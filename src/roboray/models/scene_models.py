"""Real scene-model adapters (subprocess architecture).

ViPE: monocular -> camera poses (c2w, inverted to w2c output) + near-metric depth.
Fast-FoundationStereo: rectified stereo -> disparity -> metric depth (depth = fx·B/d);
poses still come from ViPE (the design's semantics: with stereo, depth goes through
FFS and poses through ViPE).

Each model runs in its own conda env; this module only does orchestration and
contract normalization, it never imports torch.

**Resident session**: the adapter holds a ``ModelEnvSession``
(``open_session()`` starts a long-lived worker subprocess, ``close_session()`` stops
it); ``infer()`` branches internally — when ``self._session is not None`` it reuses
the loaded weights through ``session.infer``, otherwise it runs a ``run_in_modelenv``
oneshot subprocess. Session and oneshot build ``weights``/``extra`` from the same
source: the same ``os.environ.get(...)`` and the same extra dict, so the model build
key (weights, extra) is identical in both modes.
"""

import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from roboray.stages.artifacts import save_artifact

from .modelenv import ModelEnvError, ModelEnvSession, run_in_modelenv

__all__ = ["VipeSceneModel", "FastFoundationStereoModel"]

_REQUIRED_SCENE_KEYS = {"camera_poses", "intrinsics", "depth", "depth_source", "n_frames", "fps"}
_DEFAULT_SCRIPT_DIR = Path(__file__).resolve().parents[3] / "scripts" / "modelenvs"


def _env_python(env_var: str) -> str:
    return os.environ.get(env_var, "python")


def _check_python(env_var: str, build_hint: str) -> str:
    python = _env_python(env_var)
    if python != "python" and shutil.which(python) is None and not Path(python).exists():
        raise ModelEnvError(
            f"{env_var} points to a non-existent interpreter: {python}. "
            f"Build the model env with {build_hint}, or set a correct path in the "
            "config models section / environment variable."
        )
    try:
        proc = subprocess.run([python, "--version"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ModelEnvError(f"{env_var}={python} is not executable: {exc}") from exc
    if proc.returncode != 0:
        raise ModelEnvError(f"{env_var}={python} --version failed: {proc.stderr[:500]}")
    return python


def _write_input_npz(path: Path, frames: dict, calib: dict) -> None:
    arrays = {
        "frames": np.asarray(frames["frames"]),
        "n_frames": np.int32(frames["frames"].shape[0]),
        "fps": np.float32(calib.get("fps", 30.0)),
    }
    if "frames_right" in frames:
        arrays["frames_right"] = np.asarray(frames["frames_right"])
    np.savez(path, **arrays)


class _SceneBase:
    name = "scene"

    def __init__(self, script: str | None = None):
        self.script = script or str(_DEFAULT_SCRIPT_DIR / self._script_name)
        self.loaded_on: str | None = None
        self._session: ModelEnvSession | None = None

    def load(self, device: str) -> None:
        self.loaded_on = device

    def cost_hint(self, n_frames: int, resolution_key: str) -> float:
        w = int(resolution_key.split("x")[0])
        return self._sec_per_frame(w) * max(1, n_frames)

    # -- resident session --------------------------------------------------
    def open_session(self) -> float:
        """Start a long-lived worker subprocess session; return startup time (s).

        Model weights are built lazily on the **first inference** keyed by
        (weights, extra) (spec §5.3), so this only accounts for "process start +
        handshake" time.

        Subclasses may override to open **more** sessions (e.g. the vipe_model
        nested in the FFS stereo route); ``_open_own_session`` only opens its own.
        """
        t0 = time.perf_counter()
        self._open_own_session()
        return time.perf_counter() - t0

    def _open_own_session(self) -> None:
        """Open only **its own** session (no cascading).

        A separate method so a unit test can patch it to verify open_session's
        propagation logic (nested-model cascade) without actually spawning a
        subprocess.
        """
        self._session = ModelEnvSession(self._env_python(), self.script)
        self._session.start()

    def close_session(self) -> None:
        """Close the session (idempotent: no-op when not open / already closed)."""
        if self._session is not None:
            self._session.close()
            self._session = None

    # -- hooks: subclasses provide model-specific details -------------------
    def _env_python(self) -> str:
        raise NotImplementedError

    def _weights(self) -> str:
        raise NotImplementedError

    def _extra(self, calib: dict) -> dict:
        """Extra-args dict; forwarded via ``extra=json.dumps(...)`` in session mode
        and via ``--extra`` in oneshot mode (whether it is carried is decided by
        ``_oneshot_argv``)."""
        return {}

    def _write_input(self, path: Path, frames: dict, calib: dict) -> None:
        _write_input_npz(path, frames, calib)

    def _out_suffix(self) -> str:
        return ".out.npz"

    def _oneshot_argv(self, inp: Path, out: Path, weights: str, extra: dict) -> list:
        argv = ["--input", str(inp), "--output", str(out)]
        if weights:
            argv += ["--weights", weights]
        return argv

    def _fail_msg(self, summary: dict) -> str:
        return f"model inference failed: {summary}"

    # -- inference -----------------------------------------------------------
    def _infer_once(self, python: str, inp: Path, out: Path,
                    weights: str, extra: dict) -> dict:
        """Run one inference: the session branch uses the resident session, otherwise
        a oneshot subprocess.

        ``weights``/``extra`` are sourced from the same place and built the same way
        (json.dumps) as the oneshot argv, so the build key is identical in both modes.
        """
        if self._session is not None:
            return self._session.infer(str(inp), str(out), weights=weights,
                                       extra=json.dumps(extra))
        return run_in_modelenv(python, self.script,
                               self._oneshot_argv(inp, out, weights, extra),
                               timeout_s=3600.0)

    def infer(self, frames: dict, calib: dict, out_path: Path) -> None:
        """Default flow: write input -> run one inference (session/oneshot) ->
        validate -> adopt.

        Single-route models (vipe / echo test adapter) inherit directly; FFS
        overrides it with the "FFS depth + ViPE poses" composite flow.
        """
        python = self._env_python()
        weights = self._weights()
        extra = self._extra(calib)
        with tempfile.TemporaryDirectory() as td:
            inp = Path(td) / "input.npz"
            self._write_input(inp, frames, calib)
            out = Path(str(inp) + self._out_suffix())
            summary = self._infer_once(python, inp, out, weights, extra)
            if not summary.get("ok"):
                raise ModelEnvError(self._fail_msg(summary))
            self._validate_output(out)
            self._adopt(out, Path(out_path), summary)

    def _validate_output(self, out_path: Path) -> None:
        with np.load(out_path, allow_pickle=False) as z:
            missing = _REQUIRED_SCENE_KEYS - set(z.files)
        if missing:
            raise ModelEnvError(f"Scene artifact is missing keys: {sorted(missing)} ({out_path.name})")

    def _adopt(self, raw_path: Path, out_path: Path, summary: dict) -> None:
        with np.load(raw_path, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        arrays.pop("__schema_version__", None)  # the script stamped it; save_artifact re-stamps, avoid key collision
        depth_source = str(arrays["depth_source"])
        if depth_source not in ("stereo", "monocular"):
            raise ModelEnvError(f"invalid depth_source: {depth_source!r}")
        save_artifact(out_path, **arrays)


class VipeSceneModel(_SceneBase):
    """Monocular route: ViPE outputs poses (c2w, inverted to w2c) and near-metric depth."""

    name = "vipe"
    _script_name = "run_vipe.py"

    def _sec_per_frame(self, width: int) -> float:
        return 0.15 if width <= 640 else 0.4

    def _env_python(self) -> str:
        return _env_python("ROBORAY_VIPE_PYTHON")

    def _weights(self) -> str:
        return os.environ.get("ROBORAY_VIPE_WEIGHTS", "")

    def _fail_msg(self, summary: dict) -> str:
        return f"ViPE inference failed: {summary}"

    def load(self, device: str) -> None:
        super().load(device)
        _check_python("ROBORAY_VIPE_PYTHON", "deploy/modelenvs/build_vipe.sh")


class FastFoundationStereoModel(_SceneBase):
    """Stereo route: FFS depth + ViPE poses (composite output)."""

    name = "fast_foundation_stereo"
    _script_name = "run_ffs.py"

    def __init__(self, script: str | None = None, vipe_script: str | None = None):
        super().__init__(script)
        self.vipe_model = VipeSceneModel(script=vipe_script)

    def open_session(self) -> float:
        """Cascade-open sessions on the stereo route: both FFS itself and the nested
        vipe_model.

        Poses come from ``self.vipe_model.infer`` (FFS only produces depth); without
        the cascade, nested calls in session mode would go through oneshot and ViPE
        weights would be rebuilt on every stereo scene inference — "weights loaded
        once" would not hold for the stereo route.
        """
        t0 = time.perf_counter()
        self._open_own_session()
        if hasattr(self.vipe_model, "open_session"):
            self.vipe_model.open_session()
        return time.perf_counter() - t0

    def close_session(self) -> None:
        """Cascade-close sessions: first its own (via the parent implementation), then
        the nested vipe_model."""
        super().close_session()
        if hasattr(self.vipe_model, "close_session"):
            self.vipe_model.close_session()

    def _sec_per_frame(self, width: int) -> float:
        return 0.05 if width <= 640 else 0.15

    def _env_python(self) -> str:
        return _env_python("ROBORAY_FFS_PYTHON")

    def _weights(self) -> str:
        return os.environ.get("ROBORAY_FFS_WEIGHTS", "")

    def _out_suffix(self) -> str:
        return ".ffs.npz"

    def _oneshot_argv(self, inp: Path, out: Path, weights: str, extra: dict) -> list:
        # FFS's oneshot argv always carries --weights (possibly empty) and --extra
        # (built the same way as the session path).
        return ["--input", str(inp), "--output", str(out),
                "--weights", weights, "--extra", json.dumps(extra)]

    def _fail_msg(self, summary: dict) -> str:
        return f"FFS inference failed: {summary}"

    def load(self, device: str) -> None:
        super().load(device)
        _check_python("ROBORAY_FFS_PYTHON", "deploy/modelenvs/build_ffs.sh")
        weights = os.environ.get("ROBORAY_FFS_WEIGHTS", "")
        # Weights-directory validation only runs when explicitly set: when unset we do
        # not hard-fail here, the production script run_ffs.py gives the same readable
        # guidance (stand-in scripts need no weights).
        if weights and not Path(weights).exists():
            raise ModelEnvError(
                "ROBORAY_FFS_WEIGHTS is unset or its directory does not exist; "
                "download the weights from HF nvidia/c-fast-foundationstereo and set "
                "the variable (see docs/models.md)."
            )
        self.vipe_model.load(device)

    def infer(self, frames: dict, calib: dict, out_path: Path) -> None:
        if not calib.get("has_stereo") or "frames_right" not in frames:
            raise ModelEnvError(
                "fast_foundation_stereo requires frames_right stereo input; "
                "monocular episodes should take the vipe route (check resolve_scene_model's "
                "routing conditions)."
            )
        python = self._env_python()
        t_l2r = np.asarray(calib.get("T_left_to_right", np.eye(4)), dtype=np.float64)
        baseline = float(t_l2r[0, 3])
        w = int(frames["frames"].shape[2])
        fx = float(np.asarray(calib["K"])[0, 0]) if calib.get("K") is not None else float(w)
        extra = {"baseline_m": baseline, "fx": fx}
        # spec §6's FFS input-resolution control: forwarded to run_ffs.py via --extra
        # (0/absent = no resize).
        if "stereo_max_image_side_px" in calib:
            extra["stereo_max_image_side_px"] = int(calib["stereo_max_image_side_px"] or 0)
        with tempfile.TemporaryDirectory() as td:
            inp = Path(td) / "input.npz"
            self._write_input(inp, frames, calib)
            ffs_out = Path(str(inp) + self._out_suffix())
            summary = self._infer_once(python, inp, ffs_out, self._weights(), extra)
            if not summary.get("ok"):
                raise ModelEnvError(self._fail_msg(summary))
            self._validate_output(ffs_out)
            with np.load(ffs_out, allow_pickle=False) as z:
                arrays = {k: z[k] for k in z.files}
            arrays.pop("__schema_version__", None)  # the script stamped it; save_artifact re-stamps, avoid key collision
            # Poses go through ViPE (separate output file; on success take the
            # poses/intrinsics to replace FFS's placeholders)
            vipe_out = Path(td) / "vipe.npz"
            self.vipe_model.infer(frames, calib, vipe_out)
            with np.load(vipe_out, allow_pickle=False) as z:
                arrays["camera_poses"] = z["camera_poses"]
                arrays["intrinsics"] = z["intrinsics"]
            arrays["depth_source"] = np.array("stereo")
            save_artifact(out_path, **arrays)
