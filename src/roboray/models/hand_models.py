"""Real hand-model adapter: HaMeR (subprocess architecture).

HaMeR runs in its own conda env; this module only does orchestration and contract
normalization, it never imports torch.

MANO_RIGHT.pkl is a registration-gated download (https://mano.is.tue.mpg.de); the
adapter's load() hard-validates its presence — fail early rather than allow a
half-usable state.

**Resident session**: like the scene adapter — ``open_session()`` starts a long-lived
worker subprocess, ``close_session()`` stops it; ``infer()`` goes through
``session.infer`` when ``self._session`` is not None, otherwise through a
``run_in_modelenv`` oneshot. Session and oneshot build ``weights``/``extra`` from
the same source (same ``os.environ.get`` / same extra dict; extra has the optional
key ``checkpoint``).
"""

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from roboray.stages.artifacts import save_artifact

from .modelenv import ModelEnvError, ModelEnvSession, run_in_modelenv

__all__ = ["HamerHandModel"]

_REQUIRED_HAND_KEYS = {
    "hand_present", "joints_2d", "joints_3d_cam", "mano_pose", "mano_shape",
    "mano_trans", "confidence", "n_frames", "fps",
}
_DEFAULT_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "modelenvs" / "run_hamer.py"
_MANO_HINT = (
    "MANO_RIGHT.pkl must be downloaded after registering at "
    "https://mano.is.tue.mpg.de and placed in the mano/ subdirectory of the "
    "weights directory (layout in docs/models.md)."
)


class HamerHandModel:
    name = "hamer"

    def __init__(self, script: str | None = None):
        self.script = script or str(_DEFAULT_SCRIPT)
        self.loaded_on: str | None = None
        self._session: ModelEnvSession | None = None

    def load(self, device: str) -> None:
        self.loaded_on = device
        python = os.environ.get("ROBORAY_HAMER_PYTHON", "python")
        if python != "python" and not Path(python).exists():
            raise ModelEnvError(
                f"ROBORAY_HAMER_PYTHON points to a non-existent interpreter: {python}. "
                "Build the model env with deploy/modelenvs/build_hamer.sh."
            )
        try:
            proc = subprocess.run([python, "--version"], capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ModelEnvError(f"ROBORAY_HAMER_PYTHON={python} is not executable: {exc}") from exc
        if proc.returncode != 0:
            raise ModelEnvError(f"ROBORAY_HAMER_PYTHON={python} --version failed: {proc.stderr[:500]}")
        weights = os.environ.get("ROBORAY_HAMER_WEIGHTS", "")
        if not weights or not Path(weights).exists():
            raise ModelEnvError(f"ROBORAY_HAMER_WEIGHTS is unset or its directory does not exist. {_MANO_HINT}")
        if not (Path(weights) / "mano" / "MANO_RIGHT.pkl").exists():
            raise ModelEnvError(
                f"weights directory is missing mano/MANO_RIGHT.pkl ({weights}). {_MANO_HINT}"
            )

    def cost_hint(self, n_frames: int, resolution_key: str) -> float:
        return 0.08 * max(1, n_frames)

    # -- resident session --------------------------------------------------
    def open_session(self) -> float:
        """Start a long-lived worker subprocess session; return startup time (s)."""
        t0 = time.perf_counter()
        self._session = ModelEnvSession(self._python(), self.script)
        self._session.start()
        return time.perf_counter() - t0

    def close_session(self) -> None:
        """Close the session (idempotent: no-op when not open / already closed)."""
        if self._session is not None:
            self._session.close()
            self._session = None

    def _python(self) -> str:
        return os.environ.get("ROBORAY_HAMER_PYTHON", "python")

    def _weights(self) -> str:
        return os.environ.get("ROBORAY_HAMER_WEIGHTS", "")

    def _extra(self, calib: dict) -> dict:
        # HaMeR's extra only has the optional key checkpoint (path to the checkpoint;
        # when absent the script falls back to weights/_DATA/hamer_ckpts/.../hamer.ckpt),
        # inference_fps (inference is sampled at round(fps/inference_fps)) and max_hands
        # (detection count cap, default 4). The calibration intrinsics K are already
        # written into the input npz (see infer), no need to forward them via extra.
        # Session and oneshot build the same extra.
        extra: dict = {}
        ckpt = os.environ.get("ROBORAY_HAMER_CHECKPOINT", "")
        if ckpt:
            extra["checkpoint"] = ckpt
        # The actor has already injected payload.inference_fps into calib (same path as
        # scene's stereo_max_image_side_px); only forward when non-empty — the default 30
        # keeps stride=1 unchanged.
        if calib.get("inference_fps"):
            extra["inference_fps"] = int(calib["inference_fps"])
        # Forward the detection-count cap the same way: only when calib (payload
        # injection chain) carries max_hands; otherwise it is not sent and the script
        # falls back to DEFAULT_MAX_HANDS=4, unchanged behavior.
        if calib.get("max_hands"):
            extra["max_hands"] = int(calib["max_hands"])
        return extra

    def infer(self, frames: dict, calib: dict, out_path: Path) -> None:
        python = self._python()
        weights = self._weights()
        extra = self._extra(calib)
        with tempfile.TemporaryDirectory() as td:
            inp = Path(td) / "input.npz"
            arrays = {
                "frames": np.asarray(frames["frames"]),
                "n_frames": np.int32(frames["frames"].shape[0]),
                "fps": np.float32(calib.get("fps", 30.0)),
            }
            # Forward the calibration intrinsics as well: HaMeR's camera translation
            # scale is proportional to focal length (in `cam_crop_to_full`,
            # tz = 2*f/bs), so a measured K gives metric units; without it the official
            # demo prior focal length is used and only relative scale is obtained.
            if calib.get("K") is not None:
                arrays["K"] = np.asarray(calib["K"], np.float64)
            np.savez(inp, **arrays)
            raw = Path(str(inp) + ".out.npz")
            if self._session is not None:
                summary = self._session.infer(str(inp), str(raw), weights=weights,
                                              extra=json.dumps(extra))
            else:
                # Oneshot argv is built from the same source as the session path (the
                # same extra dict, json.dumps): --extra is always passed — when extra is
                # empty the script behavior is byte-for-byte equivalent to before
                # (fake_hamer/run_hamer parse an empty extra as {}, triggering no sampling).
                argv = ["--input", str(inp), "--output", str(raw),
                        "--weights", weights, "--extra", json.dumps(extra)]
                summary = run_in_modelenv(python, self.script, argv, timeout_s=3600.0)
            if not summary.get("ok"):
                raise ModelEnvError(f"HaMeR inference failed: {summary}")
            # allow_pickle=True: the script may return object arrays (candidates: a
            # list[dict] per frame, consumed by later association stages); all keys are
            # passed through to save_artifact, the same trust domain as load_artifact's
            # allow_pickle=True reads (output of our own scripts).
            with np.load(raw, allow_pickle=True) as z:
                missing = _REQUIRED_HAND_KEYS - set(z.files)
                arrays_out = {k: z[k] for k in z.files}
            if missing:
                raise ModelEnvError(f"Hand artifact is missing keys: {sorted(missing)}")
            arrays_out.pop("__schema_version__", None)  # the script stamped it; save_artifact re-stamps, avoid key collision
            save_artifact(out_path, **arrays_out)
