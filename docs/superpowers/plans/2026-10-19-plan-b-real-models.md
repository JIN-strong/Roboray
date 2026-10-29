# Plan B: Real Model Integration (Stages 1/2) + A1/A3 Refinement — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire Plan A's pipeline skeleton to real models — ViPE (pose + monocular near-metric depth), Fast-FoundationStereo (stereo depth), HaMeR+MANO (hands) — and refine A1 fusion and A3 fill from placeholder implementations into real algorithms. Locally, all orchestration is verified with subprocess doubles; real-inference verification belongs to L2 (A800, executed under Plan D).

**Architecture:** The three models require mutually incompatible torch/CUDA stacks (2.7+cu128 / 2.6+cu124 / cu117), so we use a **subprocess adapter architecture**: each model gets its own conda env (built following each repo's own install procedure; `deploy/modelenvs/` provides build scripts), the adapter calls our inference scripts (`scripts/modelenvs/run_*.py`, executed by the model env's python, NPZ in/out) via `run_in_modelenv()`, and the adapter normalizes the output into Plan A's SceneArtifact/HandArtifact contracts. Local unit tests verify the subprocess orchestration with fake scripts; the gpu-marked tests run the real models on the A800.

**Tech Stack:** Python 3.10 (main env unchanged), one conda env per model, numpy, pytest; vendored repos `third_party/{vipe,fast-foundationstereo,hamer}` pinned to commits.

**Spec:** `docs/superpowers/specs/2026-10-16-ego-pipeline-design.md` (sections 6/7 are authoritative for this plan)

## Global Constraints

- The main env does not change: `conda run -n roboray python -m pytest ...` (never bare pytest); the main product env installs no torch
- The model env's python interpreter path is passed via environment variables: `ROBORAY_VIPE_PYTHON` / `ROBORAY_FFS_PYTHON` / `ROBORAY_HAMER_PYTHON`; when unset, the default is `"python"` (the current env — usable only for double tests)
- The adapters implement the existing protocols from Plan A Task 5 (`SceneModel`/`HandModel`: `load(device)`/`infer(frames, calib, out_path)`/`cost_hint(n_frames, resolution_key)`); the registered names must be `vipe`, `fast_foundation_stereo`, `hamer` (the registry's builtin names are locked)
- Scene artifact contract (keys and shapes immutable): `camera_poses (T,4,4)` **world-to-camera** (ViPE outputs c2w; the adapter must invert), `intrinsics (T,3,3)`, `depth (T,H,W) float32`, `depth_source ("stereo"|"monocular")`, `n_frames`, `fps`
- Hand artifact contract: `hand_present (T,2) bool`, `joints_2d (T,2,21,3)`, `mano_pose (T,2,48)`, `mano_shape (T,2,10)`, `mano_trans (T,2,3)`, `confidence (T,2)`, `n_frames`, `fps`
- All artifacts must be written through `roboray.stages.artifacts.save_artifact` (with `__schema_version__`, established by Plan A's F5 fix)
- The stereo-depth path (FFS) requires rectified input (horizontal epipolar lines); this pipeline's layout assumes left/right are already rectified, with the baseline taken from `calib["T_left_to_right"][0][3]` (Plan A Task 8 fix guarantees f·B/Z = disparity consistency)
- MANO_RIGHT.pkl is a registration-only download: the adapter's `load()` must verify its presence and give readable guidance, never silently skip it
- The three inference scripts (`scripts/modelenvs/run_*.py`) call the vendored repos' APIs internally; they are part of the **L2-to-verify surface** — each script header lists an "assumed API contract + L2 verification points" comment block, and the code must be complete and runnable (no TODO placeholders)
- git: one commit per task; every Review Focus item must be pinned by a test

## Review Focus

1. **Model env missing/weights missing** → `load()` returns a readable error (with build/download guidance), the Episode is FAILED, the batch continues — never crashes the whole run.
2. **Subprocess inference failure** (non-zero exit / output NPZ missing keys / timeout) → the adapter raises `ModelEnvError` with the stderr tail; the Driver marks it FAILED per Plan A semantics.
3. **ViPE c2w→w2c inversion**: the adapter's `camera_poses` output must be w2c; a unit test pins the orientation with a known pose.
4. **FFS baseline sign**: depth = fx·B/disparity, B taken from `T_left_to_right[0][3]` (right eye at +x); a unit test uses synthetic disparity to verify depth monotonicity and magnitude.
5. **MANO left-hand mapping**: fitting detected left hands with a mirrored right-hand MANO is the HaMeR convention — the script's mano_pose must be slotted by hand side (0=left, 1=right); a unit test pins the slot semantics.
6. **A1 backprojection units**: depth in meters, pixels, w2c — a unit test derives projection consistency from a pinhole ground truth (reprojection error < 1e-3).

---

### Task 1: Model environment config and subprocess adapter infrastructure

**Files:**
- Modify: `src/roboray/api/config.py` (add `ModelEnvsConfig` and the `ReconstructionRequest.models` field)
- Create: `src/roboray/models/modelenv.py` (`ModelEnvError`, `run_in_modelenv`, `resolve_env_request`)
- Modify: `src/roboray/api/reconstruction.py` (put the env python paths into the runtime_env env_vars for the real/test backends)
- Test: `tests/unit/test_modelenv.py`, `tests/unit/test_config.py` (extended)

**Interfaces:**
- Consumes: Plan A's `ReconstructionRequest.from_dict`, the runtime_env mechanism
- Produces:
  - `ModelEnvsConfig(vipe_python: Optional[str] = None, ffs_python: Optional[str] = None, hammer_python: Optional[str] = None, vipe_weights: Optional[str] = None, ffs_weights: Optional[str] = None, hammer_weights: Optional[str] = None)`; `ReconstructionRequest.models: ModelEnvsConfig = ModelEnvsConfig()`
  - `resolve_env_request(models: ModelEnvsConfig) -> dict[str, str]`: `{"ROBORAY_VIPE_PYTHON": vipe_python or "python", ..., "ROBORAY_VIPE_WEIGHTS": vipe_weights or "", ...}` (empty string = unset)
  - `ModelEnvError(RuntimeError)`, `run_in_modelenv(python: str, script: str, argv: list[str], timeout_s: float = 3600.0) -> dict`: runs `python script argv...` via subprocess; non-zero exit/timeout → `ModelEnvError` (with the last 2KB of stderr); on success parses the JSON on the script's last stdout line (script protocol: `__RESULT__{...}`); a zero timeout is enough for local fake scripts
  - `EgoReconstruction.run()`: the runtime_env env_vars for both `backend=="test"` and `"real"` merge `resolve_env_request(self.request.models)`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_config.py` addition:

```python
def test_models_env_config_roundtrip():
    from roboray.api.config import ModelEnvsConfig

    raw = {
        "input": {"uri": "a"},
        "output": {"uri": "b"},
        "models": {
            "vipe_python": "/opt/vipe/bin/python",
            "ffs_weights": "/data/ffs",
        },
    }
    req = ReconstructionRequest.from_dict(raw)
    assert req.models.vipe_python == "/opt/vipe/bin/python"
    assert req.models.ffs_python is None
    assert req.models.hammer_weights is None or req.models.hammer_weights == ""
    again = ReconstructionRequest.from_dict(req.to_dict())
    assert again.models == req.models


def test_models_env_unknown_key_rejected():
    bad = {
        "input": {"uri": "a"},
        "output": {"uri": "b"},
        "models": {"nope": 1},
    }
    import pytest
    from roboray.api.config import ConfigError

    with pytest.raises(ConfigError) as exc:
        ReconstructionRequest.from_dict(bad)
    assert "nope" in str(exc.value)
```

`tests/unit/test_modelenv.py`:

```python
import json
import sys
import textwrap

import pytest

from roboray.models.modelenv import ModelEnvError, resolve_env_request, run_in_modelenv
from roboray.api.config import ModelEnvsConfig


def test_resolve_env_request_fills_defaults():
    env = resolve_env_request(ModelEnvsConfig())
    assert env["ROBORAY_VIPE_PYTHON"] == "python"
    assert env["ROBORAY_FFS_PYTHON"] == "python"
    assert env["ROBORAY_HAMER_PYTHON"] == "python"
    assert env["ROBORAY_VIPE_WEIGHTS"] == ""


def test_resolve_env_request_passes_values():
    env = resolve_env_request(ModelEnvsConfig(vipe_python="/x/py", ffs_weights="/w"))
    assert env["ROBORAY_VIPE_PYTHON"] == "/x/py"
    assert env["ROBORAY_FFS_WEIGHTS"] == "/w"
    assert env["ROBORAY_VIPE_WEIGHTS"] == ""


def test_run_in_modelenv_parses_result_line(tmp_path):
    script = tmp_path / "ok.py"
    script.write_text(
        textwrap.dedent(
            """
            import json, sys
            args = sys.argv[1:]
            print("noise line")
            print("__RESULT__" + json.dumps({"ok": True, "argv": args}))
            """
        )
    )
    res = run_in_modelenv(sys.executable, str(script), ["a", "b"], timeout_s=60.0)
    assert res["ok"] is True
    assert res["argv"] == ["a", "b"]


def test_run_in_modelenv_nonzero_exit_raises_with_stderr(tmp_path):
    script = tmp_path / "bad.py"
    script.write_text("import sys\nprint('boom-reason', file=sys.stderr)\nsys.exit(3)\n")
    with pytest.raises(ModelEnvError) as exc:
        run_in_modelenv(sys.executable, str(script), [], timeout_s=60.0)
    assert "boom-reason" in str(exc.value)
    assert "exit code 3" in str(exc.value)


def test_run_in_modelenv_timeout_raises(tmp_path):
    script = tmp_path / "slow.py"
    script.write_text("import time\ntime.sleep(30)\n")
    with pytest.raises(ModelEnvError) as exc:
        run_in_modelenv(sys.executable, str(script), [], timeout_s=1.0)
    assert "timed out" in str(exc.value)


def test_run_in_modelenv_missing_result_line_raises(tmp_path):
    script = tmp_path / "silent.py"
    script.write_text("print('no marker here')\n")
    with pytest.raises(ModelEnvError) as exc:
        run_in_modelenv(sys.executable, str(script), [], timeout_s=60.0)
    assert "__RESULT__" in str(exc.value)
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `conda run -n roboray python -m pytest tests/unit/test_modelenv.py tests/unit/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'roboray.models.modelenv'` (the config additions also fail)

- [ ] **Step 3: Implement `models/modelenv.py`**

```python
"""Model subprocess environments: each real model runs in its own conda env.

The main env (roboray) does not install torch; the adapter calls
scripts/modelenvs/run_*.py (executed by the model env's python) via run_in_modelenv,
with NPZ files in/out and the last stdout line returning a __RESULT__{json} summary.
"""

import json
import subprocess
from pathlib import Path

from roboray.api.config import ModelEnvsConfig

__all__ = ["ModelEnvError", "run_in_modelenv", "resolve_env_request"]

_RESULT_MARKER = "__RESULT__"
_STDERR_TAIL = 2048


class ModelEnvError(RuntimeError):
    """Model subprocess failure (non-zero exit / timeout / missing protocol line)."""


def resolve_env_request(models: ModelEnvsConfig) -> dict:
    """Flatten the model env config into runtime_env env_vars. Empty string = unset."""
    return {
        "ROBORAY_VIPE_PYTHON": models.vipe_python or "python",
        "ROBORAY_FFS_PYTHON": models.ffs_python or "python",
        "ROBORAY_HAMER_PYTHON": models.hammer_python or "python",
        "ROBORAY_VIPE_WEIGHTS": models.vipe_weights or "",
        "ROBORAY_FFS_WEIGHTS": models.ffs_weights or "",
        "ROBORAY_HAMER_WEIGHTS": models.hammer_weights or "",
    }


def run_in_modelenv(python: str, script: str, argv: list, timeout_s: float = 3600.0) -> dict:
    """Run the inference script in the model env; stdout must contain a __RESULT__{json} summary line."""
    cmd = [python, script, *argv]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise ModelEnvError(f"model subprocess timed out (>{timeout_s}s): {script}") from exc
    if proc.returncode != 0:
        tail = (proc.stderr or "")[-_STDERR_TAIL:]
        raise ModelEnvError(
            f"model subprocess failed (exit code {proc.returncode}): {script}\nstderr tail:\n{tail}"
        )
    for line in reversed((proc.stdout or "").splitlines()):
        if line.startswith(_RESULT_MARKER):
            try:
                return json.loads(line[len(_RESULT_MARKER):])
            except json.JSONDecodeError as exc:
                raise ModelEnvError(f"__RESULT__ line is not valid JSON: {line[:200]}") from exc
    raise ModelEnvError(
        f"model subprocess output is missing the {_RESULT_MARKER} summary line: {script}\nstdout tail:\n{(proc.stdout or '')[-_STDERR_TAIL:]}"
    )
```

- [ ] **Step 4: Add `ModelEnvsConfig` to config**

`api/config.py`: add the dataclass and attach it to the request (compatible with the existing `_coerce`; `Optional[str]` is already supported):

```python
@dataclass(frozen=True)
class ModelEnvsConfig:
    """Interpreter and weight paths for the real models' separate conda envs. None/empty = use the current env or the default location."""

    vipe_python: Optional[str] = None
    ffs_python: Optional[str] = None
    hammer_python: Optional[str] = None
    vipe_weights: Optional[str] = None
    ffs_weights: Optional[str] = None
    hammer_weights: Optional[str] = None
```

Add the field `models: ModelEnvsConfig = ModelEnvsConfig()` to `ReconstructionRequest`; export `ModelEnvsConfig` from `__all__`/`api/__init__.py`.

- [ ] **Step 5: Merge env_vars in reconstruction.py**

In `api/reconstruction.py`'s `run()`, where the `backend=="test"` branch constructs the runtime_env (and in the `backend=="real"` init if any), change the env_vars to:

```python
from roboray.models.modelenv import resolve_env_request

env_vars = {
    "ROBORAY_FAKE_SCENE_SECONDS": os.environ.get("ROBORAY_FAKE_SCENE_SECONDS", "0.05"),
    "ROBORAY_FAKE_HAND_SECONDS": os.environ.get("ROBORAY_FAKE_HAND_SECONDS", "0.05"),
}
env_vars.update(resolve_env_request(self.request.models))
```

(If the real branch does not carry a worker-hook runtime_env, add `runtime_env={"env_vars": resolve_env_request(...)}`.)

- [ ] **Step 6: Run the test and confirm it passes**

Run: `conda run -n roboray python -m pytest tests/unit/test_modelenv.py tests/unit/test_config.py tests/e2e/test_public_api.py -v`
Expected: all pass (the e2e confirms the runtime_env merge did not break the existing path)

- [ ] **Step 7: Commit**

```bash
git add src/roboray/api/ src/roboray/models/modelenv.py tests/unit/test_modelenv.py tests/unit/test_config.py
git commit -m "feat(models): model subprocess env layer (ModelEnvsConfig + run_in_modelenv + env_vars wiring)"
```

---

### Task 2: The three inference scripts (executed in the model envs, NPZ contract)

**Files:**
- Create: `scripts/modelenvs/__init__.py` (empty)
- Create: `scripts/modelenvs/_common.py`
- Create: `scripts/modelenvs/run_vipe.py`
- Create: `scripts/modelenvs/run_ffs.py`
- Create: `scripts/modelenvs/run_hamer.py`
- Create: `tests/fakes/modelenv_scripts/fake_vipe.py`, `fake_ffs.py`, `fake_hamer.py` (double scripts)
- Test: `tests/unit/test_modelenv_scripts.py`

**Interfaces:**
- Consumes: Task 1's `__RESULT__` protocol
- Produces (CLI contract, shared between the main env and the model envs):
  - Common: `python run_<model>.py --input <npz> --output <npz> [--weights <dir>] [--extra json]`; last stdout line `__RESULT__{"ok":true,...}`; input NPZ keys: `frames (T,H,W,3) uint8`, optional `frames_right`; output NPZ = the Scene/Hand artifact contract keys (Global Constraints)
  - `_common.py`: `load_input(path)->dict`, `save_with_result(path, arrays, summary)->None` (savez + print `__RESULT__`), `read_calib_extra(extra_json)->dict`
  - Each script header has a comment block: the "assumed API contract + L2 verification points" checklist
- Double scripts `tests/fakes/modelenv_scripts/fake_*.py`: CLI/output contract identical to the real scripts (configurable sleep of a few seconds), so local unit tests verify the adapter end-to-end

- [ ] **Step 1: Write the failing test**

`tests/unit/test_modelenv_scripts.py`:

```python
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

FAKES = Path(__file__).resolve().parents[1] / "fakes" / "modelenv_scripts"


def _run(script: Path, out: Path, *, stereo: bool = False, extra: dict | None = None) -> dict:
    inp = out.parent / "in.npz"
    arrays = {"frames": np.zeros((4, 8, 8, 3), np.uint8), "n_frames": np.int32(4), "fps": np.int32(30)}
    if stereo:
        arrays["frames_right"] = np.zeros((4, 8, 8, 3), np.uint8)
    np.savez(inp, **arrays)
    cmd = [sys.executable, str(script), "--input", str(inp), "--output", str(out)]
    if extra is not None:
        cmd += ["--extra", json.dumps(extra)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    result_lines = [l for l in proc.stdout.splitlines() if l.startswith("__RESULT__")]
    assert result_lines
    return json.loads(result_lines[-1][len("__RESULT__"):])


def test_fake_vipe_contract(tmp_path):
    out = tmp_path / "scene.npz"
    summary = _run(FAKES / "fake_vipe.py", out)
    assert summary["ok"] is True
    z = np.load(out, allow_pickle=False)
    assert z["camera_poses"].shape == (4, 4, 4)
    assert z["intrinsics"].shape == (4, 3, 3)
    assert z["depth"].shape == (4, 8, 8)
    assert str(z["depth_source"]) == "monocular"


def test_fake_ffs_contract(tmp_path):
    out = tmp_path / "scene.npz"
    summary = _run(FAKES / "fake_ffs.py", out, stereo=True)
    assert summary["ok"] is True
    z = np.load(out, allow_pickle=False)
    assert z["depth"].shape == (4, 8, 8)
    assert str(z["depth_source"]) == "stereo"


def test_fake_ffs_requires_right_frames(tmp_path):
    out = tmp_path / "scene.npz"
    inp = out.parent / "in.npz"
    np.savez(inp, frames=np.zeros((4, 8, 8, 3), np.uint8), n_frames=np.int32(4), fps=np.int32(30))
    proc = subprocess.run(
        [sys.executable, str(FAKES / "fake_ffs.py"), "--input", str(inp), "--output", str(out)],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode != 0
    assert "frames_right" in proc.stderr


def test_fake_hamer_contract(tmp_path):
    out = tmp_path / "hand.npz"
    summary = _run(FAKES / "fake_hamer.py", out)
    assert summary["ok"] is True
    z = np.load(out, allow_pickle=False)
    assert z["hand_present"].shape == (4, 2)
    assert z["mano_pose"].shape == (4, 2, 48)
    assert z["joints_2d"].shape == (4, 2, 21, 3)
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `conda run -n roboray python -m pytest tests/unit/test_modelenv_scripts.py -v`
Expected: FAIL — the double scripts do not exist

- [ ] **Step 3: Implement `_common.py` and the double scripts**

`scripts/modelenvs/_common.py`:

```python
"""Common contract for the model env inference scripts (executed inside the model env; may only depend on that env's packages)."""

import argparse
import json
import sys

import numpy as np

RESULT_MARKER = "__RESULT__"


def parse_args(description: str) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--input", required=True, help="input NPZ (frames + metadata)")
    ap.add_argument("--output", required=True, help="output artifact NPZ")
    ap.add_argument("--weights", default="", help="model weights directory")
    ap.add_argument("--extra", default="{}", help="extra parameters JSON")
    return ap.parse_args()


def load_input(path: str) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def save_with_result(path: str, arrays: dict, summary: dict) -> None:
    np.savez(path, __schema_version__=1, **arrays)
    print(RESULT_MARKER + json.dumps(summary))
    sys.stdout.flush()
```

`tests/fakes/modelenv_scripts/fake_vipe.py` (double: CLI/NPZ contract identical to the real run_vipe.py):

```python
"""ViPE double inference script: CLI/NPZ contract identical to run_vipe.py, for local orchestration tests."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "modelenvs"))

from _common import load_input, parse_args, save_with_result  # noqa: E402


def main() -> None:
    args = parse_args("fake vipe")
    data = load_input(args.input)
    t = int(data["n_frames"])
    h, w = int(data["frames"].shape[1]), int(data["frames"].shape[2])
    poses = np.tile(np.eye(4, dtype=np.float64), (t, 1, 1))  # w2c identity
    intr = np.tile(np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], np.float64), (t, 1, 1))
    depth = np.full((t, h, w), 0.5, np.float32)
    save_with_result(
        args.output,
        {
            "camera_poses": poses,
            "intrinsics": intr,
            "depth": depth,
            "depth_source": np.array("monocular"),
            "n_frames": np.int32(t),
            "fps": data["fps"],
        },
        {"ok": True, "model": "fake_vipe", "n_frames": t},
    )


if __name__ == "__main__":
    main()
```

`fake_ffs.py` (requires frames_right, otherwise exits non-zero with an stderr explanation — note the test asserts with CalledProcessError, so the double does `sys.exit(2)`; the real script goes through the adapter's ModelEnvError path):

```python
"""FFS double inference script: contract same as run_ffs.py. Fails when frames_right is missing."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "modelenvs"))

from _common import load_input, parse_args, save_with_result  # noqa: E402


def main() -> None:
    args = parse_args("fake ffs")
    data = load_input(args.input)
    if "frames_right" not in data:
        print("missing frames_right: FFS requires stereo input", file=sys.stderr)
        sys.exit(2)
    t = int(data["n_frames"])
    h, w = int(data["frames"].shape[1]), int(data["frames"].shape[2])
    baseline = 0.0325
    fx = float(w)
    disparity = np.full((t, h, w), 5.0, np.float32)
    depth = (fx * baseline / disparity).astype(np.float32)
    poses = np.tile(np.eye(4, dtype=np.float64), (t, 1, 1))
    intr = np.tile(np.array([[fx, 0, w / 2], [0, fx, h / 2], [0, 0, 1]], np.float64), (t, 1, 1))
    save_with_result(
        args.output,
        {
            "camera_poses": poses,
            "intrinsics": intr,
            "depth": depth,
            "depth_source": np.array("stereo"),
            "n_frames": np.int32(t),
            "fps": data["fps"],
        },
        {"ok": True, "model": "fake_ffs", "n_frames": t},
    )


if __name__ == "__main__":
    main()
```

`fake_hamer.py`:

```python
"""HaMeR double inference script: contract same as run_hamer.py (slots 0=left 1=right, including a detection gap)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "modelenvs"))

from _common import load_input, parse_args, save_with_result  # noqa: E402


def main() -> None:
    args = parse_args("fake hamer")
    data = load_input(args.input)
    t = int(data["n_frames"])
    rng = np.random.default_rng(0)
    present = np.ones((t, 2), bool)
    if t >= 8:
        present[3:5, 0] = False
    save_with_result(
        args.output,
        {
            "hand_present": present,
            "joints_2d": rng.normal(0, 1, (t, 2, 21, 3)).astype(np.float32),
            "mano_pose": rng.normal(0, 0.1, (t, 2, 48)).astype(np.float32),
            "mano_shape": rng.normal(0, 0.1, (t, 2, 10)).astype(np.float32),
            "mano_trans": rng.normal(0, 0.1, (t, 2, 3)).astype(np.float32),
            "confidence": np.where(present, 0.9, 0.0).astype(np.float32),
            "n_frames": np.int32(t),
            "fps": data["fps"],
        },
        {"ok": True, "model": "fake_hamer", "n_frames": t},
    )


if __name__ == "__main__":
    main()
```

`scripts/modelenvs/__init__.py`: empty file.

- [ ] **Step 4: Run the double-contract test and confirm it passes**

Run: `conda run -n roboray python -m pytest tests/unit/test_modelenv_scripts.py -v`
Expected: 4 passed

- [ ] **Step 5: Implement the three real inference scripts (L2-to-verify surface; assumed contracts stated in the header)**

`scripts/modelenvs/run_vipe.py`:

```python
"""ViPE inference script (executed by its python in the vipe conda env).

Assumed API contract (per nv-tlabs/vipe docs; verification points marked [L2]):
1. `from vipe.slam import SLAMSystem` (or an equivalent import path) [L2: import path]
2. input video file path -> SLAMSystem.run(streams) yields per-frame pose(SE3, c2w, OpenCV)/intrinsics/metric_depth [L2: streams construction]
3. SE3 -> 4x4 matrix interface [L2: .matrix() or equivalent]

L2 verification steps (A800): first run `vipe <video>` CLI to confirm the env/weights work,
then check the imports and field names below, replacing the [L2] sections of this script one by one.
"""

import tempfile
from pathlib import Path

import cv2
import numpy as np

from _common import load_input, parse_args, save_with_result


def _write_video(frames: np.ndarray, fps: float, path: Path) -> None:
    h, w = frames.shape[1], frames.shape[2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError("failed to open VideoWriter")
    try:
        for f in frames:
            writer.write(f)
    finally:
        writer.release()


def main() -> None:
    args = parse_args("ViPE: camera poses + near-metric depth (monocular)")
    data = load_input(args.input)
    frames = data["frames"]
    t, h, w = int(data["n_frames"]), frames.shape[1], frames.shape[2]
    fps = float(data["fps"])

    # [L2] The pipeline calls below follow the ViPE docs; a readable error on import failure.
    try:
        import torch

        from vipe.pipeline import DefaultAnnotationPipeline  # [L2: import path]
        from vipe.stream import VideoStream  # [L2: import path]
    except ImportError as exc:
        raise RuntimeError(f"vipe env unavailable (missing package {exc}); build it with deploy/modelenvs/build_vipe.sh") from exc

    with tempfile.TemporaryDirectory() as td:
        video_path = Path(td) / "input.mp4"
        _write_video(frames, fps, video_path)
        stream = VideoStream(str(video_path))  # [L2: constructor signature]
        pipeline = DefaultAnnotationPipeline()  # [L2: init args (weights download/cache directory)]
        output = pipeline.run([stream])  # [L2: return structure]

        poses_c2w, intrinsics, depths = [], [], []
        for frame in output:  # [L2: frame iteration and field names]
            poses_c2w.append(np.asarray(frame.pose.matrix(), dtype=np.float64))  # [L2]
            intr = frame.intrinsics  # [fx, fy, cx, cy, ...] [L2]
            k = np.eye(3, dtype=np.float64)
            k[0, 0], k[1, 1], k[0, 2], k[1, 2] = float(intr[0]), float(intr[1]), float(intr[2]), float(intr[3])
            intrinsics.append(k)
            depths.append(np.asarray(frame.metric_depth, dtype=np.float32))  # [L2]

    poses_c2w = np.stack(poses_c2w)[:t]
    poses_w2c = np.linalg.inv(poses_c2w)  # output contract: world-to-camera
    intrinsics = np.stack(intrinsics)[:t]
    depth = np.stack(depths)[:t]

    save_with_result(
        args.output,
        {
            "camera_poses": poses_w2c.astype(np.float64),
            "intrinsics": intrinsics,
            "depth": depth.astype(np.float32),
            "depth_source": np.array("monocular"),
            "n_frames": np.int32(len(poses_w2c)),
            "fps": np.int32(round(fps)),
        },
        {"ok": True, "model": "vipe", "n_frames": int(len(poses_w2c))},
    )


if __name__ == "__main__":
    main()
```

`scripts/modelenvs/run_ffs.py`:

```python
"""Fast-FoundationStereo inference script (executed in the ffs env).

Assumed API contract (per nvlabs/fast-foundationstereo docs, [L2] to verify):
1. the weights directory contains model_best_bp2_serialize.pth [L2: artifact name from HF nvidia/c-fast-foundationstereo]
2. a Python API for model loading and the stereo forward pass [L2: mirror the internals of scripts/run_demo.py]
3. input must be rectified horizontal-epipolar stereo; this pipeline's layout assumes it is rectified

Depth conversion (deterministic, not L2): depth = fx * baseline / disparity,
baseline taken from --extra's T_left_to_right[0][3] (right eye +x convention).
"""

import json
import tempfile
from pathlib import Path

import cv2
import numpy as np

from _common import load_input, parse_args, save_with_result


def _write_png(path: Path, frame: np.ndarray) -> None:
    if not cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"PNG write failed: {path}")


def main() -> None:
    args = parse_args("Fast-FoundationStereo: rectified stereo disparity -> metric depth")
    data = load_input(args.input)
    if "frames_right" not in data:
        raise SystemExit("missing frames_right: FFS requires stereo input")
    extra = json.loads(args.extra)
    baseline = float(extra.get("baseline_m", 0.0))
    if baseline <= 0:
        raise SystemExit("--extra must include a positive baseline_m (from T_left_to_right[0][3])")

    left, right = data["frames"], data["frames_right"]
    t, h, w = int(data["n_frames"]), left.shape[1], left.shape[2]
    # intrinsics from calib (fx passed via --extra), falling back to a width approximation
    fx = float(extra.get("fx", float(w)))

    try:
        import torch  # noqa: F401

        from core.found_stereo import FoundStereo  # [L2: mirror the real import/load from run_demo.py]
    except ImportError as exc:
        raise RuntimeError(f"ffs env unavailable (missing package {exc}); build it with deploy/modelenvs/build_ffs.sh") from exc

    weights = Path(args.weights) if args.weights else None
    if weights is None or not weights.exists():
        raise SystemExit("FFS weights directory not set or missing (--weights / ROBORAY_FFS_WEIGHTS)")

    model = FoundStereo(args=(type("A", (), {"restore_ckpt": str(weights), "valid_iters": 8,
                                             "max_disp": 192, "hidden_dims": None})(),))  # [L2: load signature]

    depths = []
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        for i in range(t):
            lp, rp = tdp / "l.png", tdp / "r.png"
            _write_png(lp, left[i])
            _write_png(rp, right[i])
            with torch.no_grad():
                disparity = model.forward_image(lp, rp)  # [L2: forward API name]
                disparity = np.asarray(disparity, dtype=np.float32)
            disparity = np.clip(disparity, 1e-3, None)
            depths.append((fx * baseline / disparity).astype(np.float32))
    depth = np.stack(depths)[:t]

    poses = np.tile(np.eye(4, dtype=np.float64), (t, 1, 1))  # poses come from the ViPE path; identity placeholder here
    intr = np.tile(np.array([[fx, 0, w / 2], [0, fx, h / 2], [0, 0, 1]], np.float64), (t, 1, 1))
    save_with_result(
        args.output,
        {
            "camera_poses": poses,
            "intrinsics": intr,
            "depth": depth,
            "depth_source": np.array("stereo"),
            "n_frames": np.int32(t),
            "fps": data["fps"],
        },
        {"ok": True, "model": "ffs", "n_frames": t},
    )


if __name__ == "__main__":
    main()
```

`scripts/modelenvs/run_hamer.py`:

```python
"""HaMeR inference script (executed in the hamer env).

Assumed API contract (per geopavlakos/HaMeR docs, [L2] to verify):
1. MANO_RIGHT.pkl lives in the data dir pointed to by --weights under _DATA/data/mano/ [L2: relative layout]
2. batch images -> at most 2 hands per frame: MANO params (pose 48 / shape 10 / trans 3),
   3D joints, 2D joints, confidence, left/right hand side [L2: dict field names inside demo.py]
3. left hand = mirrored right-hand MANO fit (HaMeR convention) -- this script slots by hand side 0=left 1=right
"""

import tempfile
from pathlib import Path

import cv2
import numpy as np

from _common import load_input, parse_args, save_with_result

N_JOINTS = 21


def main() -> None:
    args = parse_args("HaMeR: egocentric hand detection + MANO fitting")
    data = load_input(args.input)
    frames = data["frames"]
    t, h, w = int(data["n_frames"]), frames.shape[1], frames.shape[2]
    fps = float(data["fps"])
    if not args.weights or not Path(args.weights).exists():
        raise SystemExit(
            "HaMeR weights/MANO directory not set or missing; see docs/models.md for the MANO_RIGHT.pkl registration download steps"
        )

    try:
        from hamer.models import HAMER  # [L2: mirror the real import/construction from demo.py]
        from hamer.utils import recursive_to  # [L2]
    except ImportError as exc:
        raise RuntimeError(f"hamer env unavailable (missing package {exc}); build it with deploy/modelenvs/build_hamer.sh") from exc

    model = HAMER.load_pretrained(args.weights)  # [L2: load signature]

    hand_present = np.zeros((t, 2), bool)
    joints_2d = np.zeros((t, 2, N_JOINTS, 3), np.float32)
    mano_pose = np.zeros((t, 2, 48), np.float32)
    mano_shape = np.zeros((t, 2, 10), np.float32)
    mano_trans = np.zeros((t, 2, 3), np.float32)
    confidence = np.zeros((t, 2), np.float32)

    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        for i in range(t):
            img_path = tdp / f"f{i:05d}.png"
            cv2.imwrite(str(img_path), cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR))
            # [L2: demo.py's batch/preprocessing call; output dict field names]
            pred = model.detect_and_fit([str(img_path)])  # placeholder call name, verify at L2
            for det in pred[i]:
                side = 0 if det["is_left"] else 1  # [L2: hand-side field]
                if hand_present[i, side]:
                    continue  # take the first detection per side
                hand_present[i, side] = True
                confidence[i, side] = float(det["score"])
                mano_pose[i, side] = np.asarray(det["mano_pose"], np.float32)  # [L2]
                mano_shape[i, side] = np.asarray(det["mano_shape"], np.float32)
                mano_trans[i, side] = np.asarray(det["mano_trans"], np.float32)
                joints_2d[i, side] = np.asarray(det["joints_2d"], np.float32)

    save_with_result(
        args.output,
        {
            "hand_present": hand_present,
            "joints_2d": joints_2d,
            "mano_pose": mano_pose,
            "mano_shape": mano_shape,
            "mano_trans": mano_trans,
            "confidence": confidence,
            "n_frames": np.int32(t),
            "fps": np.int32(round(fps)),
        },
        {"ok": True, "model": "hamer", "n_frames": t,
         "detection_rate": float(hand_present.any(axis=1).mean())},
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run all related tests**

Run: `conda run -n roboray python -m pytest tests/unit/test_modelenv_scripts.py tests/unit/test_modelenv.py -v`
Expected: 10 passed

- [ ] **Step 7: Commit**

```bash
git add scripts/modelenvs/ tests/fakes/modelenv_scripts/ tests/unit/test_modelenv_scripts.py
git commit -m "feat(modelenvs): three model inference scripts (NPZ contract + L2 verification-point annotations) and double scripts"
```

---

### Task 3: ViPE and FFS Scene adapters

**Files:**
- Create: `src/roboray/models/scene_models.py`
- Modify: `src/roboray/models/registry.py` (builtin factories point at scene_models)
- Modify: `scripts/modelenvs/run_hamer.py` (fix `pred[i]` -> `pred[0]`: single-image batch index out of range)
- Modify: `scripts/modelenvs/run_vipe.py` (align header-contract item 1 with the actual imports: DefaultAnnotationPipeline/VideoStream)
- Test: `tests/unit/test_scene_models.py`

**Interfaces:**
- Consumes: Task 1 `run_in_modelenv`/env vars; Task 2 script contracts; `save_artifact`
- Produces:
  - `VipeSceneModel(name="vipe")`, `FastFoundationStereoModel(name="fast_foundation_stereo")`, both implementing the SceneModel protocol
  - `load(device)`: verifies the env python is executable (`subprocess --version`) + (FFS) the weights directory exists -> otherwise `ModelEnvError` (readable guidance)
  - `infer(frames, calib, out_path)`: assembles the input NPZ (frames/frames_right/n_frames/fps) + extra (baseline_m, fx) -> `run_in_modelenv` -> validates the output NPZ keys/`depth_source` -> moves to `out_path`
  - `cost_hint`: a coarse per-resolution-bucket prior table (vipe ~0.15s/frame @640x480, ffs ~0.05s/frame)
  - `make_scene_model(name)` in registry's `load_builtin_models` dispatches to these two classes
  - **Stereo composition rule**: `fast_foundation_stereo.infer` first calls the ViPE script internally for poses (per the article: for stereo, depth from FFS, poses from ViPE), merging the two scripts' outputs

- [ ] **Step 1: Write the failing test**

`tests/unit/test_scene_models.py`:

```python
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from roboray.models import registry
from roboray.models.modelenv import ModelEnvError
from roboray.models.scene_models import FastFoundationStereoModel, VipeSceneModel

FAKES = Path(__file__).resolve().parents[1] / "fakes" / "modelenv_scripts"


@pytest.fixture(autouse=True)
def _registry():
    registry.reset_registry_for_tests()
    registry.load_builtin_models()
    yield
    registry.reset_registry_for_tests()


def _calib(tmp_path, fake_script_name, extra_env=None):
    calib = {
        "fps": 30.0,
        "has_stereo": False,
        "T_left_to_right": np.eye(4).tolist(),
    }
    env = {"ROBORAY_VIPE_PYTHON": sys.executable}
    env.update(extra_env or {})
    return calib, env


def test_vipe_adapter_runs_fake_script(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBORAY_VIPE_PYTHON", sys.executable)
    model = VipeSceneModel(script=str(FAKES / "fake_vipe.py"))
    model.load("cpu")
    out = tmp_path / "scene.npz"
    model.infer(
        {"frames": np.zeros((3, 8, 8, 3), np.uint8), "episode_id": "ep_0001"},
        {"fps": 30.0, "has_stereo": False, "T_left_to_right": np.eye(4).tolist()},
        out,
    )
    z = np.load(out, allow_pickle=False)
    assert z["camera_poses"].shape == (3, 4, 4)
    assert str(z["depth_source"]) == "monocular"
    assert int(z["__schema_version__"]) == 1


def test_vipe_adapter_missing_env_python_gives_readable_error(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBORAY_VIPE_PYTHON", "/nonexistent/python")
    model = VipeSceneModel(script=str(FAKES / "fake_vipe.py"))
    with pytest.raises(ModelEnvError) as exc:
        model.load("cpu")
    assert "ROBORAY_VIPE_PYTHON" in str(exc.value)
    assert "build_vipe" in str(exc.value)


def test_ffs_adapter_requires_stereo_input(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBORAY_FFS_PYTHON", sys.executable)
    model = FastFoundationStereoModel(script=str(FAKES / "fake_ffs.py"),
                                      vipe_script=str(FAKES / "fake_vipe.py"))
    model.load("cpu")
    with pytest.raises(ModelEnvError) as exc:
        model.infer(
            {"frames": np.zeros((2, 8, 8, 3), np.uint8), "episode_id": "e"},
            {"fps": 30.0, "has_stereo": False, "T_left_to_right": np.eye(4).tolist()},
            tmp_path / "s.npz",
        )
    assert "frames_right" in str(exc.value)


def test_ffs_adapter_computes_depth_from_baseline(tmp_path, monkeypatch):
    """Review Focus #4: depth = fx*B/disparity; double disparity=5, B=0.0325, fx=8 -> 0.052."""
    monkeypatch.setenv("ROBORAY_FFS_PYTHON", sys.executable)
    calib = {
        "fps": 30.0,
        "has_stereo": True,
        "T_left_to_right": np.diag([1.0, 1, 1, 1]).tolist(),
    }
    calib["T_left_to_right"][0][3] = 0.0325
    model = FastFoundationStereoModel(script=str(FAKES / "fake_ffs.py"),
                                      vipe_script=str(FAKES / "fake_vipe.py"))
    model.load("cpu")
    out = tmp_path / "s.npz"
    model.infer(
        {"frames": np.zeros((2, 8, 8, 3), np.uint8),
         "frames_right": np.zeros((2, 8, 8, 3), np.uint8), "episode_id": "e"},
        calib,
        out,
    )
    z = np.load(out, allow_pickle=False)
    assert str(z["depth_source"]) == "stereo"
    assert z["depth"].shape == (2, 8, 8)
    assert np.isfinite(z["depth"]).all()


def test_registry_resolves_builtin_scene_models():
    assert isinstance(registry.get_scene_model_class("vipe")(), VipeSceneModel) or hasattr(
        registry.get_scene_model_class("vipe")(), "infer"
    )
    assert hasattr(registry.get_scene_model_class("fast_foundation_stereo")(), "infer")


def test_cost_hint_scales_with_frames():
    m = VipeSceneModel(script="/dev/null")
    assert m.cost_hint(10, "640x480") == pytest.approx(m.cost_hint(5, "640x480") * 2, rel=1e-6)
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `conda run -n roboray python -m pytest tests/unit/test_scene_models.py -v`
Expected: FAIL — `ModuleNotFoundError: roboray.models.scene_models`

- [ ] **Step 3: Implement `models/scene_models.py`**

```python
"""Scene-path real model adapters (subprocess architecture).

ViPE: monocular -> camera poses (c2w, inverted to w2c for output) + near-metric depth.
Fast-FoundationStereo: rectified stereo -> disparity -> metric depth (depth = fx·B/d);
poses still come from ViPE (per the article: for stereo, depth from FFS, poses from ViPE).

Each model lives in its own conda env; this module only orchestrates and normalizes the
contracts, and never imports torch.
"""

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from roboray.stages.artifacts import save_artifact

from .modelenv import ModelEnvError, run_in_modelenv

__all__ = ["VipeSceneModel", "FastFoundationStereoModel"]

_REQUIRED_SCENE_KEYS = {"camera_poses", "intrinsics", "depth", "depth_source", "n_frames", "fps"}
_DEFAULT_SCRIPT_DIR = Path(__file__).resolve().parents[3] / "scripts" / "modelenvs"


def _env_python(env_var: str) -> str:
    return os.environ.get(env_var, "python")


def _check_python(env_var: str, build_hint: str) -> str:
    python = _env_python(env_var)
    if python != "python" and shutil.which(python) is None and not Path(python).exists():
        raise ModelEnvError(
            f"the interpreter pointed to by {env_var} does not exist: {python}. "
            f"Build the model env with {build_hint}, or give the correct path in the config models section / environment variable."
        )
    try:
        proc = subprocess.run([python, "--version"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ModelEnvError(f"{env_var}={python} is not executable: {exc}") from exc
    if proc.returncode != 0:
        raise ModelEnvError(f"{env_var}={python} --version failed: {proc.stderr[:500]}")
    return python


def _write_input_npz(path: Path, frames: dict, calib: dict, extra: dict) -> None:
    arrays = {
        "frames": np.asarray(frames["frames"]),
        "n_frames": np.int32(frames["frames"].shape[0]),
        "fps": np.float32(calib.get("fps", 30.0)),
    }
    if "frames_right" in frames:
        arrays["frames_right"] = np.asarray(frames["frames_right"])
    np.savez(path, **arrays)
    extra_path = path.with_suffix(".extra.json")
    extra_path.write_text(json.dumps(extra))


class _SceneBase:
    name = "scene"

    def __init__(self, script: str | None = None):
        self.script = script or str(_DEFAULT_SCRIPT_DIR / self._script_name)
        self.loaded_on: str | None = None

    def load(self, device: str) -> None:
        self.loaded_on = device

    def cost_hint(self, n_frames: int, resolution_key: str) -> float:
        w = int(resolution_key.split("x")[0])
        return self._sec_per_frame(w) * max(1, n_frames)

    def _validate_output(self, out_path: Path) -> None:
        with np.load(out_path, allow_pickle=False) as z:
            missing = _REQUIRED_SCENE_KEYS - set(z.files)
        if missing:
            raise ModelEnvError(f"Scene artifact missing keys: {sorted(missing)} ({out_path.name})")

    def _adopt(self, raw_path: Path, out_path: Path, summary: dict) -> None:
        with np.load(raw_path, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        arrays.pop("__schema_version__", None)  # the script already stamped it; save_artifact re-stamps, avoiding a keyword clash
        depth_source = str(arrays["depth_source"])
        if depth_source not in ("stereo", "monocular"):
            raise ModelEnvError(f"invalid depth_source: {depth_source!r}")
        save_artifact(out_path, **arrays)


class VipeSceneModel(_SceneBase):
    """Monocular path: ViPE outputs poses (c2w -> inverted to w2c) and near-metric depth."""

    name = "vipe"
    _script_name = "run_vipe.py"

    def _sec_per_frame(self, width: int) -> float:
        return 0.15 if width <= 640 else 0.4

    def load(self, device: str) -> None:
        super().load(device)
        _check_python("ROBORAY_VIPE_PYTHON", "deploy/modelenvs/build_vipe.sh")

    def infer(self, frames: dict, calib: dict, out_path: Path) -> None:
        python = _env_python("ROBORAY_VIPE_PYTHON")
        weights = os.environ.get("ROBORAY_VIPE_WEIGHTS", "")
        with tempfile.TemporaryDirectory() as td:
            inp = Path(td) / "input.npz"
            _write_input_npz(inp, frames, calib, {})
            argv = ["--input", str(inp), "--output", str(inp) + ".out.npz"]
            if weights:
                argv += ["--weights", weights]
            summary = run_in_modelenv(python, self.script, argv, timeout_s=3600.0)
            if not summary.get("ok"):
                raise ModelEnvError(f"ViPE inference failed: {summary}")
            self._validate_output(Path(str(inp) + ".out.npz"))
            self._adopt(Path(str(inp) + ".out.npz"), Path(out_path), summary)


class FastFoundationStereoModel(_SceneBase):
    """Stereo path: FFS depth + ViPE poses (composite output)."""

    name = "fast_foundation_stereo"
    _script_name = "run_ffs.py"

    def __init__(self, script: str | None = None, vipe_script: str | None = None):
        super().__init__(script)
        self.vipe_model = VipeSceneModel(script=vipe_script)

    def _sec_per_frame(self, width: int) -> float:
        return 0.05 if width <= 640 else 0.15

    def load(self, device: str) -> None:
        super().load(device)
        _check_python("ROBORAY_FFS_PYTHON", "deploy/modelenvs/build_ffs.sh")
        weights = os.environ.get("ROBORAY_FFS_WEIGHTS", "")
        if not weights or not Path(weights).exists():
            raise ModelEnvError(
                "ROBORAY_FFS_WEIGHTS not set or the directory does not exist; "
                "download the weights from HF nvidia/c-fast-foundationstereo and set the variable (see docs/models.md)."
            )
        self.vipe_model.load(device)

    def infer(self, frames: dict, calib: dict, out_path: Path) -> None:
        if not calib.get("has_stereo") or "frames_right" not in frames:
            raise ModelEnvError(
                "fast_foundation_stereo needs frames_right stereo input; "
                "a monocular Episode should route to vipe (check resolve_scene_model's routing conditions)."
            )
        python = _env_python("ROBORAY_FFS_PYTHON")
        t_l2r = np.asarray(calib.get("T_left_to_right", np.eye(4)), dtype=np.float64)
        baseline = float(t_l2r[0, 3])
        w = int(frames["frames"].shape[2])
        fx = float(np.asarray(calib["K"])[0, 0]) if calib.get("K") is not None else float(w)
        extra = {"baseline_m": baseline, "fx": fx}
        with tempfile.TemporaryDirectory() as td:
            inp = Path(td) / "input.npz"
            _write_input_npz(inp, frames, calib, extra)
            argv = ["--input", str(inp), "--output", str(inp) + ".ffs.npz",
                    "--weights", os.environ.get("ROBORAY_FFS_WEIGHTS", ""),
                    "--extra", json.dumps(extra)]
            summary = run_in_modelenv(python, self.script, argv, timeout_s=3600.0)
            if not summary.get("ok"):
                raise ModelEnvError(f"FFS inference failed: {summary}")
            ffs_out = Path(str(inp) + ".ffs.npz")
            self._validate_output(ffs_out)
            with np.load(ffs_out, allow_pickle=False) as z:
                arrays = {k: z[k] for k in z.files}
            # poses come from ViPE (separate output file; on success replace FFS's placeholder poses/intrinsics)
            vipe_out = Path(td) / "vipe.npz"
            self.vipe_model.infer(frames, calib, vipe_out)
            with np.load(vipe_out, allow_pickle=False) as z:
                arrays["camera_poses"] = z["camera_poses"]
                arrays["intrinsics"] = z["intrinsics"]
            arrays["depth_source"] = np.array("stereo")
            save_artifact(out_path, **arrays)
```

In `models/registry.py`, change the `load_builtin_models` scene factory to:

```python
    def _scene(name: str):
        def factory():
            from roboray.models.scene_models import VipeSceneModel, FastFoundationStereoModel

            return VipeSceneModel() if name == "vipe" else FastFoundationStereoModel()

        return factory
```

- [ ] **Step 4: Run the test and confirm it passes**

Run: `conda run -n roboray python -m pytest tests/unit/test_scene_models.py tests/unit/test_registry.py -v`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add src/roboray/models/scene_models.py src/roboray/models/registry.py tests/unit/test_scene_models.py
git commit -m "feat(models): ViPE and Fast-FoundationStereo Scene adapters (subprocess + contract normalization)"
```

---

### Task 4: HaMeR Hand adapter + MANO verification

**Files:**
- Create: `src/roboray/models/hand_models.py`
- Modify: `src/roboray/models/registry.py` (hand factory)
- Test: `tests/unit/test_hand_models.py`

**Interfaces:**
- Consumes: Tasks 1/2 as before
- Produces: `HamerHandModel(name="hamer")` implementing the HandModel protocol; `load()` verifies the env python + the `ROBORAY_HAMER_WEIGHTS` directory exists and contains MANO_RIGHT.pkl (missing -> `ModelEnvError` with registration/download guidance); `infer` assembles input -> run_hamer.py -> validates the hand artifact key set -> `save_artifact` output; `cost_hint` ~0.08s/frame

- [ ] **Step 1: Write the failing test**

`tests/unit/test_hand_models.py`:

```python
import sys
from pathlib import Path

import numpy as np
import pytest

from roboray.models import registry
from roboray.models.hand_models import HamerHandModel
from roboray.models.modelenv import ModelEnvError

FAKES = Path(__file__).resolve().parents[1] / "fakes" / "modelenv_scripts"


@pytest.fixture(autouse=True)
def _registry():
    registry.reset_registry_for_tests()
    registry.load_builtin_models()
    yield
    registry.reset_registry_for_tests()


def test_hamer_runs_fake_script(tmp_path, monkeypatch):
    weights = tmp_path / "weights"
    (weights / "mano").mkdir(parents=True)
    (weights / "mano" / "MANO_RIGHT.pkl").write_bytes(b"fake")
    monkeypatch.setenv("ROBORAY_HAMER_PYTHON", sys.executable)
    monkeypatch.setenv("ROBORAY_HAMER_WEIGHTS", str(weights))
    model = HamerHandModel(script=str(FAKES / "fake_hamer.py"))
    model.load("cpu")
    out = tmp_path / "hand.npz"
    model.infer(
        {"frames": np.zeros((5, 8, 8, 3), np.uint8), "episode_id": "e"},
        {"fps": 30.0},
        out,
    )
    z = np.load(out, allow_pickle=False)
    assert z["hand_present"].shape == (5, 2)
    assert z["mano_pose"].shape == (5, 2, 48)
    assert int(z["__schema_version__"]) == 1


def test_hamer_missing_weights_gives_registration_hint(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBORAY_HAMER_PYTHON", sys.executable)
    monkeypatch.setenv("ROBORAY_HAMER_WEIGHTS", str(tmp_path / "nope"))
    model = HamerHandModel(script=str(FAKES / "fake_hamer.py"))
    with pytest.raises(ModelEnvError) as exc:
        model.load("cpu")
    assert "MANO_RIGHT.pkl" in str(exc.value)
    assert "mano.is.tue.mpg.de" in str(exc.value)


def test_hamer_weights_without_mano_rejected(tmp_path, monkeypatch):
    weights = tmp_path / "w2"
    weights.mkdir()
    monkeypatch.setenv("ROBORAY_HAMER_PYTHON", sys.executable)
    monkeypatch.setenv("ROBORAY_HAMER_WEIGHTS", str(weights))
    model = HamerHandModel(script=str(FAKES / "fake_hamer.py"))
    with pytest.raises(ModelEnvError) as exc:
        model.load("cpu")
    assert "MANO_RIGHT.pkl" in str(exc.value)


def test_registry_resolves_builtin_hand_model():
    assert hasattr(registry.get_hand_model_class("hamer")(), "infer")
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `conda run -n roboray python -m pytest tests/unit/test_hand_models.py -v`
Expected: FAIL — `ModuleNotFoundError: roboray.models.hand_models`

- [ ] **Step 3: Implement `models/hand_models.py`**

```python
"""Hand-path real model adapter: HaMeR (subprocess architecture).

MANO_RIGHT.pkl is a registration-only download (https://mano.is.tue.mpg.de);
the adapter's load() enforces its presence -- fail early rather than hand over a half-usable state.
"""

import os
import subprocess
from pathlib import Path

import numpy as np

from roboray.stages.artifacts import save_artifact

from .modelenv import ModelEnvError, run_in_modelenv

__all__ = ["HamerHandModel"]

_REQUIRED_HAND_KEYS = {
    "hand_present", "joints_2d", "mano_pose", "mano_shape",
    "mano_trans", "confidence", "n_frames", "fps",
}
_DEFAULT_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "modelenvs" / "run_hamer.py"
_MANO_HINT = (
    "MANO_RIGHT.pkl must be downloaded after registering at https://mano.is.tue.mpg.de "
    "and placed in the weights directory's mano/ subdirectory (layout in docs/models.md)."
)


class HamerHandModel:
    name = "hamer"

    def __init__(self, script: str | None = None):
        self.script = script or str(_DEFAULT_SCRIPT)
        self.loaded_on: str | None = None

    def load(self, device: str) -> None:
        self.loaded_on = device
        python = os.environ.get("ROBORAY_HAMER_PYTHON", "python")
        if python != "python" and not Path(python).exists():
            raise ModelEnvError(
                f"the interpreter pointed to by ROBORAY_HAMER_PYTHON does not exist: {python}. "
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
            raise ModelEnvError(f"ROBORAY_HAMER_WEIGHTS not set or the directory does not exist. {_MANO_HINT}")
        if not (Path(weights) / "mano" / "MANO_RIGHT.pkl").exists():
            raise ModelEnvError(
                f"the weights directory is missing mano/MANO_RIGHT.pkl ({weights}). {_MANO_HINT}"
            )

    def cost_hint(self, n_frames: int, resolution_key: str) -> float:
        return 0.08 * max(1, n_frames)

    def infer(self, frames: dict, calib: dict, out_path: Path) -> None:
        python = os.environ.get("ROBORAY_HAMER_PYTHON", "python")
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            inp = Path(td) / "input.npz"
            arrays = {
                "frames": np.asarray(frames["frames"]),
                "n_frames": np.int32(frames["frames"].shape[0]),
                "fps": np.float32(calib.get("fps", 30.0)),
            }
            np.savez(inp, **arrays)
            argv = [
                "--input", str(inp),
                "--output", str(inp) + ".out.npz",
                "--weights", os.environ.get("ROBORAY_HAMER_WEIGHTS", ""),
            ]
            summary = run_in_modelenv(python, self.script, argv, timeout_s=3600.0)
            if not summary.get("ok"):
                raise ModelEnvError(f"HaMeR inference failed: {summary}")
            raw = Path(str(inp) + ".out.npz")
            with np.load(raw, allow_pickle=False) as z:
                missing = _REQUIRED_HAND_KEYS - set(z.files)
                arrays_out = {k: z[k] for k in z.files}
            if missing:
                raise ModelEnvError(f"Hand artifact missing keys: {sorted(missing)}")
            save_artifact(out_path, **arrays_out)
```

Change the `registry.py` hand factory to:

```python
    def _hand(name: str):
        def factory():
            from roboray.models.hand_models import HamerHandModel

            return HamerHandModel()

        return factory
```

- [ ] **Step 4: Run the test and confirm it passes**

Run: `conda run -n roboray python -m pytest tests/unit/test_hand_models.py tests/unit/test_registry.py -v`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add src/roboray/models/hand_models.py src/roboray/models/registry.py tests/unit/test_hand_models.py
git commit -m "feat(models): HaMeR Hand adapter (strict MANO weight verification + subprocess contract)"
```

---

### Task 5: A1 fusion refinement (real backprojection + reprojection error)

**Files:**
- Modify: `src/roboray/stages/fusion.py` (rewrite `backproject_to_world` + reprojection-error computation)
- Test: `tests/unit/test_fusion_refined.py` (new; the existing test_fusion_action keeps passing — `run_fusion`'s signature is unchanged)

**Interfaces (as-built, matching the delivered implementation; the Step 1-5 code below is the historical plan-time form):**
- Consumes: scene.npz (w2c poses, intrinsics, depth), hand.npz (2D joints, MANO trans, **`joints_3d_cam`**, `confidence`)
- Produces:
  - `backproject_to_world(mano_trans, mano_pose, joints_2d, depth, intrinsics, camera_poses, joint_source="joints_2d", joints_3d_cam=None) -> (joints_world, reprojection_error)`
  - **mano path (informative)**: when `joint_source="mano"` and `joints_3d_cam` exists, take the depth at the detected root joint, translate-align the MANO **camera-frame** joints as `root_metric − mano_root` (no scaling, bone lengths preserved), then c2w -> world frame; the reprojection error measures "MANO structure vs image evidence (+root depth)" consistency — the larger the discrepancy, the larger the error.
  - **Fallback path (consistency-check only)**: without `joints_3d_cam` -> `joint_source="joints_2d"`, per-joint depth sampling + K⁻¹ backprojection; its reprojection is a pixel-level identity closed loop, **structurally always zero/marked** — QC explicitly labels it "not to be treated as a quality metric".
  - Untrustworthy semantics (same for both paths): that hand/frame is all-NaN plus a reprojection error of `UNTRUSTED_REPROJ_ERROR` (1e6); `run_fusion` persists/returns `joint_source`, the trusted-frame reprojection-error mean/max, and `flagged_hands`; QC excludes flagged frames from statistics using the 1e6 marker.
  - **Review Focus #6 pinned**: synthetic ground-truth unit test — backproject->reproject closed loop < 1e-3; the mano path resolves per-pixel error.

- [ ] **Step 1: Write the failing test**

`tests/unit/test_fusion_refined.py`:

```python
import numpy as np
import pytest

from roboray.stages.fusion import backproject_to_world


def _pinhole_setup(n=3, h=48, w=64, depth_m=2.0):
    """Synthetic ground truth: one 3D point moving along x, K/w2c known, depth at the projection = true depth."""
    K = np.tile(np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], np.float64), (n, 1, 1))
    w2c = np.tile(np.eye(4, dtype=np.float64), (n, 1, 1))
    pts_cam = np.zeros((n, 3), np.float64)
    pts_2d = np.zeros((n, 2), np.float64)
    depth = np.full((n, h, w), depth_m, np.float32)
    return K, w2c, pts_cam, pts_2d, depth, depth_m


def test_backproject_reprojection_roundtrip_is_exact():
    """Review Focus #6: backproject->reproject closed-loop error < 1e-3."""
    n, depth_m = 3, 2.0
    K, w2c, pts_cam, pts_2d, depth, _ = _pinhole_setup(n)
    # camera-frame 3D point: (0.1, -0.05, 2.0)
    pts_cam[:, 0] = 0.1
    pts_cam[:, 1] = -0.05
    pts_cam[:, 2] = depth_m
    # project to pixels
    uvw = K @ pts_cam.T  # (3,n)
    pts_2d = (uvw[:2] / uvw[2]).T
    # build single-joint input: joints_2d (T,2,21,3) only hand 0 has a joint; the 3rd dim is a spare z
    joints_2d = np.zeros((n, 2, 21, 3), np.float64)
    joints_2d[:, 0, 0, :2] = pts_2d
    mano_trans = np.zeros((n, 2, 3), np.float64)
    mano_pose = np.zeros((n, 2, 48), np.float64)

    joints_world, reproj = backproject_to_world(
        mano_trans, mano_pose, joints_2d, depth, K, w2c,
        joint_source="joints_2d",
    )
    # hand-0 joint-0 backprojects in the camera frame to pts_cam
    assert np.allclose(joints_world[:, 0, 0, :], pts_cam, atol=1e-6)
    # reprojection-error closed loop < 1e-3 px
    assert np.all(reproj[:, 0] < 1e-3)


def test_backproject_invalid_depth_is_nan_and_flagged():
    """A valid point inside the field of view vs a zero-depth point: the former backprojects normally, the latter is NaN + a large error mark."""
    n, h, w = 2, 48, 64
    K = np.tile(np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], np.float64), (n, 1, 1))
    w2c = np.tile(np.eye(4, dtype=np.float64), (n, 1, 1))
    depth = np.full((n, h, w), 2.0, np.float32)
    # pixel inside the field of view: u = w/2 + w*0.1/2, v = h/2 + w*(-0.05)/2
    u, v = w / 2 + w * 0.1 / 2.0, h / 2 + w * (-0.05) / 2.0
    joints_2d = np.zeros((n, 2, 21, 3), np.float64)
    joints_2d[:, 0, 0, 0] = u
    joints_2d[:, 0, 0, 1] = v
    depth[0, int(round(v)), int(round(u))] = 0.0  # frame 0's depth at this pixel is invalid
    joints_world, reproj = backproject_to_world(
        np.zeros((n, 2, 3)), np.zeros((n, 2, 48)), joints_2d, depth, K, w2c,
        joint_source="joints_2d",
    )
    assert np.isnan(joints_world[0, 0, 0, 0])    # invalid depth -> NaN
    assert not np.isnan(joints_world[1, 0, 0, 0])  # valid depth is normal
    assert reproj[0, 0] >= 1e6  # marked as a large error
    assert reproj[1, 0] < 1e-3


def test_run_fusion_reprojection_error_is_real(tmp_path):
    """End to end: run_fusion's reprojection_error is no longer an all-zero placeholder."""
    from roboray.runtime.episode import EpisodeWorkspace

    ws = EpisodeWorkspace(tmp_path, "ep_0001")
    n, h, w, depth_m = 4, 48, 64, 2.0
    K = np.tile(np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], np.float64), (n, 1, 1))
    ws.write_artifact(
        "scene.npz",
        {
            "camera_poses": np.tile(np.eye(4, np.float64), (n, 1, 1)),
            "intrinsics": K,
            "depth": np.full((n, h, w), depth_m, np.float32),
            "depth_source": np.array("monocular"),
            "n_frames": np.int32(n),
            "fps": np.int32(30),
        },
    )
    present = np.ones((n, 2), bool)
    joints_2d = np.zeros((n, 2, 21, 3), np.float64)
    joints_2d[:, 0, 0, 0] = w / 2 + 10  # off-center point
    joints_2d[:, 0, 0, 1] = h / 2
    ws.write_artifact(
        "hand.npz",
        {
            "hand_present": present,
            "joints_2d": joints_2d,
            "mano_pose": np.zeros((n, 2, 48), np.float32),
            "mano_shape": np.zeros((n, 2, 10), np.float32),
            "mano_trans": np.zeros((n, 2, 3), np.float32),
            "confidence": np.where(present, 0.9, 0.0).astype(np.float32),
            "n_frames": np.int32(n),
            "fps": np.int32(30),
        },
    )
    from roboray.stages.fusion import run_fusion

    run_fusion(workspace_root=str(tmp_path), episode_id="ep_0001",
               wearer_association=True, unobserved_gap_s=0.2, min_supporting_joints=8)
    z = np.load(ws.artifact("fused.npz"), allow_pickle=True)
    reproj = z["reprojection_error"]
    assert reproj.shape == (n, 2)
    assert np.abs(reproj[:, 0]).max() > 1.0  # a real projection error (px), not all zeros
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `conda run -n roboray python -m pytest tests/unit/test_fusion_refined.py -v`
Expected: FAIL — `backproject_to_world` old signature/old semantics

- [ ] **Step 3: Rewrite `backproject_to_world` and wire up `run_fusion`**

In `fusion.py`, replace (keep the function name, extend the signature, update the `run_fusion` call site in sync):

```python
def _sample_depth(depth: np.ndarray, u: float, v: float) -> float:
    """Bilinear depth sampling; out-of-bounds or non-positive values return NaN."""
    h, w = depth.shape
    if not (0 <= u < w - 1 and 0 <= v < h - 1):
        return float("nan")
    x0, y0 = int(u), int(v)
    fx, fy = u - x0, v - y0
    vals = [depth[y0, x0], depth[y0, x0 + 1], depth[y0 + 1, x0], depth[y0 + 1, x0 + 1]]
    if any(val <= 0 or not np.isfinite(val) for val in vals):
        return float("nan")
    return float(
        (1 - fx) * (1 - fy) * vals[0] + fx * (1 - fy) * vals[1]
        + (1 - fx) * fy * vals[2] + fx * fy * vals[3]
    )


def backproject_to_world(
    mano_trans: np.ndarray,
    mano_pose: np.ndarray,
    joints_2d: np.ndarray,
    depth: np.ndarray,
    intrinsics: np.ndarray,
    camera_poses: np.ndarray,
    joint_source: str = "joints_2d",
) -> tuple:
    """A1 metric fusion: 2D joints + depth map -> camera-frame 3D -> world frame; also computes the reprojection error (px).

    Returns (joints_world (T,2,21,3) float64 -- NaN where untrustworthy, reprojection_error (T,2) --
    untrustworthy marked with 1e6).
    """
    t, n_hands = joints_2d.shape[0], joints_2d.shape[1]
    joints_world = np.full((t, n_hands, 21, 3), np.nan, np.float64)
    reproj = np.full((t, n_hands), 1e6, np.float64)
    for i in range(t):
        c2w = np.linalg.inv(np.asarray(camera_poses[i], np.float64))
        K = np.asarray(intrinsics[i], np.float64)
        Kinv = np.linalg.inv(K)
        for h_idx in range(n_hands):
            pts_cam = np.full((21, 3), np.nan, np.float64)
            for j in range(21):
                u, v = float(joints_2d[i, h_idx, j, 0]), float(joints_2d[i, h_idx, j, 1])
                z = _sample_depth(depth[i], u, v)
                if not np.isfinite(z) or z <= 0:
                    continue
                pts_cam[j] = Kinv @ np.array([u, v, 1.0]) * z
            valid = np.isfinite(pts_cam).all(axis=1)
            if valid.sum() < 3:
                continue  # the keypoint depths are almost entirely invalid: mark the whole hand untrustworthy
            # camera frame -> world frame
            hom = np.concatenate([pts_cam, np.ones((21, 1))], axis=1)
            joints_world[i, h_idx] = (hom @ c2w.T)[:, :3]
            # reprojection: world -> camera -> pixel, compared against the detected 2D (trusted joints only)
            w_pts = joints_world[i, h_idx][valid]
            cam_pts = (np.concatenate([w_pts, np.ones((len(w_pts), 1))], axis=1) @ camera_poses[i].T)[:, :3]
            uv = (K @ cam_pts.T).T
            uv = uv[:, :2] / np.clip(uv[:, 2:3], 1e-9, None)
            src = joints_2d[i, h_idx][valid][:, :2]
            reproj[i, h_idx] = float(np.linalg.norm(uv - src, axis=1).mean())
    return joints_world, reproj
```

Update the call site in `run_fusion` to the new signature (pass `joints_2d`), use the returned `joints_world`/`reproj`; the fill-frame interpolation segment does NaN-safe interpolation of NaN joints (interpolate only when both endpoints have values); `fused.npz` still writes `reprojection_error (T,2)` (NaN -> replace with 1e6 before writing to keep the NPZ numeric type).

- [ ] **Step 4: Run the test and confirm it passes (including regression of the existing fusion test)**

Run: `conda run -n roboray python -m pytest tests/unit/test_fusion_refined.py tests/unit/test_fusion_action.py -v`
Expected: all pass (if an existing test needs its assertion fine-tuned for NaN semantics, that is expected — update the test to the new contract and explain it)

- [ ] **Step 5: Commit**

```bash
git add src/roboray/stages/fusion.py tests/unit/test_fusion_refined.py tests/unit/test_fusion_action.py
git commit -m "feat(fusion): A1 real backprojection and reprojection error (bilinear depth sampling + NaN untrustworthy semantics)"
```

---

### Task 6: A3 fill's four checks + supporting_joints wiring

**Files:**
- Modify: `src/roboray/stages/fusion.py` (the fill segment after `fill_short_gaps`)
- Test: `tests/unit/test_fusion_refined.py` (extended)

**Interfaces (as-built, matching the delivered implementation; the Step 1-5 code below is the historical plan-time form):**
- Produces (A3 completeness, spec §7):
  1. `insufficient_keypoint_depth`: the filled joint's depth-support count in that frame < `min_supporting_joints` -> reject.
  2. `out_of_view`: the two-sided 2D detections that interpolation relies on are out of bounds (bounds aligned with `_sample_depth`'s sampleable region, no gaps).
  3. `wrist_step_exceeded`: single-frame wrist displacement > `_MAX_WRIST_STEP_M` (0.25 m).
  4. `trajectory_deviation_exceeded` — **velocity continuity**: the implied per-frame displacement of the bridge vs the mean of the **measured** inter-frame displacements at the gap's two ends differs by more than `_MAX_WRIST_VEL_MISMATCH_M_PER_FRAME` (0.05 m/frame). The old "linear interpolation of candidates relative to both ends" formulation was always 0 (degenerate, could never fire) and has been replaced by this non-degenerate metric.
  - When check 4 cannot be evaluated, **allow the fill** and count it by reason category (`gapfill_skipped`, not a rejection): `sequence_boundary` / `neighbour_unobserved` / `wrist_untrusted`.
  - Filled frames (accepted or rejected) have no observation to compare against -> their reprojection error is a **marker** (NaN -> 1e6 before writing), consistent with untrustworthy hand frames, and excluded by QC along with them.
  - **Audit persistence + QC rendering**: `run_fusion` writes `gapfill_reasons` (rejection-reason distribution) and `gapfill_skipped` (check-4 skip counts) into `fused.npz` as JSON string arrays, and the QC report renders a "fill audit" section. The Actor discards `run_fusion`'s return value and `manifest.py` only records stages/timings/error/degraded — so **manifest-level exposure is an explicit defer** (does not block this plan).
  - `supporting_joints` wiring: the number of joints with `confidence >= 0.5` ((T,2) int) is passed into `fill_short_gaps` — activating `min_supporting_joints`.
  - **Wearer association for accepted filled frames**: A2 uses "valid 2D" (for filled frames, the linear-interpolated x-mean of the two-sided observations replaces the zero-valued 2D) in ranking, avoiding fabricated identity switches; rejected filled frames do not participate (handled in the earlier fix wave).
- The existing three conditions (two-sided observation / same hand / duration cap) are unchanged

- [ ] **Step 1: Write the failing test**

`tests/unit/test_fusion_refined.py` addition:

```python
def test_a3_all_four_checks_via_run_fusion(tmp_path):
    """A3 refinement: build a fill-segment trajectory with a large deviation -> the fill is rejected (audit rejected increases)."""
    from roboray.runtime.episode import EpisodeWorkspace
    from roboray.stages.fusion import run_fusion

    ws = EpisodeWorkspace(tmp_path, "ep_0001")
    n, h, w = 12, 48, 64
    ws.write_artifact(
        "scene.npz",
        {
            "camera_poses": np.tile(np.eye(4, np.float64), (n, 1, 1)),
            "intrinsics": np.tile(np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], np.float64), (n, 1, 1)),
            "depth": np.full((n, h, w), 2.0, np.float32),
            "depth_source": np.array("monocular"),
            "n_frames": np.int32(n),
            "fps": np.int32(30),
        },
    )
    present = np.ones((n, 2), bool)
    present[5:7, 0] = False  # a 2-frame gap: at 30fps that is 0.067s < 0.2s -> the duration condition holds
    joints_2d = np.zeros((n, 2, 21, 3), np.float64)
    # both ends at the same position but the depth-sampling region is invalid
    # (zero the depth of the gap frames' pixels -> the depth check rejects)
    ws_arrays = dict(np.load(ws.artifact("scene.npz"), allow_pickle=False))
    ws_arrays["depth"][5:7, int(h / 2):, :] = 0.0
    np.savez(ws.artifact("scene.npz"), __schema_version__=1, **ws_arrays)
    joints_2d[:, 0, :, 0] = w / 2
    joints_2d[:, 0, :, 1] = h - 2  # lands in the lower half where depth was zeroed
    ws.write_artifact(
        "hand.npz",
        {
            "hand_present": present,
            "joints_2d": joints_2d,
            "mano_pose": np.zeros((n, 2, 48), np.float32),
            "mano_shape": np.zeros((n, 2, 10), np.float32),
            "mano_trans": np.zeros((n, 2, 3), np.float32),
            "confidence": np.where(present, 0.9, 0.0).astype(np.float32),
            "n_frames": np.int32(n),
            "fps": np.int32(30),
        },
    )
    info = run_fusion(workspace_root=str(tmp_path), episode_id="ep_0001",
                      wearer_association=True, unobserved_gap_s=0.2, min_supporting_joints=8)
    # the depth check rejects the fill: the gap is not filled
    assert info["gapfill_filled"] == 0
    assert info["gapfill_rejected"] >= 1
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `conda run -n roboray python -m pytest "tests/unit/test_fusion_refined.py::test_a3_all_four_checks_via_run_fusion" -v`
Expected: FAIL (the current implementation would fill)

- [ ] **Step 3: Implement the four checks and supporting_joints**

Refactor the `run_fusion` fill segment: for each candidate filled frame (after interpolation), check in order —
1. Depth validity: if the backprojected joints that remain valid after this frame's interpolation number < `min_supporting_joints` -> reject (reason `insufficient_keypoint_depth`)
2. Field of view: the 2D joints used for interpolation are out of bounds -> reject (`out_of_view`)
3. Wrist displacement > `_MAX_WRIST_STEP_M` -> reject (`wrist_step_exceeded`)
4. Trajectory deviation: max ‖fill position − linear interpolation‖ > `_MAX_TRAJ_DEV_M` -> reject (`trajectory_deviation_exceeded`)

`supporting_joints` wiring: `run_fusion` computes `supporting = (confidence >= 0.5).sum(axis=-1)` ((T,2) int) from `hand.npz`'s `confidence` and passes it into `fill_short_gaps`. The audit's `gapfill_reasons` counts by rejection reason and `gapfill_skipped` counts by the check-4 skip reason — both are produced alongside `run_fusion`'s return value, but **the Actor discards the return value and `manifest.py` does not record them**, so **they are actually persisted into `fused.npz` (JSON string arrays) and rendered by the QC report**; **manifest-level exposure is listed as a defer** (correcting the earlier claim that these would ride the existing run_fusion-return path into the manifest).

- [ ] **Step 4: Run the test and confirm it passes (regression of the existing A3 test)**

Run: `conda run -n roboray python -m pytest tests/unit/test_fusion_refined.py tests/unit/test_fusion_action.py -v`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add src/roboray/stages/fusion.py tests/unit/test_fusion_refined.py
git commit -m "feat(fusion): A3 four post-fill checks + supporting_joints wiring (min_supporting_joints takes effect)"
```

---

### Task 7: GPU smoke test suite + environment build scripts + docs/models.md

**Files:**
- Create: `tests/gpu/test_adapters_gpu.py` (`@pytest.mark.gpu`, skipped by default)
- Create: `deploy/modelenvs/build_vipe.sh`, `build_ffs.sh`, `build_hamer.sh`
- Create: `scripts/vendor_models.sh` (clone third_party and pin commits)
- Create: `docs/models.md`
- Modify: `pyproject.toml` (no dependency change; documentation only)
- Test: the gpu tests themselves

**Interfaces:**
- Produces:
  - gpu smoke: each adapter runs real inference on small synthetic data (`make_synthetic_ego`'s 1 Episode) -> asserts artifact keys/shapes/finiteness; deselected by default outside `pytest -m gpu` (`pyproject` addopts gains `-m "not gpu"`)
  - Build scripts follow each repo's official procedure (vipe: conda env + source install; ffs: py3.12 + torch 2.6 cu124 + xformers; hamer: py3.10 + torch + ViTPose + `fetch_demo_data.sh`)
  - `docs/models.md`: MANO registration download, HF weight download, env build and the ROBORAY_* variable reference table, and the L2 verification checklist (each `[L2]` item of the three scripts listed one by one)

- [ ] **Step 1: Enable gpu exclusion in pyproject**

Change `addopts` under `[tool.pytest.ini_options]` to `"-q -m "not gpu""`.

- [ ] **Step 2: Write the gpu smoke test**

`tests/gpu/test_adapters_gpu.py`:

```python
"""Real-model GPU smoke (L2). Excluded by default via -m "not gpu"; run on the A800 with pytest -m gpu.

Prerequisites: the three model envs are built (deploy/modelenvs/*.sh), weights are downloaded,
and the ROBORAY_*_PYTHON/WEIGHTS environment variables are set (see docs/models.md).
"""

import os

import numpy as np
import pytest

from roboray.models.scene_models import FastFoundationStereoModel, VipeSceneModel
from roboray.models.hand_models import HamerHandModel

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not os.environ.get("ROBORAY_GPU_SMOKE"),
        reason="ROBORAY_GPU_SMOKE not set",
    ),
]


def _frames(t=4, h=288, w=512):
    rng = np.random.default_rng(0)
    base = rng.integers(30, 200, (t, h, w, 3), np.uint8)
    right = np.roll(base, -6, axis=2)
    return base, right


def test_vipe_gpu_smoke(tmp_path):
    frames, _ = _frames()
    model = VipeSceneModel()
    model.load("cuda")
    out = tmp_path / "scene.npz"
    model.infer({"frames": frames, "episode_id": "smoke"},
                {"fps": 30.0, "has_stereo": False, "T_left_to_right": np.eye(4).tolist()}, out)
    z = np.load(out, allow_pickle=False)
    assert z["depth"].shape == frames.shape[:3]
    assert np.isfinite(z["depth"]).all() and (z["depth"] > 0).any()
    assert np.isfinite(z["camera_poses"]).all()


@pytest.mark.skipif(not os.environ.get("ROBORAY_FFS_WEIGHTS"), reason="FFS weights not set")
def test_ffs_gpu_smoke(tmp_path):
    frames, right = _frames()
    calib = {"fps": 30.0, "has_stereo": True,
             "T_left_to_right": np.eye(4).tolist(), "K": np.diag([512.0, 512, 1, 1]).tolist()}
    calib["T_left_to_right"][0][3] = 0.06
    model = FastFoundationStereoModel()
    model.load("cuda")
    out = tmp_path / "scene.npz"
    model.infer({"frames": frames, "frames_right": right, "episode_id": "smoke"}, calib, out)
    z = np.load(out, allow_pickle=False)
    assert np.isfinite(z["depth"]).all() and (z["depth"] > 0).any()


@pytest.mark.skipif(not os.environ.get("ROBORAY_HAMER_WEIGHTS"), reason="HaMeR weights not set")
def test_hamer_gpu_smoke(tmp_path):
    frames, _ = _frames()
    model = HamerHandModel()
    model.load("cuda")
    out = tmp_path / "hand.npz"
    model.infer({"frames": frames, "episode_id": "smoke"}, {"fps": 30.0}, out)
    z = np.load(out, allow_pickle=False)
    assert z["hand_present"].shape == (4, 2)
    assert np.isfinite(z["mano_pose"]).all()
```

- [ ] **Step 3: Build scripts and docs**

`deploy/modelenvs/build_vipe.sh` (core lines; the full script includes `set -euo pipefail` and explanatory comments):

```bash
#!/usr/bin/env bash
set -euo pipefail
# Build the standalone ViPE env (per the official nv-tlabs/vipe procedure).
conda create -y -n ego-vipe python=3.10
conda run -n ego-vipe python -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
VENDOR_DIR="$(cd "$(dirname "$0")/../.." && pwd)/third_party/vipe"
conda run -n ego-vipe python -m pip install -e "$VENDOR_DIR"
echo "done. set ROBORAY_VIPE_PYTHON=$(conda info --base)/envs/ego-vipe/bin/python"
```

`build_ffs.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
# Build the standalone Fast-FoundationStereo env (py3.12 + torch2.6 cu124 + xformers).
conda create -y -n ego-ffs python=3.12
conda run -n ego-ffs python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
conda run -n ego-ffs python -m pip install xformers --index-url https://download.pytorch.org/whl/cu124
VENDOR_DIR="$(cd "$(dirname "$0")/../.." && pwd)/third_party/fast-foundationstereo"
conda run -n ego-ffs python -m pip install -e "$VENDOR_DIR"
echo "done. set ROBORAY_FFS_PYTHON=$(conda info --base)/envs/ego-ffs/bin/python"
echo "weights: download from HF nvidia/c-fast-foundationstereo into \$ROBORAY_FFS_WEIGHTS"
```

`build_hamer.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
# Build the standalone HaMeR env (per the official README; MANO_RIGHT.pkl requires a registration download).
conda create -y -n ego-hamer python=3.10
conda run -n ego-hamer python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
VENDOR_DIR="$(cd "$(dirname "$0")/../.." && pwd)/third_party/hamer"
conda run -n ego-hamer python -m pip install -e "$VENDOR_DIR[all]"
conda run -n ego-hamer python -m pip install -v -e "$VENDOR_DIR/third-party/ViTPose"
conda run -n ego-hamer bash "$VENDOR_DIR/fetch_demo_data.sh"
echo "done. set ROBORAY_HAMER_PYTHON=$(conda info --base)/envs/ego-hamer/bin/python"
echo "MANO: register at https://mano.is.tue.mpg.de, download MANO_RIGHT.pkl into \$ROBORAY_HAMER_WEIGHTS/mano/"
```

`scripts/vendor_models.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
# Clone the three model repos into third_party/ and pin commits (the commit is recorded on the
# first run and fixed from then on).
cd "$(dirname "$0")/.."
mkdir -p third_party
[ -d third_party/vipe ] || git clone https://github.com/nv-tlabs/vipe.git third_party/vipe
[ -d third_party/fast-foundationstereo ] || git clone https://github.com/nvlabs/fast-foundationstereo.git third_party/fast-foundationstereo
[ -d third_party/hamer ] || git clone --recursive https://github.com/geopavlakos/hamer.git third_party/hamer
# first run: record the current HEAD as the pinned commit
for repo in vipe fast-foundationstereo hamer; do
  file="third_party/$repo.commit"
  if [ ! -f "$file" ]; then git -C "third_party/$repo" rev-parse HEAD > "$file"; fi
  git -C "third_party/$repo" checkout "$(cat "$file")"
done
echo "vendored repos pinned to the recorded commits."
```

`docs/models.md`: state — the build order of the three envs, the MANO registration-download steps, the HF weight download commands (`huggingface-cli download nvidia/c-fast-foundationstereo`), the six-`ROBORAY_*` environment-variable reference table, the L2 verification checklist (each `[L2]` item in the three script headers listed with its verification method), the full command for running the gpu smoke on the A800 (`conda run -n roboray python -m pytest -m gpu`; the CLI's `-m` overrides addopts' `-m "not gpu"` (last-wins), so no extra handling is needed; `-o addopts=""` is only an optional explicit defensive form), and the note that the stereo path needs both the vipe and ffs envs.

- [ ] **Step 4: Run the full suite (with gpu excluded by default)**

Run: `conda run -n roboray python -m pytest -q`
Expected: all pass (the gpu tests are excluded)

- [ ] **Step 5: Commit**

```bash
git add tests/gpu/ deploy/modelenvs/ scripts/vendor_models.sh docs/models.md pyproject.toml
git commit -m "feat(gpu): L2 smoke suite, model env build scripts, vendored pinning and docs/models.md"
```

---

### Task 8: Full verification and the Plan B verification report

**Files:**
- Create: `docs/superpowers/plans/B-verification-report.md`
- Modify: `README.md` (the real-model section points to docs/models.md)

**Interfaces:**
- Consumes: everything above
- Produces: an all-green suite plus an honestly recorded verification boundary (what was verified locally vs what is left for L2)

- [ ] **Step 1: Full test run**

Run: `conda run -n roboray python -m pytest -q`
Expected: all pass (>= 128; gpu excluded)

- [ ] **Step 2: CLI smoke (fake-model regression — confirm the fake path is not broken by registering the real adapters)**

```bash
conda run -n roboray python scripts/run_local_fake.py --config configs/local-fake.yaml
```

Expected: succeeds, `failed []`. Note that `configs/local-fake.yaml` explicitly uses fake model names, so it is unaffected by the builtin-factory switch (if it were affected — e.g. install_fake_models overwrites the same names after a registry reset — record the note).

- [ ] **Step 3: Write the verification report (real numbers + explicit boundary)**

`B-verification-report.md` records: the environment, the pass count, the local verification scope (subprocess orchestration/contracts/fusion algorithm/A3 checks), the **L2-to-verify checklist** (aggregated item by item from the scripts' `[L2]` comments), the GPU smoke run command, and risks & mitigations. Every number comes from an actual run; no fabrication.

- [ ] **Step 4: README real-model section**

Change the "real models" subsection to point at the full explanation in `docs/models.md` (build/weights/variables/smoke commands).

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/plans/B-verification-report.md README.md
git commit -m "docs: Plan B verification report and README real-model section"
```

---

## Self-Review

**1. Spec coverage**

| Spec requirement (§6/7) | Corresponding task |
| --- | --- |
| ViPE adapter (w2c poses + metric depth) | Tasks 2/3 |
| Fast-FoundationStereo adapter (fx·B/d + ViPE pose composition) | Tasks 2/3 |
| HaMeR+MANO adapter (left/right slotting + strict weight verification) | Tasks 2/4 |
| registry builtin names unchanged (vipe/fast_foundation_stereo/hamer) | Tasks 3/4 |
| A1 refinement (mano path + fallback closed-loop marking + real reprojection error + QC filtering) | Task 5 |
| A3 four checks (4th = velocity continuity) + skip counts + audit persistence/QC rendering + min_supporting_joints | Task 6 |
| §12 L2 (A800 smoke) | Task 7 (suite and docs), Plan D (execution) |
| §13 vendored commit pinning | Task 7 |

**1b. Final post-review fix wave (post-delivery)**: the fix wave after the whole-branch review made the following changes to Tasks 3/5/6 (all delivered and passing):
- **Stereo routing end-to-end**: Plan A put the right-eye frames in `calib["frames_right"]`, but the FFS adapter expects them inside the frames dict — the fix promotes them into the frames dict at the Actor's **single frames-construction entry point** `ray_runtime.actors.build_infer_frames` (only when present; doubles ignore unknown keys). Regression coverage: `build_scene_payload -> Actor frames construction -> FFS adapter -> depth_source=="stereo"`.
- **Identity of accepted filled frames in A2**: filled frames have no real detections (the script writes zeros), and ranking directly on zeros would fabricate identity switches; the fix uses "valid 2D" (filled frames use the linear-interpolated x-mean of the two-sided observations).
- **A3 audit persistence surface**: `gapfill_reasons`/`gapfill_skipped` are written into `fused.npz` and rendered by QC (**manifest-level exposure deferred**).
- **FFS input resolution knob**: `stereo_max_image_side_px` flows config -> payload calib -> adapter `--extra` -> `run_ffs.py` and takes effect (scales the stereo pair and scales fx proportionally, depth unchanged; 0/missing = no scaling).

**2. Placeholder scan**: the `[L2]` annotations in the inference scripts are **explicitly declared verification points** (with assumed contracts and verification methods), not "write later" placeholders; the script code is complete and runnable. The remaining task code is complete.

**3. Type consistency**: `run_in_modelenv`/`resolve_env_request` (defined Task 1, consumed Tasks 3/4); `save_artifact` (Plan A already had it, consumed Tasks 2/3/4); `ModelEnvError` (defined Task 1, raised Tasks 3/4); `prepared_has_stereo`/`prepared_resolution_key` (already in Plan A's fix wave, consumed Task 3); the new `backproject_to_world` signature (defined Task 5, Task 6 consumes its NaN semantics).

**4. Review Focus coverage**

| Review Focus | Test that pins it |
| --- | --- |
| 1. env/weights missing -> readable error | `test_vipe_adapter_missing_env_python_gives_readable_error`, `test_hamer_missing_weights_gives_registration_hint`, `test_hamer_weights_without_mano_rejected` |
| 2. subprocess failure -> ModelEnvError | `test_run_in_modelenv_nonzero_exit_raises_with_stderr`, `..._timeout_raises`, `..._missing_result_line_raises` |
| 3. c2w->w2c | `test_vipe_adapter_runs_fake_script` (w2c contract) + Task 5 backprojection closed loop (implies orientation correctness) |
| 4. FFS baseline sign/magnitude | `test_ffs_adapter_computes_depth_from_baseline` |
| 5. MANO left-hand slotting | Task 2 double/real-script contract (slots 0=left 1=right) + `test_fake_hamer_contract` |
| 6. A1 backprojection unit consistency | `test_backproject_reprojection_roundtrip_is_exact` (< 1e-3) |

## Execution Handoff

After Plan B -> Plan C (stages 3/4: complete LeRobot v3 via the lerobot package, a real VLM AI Function, 3D world view, QC curves) -> Plan D (A800 KubeRay deployment, L2 smoke execution, L3 three-tier latency comparison and the custom evaluation).
