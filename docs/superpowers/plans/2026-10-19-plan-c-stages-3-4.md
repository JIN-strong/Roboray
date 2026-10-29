# Plan C: Stages 3/4 Full Implementation (LeRobot v3 / Real VLM / 3D World View / QC Curves) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring the Plan A/B pipeline up to the complete form in spec sections 9/10 — write a genuine LeRobot v3 dataset through the **lerobot public API**, do action semantics with a real OpenAI-compatible VLM, render a 3D world view plus a camera-view overlay, and output a QC report with error curves; and close Plan B's leftover hard gate on stereo-depth grid mismatch.

**Architecture:** Both the LeRobot write and the VLM call follow the **subprocess/sparse-dependency** approach established in Plan B: lerobot hard-depends on torch>=2.7 and numpy>=2.0 (which conflicts with the main env's numpy<2.0), so a separate CPU env `ego-lerobot` is created and the export runs as a subprocess via `scripts/modelenvs/run_lerobot_export.py` (the same `__RESULT__` protocol), **which can be genuinely verified locally** (CPU torch does not need a GPU); the VLM goes through the `openai` package's OpenAI-compatible interface (a light dependency that enters the main env), with an injectable client for testing. Visualization and QC run inside the main env with matplotlib (Agg backend, no EGL required).

**Tech Stack:** Python 3.10 (main env), separate `ego-lerobot` env (CPU torch + lerobot[dataset]), `openai` (main env), matplotlib(Agg), jinja2, pytest.

**Spec:** `docs/superpowers/specs/2026-10-16-ego-pipeline-design.md` (sections 9/10 are authoritative for this plan; section 6's semantic defect in the FFS resolution knob is this plan's first hard gate)

## Global Constraints

- Main env dependency: `conda run -n roboray python -m pytest ...` (never bare `conda run -n roboray pytest`); the main env **does not install torch or lerobot**; the main env's `numpy<2.0` pin must not be relaxed
- lerobot version pinned to **0.6.1**; its env is built by `deploy/modelenvs/build_lerobot.sh` (CPU torch: `--index-url https://download.pytorch.org/whl/cpu`), the interpreter injected via `ROBORAY_LEROBOT_PYTHON` (default `"python"`), matching Plan B's `ROBORAY_*` pattern
- The LeRobot dataset must be written via `LeRobotDataset.create/add_frame/save_episode/finalize` (spec §13 explicitly says "export with its public API rather than hand-written files"); the feature dict must declare dtype/shape explicitly; `use_videos=False` (this pipeline's training tensors are numeric features, avoiding the ffmpeg dependency)
- Subprocess script protocol (existing in Plan B): `--input <npz> --output <npz|dir> [--weights] [--extra json]`, last line `__RESULT__{json}`
- Every failure must be readable and **not silent**: an accepted config item must either take effect or be explicitly declared unimplemented (the lesson from Plan B's `stereo_max_image_side_px`)
- VLM config keys reuse the existing schema: `action_understanding.{vlm_base_url, vlm_api_key_env, vlm_model, keyframe_stride, model}`; the builtin name for `model` is `openai_vlm` (`configs/default.yaml` currently writes `"vlm"`, an existing defect this plan fixes)
- Visualization must run in a container without EGL/display (matplotlib Agg + cv2)
- One commit per task; every Review Focus item must be pinned by a test

## Review Focus

1. **lerobot env missing/version mismatch** -> the export stage returns a readable `ModelEnvError` (with the build-script guidance), the Episode is FAILED, the batch continues — never silently produces a half-baked dataset.
2. **The LeRobot dataset can be read back**: the written directory must load with `LeRobotDataset(root=...)` and `len(dataset)/features/fps` must match the source data (self-evidence of a valid format, not just "the file exists").
3. **VLM unreachable/auth failure** (no `vlm_base_url`, missing API key, non-2xx HTTP) -> clear error naming the needed config item; **no silent degradation** to no labels (Plan A decision D6).
4. **VLM returns invalid/excess labels** -> truncate/pad to the action-segment count; never crash the whole Episode because of a volatile model output format.
5. **Stereo depth grid** (Plan B leftover): when `stereo_max_image_side_px` takes effect, the depth must be geometrically consistent with the original frames (resampled back to the original resolution), otherwise fusion sampling misaligns — this must be falsifiable locally.
6. **QC numbers must come from real measurements**: the error curves are drawn from ground truth (not placeholders), the fill audit and thresholds agree between figure and text; empty/all-untrustworthy data must be recognized rather than drawing an empty figure and treating it as normal.

---

### Task 1: Stereo depth grid fix (Plan B leftover hard gate)

**Files:**
- Modify: `scripts/modelenvs/run_ffs.py` (resample the depth back to the original geometry after inference)
- Modify: `tests/fakes/modelenv_scripts/fake_ffs.py` (mirror the same semantics)
- Test: `tests/unit/test_scene_models.py` (extended)

**Interfaces:**
- Consumes: the existing `--extra` keys (`baseline_m`, `fx`, `max_side_px`)
- Produces: `run_ffs.py`'s output `depth` shape **always equals the input frame geometry** `(T, H, W)`; `__RESULT__` gains `"depth_grid": [H, W]` and `"model_input_size": [h, w]` (for verification/QC); `intrinsics` always uses the original-geometry intrinsics

- [ ] **Step 1: Write the failing test**

`tests/unit/test_scene_models.py` addition:

```python
def test_ffs_depth_returns_to_original_grid_after_resize(tmp_path, monkeypatch):
    """Plan B leftover gate: when the knob takes effect, depth must return to the original frame geometry (otherwise fusion sampling misaligns)."""
    import json as _json
    import subprocess as _sp

    monkeypatch.setenv("ROBORAY_FFS_PYTHON", sys.executable)
    # original 16x16, max_side=8 -> model input 8x8, but the output depth must be 16x16
    calib = {"fps": 30.0, "has_stereo": True, "T_left_to_right": np.eye(4).tolist(),
             "K": np.diag([16.0, 16.0, 1.0, 1.0]).tolist(),
             "stereo_max_image_side_px": 8}
    calib["T_left_to_right"][0][3] = 0.0325
    frames = np.zeros((2, 16, 16, 3), np.uint8)
    right = np.zeros((2, 16, 16, 3), np.uint8)
    model = FastFoundationStereoModel(script=str(FAKES / "fake_ffs.py"),
                                      vipe_script=str(FAKES / "fake_vipe.py"))
    model.load("cpu")
    out = tmp_path / "s.npz"
    model.infer({"frames": frames, "frames_right": right, "episode_id": "e"}, calib, out)
    z = np.load(out, allow_pickle=False)
    assert z["depth"].shape == (2, 16, 16)          # original geometry
    assert z["intrinsics"].shape == (2, 3, 3)


def test_fake_ffs_reports_both_grids(tmp_path):
    """The script's __RESULT__ records both the model input grid and the output depth grid; they differ when scaled."""
    import json as _json
    import subprocess as _sp

    inp = tmp_path / "in.npz"
    np.savez(inp, frames=np.zeros((2, 16, 16, 3), np.uint8),
             frames_right=np.zeros((2, 16, 16, 3), np.uint8),
             n_frames=np.int32(2), fps=np.float32(30.0))
    out = tmp_path / "o.npz"
    proc = _sp.run([sys.executable, str(FAKES / "fake_ffs.py"),
                    "--input", str(inp), "--output", str(out),
                    "--extra", _json.dumps({"baseline_m": 0.0325, "fx": 16.0, "max_side_px": 8})],
                   capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    res = _json.loads([l for l in proc.stdout.splitlines() if l.startswith("__RESULT__")][-1][10:])
    assert res["model_input_size"] == [8, 8]
    assert res["depth_grid"] == [16, 16]
    assert np.load(out, allow_pickle=False)["depth"].shape == (2, 16, 16)
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `conda run -n roboray python -m pytest tests/unit/test_scene_models.py -k "grid" -v`
Expected: FAIL (the depth currently stays on the scaled grid, shape `(2,8,8)`)

- [ ] **Step 3: Implement the fix**

In `scripts/modelenvs/run_ffs.py`, after computing the depth on the scaled grid and before assembling the output, insert:

```python
    # depth must return to the caller's (original frame) geometry: fusion samples depth at
    # original pixels, and a scaled grid would make _sample_depth go out of bounds or misalign
    # (Plan B review residual #1).
    orig_h, orig_w = int(left.shape[1]), int(left.shape[2])
    if (model_h, model_w) != (orig_h, orig_w):
        depth = np.stack([
            cv2.resize(d, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            for d in depth
        ]).astype(np.float32)
```

where `model_h, model_w` are the resized grid used for inference (already obtainable at `run_ffs.py`'s existing resize logic). Add to the `__RESULT__` summary:

```python
        {"ok": True, "model": "ffs", "n_frames": t,
         "model_input_size": [int(model_h), int(model_w)],
         "depth_grid": [int(depth.shape[1]), int(depth.shape[2])]},
```

`tests/fakes/modelenv_scripts/fake_ffs.py` mirrors the same semantics (compute the constant disparity on the scaled grid -> depth -> resize back to the original geometry) and likewise outputs both grids in `__RESULT__`.

- [ ] **Step 4: Run the test and confirm it passes (including regression of the existing FFS tests)**

Run: `conda run -n roboray python -m pytest tests/unit/test_scene_models.py tests/unit/test_modelenv_scripts.py -v`
Expected: all pass (including the existing `test_ffs_adapter_resize_keeps_depth_invariant`)

- [ ] **Step 5: Commit**

```bash
git add scripts/modelenvs/run_ffs.py tests/fakes/modelenv_scripts/fake_ffs.py tests/unit/test_scene_models.py
git commit -m "fix(ffs): resample depth back to the original frame geometry (closes the Plan B stereo grid mismatch gate)"
```

---

### Task 2: Real LeRobot v3 export (separate CPU env + subprocess)

**Files:**
- Create: `deploy/modelenvs/build_lerobot.sh`
- Create: `scripts/modelenvs/run_lerobot_export.py`
- Create: `tests/fakes/modelenv_scripts/fake_lerobot_export.py`
- Create: `src/roboray/export/__init__.py`
- Create: `src/roboray/export/lerobot_writer.py`
- Modify: `src/roboray/stages/publish.py` (call the export module, replacing Plan A's hand-written data.npz/meta.json)
- Test: `tests/unit/test_lerobot_writer.py`

**Interfaces:**
- Consumes: `action.npz` (state/action/labels/segments/fps), `fused.npz` (joints_world), `EpisodeWorkspace`
- Produces:
  - `LeRobotWriter(env_python_var="ROBORAY_LEROBOT_PYTHON", script=None)`: `write(episode_id, state, action, tasks, fps, out_dir) -> dict` (calls the script via `run_in_modelenv`; returns `{"dataset_dir","n_frames","n_episodes"}`)
  - Script contract: `--input <npz>` (`state (T,D) float32`, `action (T,D) float32`, `task_str`), `--output <dir>` (dataset root), `--extra {"repo_id": ..., "fps": ...}`; last stdout line `__RESULT__{"ok":true,"dataset_dir":...,"n_frames":...}`
  - `publish`'s `lerobot_v3/<ep>/` is a **real LeRobot dataset** (`meta/info.json`, `data/`, `meta/episodes/`), loadable via `LeRobotDataset(root=...)`
  - Features: `observation.state` and `action`, `dtype float32`, `shape (21*2*3,) = (126,)` (left/right hand 21 joints × 3 coords flattened), with `names`
- Note: if the `ego-lerobot` env can be built, this task is **genuinely verified locally** (CPU torch is enough); if the env is unavailable, fall back to verifying the orchestration with the fake script and mark the real-API verification [L2] — but the report must explicitly state which case it is

- [ ] **Step 1: Write the build script and the failing test**

`deploy/modelenvs/build_lerobot.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
# Build the LeRobot export-only env (CPU torch; lerobot 0.6.1 hard-depends on torch>=2.7 and
# numpy>=2.0, which conflicts with the main env's numpy<2.0, so it must be separate).
conda create -y -n ego-lerobot python=3.12  # lerobot 0.6.1 Requires-Python>=3.12 (3.10 fails to install, verified)
conda run -n ego-lerobot python -m pip install "torch>=2.7" torchvision --index-url https://download.pytorch.org/whl/cpu
conda run -n ego-lerobot python -m pip install "lerobot[dataset]==0.6.1"
conda run -n ego-lerobot python -c "import lerobot, torch; print('lerobot', lerobot.__version__, 'torch', torch.__version__)"
echo "done. set ROBORAY_LEROBOT_PYTHON=$(conda info --base)/envs/ego-lerobot/bin/python"
```

`tests/unit/test_lerobot_writer.py`:

```python
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from roboray.export.lerobot_writer import LeRobotWriter

FAKES = Path(__file__).resolve().parents[1] / "fakes" / "modelenv_scripts"


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("ROBORAY_LEROBOT_PYTHON", sys.executable)
    yield


def test_writer_creates_dataset_dir_via_fake(tmp_path):
    writer = LeRobotWriter(script=str(FAKES / "fake_lerobot_export.py"))
    t = 6
    state = np.zeros((t, 126), np.float32)
    action = np.ones((t, 126), np.float32)
    info = writer.write("ep_0001", state, action, tasks=["reach"], fps=30, out_dir=tmp_path / "ds")
    assert info["n_frames"] == t
    ds = tmp_path / "ds"
    assert (ds / "meta" / "info.json").exists()
    assert (ds / "data").is_dir()
    assert (ds / "meta" / "episodes").is_dir()


def test_writer_rejects_mismatched_lengths(tmp_path):
    writer = LeRobotWriter(script=str(FAKES / "fake_lerobot_export.py"))
    with pytest.raises(ValueError) as exc:
        writer.write("ep", np.zeros((4, 126), np.float32), np.zeros((3, 126), np.float32),
                     tasks=["t"], fps=30, out_dir=tmp_path / "ds")
    assert "length" in str(exc.value) or "shape" in str(exc.value)


def test_writer_missing_env_gives_build_hint(tmp_path, monkeypatch):
    from roboray.models.modelenv import ModelEnvError

    monkeypatch.setenv("ROBORAY_LEROBOT_PYTHON", "/nonexistent/python")
    writer = LeRobotWriter(script=str(FAKES / "fake_lerobot_export.py"))
    with pytest.raises(ModelEnvError) as exc:
        writer.write("ep", np.zeros((2, 126), np.float32), np.zeros((2, 126), np.float32),
                     tasks=["t"], fps=30, out_dir=tmp_path / "ds")
    assert "build_lerobot" in str(exc.value)
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `conda run -n roboray python -m pytest tests/unit/test_lerobot_writer.py -v`
Expected: FAIL — `ModuleNotFoundError: roboray.export.lerobot_writer`

- [ ] **Step 3: Implement the fake script + export module**

`tests/fakes/modelenv_scripts/fake_lerobot_export.py` (mirrors the real script's on-disk layout without depending on lerobot):

```python
"""LeRobot export double: writes the same directory skeleton as the real script, for local orchestration tests."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "modelenvs"))

from _common import load_input, parse_args, save_with_result  # noqa: E402


def main() -> None:
    args = parse_args("fake lerobot export")
    data = load_input(args.input)
    extra = json.loads(args.extra)
    root = Path(args.output)
    (root / "meta" / "episodes").mkdir(parents=True, exist_ok=True)
    (root / "data").mkdir(parents=True, exist_ok=True)
    n = int(data["state"].shape[0])
    (root / "meta" / "info.json").write_text(json.dumps(
        {"codebase_version": "v3.0", "fps": int(extra.get("fps", 30)),
         "features": {"observation.state": {"dtype": "float32", "shape": list(data["state"].shape[1:])},
                      "action": {"dtype": "float32", "shape": list(data["action"].shape[1:])}},
         "total_episodes": 1, "total_frames": n}, ensure_ascii=False))
    (root / "data" / "chunk-000.parquet").write_bytes(b"fake-parquet")
    (root / "meta" / "episodes" / "chunk-000.parquet").write_bytes(b"fake-parquet")
    (root / "meta" / "tasks.jsonl").write_text(json.dumps({"task": extra.get("task", "unknown")}) + "\n")
    save_with_result(args.output + ".summary.npz",
                     {"n_frames": __import__("numpy").int32(n)},
                     {"ok": True, "dataset_dir": str(root), "n_frames": n, "n_episodes": 1})


if __name__ == "__main__":
    main()
```

`src/roboray/export/lerobot_writer.py`:

```python
"""LeRobot v3 export: calls the lerobot public API via a subprocess in a separate env.

lerobot hard-depends on torch>=2.7 and numpy>=2.0, which conflicts with the main env's
numpy<2.0, hence the subprocess (the same pattern as Plan B's model adapters).
"""

import os
import tempfile
from pathlib import Path

import numpy as np

from roboray.models.modelenv import ModelEnvError, run_in_modelenv

__all__ = ["LeRobotWriter"]

_DEFAULT_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "modelenvs" / "run_lerobot_export.py"


class LeRobotWriter:
    def __init__(self, script: str | None = None, env_var: str = "ROBORAY_LEROBOT_PYTHON"):
        self.script = script or str(_DEFAULT_SCRIPT)
        self.env_var = env_var

    def write(self, episode_id, state, action, tasks, fps, out_dir) -> dict:
        state = np.asarray(state, np.float32)
        action = np.asarray(action, np.float32)
        if state.shape[0] != action.shape[0]:
            raise ValueError(
                f"state/action length mismatch: state shape {state.shape} vs action shape {action.shape}"
            )
        if state.ndim != 2 or action.ndim != 2 or state.shape[1] != action.shape[1]:
            raise ValueError(f"state/action must be same-width 2D (T,D): {state.shape} / {action.shape}")
        python = os.environ.get(self.env_var, "python")
        if python != "python" and not Path(python).exists():
            raise ModelEnvError(
                f"the interpreter pointed to by {self.env_var} does not exist: {python}. "
                "Build the export env with deploy/modelenvs/build_lerobot.sh."
            )
        out_dir = Path(out_dir)
        out_dir.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as td:
            inp = Path(td) / "export_in.npz"
            np.savez(inp, state=state, action=action,
                     task_str=np.array(tasks[0] if tasks else "unknown"))
            argv = ["--input", str(inp), "--output", str(out_dir),
                    "--extra", _json({"repo_id": episode_id, "fps": int(fps),
                                      "task": (tasks[0] if tasks else "unknown")})]
            summary = run_in_modelenv(python, self.script, argv, timeout_s=3600.0)
        if not summary.get("ok"):
            raise ModelEnvError(f"LeRobot export failed: {summary}")
        if not (out_dir / "meta" / "info.json").exists():
            raise ModelEnvError(f"LeRobot export produced no meta/info.json: {out_dir}")
        return {"dataset_dir": str(out_dir), "n_frames": int(summary["n_frames"]),
                "n_episodes": int(summary.get("n_episodes", 1))}


def _json(obj) -> str:
    import json

    return json.dumps(obj, ensure_ascii=False)
```

`src/roboray/export/__init__.py`:

```python
"""LeRobot v3 export."""

from .lerobot_writer import LeRobotWriter  # noqa: F401

__all__ = ["LeRobotWriter"]
```

- [ ] **Step 4: Implement the real script (using the lerobot public API)**

`scripts/modelenvs/run_lerobot_export.py`:

```python
"""LeRobot v3 export script (executed in the ego-lerobot env).

Uses the lerobot public API (spec §13 requirement), no hand-written file format:
  LeRobotDataset.create(repo_id, fps, features, root=..., use_videos=False)
  -> add_frame({...,"task": ...}) -> save_episode() -> finalize()
Features: observation.state / action, both float32 (21*2*3,)=(126,) flattened joint coordinates.
use_videos=False: this pipeline's training tensors are numeric features, no ffmpeg needed.

L2 verification point (A800/first real run): [L2] lerobot 0.6.1's create() parameter names and
root semantics (if root must be the dataset root rather than the parent directory, adjust per the
actual error and record it in docs/models.md).
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import load_input, parse_args, save_with_result  # noqa: E402


def main() -> None:
    args = parse_args("LeRobot v3 export")
    data = load_input(args.input)
    extra = json.loads(args.extra)
    state = np.asarray(data["state"], np.float32)
    action = np.asarray(data["action"], np.float32)
    task = str(extra.get("task", "unknown"))
    fps = int(extra.get("fps", 30))
    root = Path(args.output)

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset  # [L2: import path]
    except ImportError as exc:
        raise RuntimeError(
            f"lerobot env unavailable (missing package {exc}); build it with deploy/modelenvs/build_lerobot.sh"
        ) from exc

    dim = state.shape[1]
    features = {
        "observation.state": {"dtype": "float32", "shape": (dim,),
                              "names": [f"j{i}" for i in range(dim)]},
        "action": {"dtype": "float32", "shape": (dim,),
                   "names": [f"a{i}" for i in range(dim)]},
    }
    ds = LeRobotDataset.create(  # [L2: parameter names (repo_id/root/use_videos)]
        repo_id=str(extra.get("repo_id", "roboray_episode")),
        fps=fps,
        features=features,
        root=root,
        use_videos=False,
    )
    for i in range(state.shape[0]):
        ds.add_frame({  # [L2: frame key names and the task field requirement]
            "observation.state": state[i],
            "action": action[i],
            "task": task,
        })
    ds.save_episode()
    ds.finalize()

    save_with_result(
        args.output + ".summary.npz",
        {"n_frames": np.int32(state.shape[0])},
        {"ok": True, "dataset_dir": str(root), "n_frames": int(state.shape[0]), "n_episodes": 1},
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Point publish at the export module**

In `src/roboray/stages/publish.py`, replace the Plan A hand-written `data.npz + meta.json` block with:

```python
    # --- LeRobot v3 (spec §13: must go through the lerobot public API) ---
    from roboray.export.lerobot_writer import LeRobotWriter

    ds_dir = out / "lerobot_v3" / episode_id
    t_state = np.asarray(state_arr, np.float32)
    t_action = np.asarray(action, np.float32)
    if t_state.ndim == 4:  # (T, hands, joints, 3) -> (T, 126)
        t_state = t_state.reshape(t_state.shape[0], -1)
    if t_action.ndim == 4:
        t_action = t_action.reshape(t_action.shape[0], -1)
    info = LeRobotWriter().write(
        episode_id=episode_id,
        state=t_state,
        action=t_action,
        tasks=[labels[0] if labels else "unknown"],
        fps=fps,
        out_dir=ds_dir,
    )
    outputs["dataset"] = info["dataset_dir"]
```

(And remove the original `np.savez_compressed(ds_dir / "data.npz", ...)` and `meta.json` write block; the labels/segments info from `meta.json` is moved to `out / "qc" / episode_id / "segments.json"` to keep traceability.)

- [ ] **Step 6: Build the ego-lerobot env and verify for real (if feasible)**

```bash
bash deploy/modelenvs/build_lerobot.sh
conda run -n roboray python -c "
import os, subprocess
p = subprocess.run(['conda','info','--base'], capture_output=True, text=True).stdout.strip() + '/envs/ego-lerobot/bin/python'
os.environ['ROBORAY_LEROBOT_PYTHON'] = p
print('env python:', p)
"
```
If the build succeeds, run one export with the **real script** and read it back:

```bash
conda run -n roboray python - <<'PY'
import os, numpy as np, subprocess, sys
base = subprocess.run(['conda','info','--base'], capture_output=True, text=True).stdout.strip()
os.environ['ROBORAY_LEROBOT_PYTHON'] = f'{base}/envs/ego-lerobot/bin/python'
from roboray.export.lerobot_writer import LeRobotWriter
t = 8
w = LeRobotWriter()
info = w.write('ep_test', np.zeros((t,126),np.float32), np.ones((t,126),np.float32), ['reach'], 30, '/tmp/lerobot_probe')
print(info)
# read-back self-verification
from pathlib import Path
assert (Path('/tmp/lerobot_probe')/'meta'/'info.json').exists()
print('OK: directory layout ready')
PY
```

If the build fails (disk/network/deps): **explicitly record the failure** in the report, mark the real-API verification as [L2], and keep the entire fake-script path's tests passing.

- [ ] **Step 7: Run the tests and commit**

Run: `conda run -n roboray python -m pytest tests/unit/test_lerobot_writer.py tests/unit/test_fusion_action.py -v`, then the full `conda run -n roboray python -m pytest -q`
Expected: all pass (if a publish-related assertion needs adjusting for the format change, that is expected — update to the real LeRobot contract and explain it)

```bash
git add deploy/modelenvs/build_lerobot.sh scripts/modelenvs/run_lerobot_export.py \
        tests/fakes/modelenv_scripts/fake_lerobot_export.py src/roboray/export/ \
        src/roboray/stages/publish.py tests/unit/test_lerobot_writer.py
git commit -m "feat(export): export LeRobot v3 via the lerobot public API (separate CPU env + subprocess)"
```

---

### Task 3: Real VLM AI Function (OpenAI-compatible)

**Files:**
- Create: `src/roboray/models/semantic_models.py`
- Modify: `src/roboray/models/registry.py` (the builtin `openai_vlm` points at the real implementation)
- Modify: `pyproject.toml` (promote `openai>=1.40` from the `models` extra to a **base dependency**)
- Modify: `configs/default.yaml` (`action_understanding.model: "vlm"` -> `"openai_vlm"`)
- Modify: `src/roboray/stages/action.py` (actually pass the vlm config to the semantic model)
- Modify: `src/roboray/models/protocol.py` (`SemanticModel.label` gains an optional `n_segments`)
- Modify: `tests/fakes/models.py` (`FakeSemanticModel.label` synced to the signature)
- Test: `tests/unit/test_semantic_models.py`

**Interfaces:**
- Produces:
  - `OpenAIVlm(name="openai_vlm", client_factory=None)` implementing `SemanticModel`: `label(prompt, images, out_path)` calls `client.chat.completions.create(model=..., messages=[...])`; `base_url` must come from the config, and the API key is read from the env var named by `vlm_api_key_env`; missing/non-2xx/parse failure -> `VlmError` (readable, naming the missing item)
  - The returned content is a JSON array (action labels); on a parse failure or a count mismatch -> truncate/pad to the `segments` count (no crash)
  - `client_factory` is injectable (tests pass a fake client)
  - `run_action` in `stages/action.py` passes `vlm_base_url`/`vlm_api_key_env`/`vlm_model` to the instance from `registry.get_semantic_model_class(...)()` (via constructor args or `configure()`)
  - **Keyframes must actually enter the request** (spec §9: "input keyframes + trajectory summary"): `run_action` samples `prepared/frames_left.npy` by `keyframe_stride` (always includes the first/last frame, capped at `MAX_KEYFRAMES=16` by uniform downsampling), encodes via `cv2.imencode(".jpg")` -> base64 data URI passed to `label(..., images=[...])`; `OpenAIVlm` builds multimodal `content = [{"type":"text",...}, {"type":"image_url","image_url":{"url": uri}}, ...]`, falling back to text-only when `images` is empty; a missing frame or encoding failure falls back to text-only and never crashes the Episode
- Decision D6: VLM unreachable/unconfigured -> stage 3 reports an explicit error (**no geometric degradation**)

- [ ] **Step 1: Write the failing test**

`tests/unit/test_semantic_models.py`:

```python
import json

import numpy as np
import pytest

from roboray.models.semantic_models import OpenAIVlm, VlmError


class _FakeMsg:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMsg(content)


class _FakeCompletions:
    def __init__(self, content):
        self._content = content
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return type("R", (), {"choices": [_FakeChoice(self._content)]})()


class _FakeClient:
    def __init__(self, content):
        self.chat = type("C", (), {"completions": _FakeCompletions(content)})()


@pytest.fixture(autouse=True)
def _key(monkeypatch):
    monkeypatch.setenv("VLM_API_KEY", "test-key")
    yield


def test_label_writes_parsed_labels(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBORAY_VLM_BASE_URL", "https://vlm.example/v1")
    client = _FakeClient(json.dumps(["reach", "grasp"]))
    model = OpenAIVlm(client_factory=lambda **kw: client)
    out = tmp_path / "sem.npz"
    model.label("segments=2", [], out, n_segments=2)
    z = np.load(out, allow_pickle=True)
    assert [str(x) for x in z["labels"]] == ["reach", "grasp"]
    assert client.chat.completions.calls[0]["model"] == "qwen-vl"


def test_label_pads_and_truncates_to_segment_count(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBORAY_VLM_BASE_URL", "https://vlm.example/v1")
    model = OpenAIVlm(client_factory=lambda **kw: _FakeClient(json.dumps(["a"])))
    out = tmp_path / "sem.npz"
    model.label("p", [], out, n_segments=3)
    assert len([str(x) for x in np.load(out, allow_pickle=True)["labels"]]) == 3


def test_missing_base_url_raises_naming_the_config_key(tmp_path, monkeypatch):
    monkeypatch.delenv("ROBORAY_VLM_BASE_URL", raising=False)
    model = OpenAIVlm(client_factory=lambda **kw: _FakeClient("[]"))
    with pytest.raises(VlmError) as exc:
        model.label("p", [], tmp_path / "s.npz", n_segments=1)
    assert "vlm_base_url" in str(exc.value) or "ROBORAY_VLM_BASE_URL" in str(exc.value)


def test_missing_api_key_raises_naming_the_env_var(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBORAY_VLM_BASE_URL", "https://vlm.example/v1")
    monkeypatch.delenv("VLM_API_KEY", raising=False)
    model = OpenAIVlm(client_factory=lambda **kw: _FakeClient("[]"))
    with pytest.raises(VlmError) as exc:
        model.label("p", [], tmp_path / "s.npz", n_segments=1)
    assert "VLM_API_KEY" in str(exc.value)


def test_non_json_reply_is_not_fatal(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBORAY_VLM_BASE_URL", "https://vlm.example/v1")
    model = OpenAIVlm(client_factory=lambda **kw: _FakeClient("this is not JSON"))
    out = tmp_path / "s.npz"
    model.label("p", [], out, n_segments=2)  # does not raise
    labels = [str(x) for x in np.load(out, allow_pickle=True)["labels"]]
    assert len(labels) == 2
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `conda run -n roboray python -m pytest tests/unit/test_semantic_models.py -v`
Expected: FAIL — `ModuleNotFoundError: roboray.models.semantic_models`

- [ ] **Step 3: Implement `semantic_models.py`**

```python
"""Action-semantics model: OpenAI-compatible VLM (pluggable).

Decision D6: an unreachable VLM errors out; no geometric degradation.
Config sources: env vars ROBORAY_VLM_BASE_URL / ROBORAY_VLM_MODEL / <vlm_api_key_env>.
"""

import json
import os
from pathlib import Path

import numpy as np

__all__ = ["OpenAIVlm", "VlmError"]


class VlmError(RuntimeError):
    """VLM call failed or is unconfigured."""


def _extract_labels(raw, n_segments: int) -> list:
    """Try to pull a label list from the model reply; fall back to generic names on failure (no crash)."""
    labels = None
    text = (raw or "").strip()
    if text:
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
                labels = parsed
        except json.JSONDecodeError:
            start, end = text.find("["), text.rfind("]")
            if 0 <= start < end:
                try:
                    parsed = json.loads(text[start:end + 1])
                    if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
                        labels = parsed
                except json.JSONDecodeError:
                    labels = None
    if not labels:
        labels = [f"segment_{i}" for i in range(n_segments)]
    if len(labels) < n_segments:
        labels = labels + [f"segment_{i}" for i in range(len(labels), n_segments)]
    return labels[:n_segments]


class OpenAIVlm:
    name = "openai_vlm"

    def __init__(self, client_factory=None, base_url=None, model=None, api_key_env="VLM_API_KEY"):
        self._client_factory = client_factory
        self._base_url = base_url
        self._model = model
        self._api_key_env = api_key_env

    def _resolve(self):
        base_url = self._base_url or os.environ.get("ROBORAY_VLM_BASE_URL", "")
        model = self._model or os.environ.get("ROBORAY_VLM_MODEL", "qwen-vl")
        api_key = os.environ.get(self._api_key_env, "")
        if not base_url:
            raise VlmError(
                "VLM is unconfigured: set action_understanding.vlm_base_url (or the env var ROBORAY_VLM_BASE_URL)."
            )
        if not api_key:
            raise VlmError(
                f"VLM auth missing: env var {self._api_key_env} is empty "
                "(config action_understanding.vlm_api_key_env names it)."
            )
        return base_url, model, api_key

    def label(self, prompt: str, images: list, out_path: Path, n_segments: int = 0) -> None:
        base_url, model, api_key = self._resolve()
        factory = self._client_factory
        if factory is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - the main env already installs openai
                raise VlmError("the openai package is required (base dependency)") from exc
            factory = OpenAI
        try:
            client = factory(base_url=base_url, api_key=api_key)
            content = []
            if images:
                for uri in images:
                    content.append({"type": "image_url", "image_url": {"url": uri}})
                content.append({"type": "text", "text": prompt})
                messages = [{"role": "user", "content": content}]
            else:
                messages = [{"role": "user", "content": prompt}]
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
            )
            raw = resp.choices[0].message.content or ""
        except VlmError:
            raise
        except Exception as exc:
            raise VlmError(f"VLM call failed (base_url={base_url}, model={model}): {exc}") from exc
        labels = _extract_labels(raw, n_segments)
        np.savez(out_path, labels=np.array(labels, dtype=object),
                 segment_start=np.zeros(len(labels), np.int32),
                 segment_end=np.zeros(len(labels), np.int32))
```

Change the `registry.py` semantic factory to:

```python
    def _semantic(name: str):
        def factory():
            from roboray.models.semantic_models import OpenAIVlm

            return OpenAIVlm()

        return factory
```

- [ ] **Step 4: Pass the config down in the action stage**

In `stages/action.py`, after instantiating the semantic model, inject the config (keep the existing `SemanticModel` protocol signature; assign via attributes):

```python
        factory = registry.get_semantic_model_class(semantic_model_name)
        model = factory()
        for attr, key in (("_base_url", "vlm_base_url"), ("_model", "vlm_model"),
                          ("_api_key_env", "vlm_api_key_env")):
            value = semantic_kwargs.get(key)
            if value is not None and hasattr(model, attr):
                setattr(model, attr, value)
        model.label(summary, [], semantic_path, n_segments=len(segments))
```

`pyproject.toml`: move `openai>=1.40` into `dependencies` (keeping it in the `models` extra is also fine, but the base dependency is required). `configs/default.yaml`: `model: "openai_vlm"`.

- [ ] **Step 4b: Sync the double and protocol signatures (cross-task seam)**

`SemanticModel.label` gains the optional `n_segments: int = 0`, so:
- the `SemanticModel.label` signature in `src/roboray/models/protocol.py` becomes `label(prompt, images, out_path, n_segments: int = 0)`;
- `FakeSemanticModel.label` in `tests/fakes/models.py` accepts `n_segments` too (keeping the existing 3-arg calls working);
- the routing call site in `stages/action.py` already passes `n_segments=len(segments)` per Step 4.
Not updating the double -> the semantic-path calls in the existing `test_fusion_action` would `TypeError`.

- [ ] **Step 5: Run the tests and commit**

Run: `conda run -n roboray python -m pytest tests/unit/test_semantic_models.py tests/unit/test_fusion_action.py tests/unit/test_registry.py -v`, then the full `conda run -n roboray python -m pytest -q`

```bash
git add src/roboray/models/semantic_models.py src/roboray/models/registry.py \
        src/roboray/stages/action.py pyproject.toml configs/default.yaml tests/unit/test_semantic_models.py
git commit -m "feat(action): real OpenAI-compatible VLM (readable errors on missing key/url, non-fatal on malformed output)"
```

---

### Task 4: 3D world view and camera-view overlay

**Files:**
- Create: `src/roboray/viz/__init__.py`
- Create: `src/roboray/viz/world_view.py`
- Create: `src/roboray/viz/camera_view.py`
- Modify: `src/roboray/stages/publish.py` (call the render modules, replacing Plan A's placeholder)
- Test: `tests/unit/test_viz.py`

**Interfaces:**
- Produces:
  - `render_world_view(joints_world, camera_poses, depth, fps, out_path, *, pointcloud_stride=8, future_s=1.0) -> dict`: matplotlib Agg 3D; per frame draws the left/right hand skeletons (21-point connections, trusted joints solid, NaN joints skipped), the camera trajectory (history), a downsampled point cloud (depth backprojection of the first frame), and the next-1s wrist trajectory (dashed); outputs mp4; returns `{"frames": n, "out_path": ...}`
  - `render_camera_view(frames, joints_2d, present, intrinsics, out_path, fps) -> dict`: overlays the 2D keypoints and connections on the original frames (absent hands skipped); keeps the original aspect ratio
  - Both run in a display-less environment (Agg); empty/all-untrustworthy data outputs **empty with a reason** (no misleading video)
- Note: the world view renders frame-by-frame with matplotlib (small resolution + short clips, keeping it testable; performance optimization deferred to Plan D)

> **[Honest deferral · Plan D]** spec §10 requires the camera view to overlay **two** elements: "2D keypoints" **and** "MANO reprojection".
> This task delivers only the **2D keypoint overlay** (`render_camera_view`); the **MANO reprojection overlay is not implemented** and is registered as a
> **Plan D item**. The needed data is already available (`fused.npz`'s hand 3D keypoints/reprojection error + `scene.npz`'s intrinsics/extrinsics),
> so it is rendering work rather than a data gap — same source as the QC reprojection curve, it only needs the world keypoints projected back to pixels and overlaid.

- [ ] **Step 1: Write the failing test**

`tests/unit/test_viz.py`:

```python
import numpy as np
import pytest

from roboray.viz.camera_view import render_camera_view
from roboray.viz.world_view import render_world_view


def _fixture_hands(t=6, h=48, w=64):
    joints_world = np.zeros((t, 2, 21, 3), np.float64)
    joints_world[:, 0, :, 0] = 0.1
    joints_world[:, 0, :, 1] = np.linspace(0, 0.2, t)
    joints_world[:, 1, :, 0] = -0.1
    joints_world[2, 1, :, :] = np.nan  # one frame untrustworthy
    cam = np.tile(np.eye(4), (t, 1, 1))
    depth = np.full((t, h, w), 0.5, np.float32)
    return joints_world, cam, depth


def test_world_view_writes_playable_video(tmp_path):
    joints, cam, depth = _fixture_hands()
    out = tmp_path / "world.mp4"
    info = render_world_view(joints, cam, depth, 30, out)
    assert info["frames"] == 6
    assert out.exists() and out.stat().st_size > 0
    import cv2

    cap = cv2.VideoCapture(str(out))
    n = 0
    while cap.read()[0]:
        n += 1
    cap.release()
    assert n == 6


def test_world_view_all_nan_is_reported_not_faked(tmp_path):
    joints = np.full((4, 2, 21, 3), np.nan)
    cam = np.tile(np.eye(4), (4, 1, 1))
    depth = np.full((4, 8, 8), 0.5, np.float32)
    out = tmp_path / "world.mp4"
    info = render_world_view(joints, cam, depth, 30, out)
    assert info["frames"] == 0
    assert info.get("skipped_reason")
    assert not out.exists()


def test_camera_view_overlays_and_keeps_aspect(tmp_path):
    import cv2

    t, h, w = 4, 48, 64
    frames = np.zeros((t, h, w, 3), np.uint8)
    joints_2d = np.zeros((t, 2, 21, 3), np.float64)
    joints_2d[:, 0, :, 0] = w / 2
    joints_2d[:, 0, :, 1] = h / 2
    present = np.ones((t, 2), bool)
    present[:, 1] = False
    out = tmp_path / "cam.mp4"
    info = render_camera_view(frames, joints_2d, present, 30, out)
    assert info["frames"] == t
    cap = cv2.VideoCapture(str(out))
    ok, frame = cap.read()
    cap.release()
    assert ok and frame.shape[:2] == (h, w)
    assert frame.sum() > 0  # keypoints overlaid
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `conda run -n roboray python -m pytest tests/unit/test_viz.py -v`
Expected: FAIL — `ModuleNotFoundError: roboray.viz`

- [ ] **Step 3: Implement the two render modules**

`src/roboray/viz/world_view.py`:

```python
"""World view: hand skeletons + camera trajectory + downsampled point cloud + next-1s wrist trajectory.

matplotlib Agg (no EGL dependency); renders frame-by-frame and writes the mp4 via cv2.
"""

from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

__all__ = ["render_world_view"]

_HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
               (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16),
               (0, 17), (17, 18), (18, 19), (19, 20)]


def _draw_hand(ax, pts, color):
    for a, b in _HAND_EDGES:
        if np.isfinite(pts[a]).all() and np.isfinite(pts[b]).all():
            ax.plot([pts[a][0], pts[b][0]], [pts[a][1], pts[b][1]], [pts[a][2], pts[b][2]],
                    color=color, linewidth=1.0)
        ax.scatter(pts[i, 0], pts[i, 1], pts[i, 2], color=color, s=4)


def render_world_view(joints_world, camera_poses, depth, fps, out_path, *,
                      pointcloud_stride: int = 8, future_s: float = 1.0) -> dict:
    joints_world = np.asarray(joints_world, np.float64)
    camera_poses = np.asarray(camera_poses, np.float64)
    depth = np.asarray(depth)
    t = joints_world.shape[0]
    if t == 0 or not np.isfinite(joints_world).any():
        # all untrustworthy: produce no misleading video, give the reason
        for cand in (Path(out_path),):
            if cand.exists():
                cand.unlink()
        return {"frames": 0, "out_path": str(out_path), "skipped_reason": "no trusted hand joints"}

    writer = None
    frames_written = 0
    fig = plt.figure(figsize=(4, 3), dpi=100)
    try:
        for i in range(t):
            fig.clf()
            ax = fig.add_subplot(111, projection="3d")
            ax.set_xlim(-0.5, 0.5); ax.set_ylim(-0.5, 0.5); ax.set_zlim(-1.0, 0.2)
            ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
            _draw_hand(ax, joints_world[i, 0], "tab:blue")
            _draw_hand(ax, joints_world[i, 1], "tab:red")
            # camera trajectory (c2w translation)
            if camera_poses.ndim == 3 and i > 0:
                pos = camera_poses[:i + 1, :3, 3]
                ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], color="black", linewidth=1.0)
            # point cloud: downsampled depth backprojection of the first frame (camera frame)
            if i == 0:
                dh, dw = depth.shape[1], depth.shape[2]
                ys, xs = np.mgrid[0:dh:pointcloud_stride, 0:dw:pointcloud_stride]
                zs = np.asarray(depth[0])[::pointcloud_stride, ::pointcloud_stride]
                ax.scatter(xs.ravel() * 0.001, ys.ravel() * 0.001, zs.ravel(),
                           c=zs.ravel(), cmap="viridis", s=1, alpha=0.4)
            # next-1s wrist trajectory
            k = max(1, int(round(future_s * fps)))
            if i + k < t:
                w0 = joints_world[i:i + k + 1, 0, 0]
                if np.isfinite(w0).all():
                    ax.plot(w0[:, 0], w0[:, 1], w0[:, 2], "b--", linewidth=0.8)
            fig.canvas.draw()
            buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
            frame = cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)
            if writer is None:
                fh, fw = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(out_path), fourcc, fps, (fw, fh))
                if not writer.isOpened():
                    raise RuntimeError(f"cannot open VideoWriter for {out_path}")
            writer.write(frame)
            frames_written += 1
    finally:
        plt.close(fig)
        if writer is not None:
            writer.release()
    return {"frames": frames_written, "out_path": str(out_path)}
```

`src/roboray/viz/camera_view.py`:

```python
"""Camera view: overlays the 2D keypoints and skeleton connections on the original frames (keeps the original aspect ratio)."""

from pathlib import Path

import cv2
import numpy as np

__all__ = ["render_camera_view"]

_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
          (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16),
          (0, 17), (17, 18), (18, 19), (19, 20)]
_COLORS = {0: (255, 128, 0), 1: (0, 128, 255)}


def render_camera_view(frames, joints_2d, present, fps, out_path) -> dict:
    frames = np.asarray(frames)
    joints_2d = np.asarray(joints_2d)
    present = np.asarray(present, bool)
    t = frames.shape[0]
    if t == 0:
        return {"frames": 0, "out_path": str(out_path), "skipped_reason": "no frames"}
    h, w = frames.shape[1], frames.shape[2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"cannot open VideoWriter for {out_path}")
    try:
        for i in range(t):
            frame = frames[i].copy()
            for hand in range(joints_2d.shape[1]):
                if not present[i, hand]:
                    continue
                pts = joints_2d[i, hand][:, :2]
                color = _COLORS.get(hand, (0, 255, 0))
                for a, b in _EDGES:
                    pa, pb = pts[a], pts[b]
                    if np.isfinite(pa).all() and np.isfinite(pb).all():
                        cv2.line(frame, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), color, 1)
                for p in pts:
                    if np.isfinite(p).all():
                        cv2.circle(frame, (int(p[0]), int(p[1])), 2, color, -1)
            writer.write(frame)
    finally:
        writer.release()
    return {"frames": t, "out_path": str(out_path)}
```

`src/roboray/viz/__init__.py`:

```python
"""Visualization: camera view and 3D world view."""

from .camera_view import render_camera_view  # noqa: F401
from .world_view import render_world_view  # noqa: F401

__all__ = ["render_camera_view", "render_world_view"]
```

- [ ] **Step 4: Point publish at the render modules**

In `stages/publish.py`, replace Plan A's placeholder `_write_viz` (frame copy + black band) with:

```python
    if make_visualization:
        viz_dir = out / "viz" / episode_id
        viz_dir.mkdir(parents=True, exist_ok=True)
        if camera_view:
            from roboray.viz.camera_view import render_camera_view

            info = render_camera_view(frames, joints_2d, present, fps, viz_dir / "camera.mp4")
            outputs["camera_view"] = info["out_path"]
        if world_view:
            from roboray.viz.world_view import render_world_view

            info = render_world_view(joints_world, camera_poses, depth, fps, viz_dir / "world.mp4")
            if info["frames"] == 0:
                outputs["world_view_skipped"] = info.get("skipped_reason", "unknown")
            else:
                outputs["world_view"] = info["out_path"]
```

**Artifacts and keys `publish` must read (cross-file seams, confirm one by one)**:
- `prepared/frames_left.npy` -> `frames (T,H,W,3)` (existing)
- `hand.npz` -> `joints_2d (T,2,21,3)`, `hand_present (T,2)` (**newly read in Task 4**)
- `scene.npz` -> `camera_poses (T,4,4)`, `depth (T,H,W)` (**newly read in Task 4**)
- `fused.npz` -> `joints_world`, `reprojection_error`, `depth_source`, `joint_source`, `gapfill_reasons`, `gapfill_skipped`, `identity_switches` (existing + new in Plan B)
Any missing/corrupt artifact -> skip the corresponding view and record the reason in `outputs`, **no crash** (partial visualization beats total failure).

- [ ] **Step 5: Run the tests and commit**

Run: `conda run -n roboray python -m pytest tests/unit/test_viz.py tests/unit/test_fusion_action.py -v`, then the full suite

```bash
git add src/roboray/viz/ src/roboray/stages/publish.py tests/unit/test_viz.py
git commit -m "feat(viz): 3D world view (skeletons/camera trajectory/point cloud/future wrist trajectory) and camera-view overlay"
```

---

### Task 5: QC report upgrade (error curves / depth source / per-frame diagnostics)

**Files:**
- Modify: `src/roboray/stages/publish.py` (refactor the QC section + curve plotting)
- Modify: `src/roboray/viz/__init__.py` (export `plot_reprojection_curve`)
- Create: `src/roboray/viz/qc_plots.py`
- Test: `tests/unit/test_qc_report.py`

**Interfaces:**
- Produces:
  - `plot_reprojection_curve(reprojection_error, out_png, *, flag_value) -> dict`: one curve per hand (trusted frames participate; flagged frames are excluded and the flagged intervals drawn); empty/all-flagged -> returns `{"plotted": False, "reason": ...}` and produces no misleading figure
  - The QC HTML gains: the error-curve PNG (base64 embedded), depth-source statistics (stereo/monocular frame counts), a per-frame diagnostics table (sparse, downsampled by a threshold), the fill audit (Plan B's already-persisted reasons/skipped)
- Note: the report must self-evidence the data is real — the curve numbers match `fused.npz`'s `reprojection_error`; with no data, write "no trusted frames" explicitly

- [ ] **Step 1: Write the failing test**

`tests/unit/test_qc_report.py`:

```python
import json
from pathlib import Path

import numpy as np
import pytest

from roboray.runtime.episode import EpisodeWorkspace
from roboray.stages.publish import run_publish
from roboray.viz.qc_plots import plot_reprojection_curve

FLAG = 1e6
FAKES = Path(__file__).resolve().parents[1] / "fakes" / "modelenv_scripts"


def _ws_with_artifacts(tmp_path, t=8):
    ws = EpisodeWorkspace(tmp_path, "ep_0001")
    ws.write_artifact("fused.npz", {
        "joints_world": np.zeros((t, 2, 21, 3), np.float32),
        "reprojection_error": np.array([[1.0, 2.0]] * t, np.float32),
        "depth_source": np.array(["stereo"] * t, dtype=object),
        "identity_switches": np.zeros((0, 2), np.int32),
        "gapfill_filled": np.int32(0), "gapfill_rejected": np.int32(1),
        "gapfill_reasons": np.array(json.dumps({"insufficient_keypoint_depth": 1})),
        "gapfill_skipped": np.array(json.dumps({})),
        "joint_source": np.array("mano"),
        "n_frames": np.int32(t), "fps": np.int32(30),
    })
    ws.write_artifact("action.npz", {
        "labels": np.array(["reach"], dtype=object),
        "segment_start": np.array([0], np.int32), "segment_end": np.array([t], np.int32),
        "state": np.zeros((t, 2, 21, 3), np.float32), "action": np.zeros((t, 2, 21, 3), np.float32),
        "n_frames": np.int32(t), "fps": np.int32(30),
    })
    np.save(Path(ws.prepared_dir) / "frames_left.npy", np.zeros((t, 16, 16, 3), np.uint8))
    (Path(ws.dir) / "state.json").write_text(json.dumps({"timings": {"fusion": 0.1}, "error": None}))
    return ws


def test_curve_plot_marks_flagged_frames(tmp_path):
    err = np.array([[1.0, 1.0], [FLAG, 1.0], [3.0, 1.0]], np.float32)
    res = plot_reprojection_curve(err, tmp_path / "c.png", flag_value=FLAG)
    assert res["plotted"] is True
    assert (tmp_path / "c.png").exists() and (tmp_path / "c.png").stat().st_size > 0


def test_curve_plot_all_flagged_reports_instead_of_faking(tmp_path):
    err = np.full((4, 2), FLAG, np.float32)
    res = plot_reprojection_curve(err, tmp_path / "c.png", flag_value=FLAG)
    assert res["plotted"] is False
    assert res.get("reason")
    assert not (tmp_path / "c.png").exists()


def test_qc_report_contains_curve_and_depth_source(tmp_path):
    ws = _ws_with_artifacts(tmp_path)
    info = run_publish(workspace_root=str(tmp_path), output_root=str(tmp_path / "out"),
                       episode_id="ep_0001", make_visualization=False,
                       camera_view=False, world_view=False, fps=30,
                       export_script=str(FAKES / "fake_lerobot_export.py"))
    html = (tmp_path / "out" / "qc" / "ep_0001" / "report.html").read_text()
    assert "reprojection error" in html
    assert "stereo" in html
    assert "data:image/png;base64," in html or "no curve" in html
    assert info["outputs"]["qc_report"]
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `conda run -n roboray python -m pytest tests/unit/test_qc_report.py -v`
Expected: FAIL (`roboray.viz.qc_plots` does not exist; `run_publish` has no `export_script` parameter yet)

- [ ] **Step 3: Implement `qc_plots.py` and the QC refactor**

`src/roboray/viz/qc_plots.py`:

```python
"""QC charts: reprojection-error curve (recognizes flagged frames; draws no fake figure when there is no data)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

__all__ = ["plot_reprojection_curve"]


def plot_reprojection_curve(reprojection_error, out_png, *, flag_value: float = 1e6) -> dict:
    err = np.asarray(reprojection_error, np.float64)
    if err.ndim != 2 or err.size == 0:
        return {"plotted": False, "reason": "no usable reprojection-error data"}
    trusted = np.isfinite(err) & (err < flag_value)
    if not trusted.any():
        return {"plotted": False, "reason": "all frames are flagged (no observation to compare against)"}
    t, hands = err.shape
    fig, ax = plt.subplots(figsize=(6, 2.4), dpi=100)
    names = ["left hand", "right hand"]
    for h in range(hands):
        y = np.where(trusted[:, h], err[:, h], np.nan)
        ax.plot(np.arange(t), y, linewidth=1.0, label=names[h] if h < len(names) else f"hand{h}")
        flagged = np.where(~trusted[:, h])[0]
        if flagged.size:
            ax.scatter(flagged, np.zeros_like(flagged), marker="x", s=12,
                       color="red", label="flagged (untrusted/filled)" if h == 0 else None)
    ax.set_xlabel("frame"); ax.set_ylabel("reprojection error (px)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)
    return {"plotted": True, "out_png": str(out_png),
            "trusted_frames": int(trusted.sum()), "flagged_frames": int((~trusted).sum())}
```

The QC section of `stages/publish.py`: replace the original `_HTML` template and summary logic (key points):
- read `reprojection_error`/`depth_source`/`gapfill_reasons`/`gapfill_skipped`/`identity_switches`/`joint_source` from `fused.npz`
- call `plot_reprojection_curve(...)` -> PNG -> base64-embedded `<img src="data:image/png;base64,...">`; when `plotted=False`, write "no curve: {reason}"
- depth-source statistics: `Counter(str(x) for x in depth_source)` into the table
- per-frame diagnostics: downsampled (one row per `max(1, t//40)` frames) listing frame number / left-right errors / depth_source
- fill audit: Plan B's already-persisted `gapfill_reasons`/`gapfill_skipped` JSON rendered as per-item counts
- `run_publish` gains the keyword `export_script: str | None = None` (default None -> `LeRobotWriter()` uses the real script; tests pass the fake-script path). Product code must not reference `tests/`.
- **Must update the call site written in Task 2**: `LeRobotWriter(script=export_script)` rather than `LeRobotWriter()`, otherwise the new parameter is a dead letter (cross-task seam).
- On empty/all-flagged data, QC states the reason explicitly, drawing no fake figure

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `conda run -n roboray python -m pytest tests/unit/test_qc_report.py tests/unit/test_viz.py -v`, then the full suite

- [ ] **Step 5: Commit**

```bash
git add src/roboray/viz/qc_plots.py src/roboray/stages/publish.py tests/unit/test_qc_report.py
git commit -m "feat(qc): reprojection-error curve, depth-source statistics and per-frame diagnostics (states it explicitly instead of drawing fake figures when there is no data)"
```

---

### Task 6: Full verification and the Plan C verification report

**Files:**
- Create: `docs/superpowers/plans/C-verification-report.md`
- Modify: `README.md` (the outputs section updated to the real LeRobot dataset + both views + QC curves)
- Modify: `docs/models.md` (add `ROBORAY_LEROBOT_PYTHON` and the `ROBORAY_VLM_*` variables; record that Plan B's knob is fixed)

**Interfaces:**
- Consumes: everything above
- Produces: an all-green suite plus an honest boundary (which are real calls, which are still [L2])

- [ ] **Step 1: Full test run**

Run: `conda run -n roboray python -m pytest -o addopts="" -m "not gpu"`
Expected: all pass (>= 180)

- [ ] **Step 2: End-to-end CLI smoke (fake-model full chain, including the new export/visualization/QC)**

```bash
conda run -n roboray python scripts/run_local_fake.py --config configs/local-fake.yaml --run-root /tmp/plan-c-smoke
```
(If `configs/local-fake.yaml` has no `ROBORAY_LEROBOT_PYTHON`, the export runs the real script and fails — during the smoke, set `ROBORAY_LEROBOT_PYTHON=<ego-lerobot python>` or provide a fake export script; choose per how you built the env, and record the method and result in the report.)

> **Deprecated (superseded by a9f486e)**: the temporary workaround of "`export ROBORAY_LEROBOT_PYTHON` in the shell"
> above **no longer works**. a9f486e made `resolve_env_request` unconditionally push `ROBORAY_LEROBOT_PYTHON`
> into `runtime_env.env_vars` (default `"python"`), and runtime_env's explicit values **override** inherited shell
> variables — so a shell export is overwritten and has no effect. The correct approach is to configure it in the
> config's `models:` section: `configs/local-fake.yaml` already has `models.lerobot_python: "python"` +
> `models.lerobot_script: tests/fakes/modelenv_scripts/fake_lerobot_export.py` (local fake mode, torch-free);
> on a multi-node cluster with real models you must set `lerobot_python: /.../ego-lerobot/bin/python` in the config
> `models:` section (see the comment block in `configs/default.yaml`).
Expected: `failed []`, with outputs containing `lerobot_v3/<ep>/meta/info.json`, `viz/<ep>/{camera,world}.mp4`, `qc/<ep>/report.html`

- [ ] **Step 3: If the ego-lerobot env is available, verify the read-back**

```bash
ROBORAY_LEROBOT_PYTHON=$(conda info --base)/envs/ego-lerobot/bin/python python - <<'PY'
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset(repo_id="roboray_episode", root="/tmp/plan-c-smoke-out/lerobot_v3/ep_0001")  # repo_id is a required arg (verified on 0.6.1)
print("frames:", len(ds), "fps:", ds.fps, "features:", sorted(ds.features))
PY
```
Expected: prints the frame count/features without error (**this is the hard evidence of a "valid format"**). If the env is unavailable, record it as [L2].

- [ ] **Step 4: Write the verification report**

`C-verification-report.md`: environment (including the ego-lerobot env build result), test counts, real-verification coverage (LeRobot write/read-back, the VLM client-injection path and its error branches, visualization frame production, QC curves and statistics), **items still unverified** (real VLM service end-to-end, real-model inference -> Plan D/L2), risks (lerobot version API drift, VLM service availability, visualization performance). All numbers actually measured.

- [ ] **Step 5: Update README and docs/models.md and commit**

```bash
git add docs/superpowers/plans/C-verification-report.md README.md docs/models.md
git commit -m "docs: Plan C verification report, README outputs description and docs/models.md variable additions"
```

---

## Self-Review

**1. Spec coverage**

| Spec requirement | Corresponding task |
| --- | --- |
| §9 action understanding + AI Function (real VLM, pluggable, errors when unreachable) | Task 3 |
| §9 state/action tensors | Consumed by Tasks 4/5; flattened before handoff to LeRobot |
| §10 LeRobot output (**via the lerobot public API**) | Task 2 |
| §10 camera view (original field of view/aspect ratio + 2D keypoint overlay) | Task 4 (**2D keypoints only**; the MANO reprojection overlay is deferred as a Plan D item, the data is already available) |
| §10 world view (hands/camera/point cloud/next-1s wrist trajectory) | Task 4 |
| §10 QC report (reprojection error, depth source, per-frame diagnostics, fill audit) | Task 5 (+Plan B's already-persisted audit) |
| §10 publish in completion order + manifest | Delivered by Plan A; this plan does not change the publish order |
| §6/Plan B leftover: FFS resolution knob's depth-grid semantics | Task 1 |

**2. Placeholder scan**: `run_lerobot_export.py`'s `[L2]` annotations are explicitly declared verification points (with verification methods); the code is complete. The remaining task code is complete. No `use_fake_export`-style leftovers (product code does not reference tests/; tests inject the fake script via `export_script=`).

**3. Type consistency**: `LeRobotWriter.write` (defined Task 2, consumed Task 5); `render_world_view`/`render_camera_view` (defined Task 4, consumed Task 5/publish); `plot_reprojection_curve` (defined and consumed Task 5); `OpenAIVlm.label(prompt, images, out_path, n_segments)` (Task 3 extended the third/fourth args; the `SemanticModel` protocol call site in `action.py` from Plan A updated in sync); `ModelEnvError`/`run_in_modelenv` (existing from Plan B, reused by Task 2).

**4. Review Focus coverage**

| Review Focus | Test that pins it |
| --- | --- |
| 1. lerobot env missing -> readable error | `test_writer_missing_env_gives_build_hint` (Task 2) |
| 2. dataset read-back | Task 2 Step 6 real verification + Task 6 Step 3 read-back self-verification |
| 3. VLM unconfigured/missing key -> error naming it | `test_missing_base_url_raises_naming_the_config_key`, `test_missing_api_key_raises_naming_the_env_var` (Task 3) |
| 4. VLM malformed output not fatal | `test_label_pads_and_truncates_to_segment_count`, `test_non_json_reply_is_not_fatal` (Task 3) |
| 5. stereo depth grid (Plan B gate) | `test_ffs_depth_returns_to_original_grid_after_resize`, `test_fake_ffs_reports_both_grids` (Task 1) |
| 6. QC numbers real, no fake figures on empty data | `test_curve_plot_all_flagged_reports_instead_of_faking`, `test_qc_report_contains_curve_and_depth_source`, `test_world_view_all_nan_is_reported_not_faked` (Tasks 4/5) |

## Execution Handoff

After Plan C -> Plan D (A800 KubeRay deployment, model-env build and L2 smoke execution, L3 three-tier latency comparison and the custom evaluation).
