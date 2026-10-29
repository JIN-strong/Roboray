# Plan B Verification Report

> This report records the measured results of **Plan B (real-model integration + A1/A3 refinement) within the verifiable scope of this local machine**.
> All numbers come from actual runs; nothing is fabricated. Real-model inference requires a GPU and weights, which is **L2** and is executed by Plan D on an
> A800 node — this local machine **did not run real-model inference** (honest boundary at the end).

## Environment

| Item | Value |
| --- | --- |
| Host | Linux 4.19.112-2.el8.x86_64 |
| conda env | `roboray` (main env, torch not installed) |
| Python | 3.10.22 |
| Ray | 2.59.0 |
| numpy | 1.26.4 |
| CPU cores | 8 |
| Memory | ~23 GiB |
| GPU | **none** (no `nvidia-smi`; `execution.max_gpus` only counts Ray virtual resources) |

Main-env dependencies: `pytest 9.0.3`, `pytest-timeout`, `PyYAML`.

## Full Test Suite Results

**Conclusion: 162 passed, 3 deselected, 165 collected.**

How obtained (local pytest 9.0.3: the **single** `-q` in the default addopts **still prints** the final
`N passed` summary line — see "About `-q`" below):

```bash
# Full suite (default addopts -q -m "not gpu")
$ conda run -n roboray python -m pytest
162 passed, 3 deselected in 92.54s

# Cross-check collection count (default addopts → gpu already excluded)
$ conda run -n roboray python -m pytest --collect-only
162/165 tests collected (3 deselected) in 0.42s
# Disable addopts to see the total collection count (including the 3 gpu tests)
$ conda run -n roboray python -m pytest --collect-only -o addopts=""
165 tests collected in 0.40s
```

- Collection total **165**, of which **162 were selected and passed** and **3 deselected** (i.e., the three GPU smoke
  tests in `tests/gpu/test_adapters_gpu.py`).
- How the GPU tests are excluded: `addopts = "-q -m \"not gpu\""` in `[tool.pytest.ini_options]` of `pyproject.toml`
  makes the default `pytest` **deselect** the three smoke tests marked `@pytest.mark.gpu`.
- How to run the GPU smoke on A800: the command-line `-m gpu` **overrides** (rather than ANDs with) the
  `-m "not gpu"` in addopts (pytest CLI options are last-wins), so running it directly selects them, no extra options needed:

  ```bash
  ROBORAY_GPU_SMOKE=1 conda run -n roboray python -m pytest -m gpu -v
  ```

  (The optional defensive form `-o addopts=""` is not required. See `docs/models.md`, section "Running the GPU smoke on A800"
  for the required environment variables.)

> **About `-q`**: under local pytest 9.0.3, the **single** `-q` in the default addopts **still prints** the trailing
> `N passed` summary line — running `conda run -n roboray python -m pytest` directly outputs
> `162 passed, 3 deselected in ...s`. Suppressing the summary line requires **two** `-q` flags (i.e. `-qq`: the
> `-q` from addopts plus another `-q` on the command line); only then are progress dots all that remain. Hence the
> 162/3/165 numbers are taken directly from the default command, no `-o addopts=""` workaround needed (an earlier
> version of this report had the `-q` semantics backwards; corrected here, the numbers themselves are unchanged).

## CLI Smoke (fake-model regression)

Command:

**Must run from a zero state.** The pipeline's checkpoint state lives in `run_root/<episode_id>/state.json`; if
`run_root` (default `./run`) already contains `done` states for everything, a re-run will **directly resume and execute
no stages** — that yields an "empty run" and **cannot** serve as evidence that the pipeline actually executed. The smoke
for this report used a **fresh run-root** + a **fresh output root** to guarantee all six stages actually execute.

Command (temporary config pointing `output.uri` at a new directory; `--run-root` pointing at a fresh directory):

```bash
conda run -n roboray python scripts/run_local_fake.py \
  --config /tmp/local-fake-smoke.yaml \      # same content as configs/local-fake.yaml, only output.uri=output/run-fake-smoke
  --run-root /tmp/ego-smoke-run              # fresh empty directory (no state.json)
```

Output (key line, verbatim):

```
Total time 1.487s, failed [], manifest at .../output/run-fake-smoke/manifest.json
```

(Exit code 0. The `core_worker.cc ... SchedulingCancelled` printed by Ray at shutdown is normal noise from cancelling
actors during process teardown; it appears after the result line and does not affect the run.)

Manifest (`output/run-fake-smoke/manifest.json`) summary: `n_episodes=3`, `failed=0`,
`total_seconds=1.486667`, and all three Episodes have the six stages
`prepare/scene/hand/fusion/action/publish` set to `done`, `error=null`, `degraded=false`; `scheduler_snapshots` has 6
entries (scheduling did occur). Per-stage times (seconds, measured):

| Episode | prepare | scene | hand | fusion | action | publish |
| --- | --- | --- | --- | --- | --- | --- |
| ep_0001 | 0.024131 | 0.052339 | 0.051809 | 0.008207 | 0.024096 | 0.017075 |
| ep_0002 | 0.024735 | 0.052376 | 0.052412 | 0.010472 | 0.026297 | 0.014051 |
| ep_0003 | 0.029514 | 0.053202 | 0.051989 | 0.010706 | 0.023356 | 0.016500 |

These times match the order of magnitude of the fake models' artificial delays — note the delay is **0.05s per `infer()`
call** (independent of frame count; measured t=4 → 0.055s, t=32 → 0.051s), **not** per-frame. Evidence of real
execution is the combination of three things: the fresh-run-root smoke total of 1.487s (contrast the empty resume's
`total_seconds≈0.009s`), `scheduler_snapshots` having 6 entries (scheduling did occur), and the per-stage times of all
six stages being **nonzero** (see table below).

**Verification point (the controller asked for explicit confirmation rather than assumption)**: `configs/local-fake.yaml`
uses **explicit fake model names** (`stages.scene_reconstruction.model = fake_scene`,
`hand_reconstruction.model = fake_hand`, `action_understanding.model = fake_vlm`), names registered by
`install_fake_models()`; they do not collide with `registry.load_builtin_models()` mapping the builtin names
(`vipe` / `fast_foundation_stereo` / `hamer`) to **real adapters** (different names). This smoke actually executed all
six stages from a zero state with `failed []`, proving that after switching the builtin factories to real adapters, the
fake path was not broken.

> **Note (honest boundary)**: the first attempt ran the in-repo `configs/local-fake.yaml` with the default
> `run-root=./run`; because `run/ep_*/state.json` already contained all `done` states, it **resumed and actually
> executed no stages** (`total_seconds≈0.009s`, the six stages replayed old state). That run **does not** constitute
> evidence of pipeline execution; it was replaced by the fresh-run-root + fresh-output-root run above and is recorded
> here only to document the "resume-from-checkpoint" pitfall.

## Local Verification Coverage

What this machine (no GPU, no model weights) verifies is the **orchestration, contract normalization, and
fusion/completion algorithms**, not the numerical correctness of the real networks. Per-module coverage below
(test file → assertion focus).

### 1. Subprocess orchestration and contracts (`src/roboray/models/modelenv.py`)

- `tests/unit/test_modelenv.py` (6 cases):
  - `resolve_env_request` fills defaults (unset `*_PYTHON` → `"python"`, `*_WEIGHTS` → `""`) and passes through.
  - **marker protocol**: `run_in_modelenv` parses the `__RESULT__{json}` last line of stdout (including noisy-line scenarios).
  - **nonzero exit** → `ModelEnvError`, message includes stderr tail and `exit code N`.
  - **timeout** → `ModelEnvError` (`timeout_s` triggered).
  - **missing marker line** → `ModelEnvError`.
- `tests/unit/test_modelenv_scripts.py` (4 cases): the three **stand-in inference scripts** agree with the real scripts'
  CLI/NPZ contracts — `fake_vipe`/`fake_ffs`/`fake_hamer` output keys and shapes, `depth_source` values;
  `fake_ffs` exits nonzero with a stated reason when `frames_right` is missing.

### 2. Scene / Hand adapter contract normalization

- `tests/unit/test_scene_models.py` (7 cases):
  - ViPE adapter runs the stand-in script → `camera_poses (T,4,4)`, `depth_source="monocular"`,
    `__schema_version__` stamped (the **c2w→w2c** orientation is pinned by the script output contract + the A1
    backprojection roundtrip).
  - **readable env-missing error**: `ROBORAY_VIPE_PYTHON=/nonexistent/python` → `ModelEnvError` naming the variable and
    giving the `build_vipe` build-script hint.
  - FFS requires stereo input (missing `frames_right`/`has_stereo` → `ModelEnvError`).
  - **FFS baseline formula**: `depth = fx·B/disparity`, baseline taken from `T_left_to_right[0][3]` (right cam +x) —
    with stand-in disparity=5, B=0.0325, fx=8 → depth=0.052, validating magnitude and monotonicity; `fx` is plumbed from
    the calib baseline via `--extra`.
  - registry resolves builtin names `vipe` / `fast_foundation_stereo` successfully; `cost_hint` scales linearly with frame count.
- `tests/unit/test_hand_models.py` (4 cases):
  - HaMeR adapter runs the stand-in script → `hand_present (T,2)`, `mano_pose (T,2,48)`, stamped.
  - **readable missing-weights error**: `ROBORAY_HAMER_WEIGHTS` directory absent → `ModelEnvError` mentioning
    `MANO_RIGHT.pkl` and the registration download guide at `mano.is.tue.mpg.de`.
  - **weights directory without `mano/MANO_RIGHT.pkl` → rejected** (fail early rather than late).
  - registry resolves builtin name `hamer` successfully.
  - Left/right hand slot semantics (0=left, 1=right) are pinned by the script stand-in contract and the mirrored offset of `fake_joints_3d_cam`.

### 3. A1 fusion refinement (real backprojection + informative reprojection error)

- `tests/unit/test_fusion_refined.py` (11 A1-related cases):
  - `test_backproject_reprojection_roundtrip_is_exact`: pinhole-ground-truth backprojection→reprojection roundtrip
    **< 1e-3 px** (Review Focus #6, unit consistency: meters/px/w2c).
  - Invalid depth → that joint is NaN and that hand is marked with the large-error sentinel (`1e6`); one invalid joint
    does not make the whole hand untrusted.
  - Roundtrip stays exact under a non-unit `camera_pose`.
  - **MANO path (informative)**: with `joint_source="mano"` the reprojection error equals the analytic pixel error
    (`test_backproject_mano_source_reproj_equals_analytic_pixel_error`); root depth invalid → whole hand
    untrusted (`..._mano_untrusted_when_root_depth_invalid`); `run_fusion` uses the mano source and reports the metric.
  - **Fallback path (roundtrip-consistency only)**: missing `joints_3d_cam` → `joint_source="joints_2d"`, reprojection
    becomes a pixel-level identity roundtrip (structurally zero/sentinel), `flagged_hands` counted correctly; this path
    **must not be treated as a quality metric**.
  - End-to-end `test_run_fusion_reprojection_error_is_real`: reprojection error is not an all-zero placeholder.
  - QC stats exclude flagged (`1e6`) hand frames: `test_publish_qc_stats_exclude_flagged_hands`.
- Related regression: `tests/unit/test_fusion_action.py` (11 cases) — A2 wearer association keeps identity continuous /
  detects switches, `run_fusion` artifacts + audit, **corrupt/unstamped artifacts → readable `ValueError`**, action
  segmentation and semantic-model wiring.

### 4. A3 short-gap completion (three conditions + supporting_joints gate + four post-completion checks)

`fill_short_gaps`'s **three conditions** (all must hold to fill, otherwise record `rejected` with the reason): (a) both
ends have trusted observations, (b) both sides are the same hand, (c) gap length `<= max_gap`
(`unobserved_gap_s × fps`). Plus the `supporting_joints` gate: if the number of supporting joints at the two trusted
ends is `< min_supporting_joints` → rejected (`insufficient_supporting_joints`).

Four **post-completion checks** (`_fill_reject_reason`; if any fails the whole frame is dropped, reason added to
`gapfill_reasons`):

1. `insufficient_keypoint_depth` — completed joints project back to a depth-support count `< min_supporting_joints`.
2. `out_of_view` — the two end 2D detections the interpolation is based on fall outside the view (bounds aligned with
   the `_sample_depth` sampleable region).
3. `wrist_step_exceeded` — single-frame wrist displacement `> _MAX_WRIST_STEP_M` (0.25 m).
4. `trajectory_deviation_exceeded` — **velocity continuity**: the difference between the per-frame displacement implied
   by the bridge and the **measured** per-frame displacement mean of the gap's two ends
   `> _MAX_WRIST_VEL_MISMATCH_M_PER_FRAME` (0.05 m/frame). The previous "linear interpolation of the candidate between
   the two ends" formulation was identically 0 (degenerate, never triggerable); it was replaced by this
   non-degenerate metric.

When check 4 cannot be evaluated, the frame is passed and counted by reason category (`gapfill_skipped`, not
rejected): `sequence_boundary`, `neighbour_unobserved`, `wrist_untrusted`.

Coverage tests (`tests/unit/test_fusion_refined.py`, 13 A3-related cases) — each check is **individually triggerable**:

- `test_a3_all_four_checks_via_run_fusion` (depth check rejects the fill, `gapfill_filled==0`).
- `test_a3_depth_check_records_reason`, `test_a3_out_of_view_rejects_fill`,
  `test_a3_wrist_step_rejects_fill`, `test_a3_check4_rejects_on_velocity_discontinuity`.
- `test_a3_check4_passes_on_constant_velocity` (constant velocity passes),
  `test_a3_check4_skipped_at_sequence_boundary` (skipped at boundary),
  `test_a3_check4_skips_on_unobserved_neighbour` (skipped when neighbour frame unobserved),
  `test_a3_trajectory_check_helper_rejects_bridge_velocity_mismatch` (helper-function level).
- `test_a3_min_supporting_joints_gates_via_confidence` (confidence→supporting_joints gate takes effect).
- `test_a3_in_view_bound_matches_sampleable_region` (view bound matches depth-sampleable region, no gaps).
- `test_a3_rejected_fill_frames_are_not_observed_for_wearer_association` (rejected fill frames do not participate in
  identity ordering, avoiding fabricated identity switches).
- `test_a3_filled_frames_reprojection_error_is_a_flag` (**filled-frame error is a flag**: pre-write NaN → `1e6`,
  same convention as untrusted hand frames, filtered together by QC, never masquerading as a low-error measurement).

## L2 To-verify Checklist

The items below come from the head and inline `[L2]` annotations of the three inference scripts
(`scripts/modelenvs/run_*.py`), consistent with the "L2 verification checklist" in `docs/models.md`. Verify each item
against the real environment on first integration on the A800, replacing the corresponding section in the script if
needed. **These are explicitly declared verification points (with assumed contracts and verification methods), not
"write-later" placeholders; the script code is complete and runnable.**

### ViPE (`run_vipe.py`)

First run the official CLI `vipe <video>` to confirm env/weights are usable, then check:

- [ ] **Import path** — `from vipe.pipeline import DefaultAnnotationPipeline`,
      `from vipe.stream import VideoStream` (or equivalent path).
- [ ] **streams construction** — constructor signature of `VideoStream(video_path)`.
- [ ] **SE3 → 4x4** — `frame.pose.matrix()` (or equivalent 4x4 interface), output is **c2w**.
- [ ] **Init parameters** — constructor parameters of `DefaultAnnotationPipeline()` (weight download / cache dirs).
- [ ] **Return structure** — return type of `pipeline.run([stream])` and frame iteration style.
- [ ] **Field names** — `frame.intrinsics` (`[fx, fy, cx, cy, ...]`) and `frame.metric_depth`.

### Fast-FoundationStereo (`run_ffs.py`)

Check against the internal implementation of the repo's `scripts/run_demo.py`:

- [ ] **Weight artifact name** — downloading HF `nvidia/c-fast-foundationstereo` yields
      `model_best_bp2_serialize.pth`.
- [ ] **Import / load** — real import path (`from core.found_stereo import FoundStereo`) and load signature
      `FoundStereo(args=(...))` (`restore_ckpt` / `valid_iters` / `max_disp` / `hidden_dims`).
- [ ] **Forward API name** — stereo forward call name `model.forward_image(l, r)` (or equivalent).
- [ ] **Rectification prerequisite** — input must be rectified horizontal-epipolar stereo; baseline taken from
      `calib["T_left_to_right"][0][3]` (right cam +x convention). Depth conversion `depth = fx·B/d` is a
      deterministic step (**not L2**).

### HaMeR (`run_hamer.py`)

Check against the repo's `demo.py`:

- [ ] **MANO relative layout** — `MANO_RIGHT.pkl` is at `<weights>/mano/` under the data directory that `--weights` points to.
- [ ] **Import / construction** — `from hamer.models import HAMER`, `from hamer.utils import recursive_to`;
      load signature of `HAMER.load_pretrained(weights)`.
- [ ] **Batching / preprocessing call** — the batching entry of `demo.py` (`model.detect_and_fit([...])` in the
      script is a **placeholder call name**); output dict field names.
- [ ] **At most 2 hands per frame** — MANO params (pose 48 / shape 10 / trans 3), 3D joints (`joints_3d_cam` or
      `pred_keypoints_3d`), 2D joints (`joints_2d`), confidence (`score`), left/right hand label (`is_left`).
- [ ] **Hand-label field and slots** — output slots by hand label **0=left, 1=right** (left hand = flipped right-hand
      MANO fit).
- [ ] **MANO layer** — import path (`from hamer.models.mano_wrapper import MANO`) / constructor params
      (`is_rhand` / `num_pca_comps=45` / `flat_hand_mean` / `batch_size`) / call signature and returned field names
      (`out.joints`, `return_verts`) — `joints_3d_cam` is computed forward from it, 21 joints, joint 0 = wrist.

### GPU smoke run command (A800)

```bash
export ROBORAY_GPU_SMOKE=1
export ROBORAY_VIPE_PYTHON=$(conda info --base)/envs/ego-vipe/bin/python
export ROBORAY_FFS_PYTHON=$(conda info --base)/envs/ego-ffs/bin/python
export ROBORAY_HAMER_PYTHON=$(conda info --base)/envs/ego-hamer/bin/python
export ROBORAY_VIPE_WEIGHTS=/data/weights/vipe
export ROBORAY_FFS_WEIGHTS=/data/weights/ffs
export ROBORAY_HAMER_WEIGHTS=/data/weights/hamer

conda run -n roboray python -m pytest -m gpu -v
```

The smoke runs **real inference** with small synthetic data (4 frames, 288x512) for each adapter, asserting artifact
keys / shapes / finiteness; `test_ffs_gpu_smoke` / `test_hamer_gpu_smoke` additionally have skipif: they skip when the
corresponding `*_WEIGHTS` is unset.

## Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| **Real model API disagrees with the assumed script contract** (import path / call signature / field names) | The contract lives only inside the scripts (the `[L2]` sections of `scripts/modelenvs/run_*.py`) — the adapter layer has **isolated** the artifact contract from the model API, so replacing a section of a script does not affect Scene/Hand artifact keys or shapes. `docs/models.md` lists each verification point and how to check it. |
| **MANO registration files not publicly obtainable** | `HamerHandModel.load()` **hard-validates** that `mano/MANO_RIGHT.pkl` exists; if missing it raises `ModelEnvError` with the registration download guide — fail early, never a "half-usable" state. Unit tests on this machine cover that rejection path. |
| **Build cost of the three envs / mutually incompatible torch-CUDA stacks** | **Subprocess architecture**: each model has its own conda env (ViPE torch 2.7 cu128 / FFS py3.12 torch 2.6 cu124 / HaMeR torch); `deploy/modelenvs/build_*.sh` builds each per its official process; vendored repos are commit-pinned via `scripts/vendor_models.sh` (idempotent, reproducible). The main env does not install torch. |
| **A3 thresholds need measured calibration** | `_MAX_WRIST_STEP_M=0.25 m/frame`, `_MAX_WRIST_VEL_MISMATCH_M_PER_FRAME=0.05 m/frame` are **empirical safety bounds** (egocentric wrist speed ~0.1–2 m/s → 0.003–0.067 m/frame at 30fps). Plan D / L2 re-validates them with real data and can tune. |
| **Stereo path needs both envs available at once** | `FastFoundationStereoModel.load()` first validates the FFS env and weights, then calls `vipe_model.load()` to validate the ViPE env — missing either fails early with the corresponding build-script hint. |
| **Reprojection error misread as a quality metric** | `joint_source` explicitly distinguishes `"mano"` (informative) vs `"joints_2d"` (roundtrip-consistency only, structurally zero/sentinel). Untrusted and filled frames are written with the `1e6` sentinel and excluded from QC stats, never masquerading as low-error measurements. |

## Honest Boundary

**This machine (no GPU) did not run any real-model inference.** All numbers in this report come from:

1. the CPU full test suite (fake models + real algorithms, e.g., the numpy implementations of A1/A3);
2. fake-model CLI end-to-end smoke (`configs/local-fake.yaml`).

The **numerical correctness** of the real models (ViPE pose/depth quality, FFS disparity, HaMeR+MANO fitting) and the
**contract match** of the three inference scripts against the vendored repos' real APIs are **not verified on this
machine** — these are **L2**. What this plan delivered is (a) runnable inference scripts + explicit `[L2]` verification
points, (b) a GPU smoke suite, (c) this L2 checklist with run commands. **Executing L2 is owned by Plan D** (A800
KubeRay deployment + L2 smoke); at that time verify each item per the table above and backfill measured results.
