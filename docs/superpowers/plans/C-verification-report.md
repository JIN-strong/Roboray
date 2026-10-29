# Plan C Verification Report

> This report records the measured results of **Plan C (Stage 3/4 making-real: real LeRobot v3 export, real VLM,
> dual-view visualization, QC report) within the verifiable scope of this local machine**. All numbers come from
> actual runs; nothing is fabricated. **Still on the L2 (A800 / Plan D) boundary**: end-to-end calls to a real VLM
> service and real-model inference. This machine **did not complete a VLM end-to-end run** (no usable service, and the
> `socks://` proxy breaks the real `openai` client) — that failure **was correctly wrapped as `VlmError` by the code**,
> recorded honestly as [L2] below.

## Environment

| Item | Value |
| --- | --- |
| Host | Linux 4.19.112-2.el8.x86_64 |
| conda main env | `roboray` (**torch not installed**) |
| Python | 3.10.22 |
| Ray | 2.59.0 |
| numpy | 1.26.4 |
| pytest | 9.0.3 (`addopts = -q -m "not gpu"`) |
| CPU cores | 8 |
| GPU | **none** (no `nvidia-smi`; `execution.max_gpus` only counts Ray virtual resources) |
| Export env | `ego-lerobot` (Python 3.12 + CPU torch + `lerobot==0.6.1`) |
| Export env interpreter (measured path) | `~/miniforge3/envs/default/envs/ego-lerobot/bin/python` |
| lerobot version (measured) | **0.6.1** (`import lerobot; lerobot.__version__` → `0.6.1`) |

> **The export-env path pitfall (worth recording)**: `deploy/modelenvs/build_lerobot.sh` echoes
> `$(conda info --base)/envs/ego-lerobot/bin/python` at the end. But on this machine the env was created inside the
> `default` env, so conda **nested** it at `~/miniforge3/envs/default/envs/ego-lerobot/` —
> i.e., `envs/ego-lerobot/bin/python` under `conda info --base` (`~/miniforge3`) **does not exist**
> (measured: `ls` reports "no such file or directory"). `conda env list` lists only the nested path.
> Therefore `ROBORAY_LEROBOT_PYTHON` must point at **the measured absolute path above**, not
> `$(conda info --base)/envs/ego-lerobot/bin/python`.

## Full Test Suite Results

**Conclusion: 206 passed, 3 deselected @HEAD (collected 209 → 3 deselected → 206 selected), 144.34s.**
(**205 @77329f0 before the fix / 206 @HEAD**: a9f486e added `test_resolve_env_request_passes_lerobot_values`,
one case. The pre-fix full run was `205 passed, 3 deselected in 150.97s`; no other numbers are affected.)

```bash
$ conda run -n roboray python -m pytest -o addopts="" -m "not gpu"
...
collected 209 items / 3 deselected / 206 selected
...
================ 206 passed, 3 deselected in 144.34s (0:02:24) =================
```

- `-o addopts=""` is **not** required to see the counts (measured on local pytest 9.0.3): the **single** `-q` in the
  default `addopts = "-q -m \"not gpu\""` **still prints** the final `N passed` summary line; suppressing the summary
  requires **two** `-q` flags (i.e. `-qq`: addopts' `-q` plus one more command-line `-q`). The explicit
  `-o addopts=""` here is only to get detail not compressed by `-q`; semantics agree with "About `-q`" in
  `B-verification-report.md`. (An earlier version of this file had the `-q` semantics backwards and wrongly claimed a
  single `-q` suppresses the summary — corrected here, the numbers themselves are unchanged.)
- The 3 deselected are the three GPU smoke tests in `tests/gpu/test_adapters_gpu.py` (`@pytest.mark.gpu`,
  and they only really execute with `ROBORAY_GPU_SMOKE`) — this machine has no GPU, correctly excluded.
- Versus Plan B (162 passed): this plan adds 44 cases (Task 1 depth grid, Task 2 export, Task 3 VLM,
  Task 4 visualization, Task 5 QC; +1 is a9f486e's `test_resolve_env_request_passes_lerobot_values`).

## End-to-End CLI Smoke (real LeRobot export, all six stages)

**Run from a zero state**: checkpoint state lives in `run_root/<episode>/state.json`; if `run_root` already has all
`done`, a re-run directly resumes and **executes no stages** (a pitfall already recorded in the Plan B report). This run
used a **fresh run-root** + a **fresh output root** to guarantee all six stages actually execute.

To avoid cross-talk with the historical `output.run-fake`, a temporary config points `output.uri` at a new directory
(content same as `configs/local-fake.yaml`, only `output.uri` points at the new directory and `models.lerobot_python`
points at the ego-lerobot interpreter):

```bash
P=~/miniforge3/envs/default/envs/ego-lerobot/bin/python
rm -rf /tmp/plan-c-smoke2 /tmp/plan-c-smoke-out2
# /tmp/plan-c-smoke2.yaml: input.uri=data/synthetic (absolutized),
#                          output.uri=/tmp/plan-c-smoke-out2, models.lerobot_python=$P
conda run -n roboray python scripts/run_local_fake.py \
    --config /tmp/plan-c-smoke2.yaml --run-root /tmp/plan-c-smoke2
```

Output (result line, verbatim):

```
Total time 14.324s, failed [], manifest at /tmp/plan-c-smoke-out2/manifest.json
```

> **Batch labels for the two smokes (do not confuse smoke-1 / smoke-2)**: this task ran two real-export smokes;
> every number below carries a batch label to avoid mixing them up:
> - **smoke-1 (first run, before the `ROBORAY_LEROBOT_PYTHON` delivery fix)**: relied on shell `export` inherited to
>   the worker via the process environment; result line `Total time 14.834s, failed []`; output root
>   `/tmp/plan-c-smoke-out`.
> - **smoke-2 (second run, driven by the config's `models.lerobot_python`, no shell export)**: i.e., the command above;
>   result line `Total time 14.324s, failed []` (verbatim); output root `/tmp/plan-c-smoke-out2`; artifacts likewise
>   real lerobot v3 layout (`meta/stats.json`, `meta/tasks.parquet`,
>   `meta/episodes/chunk-000/file-000.parquet`); **smoke-2**'s publish is 12.4–12.7s/episode.
>   **The config path is validated end to end.**

**smoke-1**'s manifest summary (measured): `{"n_episodes": 3, "failed": 0, "total_seconds": 14.833603}`,
`scheduler_snapshots` 6 entries (scheduling did occur). **smoke-1**'s three Episodes have all six stages
`prepare/scene/hand/fusion/action/publish` **`done`, `error=null`, `degraded=false`**. **smoke-1** per-stage times
(seconds, measured):

| Episode | prepare | scene | hand | fusion | action | **publish** |
| --- | --- | --- | --- | --- | --- | --- |
| ep_0001 | 0.0251 | 0.0531 | 0.0514 | 0.0166 | 0.0234 | **13.3308** |
| ep_0002 | 0.0336 | 0.0528 | 0.0517 | 0.0107 | 0.0256 | **12.8055** |
| ep_0003 | 0.0194 | 0.0528 | 0.0516 | 0.0116 | 0.0243 | **13.3043** |

**smoke-1**'s publish stage is ~13s/episode (contrast ~0.017s for Plan B's fake scripts), consistent with "the real
lerobot subprocess writes parquet" — direct order-of-magnitude evidence that the real export actually ran (fake
scene/hand inference is still ~0.05s-scale).

### How `ROBORAY_LEROBOT_PYTHON` reaches the Ray worker (after fix: config → runtime_env)

Inside the **Ray worker**, the export is done by `LeRobotWriter` reading `os.environ.get("ROBORAY_LEROBOT_PYTHON")`.

**Initial defect**: with `backend="test"`, `runtime_env.env_vars` contained only
`ROBORAY_FAKE_SCENE_SECONDS / ROBORAY_FAKE_HAND_SECONDS` + `resolve_env_request(models)`
(at that time only `ROBORAY_{VIPE,FFS,HAMER}_*`) — **not** `ROBORAY_LEROBOT_PYTHON`. The first smoke succeeded only
because under local `ray.init` the driver process environment is inherited by raylet→worker (measured visible to the
worker); but that is a **local coincidence**: a multi-node/remote cluster does not inherit the driver environment, so
the worker side would degrade to `"python"` → main env without `lerobot` → every episode silently fails (the same
class of defect as Plan B's broken `stereo_max_image_side_px` knob).

**Fix** (see "Fix record" at the end): `ModelEnvsConfig` gains `lerobot_python` / `lerobot_script`;
`resolve_env_request` flattens out `ROBORAY_LEROBOT_PYTHON` (default `"python"`) and
`ROBORAY_LEROBOT_SCRIPT` (default `""` = real script), carried into `runtime_env` via `env_vars`. **Confirmed by
reading the code** that `env_vars.update(resolve_env_request(self.request.models))` in `api/reconstruction.py` applies
to both the `test` and `real` backends (the former merged into `worker_process_setup_hook`'s `runtime_env`, the latter
into `{"env_vars": env_vars}`) — **no other wiring needed**.

Measured precedence (also probing: which wins when shell `export` and `runtime_env.env_vars` conflict):

```
driver shell value: /from/shell
worker sees: python (runtime_env wins -> shell inheritance superseded)
```

i.e., **the explicit value in `runtime_env.env_vars` overrides the inherited value** — so config is the authoritative
source: when config is unset it is `"python"` (which **overrides** the shell `export`); only when config is set is the
real interpreter path delivered. The second smoke, driven by the `models.lerobot_python` config (no shell export), ran
through, proving the fix works. **On a multi-node cluster, the "config delivers real interpreter → worker reads it →
real export" chain itself is [L2]** and must be re-verified on Plan D's KubeRay.

### Artifact tree (**smoke-1** measured, output root `/tmp/plan-c-smoke-out/`)

```
/tmp/plan-c-smoke-out/
├── lerobot_v3/ep_000{1,2,3}/
│   ├── data/chunk-000/file-000.parquet          # real lerobot parquet (not a fake skeleton)
│   ├── meta/info.json
│   ├── meta/stats.json                          # only written by real lerobot
│   ├── meta/tasks.parquet                       # only written by real lerobot
│   └── meta/episodes/chunk-000/file-000.parquet # real lerobot v3 layout (dir + file-000)
├── lerobot_v3/ep_000{1,2,3}.summary.npz         # subprocess __RESULT__ summary
├── viz/ep_000{1,2,3}/camera.mp4                 # original field of view + 2D keypoint overlay
├── viz/ep_000{1,2,3}/world.mp4                  # 3D world view
├── qc/ep_000{1,2,3}/report.html                 # includes base64 curves + depth-source table
├── qc/ep_000{1,2,3}/reprojection.png            # raw curve image (12,863 B)
├── qc/ep_000{1,2,3}/segments.json               # segment labels (no matching location in real dataset, archived)
└── manifest.json
```

Note the directory layout is the shape of **real lerobot 0.6.1** (`meta/stats.json`, `meta/tasks.parquet`,
`meta/episodes/chunk-000/file-000.parquet`), clearly different from the **fake skeleton** written by
`tests/fakes/modelenv_scripts/fake_lerobot_export.py` (the stand-in used by the local fake mode
`configs/local-fake.yaml`): the fake script writes `meta/tasks.jsonl` (real lerobot does **not** write
`tasks.jsonl`), `meta/episodes/chunk-000.parquet` (real lerobot is the **dir+file** form
`meta/episodes/chunk-000/file-000.parquet`), and **lacks** `meta/stats.json` and `meta/tasks.parquet` — the two can be
distinguished this way, ruling out a fake script impersonating a real export.

## LeRobot Read-back Self-Proof (hard evidence of format validity, based on **smoke-1** artifacts `/tmp/plan-c-smoke-out/`)

Read the just-exported episode back from disk with the **real public lerobot API** (`repo_id` is a required param in
0.6.1, `root` takes precedence):

```bash
P=~/miniforge3/envs/default/envs/ego-lerobot/bin/python
"$P" - <<'PY'
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset(repo_id="roboray_episode", root="/tmp/plan-c-smoke-out/lerobot_v3/ep_0001")
print("frames:", len(ds), "fps:", ds.fps, "features:", sorted(ds.features))
item = ds[0]
print("sample keys:", sorted(item.keys()))
print("observation.state shape:", tuple(item["observation.state"].shape))
print("action shape:", tuple(item["action"].shape))
PY
```

Output (key lines, verbatim):

```
lerobot 0.6.1
frames: 30 fps: 30 features: ['action', 'episode_index', 'frame_index', 'index', 'observation.state', 'task_index', 'timestamp']
sample keys: ['action', 'episode_index', 'frame_index', 'index', 'observation.state', 'task', 'task_index', 'timestamp']
observation.state shape: (126,)
action shape: (126,)
```

`len(ds)=30` matches the synthetic episode's frame count, `fps=30`, and `observation.state/action` are both `(126,)`
flattened joints (2 hands x 21 joints x 3). **This is hard evidence of format validity**: lerobot's official reader
loads and fetches frames without error.

> The read-back prints a `libtorchcodec ... falling back to 'pyav'` warning — that is lerobot probing its video
> decoding backend, unrelated to this dataset (export used `use_videos=False`; training tensors are numeric features);
> ignore it.

## Visualization Frames (dual view, actually playable; based on **smoke-1** artifacts)

Six mp4s read with `cv2.VideoCapture` (measured):

| File | frames | fps | resolution | first frame readable |
| --- | --- | --- | --- | --- |
| viz/ep_0001/camera.mp4 | 30 | 30.00 | 96x72 | True |
| viz/ep_0001/world.mp4 | 30 | 30.00 | 400x300 | True |
| viz/ep_0002/camera.mp4 | 30 | 30.00 | 96x72 | True |
| viz/ep_0002/world.mp4 | 30 | 30.00 | 400x300 | True |
| viz/ep_0003/camera.mp4 | 30 | 30.00 | 96x72 | True |
| viz/ep_0003/world.mp4 | 30 | 30.00 | 400x300 | True |

The camera view keeps the **original field of view / aspect ratio** (96x72 = the synthetic frame's native size); the
world view is a 3D-rendered canvas.

> **[Honest declaration - deferred item] spec §10 requires the camera view to overlay two elements — "2D keypoints"
> and "MANO reprojection". This branch delivered only the 2D keypoint overlay; the MANO reprojection overlay is not
> implemented and is registered as a **Plan D item** (honest deferral, not missing data). The required data already
> exists: metric 3D hand keypoints and reprojection error in `fused.npz`, camera intrinsics/extrinsics in `scene.npz`
> — this is **rendering work, not a data gap**: same source as the QC curves, it only needs to project the world-frame
> keypoints back to pixels via the intrinsics/extrinsics and overlay them on the camera view.

Geometry conventions are pinned by unit tests (`tests/unit/test_viz.py`):

- `test_world_view_camera_trail_uses_c2w_not_w2c`: camera trail uses **c2w** (artifacts store w2c, must invert) —
  writing the orientation backwards yields a mirrored trail; this test locks it.
- `test_world_view_pointcloud_is_backprojected_to_world`: point cloud **truly backprojected to world frame**
  (`K⁻¹·p·depth` → invert w2c → world), not emitted directly in camera frame.
- `test_world_view_draws_future_wrist_trail`: world view includes the **future 1s wrist trail**.
- `test_world_view_without_intrinsics_skips_only_pointcloud`: missing intrinsics drops **only the point cloud**, not
  the whole view.
- `test_world_view_all_nan_is_reported_not_faked` / `..._removes_stale_output`: all-NaN reports the reason, produces
  **no misleading video**, and cleans up stale artifacts.
- `test_camera_view_overlays_and_keeps_aspect` / `test_camera_view_no_frames_removes_stale_output`:
  2D keypoint overlay preserves aspect ratio; no frames → stale output removed.

## QC Report (all numbers from `fused.npz`, no fake plots; based on **smoke-1** artifacts)

All three `qc/ep_000*/report.html` measured to contain:

- **base64-embedded reprojection-error curve** (`data:image/png;base64` appears once; accompanied by
  `qc/ep_000*/reprojection.png`, 12,863 B).
- **Depth-source table**: `stereo → 30 frames` (from `fused.npz["depth_source"]`).
- Reprojection error: **mean 1.4567 / max 2.1504 px** (only trusted hand frames counted; `1e6`-flagged frames removed).
- The title renders correctly; **U+FFFD (tofu) count = 0** (Task 5 confirmed CJK font parsing / ASCII fallback).

Number-source discipline is pinned by unit tests (`tests/unit/test_qc_report.py`, `test_viz.py`,
`test_fusion_refined.py`):

- `test_qc_report_contains_curve_and_depth_source`: curve and depth-source table are indeed in the report.
- `test_curve_plot_all_flagged_reports_instead_of_faking`: all-flagged reports the reason, draws no fake plot.
- `test_curve_rendering_has_no_missing_glyph_warning`: curve rendering has no missing-glyph warning (no tofu).
- `test_publish_degraded_qc_report_has_no_fabricated_stats`: the degraded report (missing `fused.npz`) does **not**
  print "mean 0.0000 / 0 samples"-style values fabricated from missing data.
- `test_publish_qc_stats_exclude_flagged_hands`: `1e6`-flagged frames excluded from mean/max.
- `test_publish_qc_report_renders_gapfill_audit`: completion audit (rejected-reason distribution / skipped counts) rendered.
- `test_publish_qc_stats_exclude_flagged_hands` (in `tests/unit/test_fusion_refined.py`);
  the full `tests/unit/test_qc_report.py` suite of 4 cases passes.

## Real VLM (client injection + error branches + keyframes in the request)

The VLM's **real client path** and **error branches** are covered by unit tests (`tests/unit/test_semantic_models.py`,
7 cases), all passing:

- `test_label_writes_parsed_labels`: injects a fake `client_factory`, parses the reply JSON → labels.
- `test_label_pads_and_truncates_to_segment_count`: non-conforming output count is **padded/truncated to segment count** (not fatal).
- `test_missing_base_url_raises_naming_the_config_key`: missing base_url → `VlmError`, message names
  `vlm_base_url` / `ROBORAY_VLM_BASE_URL`.
- `test_missing_api_key_raises_naming_the_env_var`: missing key → `VlmError`, message names `VLM_API_KEY`.
- `test_non_json_reply_is_not_fatal`: non-JSON reply does **not raise**, falls back to generic names.
- `test_label_sends_keyframes_as_multimodal_content` (spec §9): keyframes **actually enter the request** —
  `messages[0].content` is a multimodal list `[{"type":"text"}, {"type":"image_url"}...]`, with image_url as
  `data:image/jpeg;base64,...`, in the same order as the passed-in images.
- `test_label_without_images_stays_text_only`: no keyframes → degrades to text-only.

On the `run_action` side (`tests/unit/test_fusion_action.py`):
`test_run_action_passes_encoded_keyframes_to_semantic_model` (encoded keyframes passed to the semantic model),
`test_run_action_tolerates_missing_prepared_frames` (missing frames don't abort),
`test_select_keyframe_indices_keeps_first_last_and_caps` (first/last frames kept + capped sampling).

### [L2] End-to-end call to a real VLM service **not verified on this machine** (honest record)

This machine has **no VLM service configured**, and the environment has `all_proxy=socks://<internal-proxy>:80`.
Measured with the **real `openai` client** (`client_factory=None`):

```
VlmError (correctly wrapped): VLM call failed (base_url=https://vlm.example/v1, model=qwen-vl):
Unknown scheme for proxy URL URL('socks://<internal-proxy>:80')
```

i.e., the `socks://` proxy makes the real `openai` client fail at construction/request time, but that failure is
**correctly wrapped as `VlmError`** (unreachable → error, no geometric degradation — decision D6). **"Connecting to a
real VLM service and getting real labels" is therefore [L2]**; this machine does not fake success. End-to-end against a
real service should be verified in Plan D / an environment that can reach a VLM directly (if needed, clear
`NO_PROXY`/`all_proxy`, or make `openai` use an HTTP proxy).

## Task 1: FFS depth grid (semantic fix of Plan B's resolution knob)

Plan B's `stereo_max_image_side_px` knob scales the stereo pair before sending it to FFS, and **the original
implementation returned the scaled depth grid**, inconsistent with the downstream assumption "depth grid = original-frame
grid". Task 1 fixes it: `run_ffs.py` **resamples the depth back to the original resolution** after obtaining the
scaled grid's disparity/depth (`cv2.resize(d, (orig_w, orig_h), INTER_LINEAR)`), and the depth is **numerically
invariant** under scaling (disparity scales proportionally with resolution). Pinned by unit tests:

- `test_ffs_depth_returns_to_original_grid_after_resize`: depth returns to the **original grid** after resizing.
- `test_fake_ffs_reports_both_grids`: the stand-in script truthfully reports both grids.
- `test_ffs_adapter_resize_keeps_depth_invariant` / `test_fake_ffs_resize_scales_disparity_depth_invariant`:
  depth **unchanged** before/after scaling (numerical invariant).
- `test_stages_io.py::test_gpu_payload_for_stereo_carries_frames_right_and_resize_knob`:
  stereo GPU payload carries `frames_right` and the resize knob.

## L2 To-verify Checklist

| Item | Current status | Verification owner |
| --- | --- | --- |
| **Real VLM service end-to-end** (connectivity + real label quality + images actually read by the model) | **Unreachable on this machine** (no service + socks proxy), failure wrapped as `VlmError` | Plan D (env that can reach VLM directly) |
| lerobot `create()` parameter names and `root` semantics (`[L2]` annotation in script header) | **Verified working** locally on 0.6.1 (wrote + read back); annotation kept for first real-machine recheck | Plan D first-run recheck |
| Real-model inference (numerical correctness of ViPE / FFS / HaMeR+MANO) | Not run (no GPU/weights locally) | Plan D (A800 GPU smoke) |

## Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| **lerobot version API drift** (`LeRobotDataset.create` param names / `root` semantics / directory layout) | Version locked at **0.6.1** (fixed by `build_lerobot.sh`); export runs in a **separate env subprocess**, the API is exposed only in `scripts/modelenvs/run_lerobot_export.py`, so replacing that script does not affect the rest of publish; locally proven feasible with real export + read-back on 0.6.1. The `[L2]` annotation stays in the script header for first real-machine recheck. |
| **`ROBORAY_LEROBOT_PYTHON` was not delivered via `runtime_env`** (**fixed**) | Previously relied on process-env inheritance locally (a local coincidence; multi-node would degrade to `"python"` → silent export failure, same class as Plan B's dead knob). Fixed: `ModelEnvsConfig` gains `lerobot_python`/`lerobot_script`; `resolve_env_request` flattens them and delivers via `runtime_env.env_vars`; the explicit `runtime_env` value overrides inherited values, so the config `models:` section is authoritative. On multi-node, the "config delivery → worker real export" chain itself is still **[L2]**, re-verified on Plan D's KubeRay. |
| **VLM service availability / proxy** | Unreachable → `VlmError` (no degradation); missing any of `ROBORAY_VLM_BASE_URL` / `ROBORAY_VLM_MODEL` / `<vlm_api_key_env>` names it in the error. The `socks://` proxy breaks `openai` — at deploy time ensure the VLM endpoint is directly reachable (or configure an HTTP proxy). |
| **Visualization performance** (matplotlib Agg per-frame 3D + point cloud) | Per-frame rendering; long episodes (thousands of frames) grow linearly with frame count; world view fixed at 400x300. If real long sequences need speed, evaluate downsampling/skip-frame in Plan D. |
| **Export-env path nested by conda** | The `$(conda info --base)/envs/ego-lerobot/bin/python` echoed by the build script **does not exist** on this machine; `ROBORAY_LEROBOT_PYTHON` must point at the measured nested path. Documented explicitly (this report + `docs/models.md`). |

## Honest Boundary

**What this machine (no GPU, no VLM service) verified**: orchestration, real LeRobot write-out + **read-back**,
visualization frame production and geometry conventions, QC number sources and the "no data → no fake plot"
discipline, VLM client injection and all error branches (including keyframes in the request).
**Not verified on this machine**: reaching a real VLM service for real labels (no service + socks proxy, [L2]),
numerical correctness of real-model inference (Plan D). All numbers come from actual runs; items not run through are
honestly marked [L2], never passed off as fake success.

## Fix record: `ROBORAY_LEROBOT_PYTHON` delivery (closing a same-class defect)

**Defect**: `LeRobotWriter` reads `ROBORAY_LEROBOT_PYTHON` inside the Ray worker, but the variable was **not** in
`runtime_env.env_vars`; under local `ray.init` it relied on driver-process environment inheritance to the worker (the
first smoke succeeded), but on a multi-node/remote cluster it would degrade to `"python"` → main env without `lerobot`
→ **per-episode silent failure** — the same class of "looks wired but is not" defect as Plan B's dead
`stereo_max_image_side_px` knob.

**Fix** (no change to export/write logic; only the delivery wiring):

1. `src/roboray/api/config.py`: `ModelEnvsConfig` adds `lerobot_python: Optional[str] = None`,
   `lerobot_script: Optional[str] = None` (the latter is only a test-injection point; empty/None = real script).
2. `src/roboray/models/modelenv.py`: `resolve_env_request` adds
   `ROBORAY_LEROBOT_PYTHON` (`lerobot_python or "python"`) and `ROBORAY_LEROBOT_SCRIPT`
   (`lerobot_script or ""`).
3. **Wiring confirmed**: `env_vars.update(resolve_env_request(self.request.models))` in `api/reconstruction.py`
   applies to both backends, `test` (merged into `worker_process_setup_hook`'s `runtime_env`) and `real`
   (merged into `{"env_vars": env_vars}`) — **no other change needed** (confirmed by reading the code).
4. Tests: `test_modelenv.py::test_resolve_env_request_fills_defaults` asserts both keys' defaults;
   new `test_resolve_env_request_passes_lerobot_values` asserts pass-through;
   `test_config.py::test_models_env_config_roundtrip` asserts `lerobot_python`/`lerobot_script` round-trip through
   `from_dict`/`to_dict`.

**Verification**: focused `test_modelenv.py` + `test_config.py` → **15 passed**; full suite
(`-o addopts="" -m "not gpu"`) → **206 passed, 3 deselected** (one more than the pre-fix 205, the new
`test_resolve_env_request_passes_lerobot_values`). The second end-to-end smoke (config `models.lerobot_python`-driven,
no shell export) → `failed []`, real lerobot v3 layout. Measured precedence: the explicit `runtime_env` value
**overrides** the inherited value, so config is authoritative. **On multi-node, the "config delivery → worker reads it
→ real export" chain itself is still [L2]** and must be re-verified on Plan D's KubeRay.
