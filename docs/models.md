# Real-Model Integration Guide (Plan B / L2)

This document is for operators wiring `roboray` to real models on an **A800 (L2)** node: how to build a conda env for each of the three models, how to obtain weights (including MANO, which requires registration), how the `ROBORAY_*` environment variables map, the **L2 verification points** for each adapter and inference script, and how to run the GPU smoke tests.

## Architecture Overview

The main env (`roboray`) does **not** install torch. Each real model runs in its own conda env, invoked as a subprocess by `src/roboray/models/{scene_models,hand_models}.py` through `run_in_modelenv` (`src/roboray/models/modelenv.py`), which calls `scripts/modelenvs/run_*.py`. Data is exchanged as NPZ files; the last line of stdout returns a summary via `__RESULT__{json}`. Adapters only do orchestration and contract normalization.

| Lane | Model | Inference script | Output |
| --- | --- | --- | --- |
| Scene (monocular) | ViPE | `scripts/modelenvs/run_vipe.py` | Pose (c2w inverted to w2c) + near-metric depth |
| Scene (stereo) | Fast-FoundationStereo + ViPE | `scripts/modelenvs/run_ffs.py` + `run_vipe.py` | Disparity → metric depth (`depth = fx·B/d`) + ViPE pose |
| Hand | HaMeR + MANO | `scripts/modelenvs/run_hamer.py` | Up to 2 hands per frame: MANO params + 3D/2D joints + confidence |

## Build Order

1. **Vendored repos** (do this first; the build scripts depend on `third_party/`):

   ```bash
   bash scripts/vendor_models.sh    # clone and pin commits (idempotent; safe to re-run)
   ```

   The script clones `vipe` / `fast-foundationstereo` / `hamer` into `third_party/`. On first run it records each HEAD as `third_party/<repo>.commit`; on every subsequent run it checks out that commit (version pinning). Repo contents are ignored by `.gitignore`; only the `<repo>.commit` lock files are committed.

2. **Scene envs** (independent of each other; either order, can be parallel):

   ```bash
   bash deploy/modelenvs/build_vipe.sh    # → venv (**must run inside a container**, see below)
   bash deploy/modelenvs/build_ffs.sh     # → conda env `ego-ffs` (python 3.12 + torch 2.6 cu124 + xformers)
   ```

   > **ViPE differs from the other two envs: it does not build a conda env on the host.** It requires Python ≥ 3.11, and `csrc/` holds DROID-SLAM's CUDA operators that **must be compiled**, which in turn needs nvcc. See the recipe at the top of `deploy/modelenvs/build_vipe.sh` (four pitfalls hit in practice: PyPI's `nvidia-cuda-nvcc-cu12` ships only `ptxas` without `nvcc`, `nvidia-cuda-cccl-cu12` has no `nv/`, the container seccomp blocks `pthread_create`, and the internal mirror's hydra-core/omegaconf are from 2019). **Already verified end-to-end on A800.**

3. **Hand env** (depends on the `third-party/ViTPose` submodule of `third_party/hamer`, so it must come after `vendor_models.sh`):

   ```bash
   bash deploy/modelenvs/build_hamer.sh   # → venv (**must run inside a container**, see below)
   ```

   > **HaMeR likewise uses a venv rather than a host conda env**: `hamer.utils` pulls in pyrender/OpenGL at import time, and a missing `libEGL` causes an immediate ImportError, so the base image must install `libegl1` (`deploy/Dockerfile.hamer`). The recipe and five more pitfalls are in the header of `build_hamer.sh`. **Already verified end-to-end on A800.**

Each build script echoes the `ROBORAY_*_PYTHON` to set at the end, i.e. `$(conda info --base)/envs/<env>/bin/python`.

## MANO Registered Download

`MANO_RIGHT.pkl` is not open-source; you must register to download it:

1. Register and sign in at <https://mano.is.tue.mpg.de>.
2. Download **MANO_RIGHT.pkl** (right-hand model).
3. Place it in the `mano/` subdirectory of the weights dir, i.e. `$ROBORAY_HAMER_WEIGHTS/mano/MANO_RIGHT.pkl`.

```bash
export ROBORAY_HAMER_WEIGHTS=/data/weights/hamer
mkdir -p "$ROBORAY_HAMER_WEIGHTS/mano"
cp /path/to/MANO_RIGHT.pkl "$ROBORAY_HAMER_WEIGHTS/mano/"
```

`HamerHandModel.load()` **hard-checks** that `mano/MANO_RIGHT.pkl` exists (if missing it raises `ModelEnvError` with the download instructions from this doc) — fail fast rather than ship a half-usable state.

## HF Weight Downloads

Fast-FoundationStereo weights (including `model_best_bp2_serialize.pth`) are downloaded from HuggingFace `nvidia/c-fast-foundationstereo`:

```bash
huggingface-cli download nvidia/c-fast-foundationstereo --local-dir /data/weights/ffs
export ROBORAY_FFS_WEIGHTS=/data/weights/ffs
```

ViPE's weights are auto-downloaded to the cache dir on first inference per the official flow; to set a specific directory, set `ROBORAY_VIPE_WEIGHTS`. **Cluster nodes have no external network**: fetch the weights on a machine that does, then place them under `$TORCH_HOME/hub/` per the layout in `deploy/assets/weights/vipe/README.md` (4 items: GeoCalib / MoGe-2 / DROID-SLAM / SuperPoint), and set `HF_HUB_OFFLINE=1`. HaMeR's detection/regression weights come from the official `hamer_demo_data.tar.gz` (the **full archive is 6.04GB**, of which 3.8GB is ViTPose weights this repo does not use); place them per the layout in `deploy/assets/weights/hamer/README.md` under `ROBORAY_HAMER_WEIGHTS` (must contain `_DATA/` and `mano/`), and put MANO under `mano/`.

## `ROBORAY_*` Variable Reference

The six variables are flattened by `resolve_env_request(models: ModelEnvsConfig)` and injected via the runtime `env_vars`; an empty string means "unset". When `*_PYTHON` is unset it defaults to `"python"` (usable only with stub tests; real inference fails for lack of torch).

| Env var | Config key (`models:` section) | Meaning | When unset |
| --- | --- | --- | --- |
| `ROBORAY_VIPE_PYTHON` | `models.vipe_python` | Interpreter path of the `ego-vipe` env | `"python"` |
| `ROBORAY_FFS_PYTHON` | `models.ffs_python` | Interpreter path of the `ego-ffs` env | `"python"` |
| `ROBORAY_HAMER_PYTHON` | `models.hammer_python` | Interpreter path of the `ego-hamer` env | `"python"` |
| `ROBORAY_VIPE_WEIGHTS` | `models.vipe_weights` | ViPE weights/cache dir | empty string |
| `ROBORAY_FFS_WEIGHTS` | `models.ffs_weights` | FFS weights dir (contains `.pth`) | empty string |
| `ROBORAY_HAMER_WEIGHTS` | `models.hammer_weights` | HaMeR weights dir (contains `mano/MANO_RIGHT.pkl`) | empty string |

> Spelling note: the env var prefix is `HAMER` (`ROBORAY_HAMER_PYTHON` / `ROBORAY_HAMER_WEIGHTS`), while the config/yaml keys are `hammer_*` (`models.hammer_python` / `models.hammer_weights`) — the two spellings differ; do not mix them up.

Example `configs/*.yaml`:

```yaml
models:
  vipe_python: /opt/conda/envs/ego-vipe/bin/python
  ffs_python: /opt/conda/envs/ego-ffs/bin/python
  hammer_python: /opt/conda/envs/ego-hamer/bin/python
  vipe_weights: /data/weights/vipe
  ffs_weights: /data/weights/ffs
  hammer_weights: /data/weights/hamer
  lerobot_python: /opt/conda/envs/ego-lerobot/bin/python   # export env; must be set in config on multi-node
  # lerobot_script: ...                                     # test-only injection point; do not set in production
```

> There is also a switch used **only for GPU smoke tests**, `ROBORAY_GPU_SMOKE` (see below), which is not one of the six variables above.

## Resident Worker Mode

How real models execute is controlled by `ROBORAY_MODEL_WORKER` (config key `execution.model_worker`), propagated to Ray worker processes via `runtime_env` and read by `GpuModelActor` (default `"resident"`):

| Value | Behavior |
| --- | --- |
| `resident` (default) | The adapter holds a resident `ModelEnvSession` — `run_vipe.py` / `run_ffs.py` / `run_hamer.py` run with `--serve` in **long-lived subprocesses**; model weights are built once and **reused across Episodes** |
| `oneshot` | Spawn a one-shot subprocess per inference (the pre-change path without `--serve`, **byte-for-byte equivalent to before the change**), for local debugging comparison |

**Protocol**: serve mode uses JSON-lines over stdin/stdout, reusing the existing `__RESULT__{json}` marker and matching responses by request `id` (the protocol core is `run_serve_loop` in `scripts/modelenvs/_common.py`). Command sequence: `hello` (handshake; subprocess ready) → `infer` (`input`/`output` are NPZ paths, `weights`/`extra` participate in the cache key) → `ping` → `shutdown`. Key points:

- Models are built lazily keyed by **`(weights, extra)`** and built **only once** (rebuilt only when the key changes) — this is the core benefit of resident mode.
- **Exception (FFS stereo lane)**: `FastFoundationStereoModel`'s `extra` contains **per-Episode calibration** (`baseline_m` / `fx` / `stereo_max_image_side_px`), so Episodes with different calibration each rebuild the FFS model — "built only once" does not hold for FFS. ViPE's `extra` is always `{}` and HaMeR's `extra` is always `{checkpoint}`, so they are unaffected. Separating build params from infer params (build-extra vs. infer-extra split) is registered as **E3**.
- If infer throws (including `ValueError`), only an `{"ok":false,"error":...}` error line is returned; **the process is not killed** — one bad input should not discard already-loaded weights.
- If the subprocess crashes (OOM/segfault), the **next** infer automatically restarts and reloads (the request that crashed at the moment of the crash is not silently retried).

**Three-level idle** (thresholds **must be increasing**, otherwise the node never sees the actor exit):

| Level | Threshold | Config key / location | Action |
| --- | --- | --- | --- |
| Unload model | 30s | `execution.actor_idle_unload_s` | Actor stays alive, GPU memory returned (`unload()` closes the session) |
| Actor exit | 240s | `execution.actor_idle_exit_s` | `unload()` first, then `ray.kill` (`ray.kill` is SIGKILL and does no cleanup, so unload must come first) |
| Node reclamation | 300s | RayCluster `idleTimeoutSeconds` (**part of the E3 deployment artifacts**, not in this repo's code) | Idle node returned to the cluster |

**Stereo-lane cascade**: `FastFoundationStereoModel.open_session()` also opens the session of the embedded `vipe_model` at the same time — both models of the stereo lane (FFS for depth, ViPE for pose) are resident. ViPE weights are built once and reused across Episodes; **FFS weights are not guaranteed to load only once**: its build cache key includes per-Episode calibration (`baseline_m` / `fx` / resolution cap, see "Exception" above), so it is rebuilt when calibration changes.

## Export (LeRobot) and VLM Environment Variables

Beyond the six model env variables above, the **export** and the **action-understanding VLM** each have their own set of `ROBORAY_*` variables:

| Env var | Config key (`models:` section) | Meaning | When unset |
| --- | --- | --- | --- |
| `ROBORAY_LEROBOT_PYTHON` | `models.lerobot_python` | **Absolute path** to the `ego-lerobot` env interpreter (`LeRobotWriter` reads it inside the worker; **it is a path, not an env name**) | `"python"` (the main env has no `lerobot`, so export always fails) |
| `ROBORAY_LEROBOT_SCRIPT` | `models.lerobot_script` | **Test-only injection point**: overrides the export script path. Production uses the real in-repo script `scripts/modelenvs/run_lerobot_export.py`; e2e uses it to swap in a fake script (`publish` cannot be swapped via the registry, hence this injection point). **Do not set in production** | The real in-repo script |
| `ROBORAY_VLM_BASE_URL` | `action_understanding.vlm_base_url` | Base URL of the OpenAI-compatible VLM service | empty (a call raises `VlmError`) |
| `ROBORAY_VLM_MODEL` | `action_understanding.vlm_model` | VLM model name | `"qwen-vl"` |
| `VLM_API_KEY` (name set by config `action_understanding.vlm_api_key_env`) | — | VLM auth key | empty (a call raises `VlmError`) |

The export env (`ego-lerobot`) is built by `deploy/modelenvs/build_lerobot.sh` (Python 3.12 + CPU torch + `lerobot==0.6.1` — lerobot ≥ 0.5 declares `Requires-Python>=3.12` and hard-depends on torch ≥ 2.7 and numpy ≥ 2.0, which conflicts with the main env's numpy<2.0, so it **must be a separate subprocess env**, same pattern as the model adapters).

> **Path pitfall (tested in practice)**: the build script echoes `$(conda info --base)/envs/ego-lerobot/bin/python`, but if the env was created inside another conda env, conda installs it **nested** (here `.../envs/default/envs/ego-lerobot/bin/python`), so the `$(conda info --base)` path above **does not exist**. Always set `ROBORAY_LEROBOT_PYTHON` to the **actual absolute path** reported by `conda env list` / the script output.
>
> **Propagation (must go through config)**: `ROBORAY_LEROBOT_PYTHON` / `ROBORAY_LEROBOT_SCRIPT` are flattened by `resolve_env_request(models)` and go into `runtime_env.env_vars` along with the six model env variables (`api/reconstruction.py`), where Ray sets them in the worker process. **On multi-node / remote clusters be sure to set `lerobot_python` in the `models:` section of the config (or place it directly into `runtime_env`)** — under local `ray.init` the driver process env can be inherited raylet→worker, and `export ROBORAY_LEROBOT_PYTHON=...` in the shell also happens to work, but that is a **local coincidence, not a guarantee**: explicit values in `runtime_env.env_vars` **override** inherited values (tested in practice), and multi-node clusters do **not** inherit the driver env. If neither is set, the worker side silently falls back to `"python"` (the main env has no `lerobot`) → every export silently fails. Hence **config is the authoritative source**.

## Artifact Contract

Artifacts written to disk by adapters (via `roboray.stages.artifacts.save_artifact`, including `__schema_version__`):

**Scene artifacts** (`VipeSceneModel` / `FastFoundationStereoModel`) required keys:

| Key | Shape / type | Description |
| --- | --- | --- |
| `camera_poses` | `(T,4,4) float64` | **world-to-camera** (the script inverts c2w before output) |
| `intrinsics` | `(T,3,3) float64` | Per-frame intrinsics |
| `depth` | `(T,H,W) float32` | Metric / near-metric depth (meters) |
| `depth_source` | scalar string | `"monocular"` (ViPE) or `"stereo"` (FFS) |
| `n_frames` | `int32` | Frame count |
| `fps` | `int32` | Frame rate |

**Hand artifacts** (`HamerHandModel`) required keys:

| Key | Shape / type | Description |
| --- | --- | --- |
| `hand_present` | `(T,2) bool` | Slot 0=left, 1=right (HaMeR's left hand = fitted flipped right hand) |
| `joints_2d` | `(T,2,21,3) float32` | Image joints (includes confidence dimension) |
| `joints_3d_cam` | `(T,2,21,3) float32` | **Camera-frame** MANO forward-kinematics joints (joint 0 = wrist = `mano_trans`), in meters; A1's mano path uses it to measure "MANO structure vs. image evidence" consistency |
| `mano_pose` | `(T,2,48) float32` | `global_orient(3) + hand_pose(45, PCA)` |
| `mano_shape` | `(T,2,10) float32` | MANO betas |
| `mano_trans` | `(T,2,3) float32` | Hand root translation (camera frame, meters) |
| `confidence` | `(T,2) float32` | Per-side detection confidence |
| `n_frames` / `fps` | `int32` | Frame count / frame rate |

**Fused artifacts** (A1/A3, `src/roboray/stages/fusion.py`) persisted keys: `joints_world`, `hand_ids`, `reprojection_error`, `joint_source` (`"mano"` is informative / `"joints_2d"` only closed-loop consistency), `depth_source`, `identity_switches`, `gapfill_filled`, `gapfill_rejected`, `n_frames`, `fps`. Among these, `gapfill_skipped` and `gapfill_reasons` are returned only in the **run summary** and are not written to artifacts.

## L2 Verification Checklist

The `[L2]` items below come from the headers and inline annotations of the three inference scripts: check each one on first A800 integration and, where necessary, replace the corresponding sections in the scripts (`scripts/modelenvs/run_*.py`).

### ViPE (`run_vipe.py`) — ✅ verified on A800

The `[L2]` assumptions written from the official docs were **all wrong**; rewritten from actual measurements. Verification results:

- ✅ **Import paths** — not `vipe.pipeline.DefaultAnnotationPipeline` / `vipe.stream.VideoStream`, but `vipe.config.validate_typed_config` + `vipe.streams.base.StreamList.make` + `vipe.pipeline.make_pipeline`.
- ✅ **hydra config nesting** — the pipeline group lives under **`cfg.pipeline`** (not flattened to the root); overrides must be written as `pipeline.output.path=...`; `pipeline.init.instance` is not yet in the struct at compose time, so you can only `OmegaConf.update(cfg, "pipeline.init.instance", None)` after compose.
- ✅ **Return structure** — with `pipeline.return_payload = True`, `run(stream).payload` is the `SLAMOutput`.
- ✅ **SE3 → 4×4** — `SLAMOutput.get_view_trajectory(0).matrix()` is **(N,4,4) c2w** (per `scripts/vipe_to_colmap.py`, which inverts it to w2c).
- ✅ **Field names** — `SLAMOutput.intrinsics[0]` is `[fx, fy, cx, cy]`.
- ✅ **Depth** — not in `SLAMOutput`; you must back-project from the sparse SLAM point cloud with
  `slam.slam_map.project_map(frame_tstamp, view_idx, target_size, target_intrinsics, target_pose, target_camera_type, infill=True, tstamp_nn=3)`.
  **Note: this path requires textured input** — measured on a synthetic random-noise video, DROID-SLAM finds no trackable structure (the whole segment's keyframe cloud has only ~118 points per frame, and most frames project to 0 inliers), and `project_map`'s CUDA kd-tree raises `cudaErrorInvalidValue`. In that case `run_vipe.py` **fails loudly and explains why**.

L2 measured numbers (`dog-example.mp4`, 40 frames / 13.4 s / A800): intrinsics fx=fy≈984.26, cx=640, cy=360 (exactly the 1280×720 principal point), depth 2.011–5.439 m all finite, 10 keyframes, c2w[0]≈I.

### Fast-FoundationStereo (`run_ffs.py`)

Checked against the reference repo's `scripts/run_demo.py` internals:

- [ ] **Weight artifact name** — after HF `nvidia/c-fast-foundationstereo` download, contains `model_best_bp2_serialize.pth`.
- [ ] **Import/load** — the real import path (e.g. `core.found_stereo.FoundStereo`) and the loading signature `FoundStereo(args=(...))` (`restore_ckpt` / `valid_iters` / `max_disp` / `hidden_dims`).
- [ ] **Forward API name** — the stereo forward call name (`forward_image(l, r)` or equivalent).
- [ ] **Rectification precondition** — input must be rectified horizontal-epipolar stereo; this pipeline's layout convention is rectified, with baseline taken from `calib["T_left_to_right"][0][3]` (right-cam +x convention). The depth conversion `depth = fx·B/d` is a deterministic step (not L2).

### HaMeR (`run_hamer.py`) — ✅ verified on A800

All 6 original `[L2]` assumptions were **wrong** (`HAMER.load_pretrained`, `model.detect_and_fit`, etc. do not exist in HaMeR at all); rewritten from actual measurements. Verification results:

- ✅ **Import/construction** — `from hamer.models import load_hamer` → `model, model_cfg = load_hamer("<weights>/_DATA/hamer_ckpts/checkpoints/hamer.ckpt")`. The config is derived from `Path(ckpt).parent.parent/'model_config.yaml'`.
- ✅ **Batch entry** — `ViTDetDataset(model_cfg, img_bgr, boxes, right, rescale_factor=2.0)` + `DataLoader` → `out = model(batch)`. `out` keys: `pred_cam (B,3)`, `pred_cam_t (B,3)`, `pred_mano_params`, `pred_keypoints_3d (B,21,3)`, `pred_keypoints_2d (B,21,2)`, `pred_vertices`.
- ✅ **MANO params and frames** — `pred_mano_params = {global_orient (B,1,3,3), hand_pose (B,15,3,3), betas (B,10)}`, **all rotation matrices and no transl** (`hamer/models/heads/mano_head.py:108`). `hamer/models/hamer.py:120` forwards with `pose2rot=False`, so `pred_keypoints_3d` is **root-relative**.
- ✅ **Full-image camera translation** — `hamer.utils.renderer.cam_crop_to_full(pred_cam, box_center, box_size, img_size, focal_length)` (call `pred_cam[:,1] *= (2*right-1)` beforehand). **Camera-frame joints = pred_keypoints_3d + cam_t_full[:,None,:]**.
- ✅ **Scale** — in `cam_crop_to_full`, `tz = 2*f/bs`, proportional to f. The official demo uses `FOCAL_LENGTH/IMAGE_SIZE*img_size.max()` (a prior focal length, ~25000px) → **relative scale**; this adapter uses the **measured fx** when calib provides K → **metric**. Measured on the same image: without K, `mano_trans` median norm **17.6** (relative); with K=0.9W, **0.82 m** (physically plausible).
- ✅ **MANO layer** — uses `model.mano` directly (an `smplx.MANOLayer` subclass that HAMER builds from `cfg.MANO`), **full pose, not PCA** (`posedirs (778,3,135)` = 45×3 rotation dims), so the axis-angle representation of `mano_pose (48,) = global_orient(3) + hand_pose(45)` is well-defined.
- ✅ **CWD constraint** — `CACHE_DIR_HAMER = "./_DATA"` in `hamer/configs/__init__.py` is a **CWD-relative** module constant → the running process's CWD must be inside the weights dir; the adapter does `os.chdir` itself.

**Detection-box source (important, stated honestly)**: the official demo uses detectron2 (ViTDet) + mmpose (ViTPose) to get hand keypoints then infer boxes. In this environment **neither can be installed** (`mmcv==1.3.9` is incompatible with torch 2.x; ViTDet weights also need another 2.4GB), so the adapter takes boxes in the order `--extra {"hand_boxes": ...}` → **mediapipe hand detection** (the fallback) → error. **HaMeR's MANO fitting part was not modified at all.**

L2 measured numbers (HaMeR's bundled `example_data/test1.jpg`, 4 frames / A800): detection rate **1.0**, 2 hands per frame, `joints_3d_cam (4,2,21,3)` and `mano_pose (4,2,48)` all finite, `mano_trans` median norm **0.82 m** (after giving K), confidence taken from the detector's real score (0.988) rather than a placeholder 1.0.

## Running the GPU Smoke Test on A800

The smoke test lives in `tests/gpu/test_adapters_gpu.py` with double gating: it runs only when `@pytest.mark.gpu` **and** the env var `ROBORAY_GPU_SMOKE` is non-empty (otherwise skipped). The default `pytest` run will **deselect** it because of `addopts = "-q -m \"not gpu\""` in `pyproject.toml`.

Set the prerequisite variables first, then run:

```bash
export ROBORAY_GPU_SMOKE=1
export ROBORAY_VIPE_PYTHON=<vipe venv>/bin/python   # ViPE is a venv, not a conda env; see build_vipe.sh
export ROBORAY_FFS_PYTHON=$(conda info --base)/envs/ego-ffs/bin/python
export ROBORAY_HAMER_PYTHON=$(conda info --base)/envs/ego-hamer/bin/python
export ROBORAY_VIPE_WEIGHTS=/data/weights/vipe
export ROBORAY_FFS_WEIGHTS=/data/weights/ffs
export ROBORAY_HAMER_WEIGHTS=/data/weights/hamer

# A command-line -m gpu **overrides** (rather than ANDs with) the -m "not gpu" in addopts,
# so running it directly selects the gpu tests; no extra options needed.
conda run -n roboray python -m pytest -m gpu -v
```

Optional (defensively clearing addopts explicitly; not required — the command above suffices):

```bash
conda run -n roboray python -m pytest -o addopts="" -m gpu -v
```

Test behavior: for each adapter it runs **real inference** on small synthetic data (4 frames, 288×512) and asserts artifact keys / shapes / finiteness — ViPE asserts `depth` shape and positive values, and `camera_poses` finite; FFS asserts `depth` / `camera_poses` / `intrinsics` shapes, and `depth` finite with positive values; HaMeR asserts `hand_present == (4,2)`, `joints_3d_cam == (4,2,21,3)`, and `mano_pose` finite. `test_ffs_gpu_smoke` / `test_hamer_gpu_smoke` additionally have a skipif: skipped when the corresponding `*_WEIGHTS` is unset.

## Stereo-Lane Env Dependencies

The stereo (two-camera) scene lane needs **two** envs: depth via `ego-ffs` (`ROBORAY_FFS_PYTHON`), pose via `ego-vipe` (`ROBORAY_VIPE_PYTHON`). `FastFoundationStereoModel.load()` first validates the FFS env and, when `ROBORAY_FFS_WEIGHTS` is **explicitly set**, validates that the weights dir exists — when unset it does not hard-fail at `load()` but lets `run_ffs.py` give the same readable guidance at `infer()` time; it then calls `self.vipe_model.load(device)` to validate the ViPE env — a missing env fails early with the matching build-script hint. `infer()` runs FFS first for depth, then ViPE for pose/intrinsics to replace FFS's placeholder identity matrix, and finally sets `depth_source="stereo"`. Input resolution is controlled by `stages.scene_reconstruction.stereo_max_image_side_px` (spec §6): via `--extra` it tells `run_ffs.py` to scale the stereo pair to `max(H, W) <= cap` with `fx` scaled proportionally; **depth is invariant to the scaling** (disparity scales proportionally with resolution), and the knob trades accuracy for speed; `0`/absent means no scaling.

**(Plan C fix)** The above scaling was fixed in Plan C for grid semantics: the scaled FFS produces disparity/depth on the **scaled grid**, and `run_ffs.py` now **resamples it back to the original frame resolution** (`cv2.resize(..., (orig_w, orig_h))`) before returning — i.e. the depth grid always equals the **original frame grid**, so downstream fusion/back-projection never needs to know whether scaling occurred; after resampling the depth values remain invariant to scaling (proportional disparity + linear resampling). Pinned by `test_ffs_depth_returns_to_original_grid_after_resize`, `test_fake_ffs_reports_both_grids`, `test_ffs_adapter_resize_keeps_depth_invariant`.
