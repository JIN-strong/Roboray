# Plan D Verification Report (deployment + L2/L3 measurements)

Date: 2026-10-20  Branch: main  Target cluster: A800 K8s (node `<gpu-node-ip>`)

## 1. Status Determination

Per the three-state definition of Plan D Task 6, this plan is **partially executed**:

- ✅ **L2 all three models pass**: Fast-FoundationStereo, ViPE, and **HaMeR** all ran end to end with **real weights
  on the A800**, artifact contracts complete (numbers below).
- ⚠️ **HaMeR's official detection chain is a substitute**: the official detectron2(ViTDet)+mmpose(ViTPose) does not
  install in this environment (`mmcv==1.3.9` is incompatible with torch 2.x; ViTDet weights additionally need 2.4GB),
  so **mediapipe** is used to obtain hand boxes; **the MANO fitting part of HaMeR was not changed at all**. Explicitly
  noted in the docs.
- ✅ **L3 three-mode comparison ran on a real Ray cluster** and reproduced over two rounds (`D-l3-results.md`).
- ✅ **Real-model end-to-end reached fusion**: after MANO was in place, re-ran `backend=real` with **real video**
  input; prepare / **scene(ViPE)** / **hand(HaMeR)** / fusion — all four stages done (see section 7 of `D-l3-results.md`).
- ⛔ **action stage not verified**: no reachable VLM service (the cluster has vLLM, but the `roboray` Pod cannot
  connect — network isolation). Error branches and the keyframe chain are covered by unit tests.
- ⛔ **three-mode comparison with real models not done**: the real-model path only ran the adaptive mode. The
  three-mode comparison is still **stand-in model** results (measures the scheduler, not compute).

**Forbidden** to write item 4 above as "verified".

## 2. Full Test Suite

```
$ RAY_ADDRESS=local conda run -p <envs>/roboray python -m pytest -q
210 passed (EXIT=0)
```

> `RAY_ADDRESS=local` is necessary: a stale `/tmp/ray/ray_current_cluster` on this machine may point at an existing
> cluster, which makes `ray.init(num_cpus=..., num_gpus=...)` directly fail with
> `ValueError: When connecting to an existing cluster, ...`.

| Plan | Status | Test count |
| --- | --- | --- |
| A foundation + scheduling core | ✅ merged to main | 115 |
| B real models | ✅ merged to main | 171 |
| C stages 3/4 | ✅ merged to main | 206 |
| D deployment + measurements | 🔄 partially executed | **210** (+4: `test_ray_init.py`) |

## 3. L2 Measurements (real weights, real GPU)

| Model | Status | Key numbers |
| --- | --- | --- |
| **Fast-FoundationStereo** | ✅ pass | 4 frames; depth 0.234–17.452 m, median 0.410 m; `depth_source=stereo`; all contract keys present |
| **ViPE** | ✅ pass | 40 frames / 13.4 s; intrinsics fx=fy≈984.26, cx=640, cy=360 (exactly the 1280x720 principal point); depth 2.011–5.439 m all finite; 10 keyframes; c2w[0]≈I |
| **HaMeR** | ✅ pass | detection rate 1.0, 2 hands/frame; `joints_3d_cam (4,2,21,3)` / `mano_pose (4,2,48)` all finite; `mano_trans` median norm **0.82 m** (metric after giving K); confidence is the detector's real score 0.988 |

**Honest boundary**: ViPE's `ba_residual=0.0` is the **unfilled default value** of `SLAMOutput`
(`return SLAMOutput(...)` at `vipe/slam/system.py:327` does not pass that field); not claimed as an accuracy statement.

## 4. L3 Measurements

Real RayCluster (head CPU + 1x A800 worker, Ray 2.59.0), same session, same batch of inputs
(3 clips x 30 frames @320x240), stand-in stage times set to scene=2.0 s / hand=1.0 s:

| Mode | two-run total_seconds | vs serial | scheduler snapshots |
| --- | --- | --- | --- |
| serial | 21.952 / 21.332 | 1.000 | 0 |
| pipeline | 18.774 / 17.577 | 1.169x / 1.214x | 0 |
| adaptive | 14.287 / 13.519 | **1.536x / 1.578x** | 6 |

Both runs: failed = 0. **This is a scheduler-behavior comparison, not a GPU-compute comparison** (stand-ins do not touch the GPU).

## 5. Real Issues Found and Fixed During Plan D

| # | Issue | Nature | Disposition |
| --- | --- | --- | --- |
| 1 | `can't start new thread` inside container | **platform convention** (seccomp blocks `pthread_create`) | `--security-opt seccomp=unconfined` / `seccompProfile: Unconfined`; previously misdiagnosed as ulimit/cgroup |
| 2 | No CUDA inside container | **platform convention** (driver userspace libs must be bundled in the image and go through K8s) | image that ships its own `libcuda.so.1` + K8s Pod |
| 3 | PyPI's `nvidia-cuda-nvcc-cu12` has no `nvcc`; `nvidia-cuda-cccl-cu12` has no `nv/` | upstream packaging fact | take `cuda-nvcc`/`cuda-cccl` from the conda `nvidia` channel, assemble `CUDA_HOME` |
| 4 | internal mirror resolves `hydra-core`/`omegaconf` to 2019 versions | mirror data issue | pin `1.3.2` / `2.3.1` explicitly; `gtsam-develop` has no source → `gtsam==4.3.0` |
| 5 | **`--ray-address` to an existing cluster always crashed** | **product bug** | extract `ray_init_kwargs()`, add 4 unit tests |
| 6 | Ray `working_dir` packaged 271 MB + 265 MB, over the limit | config gap | add `.rayignore` (includes `third_party/`, `deploy/assets/`, `assets/`) |
| 7 | both existing KubeRay operators are `--watch-namespace`-restricted, CRs never reconciled | platform convention | `deploy/k8s/setup_operator.sh` starts a namespace-local operator in `roboray` |
| 8 | `run_vipe.py` silently dropped the `depth` key when depth unavailable | diagnosability defect | fail loudly with the real reason |
| 9 | all `[L2]`-assumed ViPE APIs were wrong | docs/adapter | rewrite `run_vipe.py` per the measured API, itemized in `D-l2-results.md` |
| 10 | all `[L2]`-assumed HaMeR APIs were wrong (`HAMER.load_pretrained` / `model.detect_and_fit` do not exist) | docs/adapter | rewrite `run_hamer.py` per measurements |
| 11 | **`CpuStageActor` does not register builtin models** → action stage inevitably `UnknownModelError` | **product bug** | register in actor + regression test (mutation-verified) + diagnostic method `registered_models()` |
| 12 | ViPE adapter `depth_source` went outside the contract enum | adapter | revert to `monocular`, move fine-grained source into the summary's `depth_detail` |
| 13 | HaMeR output was always relative scale (contract says meters) | contract/adapter | `hand_models.py` delivers `calib["K"]`; adapter prefers the measured focal length |

## 6. Cleanup Status

`kubectl -n roboray delete raycluster --all` executed; probe Pods deleted.
GPU recheck: `USED_GPUS=2` (all pre-existing platform vLLM workloads, `qwen38-flash-next-varg`),
**our occupancy is 0**. Only the lightweight KubeRay operator (no GPU) remains in the `roboray` namespace;
for full removal: `bash deploy/k8s/teardown.sh --all`.

## 7. Remaining Work

1. **Real-model three-mode comparison**: the MANO blocker is cleared (HaMeR L2 passed). Point `/ego/l3-real.yaml`'s
   `models.hammer_weights` at `/ego/weights/hamer`, **and switch the input to real video** (synthetic noise makes
   ViPE's dense-depth path fail, see `D-l3-results.md`), then re-run per section 5 of that report; real GPU
   utilization mean/peak can then be collected too.
2. **HaMeR official detection chain** (optional, non-blocking): on a machine where `mmcv` and detectron2 install,
   switch back to ViTDet + ViTPose and compare against this environment's mediapipe substitute.
3. **ViPE dense depth**: this round confirmed **synthetic noise data is unusable** (DROID-SLAM has no trackable
   structure). The real-model three-mode comparison should switch to **real video** input.
