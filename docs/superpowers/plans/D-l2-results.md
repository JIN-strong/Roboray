# Plan D - L2 Execution Record (in progress, with blockers)

Date: 2026-10-19  Branch: main

## Completed (real execution)

| Step | Result |
| --- | --- |
| GPU capacity | Node `<gpu-node-ip>` **2 A800 free** (`used=0`), GATE=OPEN |
| vendored repos | vipe / fast-foundationstereo / hamer cloned and commit-pinned (476f424 / 3a01849 / 8c9f361) |
| model env | single env `ego-models` (python 3.10 + **torch 2.7.0+cu128**, 7.1GB), FFS deps installed (timm/einops/omegaconf/scikit-image/opencv/imageio) |
| FFS weights | **downloaded** (265MB, `model_best_bp2_serialize.pth` + `cfg.yaml`, HF `nvidia/c-fast-foundationstereo`) |
| FFS real API | **verified** and written back to `scripts/modelenvs/run_ffs.py`: `torch.load(ckpt, weights_only=False)` full model + `InputPadder(divis_by=32)` + `model.forward(iters, test_mode, optimize_build_volume='pytorch1')` + bf16 autocast — the original `[L2]` assumption `FoundStereo(args=...)` **was wrong**, corrected |
| env migration | tar-streamed 7.1GB to node `/home/ubuntu/ego/ego-models`, `conda-meta` complete |

## Two key environment conclusions (measured on this machine, written into the runbook)

1. **The `socks://` proxy breaks every Python HTTP client** (httpx directly raises
   `ValueError: Unknown scheme for proxy URL`), but `HTTP_PROXY/HTTPS_PROXY` (http proxy) works (`huggingface.co` 200 through it).
   → Correct practice: `env -u all_proxy -u ALL_PROXY <command>`, keeping the http proxy. HF weights are therefore downloadable.
2. **"Do not import from the source directory" from numpy inside the container is misleading**: the real cause is
   `OpenBLAS blas_thread_init: RLIMIT_NPROC`.
   → Must set `OPENBLAS_NUM_THREADS=1` (applies on this machine and in containers).

## Current blocker: CUDA unavailable in container

Symptom (node `A800 GPU node 2`, image `registry.example.com/admin/ray:v2.53.0`):

- `nvidia-smi -L` **correctly** lists 2 A800s
- `torch.cuda.device_count == 2`
- but `torch.cuda.is_available() == False`; `torch.cuda.init()` reports
  `RuntimeError: Found no NVIDIA driver on your system`
- `find / -name libcuda.so.1` inside the container: **no result**; `--runtime=nvidia`, `--gpus all`,
  `-e LD_LIBRARY_PATH=/usr/local/nvidia/lib64` all have no effect

Assessment: this cluster uses a **vendor-customized GPU stack** (`op-adrm-nvidiagpu` / `nvidiagpu-devplugin`); `docker --gpus`
only passes through the `/dev/nvidia*` device nodes, without injecting `libcuda.so.1`, so torch can enumerate devices
but cannot establish a CUDA context.

**Next step (one of two)**:
- (a) copy the spec of an existing working GPU pod on the platform (a shared RayCluster's worker holds 2
  cards) — its `annotation`/`runtimeClassName`/device-plugin resource declarations — and start our pod the same way;
- (b) take the KubeRay route (Plan D Task 3's CR), letting the platform's existing KubeRay operator handle GPU
  injection — a platform-verified working path.

## Not executed

- L2 three-model smoke (blocked by the above)
- L3 three-mode comparison (blocked by the above)
- ViPE: local build needs nvcc (its `csrc/` is the DROID-SLAM core operators); this machine has no CUDA toolkit; need to
  build the wheel inside a CUDA devel image
- HaMeR: `MANO_RIGHT.pkl` requires a manual registration download, currently missing → that smoke skipped by design

---

## Addendum (KubeRay / Pod route measurements)

To avoid hand-run `docker run`, switched to K8s: created the `roboray` namespace + Pod (`nvidia.com/gpu: 1`,
hostPath mount `/home/ubuntu/ego`, image = the node's existing `registry.example.com/admin/ray:v2.53.0`).

**Result**: scheduling succeeded, hostPath mount succeeded, `nvidia-smi -L` correctly lists 2x A800,
`torch.cuda.device_count == 2`, but **`torch.cuda.is_available()` is still False**, `dev None`.

Exhausted attempts (all with the same result):

| Method | Result |
| --- | --- |
| `docker run --gpus all` | device_count=2, init failed |
| `docker run --runtime=nvidia --gpus all` | same |
| `-e LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/cuda/lib64` | same (no libcuda at that path) |
| K8s Pod + `nvidia.com/gpu: 1` | same |
| K8s Pod + `runtimeClassName: nvidia` (RuntimeClass exists, handler=nvidia) | same |

**Root-cause assessment**: `libcuda.so.1` is never found inside the container (`find / -name libcuda.so.1` has no
result). The platform passes through device nodes and `nvidia-smi` but **does not inject the driver libraries**; torch
must have `libcuda.so.1` to establish a CUDA context.

The platform's own GPU workloads (vLLM, `tengxiao-share-raycluster`) use the cards fine, meaning there is a convention
we have not yet reproduced.

**Next step (needs one piece of info)**: inspect the spec of a pod **actually running on the GPU** (my earlier query
mistakenly matched a CPU worker); focus on:
- whether its image bundles CUDA driver-compatible libraries (e.g., nvidia/cuda-based with `LD_LIBRARY_PATH` pointing
  at in-image CUDA)
- whether a `hostPath` mounts host driver libraries into the container (e.g., `/usr/lib64/libcuda*`,
  `/usr/lib/x86_64-linux-gnu/libcuda*`)
- the nvidia runtime config on the containerd side (whether `nvidia` runtime is default in
  `/etc/containerd/config.toml`)

Once we have that spec we can copy it, and L2 can run (FFS weights and the verified real API are both ready).

## Still not executed

- L2 three-model smoke (blocked by the driver-injection above)
- L3 three-mode comparison (same)

---

## Breakthrough: GPU usable in a K8s Pod (2026-10-20)

### Platform convention (measured and confirmed)

**K8s Pod + image bundling its own `libcuda.so.1`** is required to get the GPU:

| Combination | Result |
| --- | --- |
| `docker run --gpus all` + ray image (no libcuda) | `cuda False` |
| `docker run --gpus all` + vLLM image (has libcuda) | `cuda False` ← plain docker never works |
| **K8s Pod + ray image (no libcuda)** | `cuda False` |
| **K8s Pod + vLLM image (bundles `/usr/lib/x86_64-linux-gnu/libcuda.so.1`)** | ✅ **`our torch 2.7.0+cu128 cuda True`** |

i.e., the device nodes (`/dev/nvidia*`) are injected by the platform, but the **driver userspace libraries must be
bundled in the image**; and it must go through K8s (`nvidia.com/gpu`), not hand-run docker.

### Another key finding: the node can reach the internal Artifactory

The node has no external network, but it **can reach the pypi repo at `<internal-pypi-mirror>`** (302/200; `pip install six`
verified). → Model-env dependencies can be completed directly on the node, **no need to transfer 7GB again**. FFS deps
were completed this way: `FFS deps ready | torch 2.7.0+cu128` (including importable `core.utils.utils.InputPadder`).

### FFS real-inference progress

Reached inside the real model code; currently stuck at:

```
omegaconf.errors.ConfigAttributeError: Missing key normalize
  full_key: normalize   object_type=dict
```

Ruled out: the `cfg.yaml` next to the weights does **not** contain `normalize`; `run_demo.py`'s argparse also has no
`--normalize`; `model.args` is a plain dict (changed to `OmegaConf.create({**model.args, **cfg})` merged, key still missing).

**Next step**: locate where `normalize` is read inside `core/`, confirm who should supply it (possibly injected by a
default somewhere in the demo path other than `cfg.yaml`); once supplied, the first real-model L2 can complete.

### ✅ FFS L2 passed (2026-10-20)

On a K8s Pod (`registry.example.com/admin/vllm:v0.26.0` bundles libcuda) + A800, ran `run_ffs.py` with **real weights**:

```
__RESULT__{"ok": true, "model": "ffs", "n_frames": 4, "model_input_size": [288, 512], "depth_grid": [288, 512]}
```

Artifact verification (`ffs_out.npz`):

| Item | Value |
| --- | --- |
| keys | `__schema_version__ / camera_poses / depth / depth_source / fps / intrinsics / n_frames` (contract complete) |
| depth | `(4, 288, 512) float32`, all finite, **0.2344 ~ 17.452 m** (median 0.4095 m, physically plausible) |
| depth_source | `stereo` (took the stereo path) |
| camera_poses / intrinsics | `(4,4,4)` / `(4,3,3)` |

i.e., **the Fast-FoundationStereo adapter works end to end on a real GPU**, and all the script's `[L2]` assumptions
have been replaced with the real API (`torch.load` full model / `InputPadder(/32)` /
`forward(optimize_build_volume='pytorch1')` / `cfg.yaml` merged into `model.args` / supply `normalize=True`).

### FFS real-API correction list ([L2] → verified)

1. `torch.load(ckpt, map_location='cpu', weights_only=False)` loads the **full model** (not `FoundStereo(args=...)`)
2. `InputPadder(img.shape, divis_by=32, force_square=False)` pads first, `unpad` after
3. `model.forward(p0, p1, iters=8, test_mode=True, optimize_build_volume='pytorch1')`, bf16 autocast
4. the `cfg.yaml` beside the weights must be merged into `model.args` (`OmegaConf.create`)
5. `args.normalize` is **missing from both** the pickle and `cfg.yaml`; supply `True` (read at `foundation_stereo.py:204`)

### ViPE blocker: needs Python >= 3.11

After installing nvcc on the node (Artifactory: `nvidia-cuda-nvcc-cu12 12.9.86`), compiling ViPE is stopped by its metadata:

```
ERROR: Package 'nvidia-vipe' requires a different Python: 3.10.22 not in '<3.15,>=3.11'
```

i.e., ViPE **requires Python >=3.11**, but the `ego-models` env is 3.10 (chosen originally for HaMeR/pytorch3d compatibility).

**Feasible paths** (next step):
- on the node, build a separate venv with the **Python 3.11 bundled in the vLLM image**, install torch from Artifactory
  (the PyPI torch bundles the CUDA runtime), then compile-install ViPE;
- or create a local `ego-vipe` (py3.11) env and transfer the whole directory (~6–7GB).

Also note: under `nvidia-cuda-nvcc-cu12`'s `bin/` only `ptxas` is visible; the nvcc binary itself may be in another
subdirectory — confirm the actual path when assembling `CUDA_HOME`.

### HaMeR blocker: MANO registration files missing

`MANO_RIGHT.pkl` requires registering at https://mano.is.tue.mpg.de then downloading manually; it is not in the current
environment → that model's smoke is **skipped by design** (no faked success).

## This round's L2 summary

| Model | Status |
| --- | --- |
| **Fast-FoundationStereo** | ✅ **pass** (A800 real weights, depth 0.23–17.45m, contract complete) |
| ViPE | ⛔ needs Python >=3.11 (environment issue, not algorithmic) |
| HaMeR | ⛔ needs manual registration download of MANO_RIGHT.pkl |

### ViPE progress (2026-10-20, incomplete)

Resolved: the vLLM image bundles **Python 3.12** (satisfies ViPE's `>=3.11`) and **nvcc** (`/usr/local/cuda/bin/nvcc`),
so a `vipe-venv` was created inside the image to compile ViPE.

Achieved: `vipe import OK` (the ViPE package itself imports).

**Current blocker**: **DNS resolution fails** inside the container (`Temporary failure in name resolution` toward
`<internal-pypi-mirror>`), so torch cannot install into the venv (`No matching distribution found for torch`). The node host
can resolve that domain (`<internal-pypi-ip>`), but the container cannot; `--add-host <internal-pypi-mirror>:<ip>` did not work.

**Next step**: give the container explicit DNS (`docker run --dns <internal-DNS>`), or pre-install torch into the venv
on the host (where DNS works) before entering the container to compile.

### ViPE progress (cont.): container networking and thread limits

The user pointed out the container network must use `--network host` or `dockerBridge`. Measured (node `A800 GPU node 2`):

| Config | DNS | connectivity | pip |
| --- | --- | --- | --- |
| default bridge | ❌ resolution fails | ❌ no route | ❌ |
| `--dns <internal-dns>` | ❌ | ❌ | ❌ |
| `--add-host <internal-pypi-mirror>:<internal-pypi-ip>` | ❌ | ❌ | ❌ |
| **`--network host`** | ✅ resolves to <internal-pypi-ip> | curl 000 (image lacks CA etc.) | ✅ **steps executable** |

i.e., **the container has no network by default; `--network host` is required** (a `dockerBridge` network also exists on the node).

Once into pip, a new blocker: **`RuntimeError: can't start new thread`**
(occurs when pip installs vipe's build deps; host `ulimit -u`=10000; adding
`--ulimit nproc=65535:65535 --ulimit nofile=65535:65535` still reproduces it). The venv stalled at 13MB, torch not loaded.

**Feasible next steps**:
1. use `ego-models`'s (py3.10) pip to cross-version download py3.12 wheels:
   `pip download --python-version 3.12 --only-binary=:all: --platform manylinux2014_x86_64 torch -d /ego/wheels`
   then `pip install --no-index --find-links /ego/wheels` inside the container (bypassing container networking);
2. or fix the container's thread cap (check the container PID cgroup limit, not just ulimit);
3. or create a local `ego-vipe` (py3.11+) env and transfer the whole directory.

### ViPE offline-install progress (partially successful)

Using "download wheels on host → offline install in container" to bypass container networking, **torch 2.6.0+cu124
installed into the venv successfully** (`--no-index --find-links` avoided `can't start new thread`).

The `vipe` package itself still will not install, because its dependency tree is not fully prepared:

```
ERROR: Could not find a version that satisfies the requirement click (from nvidia-vipe)
```

ViPE's dependency list (`third_party/vipe/pyproject.toml`):

```
torch torchvision click einops gdown gtsam-develop==4.3a2.dev202609030306
huggingface_hub hydra-core imageio[ffmpeg] kornia matplotlib ninja numpy omegaconf
opencv-python<5 OpenEXR<3.3.0 pydantic>=2,<3 pillow python-pycg ray rerun-sdk
safetensors scipy timm tqdm transformers>=4,<5 viser
```

Among these, `gtsam-develop==4.3a2.dev202609030306` (a specific dev version), `OpenEXR<3.3.0`, `python-pycg`, `viser`,
`rerun-sdk`, etc., may not be available on the internal mirror or may need compilation.

**Next step**: `pip download --python-version 3.12` all 27 deps to `/ego/wheels`, then install offline; for missing
items, confirm the internal mirror has a source one by one, downgrading/substituting when needed.

### ViPE dependency-tree completion result (2026-10-20)

**All dependencies installed offline into the venv** (py3.12, via `--network host` + host-downloaded wheels +
`--no-index` install):

```
click einops gdown hydra-core kornia matplotlib ninja numpy omegaconf opencv-python
safetensors scipy timm tqdm huggingface_hub pydantic pillow imageio transformers
OpenEXR viser torch torchvision setuptools wheel … (~70 packages, incl. nvidia-* CUDA runtimes)
```

**Two items with no source on the internal mirror**:
- `gtsam-develop==4.3a2.dev202609030306` (the dev version ViPE pins) → `No matching distribution`
- `rerun-sdk` → `No matching distribution`

**Remaining blocker**: `pip install -e /ego/vipe` fails building the **editable wheel**
(`Failed building editable for nvidia-vipe`) — i.e., its **CUDA extensions (csrc/: corr_ext / droid_net_ext /
lietorch_ext etc.) did not compile**. Detailed compile errors need extraction from the full build log
(`sudo docker logs vipebuild | grep -i nvcc`).

**Next step**:
1. extract the full compile error (common causes: `TORCH_CUDA_ARCH_LIST` unset making compilation target all
   architectures and slow/fail; missing CUDA header paths; `CUDA_HOME` layout not matching what
   `torch.utils.cpp_extension` expects);
2. `gtsam-develop` has no source → try the internal `gtsam` release as a substitute, or compile from source;
3. if `rerun-sdk` is only for visualization, evaluate whether it can be optional in ViPE.

---

## ✅ ViPE L2 passed (2026-10-20)

On a K8s Pod (`registry.example.com/admin/vllm:v0.26.0`) + A800, ran **`PoseOnlyAnnotationPipeline`** with **real
weights** (`dog-example.mp4`, 40 frames):

```json
{"ok": true, "model": "vipe", "n_frames": 40, "seconds": 13.4,
 "keyframe_ids": [0,5,9,13,16,19,21,27,32,39],
 "intrinsics": [984.259, 984.396, 640.0, 360.0],
 "depth_source": "monocular_slam_map_keyframe",
 "depth_finite": true, "depth_min": 2.011, "depth_max": 5.439,
 "gpu": "NVIDIA A800 80GB PCIe"}
```

Artifact verification (`vipe_out.npz`):

| Item | Value |
| --- | --- |
| keys | `camera_poses / depth / depth_source / fps / intrinsics / n_frames` |
| camera_poses | `(40,4,4)` float64 (w2c, inverted from c2w) |
| intrinsics | `(40,3,3)` float64, fx≈fy≈984.26, cx=640, cy=360 (**exactly the 1280x720 principal point**, physically plausible) |
| depth | `(1,720,1280)` float32, **all finite**, 2.011 ~ 5.439 m |
| fps | 30 |

**Honest boundary**: `ba_residual` reports `0.0`, but this is the **default** of ViPE-side `SLAMOutput` —
`return SLAMOutput(...)` in `SLAMSystem.run` (`vipe/slam/system.py:327`) **does not pass that field**, so it is
"unfilled", not "perfect BA". This record makes no accuracy claim from it.

### Breakthrough point 1: the real cause of the container's `can't start new thread` is **seccomp**

The earlier "container thread cap" assessment **was wrong**. Measured:

| Metric | Value |
| --- | --- |
| `ulimit -u` | 2058893 (huge, not the bottleneck) |
| cgroup `pids.max` | does not exist |
| `/proc/self/status` `Seccomp` | **2 (filter mode)** |
| plain `threading.Thread().start()` x 200 | **fails on the 0th** `RuntimeError: can't start new thread` |

i.e., the platform's default seccomp policy **directly blocks `pthread_create`**. Adding
`--security-opt seccomp=unconfined` gives `Seccomp: 0`, all 200 threads succeed, **and pip can connect to the
Artifactory directly** — the "download wheels on host → offline install in container" workaround is thereby obsolete.

### Breakthrough point 2: PyPI's `nvidia-cuda-nvcc-cu12` **has no nvcc driver**

Compiling ViPE's `vipe_ext` (DROID-SLAM operators) needs nvcc, and:

- `/usr/local/cuda` in the container is **CUDA 13.0**, torch in the venv is **2.6.0+cu124**
  → `cpp_extension._check_cuda_version` reports a major mismatch (12 vs 13)
- the PyPI `nvidia-cuda-nvcc-cu12` wheel has **only `ptxas` in `bin/`**, **no `nvcc`**
- the PyPI `nvidia-cuda-cccl-cu12` wheel has **no `nv/`**, causing
  `cuda_fp16.h:4100: fatal error: nv/target: No such file or directory`

Solution (fetch the packages on the **laptop** — it has external network — then transfer to the node):

| Item | Source | Destination |
| --- | --- | --- |
| nvcc 12.4.131 | conda `nvidia` channel `cuda-nvcc-12.4.131-0.tar.bz2` (65MB) | `$CH/bin/nvcc`, `$CH/nvvm/bin/cicc` |
| CCCL 12.4.127 | conda `cuda-cccl-12.4.127-0.tar.bz2` | `$CH/include/nv/` |
| other CUDA libs/headers | each `nvidia/*/include`, `nvidia/*/lib` in the venv | merged into `$CH/include`, `$CH/lib64` |
| `libcudart.so` etc. without versioned names | `lib*.so.*` → `lib*.so` symlink | `$CH/lib64` |

With `CUDA_HOME=/ego/cuda-12.4`, `TORCH_CUDA_ARCH_LIST=8.0` (A800=sm80),
`MAX_JOBS=16`, `pip install --no-deps --no-build-isolation -e /ego/vipe` passes.

(CUDA 12.9 and torch cu124 differ only in minor version; torch warns but does not error; still took 12.4 for safety.)

### Breakthrough point 3: the internal mirror served a **ten-year-old** omegaconf / hydra-core

`pip install hydra-core` on the internal mirror installed `hydra-core 0.11.3` + `omegaconf 1.4.1`
(2019 versions), so `vipe` crashed on import:

```
AttributeError: type object 'OmegaConf' has no attribute 'has_resolver'
```

The mirror actually **does** have newer versions; they must be pinned explicitly: `omegaconf==2.3.1`,
`hydra-core==1.3.2`. Also: `gtsam-develop==4.3a2.dev202609030306` has no source on the mirror; the release
**`gtsam==4.3.0`** is used instead; `rerun-sdk 0.39.0`, `python-pycg 1.0.3`, `ray 2.59.0` all install.
Large packages (rerun-sdk ~100MB) need `--timeout 180 --retries 6`, otherwise Read timed out.

### ViPE weight manifest (placed offline, `TORCH_HOME=/ego/torch_home`)

| Model | Source | Destination |
| --- | --- | --- |
| GeoCalib pinhole | `github.com/cvg/GeoCalib/releases/.../geocalib-pinhole.tar` (116MB) | `hub/geocalib/pinhole.tar` |
| MoGe-2 vitl | HF `Ruicheng/moge-2-vitl` → `model.pt` (1.3GB) | `hub/moge2/moge-2-vitl.pt` |
| DROID-SLAM | gdown id `1PpqVt1H4maBa_GbPJp4NwxRsd9jk-elh` (16MB) | `hub/droid_slam/droid.pth` |
| SuperPoint | `github.com/rpautrat/SuperPoint/raw/.../superpoint_v6_from_tf.pth` (2.6MB) | `hub/checkpoints/superpoint_v6_from_tf.pth` |

Also set `HF_HUB_OFFLINE=1`. `pipeline.init.instance=null` skips TrackAnything (GroundingDINO+SAM); not needed for
pose/depth only.

### ViPE real-API correction list (`[L2]` → verified)

All of the `[L2]` assumptions in `scripts/modelenvs/run_vipe.py` **were wrong** and have been replaced:

1. import path: not `vipe.pipeline.DefaultAnnotationPipeline` / `vipe.stream.VideoStream`,
   but `vipe.config.validate_typed_config` + `vipe.streams.base.StreamList.make` + `vipe.pipeline.make_pipeline`
2. **hydra's pipeline group is nested under `cfg.pipeline`** (not flattened to the root):
   overrides must be written as `pipeline.output.path=...`; `pipeline.init.instance` is not yet in the struct at
   compose time; it can only be `OmegaConf.update(cfg, "pipeline.init.instance", None)` after composing
3. `pipeline.return_payload = True` → `run(stream).payload` is the `SLAMOutput`
4. `SLAMOutput.get_view_trajectory(0).matrix()` is **(N,4,4) c2w**
   (per `scripts/vipe_to_colmap.py`: it inverts that matrix to get w2c)
5. `SLAMOutput.intrinsics[0]` is `[fx,fy,cx,cy]`
6. dense depth is not in `SLAMOutput`; must use
   `slam.slam_map.project_map(frame_tstamp, view_idx, target_size, target_intrinsics, target_pose, target_camera_type, infill=True, tstamp_nn=3)`
   to backproject from the sparse SLAM point cloud (`pose_only` has no per-frame dense-depth postprocessing)

### Another fix: `.rayignore` (third_party was crushing Ray packaging)

The vendored `third_party/` (vipe/fast-foundationstereo/hamer, **271MB**) was packaged into Ray's `working_dir`
(beyond what `.gitignore` excluded), so every `ray.init()` compressed and uploaded 265MB, crushing the local Ray node
(actors reported `Owner's node has crashed`), failing 2 cases in `tests/unit/test_actors.py`.

Added `.rayignore` (Ray reads only it, no longer `.gitignore`, so it must be listed fully) excluding
`third_party/` etc.; after the fix: **206 passed / 0 failed (EXIT=0)**.

## L2 three-model master table (updated)

| Model | Status |
| --- | --- |
| **Fast-FoundationStereo** | ✅ **pass** (depth 0.23–17.45m, contract complete) |
| **ViPE** | ✅ **pass** (40 frames / 13.4s / A800; intrinsics 984.26/640/360; depth 2.01–5.44m) |
| HaMeR | ⛔ needs manual registration download of `MANO_RIGHT.pkl` (https://mano.is.tue.mpg.de), not automatable |

---

## ✅ HaMeR L2 passed (2026-10-20) — all three models done

The MANO registration file was provided by the user (`MANO_RIGHT.pkl`, 3.8MB, added to `.gitignore` — the MPI license
forbids committing/redistributing it). On a K8s Pod (`ego-hamer:l2` image) + A800, ran `run_hamer.py` with **real weights**:

```json
{"ok": true, "model": "hamer", "n_frames": 4,
 "detection_rate": 1.0, "n_boxes": 8,
 "focal_px": 763.2, "box_source": "mediapipe",
 "joints_3d_finite": true,
 "mano_trans_median_norm": 0.8195}
```

Artifact verification (`hamer_probe_out.npz`):

| Key | shape / value |
| --- | --- |
| `hand_present` | `(4,2)` bool, right slot 4/4 hits |
| `joints_3d_cam` | `(4,2,21,3)` float32, **all finite** |
| `joints_2d` | `(4,2,21,3)` float32 |
| `mano_pose` | `(4,2,48)` float32, all finite |
| `mano_shape` / `mano_trans` | `(4,2,10)` / `(4,2,3)` |
| `confidence` | **detector's real score 0.988** (previously placeholder 1.0, fixed) |
| `mano_trans[0,right]` | `[-0.020, 0.114, 0.811]` → **0.811 m** |

### HaMeR real-API correction list (`[L2]` → verified)

All 6 `[L2]` assumptions in `run_hamer.py` **were wrong**; replaced one by one:

1. **import/construction**: not `HAMER.load_pretrained(weights)`, but
   `from hamer.models import load_hamer` → `load_hamer("<weights>/_DATA/hamer_ckpts/checkpoints/hamer.ckpt")`
2. **batching**: `ViTDetDataset(model_cfg, img_bgr, boxes, right, rescale_factor=2.0)` + `DataLoader` → `out = model(batch)`
3. the original **`model.detect_and_fit(...)` was made up** (HaMeR has no such method); the actual output keys are
   `pred_cam / pred_cam_t / pred_mano_params / pred_keypoints_3d / pred_keypoints_2d / pred_vertices`
4. **MANO params**: `pred_mano_params` are **rotation matrices** and **do not include transl**; `pred_keypoints_3d`
   is **relative to the root joint**
5. **full-image camera translation**: `cam_crop_to_full(pred_cam, box_center, box_size, img_size, focal)`
   (before calling, `pred_cam[:,1] *= (2*right-1)`); **camera-frame joints = pred_keypoints_3d + cam_t_full**
6. **MANO layer**: use `model.mano` directly (full pose, not PCA, `posedirs (778,3,135)`)

### Scale: with vs. without K differs a lot (wired into the pipeline)

In `cam_crop_to_full`, `tz = 2f/bs`, **proportional to the focal length**. The official demo uses a prior focal
(`FOCAL_LENGTH/IMAGE_SIZE*img_size.max()` ≈ 25000px) → relative scale; with the **measured K** → metric.
Measured on the same image:

| Focal source | `mano_trans` median norm |
| --- | --- |
| official prior (no K) | **17.6** (relative scale, not meters) |
| measured K (f=763.2) | **0.8195 m** (physically plausible) |

The product code was changed accordingly: `hand_models.py::HamerHandModel.infer` now writes `calib["K"]` into the
delivered npz, and the adapter prefers it. **Previously HaMeR output was always relative scale**, so the "unit: meters"
written in the contract was not actually true — it is now (when K is given).

### Detection-box source (honestly noted substitute)

The official detection chain is detectron2(ViTDet) + mmpose(ViTPose); **neither installs in this environment**:

- `mmcv==1.3.9` (a ViTPose dependency) is a 2021 version, incompatible with torch 2.x
- detectron2 additionally needs the 2.4GB ViTDet weights (`model_final_f05665.pkl`)

So the adapter takes boxes in the order `--extra {"hand_boxes": ...}` → **mediapipe hand detection** → error.
**The MANO fitting part of HaMeR was not changed at all**; only "where the hand boxes come from" was replaced. A
machine with the official detection chain installed can switch back itself.

### Environment pitfalls hit and fixed this round (all written into the header of build_hamer.sh)

| # | Symptom | Root cause | Disposition |
| --- | --- | --- | --- |
| 1 | `from hamer.models import ...` → `ImportError: Unable to load EGL library` | `hamer.utils` pulls pyrender/OpenGL at import time | install `libegl1` in the image (`deploy/Dockerfile.hamer`) |
| 2 | `docker build` apt fails exit 100 | the base image's `APT::Update::Post-Invoke*` hook returns nonzero in the build env | `sed -i "/Post-Invoke/d"` remove the hook; also require `--network host` |
| 3 | `import chumpy` → `cannot import name 'bool' from 'numpy'` | chumpy 0.70 uses a numpy<1.24 alias | runtime numpy-alias shim (in the adapter) |
| 4 | chumpy will not install | its setup.py imports numpy at build time | `--no-build-isolation` |
| 5 | `-e /ego/hamer` hangs | build isolation contacts pypi.org (node has no external network) | `--no-deps --no-build-isolation` |
| 6 | `ModuleNotFoundError: webdataset` | `hamer.datasets` imports it at top level | install webdataset |
| 7 | `invalid index of a 0-dim tensor` | the official prior focal is a 0-dim scalar | make it a `(B,)` tensor uniformly |
| 8 | weights path not found | `CACHE_DIR_HAMER="./_DATA"` is relative to CWD | adapter `os.chdir(weights)` |

## L2 three-model master table (final)

| Model | Status | Key numbers |
| --- | --- | --- |
| **Fast-FoundationStereo** | ✅ pass | depth 0.234–17.452 m (median 0.410 m), `depth_source=stereo` |
| **ViPE** | ✅ pass | intrinsics 984.26/640/360 (exactly the 1280x720 principal point), depth 2.011–5.439 m |
| **HaMeR** | ✅ **pass** | detection rate 1.0, 2 hands/frame, `mano_trans` 0.82 m, joints/pose all finite |

**L2 has no remaining gaps** (HaMeR's official detection chain is the only substitute item, explicitly noted in the docs).
