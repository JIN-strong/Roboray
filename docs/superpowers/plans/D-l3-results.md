# Plan D - L3 Three-Mode Timing Comparison Execution Record

Date: 2026-10-20  Branch: main  Node: `<gpu-node-ip>` (A800 cluster, 2x A800, this run used 1 card)

## 1. Execution Environment (all measured)

| Item | Value |
| --- | --- |
| K8s | v1.22.3, namespace `roboray` |
| RayCluster | `ray.io/v1`, head (CPU) + 1x GPU worker (`nvidia.com/gpu: 1`) |
| Image | `registry.example.com/admin/vllm:v0.26.0` (bundles `libcuda.so.1`, per the D-l2-results platform convention) |
| Ray | **2.59.0** (both head/worker `ray` resolve to `/ego/roboray-venv/bin/ray`) |
| App interpreter | `/ego/roboray-venv/bin/python` (Python 3.10.22, venv built from `ego-models`'s 3.10) |
| Code and data | hostPath `/home/ubuntu/ego` → `/ego` (`/ego/app` is a repo copy) |

### Two pitfalls fixed along the way to deployment

1. **Neither of the platform's two existing KubeRay operators manages `roboray`**: one runs with
   `--watch-namespace` scoped to its own namespace, the other to another namespace. After the CR was created there were
   **no Pod, no Event** (not Pending — nobody reconciled it at all).
   Fix: start an operator in `roboray` that watches only this namespace (reusing the existing
   `ClusterRole/kuberay-operator`, adding a ClusterRoleBinding, not touching existing bindings).
   See `deploy/k8s/` and the "Reproduction commands" section of this file.
2. **Ray's `working_dir` packaging uploaded the repo's large directories too**: `third_party/` (271MB)
   and `deploy/assets/weights/ffs/` (265MB) both pushed the package past the 512MiB limit, failing with
   `RuntimeEnvSetupError`. Fix: exclude them explicitly via `.rayignore` (Ray reads only `.rayignore`,
   no longer `.gitignore`, so it must be listed fully). After the fix the package is **0.48MiB**.

### A product bug fixed in passing

`--ray-address` (connecting to an existing cluster) previously **always crashed**:

```
ValueError: When connecting to an existing cluster, num_cpus and num_gpus must not be provided.
```

Cause: when connecting to an existing cluster it still unconditionally passed `num_cpus=os.cpu_count()`. That path had
no test coverage (existing e2e all start a local cluster). Extracted `ray_init_kwargs()` and added 4 unit tests
(`tests/unit/test_ray_init.py`); full suite **210 passed / 0 failed**.

## 2. Three-Mode Comparison (same session, same batch of inputs)

- Input: `scripts/make_l3_dataset.py` self-built subset, **3 clips x 30 frames @ 320x240 @30fps** (stereo).
- Config: `/ego/l3-test.yaml`, `execution.max_gpus=1`, `num_cpu_actors=4`,
  `max_episodes_in_flight=2`, `backend=test`.
- Mode semantics (`src/roboray/runtime/driver.py` file header):
  `serial = --serial-baseline`; `pipeline = --no-adaptive`; `adaptive = default`.
- Stand-in stage times set via `ROBORAY_FAKE_SCENE_SECONDS=2.0` / `ROBORAY_FAKE_HAND_SECONDS=1.0`
  (**stand-in models, not real models** — reason in section 4).

| Mode | 1st run total_seconds | 2nd run total_seconds | vs serial (2nd/1st) | scheduler snapshots |
| --- | --- | --- | --- | --- |
| serial (per-episode per-stage serial) | 21.952 | 21.332 | 1.000 / 1.000 | 0 |
| pipeline (cross-stage pipelining + static pooling) | 18.774 | 17.577 | **1.169x / 1.214x** | 0 |
| adaptive (cross-stage pipelining + adaptive scheduling) | 14.287 | 13.519 | **1.536x / 1.578x** | **6** |

- adaptive vs pipeline: **1.314x / 1.300x** (consistent across the two rounds).
- Both rounds: 3 episodes each, **failed = 0 in both**.
- `scheduler_snapshots` is non-empty only in adaptive mode (6 entries), shaped like
  `{"slot_id":0,"chosen":"scene","wanted":"scene","needs_load":true,"scene_q":1,"hand_q":1,"predict":1.0,"loaded":null}`
  — direct evidence that "adaptive really decides from queues and cost predictions"; serial/static modes being empty
  matches the design.

**Conclusion**: on a real Ray cluster the three modes **reproducibly** separate; adaptive vs serial **≈1.54–1.58x**,
vs static pipeline **≈1.30x**. These ratios are measured here, not HOT3D paper values.

### A confound that must be stated

The first-round measurement (head pod given only 2 CPU) showed almost no difference between the three modes (22.96 /
23.29 / 23.02). Checking per-stage timing showed `publish` (the LeRobot export subprocess) stretched to 4–10s with
wild variance, eating the pipelining gain. After raising head CPU to 6 (worker still 8 CPU), the signal immediately and
stably reproduced. **i.e., insufficient head-side CPU masks the scheduling gain** — a real deployment-side constraint,
not a scheduler problem.

## 3. Real-Model Path Performance on the Cluster (partial success, honestly recorded)

On the same RayCluster with `backend=real` (`/ego/l3-real.yaml`, ViPE=`/ego/vipe-venv`,
FFS=`/ego/ego-models`, FFS weights=`/ego/weights/ffs`), ran 3 clips:

| Stage | Result |
| --- | --- |
| prepare | ✅ done (0.04–0.07s) |
| scene | ❌ failed |
| hand | ❌ failed |

- **The real models did use the GPU**: during execution the worker sampled
  `utilization.gpu=96%, memory.used=10225 MiB` and `65%, 10333 MiB` (`nvidia-smi` via `kubectl exec` into the worker pod).
- **The episodes' ultimate failure cause is HAND**:
  `ModelEnvError: ROBORAY_HAMER_WEIGHTS unset or directory missing. MANO_RIGHT.pkl must be downloaded after registering at
  https://mano.is.tue.mpg.de …`
- `scene` was marked failed because `SCENE ∥ HAND` run in parallel and HAND failing first makes the episode fail; the
  manifest keeps only one error — **scene itself did not crash**.

### Independent evidence for the real-model path (bypassing HAND)

To establish scene separately, two probes were made:

1. **FFS via a Ray GPU actor** (`num_gpus=1`): the FFS subprocess **succeeded** (no error); the failure occurred after
   its internal ViPE pose subprocess — i.e., FFS itself runs on the cluster worker.
2. **ViPE adapter run directly** (GPU pod, `run_vipe.py`):

```json
{"ok": true, "model": "vipe", "n_frames": 30,
 "intrinsics": [402.03, 402.03, 160.0, 120.0],
 "keyframe_ids": [0,1,2,...,26,28,29],
 "depth_source": "unavailable: RuntimeError: __copy::trivial_device_copy H->D: failed: cudaErrorInvalidValue"}
```

**Root cause located, and it is neither the adapter nor ViPE**: the self-built subset is **random noise images**;
DROID-SLAM finds no trackable structure in them → the dense-disparity point cloud for the whole 27 keyframes has only
**3191 points** (~118 per frame), frame 0's neighborhood only **38 points**, and frame 15/29 project to **0 inliers**;
`project_map`'s CUDA kd-tree hits `cudaErrorInvalidValue` on such degenerate input.

i.e., **synthetic noise data cannot be used to validate ViPE's dense-depth path**. ViPE's dense depth was previously
validated on **real video** (`dog-example.mp4`) (2.011–5.439 m, all finite; see D-l2-results).

### Code improvement made as a result

`scripts/modelenvs/run_vipe.py` previously **silently dropped the `depth` key** when depth was unavailable, and the
caller would only see a vague `Scene artifact missing keys: ['depth']`. Now it **fails loudly with the real reason**:

```
RuntimeError: ViPE could not produce per-frame dense depth (depth_source=unavailable: ...):
the SLAM point cloud is too sparse or projection has no inliers. Common for synthetic inputs
without real texture; switch to real video, or switch to the dense-depth postprocessing of
pipeline=default.
```

## 4. Honest Boundary (no overclaiming)

1. **The three-mode comparison uses stand-in models, not real models.** It measures the **scheduler's** behavior, not
   GPU compute utilization. Why: the real path must pass the HAND stage, and `MANO_RIGHT.pkl` requires a manual
   registration download, unavailable in this environment (see D-l2-results). **This L3 therefore does not constitute
   a "real-model end-to-end three-mode comparison".**
2. **No GPU-utilization curve was captured under real models.** Stand-ins do not touch the GPU; the `backend=real` run
   only captured instantaneous points (96% / 10.2GB), not enough for mean/peak.
3. **No HOT3D absolute-metric reproduction.** The self-built subset gives only **relative** timings between the three modes.
4. **`ba_residual=0.0` is ViPE-side `SLAMOutput`'s default (unfilled); no accuracy claim.**
5. The 1.54–1.58x ratios (adaptive vs serial) are measured **on this cluster, this input, these stand-in timing
   settings**; different input sizes or stage-time ratios will change them.

## 5. Reproduction Commands

```bash
# 0) sync code and data to the node's hostPath
rsync -a --exclude=.git --exclude=third_party src scripts configs tests deploy \
      pyproject.toml README.md .rayignore <node>:/home/ubuntu/ego/app/

# 1) start a KubeRay operator inside roboray that watches only this namespace (the platform's existing operators don't manage it)
bash deploy/k8s/setup_operator.sh          # see pitfall 1 in "Pitfalls fixed along the way"

# 2) submit the RayCluster
IMAGE=registry.example.com/admin/vllm:v0.26.0 NODE=<gpu-node-ip> bash deploy/k8s/apply.sh
kubectl -n roboray get pods -o wide

# 3) generate the unified input and config
docker run --rm --network host --entrypoint bash --security-opt seccomp=unconfined \
  -v /home/ubuntu/ego:/ego registry.example.com/admin/vllm:v0.26.0 /ego/gen_l3.sh

# 4) three-mode comparison (same session)
H=$(kubectl -n roboray get pod -l ray.io/node-type=head -o jsonpath="{.items[0].metadata.name}")
kubectl -n roboray exec $H -- bash -lc \
  "export ROBORAY_FAKE_SCENE_SECONDS=2.0 ROBORAY_FAKE_HAND_SECONDS=1.0; \
   bash /ego/app/deploy/run_l3.sh"

# 5) GPU idle check / teardown
bash deploy/k8s/status.sh
bash deploy/k8s/teardown.sh
```

## 6. Status Determination

Per the three-state definition of Plan D Task 6, this item is **partially executed**:

- ✅ the three-mode comparison ran on a **real Ray cluster** and reproduced over two rounds (numbers in section 2);
- ✅ ViPE validated on real video (L2, see D-l2-results); FFS validated on real video (L2);
- ⛔ **real-model end-to-end three-mode comparison not completed**: blocked by HAND's `MANO_RIGHT.pkl`
  (requires manual registration download at https://mano.is.tue.mpg.de), not a code problem.
  Once the file is obtained, point `/ego/l3-real.yaml`'s `models.hammer_weights` at its parent directory and re-run
  the section-4 commands.

---

## 7. Real-Model End-to-End (2026-10-20, redone after MANO arrived)

After HaMeR was unblocked, re-ran `backend=real`, input switched to **real video**
(first 8 frames of `vipe/assets/examples/dog-example.mp4` x 2 clips, monocular → `resolve_scene_model` picks ViPE):

| Stage | Result | time (per clip) |
| --- | --- | --- |
| prepare | ✅ done | 0.06 / 0.07 s |
| **scene (ViPE real weights)** | ✅ **done** | **16.34 / 16.64 s** |
| **hand (HaMeR real weights)** | ✅ **done** | **12.39 / 10.21 s** |
| fusion | ✅ done | 0.04 / 0.04 s |
| action | ❌ `VlmError: VLM not configured: set action_understanding.vlm_base_url` | — |
| publish | ⏳ pending | — |

During execution the GPU worker sampled `utilization.gpu=96%, memory.used=10225 MiB` (real models running).

**i.e., the first four of the stages (prepare/scene/hand/fusion) ran through on a real Ray cluster with real
models**; the only one not passing is action — because **no reachable VLM service exists**: the cluster does have vLLM
(`qwen38-flash-next`, LoadBalancer `<management-node-ip>:28100`), but Pods in the `roboray` namespace **cannot
connect** (`curl` returns 000). This is network isolation, not a code problem.

### Two real bugs exposed and fixed this round on the real cluster

1. **`CpuStageActor` does not register builtin models** (`src/roboray/ray_runtime/actors.py`).
   A Ray worker is a separate process; the driver-side `load_builtin_models()` does not propagate;
   `GpuModelActor.load_model` registers on its own, but the CPU actor does not → the action stage's VLM inevitably
   `UnknownModelError: unregistered semantic model 'openai_vlm'; registered: []`.
   This path had only been exercised under `backend=test` (registered by `worker_process_setup_hook` with stand-ins),
   **so local tests never exposed it**. Fixed, with a regression test
   `tests/unit/test_actors.py::test_cpu_actor_registers_builtin_models_for_action_stage`
   (mutation-verified: removing the fix makes the test fail). Also added a read-only diagnostic method
   `registered_models()` to the actor — to debug "model not found in worker", ask the worker itself.

2. **`depth_source` enum out of range** (`scripts/modelenvs/run_vipe.py`).
   The ViPE adapter rewrite set `depth_source` to `"monocular_slam_map_keyframe"`, but the contract enum only allows
   `"monocular"`/`"stereo"`, so the scene stage failed at validation with `ModelEnvError: illegal depth_source`.
   Reverted to the enum value; the fine-grained source moved into the summary's `depth_detail`.

### Honest boundary (updated)

- The real-model **action stage is still unverified** (no reachable VLM service); error branches and the keyframe
  chain are covered by unit tests.
- This real-model run **only ran the adaptive mode**, not a three-mode comparison. The three-mode comparison remains
  the **stand-in-model** result in section 2 (measures the scheduler, not compute).
- The input is ViPE's bundled example video, **not ego first-person data**; calibration K comes from ViPE L2's measured
  estimate (fx≈984.26), explicitly marked as an assumed value in the config, used only for HaMeR's metric scale.

---

## 8. Final Result: real-model six-stage full pass + resident-worker gain measured (2026-10-25, real A800 cluster)

The complete runtime delivered by plans E1/E2/E3 was validated end to end on a real A800 cluster
(standalone RayCluster + execution inside head; the RayJob submission mechanism was validated separately, below).

### Result

Input: first 8 frames of dog-example x 2 clips (monocular → ViPE path), `action_understanding.model: "none"`
(no reachable VLM; labels via deterministic fallback), `model_worker: resident`.

| Episode | Stage results | Times |
|---|---|---|
| ep_0001 | **all six stages done** | prepare 0.05 / **scene 15.74 (incl. ViPE load)** / **hand 11.60 (incl. HaMeR load)** / fusion 0.05 / action 0.0 / **publish 9.17** |
| ep_0002 | **all six stages done** | **scene 2.57 / hand 0.66** / publish 9.03 |

**failed = 0, CLI_EXIT = 0.** All artifacts present: real LeRobot v3.0 datasets x2 (fps 30, 8 frames, readable back via
`LeRobotDataset`), camera/world visualizations x4, QC reports (report.html + reprojection.png
+ segments.json) x2, manifest.json.

### Resident-worker gain measured (real models, first numbers)

| Stage | ep_0001 (incl. load) | ep_0002 (reused resident worker) | Gain |
|---|---|---|---|
| scene (ViPE) | 15.74 s | **2.57 s** | **6.1x** |
| hand (HaMeR) | 11.60 s | **0.66 s** | **17.6x** |

The scheduler snapshots `loaded: None → 'vipe' → 'hamer'` record the role switch and model-residency chain; the
run-end teardown unload (E2 Task 5) returns the memory (GPU sampling: 14.5 GiB during inference → 0 after teardown).

### RayJob submission mechanism validation (separate round)

RayJob(K8sJobMode) on a real cluster: submission accepted → cluster auto-created (random suffix name, confirming E3's
wording correction) → head up → job ran to completion → SUCCEEDED → `shutdownAfterJobFinishes` tore the cluster down.
**Confirmed pitfalls** (all fixed into the template and pinned by structural tests):

1. `submitterPodTemplate` lacking `restartPolicy: Never` → the submitted Job is judged invalid by K8s, and the RayJob
   stays Initializing forever (the operator inherits only the image; the rest merged from the default template gets
   overridden by the custom template).
2. The submission client's 2Gi memory is too small (the driver process runs inside it) → OOMKill; raised to 4Gi/8Gi.
3. **In K8sJobMode, `ray.init` inside the job pod without a cluster address starts a local mini-cluster**
   (RAY_ADDRESS not injected into the job pod) — the workload lands in the 8Gi submit pod instead of the cluster's GPU
   workers. E3's final validation therefore used a standalone cluster + execution inside head (the E2-validated mode).
   For the RayJob path to actually put the workload on the cluster, the head address must be explicitly injected into
   the entrypoint (`<cluster>-head-svc` random suffix must be read from status) — registered as an E3 follow-up.

### Honest boundary (added)

- action stage used `model: "none"` (no reachable VLM service) — action semantics go through the deterministic fallback
  labels, not VLM-generated.
- Input is short clips of 8 frames x 2 — the validation target is "six-stage closed loop + existence of the resident
  gain", not a throughput benchmark.
- The resident-gain multiples (6.1x/17.6x) are measured **on this input, this cluster, single worker** and vary with input scale.
