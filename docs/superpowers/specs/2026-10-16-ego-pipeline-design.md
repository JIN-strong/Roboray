# Ego Video to Training-Ready Data Production Pipeline - Design Document

- Date: 2026-10-16
- Status: pending review
- Source: reproduction of the article "From Raw Ego Video to Training-Ready: Accelerating Embodied Data Production Pipelines with Ray on EMR Serverless Spark" (Ray Chinese community reprint of Alibaba Cloud, 2026-10-16)

## 1. Goals and Non-Goals

### Goals

Reproduce the **engineering system** of the embodied-data production pipeline described in the article:

1. Four-stage pipeline: unified ingestion and data preparation → Scene/Hand dual-path reconstruction and fusion → action understanding and training-data generation → visualization, publishing and QC.
2. Ray execution model: GPU actor pooling (roles switchable), cross-stage pipeline parallelism (EP01 doing ACTION while EP03 is in PREPARE), adaptive model scheduling (4-signal dispatch + measured-latency EMA feedback loop).
3. Public API consistent with the article: `from roboray import EgoReconstruction, ReconstructionRequest`; `ReconstructionRequest.from_dict(config)` runs the default config in one line, each stage independently tunable.
4. Output: standard LeRobot v3 dataset + dual-view visualization mp4 + per-frame traceable QC report HTML.
5. End-to-end validation with real models on the A800 K8s cluster (KubeRay).

### Non-Goals

- Do not reproduce EMR Serverless / OSS proprietary integrations (deployment target is a generic Ray cluster / KubeRay).
- Do not implement the article's section 09 outlook (domain LLM QC, Studio UI).
- Do not chase the article's absolute metric numbers (MPJPE 28.47mm etc. depend on the HOT3D official evaluation protocol and a full, unpublished algorithm stack); L3 validation focuses on **form reproduction** (two-level speedup, parallelism gain) and relative improvement on a self-built evaluation subset.

## 2. Key Decision Records

| # | Decision | Choice | Rationale |
|---|------|------|------|
| D1 | Model strategy | **real model code**: product code directly integrates ViPE / Fast-FoundationStereo / HaMeR(+MANO) | user explicitly chose; GPU environment already available (A800) |
| D2 | Scope | full reproduction (4 stages + scheduling + visualization + QC) | user choice |
| D3 | Target environment | generic Ray cluster; concretely deployed as **KubeRay** on the internal K8s; **Ray 2.59.0 latest stable** (consistent in and out of the image) | user-specified; verified the cluster has a KubeRay operator |
| D4 | Test strategy | product code only calls real models; **pytest uses fake model stand-ins** to validate orchestration | user confirmed; orchestration logic must be locally verifiable |
| D5 | Input format | simplified ego layout: `rgb_left.mp4 + rgb_right.mp4(optional) + calibration.json + meta.json` | easy to synthesize test data; VRS conversion tool not built (non-goal) |
| D6 | Action semantics | pluggable VLM API (OpenAI-compatible); fakes in tests; if unreachable, stage 3 errors out, no geometric degradation | user choice |
| D7 | Hand model | HandFlow paper has no public code → **HaMeR** (Meta, outputs MANO params) as reference implementation, interface kept replaceable | verified HandFlow has no repo |
| D8 | Code management | git-managed; one canonical commit per implementation phase | user requirement |

## 3. Overall Architecture

```
┌─ Driver (ray job) ──────────────────────────────────────────────┐
│  EgoReconstruction(request).run()                               │
│  ┌────────────┐   ┌──────────────────────────────────────────┐  │
│  │ Ingest pool │   │ AdaptiveScheduler (driver-side event loop) │  │
│  │ (CPU Actor) │   │ signals: queue len/EMA latency/resources/loaded models │  │
│  └─────┬──────┘   │ feedback loop: measured latency→EMA→role reassignment  │  │
│        ▼          └───────┬──────────────┬───────────────────┘  │
│   Episode local workspace  GPU Actor×N      CPU Actor pool       │
│   media/ npz/         roles switchable    FUSION/ACTION/         │
│   (reused on-node)    SCENE or HAND       EXPORT/PUBLISH         │
└─────────────────────────────────────────────────────────────────┘
  Episode state machine: PREPARE → SCENE ∥ HAND → FUSION → ACTION → PUBLISH
```

Core mechanisms (article sections 04/05/06):

- **GPU actor roles switchable**: each GPU actor can load the Scene or Hand model; tasks are assigned dynamically by AdaptiveScheduler, no fixed-proportion pool.
- **Release on write**: after the model-inference result is written to NPZ the actor returns to the idle pool; fusion/postprocessing is taken over by the CPU pool, so the next Episode's inference can overlap immediately.
- **Local data reuse**: the same Episode's media, synced frames and intermediate arrays stay in the node-local workspace; Scene and Hand arrays are fused on the same node, avoiding large-array cross-node transfer (the scheduler prefers the same node when dispatching).
- **Feedback loop**: every measured service latency (model load and inference timed separately) flows back to the scheduler, updating the (model, resolution tier)-granularity EMA prediction; after several Episodes the work-split ratios converge to the current data/hardware state.

## 4. Package Structure

```
robot/
├── pyproject.toml            # pins ray==2.59.0 etc.; python 3.10
├── configs/                  # default.yaml / a800.yaml / local-fake.yaml
├── src/roboray/
│   ├── api/                  # EgoReconstruction, ReconstructionRequest, config schema
│   ├── ingest/               # stage1: data discovery/decoding/timeline alignment/calibration-structure check
│   ├── scene/                # ViPE adapter, Fast-FoundationStereo adapter, stereo-selection logic
│   ├── hand/                 # HandModel adapter(HaMeR), MANO wrapper, A2 wearer association, A3 short-gap completion
│   ├── fusion/               # A1 metric-space fusion + pose refinement (CPU-heavy)
│   ├── action/               # action timeline segmentation + AiFunction(VLM) + state/action tensors
│   ├── export/               # LeRobot v3 dataset write-out + manifest
│   ├── viz/                  # camera-view/world-view offscreen rendering + QC report HTML
│   ├── scheduling/           # AdaptiveScheduler, cost EMA, queues, slot table
│   └── runtime/              # Ray driver orchestration, Episode state machine, workspace management, storage abstraction (posix)
├── tests/
│   ├── fakes/                # FakeSceneModel, FakeHandModel, FakeVLM (isomorphic to real interfaces)
│   ├── unit/                 # L0
│   └── e2e/                  # L1
├── scripts/                  # synthetic ego-data generator, L2/L3 smoke scripts, metric evaluation
├── deploy/                   # Dockerfile, RayCluster CR, registry push and deploy scripts
└── third_party/              # vendored: vipe, fast-foundationstereo, hamer (commit-pinned)
```

## 5. Data Model

### Input Layout (D5)

```
dataset_root/
└── ep_0001/
    ├── rgb_left.mp4            # required, first-person view
    ├── rgb_right.mp4           # optional; presence enables stereo depth
    ├── calibration.json        # intrinsics K / distortion / left-right extrinsics / timestamp-basis fps
    └── meta.json               # episode id, capture device, optional description
```

### Intermediate Artifacts (per-Episode workspace, NPZ)

`prepared/` (synced frame table, normalized calibration) → `scene.npz` (camera poses, intrinsics, depth-array reference) → `hand.npz` (per-frame 2D detections, MANO params, confidence) → `fused.npz` (world-frame hand motion, reprojection error, depth source, completion audit) → `action.npz` (action intervals, semantic labels, state/action tensors)

NPZ is the resume-from-checkpoint unit: each stage persists to disk on completion and records state; a re-run resumes from the first incomplete stage.

### Output

```
output/run-xxx/
├── lerobot_v3/ep_0001/...      # standard LeRobotDataset
├── viz/ep_0001/{camera.mp4, world.mp4}
├── qc/ep_0001/report.html
└── manifest.json               # whole-batch index (per-Episode status and timings)
```

## 6. Model Adapter Layer

Unified protocol (real implementations and fakes are isomorphic):

```python
class SceneModel(Protocol):
    def load(self, device: str) -> None: ...
    def infer(self, frames, calib) -> SceneArtifact: ...
    def cost_hint(self, n_frames: int, resolution: tuple) -> float: ...

class HandModel(Protocol):  # load / infer → HandArtifact(MANO params+2D detections) / cost_hint
```

| Path | Model | Source | Use |
|---|------|------|------|
| Scene | ViPE | github.com/nv-tlabs/vipe | monocular → camera intrinsics/poses + near-metric depth (ViPE also supplies depth when no stereo) |
| Scene | Fast-FoundationStereo | github.com/nvlabs/fast-foundationstereo | stereo → disparity → metric depth (primary depth path when stereo present; `stereo_max_image_side_px` controls input resolution) |
| Hand | HaMeR | github.com/geopavlakos/HaMeR | hand detection + MANO param fitting (reference implementation, interface replaceable) |
| Hand | MANO | mano.is.tue.mpg.de | parametric hand model; requires registration download of `MANO_LEFT.pkl / MANO_RIGHT.pkl` (one-time manual step, stated in README and validated) |

Selection rule: if `rgb_right.mp4` exists → depth from Fast-FoundationStereo, poses from ViPE; otherwise everything goes through ViPE.

## 7. Algorithm Trio (article A1/A2/A3)

- **A1 metric fusion** (fusion/, CPU): Scene metric depth/camera poses × Hand MANO output jointly optimized in metric space (hand 3D-keypoint depth alignment + overall pose correction), outputting world-coordinate hand motion + per-frame reprojection error.
- **A2 wearer association** (hand/): frame-level binding of detected hands to wearer identity (based on image-position prior + motion continuity + MANO orientation consistency), keeping identity continuous over time, outputting identity-switch events for QC.
- **A3 short-gap completion** (hand/): interpolate only when (a) both ends have trusted observations (b) before/after judged the same hand (c) consecutive missing ≤ `unobserved_gap_s` (default 0.2s, i.e., ≤6 frames at 30fps); completed results must pass 4 checks: keypoint depth, in-view, wrist-displacement cap, trajectory-deviation cap; drop on any failure. **Prefer keeping the gap over introducing hallucination.** Params: `min_supporting_joints` (default 8).

## 8. Adaptive Scheduler (scheduling/)

- Single-threaded event loop in the driver (asyncio); state: `scene_q` / `hand_q` dual queues, GPU actor slot table (`loaded_model` / `in_flight` / latency history), CPU pool.
- Cost model: per (model, resolution tier) EMA; cold start uses the `cost_hint` prior.
- Dispatch policy: when an actor is idle, decide the next task by weighted backlog of the dual queues + resource constraints; same-model tasks preferentially dispatched to actors that already loaded that model; role switch requires unload first (unload latency recorded into EMA).
- Feedback loop: completed tasks report measured latency → update EMA → converge.
- Observability: scheduling decisions, queue depths, EMA values periodically persisted as JSON for L3 comparison experiments and QC.

## 9. Stage 3: Action Understanding (action/)

1. Action timeline: world-frame hand motion → velocity threshold + contact-detection geometric segmentation → candidate action intervals.
2. AiFunction: OpenAI-compatible VLM interface (`base_url`/`api_key`/`model` configurable), input keyframes + trajectory summary → action semantic labels. The cluster default points at the internal vLLM service; if unreachable, stage 3 errors explicitly.
3. Tensor generation: state (world-frame hand joint poses) + action (next-frame joint targets), encoded per the LeRobot feature spec into the `lerobot_v3` dataset.

## 10. Stage 4: Visualization / Publishing / QC (viz/ + export/)

- Camera-view mp4: original field of view / aspect ratio overlaid with 2D keypoints and MANO reprojection.
- World-view mp4: 3D-rendered hands + camera trail + downsampled point cloud + future 1s wrist trail; matplotlib offscreen rendering (works even when the container has no EGL).
- QC report HTML: per-Episode reprojection-error curve, support-point count, depth source, identity switches, completion audit, per-frame diagnostic timeline.
- Publishing: writes LeRobot v3 + viz + QC per episode in completion order; aggregates `manifest.json` once the whole batch reaches terminal state.

## 11. A800 Deployment (deploy/)

Verified cluster facts (probed 2026-10-16):

- K8s v1.22, 4 nodes; GPU nodes x2, each 2x NVIDIA A800 80GB PCIe (driver 580.178.04, Anolis 8.5).
- KubeRay operator deployed (existing platform namespaces); internal image registry at `registry.example.com/...` (with a P2P variant).
- Cluster nodes have **no external network**; this machine has external network (pypi / hf-mirror reachable).
- All 4 GPUs are currently occupied by pre-existing workloads (vLLM + shared RayCluster); this project's GPU Pods need a window or coordination with the workload owners.

Deployment flow:

1. Build the image on this machine (incl. vendored model repos + dependencies) → push to the internal registry via an SSH tunnel → `deploy/raycluster.yaml` (ns `roboray`, head CPU + N GPU workers, each worker `nvidia.com/gpu: 1`).
2. Data upload: upload helper Pod + kubectl cp.
3. Job submission: `ray job submission` (port-forward head:8265).
4. MANO weights and ego data on hostPath/PVC mounts.

## 12. Validation Strategy (4 layers)

| Layer | Environment | Content |
|---|------|------|
| L0 unit | local CPU + fakes | alignment/calibration, A2/A3 boundaries, scheduler convergence and ordering, LeRobot validity, config parsing |
| L1 local e2e | local CPU + fakes + synthetic data | full pipeline, complete artifacts, resume-from-checkpoint, QC report, manifest |
| L2 model smoke | A800 | each real adapter independently infers on a small sample |
| L3 full pipeline | A800 | 3-5 Episode batch; compare serial / pipeline-parallel / +adaptive-scheduling three-mode timings; A1-A3 relative improvement on the self-built evaluation subset |

Synthetic data generator (scripts/): programmatically produces ego videos with known ground truth (hand trajectories, depth) for L1/L3.

## 13. Dependency Pinning (pyproject)

- python 3.10 (HaMeR/pytorch3d compatible)
- ray==2.59.0 (consistent in and out of the image; local machine gets a new conda env `roboray`, replacing the current 3.0.0.dev0)
- torch + cu124 (A800 sm80)
- opencv-python / av (decoding), numpy, lerobot (v3), manopth, openai client, matplotlib, jinja2, pytest
- third_party vendored: vipe / fast-foundationstereo / hamer, commit-pinned

## 14. Risks and Mitigations

| Risk | Mitigation |
|------|------|
| HandFlow has no public code (D7) | HaMeR as reference implementation; HandModel interface abstracted, swappable later |
| All GPUs currently occupied | L2/L3 wait for a window or coordinate; L0/L1 unaffected; CR pre-created as Pending |
| Cluster has no external network | image built locally and pushed to the registry; weights/data scp + kubectl cp |
| pytorch3d compile complexity (HaMeR dependency) | precompiled wheels in the image or conda packages; build script validated separately |
| ViPE/FFS/HaMeR APIs differ from expectations (before GPU measurement) | adapter-layer isolation; L2 smoke validates each model first before L3 |
| lerobot v3 format detail drift | pin lerobot version; export uses its public API rather than hand-written files |
