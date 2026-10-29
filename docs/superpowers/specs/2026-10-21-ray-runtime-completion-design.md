# Ray Runtime Feature Completion - Design Document

Date: 2026-10-21  Branch: main  Prerequisite: `2026-10-16-ego-pipeline-design.md`

## 1. Goals and Non-Goals

### Goals

Make the three Ray features the article describes **live up to their names**: the elastic resource model, the actor execution model (incl. resource release), and the local-reuse data model.

This document is a **completion spec**, not new requirements — the ledger in section 2 lists each item as "originally claimed by the spec / current implementation state", disposing of every item that was unimplemented or had distorted numbers.

### Non-Goals

- No algorithm-layer changes (A1/A2/A3 and the four-stage semantics stay as they are)
- No shared-workspace backend (node-local + affinity is the decided approach, see D9)
- No real multi-node cluster validation (see D13 and section 10)

## 2. Gap Ledger (spec claims vs implementation state)

| # | Original spec wording | Location | Implementation state | This round's disposition |
|---|---|---|---|---|
| G1 | "**Release on write**: after the model-inference result is written to NPZ the actor returns to the idle pool" | design doc §3 | **no `unload()`**; and **models never stay resident** — `run_in_modelenv` restarts a subprocess and reloads weights on every `subprocess.run` | make it real (section 5) |
| G2 | "**Local data reuse**… avoid large-array cross-node transfer" | design doc §3 | frame arrays are passed through the Ray object store (`build_scene_payload` does `np.load` on the driver side); hostPath does not cross nodes, **multi-node crashes** | make it real (section 4) |
| G3 | "The scheduler **prefers the same node** when dispatching" | design doc §3 | not implemented (not observable under single-node deployment) | make it real (section 4) |
| G4 | "Job submission: **`ray job submission`**" | design doc §11 | everything goes through `kubectl exec` into head to run the CLI | make it real (section 6) |
| G5 | "A Ray cluster that stays resident and **supports elastic scaling**" | design doc §3/§11 | no `enableInTreeAutoscaling`; `replicas/min/max` all hard-coded to 1 | make it real (section 6) |
| G6 | "Model-reuse hits (`needs_load=false`) **4/6 = 66.7%**, no repeated-loading overhead" | `A-verification-report.md` | **number distorted**: `needs_load` only indicates whether the adapter **object** is rebuilt, unrelated to GPU loading (`load()` only does an interpreter/path-existence check) | **invalidate the number**, note it in the A report and this spec |

## 3. Key Decision Records

| # | Decision | Choice | Rationale |
|---|------|------|------|
| D9 | Multi-node support approach | **node-local workspace + `NodeAffinitySchedulingStrategy`** | closest to the article's original "stays in the node-local workspace" intent; real 1080p long videos (~2GB) are not repeatedly sent over the network. Rejected the shared-POSIX-workspace approach (every stage read/write goes over the network and it contradicts the article's wording) |
| D10 | Actor release policy | **three-tier idle**: unload model 30s / actor exit 240s / node reclamation 300s | "resident actor" and "node elastic reclamation" are naturally at odds — Ray does not reclaim nodes with live actors. Tiers return resources layer by layer |
| D11 | Model residency implementation | each of the three model envs starts a **resident worker subprocess** + JSON-lines IPC | the three models' CUDA versions are mutually exclusive, so separate envs are required (original D1); for "resident/release" to hold, the subprocesses must stay alive |
| D12 | Elasticity scope | `enableInTreeAutoscaling` + `RayJob(K8sJobMode)` + CPU/GPU worker groups + **head `num-cpus: 0`** | user-specified; head carries no pipeline work |
| D13 | Validation boundary | **local unit tests complete (incl. fakes); no real-cluster multi-node validation** | user said explicitly "you don't have to actually test, but the code must be functionally complete" |

## 4. Data Plane (section 1)

### 4.1 Core Principle

**The driver process never touches any file in the workspace.** It only orchestrates, keeps the ledger, and writes its own small state (`state.json` / `manifest.json`).

Rationale: the workspace is node-local. The driver runs on head or a CPU worker, and **cannot even read those files**. So this is not a performance optimization; it is a precondition for multi-node.

### 4.2 Payload construction pushed down

| Stage | Current | After |
|---|---|---|
| GPU (scene / hand) | driver-side `build_scene_payload()` reads `frames_left.npy` into an array packed into the payload | payload **carries only** `workspace_root` / `episode_id` / `kind` / `model_name` / `out_path` / `stereo_max_image_side_px`; **the actor reads from disk on its own node** |
| CPU (prepare/fusion/action/publish) | already passes paths (`planner.build_cpu_kwargs`) | unchanged |

### 4.3 The last two driver-side disk reads removed

- `prepared_resolution_key()` currently reads `prepared/calib.json` → instead carried back by **the prepare stage's return value** (`run_prepare` adds `width` / `height` to its return), recorded by the driver in memory and in `state.json`
- the `calib.json` read inside `_gpu_payload()` disappears with it

After this, the driver's file access to the workspace is **zero** — backed by an executable invariant test (section 10).

### 4.4 Node pools and Episode placement

**Created on demand, not all at startup** (otherwise the autoscaler receives no scale-up signal):

1. Episode placement priority: **the node that already has this Episode's workspace** (resume-from-checkpoint is naturally correct) > the node with the fewest in-flight Episodes
2. No usable slot → **create a new GPU actor (without affinity)** → Ray finds the resource request unsatisfiable → autoscaler brings up a new node → the actor lands on the new node
3. the actor **reports its own node** via `ray.get_runtime_context().get_node_id()`, recorded by the driver as `assigned_node`
4. that Episode's CPU stages create/reuse a CPU actor on the **same node** (with `NodeAffinitySchedulingStrategy(node_id, soft=False)`)

**Reuse criterion (made explicit)**: the driver maintains a `node_id → NodePool` map; each `NodePool` = 1 GPU actor + `cpu_actors_per_node` CPU actors (new config item, default 2), **all carrying that node's `NodeAffinitySchedulingStrategy`**.

- Placement: reuse only when the roles needed by that Episode's stages can be served by actors in the pool; otherwise create new
- Conditions for creating a new pool: all existing pools are full (a pool's GPU actor has in-flight Episodes ≥ 1, or all CPU actors are busy)
- A new GPU actor is created **without affinity**; placement decided by Ray + autoscaler; then top up that pool's CPU actors per its `node_id`

### 4.5 Node failure = full re-run

The workspace disappears with the node, so **a RUNNING episode cannot simply be pushed back to PENDING** (that would read a nonexistent `prepared/`). `reconcile_states()` gains a rule: if `state.json`'s `assigned_node` is not in the current node list, or the workspace on that node no longer exists → **discard all of that Episode's stage states and start over from PREPARE**.

### 4.6 Storage split into three layers

| Role | Where | Rationale |
|---|---|---|
| `input.uri` (raw dataset) | shared volume | prepare must be able to read the media from any node |
| `workspace_root` | node-local hostPath | large arrays read/written locally, pinned by node affinity |
| `output.uri` (LeRobot/viz/QC) | shared volume | the user picks up finished artifacts from the shared volume |

## 5. Actor Lifecycle (section 2)

### 5.1 Current-state correction

`grep -rn "Popen|persistent" src/roboray/` is empty; `_SceneBase.load()` is a no-op `self.loaded_on = device`; `VipeSceneModel.load()` only does an interpreter-existence check; `run_in_modelenv` does `subprocess.run` every time.

**Conclusion: models never stay resident; every inference restarts a process and reloads the weights, so GPU memory was already returned to zero between calls.** "Release" previously had no object to release.

### 5.2 IPC protocol

JSON-lines over stdin/stdout, reusing the existing `__RESULT__{json}` marker.

```
→ {"id":7,"cmd":"infer","input":"/p/in.npz","output":"/p/out.npz","weights":"...","extra":"{...}"}
← __RESULT__{"id":7,"ok":true,"model":"vipe","n_frames":40,...}

→ {"id":8,"cmd":"ping"}       ← __RESULT__{"id":8,"ok":true,"idle_s":12.3}
→ {"id":9,"cmd":"shutdown"}   ← process exits
```

A failed inference only replies `{"id":N,"ok":false,"error":...}`, **killing nothing** — one bad input should not discard an already-loaded model.

### 5.3 Script split into three sections

`run_vipe.py` / `run_ffs.py` / `run_hamer.py` each split into:

```
build_model(weights, extra) -> model      # load weights; done only once in serve mode
infer_once(model, in_path, out_path)      # existing inference body moved as-is
main()                                    # one-shot CLI (kept compatible: L2 smoke and existing tests use it)
serve()                                   # --serve: loop reading stdin
```

### 5.4 `ModelEnvSession`

```python
class ModelEnvSession:
    def start(self) -> None                  # Popen([python, script, "--serve", ...])
                                             # ready only when __RESULT__{"ready":true} arrives
    def infer(self, in_npz, out_npz, timeout_s) -> dict
    def ping(self) -> bool
    @property
    def alive(self) -> bool                  # poll() is None
    def close(self) -> None                  # close stdin → wait for exit → kill process group as fallback
```

A dead subprocess is **not silently retried for the same request**: mark it dead, restart and reload on the next request.

### 5.5 `GpuModelActor` changes

- `load_model()` now creates and `start()`s a session — **`load_seconds` is only now truly the "load weights" time** (previously it measured a `--version` subprocess, meaningless)
- new `unload() -> dict`: close the session, clear `loaded_kind`/`model_name`, return the unloaded role and idle duration
- `run_task` records `last_used` for idle determination
- new `idle_seconds()`, `node_id()`

### 5.6 Three-tier idle policy (mechanism in actor, policy in driver)

| Threshold | Action | Returns |
|---|---|---|
| `actor_idle_unload_s` = 30s | `unload()` | GPU memory (actor still alive) |
| `actor_idle_exit_s` = 240s | **driver `unload()` first, then `ray.kill(actor)`**, and removes it from the pool | the actor's occupancy on the node |
| RayCluster `idleTimeoutSeconds` = 300s | autoscaler reclaims | the whole machine |

**Must be increasing and 240 < 300**, otherwise the node never gets to see the actor exit.

**Why `unload()` must come before `ray.kill()`**: `ray.kill` SIGKILLs the actor process, **running no cleanup**, and the resident subprocess becomes an orphan that keeps holding GPU memory. So the exit path must be "gracefully close the session first, then kill the actor", not just `ray.kill`.

Each driver event-loop pass scans idle actors; the `ray.wait` timeout is set to "time until the nearest unload comes due" to implement timeout wakeup. Unload events are written into `timeline` and `scheduler_snapshots`. At run end, `unload()` is called on all actors — otherwise the Ray actor stays alive, the subprocess hangs, and GPU memory is not returned.

### 5.7 Compatibility switch

`execution.model_worker: "resident" | "oneshot"`, default `resident`; `oneshot` keeps the original `run_in_modelenv` path, so the **existing 211 tests are unaffected**.

## 6. Elastic Resource Model (section 3)

### 6.1 RayCluster

```yaml
spec:
  enableInTreeAutoscaling: true
  autoscalerOptions:
    idleTimeoutSeconds: 300
  headGroupSpec:
    rayStartParams: {num-cpus: "0", dashboard-host: "0.0.0.0"}
  workerGroupSpecs:
  - groupName: gpu-workers
    replicas: 0; minReplicas: 0; maxReplicas: N
    rayStartParams: {num-gpus: "1"}
  - groupName: cpu-workers
    replicas: 0; minReplicas: 0; maxReplicas: M
    rayStartParams: {num-cpus: "8", num-gpus: "0"}
```

**Starting from 0** (not 1) gives the autoscaler something real to do. head `num-cpus: 0` ensures no pipeline work ever lands on head.

### 6.2 RayJob

```yaml
kind: RayJob
spec:
  entrypoint: cd /ego/app && python -m roboray.cli run --config /ego/configs/job.yaml
  submissionMode: K8sJobMode          # the operator starts a separate K8s Job pod as the submission client
  shutdownAfterJobFinishes: true
  ttlSecondsAfterFinished: 3600
  rayClusterSpec: {…isomorphic to the standalone RayCluster…}
```

**The CLI already works inside a Job**: `run()` contains `if not ray.is_initialized(): ray.init(...)`, so inside a Ray Job where Ray is already initialized it just skips. No change needed.

### 6.3 CPU / GPU division of labor (correction to the previous claim)

Section 4 decided "all of an Episode's stages must be pinned to the GPU node holding its workspace", so **CPU actors can only hang off GPU nodes** and cannot move to separate CPU nodes. The real duties of the CPU worker group:

1. **Host the driver** (the K8s Job pod started by `K8sJobMode` is it) — the correct answer to "don't crowd head"
2. host **CPU work that does not need an Episode workspace** (none currently; kept as an extension point)

> **Correction 2026-10-24**: the K8s Job pod is the **submission client** (runs `ray job submit`); the Ray driver process,
> per Ray's design, **runs inside the head container**. The above item 1 "host the driver" is superseded by this
> correction; the cpu-workers extension-point semantics are unchanged.

## 7. Deployment Artifacts (section 4)

| File | Change |
|---|---|
| `deploy/k8s/raycluster.yaml` | head `num-cpus: 0`; workers in two groups; autoscaling on; `replicas` starting from 0 |
| `deploy/k8s/rayjob.yaml` | added |
| `deploy/k8s/submit_job.sh` | added: render CR and submit |
| `deploy/k8s/status.sh` | extended: print node pools, autoscaler state, idle-unload event count (keep the original `TOTAL_FREE_GPUS` check) |
| `deploy/k8s/apply.sh` / `teardown.sh` | add RayJob support |

Mounts: `input` and `output` on the shared volume, `workspace` on hostPath (see 4.6).

> **Correction 2026-10-24**: the "idle-unload event count" on the status.sh row above **was not implemented** — it is a
> driver-side run artifact and must be queried from the run's timeline/manifest (`scheduler_snapshots`/`timeline` in
> `manifest.json`); `status.sh` only reads k8s, not the workspace, so it is registered but not output (consistent with
> the header note of `deploy/k8s/status.sh`).

## 8. Config Items

| Key | Default | Meaning |
|---|---|---|
| `execution.model_worker` | `"resident"` | `resident` keeps resident workers; `oneshot` starts a process each time (for tests) |
| `execution.actor_idle_unload_s` | `30.0` | how long idle before unloading the model; `0` = unload only on role switch and at teardown |
| `execution.actor_idle_exit_s` | `240.0` | how long idle before the actor exits; `0` = never exit on its own |
| `execution.node_wait_timeout_s` | `600.0` | upper bound for waiting on a node/slot under `soft=False`; timeout counts as failure, not silent hang |
| `execution.cpu_actors_per_node` | `2` | number of CPU actors per node pool (same pool and affinity as the 1 GPU actor) |

## 9. Validation Strategy (D13)

### 9.1 Must do (locally verifiable, into `pytest`)

| Item | Means |
|---|---|
| `ModelEnvSession` protocol | fake worker script (a python echo service), test `start/infer/ping/close/crash-restart` |
| three-tier idle | fake clock; test increasing thresholds, `unload` trigger, actor-exit trigger |
| node placement | mock `ray.nodes()`; test "prefer existing workspace → lightest load → create on demand" |
| **driver never touches the workspace** (hard invariant) | monkeypatch `Path.read_bytes` / `np.load`; assert the driver process accesses `workspace_root` **0** times across the whole run |
| CR rendering | rendered CR parses via `kubectl --dry-run=client`. **Correction 2026-10-24**: no CRD schema locally, so client dry-run has no validating power over CRs; the actual delivery is python rendering-structure assertions (more portable) |
| **Regression** | the `model_worker: oneshot` path stays; **all existing 211 tests must stay green** |

### 9.2 Explicitly not done ("unverified" column; must not claim verified)

- real multi-node cluster run (needs >=2 GPU nodes)
  > **Status (2026-10-24, after E3 delivery)**: **still unverified**. E3 delivered only rendering-structure assertions
    (manifest rendering + `kubectl --dry-run` parsing); it was not run through on any real multi-node cluster.
- autoscaling really scaling up / really reclaiming
  > **Status (2026-10-24, after E3 delivery)**: **still unverified**. E3 only asserts **structure** in the rendered CR —
    `enableInTreeAutoscaling`/`idleTimeoutSeconds: 300`/worker `minReplicas: 0` etc. — the real-cluster
    scale-up/reclaim end-to-end behavior was not done.
- RayJob really submitting and running to completion
  > **Status (2026-10-24, after E3 delivery)**: **still unverified**. E3 only asserts the RayJob manifest's field
    structure — `K8sJobMode`/`shutdownAfterJobFinishes`/entrypoint etc. — the real-cluster end-to-end submit-and-run was not done.
- measured GPU-memory curve of resident subprocesses
  > **Status (2026-10-24, after E3 delivery)**: **still unverified**. No measured GPU-memory data on a real GPU.

## 10. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| resident subprocess GPU-memory leak (actor killed by Ray but subprocess not reclaimed) | actor adds a `__ray_terminate__` hook calling `close()`; session uses a process group, killing the whole group on exit |
| subprocess stdout buffering makes `__RESULT__` unreadable | `python -u` + explicit `flush()` |
| node dies → Episode workspace disappears | `reconcile_states()` detects an invalid `assigned_node` → discard all stage states and re-run from PREPARE (4.5) |
| under `soft=False`, nodes full → task waits forever | `node_wait_timeout_s` timeout counts as failure, no silent hang |
| autoscaling vs node-affinity interaction | actor created **without affinity** (lands on a newly scaled node); only subsequent tasks carry affinity (4.4) |
| resident models fill GPU memory | the first tier of the three-tier idle (30s unload) catches it; on an A800's 80GB, all three models resident simultaneously is within capacity |

## 11. Acceptance Criteria

1. Local `pytest` all green, and every item in the 9.1 table has a corresponding test
2. behavior under `model_worker: oneshot` is exactly as before the change (regression)
3. driver file-access count on the workspace is 0 (9.1 hard invariant)
4. CR manifests parse via `kubectl --dry-run=client`
5. the "unverified" items in section 9.2 of this spec are **each marked unverified in the delivery report**, never written as verified

## 12. Suggested Implementation Split

The four sections of this document naturally form three relatively independent workstreams; **suggest splitting into three implementation plans** (order = dependency order):

| Plan | Content | Why independent |
|---|---|---|
| E1 | data plane (section 4): thin driver, payload pushed down, node pools and placement, node-failure re-run | the other two build on "workspace already node-localized" |
| E2 | actor lifecycle (section 5): resident workers + IPC + three-tier idle | depends on E1's actor-pool structure |
| E3 | elastic resource model and deployment artifacts (sections 6, 7): autoscaling, RayJob, CR changes | depends on E1's node-pool semantics; coupled to E2 only through config items |

## 13. Relationship to the Prerequisite Spec

This document does **not overturn** the algorithm layer or the four-stage semantics of `2026-10-16-ego-pipeline-design.md`; it only completes the runtime features that §3 and §11 of that spec claimed but did not implement. After the implementation lands, the G1–G5 wording in the original spec becomes true; G6's number is invalidated and must be noted as corrected in `A-verification-report.md`.
