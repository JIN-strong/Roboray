# Real-Model Three-Tier Benchmark Results (A800 real cluster, 2026-10-26)

- Cluster: `roboray` namespace, standalone RayCluster (not RayJob), head + on-demand GPU workers
- Node: `<gpu-node-ip>` (A800 GPU node 2), hostPath `/home/ubuntu/ego` → Pod `/ego`
- Image: `ego-hamer:l2` (platform vLLM image + `libegl1/libgl1`, required by HaMeR's pyrender/OpenGL)
- Input: `/ego/bench-real-data`, 6 clips × 60 frames (real example video slices, see §5)
- Config: `/ego/bench.yaml` (derived from the node's `/ego/l3-final.yaml`, only input/output URI + VLM branch changed)
- Models: ViPE (pose + monocular depth), HaMeR (hands), Fast-FoundationStereo (monocular input this batch; not triggered)
- VLM: platform qwen38 `Qwen3.8-Flash-Next` (`openai_vlm`, dummy key `EMPTY`)
- Execution: driver inside the head container connects `--ray-address auto`; the Ray autoscaler brings up 1–2 GPU workers on demand

## 1. Three-Tier Table (full `bench_report.md`)

| Tier | total_seconds | vs. serial | failed | episodes | Scheduler snapshots | Load events |
| --- | --- | --- | --- | --- | --- | --- |
| serial | 396.726856 | — | 0 | 6 | 0 | 12 |
| pipeline | 231.817723 | 1.7114 | 0 | 6 | 0 | 10 |
| adaptive | 145.506701 | 2.7265 | 0 | 6 | 12 | 2 |

action model: openai_vlm

Methodology alignment: serial↔902.499 / pipeline↔644.439 / adaptive↔536.550 (checklist doc); absolute values are not comparable — hardware and data differ (spec decision D-B3), only the three-tier ratios are comparable within this batch.

Raw artifacts: `bench_report.md` / `bench_report.json` (this directory).

## 2. Overlap Evidence

- Programmatic determination (`run-adaptive/timeline/timeline_evidence.json`):
  `max_concurrent_episodes=5`, `max_concurrent_stages=5`, `open_intervals=0`.
- Snapshot at peak `t=0.369s`: `ep_0001: prepare; ep_0002: prepare; ep_0003: prepare; ep_0005: prepare; ep_0004: scene`.
- Gantt: [timeline.png](timeline.png) (auto-rendered by `render_timeline` from the adaptive manifest).
- Interpretation: the 5-way concurrency is **batch-level overlap of the prepare (CPU) stage**, not 5-way GPU concurrency —
  this batch had only 2 free cards, GPU worker cap 2; `max_concurrent_episodes` counts "Episodes with at least one
  stage in flight at the same time". Adaptive mode does not set the `max_episodes_in_flight` gate (the scheduler queue
  balances concurrency), so all 6 clips can enter prepare together; the static pipeline tier is still gated by
  `max_episodes_in_flight=2`. The conclusion satisfies spec §9's `≥3`, but it **cannot** be read as 3+ GPU lanes in parallel.

## 3. VLM Branch Conclusion

- Probe (inside the head Pod, `scripts/probe_vlm.py`):
  `PROBE_RESULT=reachable_multimodal` (models / text chat / 8×8 PNG multimodal — all three levels pass).
- Endpoint: `http://<management-node-ip>:28100/v1` (platform qwen38-flash-next, ClusterIP:28100);
  model name: `Qwen3.8-Flash-Next`.
- Auth: the platform vLLM has no auth, so per spec decision D-B7 we use the dummy key `EMPTY`; the driver side sets
  `VLM_API_KEY=EMPTY` and it is propagated to workers via `runtime_env` (a fix item in this campaign, see §5).
- Branch: `bench.yaml`'s `action_understanding` is set to
  `{model: openai_vlm, vlm_base_url: http://<management-node-ip>:28100/v1, vlm_model: Qwen3.8-Flash-Next}`;
  all three tiers' `action` stages are `done`, ~3–9s per clip, and `bench_report` notes `action model: openai_vlm`.
- Platform service capacity/rate limits are outside this repo's control; no throttling observed this batch.
- **Freshness**: the ClusterIP/model name above reflect the platform side as of now; re-probe with
  `scripts/probe_vlm.py` before any rerun (the endpoint here is invalid once the platform migrates/renames).

## 4. Interference, Environment (incl. Load-Event Reading)

- GATE recheck: `TOTAL_GPUS=4`, `USED_GPUS=2` (platform qwen38-flash-next-varg on `<gpu-node-ip>`,
  a platform service, never touched), `TOTAL_FREE_GPUS=2`, `GATE=OPEN`.
- This campaign used the 2 free cards of `<gpu-node-ip>`; the Ray autoscaler brought up 1 GPU worker for the serial
  tier and up to 2 for pipeline/adaptive. GPU worker memory limit
  `32Gi → 96Gi` (the 4th clip of the 6-clip real-model batch once triggered a Ray memory kill; node physical memory
  is 502Gi, so it was a Pod limit, not a host shortage).
- Batch: 6 clips × 60 frames, 1280×720 each, 30fps; no timeouts, no clip dropping.
  1280×720 deviates from spec D-B2's 320×240 stereo (both resolution and stereo deviate).
- Load-event column (spec §3.3 cold-start evidence; all tiers >0):
  - serial = 12: the serial stage alternates between scene↔hand, each Episode triggers 2 model
    loads (scene and hand once each), 6 clips = 12;
  - pipeline = 10: the static pipeline cut some round trips, but slot role switching still triggers loads;
  - adaptive = 2: the scheduler reuses loaded models by "load cost vs. queue backlog", and the two GPU
    slots each load only once — fewest load events.
  - Count source: the driver appends `event="load"` to the manifest `timeline` when a GPU task has
    `was_load=True`; `run_benchmark.load_run` counts by event (serial/static pipeline have no scheduler
    snapshots, so they cannot use the snapshot `was_load` count).
- Platform interference: the qwen38 cards and this batch's 2 free cards are not on the same node; VLM calls reach
  the platform LoadBalancer/ClusterIP over the cluster network, no host-side preemption.

## 5. Honest Boundaries

Spec §9's four items (quoted verbatim):

- Synthetic data (generated by `make_synthetic_ego`), not HOT3D (this batch already deviates: real video slices, see above);
  absolute times are not compared against the article (D-B3),
  ratios depend on the batch and hardware and are only comparable within this batch.
- The serial tier is also real models and real export (not a stub) — it is a baseline, not a strawman.
- `max_concurrent_episodes ≥ 3` depends on the 6-clip batch and the scheduling window; if measurement only reaches 2,
  record the actual value honestly and analyze (CPU actor count / max_episodes_in_flight are candidate bottlenecks),
  do not tune parameters to fake the number.
- The platform vLLM is a platform service; this spec only does client-side integration; its service-side
  capacity/rate limits are out of scope.

Additional boundaries from this run:

- **Synthetic data rejected by the real ViPE**: after generating 6×60 with `make_l3_dataset` per the briefing,
  ViPE reported "cannot produce per-frame dense depth (SLAM point cloud too sparse / project_map
  cudaErrorInvalidValue)" on all 6 clips. The input was per-frame random noise + green dots; DROID-SLAM has no
  trackable structure; `run_vipe.py` explicitly lists this case as "please use real video".
- The campaign therefore used **real example video** slices already on the node: dog-example 122 frames sliced
  into 3 (0–59 / 30–89 / 62–121), cosmos-example 81 frames into 2 (0–59 / 21–80),
  intro_c as 1 (100–159, scaled to 1280×720), 6×60 frames total. Still **not HOT3D**,
  not the paper's data; and **monocular** (only `rgb_left`), so the FFS stereo depth lane was not triggered.
  Absolute times are not comparable with the article; only the three-tier ratios are comparable within this batch.
- **Concurrency semantics**: `max_concurrent_episodes=5` is the prepare-stage measure (CPU batch overlap);
  the actual GPU concurrency cap = 2 (2 free cards / 2 GPU workers). Do not read it as 5-card parallelism.
- This campaign also surfaced and fixed 6 real-integration defects (all committed, unit-tested):
  per-tier output.uri was not isolated (manifests overwrote each other), VLM auth was not propagated via
  `runtime_env`, scene↔hand switching did not close the old resident session (memory leak causing CUDA OOM on
  the 6th clip), `_create_pool("")` branch's `NodeAffinitySchedulingStrategy` was not imported (pipeline would
  crash), serial/static pipeline had no countable load events, and the GPU worker's 32Gi memory cap was too small.
- `load_events` only means **model load count**, not "weight cold-start seconds"; load time is already included in
  each tier's total_seconds and was not split out separately.

## 6. Teardown Recheck

`KUBECTL=/tmp/a800-kubectl bash deploy/k8s/teardown.sh`, then `status.sh`:
`TOTAL_GPUS=4`, `USED_GPUS=2` (only the platform qwen38-flash-next-varg on `<gpu-node-ip>`),
`TOTAL_FREE_GPUS=2`, no roboray RayCluster/RayJob leftovers.
GPUs returned to the platform baseline (2), the campaign left nothing held; the platform vLLM was never touched.
