# Benchmark-Closure Spec: Real-Model Three-Mode Benchmark + Cross-Stage Overlap Evidence + Real VLM Connection

Date: 2026-10-26
Status: scope aligned with the user (G1+G2+G3; benchmark scale 6 clips x 60 frames)
Prerequisite: `2026-10-21-ray-runtime-completion-design.md` (E1/E2/E3 all executed)

## 1. Background and Goals

The validation checklist gained two sections (cross-stage parallelism, adaptive model scheduling) giving the
methodology for the three-mode timing comparison and the article's numbers (902.499 s → 644.439 s, ↓28.6%;
→ 536.550 s, another ↓16.7%). Current repo state:

| Capability | Status | Evidence location |
| --- | --- | --- |
| three-mode mechanism (serial / pipeline / adaptive) | ✅ implemented | `src/roboray/runtime/driver.py` file header + `_run_serial` / `_run_pipeline` |
| adaptive scheduling (4 signals + EMA feedback loop) | ✅ implemented | `src/roboray/scheduling/scheduler.py`, `cost_model.py` |
| three-mode comparison (stand-in models) | ✅ measured | `docs/superpowers/plans/D-l3-results.md` §2 (3 clips x 30 frames, backend=test) |
| real-model end-to-end | ✅ measured | same doc §8 (2 clips x 8 frames, **single mode** adaptive) |
| **real-model x three-mode benchmark** | ❌ not done | the two pieces of evidence above cannot assemble the checklist's table |
| **showable cross-stage overlap evidence** | ❌ not done | the timeline is recorded every run, but no renderer exists (neither viz/ nor scripts/ has one) |
| **real VLM action-semantics connection** | ❌ not done | the `openai_vlm` path exists, but every run used `model "none"` deterministic fallback |

This spec does three things: **G1** completes the three-mode comparison on real models (aligned to the article's
methodology); **G2** turns "EP 01 action understanding ‖ EP 02 dual-path inference ‖ EP 03 input preparation"-style
overlap into programmatically decidable + showable evidence; **G3** gives stage 3's "AI Function extracts action
semantics" one real VLM connection attempt (probe first; both branches count as complete).

## 2. Decision Records

| # | Decision | Rationale | Cost (if wrong) |
| --- | --- | --- | --- |
| D-B1 | the benchmark runs the three modes sequentially on the same resident cluster, each mode its own run-root, driver created fresh per mode | the alternative "separate cluster per mode" triples startup time with no extra fairness gain; a fresh driver per run makes CostModel/EMA cold starts naturally identical | if state leaks between modes (actors retaining loaded models) it would favor later modes — see §5 fairness constraints; teardown unload as fallback |
| D-B2 | batch of 6 clips x 60 frames @ 320x240 stereo 30fps; `make_l3_dataset.py`'s existing params support it directly | serial is estimated at 10–15 min; three modes total ~30–40 min GPU; high GPU share → clear inter-mode ratio signal | if measured serial exceeds expectations by >2x, allow dropping to 4 clips x 45 frames and record it in the results doc (changing the batch, not the method) |
| D-B3 | absolute times are not matched against the article, only the methodology (three-mode ratios) | hardware, data and model versions all differ; the article's numbers are its cluster's measurements | none — this is an honest boundary, not a compromise |
| D-B4 | mode order serial → pipeline → adaptive | run the slowest baseline first; even an interrupted run leaves baseline data | if the cluster is squeezed by the platform mid-run, the last two modes may be gated — recorded honestly in the results doc |
| D-B5 | overlap evidence = matplotlib Gantt PNG + programmatic decision function, dual artifacts | the pure-ASCII alternative reads poorly; the checklist wants "showable", and programmatic decision lets evidence be test-asserted rather than eyeballed | matplotlib is already a qc_plots dependency; no new dependency risk |
| D-B6 | VLM probe first: reachable → real run / unreachable → keep none and record the reason; **both branches count as complete** | the platform vLLM's reachability is unknown (a platform qwen38 service holds 2 cards); not betting on reachability | if the probe misjudges reachable but runtime is unstable, bench's action mode would slow adaptive — run_benchmark records per-episode action times so anomalies are attributable |
| D-B7 | probe/run both use a dummy API key (vLLM convention `EMPTY`); do not change `OpenAIVlm._resolve`'s non-empty validation | product code already validated; the probe script can inject a client_factory for unit tests | if the platform vLLM requires real auth and the user cannot provide a key → take the unreachable branch, recorded honestly |

## 3. G1 - Real-Model Three-Mode Benchmark

### 3.1 Deliverables

**`scripts/run_benchmark.py`** — benchmark orchestration CLI:

```
python scripts/run_benchmark.py --config <bench.yaml> --bench-root <dir> \
    [--modes serial,pipeline,adaptive] [--keep-going]
```

- for each mode, invoke `python -m roboray run --config <bench.yaml> --run-root <bench-root>/run-<mode>/` as a
  subprocess with the mode flags: serial=`--serial-baseline`, pipeline=`--no-adaptive`, adaptive=default (flag names per
  `src/roboray/cli.py` as actually implemented; verify during implementation; the three-mode semantics recorded in
  D-l3-results §2 are the same source).
- subprocess isolation guarantees each mode has a fresh driver, CostModel, and `ray.init`.
- after each mode, read `<run-root>/out/manifest.json` and extract: `total_seconds`, `failed`,
  per-episode per-stage `timings`, `scheduler_snapshots` count, timeline count, and the model the action mode actually
  used (read back from manifest or config).
- aggregate into **`<bench-root>/bench_report.json`** and **`<bench-root>/bench_report.md`**: a three-mode table
  (total time / speedup vs serial / failed count / scheduler-snapshot count / action model), with a one-line note under
  the table: "methodology aligned with the article's three modes (serial↔902.499 / pipeline↔644.439 /
  adaptive↔536.550); absolute values not compared — different hardware and data (decision D-B3)".
- the aggregation logic is a pure function (e.g., `aggregate(modes_results: dict) -> dict`) for fake-manifest unit tests.
- a mode's CLI nonzero exit: by default abort and keep the completed modes' report; with `--keep-going` continue the
  remaining modes, marking that mode `error` in the report.

### 3.2 Dataset and Config

- dataset: `python scripts/make_l3_dataset.py --out <shared>/bench-data --episodes 6 --frames 60`
  (stereo 320x240 @30fps; the script can also write the run config).
- bench.yaml: template from the final validation config `/ego/l3-final.yaml` — real-model env paths (vipe-venv / FFS /
  HaMeR weights), `model_worker: resident`, `max_gpus`, `num_cpu_actors`, `max_episodes_in_flight` reuse the validated
  values; only `input.uri` points at the bench dataset. bench.yaml is written directly on the cluster node (same
  convention as `/ego/l3-final.yaml`; paths are node-local); the repo stores a sample `deploy/bench/bench-l3.yaml`
  (same values, commented) for reproducibility without a second rendering mechanism.

### 3.3 Fairness constraints (implemented as run_benchmark pre-checks + doc assertions)

1. the three modes share the same cluster and the same input directory, each with its own run-root (artifacts do not
   collide).
2. driver fresh per mode (subprocess guarantees it); inter-mode model unload guaranteed by `_teardown_unload_all`
   (the driver's finally block, already exists) — run_benchmark does **no** extra cleanup before each mode, but reads
   back in the report whether the first episode of a mode contains load time (a `was_load=true` snapshot) as cold-start evidence.
3. the three modes use the same config, differing only in the mode flag; `ROBORAY_MODEL_WORKER=resident` across all three.
4. environment gate: `deploy/k8s/status.sh` must report GATE=OPEN before running; the results doc records the then-current USED/FREE GPUS.

## 4. G2 - Cross-Stage Overlap Evidence

### 4.1 `src/roboray/viz/timeline_view.py`

- **`overlap_evidence(timeline: list[dict]) -> dict`** (pure function)
  - input is the RunResult/manifest timeline events: `{"t", "episode_id", "stage", "event"}`
    (event ∈ start/end; idle events have episode_id None, skipped).
  - pair start/end into each (episode, stage) interval; unclosed intervals (crash scenario) are truncated to the last
    event time and counted as `open_intervals` in the result.
  - output:
    ```json
    {
      "max_concurrent_episodes": 3,
      "max_concurrent_stages": 4,
      "evidence": [{"t": 12.3, "episodes": {"ep_0001": ["action"], "ep_0002": ["scene", "hand"], "ep_0003": ["prepare"]}}],
      "open_intervals": 0
    }
    ```
    `evidence` takes the (first) time snapshot at which the concurrent-episode count peaks; multiple peak times record only the first.
- **`render_gantt(timeline, out_png, title)`**: one horizontal bar chart row per Episode
  (`broken_barh`), stages color-coded, x-axis seconds; matplotlib Agg backend (no display in test env).
- unit tests: synthetic three-overlap timeline asserts the output structure above; a single serial timeline asserts
  max=1; unclosed-interval counting; gantt smoke (artifact exists and is non-empty).

### 4.2 `scripts/render_timeline.py`

```
python scripts/render_timeline.py <manifest.json> --out-dir <dir>
```

- reads the manifest's timeline, writes `<dir>/timeline.png` + `<dir>/timeline_evidence.json`,
  prints a one-line markdown summary to stdout (`max_concurrent_episodes=N (EP a: stage1; EP b: …)`), ready to paste
  into the results doc.
- after run_benchmark finishes the adaptive mode it automatically calls this script on that manifest; artifacts land in
  `<bench-root>/run-adaptive/timeline/`.

## 5. G3 - VLM Probe and Real Connection

### 5.1 `scripts/probe_vlm.py`

```
python scripts/probe_vlm.py --base-url URL --model NAME [--api-key-env VAR] [--timeout 30]
```

Three-step probe, degrading step by step, outputting a classified conclusion (stdout machine-readable
`PROBE_RESULT=<tag>` + human-readable detail):

| Step | Action | Failure → tag |
| --- | --- | --- |
| 1 | `GET {base}/models` (best effort, lists model names) | `unreachable` (with reason) |
| 2 | minimal text chat.completions call | auth error → `auth_failed`; other → `unreachable` |
| 3 | multimodal call with a 1-frame 8x8 base64 PNG data-URI | multimodal rejected → `reachable_text_only` |

- all passing → `reachable_multimodal`.
- when the API key is missing, use a dummy (`EMPTY`) and print a warning (decision D-B7); if the `--api-key-env`
  environment variable exists and is non-empty, prefer it.
- reuses `OpenAIVlm`'s `openai` SDK path; implemented with an injectable `client_factory` (the same pattern as
  `semantic_models.OpenAIVlm`), unit tests covering each tag branch (fake client).
- the probe **must run inside a cluster pod** (login-node reachability does not imply pod reachability) — following
  E3's head-exec run style.

### 5.2 Dual branches (decision D-B6)

- **reachable_multimodal** → bench.yaml's `action_understanding` set to
  `model: openai_vlm` + `vlm_base_url/vlm_model/vlm_api_key_env`; export the corresponding environment variables in the
  run environment; all three modes genuinely run VLM (action is on a CPU actor, registration path already exists).
- **other tags** → keep `model: none`; bench_report and the results doc record `action_model: none` + the probe tag +
  a one-line reason. **This is honest recording, not degradation.**
- if the probe passes but VLM fails stably at runtime (e.g., content-format incompatibility): allow a **minimal fix**
  to `src/roboray/models/semantic_models.py` (with unit tests); if it cannot be fixed, that mode's action falls back
  to none and the report marks `action_fallback: <reason>` — this does not block G1/G2.

## 6. Execution Runbook (real cluster, following validated paths)

1. local full unit suite green (`RAY_ADDRESS=local … -m pytest -q`).
2. `deploy/k8s/status.sh` → continue only if GATE=OPEN; if CLOSED stop and report honestly, no fake deployment.
3. `deploy/k8s/apply.sh` starts a standalone cluster (E3-validated path; **not RayJob** — the K8sJobMode
   workload-localization pitfall is registered, this spec does not solve it).
4. head-exec: generate the dataset (6x60) → write bench.yaml → `probe_vlm.py` decides the action branch.
5. `run_benchmark.py` three modes; the adaptive mode auto-renders overlap evidence.
6. collect artifacts (bench_report.md/json, timeline.png/evidence.json) → write the results doc (numbers, probe
   conclusion, confounds such as the GPU being squeezed by the platform) → commit.
7. `deploy/k8s/teardown.sh` + `status.sh` recheck GPU return (USED_GPUS should be back at the platform baseline).

## 7. Test Strategy

- **unit (local, must be all green)**:
  - `overlap_evidence` / `render_gantt`: synthetic timeline all branches (incl. unclosed intervals).
  - `run_benchmark.aggregate`: fake manifests (incl. a failed mode, the `--keep-going` branch, missing manifest).
  - `probe_vlm`: injected fake client_factory covering all 4 tag branches + the dummy-key warning.
  - flag check: `test_cli.py` adds assertions that the three mode flags really exist (guarding spec/CLI drift).
- **real cluster**: the section-6 runbook is itself the integration validation — three modes failed=0 + overlap
  evidence meeting the bar (adaptive mode `max_concurrent_episodes ≥ 3`) is the acceptance.

## 8. Acceptance Criteria (DoD)

1. `bench_report.md`: three modes with real models all pass (failed=0), including total time, speedup vs serial,
   scheduler-snapshot count, the action-model column, and the D-B3 methodology note.
2. adaptive-mode overlap evidence: `timeline.png` + `timeline_evidence.json`, with `max_concurrent_episodes ≥ 3`
   (the checklist's three-segment overlap shape).
3. VLM branch conclusion clearly filed: real VLM ran, or `none + probe tag + reason`.
4. results doc committed into the repo (numbers + confounds + honest boundary); GPU-return recheck passes.
5. full unit suite green; every new pure function has unit tests.

## 9. Honest Boundary

- synthetic data (generated by `make_synthetic_ego`), not HOT3D; absolute times are not compared against the article
  (D-B3); the ratios depend on the batch and hardware and are only comparable within this batch.
- the serial mode is also real models and a real export (not a stub) — it is a baseline, not a strawman.
- `max_concurrent_episodes ≥ 3` depends on the 6-clip batch and the scheduling window; if measurement only reaches 2,
  honestly record the actual value and analyze (CPU actor count / max_episodes_in_flight are candidate bottlenecks),
  not tune parameters to fake the number.
- the platform vLLM is a platform service; this spec only does client-side connection; its service-side
  capacity/rate-limiting is outside our control.
