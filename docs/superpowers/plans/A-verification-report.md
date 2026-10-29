# Plan A Verification Report

## Environment
- Local machine: Python 3.10.22, 8 cores, no GPU (Ray runs with `num_gpus` as a virtual resource)
- Ray version: 2.59.0

## Full Test Suite
- `pytest`: 99 passed (72.86s, unit + local end-to-end, fake models)

## Parallelism Benefit Reproduction (synthetic data, 3 Episodes / 30 frames)
| Mode | Total time (s) |
| --- | --- |
| Serial baseline | 1.780 |
| Cross-stage pipelining + adaptive scheduling | 1.394 |
| Speedup | 1.28x |

## Observed Scheduling Behavior
- Number of scheduling-decision snapshots: 6 (2 GPU actor slots, 3 decisions each)
- Model-reuse hits (share with `needs_load=False`): 4/6 = 66.7%
  > ⚠️ **This figure was invalidated on 2026-10-21** (see G6 in `docs/superpowers/specs/2026-10-21-ray-runtime-completion-design.md`).
  > `needs_load` actually only indicates whether the **adapter object** is rebuilt; the `load()` of that time only did an interpreter/path-existence check,
  > **unrelated to GPU loading** — every inference restarted a subprocess via `run_in_modelenv` and reloaded the weights.
  > Therefore 66.7% **cannot** be used to support the conclusion "no repeated-loading overhead".
- Note: the first 2 snapshots were cold starts (`needs_load=True`, `loaded=null`) that first loaded scene/hand into their slots; the remaining 4 snapshots all hit already-loaded models (`needs_load=False`), with no repeated-loading overhead. Pipelined total time 1.394s < serial baseline 1.780s, speedup 1.28x, consistent with the direction of the "cross-stage pipelining + adaptive scheduling" benefit described in the article.
- Note: the first adaptive run was deleted and re-measured because it was affected by resume-from-checkpoint; both CLI runs shared `output/run-fake`, so the adaptive manifest was overwritten by the serial run. The 6-snapshot evidence is in the Task 14 report along with a reproducible command.
