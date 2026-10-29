# Audit-Improvements Spec: Code-Level Fixes for the 8 Findings of the Third-Party Review

Date: 2026-10-26
Status: scope aligned with the user (tier 1 engineering defects + tier 2a A1 joint optimization + tier 2b A2 real multi-person + tier 2c sync/contact, all executed; an improvements report follows)
Prerequisite: `2026-10-26-bench-vlm-overlap-design.md` (completed); E1/E2/E3 all completed

## 1. Background and Scope

The third-party completeness review gave 8 findings; the controller verified each one and **all are real**. This spec
handles the 7 that code can solve (#1-#7); #8 (the real-model batch is monocular; stereo + fusion + benchmark have no
joint measurement) needs the user to provide real stereo video and cannot be solved in code — honestly kept as a boundary.

| # | Finding (verified) | Disposition |
| --- | --- | --- |
| 1 | only 1 GPU actor per node (`build_node_pools` counts nodes, not cards; `NodeInfo.gpu_count` already exists but is ignored) | §3.1 multi-card slots |
| 2 | `NodePoolRegistry.register` overwrites outright on the same node | §3.1 eliminated along with multi-slotting (merge semantics) |
| 3 | A2 is not multi-person association (x-rank heuristic; `run_hamer.py` drops same-side detections with HANDS=2) | §3.3 real multi-person association |
| 4 | A1 only does translation anchoring, no "overall orientation" | §3.2 Kabsch rigid alignment |
| 5 | stage 1 has no timestamp alignment (only frame-count truncation) | §3.4 timestamp-synced frame table |
| 6 | stage 3 has no contact detection; tensors have no gripper channel | §3.5 contact segmentation + gripper |
| 7 | `inference_fps` is a dead config | §3.6 stride sampling landed |
| 8 | real-model batch is monocular | outside code scope (honest boundary §7) |

## 2. Decision Records

| # | Decision | Rationale | Cost (if wrong) |
| --- | --- | --- | --- |
| D-I1 | multi-card slots change `NodePool.gpu_actor` to a list + idle count; `slot_id` stays a global actor index (zero driver dispatch changes) | the current invariant (slot_id=gpu_actors index) is the foundation of all dispatch logic; keeping it confines the change to the pool layer | if slot mapping misaligns, dispatch can go to a dead actor — pinned with the existing test_node_pools technique |
| D-I2 | `_create_pool("")` landing on an existing pool → append the actor to the existing pool and return it (no new pool registered) | same-node multiple pools are exactly what triggers finding #2's overwrite; merge semantics naturally eliminate it | if pool capacity and in_flight count desync after append, over-dispatch — take/release changed to count by list length |
| D-I3 | `inference_fps` implemented as hand-inference stride sampling (`stride = round(video_fps / inference_fps)`); skipped frames have `present=False` and go to A3 | consistent with the article's "HandFlow observation + A3 completion" semantics; default 30/30 → stride=1 behavior is byte-identical | with stride>1 hand observations thin out; gaps beyond A3's 6-frame cap are kept (keeping gaps honestly matches A3's design); unknown fps falls back to stride=1 |
| D-I4 | A1 upgraded to Kabsch rigid alignment (rotation + translation, **no scaling**); when depth-supporting joints <4 or degenerate (singular covariance), fall back to root translation | the article's "stable position and overall orientation"; no scaling preserves MANO bone lengths (existing spec decision unchanged) | the fallback branch guarantees worst case is no worse than today; QC records the alignment source for attribution |
| D-I5 | detection protocol: `run_hamer.py`'s `max_num_hands` configurable (default 4); npz gains optional `candidates` field; the main-array semantics become "post-association wearer's two hands"; **old npz without candidates → x-rank fallback** | protocol addition does not break fake models/old data; the main-array semantic change makes downstream (post-fusion) unaffected | fake models produce no candidates → e2e goes fully through the fallback path; multi-person path covered only by unit tests + real cluster — the real-cluster regression task pins it |
| D-I6 | A2 association = wearer-prior validation (image lower-center proximity + bbox area) + inter-frame nearest-neighbor (center distance, threshold fps-dependent) + left/right labels overridable by motion; `identity_switches` upgraded to an entity-transition count | lands spec §7's original "position prior + motion continuity"; "MANO orientation consistency" deferred for now (candidates have no usable orientation prior, registered) | the wearer prior can pick wrong in atypical compositions (hands often above the frame) — prior weight configurable, observed in real-cluster regression |
| D-I7 | timestamp sync: `meta.json` optional `timestamps_left/right` (seconds arrays) → nearest-neighbor pairing (tolerance `0.5/fps`) → after alignment left/right share indices; no timestamps → current truncation fallback; `manifest` records `alignment: timestamps|truncation` | alignment stays inside prepare; downstream (scene/FFS assume same index) unaffected | out-of-tolerance frames dropped make n_frames smaller — manifest already records the degraded semantics; extend reason values with `timestamp_align_dropped` |
| D-I8 | contact segmentation: `segment_motions` low-speed dwell (speed < idle threshold for >=3 frames) emits `kind="contact"` segments; tensors gain a `gripper` channel = thumb-tip(4)−index-tip(8) distance / max in-dataset distance | completes the article's "velocity threshold + contact detection"; gripper is a **distance ratio**, not real gripper state; semantic boundary written into docs and field comments | if gripper semantics get misread as a real gripper → docs state it explicitly; missing-hand frames use gripper=NaN, never filled with 0 |

## 3. Design

### 3.1 Tier 1-1: multi-card slots + registration merge (#1+#2)

**`nodes.py`**:
- `NodePool` field `gpu_actor: object` → `gpu_actors: list` (keep a `gpu_actor` property returning the first for a
  transitional period? **no** — change all call sites in one go; the repo is small); `has_free_gpu()` →
  `in_flight < len(gpu_actors)`; take/release unchanged (the counting semantics naturally work out).
- new `NodePool.add_gpu(actor)`: append + sync driver-side `_actor_node`/slots (handled by driver._create_pool).
- `register()`: when a pool for the same node_id already exists, **merge** (`existing.gpu_actors.extend(pool.gpu_actors)`)
  instead of overwriting — defensive semantics; the normal path (_create_pool already returns the existing pool) never reaches it.

**`actors.py build_node_pools`**: `for node in nodes[:num_gpus]` → distribute by card count:

```python
remaining = max(1 if not nodes else 0, num_gpus)  # no nodes → keep the existing flat fallback
for node in nodes:
    if remaining <= 0: break
    k = min(node.gpu_count, remaining)
    for _ in range(k):  # k GPU actors on this node + same affinity
        ...
    remaining -= k
```

**`driver.py`**: `_register_existing_pools` (multiple actors on one node → one pool holds them all); after `_create_pool`'s
two branches produce an actor, if `actual` already has a pool → `pool.add_gpu(actor)` + return the existing pool (no
duplicate register); the `exit` branch's pool-removal condition becomes "all GPU actors on that node are dead"
(keep the current per-actor `actor is st.actor` lookup of the node → change `pools.pop` to check whether the node still
has a live actor; simpler: still pop the pool, but the rebuild path `_create_pool(node_id)` rebuilds with affinity and
re-registers; under multi-slotting a rebuild means multiple actors, semantics unchanged).

**Tests**: stub `discover_gpu_nodes` returns `[NodeInfo(A,gpu_count=2), NodeInfo(B,1)]` + monkeypatch GpuModelActor →
assert A gets 2 slots and B gets 1 (num_gpus=3) / A gets 2 (num_gpus=2, a single node's two cards no longer idle);
NodePool multi-slot take/release to the bound; register merge loses no actor; `_create_pool("")` landing on an existing
node appends and returns the same pool.

### 3.2 Tier 2a A1 Kabsch rigid alignment (#4)

**`fusion.py`** new pure function (placed next to `_mano_hand_to_world`):

```python
def rigid_align(mano_pts: np.ndarray, target_pts: np.ndarray,
                valid: np.ndarray, min_points: int = 4) -> tuple[np.ndarray, str]:
    """Kabsch no-scaling rigid alignment: returns (aligned mano_pts, "kabsch"|"root_only").
    Valid points < min_points or singular centered covariance (collinear) → fall back to root translation (current behavior)."""
```

- `_mano_hand_to_world`'s single line `pts_cam += (root_metric - pts_cam[0])` becomes a call to
  `rigid_align(mano_pts, depth_backprojected, depth_valid_mask)`; target = the joint 3D backprojected from the depth map
  (reusing the existing `_sample_depth` path).
- hand npz/QC gain an `alignment` field (per-frame or per-hand, whichever is simplest to implement: a per-hand string array).
- **Tests**: synthetic known R,t + Gaussian noise → recovered angle error <2°, translation <5mm; 3 valid points →
  root_only; collinear points → root_only; after alignment the reprojection error ≤ the root-translation version
  (assert no degradation on synthetic scenes).

### 3.3 Tier 2b A2 real multi-person association (#3)

**`run_hamer.py`**:
- `HANDS=2` → `max_hands` (`parse_args` new param `--max-hands`, default 4; payload can override via `max_hands`).
- the detection loop keeps **all** detections (remove the `if hand_present[i, side]: continue` drop — all go into the
  HaMeR batch, results all stored as candidates); new output field `candidates`: an object array, per frame a list of
  `{bbox(xyxy), side, score, joints_2d(21,3), joints_3d_cam(21,3)}`.
- main arrays (hand_present/joints_2d/…): change to "**select 2 by x-rank first per frame**" (as a no-association-layer
  compatible output)? **no** — the main arrays are filled with "x-rank left/right" directly (isomorphic to today), and
  the association upgrade happens on the fusion side, reading candidates and then **overwriting** the main arrays. This
  keeps run_hamer's output fully compatible with old consumers (fakes/unit tests), and all association logic is
  concentrated in fusion (unit-testable).
- **Tests** (script-level unit tests reuse the test_modelenv_scripts technique): candidates structure for a 4-hand
  frame, repeated side kept, score passed through.

**`fusion.py`** new pure-function module (new file `src/roboray/stages/association.py`; fusion.py at 587 lines no
longer grows):

```python
def associate_candidates(candidates: list, shape: tuple,
                         prior_center=(0.5, 0.85), prior_weight=1.0,
                         size_weight=0.5, motion_thresh_px=None) -> dict
```

- Input: per-frame candidate list (from the npz candidates); output `{joints_2d, present, switches, chosen}`.
- Algorithm: first frame by wearer score (`-prior_weight·normalized center distance + size_weight·normalized bbox
  area`, best per left/right); subsequent frames nearest-neighbor-match the two hands chosen in the previous frame
  (center distance ≤ `motion_thresh_px`, default `0.15 × image width × frame interval`); unmatched → re-select by the
  prior this frame (counts one switch); **left/right labels are only a tiebreaker, not a hard constraint**.
- `run_fusion` integration: npz has `candidates` → association result overwrites joints_2d/present, `identity_switches`
  uses switches; without → current x-rank (`associate_wearer` kept).
- **Tests**: synthetic 3-hand sequence (wearer's two hands big and low + bystander small and high) → wearer selected,
  switches=0; a bystander inserted at the wearer's position for 10 frames → no false switch (nearest-neighbor keeps
  entity); wearer's two hands cross (x-order swaps) → entity continuous, switches=0 (the x-rank baseline would be ≥1);
  first frame only the bystander → later wearer entry counts one re-select.

### 3.4 Tier 2c-1: timestamp sync (#5)

**`prepare.py`**:
- `load_episode` reads meta.json's optional `timestamps_left` / `timestamps_right` (float arrays, seconds; length must
  equal each side's frame count — mismatch → ignore timestamps, fall back to truncation, record the reason).
- with both timestamp arrays: nearest-neighbor pairing (for each left frame find the nearest right frame, pair only if
  `|Δt| ≤ 0.5/fps`), producing sync pairs; after alignment `left/right` are taken per pair (same index = sync pair —
  downstream unaffected); `degrade_reason="timestamp_align_dropped"` (if dropped frames >0) but the `degraded` semantic
  stays (timestamp-align drops are not stereo degradation; has_stereo stays True — only the dropped count goes to
  calib/manifest).
- without timestamps: the current truncation path is unchanged.
- `manifest`/record gains `alignment: "timestamps"|"truncation"` (filled by prepare, through the existing
  stage_result channel).
- **Tests**: synthetic left/right timestamp drift (right eye drops 1 every 10 frames) → sync-table length/content
  correct, out-of-tolerance dropped; one side missing timestamps → truncation fallback; length mismatch → fallback + reason.

### 3.5 Tier 2c-2: contact segmentation + gripper (#6)

**`action.py`**:
- `segment_motions` adds low-speed dwell: returns `[(start, end, kind)]` with `kind ∈ "motion"|"contact"`
  (`speed < contact_thresh` (default 0.02 m/s) for >=3 consecutive frames); labels/segment_start/end stay compatible
  (kind stored in the new action.npz key `segment_kind`).
- tensors: `gripper = ||joints[:, h, 4] − joints[:, h, 8]|| / max_dist` (one channel per hand; `max_dist` = the max of
  that channel within this episode; all-zero guard against division by zero); stored in action.npz and concatenated into
  state/action (state width +n_hands, action likewise) — **or** a separate key not concatenated into tensors? **into
  the tensors** (LeRobot columns are channels, directly usable by the training side); missing-hand frames NaN.
- `lerobot_writer.write` signature unchanged (state/action already contain the gripper column, transparent); docs and
  npz-key comments state "distance ratio, not real gripper state".
- **Tests**: synthetic motion-dwell-motion sequence → three segments with correct kinds; gripper range [0,1];
  missing-hand frames NaN; after LeRobot write-out, read-back column count matches.

### 3.6 Tier 1-2: `inference_fps` landed (#7)

**`hand.py` / `run_hamer.py`**:
- payload gains `inference_fps` (passed through from config; `hand_payload_args` already has the request to read);
  `video_fps` comes from calib.
- `run_hamer.py` inference loop: `stride = max(1, round(video_fps / inference_fps))`; frames with `i % stride != 0` skip
  inference (present=False, not in the HaMeR batch); `stride==1` behavior byte-identical.
- skipped frames go to A3: the current `fill_short_gaps` naturally handles present=False gaps (cap 6 frames) — with
  stride ≤7 A3 can fill everything (stride=7 → 6-gap exactly at the cap); the doc states this boundary.
- **Tests**: synthetic input with stride=2 → present alternates + after A3 completion all True; stride=1 output
  byte-equal to today (regression pin).

## 4. Execution Order and Validation

1. tier 1 (§3.1 + §3.6) → tier 2a (§3.2) → tier 2c (§3.4 + §3.5) → tier 2b (§3.3, the largest item last).
2. each tier is its own task, own tests, own commit; run full regression after each tier.
3. **real-cluster regression (GATE-gated)**: after everything merges, run a small batch on the A800 (reusing the bench
   flow, 4 clips x 30 frames) to validate: multi-card slots (e.g., a single node's 2 slots both dispatched), real
   candidate detection for A2, and that stride and Kabsch do not blow up on real data; produce comparison numbers into
   the improvements report.
4. **improvements report** `docs/bench/2026-10-26-audit-improvements-report.md`: respond item by item to the third
   party's 8 findings (fixed / semantic upgrade / honestly kept), including real-cluster regression evidence.

## 5. Test Strategy

- all new logic is pure-functioned (rigid_align / associate_candidates / timestamp pairing / contact segmentation /
  stride) — synthetic-data unit tests, no real models.
- protocol changes (run_hamer candidates) use the existing fake-script technique of test_modelenv_scripts.
- compatibility regression pins: stride=1 byte-equal to old output; npz without candidates walks the x-rank fallback
  path without blowing up; fake-model e2e all green.
- real-cluster regression per §4.3.

## 6. Acceptance Criteria (DoD)

1. third-party #1-#7 each has a code fix + unit-test pin; #8 honestly marked as a data boundary in the improvements report.
2. full suite green (expected 320+ with some additions); fake e2e does not regress.
3. real-cluster regression runs (GATE OPEN prerequisite): the `candidates` field appears in real npz, stride/Kabsch/
   timestamps do not blow up on real data, whole-batch failed=0. **Multi-card slots are not a real-cluster validation
   item** — in the K8s form each worker pod has exactly 1 card (Ray sees 1 GPU per node), so same-node multi-card never
   occurs naturally; that item is pinned by the §3.1 unit-test stubs, and the real-cluster validation will explicitly
   note this scope.
4. improvements report filed and committed, with the item-by-item comparison table and real-cluster evidence.
5. GPU-return recheck (USED_GPUS back to the platform baseline after teardown).

## 7. Honest Boundary

- **#8 kept**: the real-model benchmark batch remains monocular; joint measurement of stereo + fusion + benchmark needs
  the user to provide real stereo video.
- **A2's "MANO orientation consistency" signal not implemented** (D-I6): candidates have no usable orientation prior;
  registered as a follow-up; the prior weight may pick wrong in atypical compositions, observed in real-cluster regression.
- **gripper is a distance ratio**, not real gripper state — stated in the field comments and the improvements report.
- **stride >7 means A3 cannot fill everything** (gap exceeds the 6-frame cap); keeping the gap matches the
  "rather no data than bad data" design.
- Kabsch's fallback branch means some frames remain root-translation — the QC alignment field distinguishes honestly,
  never masquerading as full-frame joint optimization.
