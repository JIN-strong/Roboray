# Audit Improvements Report: Third-Party 8 Items, Item-by-Item Responses + Real-Cluster Regression Evidence

Date: 2026-10-27 (Task 11, closing out plan `docs/superpowers/plans/2026-10-26-plan-audit-improvements.md`)
Code baseline: main `e16c0a9` (Tasks 1–10 all merged; prior full regression 360 passed)
Real cluster: the A800 GPU cluster at <gpu-node-ip> (GATE=OPEN: TOTAL_GPUS=4, USED_GPUS=2 [pre-existing platform vLLM pods, untouched], TOTAL_FREE_GPUS=2)

---

## 1. Item-by-Item Table (Third-Party 8 Items → Disposition + Evidence)

| # | Finding (verified) | Disposition | Code evidence (commit / test) | Real-cluster forensics |
| --- | --- | --- | --- | --- |
| 1 | Only 1 GPU actor per node (per node count, not per card; `gpu_count` ignored) | **Fixed** (Task 1/2) | `d350081` (NodePool multi-slot, behavior unchanged) + `61a2c76` (fill slots by card count) + `07b35f0` (exit removal semantics pinned); tests `test_node_pool_multi_gpu_slots`, `test_build_node_pools_fills_by_gpu_count` (2 cards → 2 actors on node A), `test_multi_gpu_slots_dispatchable` (two slots on the same node each own a distinct slot, dispatch without overlap or omission); release does not drain down (2 held, 3 free) pinned in the tail assertion of `test_node_pool_multi_gpu_slots` | **K8s view not applicable**: this cluster's RayCluster `rayStartParams.num-gpus=1` (exactly 1 card per pod), so the multi-card-per-node case is invisible on a real cluster. Per spec §6's established view we do not fake verification; correctness is pinned by the unit tests above |
| 2 | `NodePoolRegistry.register` overwrites directly on the same node | **Fixed** (Task 1/2, eliminated with multi-slot) | `d350081` (register merges same-node pools via extend) + `61a2c76` (`_create_pool("")` lands on an existing node, appends, and returns the same pool instead of creating a new one); tests `test_registry_merges_same_node_pools`, `test_create_pool_on_existing_node_merges`, `test_create_pool_empty_lands_on_existing_node_appends` | Under the real cluster's one-card view the overwrite trap cannot trigger (one pool per node); the defensive semantics are covered by unit tests |
| 3 | A2 not multi-person association (x-rank heuristic; run_hamer HANDS=2 drops same-side) | **Fixed** (Task 8/9/10, incl. Critical fix) | `c8037d1` (max_hands configurable + all detections kept into candidates) → `09fcd6a` (review: AST guard pins the call chain + pixel-unit unification) → `b25cedc` (max_hands threaded through all four config segments) → `833dee4` (associate_candidates pure function: wearer-first verification + temporal nearest-neighbor + entity continuity) → `8e6cc94` (review: restore motion-matching exclusivity, two slots never bind the same candidate) → `6b2889f` (fusion closed loop: candidate association overwrites the main data, keyless fallback to x-rank) → `e16c0a9` (review Critical: association moved up before fill/projection + main arrays reordered by labels — multi-person frames have consistent geometry and identity, RED counterexample pins the data plane) | hand.npz `candidates` key present in 4/4 episodes (object: frame → candidate list); **no multi-person frames in the data** (multi_hand_frames=0, max_per_frame=1) — the multi-person path is honestly covered by unit tests + fake e2e (`test_fusion_with_candidates_uses_association`, `hand_ids==labels` and ≠x-rank asserted via instrumentation), see honest boundaries §3.1 |
| 4 | A1 anchors translation only, no "global orientation" | **Fixed** (Task 4/5, fallback branch kept) | `083e394` (rigid_align pure function: Kabsch no-scale rotation + translation; degenerate case falls back on condition number — the briefing's "SVD throws" was a bug, NumPy does not throw) + `6293a45` (_mano_hand_to_world integration + alignment field written to npz); tests `test_rigid_align_*` (synthetic R/t recovered <5mm, partial valid points, collinear fallback), `test_mano_world_kabsch_beats_root_translation` | fused.npz `alignment` per-hand distribution: **100% kabsch on observed frames** (ep1 5, ep2 4, ep3 1, ep4 0 — the data naturally has few hand frames), root_only 115/116/119/120 (frames with no observation go root-anchored), none=0; the high fallback share is the data's scarcity of hands, not algorithmic fallback |
| 5 | Stage 1 has no timestamp alignment (frame-count truncation only) | **Fixed** (Task 6, incl. two guards) | `793ae39` (align_by_timestamps nearest-neighbor pairing, tolerance 0.5/fps, alignment converges in prepare, downstream unaffected) + `37cf8b8` (guards: ndim check prevents nested crash + zero-pairing falls back to truncation degradation); tests include correcting the briefing's arithmetic slip (0.3s lag formula) with review numerical verification | manifest `alignment=truncation, align_dropped=0` (4/4) — the real data's meta.json has **no `timestamps_left/right` field**, so by design it takes the fallback truncation path; the aligned main path has no real stereo data to trigger it (same boundary as #8), covered by unit tests |
| 6 | Stage 3 has no contact detection; tensor has no gripper channel | **Fixed** (Task 7, semantics = distance ratio) | `98508bd` (segment_motions low-speed resident contact segments + gripper = thumb(4)–index(8) fingertip distance / batch max appended to the tensor; legacy 4-dim artifact release compatible) + `d28b5af` (docstring follow-up); test `test_publish_accepts_legacy_4d_state` etc. 4-dim compatibility + 4 new tests | action.npz `gripper` present in 4/4 episodes: shape (60,2), range [0,1], NaN only on frames with no hand (ep1 6, ep2 2, ep3/ep4 0 — NaN semantics = hand absent, not filled with 0); `segment_kind` motion/contact mixed distribution (ep1 m3 / ep2 m4+c1 / ep3 m2+c1 / ep4 c3) |
| 7 | `inference_fps` dead config | **Fixed** (Task 3, incl. adapter's last mile) | `c94976b` (compute_stride + run_hamer frame-skip sampling; stride=1 behavior byte-identical) + `4249182` (review R1: inference_fps reached only the payload, not the script extra — adapter/actor passthrough chain added, the real path is no longer a dead config); tests `test_stride.py` 3 items + fake contract isomorphism | **Direct-effect evidence**: `inference_fps: 15` (source 30fps) → stride=2, observed frames in all 4 episodes land on **even frame numbers** (32/42/2…, odd frames 0 times), hand.npz `fps=30` output timeline unchanged; control fps=30 gives mixed odd/even observed frames (3/45/31/32/39/37/42/1/2/5) |
| 8 | Real-model batch is monocular (stereo + fusion + benchmark have no joint real test) | **Honestly kept** (data boundary; code cannot fix) | —— | All 6 bench-real-data clips are monocular `rgb_left.mp4`; no real stereo video is available, so #5's timestamp main path and stereo-fusion joint real test remain missing. **Needs a real stereo video from the user, then a separate regression** |

---

## 2. Real-Cluster Regression Numbers (4×60-frame real models ×2 runs)

**Environment**: standalone RayCluster `roboray-cluster` (apply.sh, IMAGE=registry.example.com/roboray/base:v0.26.0, NODE=<gpu-node-ip>); head-exec (`--ray-address auto`), autoscaling gpu-workers min=0/max=2 (1 card per pod, brought up on demand); data = a 4-clip subset of bench-real-data (ep_0001–0004, 60 frames @30fps each, monocular), actually copied to `/ego/audit-data` (symlinks are rejected by the storage-path boundary guard — expected safety behavior, recorded honestly).

**Run log**:
- run1 `/ego/audit.yaml` (inference_fps=30): **failed=0**, all six stages DONE ×4, total 39.9s
- run2 `/ego/audit-stride.yaml` (inference_fps=15, only this one knob differs + output isolation): **failed=0**, all six stages DONE ×4, total 105.2s
- manifest per episode: `degraded=False` ×8; `alignment=truncation, align_dropped=0` ×8 (the design fallback for no-timestamps meta, see row #5)

**Forensics checklist item by item** (run1 primary, run2 annotated):

| Forensic item | Result |
| --- | --- |
| manifest failed | **run1=0, run2=0** (4 episodes each; prepare/scene/hand/fusion/action/publish all done) |
| hand.npz `candidates` | key present 4/4 (object: frame → candidate list); nonempty frames ep1=5/ep2=4/ep3=1/ep4=0; **multi_hand_frames=0** (no multi-person/multi-hand frames in the data, max_per_frame=1) → the multi-person path is covered by unit tests; we do not invent evidence |
| fused.npz `alignment` | run1: kabsch=5/4/1/0 (= observed-frame count; 100% kabsch on observed frames), root_only=115/116/119/120, **none=0**; run2 isomorphic (kabsch=observed-frame count) |
| `identity_switches` | ep1=[[31,0]], ep2=[[37,0]], ep3=[], ep4=[] (bookkeeping under low-frequency single-hand observation; no multi-person scene so no entity jumps) |
| action.npz `gripper` | shape (60,2) ×4; range [0,1]; NaN only on no-hand frames (6/2/0/0), not filled with 0 — distance-ratio semantics, not a real gripper |
| `segment_kind` | ep1 motion×3; ep2 motion×4+contact×1; ep3 motion×2+contact×1; ep4 contact×3 (all-contact on the no-hand video is semantically reasonable: low-speed residence) |
| stride batch (run2) | sparse present pattern: observed frames **all even** (4/4 episodes), odd frames 0 detections (direct stride=2 evidence); total observed hand-frames 4 (fps30 control 10 — halved detection opportunities as expected) |
| A3 fill rate | post-fusion per-hand finite frames ≥ observed: run1 ep1 hand0 **2 observed → 54/60** (gapfill_filled=5 frames, rejected=10 — gaps exceeding `unobserved_gap_s=0.2s` honestly stay NaN), hand1 3→60/60, ep2 hand0 2→58/60; run2 observations too sparse (single-frame observation) → gapfill_filled all 0, gaps refused by `neighbour_unobserved` — **filling is bounded, we do not fabricate data** |
| Multi-card slots | K8s one-card-per-pod view (num-gpus=1); single-node multi-card is invisible on a real cluster — pinned by unit tests (§1 row #1), not faked as verified |

**A2 association-path real-cluster coverage boundary**: the data has no multi-person frames, so `associate_candidates` only exercises the "single/empty candidate" branch on the real cluster; dual-candidate assignment, exclusivity, reroute, identity_switches entity jumps, etc. are **covered by 4 unit-test scenarios + fake e2e (hand_ids==labels asserted via instrumentation)**.

---

## 3. Honest Boundaries (incl. Ledger-Registered Items)

1. **A2 orientation-consistency signal not done** (registered D-I6): candidates have no usable orientation prior; the association criterion is position prior + motion continuity; MANO orientation consistency is left for later.
2. **gripper semantics**: thumb–index fingertip distance / batch max ([0,1] ratio), **not real gripper state**; stated explicitly in the field comment and in two places in this doc.
3. **stride fill boundary**: stride>7 A3 cannot fully fill (already registered in the plan); this run used stride=2 with naturally hand-scarce data → A3 also has nothing to fill when observations are sparse (rejected: neighbour_unobserved) — fill rate is dominated by "hand visibility in the data", not only by stride.
4. **Kabsch fallback share**: on a full-frame view root_only dominates (kabsch only on observed frames out of 120) — data hand scarcity; on the **observed-frame view the kabsch rate is 100%**; the fallback branches (insufficient points / collinear / root NaN) were not triggered by real data and are covered by unit tests.
5. **publish camera view still draws other hands on multi-person frames** (registered outside Task 10's scope): publish renders hand.npz's raw joints_2d without filtering by labels; the main data plane is fixed (e16c0a9), the visualization plane is untouched.
6. **Hang risk for consecutive runs on the same cluster** (follow-up registered in Task 2): resident-actor unload does not return the Ray GPU slot + no post-run reaper + unbounded `_register_existing_pools` ray.get — a product decision beyond the plan's scope. Not reproduced this run (run interval >240s, the idle timer released the GPU first: at run2 start `ray.available_resources()` showed GPU=1.0 free), but **the risk remains**; fix before consecutive dense runs.
7. **No multi-person frames in real data**: real-cluster verification of the A2 multi-person main path stops here (data boundary); we do not fabricate multi-person forensics.
8. **Fused trajectories with zero observed hands**: a hand with hand_present=0 still has fully finite fused.joints_world (mano anchoring fallback, hand_ids=-1, joint_source=mano, |jw|max≈30.17) — **consumers must not read it as real motion**; reprojection_error≈1e-13 is also a numerical artifact of synthetic anchoring.
9. **Timestamp-sync main path not triggered on real data**: the data has no `timestamps_left/right` meta (corollary of the #8 monocular boundary); the manifest honestly records `truncation`.
10. **Test-environment facts** (working notes, not product defects):
    - On this platform `kubectl exec` lands in the **host** namespace (the container name does not exist / mount table is the host's); entering the container requires a node-side `sudo docker exec <ray-head container>` (same path as the bench campaign);
    - The `vllm:v0.26.0` image is missing libEGL/mesa swrast, so HaMeR (`PYOPENGL_PLATFORM=egl`) crashes at `import pyrender`. Fixed this time with **hostPath persistence**: `/ego/egl-libs` (glvnd libEGL.so.1 + libEGL_mesa.so.0 + swrast/kms_swrast dri + vendor json) + a `/ego/hamer-venv/bin/python` wrapper script (injects `LD_LIBRARY_PATH`/`LIBGL_DRIVERS_PATH`/`__EGL_VENDOR_LIBRARY_FILENAMES`, uses `exec -a` to preserve argv0 for venv probing), verified in a clean-container equivalent environment (docker run without GPU injection) for the full dlopen→pyrender→OffscreenRenderer chain. **The image itself was not changed; redo this step if the image changes**;
    - gpu-worker autoscaling min=0 → each run brings up a new pod: a one-shot `apt install libegl1` was fixed and then reclaimed (verified once), so the ephemeral approach was abandoned for hostPath persistence;
    - The VLM endpoint does not validate keys; runs pass the auth check with the placeholder `VLM_API_KEY=audit-run-any-key` (had failed=4 once due to a missing env var);
    - The multi-card-slot real-cluster view is in §1 row #1.

---

## 4. Full Test Counts

- Code baseline (after Tasks 1–10 merged): **360 passed, 3 deselected** (local main `e16c0a9`)
- Task 11 has no code changes (forensics + report), 0 new tests:
  - Final check (recheck after teardown): `RAY_ADDRESS=local … -m pytest --tb=short` → **360 passed, 3 deselected in 224.28s (0:03:44), exit 0**
  - Note: pyproject `addopts` already includes `-q`; stacking another `-q` on the command line swallows the summary line due to double-quiet (the first two runs had all-green dots + exit 0 but no summary line, so they were not counted) — the final check uses this line as authoritative.

---

## 5. Teardown and GPU Return Recheck

- 2026-10-27: `teardown.sh` (RayJob→Job→RayCluster→Pod cascade) completed; `status.sh` recheck: **TOTAL_GPUS=4, USED_GPUS=2 (back to the platform baseline: pre-existing platform vLLM @<gpu-node-ip>, never touched), TOTAL_FREE_GPUS=2, GATE=OPEN**; roboray's RayCluster/Pods are cleared. Note: `delete pod --all` also affected the kuberay-operator pod resident in the same namespace, which its deployment self-healed (self-healing inside the roboray namespace, not platform resource damage). **Registered improvement (follow-up)**: change teardown.sh's pod cleanup to a label selector (e.g. `-l ray.io/cluster`) or move kuberay-operator out of the roboray namespace, to avoid bouncing the operator on every teardown.
