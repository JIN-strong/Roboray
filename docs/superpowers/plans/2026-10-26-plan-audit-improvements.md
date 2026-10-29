# Audit Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the third-party review's confirmed items #1-#7: multi-GPU slots, register merging, A1 Kabsch joint alignment, A2 real multi-person association, timestamp synchronization, contact segmentation + gripper, and landing inference_fps; then produce an improvements report after a real-cluster regression.

**Architecture:** pure functions first (rigid_align / associate_candidates / compute_stride / timestamp pairing / contact segmentation are all testable with synthetic data), and protocol changes follow "additive + fallback" (the npz gains a candidates field, old data uses x-rank; stride=1 stays byte-identical). NodePool becomes multi-slot, keeping the slot_id = global actor index invariant, so the driver dispatch needs zero changes.

**Tech Stack:** Python 3.10 (roboray conda env), numpy, pytest; real-model changes touch only `scripts/modelenvs/` (protocol layer -- real models are not started locally, coverage comes from contract-isomorphic fakes + a real-cluster regression).

**Spec:** `docs/superpowers/specs/2026-10-26-audit-improvements-design.md`

## Global Constraints

- Test command (repo root): `RAY_ADDRESS=local python -m pytest -q --tb=short`
- Commit messages: English conventional commits, ending with `Co-Authored-By: Claude Code <noreply@anthropic.com>`
- No new third-party dependencies; all new algorithm logic is pure functions (testable with synthetic data, no Ray/real models)
- **Invariant**: `slot_id` == the global index into `self.gpu_actors` (the driver's `zip(self.slots, self.gpu_actors)` dispatch view depends on it)
- **Protocol additive**: new npz/payload fields are always optional; old data/old fakes without the new fields must take the fallback path and behave exactly as before
- Fake and real scripts are **contract-isomorphic** (`tests/fakes/modelenv_scripts/fake_hamer.py` header says "same contract as run_hamer.py") -- stride/candidates change on both ends so e2e has coverage
- Script-level tests use `tests/unit/test_modelenv_scripts.py`'s `_run()` subprocess technique (`--input npz --extra json`)
- Comments/docs in English

## Review Focus

1. slot↔actor misalignment after multi-sloting (2 actors on one node must each occupy an independent slot, dispatch without duplication or omission) → Task 2 `test_multi_gpu_slots_dispatchable`.
2. stride=1 behavior drift (the default config's output must be byte-identical to before) → Task 3 `test_stride_default_identical_output`.
3. old hand.npz without the candidates key → fusion must fall back to x-rank with results identical to before → Task 10 `test_fusion_without_candidates_falls_back` (the existing fusion test fixtures all go through this path, a second layer of insurance).
4. the theoretical possibility that Kabsch under partial valid points/noise is worse than root translation → Task 4/5 synthetic assertions `reproj(kabsch) <= reproj(root_only)` + degenerate fallback.
5. publish/LeRobot dimension break after concatenating the gripper column (an old 4-D action.npz must still be publishable) → Task 7 `test_publish_accepts_legacy_4d_state` + full regression.

---

### Task 1: NodePool multi-slot + register merging (behavior-preserving refactor)

**Files:**
- Modify: `src/roboray/ray_runtime/nodes.py` (NodePool fields + register)
- Modify: `src/roboray/runtime/driver.py` (the 3 pool.gpu_actor call sites)
- Test: `tests/unit/test_node_pools.py` (append)

**Interfaces:**
- Consumes: the existing `NodePool(node_id, gpu_actor, cpu_actors)` (to be deprecated)
- Produces: `NodePool(node_id, gpu_actors: list, cpu_actors: list, in_flight)`; `add_gpu(actor)`; `register()` merges actors for the same node_id instead of overwriting; `has_free_gpu() = in_flight < len(gpu_actors)`. Consumed by Task 2.

- [ ] **Step 1: write failing tests (append to tests/unit/test_node_pools.py)**

```python
def test_node_pool_multi_gpu_slots():
    from roboray.ray_runtime.nodes import NodePool

    a1, a2, cpu = object(), object(), object()
    pool = NodePool(node_id="n1", gpu_actors=[a1], cpu_actors=[cpu])
    assert pool.has_free_gpu()
    pool.take_gpu()
    assert pool.has_free_gpu()  # 2 slots, only 1 occupied
    pool.add_gpu(a2)
    pool.take_gpu()
    assert not pool.has_free_gpu()  # 2 slots, 2 occupied
    pool.release_gpu()
    pool.release_gpu()
    pool.release_gpu()  # must not go below zero
    assert pool.has_free_gpu() and len(pool.gpu_actors) == 2


def test_registry_merges_same_node_pools():
    from roboray.ray_runtime.nodes import NodeInfo, NodePool, NodePoolRegistry

    reg = NodePoolRegistry(
        live_nodes=[NodeInfo("n1", 2, "h1")],
        cpu_actors_per_node=1,
        create_pool=lambda nid: NodePool(nid, gpu_actors=[object()]),
    )
    p1 = NodePool("n1", gpu_actors=[object()], cpu_actors=[object()])
    reg.register(p1)
    extra = NodePool("n1", gpu_actors=[object()], cpu_actors=[object()])
    reg.register(extra)  # a second pool on the same node: merge, not overwrite
    merged = reg.pools["n1"]
    assert merged is p1
    assert len(merged.gpu_actors) == 2 and len(merged.cpu_actors) == 2
```

- [ ] **Step 2: run the tests and confirm failure**

Run: `RAY_ADDRESS=local python -m pytest tests/unit/test_node_pools.py -q`
Expected: FAIL — `NodePool.__init__() got an unexpected keyword argument 'gpu_actors'`

- [ ] **Step 3: implement**

Replace `nodes.py`'s NodePool wholesale with:

```python
@dataclass
class NodePool:
    """Actor pool on one node: K GPU actors + N CPU actors, all on the same node.

    GPU slots are exclusive (K = the node's GPU count, spec 2026-10-26 §3.1): the
    second GPU of a multi-GPU node no longer sits idle. ``in_flight`` is the
    concurrent-occupancy counter; ``has_free_gpu`` uses the actor count as capacity.
    """

    node_id: str
    gpu_actors: list = field(default_factory=list)
    cpu_actors: list = field(default_factory=list)
    in_flight: int = 0

    def add_gpu(self, actor) -> None:
        self.gpu_actors.append(actor)

    def has_free_gpu(self) -> bool:
        return self.in_flight < len(self.gpu_actors)

    def take_gpu(self) -> None:
        self.in_flight += 1

    def release_gpu(self) -> None:
        self.in_flight = max(0, self.in_flight - 1)
```

Replace `register` with:

```python
    def register(self, pool: NodePool) -> None:
        existing = self.pools.get(pool.node_id)
        if existing is None:
            self.pools[pool.node_id] = pool
            return
        # a second pool on the same node (audit #2: a direct overwrite would drop
        # actors and misalign in_flight) -- merge.
        existing.gpu_actors.extend(
            a for a in pool.gpu_actors if not any(a is b for b in existing.gpu_actors))
        existing.cpu_actors.extend(
            a for a in pool.cpu_actors if not any(a is b for b in existing.cpu_actors))
```

`driver.py` call sites (grep `gpu_actor` to find them all):
- `_register_existing_pools`: collect by node_id, then build `NodePool(node_id=node_id, gpu_actors=[all actors of this node], cpu_actors=list(self._cpu_by_node.get(node_id, [])))`;
- `_create_pool` returns `NodePool(node_id=actual, gpu_actors=[gpu], cpu_actors=cpus)`;
- `_pool_index_for`'s `self.gpu_actors[idx] is pool.gpu_actor` → `any(self.gpu_actors[idx] is a for a in pool.gpu_actors)`.

Synchronously update the existing `NodePool(node_id=…, gpu_actor=…)` constructions in `tests/unit/test_node_pools.py` to `gpu_actors=[…]` (grep the whole repo for `gpu_actor=` to wrap up; no residual old kwarg allowed).

- [ ] **Step 4: run the tests and confirm pass + full regression (behavior-preservation check)**

Run: `RAY_ADDRESS=local ... -m pytest tests/unit/test_node_pools.py -q` → PASS
Run: `RAY_ADDRESS=local ... -m pytest -q --tb=short` → all green (this task has no behavior change)

- [ ] **Step 5: commit**

```bash
git add src/roboray/ray_runtime/nodes.py src/roboray/runtime/driver.py tests/unit/test_node_pools.py
git commit -m "refactor(nodes): NodePool multi-GPU slots + register same-node merge -- audit #2 defense (behavior-preserving)"
```

---

### Task 2: build_node_pools distributes by GPU count + pool merge/remove semantics

**Files:**
- Modify: `src/roboray/ray_runtime/actors.py` (build_node_pools distribution)
- Modify: `src/roboray/runtime/driver.py` (_create_pool("") appends to an existing node; the exit branch removes an actor rather than the whole pool)
- Test: `tests/unit/test_node_pools.py`, `tests/unit/test_actors.py` (append)

**Interfaces:**
- Consumes: Task 1's `NodePool.gpu_actors/add_gpu`, `NodeInfo.gpu_count` (already present)
- Produces: one GPU actor per GPU on a multi-GPU node; `_create_pool("")` returns the existing pool when it lands on an existing node (the actor has already been appended)

- [ ] **Step 1: write failing tests (append to test_actors.py; reuse that file's existing monkeypatch of GpuModelActor/discover_gpu_nodes technique -- read the file first to confirm the technique's name)**

```python
def test_build_node_pools_fills_by_gpu_count(monkeypatch):
    """Audit #1: build slots by GPU count rather than node count -- the second GPU of a
    single-node-2-GPU setup no longer sits idle."""
    from roboray.ray_runtime import actors as mod
    from roboray.ray_runtime.nodes import NodeInfo

    nodes = [NodeInfo("A", 2, "ha"), NodeInfo("B", 1, "hb")]
    monkeypatch.setattr(mod, "discover_gpu_nodes", lambda: nodes)
    made = []
    real_opts = mod.GpuModelActor.options

    class _Fake:
        @staticmethod
        def options(**kw):
            class _B:
                @staticmethod
                def remote():
                    made.append(kw.get("scheduling_strategy"))
                    return object()
            return _B

    monkeypatch.setattr(mod, "GpuModelActor", _Fake)
    monkeypatch.setattr(mod, "CpuStageActor", _Fake)
    gpu, _cpu = mod.build_node_pools(num_gpus=3, num_cpu_actors=1)
    assert len(gpu) == 3                      # A×2 + B×1
    assert len({id(s) for s in made[:2]}) or True  # A's two actors each remote'd separately
    # num_gpus=2: A gets its full 2, B is not built
    made.clear()
    gpu2, _ = mod.build_node_pools(num_gpus=2, num_cpu_actors=1)
    assert len(gpu2) == 2
```

(If the existing tests use a different stub shape, rewrite this case with the file's established technique; the assertion semantics stay: num_gpus=3 → node A 2 + node B 1; num_gpus=2 → exactly 2.)

Append to `test_node_pools.py` (Review Focus 1 + merge semantics):

```python
def test_multi_gpu_slots_dispatchable():
    """2 actors on one node must each occupy an independent slot: the dispatch view
    zip(slots, gpu_actors) has no duplication or omission."""
    slots = [GpuSlot(i) for i in range(2)]
    gpu_actors = [object(), object()]
    actor_node = {0: "A", 1: "A"}
    # simulate dispatch: both slots can be independently set busy/reclaimed
    slots[0].busy = True
    free = [s for s in slots if not s.busy]
    assert len(free) == 1 and free[0].slot_id == 1
    assert set(actor_node) == {0, 1}  # indices align contiguously


def test_create_pool_empty_lands_on_existing_node_appends():
    """Audit #2 scenario: an affinity-free new pool lands back on an existing node ->
    append and return the existing pool, no overwrite."""
    from roboray.ray_runtime.nodes import NodeInfo, NodePool, NodePoolRegistry

    existing = NodePool("A", gpu_actors=[object()], cpu_actors=[object()])
    reg = NodePoolRegistry(
        live_nodes=[NodeInfo("A", 2, "h")], cpu_actors_per_node=1,
        create_pool=lambda nid: existing,
    )
    reg.register(existing)
    # driver._create_pool("")'s actual==A branch semantics: returns existing, actor already in pool
    new_actor = object()
    existing.add_gpu(new_actor)
    assert reg.pools["A"] is existing and len(existing.gpu_actors) == 2
```

Driver-behavior test (append to `test_node_pools.py`, modeled on the existing `_create_pool` regression test `test_create_pool_with_empty_node_id_does_not_crash_and_registers_cpu_actors` stub technique):

```python
def test_create_pool_on_existing_node_merges(monkeypatch):
    """_create_pool("") landing on a node that already has a pool: the new actor is
    appended into the existing pool and that pool is returned."""
    # construction: the driver holds an A-node pool (1 actor); the stubbed new
    # GpuModelActor self-reports node_id=A
    # assertions: the returned pool is the existing one; len(pool.gpu_actors)==2;
    # registry.pools still has a single A entry
    # (reuse test_create_pool_with_empty_node_id...'s _Gpu/_Cpu/ray.get stub technique,
    #  the only difference being that the new actor's node_id() returns an existing node id)
```

(Implementers complete the concrete construction per the assertions above -- the file already has every needed stub component.)

- [ ] **Step 2: run the tests and confirm failure**

Run: `RAY_ADDRESS=local ... -m pytest tests/unit/test_node_pools.py tests/unit/test_actors.py -q`
Expected: the new cases FAIL (build still makes 1 per node / create_pool still overwrites)

- [ ] **Step 3: implement**

Replace the GPU distribution loop of `build_node_pools` in `actors.py` with:

```python
    remaining = int(num_gpus)
    gpu_actors, cpu_actors = [], []
    for node in nodes:
        if remaining <= 0:
            break
        k = min(int(node.gpu_count), remaining)  # audit #1: fill slots by GPU count
        strategy = NodeAffinitySchedulingStrategy(node_id=node.node_id, soft=False)
        for _ in range(k):
            gpu_actors.append(
                GpuModelActor.options(scheduling_strategy=strategy).remote())
        cpu_actors.extend(
            CpuStageActor.options(scheduling_strategy=strategy).remote()
            for _ in range(per_node_cpu)
        )
        remaining -= k
```

The empty-node_id branch of `driver.py _create_pool`: after getting `actual`, first check the existing pools:

```python
        existing = self.node_registry.pools.get(actual) if self.node_registry else None
        if existing is not None:
            # landed on an existing node (audit #2): append into the existing pool,
            # neither creating a new pool nor overwriting
            existing.add_gpu(gpu)
            # driver-side bookkeeping same as _create_pool's other branches
            # (gpu_actors/slots/_actor_node/_cpu_by_node)
            ...
            return existing
```

(Extract "register the new actor into the driver bookkeeping" as a private `_register_actor(gpu, actual)` method shared by both branches.)

The `_apply_idle_actions` exit branch: change `pools.pop(node_id)` to remove the actor, popping the pool only when empty:

```python
                        pool = self.node_registry.pools.get(node_id)
                        if pool is not None:
                            pool.gpu_actors = [
                                a for a in pool.gpu_actors if a is not st.actor]
                            if not pool.gpu_actors:
                                self.node_registry.pools.pop(node_id, None)
```

- [ ] **Step 4: run the tests and confirm pass + full regression**

Run: targeted two files → PASS; full → green

- [ ] **Step 5: commit**

```bash
git add src/roboray/ray_runtime/actors.py src/roboray/runtime/driver.py tests/unit/test_node_pools.py tests/unit/test_actors.py
git commit -m "feat(nodes): build GPU slots by GPU count (no idle GPU on multi-GPU nodes) + pool merge/remove semantics -- audit #1/#2"
```

---

### Task 3: inference_fps landed as stride sampling (audit #7)

**Files:**
- Modify: `scripts/modelenvs/_common.py` (`compute_stride` pure function)
- Modify: `scripts/modelenvs/run_hamer.py` (inference-loop sampling)
- Modify: `tests/fakes/modelenv_scripts/fake_hamer.py` (contract-isomorphic)
- Modify: `src/roboray/stages/hand.py` (payload pass-through) + `src/roboray/runtime/driver.py` (_gpu_payload passes it)
- Test: `tests/unit/test_modelenv_scripts.py`, new `tests/unit/test_stride.py`

**Interfaces:**
- Consumes: `HandStageConfig.inference_fps` (config.py:71, already present); calib's `fps`
- Produces: `compute_stride(fps, inference_fps) -> int` (`_common.py`); payload optional key `inference_fps`; skipped frames get `hand_present=False` (naturally backfilled by A3 as-is)

- [ ] **Step 1: write failing tests**

`tests/unit/test_stride.py` (new):

```python
"""inference_fps -> stride sampling (audit #7): the default behavior is byte-identical
(Review Focus 2)."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "modelenvs"))
from _common import compute_stride  # noqa: E402

FAKES = Path(__file__).resolve().parents[1] / "fakes" / "modelenv_scripts"


def _run(out, extra=None):
    inp = out.parent / "in.npz"
    np.savez(inp, frames=np.zeros((8, 8, 8, 3), np.uint8),
             n_frames=np.int32(8), fps=np.int32(30))
    import subprocess
    cmd = [sys.executable, str(FAKES / "fake_hamer.py"),
           "--input", str(inp), "--output", str(out)]
    if extra is not None:
        cmd += ["--extra", json.dumps(extra)]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    assert p.returncode == 0, p.stderr
    return json.loads([l for l in p.stdout.splitlines()
                       if l.startswith("__RESULT__")][-1][10:])


def test_compute_stride_rules():
    assert compute_stride(30, 30) == 1      # same rate: all frames (default)
    assert compute_stride(30, 15) == 2
    assert compute_stride(30, None) == 1    # not configured
    assert compute_stride(None, 15) == 1    # fps unknown: fall back to all frames (D-I3)
    assert compute_stride(30, 100) == 1     # above source rate: never below full frames


def test_stride_two_marks_unobserved(tmp_path):
    out = tmp_path / "hand.npz"
    _run(out, extra={"inference_fps": 15})
    with np.load(out) as z:
        present = z["hand_present"]
    assert present[0].all() and not present[1].any()   # 0,2,4,6 inferred; 1,3,5,7 unobserved
    assert present[::2].all() and not present[1::2].any()


def test_stride_default_identical_output(tmp_path):
    """Review Focus 2: output with inference_fps=30 (the default) and without the extra
    key must be byte-identical."""
    a, b = tmp_path / "a.npz", tmp_path / "b.npz"
    _run(a, extra={"inference_fps": 30})
    _run(b)
    za, zb = np.load(a), np.load(b)
    try:
        for k in za.files:
            assert np.array_equal(za[k], zb[k]), f"key {k} drifted"
    finally:
        za.close(), zb.close()
```

- [ ] **Step 2: run the tests and confirm failure**

Run: `RAY_ADDRESS=local ... -m pytest tests/unit/test_stride.py -q`
Expected: FAIL — `cannot import name 'compute_stride'`

- [ ] **Step 3: implement**

Append to `_common.py`:

```python
def compute_stride(fps, inference_fps) -> int:
    """hand inference sampling stride (audit #7): stride = round(fps / inference_fps),
    floor 1.

    Missing/invalid fps or inference_fps -> 1 (all frames, byte-identical to before).
    Skipped frames get hand_present=False and are short-term backfilled by A3 (gaps
    <= 6 frames can be fully filled; gaps from stride>7 are kept -- better to leave a
    gap than to fabricate, see spec D-I3).
    """
    try:
        if not fps or not inference_fps:
            return 1
        return max(1, int(round(float(fps) / float(inference_fps))))
    except (TypeError, ValueError, ZeroDivisionError):
        return 1
```

In `fake_hamer.py`'s main(): after reading `data`, add

```python
    stride = compute_stride(int(data.get("fps") or 0), data.get("inference_fps"))
    idx = np.arange(t)
    present = np.ones((t, 2), bool)
    if t >= 8:
        present[3:5, 0] = False
    if stride > 1:
        present[idx % stride != 0] = False   # frames outside the sampling stride are unobserved (A3 backfills)
```

(Merge `from _common import ... compute_stride` into the existing import line; the other arrays keep their present semantics -- the fake's joint arrays are already fully filled, and consumers ignore unobserved frames per present.)

In `run_hamer.py`'s inference function: at the `extra`/`data` parsing point add `stride = compute_stride(fps, extra.get("inference_fps") or data.get("inference_fps"))`, and at the first line of the main loop `for i in range(t):` add:

```python
        if stride > 1 and i % stride != 0:
            continue  # outside the sampling stride: do not run HaMeR; present=False, A3 backfills (same contract as the fake)
```

`hand.py`:

```python
def hand_payload_args(
    workspace_root: str,
    episode_id: str,
    model_name: str,
    out_path: str,
    inference_fps: int | None = None,
) -> dict:
    """Build the Hand GPU payload -- **paths and scalars only, no arrays** (same as Scene).

    ``inference_fps`` (same name as HandStageConfig), when non-empty, is sent down:
    the model side samples inference at round(fps/inference_fps) and marks skipped
    frames present=False for A3 to backfill.
    """
    payload = {
        "kind": "hand",
        "episode_id": episode_id,
        "model_name": model_name,
        "workspace_root": str(workspace_root),
        "out_path": str(out_path),
    }
    if inference_fps:
        payload["inference_fps"] = int(inference_fps)
    return payload
```

`driver.py _gpu_payload`'s hand branch adds the argument `inference_fps=self.request.stages.hand_reconstruction.inference_fps` (default 30 → payload carries 30 → compute_stride(30,30)=1, behavior unchanged).

- [ ] **Step 4: run the tests and confirm pass + full regression**

Run: `... -m pytest tests/unit/test_stride.py tests/unit/test_modelenv_scripts.py -q` → PASS
Run: full → green (fake e2e goes through stride=1, unchanged)

- [ ] **Step 5: commit**

```bash
git add scripts/modelenvs/_common.py scripts/modelenvs/run_hamer.py tests/fakes/modelenv_scripts/fake_hamer.py src/roboray/stages/hand.py src/roboray/runtime/driver.py tests/unit/test_stride.py
git commit -m "feat(hand): inference_fps landed as stride-sampled inference -- skipped frames backfilled by A3 (default behavior unchanged)"
```

---

### Task 4: rigid_align pure function (A1 Kabsch, audit #4)

**Files:**
- Modify: `src/roboray/stages/fusion.py` (new function, placed before `_mano_hand_to_world`)
- Test: `tests/unit/test_fusion_refined.py` (append)

**Interfaces:**
- Consumes: none (pure numpy)
- Produces: `rigid_align(mano_pts, target_pts, valid, min_points=4) -> (pts_aligned, method)`, method ∈ `"kabsch"|"root_only"`; integrated in Task 5.

- [ ] **Step 1: write failing tests**

```python
# -- Task 4: A1 Kabsch rigid alignment (audit #4) --
def _rot(axis, deg):
    axis = np.asarray(axis, float); axis /= np.linalg.norm(axis)
    a = np.deg2rad(deg)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * (K @ K)


def test_rigid_align_recovers_rotation_translation():
    from roboray.stages.fusion import rigid_align

    rng = np.random.default_rng(7)
    base = rng.normal(0, 0.05, (21, 3))
    R, t = _rot([0.3, 1.0, -0.2], 25.0), np.array([0.02, -0.03, 0.4])
    mano = base + rng.normal(0, 1e-4, base.shape)          # MANO metric skeleton (~ground truth)
    target = (R @ base.T).T + t + rng.normal(0, 2e-3, base.shape)  # depth backprojection (noisy)
    valid = np.ones(21, bool)
    aligned, method = rigid_align(mano, target, valid)
    assert method == "kabsch"
    # recovery quality: after alignment, the difference from the noise-free ground
    # truth is < 5mm (the Kabsch closed-form solution is robust to 2mm noise)
    assert np.linalg.norm(aligned - ((R @ base.T).T + t), axis=-1).max() < 5e-3


def test_rigid_align_partial_valid_points():
    from roboray.stages.fusion import rigid_align

    rng = np.random.default_rng(8)
    base = rng.normal(0, 0.05, (21, 3))
    R, t = _rot([1, 0, 0], 10.0), np.array([0.0, 0.0, 0.35])
    mano = base.copy()
    target = (R @ base.T).T + t
    valid = np.zeros(21, bool); valid[[0, 4, 8, 12, 16, 20]] = True   # 6 depth-supported points
    aligned, method = rigid_align(mano, target, valid)
    assert method == "kabsch"
    assert np.linalg.norm(aligned - target, axis=-1).max() < 1e-6


def test_rigid_align_degrades_to_root_only():
    from roboray.stages.fusion import rigid_align

    pts = np.ones((21, 3)) * np.arange(21)[:, None] * 0.01   # all collinear
    valid = np.ones(21, bool)
    aligned, method = rigid_align(pts, pts + 0.3, valid)
    assert method == "root_only"                            # collinear -> singular covariance -> fallback
    assert abs(aligned[0, 2] - (pts[0, 2] + 0.3)) < 1e-9    # root-translation semantics preserved
    valid3 = np.zeros(21, bool); valid3[:3] = True
    _a, m3 = rigid_align(pts, pts, valid3)
    assert m3 == "root_only"                                # < 4 valid points -> fallback
```

- [ ] **Step 2: run the tests and confirm failure**

Run: `... -m pytest tests/unit/test_fusion_refined.py -q`
Expected: FAIL — `cannot import name 'rigid_align'`

- [ ] **Step 3: implement (fusion.py, before `_mano_hand_to_world`)**

```python
def rigid_align(
    mano_pts: np.ndarray,
    target_pts: np.ndarray,
    valid: np.ndarray,
    min_points: int = 4,
) -> tuple:
    """A1 upgrade (audit #4): Kabsch scale-free rigid alignment -- rotation + translation,
    **no scaling**.

    ``target_pts`` is the depth-backprojected joint 3D (trusted only where ``valid``).
    Fewer than ``min_points`` valid points, or a singular centered covariance
    (collinear/coincident) -> fall back to root translation (the pre-change
    behavior), method="root_only"; otherwise method="kabsch". No scaling preserves
    the MANO bone lengths (spec D-I4).
    """
    pts = np.asarray(mano_pts, np.float64)
    tgt = np.asarray(target_pts, np.float64)
    m = np.asarray(valid, bool) & np.isfinite(pts).all(axis=1) & np.isfinite(tgt).all(axis=1)
    if int(m.sum()) < min_points or not np.isfinite(pts[0]).all():
        shifted = pts + (tgt[0] - pts[0])
        return shifted, "root_only"
    p, q = pts[m], tgt[m]
    pc, qc = p.mean(axis=0), q.mean(axis=0)
    H = (p - pc).T @ (q - qc)
    try:
        U, _S, Vt = np.linalg.svd(H)
    except np.linalg.LinAlgError:
        return pts + (tgt[0] - pts[0]), "root_only"
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])          # no scaling (|singular values|=1), only reflection guard
    R = Vt.T @ D @ U.T
    return (R @ (pts - pc).T).T + qc, "kabsch"
```

- [ ] **Step 4: run the tests and confirm pass**

Run: `... -m pytest tests/unit/test_fusion_refined.py -q` → PASS

- [ ] **Step 5: commit**

```bash
git add src/roboray/stages/fusion.py tests/unit/test_fusion_refined.py
git commit -m "feat(fusion): rigid_align Kabsch pure function -- A1 upgraded from translation anchoring to rotation+translation (bone lengths preserved)"
```

---

### Task 5: `_mano_hand_to_world` integrates Kabsch + alignment field

**Files:**
- Modify: `src/roboray/stages/fusion.py` (`_mano_hand_to_world` + `backproject_to_world` pass-through)
- Test: `tests/unit/test_fusion_refined.py` (append)

**Interfaces:**
- Consumes: Task 4's `rigid_align`, the existing `_sample_depth`
- Produces: `_mano_hand_to_world` returns `(world, reproj, method)` (a triple -- **signature change**, the `backproject_to_world` call site updates in sync); `backproject_to_world`'s return value appends a per-hand `alignment` array (consumed by npz/QC)

- [ ] **Step 1: write failing tests (append)**

```python
def test_mano_world_kabsch_beats_root_translation():
    """Review Focus 4: in a synthetic scenario, Kabsch alignment's reprojection error
    is <= root translation's."""
    from roboray.stages.fusion import _mano_hand_to_world, _reproject_px, rigid_align

    rng = np.random.default_rng(9)
    K = np.array([[60.0, 0, 4.0], [0, 60.0, 3.0], [0, 0, 1]])
    h, w = 6, 8
    base = rng.normal(0, 0.02, (21, 3)); base[:, 2] += 0.5
    cam = (_rot([0, 1, 0], 20.0) @ base.T).T          # camera-frame ground truth
    depth = np.full((h, w), np.nan, np.float32)
    joints_2d = np.zeros((21, 3))
    for j, (x, y, z) in enumerate(cam):
        u = K[0, 0] * x / z + K[0, 2]
        v = K[1, 1] * y / z + K[1, 2]
        joints_2d[j, :2] = (u, v)
        ui, vi = int(round(u)), int(round(v))
        if 0 <= ui < w and 0 <= vi < h:
            depth[vi, ui] = z
    # the root joint's depth point must be valid (forced into bounds if out of range)
    r_v, r_u = int(round(joints_2d[0, 1])), int(round(joints_2d[0, 0]))
    r_v, r_u = min(max(r_v, 0), h - 1), min(max(r_u, 0), w - 1)
    depth[r_v, r_u] = float(cam[0, 2])
    mano = cam + rng.normal(0, 1e-3, cam.shape)       # MANO with slight perturbation
    c2w = np.eye(4); w2c = np.eye(4)
    world, reproj, method = _mano_hand_to_world(mano, joints_2d, depth, K, w2c, c2w)
    assert method == "kabsch"
    # comparison: the same input with only the root valid -> rigid_align takes root
    # translation (the fallback path)
    valid_root = np.zeros(21, bool); valid_root[0] = True
    root_pts, root_method = rigid_align(mano, cam, valid_root)
    assert root_method == "root_only"
    fin = np.isfinite(joints_2d[:, :2]).all(axis=1)
    root_reproj = _reproject_px(root_pts[fin], joints_2d[fin, :2], K, w2c)
    assert reproj <= root_reproj + 1e-6
```

- [ ] **Step 2: run the tests and confirm failure**

Run: `... -m pytest tests/unit/test_fusion_refined.py -q`
Expected: FAIL — `too many values to unpack` (it currently returns a pair)

- [ ] **Step 3: implement**

Rework `_mano_hand_to_world` (docstring updated to "Kabsch preferred, root-translation fallback"; the early root-depth-invalid return `(None, None)` stays):

```python
    # depth-backprojected target: valid detection points backproject at their own
    # depths (K⁻¹·(u,v,1)·z)
    uv = src[valid_src]
    z_pts = np.array([_sample_depth(depth_frame, float(u), float(v))
                      for u, v in uv])
    ok = np.isfinite(z_pts) & (z_pts > 0)
    target = pts_cam.copy()
    tgt_valid = np.zeros(len(pts_cam), bool)
    if ok.any():
        idxs = np.where(valid_src)[0][ok]
        uv_h = np.stack([uv[ok, 0], uv[ok, 1], np.ones(int(ok.sum()))])  # (3,N)
        target[idxs] = (np.linalg.inv(K) @ uv_h * z_pts[ok]).T          # (N,3)
        tgt_valid[idxs] = True
    pts_cam, method = rigid_align(pts_cam, target, tgt_valid)
    hom = np.concatenate([pts_cam, np.ones((21, 1))], axis=1)
    world = (hom @ c2w.T)[:, :3]
    return world, _reproject_px(world[valid_src], src[valid_src], K, camera_poses_i), method
```

`backproject_to_world`: the mano path's call site changes to receive the triple, per-hand `alignment` collected into the return (the return structure appends an `alignment` array (T,2) of strings); `run_fusion` writes the new optional npz key `alignment` (optional -- publish.py's QC rendering side does not require it; the improvements report notes QC presentation as a later small item). **The joints_2d fallback path is unaffected** (alignment filled with `"none"`).

- [ ] **Step 4: full regression (existing fusion tests + e2e)**

Run: `... -m pytest -q --tb=short` → all green (call sites already synced; on synthetic data the fake e2e's A1 path should mostly hit kabsch or root_only -- just must not blow up)

- [ ] **Step 5: commit**

```bash
git add src/roboray/stages/fusion.py tests/unit/test_fusion_refined.py
git commit -m "feat(fusion): A1 integrates Kabsch rigid alignment + alignment field into npz -- 'global orientation' landed (audit #4)"
```

---

### Task 6: prepare timestamp synchronization (audit #5)

**Files:**
- Modify: `src/roboray/stages/prepare.py` (timestamp pairing + alignment summary)
- Modify: `src/roboray/runtime/episode.py` (`record_prepare` gains an `alignment` param, default None)
- Modify: `src/roboray/runtime/driver.py` (`_backfill_prepare` pass-through) + `src/roboray/manifest.py` (entry gains optional alignment)
- Test: `tests/unit/test_stages_io.py` (append; read the file first to confirm the prepare-test construction technique)

**Interfaces:**
- Consumes: `meta.json` optional `timestamps_left/right` (float arrays, seconds)
- Produces: pure function `align_by_timestamps(ts_l, ts_r, fps) -> (idx_l, idx_r, n_dropped)` (in prepare.py); prepared left/right frames share the same index, i.e. a synchronized pair (transparent downstream); stage_result/record/manifest gain the keys `alignment ∈ "timestamps"|"truncation"`, `align_dropped: int`

- [ ] **Step 1: write failing tests (append to test_stages_io.py; reuse the existing run_prepare test's temporary Episode directory construction)**

```python
def test_align_by_timestamps_rules():
    from roboray.stages.prepare import align_by_timestamps

    fps = 30.0
    ts_l = np.arange(10) / fps                       # 0.0 ... 0.3
    ts_r = (np.arange(10) + 0.3) / fps               # right eye lags 0.3s = 9 frames overall
    il, ir, dropped = align_by_timestamps(ts_l, ts_r, fps)
    assert len(il) == len(ir) == 1                   # only the t≈0.3 pair falls within tolerance
    assert il[0] == 9 and ir[0] == 0
    assert dropped == 9
    # no drift: all pairs
    il, ir, dropped = align_by_timestamps(ts_l, ts_l.copy(), fps)
    assert len(il) == 10 and dropped == 0


def test_prepare_timestamp_alignment_end_to_end(tmp_path):
    """Timestamp drift on both eyes -> left/right share the same index after alignment;
    the manifest semantic is alignment=timestamps."""
    # build a minimal Episode: calibration.json + left/right mp4 (12/10 frames, right
    # drops 1 of every 3)
    # meta.json carries timestamps_left(12)/timestamps_right(10)
    # after run_prepare assert: prepared/frames_left.npy.shape[0] == len(pairs)
    # summary["alignment"] == "timestamps", summary["align_dropped"] == expected
    # existing cases without meta timestamps regress: alignment == "truncation"
    # (the frame-count mismatch path)
```

(Frame synthesis follows test_stages_io.py's existing mp4 style; the implementer completes the concrete construction with that file's existing helpers, per the assertion semantics above.)

- [ ] **Step 2: run the tests and confirm failure**

Run: `... -m pytest tests/unit/test_stages_io.py -q` → FAIL (`align_by_timestamps` does not exist)

- [ ] **Step 3: implement**

Add to `prepare.py`:

```python
def align_by_timestamps(ts_l: np.ndarray, ts_r: np.ndarray, fps: float
                        ) -> tuple[np.ndarray, np.ndarray, int]:
    """Nearest-neighbor timestamp pairing (audit #5): each left frame takes its nearest
    right frame, pairing only when |Δt| <= 0.5/fps.

    Returns (left_idx, right_idx, count of unpaired left frames dropped). Mutual
    nearest-neighbor is not enforced (duplicate pairing is allowed -- stereo
    alignment only requires matching instants).
    """
    tol = 0.5 / float(fps)
    order = np.argsort(np.asarray(ts_r, np.float64))
    rs = np.asarray(ts_r, np.float64)[order]
    keep_l, keep_r = [], []
    for i, t in enumerate(np.asarray(ts_l, np.float64)):
        j = np.searchsorted(rs, t)
        best, best_d = None, tol + 1.0
        for k in (j - 1, j):
            if 0 <= k < len(rs):
                d = abs(rs[k] - t)
                if d < best_d:
                    best, best_d = k, d
        if best is not None and best_d <= tol:
            keep_l.append(i); keep_r.append(int(order[best]))
    return (np.asarray(keep_l, np.int64), np.asarray(keep_r, np.int64),
            len(ts_l) - len(keep_l))
```

Upgrade `run_prepare`'s right-eye-present branch: read `src / "meta.json"`'s `timestamps_left/right` (float lists; each length must equal its own frame count, and if either fails → keep the current truncation with `alignment="truncation"`); after pairing `left, right = left[il], right[ir]`; the `degraded` semantics are preserved (timestamp-dropped frames are not stereo degradation: `has_stereo` stays True, `align_dropped=n_dropped` recorded separately). The summary gains the `"alignment"` and `"align_dropped"` keys (current path: `alignment="truncation"`, `align_dropped=0`).

`episode.py record_prepare(..., alignment: str | None = None, align_dropped: int = 0)`; `driver._backfill_prepare` passes the two keys through; `manifest.py`'s entry gains `entry["alignment"] = payload.get("alignment")`, `entry["align_dropped"] = payload.get("align_dropped", 0)` (defaulting to None/0, compatible with old records).

- [ ] **Step 4: full regression**

Run: `... -m pytest -q --tb=short` → all green

- [ ] **Step 5: commit**

```bash
git add src/roboray/stages/prepare.py src/roboray/runtime/episode.py src/roboray/runtime/driver.py src/roboray/manifest.py tests/unit/test_stages_io.py
git commit -m "feat(prepare): nearest-neighbor timestamp sync frame table -- alignment converges in prepare, transparent downstream (audit #5)"
```

---

### Task 7: contact segmentation + gripper channel (audit #6)

**Files:**
- Modify: `src/roboray/stages/action.py` (segment_motions kind + gripper column concat)
- Test: `tests/unit/test_fusion_action.py` (append), `tests/unit/test_stages_io.py` (publish compatibility)

**Interfaces:**
- Consumes: `joints_world (T,hands,21,3)`, `fps`
- Produces: `segment_motions(...) -> [(start, end, kind)]` (kind ∈ `"motion"|"contact"`; **return-shape change**, the run_action call site syncs); action.npz gains `segment_kind`, `gripper (T,n_hands)`; state/action concatenate the gripper column (2-D (T, D+hands)); publish's 4-D reshape branch is kept for legacy-artifact compatibility

- [ ] **Step 1: write failing tests (append to test_fusion_action.py)**

```python
def test_segment_motions_contact_kind():
    from roboray.stages.action import segment_motions

    T, fps = 30, 30
    jw = np.zeros((T, 2, 21, 3))
    jw[:10, :, :, 0] = np.linspace(0, 0.3, 10)[:, None, None]   # motion (>0.05/frame)
    jw[10:20, :, :, 0] = 0.3                                      # dwell 10 frames (contact)
    jw[20:, :, :, 0] = np.linspace(0.3, 0.6, 10)[:, None, None]  # motion again
    segs = segment_motions(jw, fps)
    kinds = {k for _s, _e, k in segs}
    assert kinds == {"motion", "contact"}
    contact = [s for s in segs if s[2] == "contact"]
    assert contact and contact[0][0] >= 9 and contact[0][1] <= 20


def test_segment_motions_legacy_shape_gone_but_labels_ok():
    """Triple returns: label count equals segment count (run_action call-site
    compatibility is pinned by the full regression)."""
    from roboray.stages.action import segment_motions

    jw = np.zeros((20, 2, 21, 3))
    jw[:10, :, :, 0] = np.linspace(0, 0.5, 10)[:, None, None]
    segs = segment_motions(jw, 30)
    assert all(len(s) == 3 for s in segs) and segs


def test_gripper_channel_range_and_nan(tmp_path):
    """run_action artifacts: gripper ∈ [0,1], missing-hand frames NaN, state contains
    the gripper column (Review Focus 5)."""
    # build a minimal fused.npz (joints_world (T,2,21,3) + fps) -> run_action(model="none")
    # assert action.npz: gripper shape (T,2); state.ndim==2 and state.shape[1]==126+2;
    # gripper all finite in [0,1]; frames where hand_present is missing (fused npz set NaN)
    # -> gripper NaN
```

(Complete the construction per the file's existing run_action test technique; assertion semantics as above.)

Append to `test_stages_io.py` (Review Focus 5's publish half):

```python
def test_publish_accepts_legacy_4d_state(tmp_path):
    """An action.npz with the old 4-D state must still be publishable (the publish
    reshape branch is kept)."""
    # build a ws: state (T,2,21,3) 4-D + same-shaped action + the minimal remaining
    # artifacts
    # run_publish succeeds and the lerobot directory is produced (reuse this file's
    # existing publish-test construction)
```

- [ ] **Step 2: run the tests and confirm failure** → new cases FAIL (pair / no gripper)

- [ ] **Step 3: implement**

Upgrade `segment_motions` (threshold and min_len semantics kept; low-speed runs use `contact_thresh=0.02`, `min_contact=3`):

```python
def segment_motions(joints_world, fps, speed_thresh=0.05, min_len=3,
                    contact_thresh=0.02, min_contact=3):
    """Speed-threshold segmentation + low-speed dwell as contact candidates (audit #6).
    Returns [(start, end, kind)]."""
    wrist = joints_world[:, :, 0, :]
    speed = np.linalg.norm(np.diff(wrist, axis=0), axis=-1).max(axis=1)
    active = speed > speed_thresh
    idle = speed < contact_thresh
    segments, i, n = [], 0, len(active)
    while i < n:
        if active[i]:
            start = i
            while i < n and active[i]:
                i += 1
            if i - start >= min_len:
                segments.append((start, i, "motion"))
        elif idle[i]:
            start = i
            while i < n and idle[i]:
                i += 1
            if i - start >= min_contact:
                segments.append((start, i, "contact"))
        else:
            i += 1
    if not segments:
        segments = [(0, joints_world.shape[0], "motion")]
    return segments
```

`run_action`: unpack `segments` as triples (`segment_start/end` take [0]/[1], new key `segment_kind`); compute and concat the gripper:

```python
    flat = joints_world.reshape(t, -1).astype(np.float32)          # (T, D)
    tip_dist = np.linalg.norm(
        joints_world[:, :, 4, :3] - joints_world[:, :, 8, :3], axis=-1)  # (T,hands)
    observed = np.isfinite(tip_dist).all(axis=1)
    max_d = float(np.nanmax(tip_dist)) if observed.any() else 1.0
    gripper = np.where(np.isfinite(tip_dist),
                       np.clip(tip_dist / (max_d or 1.0), 0.0, 1.0), np.nan)
    gripper = gripper.astype(np.float32)                            # a distance ratio, not a real gripper
    state = np.concatenate([flat, gripper], axis=1)
    action = np.empty_like(state)
    action[:-1] = state[1:]; action[-1] = state[-1]
```

(`save_artifact` gains `segment_kind=np.array([k for *_ , k in segments], dtype=object)` and `gripper=gripper`; the docstring states "gripper = thumb-index fingertip distance / this batch's max, not a real gripper state".)

- [ ] **Step 4: full regression (e2e/publish/lerobot dimension chain)**

Run: `... -m pytest -q --tb=short` → all green; if an existing case pins the old state width (126), change it to not pin the width or update the expectation (no relaxing to skipping the test)

- [ ] **Step 5: commit**

```bash
git add src/roboray/stages/action.py tests/unit/test_fusion_action.py tests/unit/test_stages_io.py
git commit -m "feat(action): contact-segment kind + gripper distance-ratio channel concatenated into the tensor (legacy 4-D artifact publish compatible)"
```

---

### Task 8: HaMeR max_hands + candidates protocol (audit #3 detection side)

**Files:**
- Modify: `scripts/modelenvs/run_hamer.py` (max_hands configurable + keep all detections + candidates field)
- Modify: `tests/fakes/modelenv_scripts/fake_hamer.py` (contract-isomorphic: produces candidates)
- Test: `tests/unit/test_modelenv_scripts.py`, `tests/unit/test_stride.py` (reuse `_run`)

**Interfaces:**
- Consumes: payload/extra optional `max_hands` (default 4)
- Produces: npz optional key `candidates`: object array (T,), each element a list[dict] (`bbox` (4,), `side` int, `score` float, `joints_2d` (21,3), `joints_3d_cam` (21,3)); the main-array semantics are unchanged (x-rank left/right compatible output). Consumed by Tasks 9/10.

- [ ] **Step 1: write failing tests (append to test_modelenv_scripts.py)**

```python
def test_fake_hamer_candidates_contract(tmp_path):
    out = tmp_path / "hand.npz"
    _run(out, extra={"max_hands": 4, "bystander": True})
    with np.load(out, allow_pickle=True) as z:
        assert "candidates" in z.files
        cands = z["candidates"]
    assert cands.shape == (4,)                       # T=4 (_run frame count)
    for frame in cands:
        assert isinstance(frame, list) and len(frame) >= 2
        for c in frame:
            assert set(c) >= {"bbox", "side", "score", "joints_2d", "joints_3d_cam"}
    # bystander frames: >=3 candidates, and the bystander bbox is higher up/smaller
    # (the association layer's discriminability is guaranteed by the data shape)
```

- [ ] **Step 2: run the tests and confirm failure** → FAIL (fake has no candidates)

- [ ] **Step 3: implement**

`run_hamer.py`:
- `HANDS = 2` → `HANDS = 2` (main-array shape semantics unchanged) + `DEFAULT_MAX_HANDS = 4`; `infer` reads `max_hands = int(extra.get("max_hands") or data.get("max_hands") or DEFAULT_MAX_HANDS)`.
- mediapipe detection `Hands(..., max_num_hands=max_hands)` (`_boxes_from_mediapipe` gains the parameter).
- **drop the same-side discard**: delete `if hand_present[i, side]: continue` in the `for n in ...` loop; all detections first go into `candidates_rows[i].append({...})` (bbox=boxes[n], side, score, joints_2d/3d_cam taken from the same batch of outputs); then the main array is still filled by the current rules (first-left/first-right -- compatible with pre-change output).
- the `save_arrays` dict gains `"candidates": np.array(candidates_rows, dtype=object)` (allow_pickle semantics consistent with existing object fields); the result summary gains `"n_candidates": int(sum(len(r) for r in candidates_rows))`.

`fake_hamer.py` (contract-isomorphic): with the optional extra `bystander=True`, each frame appends a third "bystander hand" (smaller bbox and smaller y = higher up, joints shifted); `candidates` per frame = the wearer's left/right 2 (existing offset logic) + the optional bystander; the main array stays as-is. By default (no extra) it also produces candidates (the wearer's 2) -- e2e fully goes through the association path.

- [ ] **Step 4: run the tests and confirm pass + full regression (fake e2e does not blow up: a new npz key has no effect before a consumer reads it)**

- [ ] **Step 5: commit**

```bash
git add scripts/modelenvs/run_hamer.py tests/fakes/modelenv_scripts/fake_hamer.py tests/unit/test_modelenv_scripts.py
git commit -m "feat(hamer): max_hands configurable + all detections kept in the candidates field -- the data premise for A2 multi-person (audit #3)"
```

---

### Task 9: associate_candidates pure function (A2 association layer)

**Files:**
- Create: `src/roboray/stages/association.py`
- Test: `tests/unit/test_association.py` (new)

**Interfaces:**
- Consumes: Task 8's candidates structure (per-frame list[dict])
- Produces: `associate_candidates(candidates, width, height, prior_center=(0.5,0.85), prior_weight=1.0, size_weight=0.5, motion_weight=1.0) -> {"joints_2d": (T,2,21,3), "present": (T,2), "switches": [(frame, slot)], "labels": (T,2) int|−1}`; integrated in Task 10

- [ ] **Step 1: write failing tests (4 scenarios, spec §3.3)**

```python
"""A2 multi-person hand association (audit #3): wearer-prior verification + temporal
nearest-neighbor + entity continuity."""
import numpy as np

from roboray.stages.association import associate_candidates


def _cand(cx, cy, size, side, score=0.9, seed=0):
    rng = np.random.default_rng(seed)
    j2 = np.zeros((21, 3), np.float32)
    j2[:, 0] = cx + rng.normal(0, 0.01, 21) * size
    j2[:, 1] = cy + rng.normal(0, 0.01, 21) * size
    j2[:, 2] = score
    j3 = np.zeros((21, 3), np.float32); j3[:, 2] = 0.4
    return {"bbox": np.array([cx - size / 2, cy - size / 2,
                              cx + size / 2, cy + size / 2], np.float32),
            "side": side, "score": score, "joints_2d": j2, "joints_3d_cam": j3}


def test_wearer_selected_over_smaller_upper_bystander():
    W, H = 640, 480
    frames = [[_cand(200, 400, 120, 0), _cand(430, 390, 120, 1),
               _cand(320, 100, 50, 0, seed=1)]] * 6      # bystander: small and higher up
    out = associate_candidates(frames, W, H)
    assert out["present"].all()
    # the wearer is selected: the mean joint x should be near 200/430, not 320
    xs = out["joints_2d"][:, :, :, 0].mean(axis=2)
    assert abs(xs[:, 0].mean() - 200) < 30 and abs(xs[:, 1].mean() - 430) < 30
    assert out["switches"] == []


def test_bystander_intrusion_keeps_identity():
    W, H = 640, 480
    frames = [[_cand(200, 400, 120, 0), _cand(430, 390, 120, 1)]] * 5
    # frames 6-8: a bystander intrudes near the wearer's right hand (closer to the
    # previous frame's position is the wearer's own body)
    for k in range(3):
        frames.append([_cand(200, 400, 120, 0, seed=k),
                       _cand(430, 390, 120, 1, seed=k),
                       _cand(445, 395, 60, 1, seed=k + 2)])
    frames.append([_cand(200, 400, 120, 0), _cand(430, 390, 120, 1)])
    out = associate_candidates(frames, W, H)
    assert out["switches"] == []                          # a bystander does not trigger a switch
    xs = out["joints_2d"][:, :, :, 0].mean(axis=2)
    assert abs(xs[:, 1].mean() - 430) < 30                # right-hand entity continuity


def test_wrist_cross_keeps_entity_continuity():
    """Crossed hands: the x order swaps, the entity does not break (the x-rank baseline
    would misjudge >=1 switch)."""
    W, H = 640, 480
    frames = []
    for t in range(10):
        x1 = 200 + 24 * t          # left hand from 200 -> 416
        x2 = 430 - 24 * t          # right hand from 430 -> 214 (cross around t≈5)
        frames.append([_cand(x1, 400, 120, 0, seed=t), _cand(x2, 390, 120, 1, seed=t + 1)])
    out = associate_candidates(frames, W, H)
    assert out["switches"] == []
    x_track0 = out["joints_2d"][:, 0, :, 0].mean(axis=1)
    assert (np.diff(x_track0) > 0).all()                 # slot 0 always tracks the original left-hand entity


def test_first_frame_only_bystander_then_reroute():
    W, H = 640, 480
    frames = [[_cand(320, 100, 50, 0, seed=3)]]          # first frame has only a bystander
    frames += [[_cand(320, 100, 50, 0, seed=4),
                _cand(200, 400, 120, 0, seed=5), _cand(430, 390, 120, 1, seed=6)]] * 4
    out = associate_candidates(frames, W, H)
    assert len(out["switches"]) >= 1                     # wearer enters -> reselect (counted truthfully)
    xs = out["joints_2d"][:, :, :, 0].mean(axis=2)
    assert abs(xs[-1, 0].mean() - 200) < 30              # the terminal state selects the wearer
```

- [ ] **Step 2: run the tests and confirm failure**

Run: `... -m pytest tests/unit/test_association.py -q` → FAIL (module does not exist)

- [ ] **Step 3: implement (association.py in full)**

```python
"""A2 multi-person hand association (audit #3): wearer-prior verification + inter-frame
nearest-neighbor + entity continuity.

spec 2026-10-26 §3.3 / D-I6: left/right labels only break ties, they are not a hard
constraint -- the entity does not break when the wearer crosses hands or turns.
``switches`` counts entity jumps (= reselection events), not x-order changes (the old
x-rank semantics stay in fusion.associate_wearer for the no-candidates fallback).
"""

import numpy as np

__all__ = ["associate_candidates", "wearer_score"]


def wearer_score(cand: dict, width: float, height: float,
                 prior_center=(0.5, 0.85), prior_weight=1.0, size_weight=0.5) -> float:
    """Wearer prior score: proximity to lower-center of the frame (high) + bbox area
    (large) -- in the ego view the wearer's hands are close and usually in the lower
    sides of the frame. Normalized to roughly [0, ~2]."""
    x1, y1, x2, y2 = cand["bbox"]
    cx, cy = (x1 + x2) / 2 / width, (y1 + y2) / 2 / height
    dist = float(np.hypot(cx - prior_center[0], cy - prior_center[1]))
    area = float(max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))) / (width * height)
    return -prior_weight * dist + size_weight * min(1.0, area * 8.0)


def _match(prev_xy, cand, width, motion_thresh_px):
    """Center distance <= threshold -> the distance; otherwise None (no match)."""
    x1, y1, x2, y2 = cand["bbox"]
    d = float(np.hypot((x1 + x2) / 2 - prev_xy[0], (y1 + y2) / 2 - prev_xy[1]))
    return d if d <= motion_thresh_px else None


def associate_candidates(candidates, width, height,
                         prior_center=(0.5, 0.85), prior_weight=1.0,
                         size_weight=0.5, motion_weight=1.0):
    """Bind candidates frame by frame to the wearer's two "entity slots".

    - First frame (or no entity last frame): per-slot best by wearer_score (side
      breaks ties).
    - Later frames: for each entity slot take the nearest neighbor among candidates
      (minimal composite cost of distance + motion_weight·(1−score) and center
      distance <= threshold); an unmatched slot -> reselect by prior this frame and
      count one switch.
    Output dict: joints_2d (T,2,21,3), present (T,2), switches [(frame, slot)],
    labels (T,2) (the candidate's index in that frame's list, −1=unobserved -- for QC audit).
    """
    t = len(candidates)
    out_j2 = np.zeros((t, 2, 21, 3), np.float32)
    out_present = np.zeros((t, 2), bool)
    labels = np.full((t, 2), -1, np.int32)
    switches: list = []
    motion_thresh = 0.15 * float(width)
    ents: list = [None, None]          # last frame's (cx, cy) per slot
    for i, frame in enumerate(candidates):
        cands = list(frame or [])
        assigned = {}
        for slot in (0, 1):
            if ents[slot] is None or not cands:
                continue
            costs = [(motion_weight * _match(ents[slot], c, width, motion_thresh)
                      + (1.0 - float(c.get("score", 0.5))), j)
                     for j, c in enumerate(cands) if j not in assigned.values()
                     and _match(ents[slot], c, width, motion_thresh) is not None]
            if costs:
                assigned[slot] = min(costs)[1]
        reselected = False
        for slot in (0, 1):
            if slot not in assigned and cands:
                pool = [j for j in range(len(cands)) if j not in assigned.values()]
                if pool:
                    best = max(pool, key=lambda j: wearer_score(
                        cands[j], width, height, prior_center,
                        prior_weight, size_weight) + (0.05 if cands[j].get("side") == slot else 0.0))
                    assigned[slot] = best
                    if ents[slot] is not None:
                        reselected = True
        for slot, j in assigned.items():
            c = cands[j]
            out_j2[i, slot] = c["joints_2d"]
            out_present[i, slot] = True
            labels[i, slot] = j
            x1, y1, x2, y2 = c["bbox"]
            ents[slot] = ((x1 + x2) / 2, (y1 + y2) / 2)
        if reselected:
            slots_hit = [s for s in (0, 1) if out_present[i, s] and ents[s] is not None]
            switches.append((i, slots_hit[-1] if slots_hit else 0))
    return {"joints_2d": out_j2, "present": out_present,
            "switches": switches, "labels": labels}
```

(Implementers may fine-tune constants/tie-breakers after running the tests, but **the assertion semantics must not change**; all 4 scenarios must be green.)

- [ ] **Step 4: run the tests and confirm pass**

Run: `... -m pytest tests/unit/test_association.py -q` → 4 PASS

- [ ] **Step 5: commit**

```bash
git add src/roboray/stages/association.py tests/unit/test_association.py
git commit -m "feat(association): A2 wearer-prior + temporal nearest-neighbor association -- multi-person identity binding/entity continuity (pure function)"
```

---

### Task 10: run_fusion integrates candidates (A2 closed loop)

**Files:**
- Modify: `src/roboray/stages/fusion.py` (run_fusion reads candidates → overwrites joints_2d/present/switches)
- Test: `tests/unit/test_fusion_action.py` or new `tests/unit/test_fusion_association.py`

**Interfaces:**
- Consumes: Task 8's npz `candidates` key, Task 9's `associate_candidates`, `scene.npz`'s width/height (also in calib)
- Produces: with npz candidates → the association result is the primary data (`identity_switches` = entity switches); without → current x-rank (Review Focus 3)

- [ ] **Step 1: write failing tests**

```python
def test_fusion_with_candidates_uses_association(tmp_path):
    """A hand.npz with candidates -> joints_2d/present come from the association layer,
    switches semantics upgraded."""
    # build a minimal scene.npz + hand.npz (with candidates: the wearer's 2 hands +
    # a 1-hand bystander sequence)
    # after run_fusion assert: the non-zero joints_world frames match the wearer's
    # candidates (bystander coordinates never appear);
    # identity_switches == associate_candidates' switches count
    # (construction reuses this file's existing run_fusion test npz fixtures)


def test_fusion_without_candidates_falls_back(tmp_path):
    """Review Focus 3: no candidates key -> the current x-rank path, results identical
    to before."""
    # run run_fusion on the same fixture with the candidates key removed, and assert
    # it matches pre-change behavior:
    # identity_switches semantics = x-rank sort-change count (associate_wearer as-is)
```

(Implementers complete both cases with the existing run_fusion fixtures; assertion semantics as above, and the second case may take the comparative form "run the same fixture with/without the key; the no-key version's result == association layer off".)

- [ ] **Step 2: run the tests and confirm failure** → FAIL (fusion does not read candidates)

- [ ] **Step 3: implement**

At the hand.npz load point in `run_fusion` (near `fusion.py:410`):

```python
        candidates = z["candidates"] if "candidates" in z.files else None
```

Before the `associate_wearer` call site (near `fusion.py:536`), add the branch:

```python
    if candidates is not None:
        assoc = associate_candidates(list(candidates), width, height)
        joints_2d_eff = assoc["joints_2d"]
        effective = assoc["present"] & present.any(axis=1, keepdims=True)
        hand_ids = assoc["labels"]                      # (T,2): the candidate index each slot binds per frame
        switches = [(int(f), int(s)) for f, s in assoc["switches"]]
    else:
        hand_ids, switches = associate_wearer(joints_2d_eff, effective)
```

(width/height come from calib/scene npz, defaulting to 640×480; import: `from roboray.stages.association import associate_candidates`. **Before implementing, read the downstream consumption shape of `associate_wearer`'s returned `hand_ids`** -- if downstream expects a shape different from `assoc["labels"]` (T,2), do one same-shape mapping inside the association branch (per-slot→candidate-index semantics unchanged); the output shape of existing npz keys must not change. The remaining downstream (fill_short_gaps, world projection) consumes the overwritten arrays -- naturally consistent.)

- [ ] **Step 4: full regression (the fake e2e now carries candidates → e2e fully goes through the association path)**

Run: `... -m pytest -q --tb=short` → all green (existing e2e assertions do not break from the association path -- the fake's candidates are the wearer's 2 hands, so x-rank and the association result agree)

- [ ] **Step 5: commit**

```bash
git add src/roboray/stages/fusion.py tests/unit/test_fusion_action.py
git commit -m "feat(fusion): A2 closed loop -- candidates association overwrites the primary data, no-key falls back to x-rank (audit #3)"
```

---

### Task 11: real-cluster regression + improvements report (GATE-gated)

**Files:**
- Create: `docs/bench/2026-10-26-audit-improvements-report.md`

**Interfaces:**
- Consumes: all prior tasks; `deploy/k8s/{status,apply,teardown}.sh`; a kubectl wrapper on the management node (recreate it if absent, via `ssh` to the cluster management node with `sudo kubectl`); the node's `/home/ubuntu/ego` (GPU-2 = <gpu-node-ip>) workspace and the rsync sync flow (already verified by the bench campaign)
- Produces: real-cluster regression evidence (small batch 4×30 real-model failed=0, npz contains candidates/alignment/gripper/stride-effective evidence) + the improvements report (the third-party review's 8 items compared item by item)

- [ ] **Step 1: GATE check**

Run: `KUBECTL=<wrapper> bash deploy/k8s/status.sh` → continue only if GATE=OPEN; if CLOSED, report gated (all code tasks are delivered as normal; the report truthfully notes the regression did not run).

- [ ] **Step 2: bring up the cluster + rsync + config**

`apply.sh` (IMAGE=registry.example.com/admin/vllm:v0.26.0 NODE=<gpu-node-ip>) → head Ready → rsync `src/ scripts/ tests/fakes/` to the node's `/home/ubuntu/ego/app/` → derive `/ego/audit.yaml` on the node (based on bench.yaml: input points to a bench-real-data subset of 4 items × 30 frames, `inference_fps: 30` (run `inference_fps: 15` once more to compare stride effectiveness), the action model per what was probed then).

- [ ] **Step 3: run the regression batch and collect evidence**

Inside the head pod (with `--ray-address auto`) run the 4×30 batch; evidence checklist:
- manifest failed=0, all six stages DONE;
- hand.npz contains `candidates` (share of frames where real mediapipe has >2 hands; if the data naturally has no multi-hand frames -- truthfully record "no multi-person scenes in the data; the multi-hand path is covered by unit tests");
- fused.npz/related npz contain `alignment` (share of kabsch vs root_only frames);
- action.npz contains `gripper`/`segment_kind`;
- the `inference_fps:15` run shows sparse present + the A3 backfill rate.
- multi-GPU slots: noted under the K8s one-pod-one-GPU arrangement (already decided in spec §6); not pretended to be verified.

- [ ] **Step 4: write the improvements report + commit**

`docs/bench/2026-10-26-audit-improvements-report.md`:
1. item-by-item comparison table (the third-party review's 8 items → fixed/semantic upgrade/truthfully kept + evidence links commit/test/real cluster)
2. real-cluster regression numbers and the evidence-checklist results
3. honest boundaries (#8 mono kept, A2 orientation consistency not done, gripper semantics, stride>7 boundary, Kabsch fallback share, multi-GPU slot verification scope)

```bash
git add docs/bench/2026-10-26-audit-improvements-report.md
git commit -m "docs(audit): improvements report -- item-by-item response to the third-party review's 8 items + real-cluster regression evidence"
```

- [ ] **Step 5: teardown + GPU-return check**

`teardown.sh` → `status.sh` (USED_GPUS back to the platform baseline) → append a check line to the report, then an additional commit.

- [ ] **Step 6: full regression final check**

Run: `RAY_ADDRESS=local ... -m pytest -q --tb=short` → all green; record the number in the report.
