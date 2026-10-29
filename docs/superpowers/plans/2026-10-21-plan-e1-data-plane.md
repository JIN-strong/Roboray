# E1 Data Plane Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep the driver process **entirely away from workspace files**, and pin each Episode to the node hosting its workspace, so the pipeline runs on multi-node clusters with node-local workspaces.

**Architecture:** Three changes interlock — (1) stage function return values come back through the Actor, the driver no longer reads `prepared/calib.json` itself; (2) the GPU payload changes from "carrying frame arrays" to "paths and scalars only", and frames are read by the Actor on its own node; (3) a new node pool registry — the Actor creates pools on demand with `NodeAffinitySchedulingStrategy`, and once an Episode picks a node all its stages stay pinned there.

**Tech Stack:** Python 3.10, Ray 2.59.0, pytest (`RAY_ADDRESS=local`), Ray's `NodeAffinitySchedulingStrategy` / `get_runtime_context().get_node_id()`

**Spec:** `docs/superpowers/specs/2026-10-21-ray-runtime-completion-design.md` section 4

## Global Constraints

- **Python environment:** `python` (conda env `roboray`); test commands are always `RAY_ADDRESS=local <python> -m pytest` (`RAY_ADDRESS=local` is required, otherwise the run connects to a leftover local Ray cluster and errors with `num_cpus and num_gpus must not be provided`).
- **Regression baseline:** before the change `211 passed / 0 failed`; the number only grows at the end of each task and **must never have failures**.
- **`model_worker` default:** this plan does not involve resident workers (that is E2); GPU still runs via `run_in_modelenv` one-shot subprocesses.
- **No algorithm-layer changes:** A1/A2/A3 and the four-stage semantics and the LeRobot output format all stay untouched.
- **No CR manifest changes:** `deploy/k8s/*` belongs to E3.
- **Zero driver workspace access:** this is the plan's final invariant, accepted by Task 7.
- **Comments/docstrings:** write them in English, and explain "why", not "what".

## Review Focus

These are inputs/conditions the spec hints at but no task naturally covers; each has a matching test added in the task in parentheses.

1. **Legacy `state.json` deserialization** — when a user upgrades the code and reruns an **existing workspace**, `state.json` has none of the new fields. Expected: it loads fine, the new fields take defaults, **no exception and no lost completed stage state**. (Task 2)
2. **CPU-only machine / a cluster with no GPU nodes at all** — all 211 existing tests run on machines without GPUs (Ray uses a virtual `num_gpus`). Expected: when node discovery returns empty, **fall back to the original single-pool behavior** and all tests stay green. (Task 4)
3. **Single-node cluster (only 1 GPU node)** — Expected: node selection still works, all Episodes land on the same node, behavior equivalent to before the change. (Task 4)
4. **Node at full load under `soft=False` affinity** — Expected: waiting is capped (`node_wait_timeout_s`); on timeout the Episode is counted as failed and leaves a searchable error, rather than silently hanging. (Task 5)
5. **fake model paths** — once the payload no longer carries arrays, `FakeSceneModel` / `FakeHandModel` must still be feedable with frames by the Actor. Expected: fake and real model paths share the same "Actor-side disk read" logic; e2e does not regress. (Task 3)

---

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `src/roboray/ray_runtime/actors.py` | Actor execution bodies; add `run_cpu_stage()` / `run_gpu_task()` pure function bodies, add `build_node_pools()` | Modify |
| `src/roboray/ray_runtime/nodes.py` | **New**: node discovery, `NodePool`, `NodePoolRegistry` selection logic (pure logic, no Ray runtime dependency) | Create |
| `src/roboray/runtime/episode.py` | `EpisodeRecord` gains `has_stereo` / `resolution_key` / `assigned_node`; `reconcile_for_new_run` supports dead nodes | Modify |
| `src/roboray/runtime/driver.py` | Drop the `_gpu_payload` disk reads; dispatch via `NodePoolRegistry`; `_reconcile_states` passes live nodes | Modify |
| `src/roboray/stages/scene.py` | `build_scene_payload` → `scene_payload_args` (paths and scalars only) | Modify |
| `src/roboray/stages/hand.py` | `build_hand_payload` → `hand_payload_args` (same) | Modify |
| `src/roboray/stages/prepare.py` | `run_prepare` return value gains `width` / `height` | Modify |
| `src/roboray/api/reconstruction.py` | `build_actor_pool` → `build_node_pools` | Modify |
| `src/roboray/api/config.py` | `ExecutionConfig` gains `cpu_actors_per_node` / `node_wait_timeout_s` | Modify |
| `tests/unit/test_node_pools.py` | **New**: node selection pure-logic tests | Create |
| `tests/e2e/test_driver_workspace_isolation.py` | **New**: hard-invariant acceptance that the driver has zero workspace access | Create |

---

### Task 1: CPU Actor returns stage function results

**Files:**
- Modify: `src/roboray/ray_runtime/actors.py:96-141`
- Test: `tests/unit/test_actors.py` (append at end of file)

**Interfaces:**
- Consumes: `roboray.stages.STAGE_FUNCTIONS` (existing)
- Produces: `run_cpu_stage(payload: dict) -> dict`, returning keys `episode_id` / `kind` / `seconds` / `ok` / `error` / **`stage_result`** (the dict returned by the stage function, or `None` when it is not a dict or raises)

**Why:** `CpuStageActor.run_task` currently **discards the return value** after `fn(**kwargs)`. Yet prepare's `width` / `height` / `has_stereo` are the only source the driver uses for later routing — the driver cannot read the workspace itself (unreadable under multi-node), so the Actor must return them.

- [ ] **Step 1: write a failing test**

Append to `tests/unit/test_actors.py`:

```python
def test_run_cpu_stage_returns_stage_function_result(monkeypatch):
    """Regression: the CPU Actor previously discarded the stage function's return value.

    prepare's width/height/has_stereo are the only source for the driver's later
    routing (scene stereo decision, resolution tier); the driver cannot read the
    workspace itself (unreadable under multi-node), so these values must be
    returned through the Actor.

    Call the **module-level pure function body** directly instead of ``.remote()``:
    Ray actors run in separate processes, so a driver-side monkeypatch does not
    propagate and ``.remote()`` cannot test this path.
    """
    from roboray import stages as stages_mod
    from roboray.ray_runtime.actors import run_cpu_stage

    monkeypatch.setitem(
        stages_mod.STAGE_FUNCTIONS, "__probe__", lambda **kw: {"width": 320, "height": 240}
    )
    res = run_cpu_stage(
        {"episode_id": "ep_probe", "kind": "prepare", "fn_name": "__probe__", "kwargs": {}}
    )
    assert res["ok"] is True, res["error"]
    assert res["stage_result"] == {"width": 320, "height": 240}


def test_run_cpu_stage_stage_result_is_none_on_failure(monkeypatch):
    """stage_result must be None when the stage function raises (no stale value leaks)."""
    from roboray import stages as stages_mod
    from roboray.ray_runtime.actors import run_cpu_stage

    def _boom(**kw):
        raise ValueError("intentional failure")

    monkeypatch.setitem(stages_mod.STAGE_FUNCTIONS, "__boom__", _boom)
    res = run_cpu_stage(
        {"episode_id": "ep_probe", "kind": "fusion", "fn_name": "__boom__", "kwargs": {}}
    )
    assert res["ok"] is False
    assert res["stage_result"] is None
    assert "ValueError" in res["error"]
```

- [ ] **Step 2: run the test and confirm it fails**

```bash
cd <repo-root>
RAY_ADDRESS=local python -m pytest \
  tests/unit/test_actors.py -q -o addopts="" -k "stage_function_result or stage_result_is_none"
```
Expected: FAIL, `ImportError: cannot import name 'run_cpu_stage'`

- [ ] **Step 3: extract the execution body into a module-level pure function**

`src/roboray/ray_runtime/actors.py`: move the whole body of `CpuStageActor.run_task` into a module-level function; the method just delegates.

Insert after `build_infer_frames` and before `@ray.remote(num_gpus=1)`:

```python
def run_cpu_stage(payload: dict) -> dict:
    """CPU stage execution body (module-level pure function, directly unit-testable).

    It is extracted to a module-level function rather than an Actor method because
    Ray actors run in separate processes: a test's monkeypatch of ``STAGE_FUNCTIONS``
    does not propagate on the driver side, so only calling the function body
    directly can cover this path.
    """
    from roboray.stages import STAGE_FUNCTIONS

    # A Ray worker is a **separate process**, so the driver-side registry
    # registration does not propagate. The action stage (VLM) runs on the CPU
    # actor; without registering here it fails with
    # `UnknownModelError: semantic model 'openai_vlm' is not registered; registered: []`
    # -- that path had only been exercised under backend=test (a stub registered by
    # worker_process_setup_hook), which is why it never surfaced. Registration is
    # idempotent (internally guarded by `if name not in ...`).
    registry.load_builtin_models()

    result: dict[str, Any] = {
        "episode_id": payload.get("episode_id"),
        "kind": payload.get("kind"),
        "seconds": 0.0,
        "ok": True,
        "error": None,
        "stage_result": None,
    }
    t0 = time.perf_counter()
    try:
        fn = STAGE_FUNCTIONS[payload["fn_name"]]
        out = fn(**payload["kwargs"])
        result["seconds"] = time.perf_counter() - t0
        # The stage function's return value must be sent back: prepare's
        # width/height/has_stereo are the only source for the driver's later
        # routing (the driver cannot read the workspace under multi-node).
        result["stage_result"] = out if isinstance(out, dict) else None
    except Exception as exc:
        result["ok"] = False
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result
```

Then replace the whole method body of `CpuStageActor.run_task` with:

```python
    def run_task(self, payload: dict) -> dict:
        return run_cpu_stage(payload)
```

And change `__all__` to:

```python
__all__ = ["GpuModelActor", "CpuStageActor", "build_infer_frames", "build_actor_pool",
           "run_cpu_stage"]
```

- [ ] **Step 4: run the test and confirm it passes**

```bash
RAY_ADDRESS=local python -m pytest \
  tests/unit/test_actors.py -q -o addopts=""
```
Expected: 8 passed (previous 6 + new 2)

- [ ] **Step 5: commit**

```bash
git add src/roboray/ray_runtime/actors.py tests/unit/test_actors.py
git commit -m "refactor(actors): extract run_cpu_stage pure function body and return stage function results

prepare's width/height/has_stereo are the only source for the driver's later
routing and were previously discarded by CpuStageActor. E1 requires the driver
to stop reading the workspace, so the Actor must send them back.
Extracted to a module-level function because Ray actors run in separate
processes, monkeypatch does not propagate, and .remote() cannot cover this path."
```

---

### Task 2: `EpisodeRecord` carries prepare metadata, driver stops reading `calib.json`

**Files:**
- Modify: `src/roboray/runtime/episode.py:59-147` (`EpisodeRecord`), `src/roboray/stages/prepare.py` (`run_prepare` return value), `src/roboray/runtime/driver.py:80-87,363-389`
- Test: `tests/unit/test_episode.py`, `tests/unit/test_stages_io.py`

**Interfaces:**
- Consumes: Task 1's `stage_result`
- Produces:
  - `EpisodeRecord.has_stereo: Optional[bool]`, `EpisodeRecord.resolution_key: Optional[str]`
  - `EpisodeRecord.record_prepare(has_stereo: bool, width: int, height: int) -> None` (writes back these three fields)
  - `run_prepare(...)` return value gains `width: int` / `height: int`
  - `driver._scene_model(ep)` and the new `driver._resolution_key(ep)` read the record instead

- [ ] **Step 1: write failing tests (legacy state.json compatibility + record backfill)**

Append to `tests/unit/test_episode.py`:

```python
def test_record_loads_legacy_state_json_without_new_fields(tmp_path):
    """Review Focus #1: rerunning an **existing workspace** after upgrade, the old
    state.json has none of the new fields.

    Expected: loads fine, new fields take defaults, no exception and no lost
    completed stage state.
    """
    from roboray.runtime.episode import EpisodeRecord, EpisodeWorkspace, Stage, StageStatus

    ws = EpisodeWorkspace(tmp_path, "ep_0001")
    # hand-write an "old-version" state.json: no has_stereo / resolution_key
    ws.state_file.write_text(json.dumps({
        "episode_id": "ep_0001",
        "state_schema_version": 1,
        "stages": {"prepare": "done", "scene": "pending", "hand": "pending",
                   "fusion": "pending", "action": "pending", "publish": "pending"},
        "timings": {"prepare": 0.5},
        "error": None,
    }))

    rec = ws.load()
    assert rec.is_done(Stage.PREPARE)
    assert rec.stages[Stage.SCENE] is StageStatus.PENDING
    assert rec.has_stereo is None
    assert rec.resolution_key is None


def test_record_prepare_persists_metadata(tmp_path):
    """record_prepare writes has_stereo / resolution_key and round-trips serialization."""
    from roboray.runtime.episode import EpisodeWorkspace

    ws = EpisodeWorkspace(tmp_path, "ep_0002")
    rec = ws.load()
    rec.record_prepare(has_stereo=True, width=320, height=240)
    ws.save_record(rec)

    again = ws.load()
    assert again.has_stereo is True
    assert again.resolution_key == "320x240"
```

If `tests/unit/test_episode.py` does not already import `json` at the top, add `import json`.

- [ ] **Step 2: run the tests and confirm they fail**

```bash
RAY_ADDRESS=local python -m pytest \
  tests/unit/test_episode.py -q -o addopts="" -k "legacy_state_json or record_prepare"
```
Expected: FAIL, `AttributeError: 'EpisodeRecord' object has no attribute 'has_stereo'`

- [ ] **Step 3: add the fields and method to `EpisodeRecord`**

`src/roboray/runtime/episode.py`, add three fields to the `EpisodeRecord` dataclass (`assigned_node` is for Task 6; adding it here avoids changing the schema twice):

```python
@dataclass
class EpisodeRecord:
    episode_id: str
    stages: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)
    error: Optional[str] = None
    # backfilled after prepare completes: the driver routes on this (scene stereo
    # decision, resolution tier). Not reading workspace files is a hard constraint --
    # under multi-node the driver cannot read a node-local workspace.
    has_stereo: Optional[bool] = None
    resolution_key: Optional[str] = None
    # which Ray node this Episode is pinned to (written by Task 5, checked by Task 6).
    assigned_node: Optional[str] = None
```

Add a method (before `undone_stages`):

```python
    def record_prepare(self, has_stereo: bool, width: int, height: int) -> None:
        """Backfill the routing metadata from prepare.

        ``resolution_key`` uses the same ``"<w>x<h>"`` format as
        ``planner.DEFAULT_RESOLUTION_KEY`` (see ``stages/scene.resolution_key``).
        """
        self.has_stereo = bool(has_stereo)
        self.resolution_key = f"{int(width)}x{int(height)}"
```

Add the three fields to `to_json`:

```python
    def to_json(self) -> dict:
        return {
            "episode_id": self.episode_id,
            "state_schema_version": STATE_SCHEMA_VERSION,
            "stages": {s.value: v.value for s, v in self.stages.items()},
            "timings": dict(self.timings),
            "error": self.error,
            "has_stereo": self.has_stereo,
            "resolution_key": self.resolution_key,
            "assigned_node": self.assigned_node,
        }
```

Extend `from_json` to read them, **all via `.get()` for tolerance** (this is where Review Focus #1 lands):

```python
        return cls(
            episode_id=raw["episode_id"],
            stages=parsed,
            timings={k: float(v) for k, v in raw.get("timings", {}).items()},
            error=raw.get("error"),
            # old state.json has none of these three fields: they must tolerate
            # None, otherwise rerunning an existing workspace after upgrade
            # raises directly and loses completed stage state.
            has_stereo=raw.get("has_stereo"),
            resolution_key=raw.get("resolution_key"),
            assigned_node=raw.get("assigned_node"),
        )
```

- [ ] **Step 4: `run_prepare` return value gains width/height**

`src/roboray/stages/prepare.py`, change the `summary` construction to:

```python
    summary = {
        "episode_id": episode_id,
        "n_frames": int(n_frames),
        "has_stereo": has_stereo,
        "degraded": degraded,
        "degrade_reason": degrade_reason,
        # for the driver's routing: the driver does not read workspace files
        # (unreadable under multi-node), so these values must be returned
        # through the Actor's return value.
        "width": int(w),
        "height": int(h),
    }
```

- [ ] **Step 5: driver reads the record instead**

`src/roboray/runtime/driver.py`:

Change `_scene_model` to (`src/roboray/runtime/driver.py:80-87`):

```python
    def _scene_model(self, episode_id: str) -> str:
        """The real model name for this Episode's scene path.

        Prefer the has_stereo backfilled into the record by prepare; when prepare
        has **not** run yet (cold-start prior phase) fall back to the input-side
        raw guess ``spec.has_stereo`` -- only affects the early scheduling prior,
        not correctness. **Does not read workspace files**: under multi-node the
        driver cannot read a node-local workspace.
        """
        rec = self._ws(episode_id).load()
        has_stereo = rec.has_stereo
        if has_stereo is None:
            spec = self.specs.get(episode_id)
            has_stereo = spec.has_stereo if spec is not None else False
        return resolve_scene_model(
            self.request.stages.scene_reconstruction.model, has_stereo
        )

    def _resolution_key(self, episode_id: str) -> str:
        """This Episode's resolution tier; falls back to the placeholder tier when
        prepare is not done.

        The placeholder tier only affects the cold-start scheduling prior (same
        meaning as planner.DEFAULT_RESOLUTION_KEY); at execution time the Actor
        reports the real tier.
        """
        rec = self._ws(episode_id).load()
        return rec.resolution_key or DEFAULT_RESOLUTION_KEY
```

Change the top-of-file import to (removing `prepared_resolution_key`):

```python
from roboray.stages.scene import resolve_scene_model
```
and after `from .planner import build_cpu_kwargs, episode_ready_tasks` add a line:
```python
from .planner import DEFAULT_RESOLUTION_KEY, build_cpu_kwargs, episode_ready_tasks
```

Change the `res_key = prepared_resolution_key(...)` in `push_ready()` (`driver.py:204-206`) to:

```python
                        res_key = self._resolution_key(ep)
```

Change the `res_key = prepared_resolution_key(...)` in `_execute_one` (`driver.py:398,405`) to:

```python
            res_key = self._resolution_key(spec.episode_id)
```
and delete the `from roboray.stages.scene import prepared_resolution_key` inside that branch (`driver.py:398`).

In `_complete`, inside the `if res.get("ok"):` branch and **before** `if slot is not None:`, insert the backfill:

```python
            if kind == "prepare":
                sr = res.get("stage_result") or {}
                if "width" in sr and "height" in sr:
                    rec = ws.load()
                    rec.record_prepare(
                        has_stereo=bool(sr.get("has_stereo", False)),
                        width=int(sr["width"]),
                        height=int(sr["height"]),
                    )
                    ws.save_record(rec)
```

- [ ] **Step 6: update affected existing tests**

`tests/unit/test_stages_io.py:137-178`'s `test_build_scene_payload_uses_prepared_has_stereo`
still tests `prepared_has_stereo` / `prepared_resolution_key` themselves — they **stay** (Task 3 changes `build_scene_payload`), so this test is untouched for now.

Append to `tests/unit/test_stages_io.py` a case pinning "the driver no longer calls these two functions":

```python
def test_driver_does_not_read_prepared_calib(tmp_path, monkeypatch):
    """Sibling of Review Focus #1/#2: the driver's routing must not read prepared/calib.json.

    E1's hard constraint is "the driver never touches the workspace"; `_scene_model`
    and `_resolution_key` must depend only on EpisodeRecord.
    """
    import roboray.runtime.driver as drv
    from roboray.runtime.episode import EpisodeRecord, EpisodeWorkspace

    ws = EpisodeWorkspace(tmp_path, "ep_0001")
    rec = EpisodeRecord.new("ep_0001")
    rec.record_prepare(has_stereo=True, width=320, height=240)
    ws.save_record(rec)

    def _forbidden(*a, **k):
        raise AssertionError("driver must not read prepared/calib.json")

    monkeypatch.setattr(drv, "prepared_resolution_key", _forbidden, raising=False)

    d = drv.PipelineDriver.__new__(drv.PipelineDriver)
    d.config = drv.DriverConfig(
        storage_root=str(tmp_path), workspace_root=str(tmp_path), output_root=str(tmp_path)
    )
    d.specs = {}
    d.request = _minimal_request()
    assert d._scene_model("ep_0001") == "fast_foundation_stereo"
    assert d._resolution_key("ep_0001") == "320x240"
```

The same file needs a minimal request constructor (reuse an existing helper if present; add one if not):

```python
def _minimal_request():
    from roboray.api.config import ReconstructionRequest

    return ReconstructionRequest.from_dict({
        "input": {"uri": "data/synthetic"},
        "output": {"uri": "output/run-test", "format": "lerobot_v3"},
        "execution": {"backend": "test"},
    })
```

- [ ] **Step 7: run the tests and confirm they pass**

```bash
RAY_ADDRESS=local python -m pytest \
  tests/unit/test_episode.py tests/unit/test_stages_io.py -q -o addopts=""
```
Expected: all PASS

- [ ] **Step 8: run the full regression**

```bash
RAY_ADDRESS=local python -m pytest -q --tb=short
```
Expected: `>= 213 passed`, 0 failed

- [ ] **Step 9: commit**

```bash
git add src/roboray/runtime/episode.py src/roboray/stages/prepare.py \
        src/roboray/runtime/driver.py tests/unit/test_episode.py tests/unit/test_stages_io.py
git commit -m "feat(driver): prepare metadata returned via EpisodeRecord, driver no longer reads calib.json

The driver's routing (scene stereo decision, resolution tier) previously relied
on reading prepared/calib.json -- under multi-node the driver cannot read a
node-local workspace. Now prepare returns it through the Actor's return value
and the driver records it in EpisodeRecord. Old state.json missing the new
fields tolerates None."
```

---

### Task 3: GPU payload carries only paths and scalars, the Actor reads frames on its own node

**Files:**
- Modify: `src/roboray/stages/scene.py:65-100`, `src/roboray/stages/hand.py:13-26`, `src/roboray/ray_runtime/actors.py:18-35,64-95`
- Test: `tests/unit/test_scene_models.py:130-260`, `tests/unit/test_stages_io.py`

**Interfaces:**
- Produces:
  - `scene_payload_args(workspace_root: str, episode_id: str, model_name: str, out_path: str, stereo_max_image_side_px: int = 640) -> dict`
  - `hand_payload_args(workspace_root: str, episode_id: str, model_name: str, out_path: str) -> dict`
  - `read_prepared_frames(workspace_root: str, episode_id: str) -> tuple[dict, dict]` (returns `(frames, calib)`)
  - `GpuModelActor.run_task`'s payload **no longer contains ndarrays**
- Deletes: `build_scene_payload` / `build_hand_payload` / `build_infer_frames`

- [ ] **Step 1: write failing tests**

Append to `tests/unit/test_scene_models.py`:

```python
def test_gpu_payload_carries_no_arrays(tmp_path):
    """E1 hard constraint: the GPU payload may carry only paths and scalars.

    Frame arrays sent through the Ray object store are shared memory on a single
    node but cross the network between nodes -- exactly the "large array
    cross-node movement" the spec wants to eliminate. An ndarray appearing in the
    payload is treated as a regression.
    """
    from roboray.stages.scene import scene_payload_args

    payload = scene_payload_args(
        workspace_root=str(tmp_path), episode_id="ep_0001",
        model_name="vipe", out_path=str(tmp_path / "ep_0001" / "artifacts" / "scene.npz"),
    )
    for key, value in payload.items():
        assert not isinstance(value, np.ndarray), f"payload[{key!r}] must not be an array"
    assert payload["workspace_root"] == str(tmp_path)
    assert payload["episode_id"] == "ep_0001"


def test_read_prepared_frames_hoists_right_only_when_stereo(tmp_path):
    """C1 companion semantics unchanged: frames_right is attached only for stereo.

    Previously `build_infer_frames` hoisted it from ``calib["frames_right"]``; now
    the Actor reads from disk directly, but the "no frames_right for mono" semantic
    must be preserved.
    """
    from roboray.ray_runtime.actors import read_prepared_frames

    ws = tmp_path / "ep_0001" / "prepared"
    ws.mkdir(parents=True)
    np.save(ws / "frames_left.npy", np.zeros((4, 8, 8, 3), np.uint8))

    (ws / "calib.json").write_text(json.dumps(
        {"K": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "dist": [0] * 5, "fps": 30,
         "width": 8, "height": 8, "has_stereo": False}
    ))
    frames, calib = read_prepared_frames(str(tmp_path), "ep_0001")
    assert "frames_right" not in frames
    assert calib["has_stereo"] is False

    np.save(ws / "frames_right.npy", np.ones((4, 8, 8, 3), np.uint8))
    (ws / "calib.json").write_text(json.dumps(
        {"K": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "dist": [0] * 5, "fps": 30,
         "width": 8, "height": 8, "has_stereo": True}
    ))
    frames, calib = read_prepared_frames(str(tmp_path), "ep_0001")
    assert "frames_right" in frames
    assert frames["frames_right"].shape == (4, 8, 8, 3)
```

If `tests/unit/test_scene_models.py` does not already import `json` at the top, add `import json`.

- [ ] **Step 2: run the tests and confirm they fail**

```bash
RAY_ADDRESS=local python -m pytest \
  tests/unit/test_scene_models.py -q -o addopts="" -k "carries_no_arrays or hoists_right_only_when_stereo"
```
Expected: FAIL, `ImportError: cannot import name 'scene_payload_args'`

- [ ] **Step 3: rewrite `stages/scene.py`**

Replace the whole of `build_scene_payload` (`scene.py:65-100`) with:

```python
def scene_payload_args(
    workspace_root: str,
    episode_id: str,
    model_name: str,
    out_path: str,
    stereo_max_image_side_px: int = 640,
) -> dict:
    """Build the Scene GPU payload -- **paths and scalars only, no arrays**.

    Frame arrays are read by ``GpuModelActor`` on **its own node**. In a multi-node
    cluster the driver runs on the head/CPU worker and **cannot read** a node-local
    workspace; and passing arrays through the Ray object store across nodes is
    exactly the "large array cross-node movement" the spec wants to eliminate.

    ``stereo_max_image_side_px`` (spec §6's FFS input resolution control) takes
    effect via Actor -> adapter ``--extra`` -> ``run_ffs.py``: the stereo pair is
    scaled to ``max(H, W) <= this cap``, with ``fx`` scaled proportionally; because
    disparity scales with resolution, ``fx'·B/d' == fx·B/d``, so **depth is
    invariant to the scaling** -- this knob trades accuracy for speed. ``0`` means
    no scaling.
    """
    return {
        "kind": "scene",
        "episode_id": episode_id,
        "model_name": model_name,
        "workspace_root": str(workspace_root),
        "out_path": str(out_path),
        "stereo_max_image_side_px": int(stereo_max_image_side_px),
    }
```

In `__all__`, replace `"build_scene_payload"` with `"scene_payload_args"`.

`prepared_has_stereo` / `prepared_resolution_key` **stay** (`tests/unit/test_stages_io.py` still tests them; they are pure read utilities, the driver just no longer calls them).

- [ ] **Step 4: rewrite `stages/hand.py`**

Replace the whole of `build_hand_payload` (`hand.py:13-26`) with:

```python
def hand_payload_args(
    workspace_root: str,
    episode_id: str,
    model_name: str,
    out_path: str,
) -> dict:
    """Build the Hand GPU payload -- **paths and scalars only, no arrays** (same as Scene)."""
    return {
        "kind": "hand",
        "episode_id": episode_id,
        "model_name": model_name,
        "workspace_root": str(workspace_root),
        "out_path": str(out_path),
    }
```

Rename it in `__all__` as well.

- [ ] **Step 5: `actors.py` gains a disk-read function and the GPU execution body is rewritten**

Delete `build_infer_frames` (`actors.py:18-35`), replacing it with:

```python
def read_prepared_frames(workspace_root: str, episode_id: str) -> tuple:
    """Read the prepared frames and calibration on the **Actor's own node**, returning
    ``(frames, calib)``.

    ``frames`` is the dict the model adapter ``infer(frames, calib, out_path)``
    expects: ``frames_right`` is attached only when the calibration marks
    ``has_stereo`` (not for mono). This semantic was previously implemented by
    ``build_infer_frames`` hoisting from ``calib["frames_right"]``; it is now a
    direct disk read, semantics unchanged.
    """
    prepared = Path(workspace_root) / episode_id / "prepared"
    calib = json.loads((prepared / "calib.json").read_text(encoding="utf-8"))
    frames = {"frames": np.load(prepared / "frames_left.npy")}
    if calib.get("has_stereo"):
        frames["frames_right"] = np.load(prepared / "frames_right.npy")
    return frames, calib
```

Add imports at the top of the file: `import json`, `import numpy as np`, `from roboray.stages.scene import resolution_key` (next to the existing `from roboray.models import registry`).

Replace the whole of `GpuModelActor.run_task` (`actors.py:64-95`) with:

```python
    def run_task(self, payload: dict) -> dict:
        kind = payload["kind"]
        model_name = payload["model_name"]
        result: dict[str, Any] = {
            "episode_id": payload["episode_id"],
            "kind": kind,
            "out_path": payload.get("out_path"),
            "resolution_key": payload.get("resolution_key", "640x480"),
            "infer_seconds": 0.0,
            "load_seconds": 0.0,
            "was_load": False,
            "ok": True,
            "error": None,
        }
        try:
            if self.loaded_kind != kind or self.model_name != model_name:
                result["load_seconds"] = self.load_model(kind, model_name)
                result["was_load"] = True
            # frames are read from disk on this Actor's own node (the payload only
            # carries paths): under multi-node the driver cannot read a node-local
            # workspace, and passing arrays through the object store across nodes
            # is exactly what we want to avoid.
            frames, calib = read_prepared_frames(payload["workspace_root"], payload["episode_id"])
            if kind == "scene":
                calib["stereo_max_image_side_px"] = int(payload.get("stereo_max_image_side_px", 0))
            # the resolution tier is decided by the **real frames** and reported
            # back to the driver for the cost model
            result["resolution_key"] = resolution_key(frames["frames"].shape[2],
                                                      frames["frames"].shape[1])
            t0 = time.perf_counter()
            self.model.infer(frames, calib, Path(payload["out_path"]))
            result["infer_seconds"] = time.perf_counter() - t0
        except Exception as exc:  # failure comes back as data; the Driver decides handling
            result["ok"] = False
            result["error"] = f"{type(exc).__name__}: {exc}"
        return result
```

Change `__all__` to:
```python
__all__ = ["GpuModelActor", "CpuStageActor", "read_prepared_frames", "build_actor_pool",
           "run_cpu_stage"]
```

- [ ] **Step 6: driver switches to the new payload constructors**

`src/roboray/runtime/driver.py`, replace the whole of `_gpu_payload` (`driver.py:333-354`) with:

```python
    def _gpu_payload(self, kind, spec, item, ws) -> dict:
        """Build the GPU payload -- **paths and scalars only** (frames are read by the
        Actor on its own node)."""
        if kind == "scene":
            from roboray.stages.scene import scene_payload_args

            return scene_payload_args(
                self.config.workspace_root,
                item.episode_id,
                self._scene_model(item.episode_id),
                str(ws.artifact("scene.npz")),
                stereo_max_image_side_px=(
                    self.request.stages.scene_reconstruction.stereo_max_image_side_px
                ),
            )
        from roboray.stages.hand import hand_payload_args

        return hand_payload_args(
            self.config.workspace_root,
            item.episode_id,
            self.request.stages.hand_reconstruction.model,
            str(ws.artifact("hand.npz")),
        )
```

- [ ] **Step 7: update existing tests affected by the deleted functions**

`tests/unit/test_scene_models.py:130-260` has two tests using `build_infer_frames`:
- around lines 130-155 (the one building frames the same way the Actor does) → switch it to `read_prepared_frames`
- lines 240-260, `test_build_infer_frames_hoists_right_only_when_present` → **delete** (its semantic is now covered by the new `test_read_prepared_frames_hoists_right_only_when_stereo` from Step 1)

`tests/unit/test_stages_io.py:137-178`'s `test_build_scene_payload_uses_prepared_has_stereo` → rename to
`test_prepared_has_stereo_matches_calib`, drop the call to `build_scene_payload`, and keep only the
assertions on the two utilities `prepared_has_stereo` / `prepared_resolution_key`.

- [ ] **Step 8: run the tests and confirm they pass**

```bash
RAY_ADDRESS=local python -m pytest \
  tests/unit/test_scene_models.py tests/unit/test_stages_io.py -q -o addopts=""
```
Expected: all PASS

- [ ] **Step 9: run the full regression (the fake path must still work -- Review Focus #5)**

```bash
RAY_ADDRESS=local python -m pytest -q --tb=short
```
Expected: `>= 214 passed`, 0 failed

- [ ] **Step 10: commit**

```bash
git add src/roboray/stages/scene.py src/roboray/stages/hand.py \
        src/roboray/ray_runtime/actors.py src/roboray/runtime/driver.py \
        tests/unit/test_scene_models.py tests/unit/test_stages_io.py
git commit -m "feat(actors): GPU payload carries only paths and scalars, frames read by the Actor on its own node

Frame arrays were previously read into arrays by the driver and stuffed into the
payload, then sent to the GPU actor through the Ray object store -- shared memory
on a single node but over the network across nodes, exactly the "large array
cross-node movement" the spec wants to eliminate; and under multi-node the driver
cannot read a node-local workspace at all.
read_prepared_frames keeps the original "frames_right only for stereo" semantic."
```

---

### Task 4: node discovery and node-pool selection (pure logic)

**Files:**
- Create: `src/roboray/ray_runtime/nodes.py`
- Test: `tests/unit/test_node_pools.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) NodeInfo: node_id: str; gpu_count: int; hostname: str`
  - `discover_gpu_nodes() -> list[NodeInfo]` (queries `ray.nodes()`; returns `[]` when Ray is uninitialized or there are no GPU nodes)
  - `class NodePool: node_id: str; gpu_actor: object; cpu_actors: list; in_flight: int`, methods `has_free_gpu() -> bool`, `take_gpu() -> None`, `release_gpu() -> None`
  - `class NodePoolRegistry`, methods:
    - `__init__(self, live_nodes: list[NodeInfo], cpu_actors_per_node: int, create_pool: Callable[[str], NodePool])`
    - `pool_for_episode(self, episode_id: str, assigned: dict[str, str]) -> NodePool | None`
    - `register(self, pool: NodePool) -> None`
    - `drop(self, node_id: str) -> None`
    - `live_node_ids(self) -> set[str]`

**Why pure logic:** the selection rules (prefer an existing workspace → least-loaded → create new) are the part of E1 most prone to error and most worth pinning with unit tests. By isolating Ray calls in the `create_pool` callback and `discover_gpu_nodes`, the rules themselves can be tested directly against a fake node table.

- [ ] **Step 1: write failing tests**

Create `tests/unit/test_node_pools.py`:

```python
"""Node-pool selection rules (pure logic, no Ray runtime dependency)."""

import pytest

from roboray.ray_runtime.nodes import NodeInfo, NodePool, NodePoolRegistry


def _pool(node_id: str, cpu_actors: int = 2) -> NodePool:
    return NodePool(node_id=node_id, gpu_actor=object(), cpu_actors=[object()] * cpu_actors)


def _registry(nodes, created=None, cpu_actors_per_node=2):
    created = created if created is not None else []

    def _create(node_id):
        p = _pool(node_id, cpu_actors_per_node)
        created.append(p)
        return p

    return NodePoolRegistry(
        live_nodes=nodes, cpu_actors_per_node=cpu_actors_per_node, create_pool=_create
    ), created


def test_prefers_node_already_holding_the_episode_workspace():
    """Resume from a checkpoint: the Episode already has assigned_node and that node
    is still live -> it must return to the original node.

    If it cannot return, the workspace is not on that machine and the whole thing
    must be rerun (Task 6).
    """
    reg, _ = _registry([NodeInfo("n1", 1, "h1"), NodeInfo("n2", 1, "h2")])
    reg.register(_pool("n1"))
    reg.register(_pool("n2"))
    chosen = reg.pool_for_episode("ep_0001", {"ep_0001": "n2"})
    assert chosen.node_id == "n2"


def test_assigned_node_no_longer_live_returns_none():
    """The original node is no longer in the live list -> return None (the caller
    triggers a full rerun)."""
    reg, _ = _registry([NodeInfo("n1", 1, "h1")])
    reg.register(_pool("n1"))
    assert reg.pool_for_episode("ep_0001", {"ep_0001": "gone"}) is None


def test_picks_least_loaded_live_node():
    """With no assigned_node, pick the node with the fewest in-flight Episodes."""
    reg, _ = _registry([NodeInfo("n1", 1, "h1"), NodeInfo("n2", 1, "h2")])
    p1, p2 = _pool("n1"), _pool("n2")
    reg.register(p1)
    reg.register(p2)
    p1.take_gpu()
    assert reg.pool_for_episode("ep_x", {}).node_id == "n2"


def test_creates_new_pool_when_all_full():
    """All pools full -> create a new one (no affinity, giving the autoscaler a
    chance to add nodes)."""
    reg, created = _registry([NodeInfo("n1", 1, "h1"), NodeInfo("n2", 1, "h2")])
    p1, p2 = _pool("n1"), _pool("n2")
    reg.register(p1)
    reg.register(p2)
    p1.take_gpu()
    p2.take_gpu()

    chosen = reg.pool_for_episode("ep_x", {})
    assert chosen is created[-1]
    assert chosen.node_id not in ("n1", "n2")


def test_empty_live_nodes_returns_none_not_crash():
    """Review Focus #2: CPU-only machine / no GPU nodes -> return None, no exception.

    All 211 existing tests run on machines without GPUs (Ray uses a virtual
    num_gpus), so empty node discovery is the norm, not an anomaly.
    """
    reg, _ = _registry([])
    assert reg.pool_for_episode("ep_x", {}) is None
    assert reg.live_node_ids() == set()


def test_single_node_cluster_all_episodes_land_there():
    """Review Focus #3: in a single-node cluster all Episodes land on the same node."""
    reg, _ = _registry([NodeInfo("only", 1, "h")])
    reg.register(_pool("only"))
    assert reg.pool_for_episode("ep_a", {}).node_id == "only"
    assert reg.pool_for_episode("ep_b", {}).node_id == "only"


def test_drop_removes_pool_and_node_id():
    """After a node dies and is dropped: it is neither in the live set nor selectable."""
    reg, _ = _registry([NodeInfo("n1", 1, "h1"), NodeInfo("n2", 1, "h2")])
    reg.register(_pool("n1"))
    reg.register(_pool("n2"))
    reg.drop("n1")
    assert "n1" not in reg.live_node_ids()
    assert reg.pool_for_episode("ep_x", {}).node_id == "n2"


def test_gpu_slot_is_exclusive():
    """A node pool runs only one Episode at a time (spec §4.4's reuse criterion)."""
    p = _pool("n1")
    assert p.has_free_gpu() is True
    p.take_gpu()
    assert p.has_free_gpu() is False
    p.release_gpu()
    assert p.has_free_gpu() is True
```

- [ ] **Step 2: run the tests and confirm they fail**

```bash
RAY_ADDRESS=local python -m pytest \
  tests/unit/test_node_pools.py -q -o addopts=""
```
Expected: FAIL, `ModuleNotFoundError: No module named 'roboray.ray_runtime.nodes'`

- [ ] **Step 3: implement `nodes.py`**

Create `src/roboray/ray_runtime/nodes.py`:

```python
"""Node discovery and node-pool selection.

**Why needed**: E1 keeps each Episode's workspace on the **node-local** hostPath.
Therefore **all stages of one Episode must land on the same node** -- otherwise
``prepared/`` and ``artifacts/`` are unreadable. This module owns the rule for
"which Episode goes to which node".

**Why decoupled from Ray**: the selection rule is the part of this plan most
prone to error and most worth pinning with unit tests. Ray calls are isolated in
``discover_gpu_nodes()`` and the ``create_pool`` callback, so the rules themselves
can be tested against a fake node table without starting a cluster.
"""

from dataclasses import dataclass, field
from typing import Callable

__all__ = ["NodeInfo", "NodePool", "NodePoolRegistry", "discover_gpu_nodes"]


@dataclass(frozen=True)
class NodeInfo:
    node_id: str
    gpu_count: int
    hostname: str


def discover_gpu_nodes() -> list[NodeInfo]:
    """List the Ray nodes that currently have GPUs.

    Returns ``[]`` when Ray is uninitialized or the cluster has no GPU nodes --
    **this is the norm, not an anomaly**: the local test machine has no GPU and
    Ray runs with a virtual ``num_gpus``; the same holds for CPU-only clusters.
    The caller falls back to the old "no node pool" behavior accordingly.
    """
    try:
        import ray
    except ImportError:
        return []
    if not ray.is_initialized():
        return []
    out: list[NodeInfo] = []
    for node in ray.nodes():
        if not node.get("Alive", False):
            continue
        gpus = int(node.get("Resources", {}).get("GPU", 0) or 0)
        if gpus <= 0:
            continue
        out.append(
            NodeInfo(
                node_id=str(node.get("NodeID", "")),
                gpu_count=gpus,
                hostname=str(node.get("NodeManagerHostname", "")),
            )
        )
    return sorted(out, key=lambda n: n.node_id)


@dataclass
class NodePool:
    """Actor pool on one node: 1 GPU actor + N CPU actors, all on the same node.

    The GPU slot is **exclusive**: spec §4.4's reuse criterion is "one node runs
    only one Episode at a time", which keeps node affinity simplest at the cost of
    sacrificing same-node parallelism.
    """

    node_id: str
    gpu_actor: object
    cpu_actors: list = field(default_factory=list)
    in_flight: int = 0

    def has_free_gpu(self) -> bool:
        return self.in_flight == 0

    def take_gpu(self) -> None:
        self.in_flight += 1

    def release_gpu(self) -> None:
        self.in_flight = max(0, self.in_flight - 1)


class NodePoolRegistry:
    """Episode → node-pool selection registry.

    ``live_nodes`` is a snapshot **at discovery time**; ``register`` / ``drop``
    maintain the actual pools. The pool count may exceed ``live_nodes`` (a newly
    created pool lands on a new node not yet in the snapshot).
    """

    def __init__(
        self,
        live_nodes: list[NodeInfo],
        cpu_actors_per_node: int,
        create_pool: Callable[[str], NodePool],
    ):
        self._live = {n.node_id: n for n in live_nodes}
        self.cpu_actors_per_node = int(cpu_actors_per_node)
        self._create_pool = create_pool
        self.pools: dict[str, NodePool] = {}

    def register(self, pool: NodePool) -> None:
        self.pools[pool.node_id] = pool

    def drop(self, node_id: str) -> None:
        self.pools.pop(node_id, None)

    def live_node_ids(self) -> set[str]:
        """Available nodes = live at discovery ∪ pools built (including newly scaled-out nodes)."""
        return set(self._live) | set(self.pools)

    def pool_for_episode(self, episode_id: str, assigned: dict) -> NodePool | None:
        """Pick a node pool for this Episode; return None when no node is available and
        none can be created.

        Priority (spec §4.4):
        1. The Episode's ``assigned_node`` is still among live nodes -> **must
           return to the original node** (otherwise the workspace is not on this
           machine and the whole thing must be rerun)
        2. The free pool with the fewest in-flight Episodes
        3. All full -> create a new pool (``create_pool`` makes an affinity-free GPU
           actor; the landing node is decided by Ray + the autoscaler)
        """
        node_id = assigned.get(episode_id)
        if node_id:
            if node_id in self.pools:
                return self.pools[node_id]
            if node_id in self._live:
                pool = self._create_pool(node_id)
                self.register(pool)
                return pool
            return None  # the original node is gone: the caller must rerun the whole Episode

        if not self._live and not self.pools:
            return None  # no GPU nodes (local test machine / CPU-only cluster)

        free = [p for p in self.pools.values() if p.has_free_gpu()]
        if free:
            return min(free, key=lambda p: (p.in_flight, p.node_id))

        # all full: create a new one. node_id is left empty for now; create_pool
        # backfills it after the actor is created (the GPU actor has no affinity,
        # the landing node is decided by Ray + the autoscaler).
        pool = self._create_pool("")
        self.register(pool)
        return pool
```

- [ ] **Step 4: run the tests and confirm they pass**

```bash
RAY_ADDRESS=local python -m pytest \
  tests/unit/test_node_pools.py -q -o addopts=""
```
Expected: 8 passed

- [ ] **Step 5: commit**

```bash
git add src/roboray/ray_runtime/nodes.py tests/unit/test_node_pools.py
git commit -m "feat(nodes): node discovery and node-pool selection (pure logic, unit-testable)

Selection rules: prefer the Episode's original node (checkpoint resume) -> the
least-loaded free pool -> create new when all full (no affinity, giving the
autoscaler a chance to add nodes). Returns None with no GPU nodes and the caller
falls back to old behavior -- the local test machine has no GPU, so this is the
norm, not an anomaly."
```

---

### Task 5: `build_node_pools` and new config options

**Files:**
- Modify: `src/roboray/ray_runtime/actors.py` (`build_actor_pool` at end of file), `src/roboray/api/config.py:37-43`
- Test: `tests/unit/test_actors.py`

**Interfaces:**
- Produces:
  - `build_node_pools(num_gpus: int, num_cpu_actors: int) -> tuple[list, list]` -- **return signature unchanged** (`(gpu_actors, cpu_actors)`), so existing callers and tests need no changes
  - `ExecutionConfig.cpu_actors_per_node: int = 2`, `ExecutionConfig.node_wait_timeout_s: float = 600.0`

**Key design:** node affinity is enforced by **actor placement**, not by a per-task `scheduling_strategy` -- calling a method on an actor always executes on its node. So `build_node_pools` only needs to pin at **creation time**, and the driver only needs to route each Episode to the right actor index.

- [ ] **Step 1: write a failing test**

Append to `tests/unit/test_actors.py`:

```python
def test_build_node_pools_falls_back_on_gpu_less_machine():
    """Review Focus #2: fall back to the original single-pool behavior with no GPU nodes.

    This machine has no GPU, so Ray uses a virtual num_gpus -- node discovery
    necessarily returns empty. build_node_pools must still return usable actor
    lists, or all e2e tests break.
    """
    from roboray.ray_runtime.actors import build_node_pools

    gpu_actors, cpu_actors = build_node_pools(num_gpus=1, num_cpu_actors=2)
    assert len(gpu_actors) == 1
    assert len(cpu_actors) == 2
```

- [ ] **Step 2: run the test and confirm it fails**

```bash
RAY_ADDRESS=local python -m pytest \
  tests/unit/test_actors.py -q -o addopts="" -k falls_back_on_gpu_less_machine
```
Expected: FAIL, `ImportError: cannot import name 'build_node_pools'`

- [ ] **Step 3: add the config options**

`src/roboray/api/config.py`, change `ExecutionConfig` to:

```python
@dataclass(frozen=True)
class ExecutionConfig:
    max_gpus: int = 1
    num_cpu_actors: int = 4
    max_episodes_in_flight: int = 2
    # "real" uses the real models (production default); "test" only registers
    # stubs for local end-to-end tests.
    backend: str = "real"
    # number of CPU actors per node pool (pooled with the 1 GPU actor on the same node).
    cpu_actors_per_node: int = 2
    # cap (seconds) on waiting for "this Episode's node slot" to free up. On timeout
    # the Episode is counted as failed and leaves a searchable error rather than
    # silently hanging -- node affinity is hard; if it cannot be waited for, it
    # simply cannot be.
    node_wait_timeout_s: float = 600.0
```

- [ ] **Step 4: implement `build_node_pools`**

`src/roboray/ray_runtime/actors.py`, replace the whole of `build_actor_pool` with:

```python
def build_node_pools(num_gpus: int, num_cpu_actors: int) -> tuple:
    """Create GPU / CPU actor pools, returning ``(gpu_actors, cpu_actors)``.

    **The return structure is unchanged** (existing callers and tests need no
    changes). With GPU nodes, each node gets 1 GPU actor + ``num_cpu_actors`` CPU
    actors, all pinned to that node via
    ``NodeAffinitySchedulingStrategy(soft=False)`` -- **affinity is enforced by
    actor placement**: calling a method on an actor always executes on its node,
    so no per-task strategy is needed.

    With no GPU nodes (local test machine / CPU-only cluster) it falls back to the
    original behavior: flat pools by count, no affinity applied. This is **the
    norm, not an anomaly** -- all e2e tests run in this environment.
    """
    from roboray.ray_runtime.nodes import discover_gpu_nodes

    nodes = discover_gpu_nodes()
    if not nodes:
        gpu_actors = [GpuModelActor.remote() for _ in range(max(0, num_gpus))]
        cpu_actors = [CpuStageActor.remote() for _ in range(max(1, num_cpu_actors))]
        return gpu_actors, cpu_actors

    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    per_node_cpu = max(1, int(num_cpu_actors))
    gpu_actors, cpu_actors = [], []
    for node in nodes[: max(1, num_gpus)]:
        strategy = NodeAffinitySchedulingStrategy(node_id=node.node_id, soft=False)
        gpu_actors.append(GpuModelActor.options(scheduling_strategy=strategy).remote())
        cpu_actors.extend(
            CpuStageActor.options(scheduling_strategy=strategy).remote()
            for _ in range(per_node_cpu)
        )
    return gpu_actors, cpu_actors


# compatibility alias: existing tests and callers still use the old name.
build_actor_pool = build_node_pools
```

Add `"build_node_pools"` to `__all__`.

- [ ] **Step 5: run the tests and confirm they pass**

```bash
RAY_ADDRESS=local python -m pytest \
  tests/unit/test_actors.py -q -o addopts=""
```
Expected: 9 passed

- [ ] **Step 6: commit**

```bash
git add src/roboray/ray_runtime/actors.py src/roboray/api/config.py tests/unit/test_actors.py
git commit -m "feat(actors): build_node_pools builds pools per GPU node (affinity enforced by actor placement)

With no GPU nodes it falls back to the original flat behavior -- the local test
machine has no GPU, this is the norm. Return structure unchanged. Affinity needs
no per-task scheduling_strategy: calling a method on an actor always executes on
its node."
```

---

### Task 6: driver integrates the node pool (selection + `assigned_node` + dispatch routing)

**Files:**
- Modify: `src/roboray/ray_runtime/actors.py` (add `node_id()` to both Actors), `src/roboray/runtime/driver.py:46-64,165-288`
- Test: `tests/unit/test_actors.py`, `tests/e2e/test_public_api.py` (regression)

**Interfaces:**
- Consumes: Task 4's `NodePoolRegistry` / `NodePool` / `discover_gpu_nodes`; Task 5's `build_node_pools`
- Produces:
  - `GpuModelActor.node_id() -> str`, `CpuStageActor.node_id() -> str` (`ray.get_runtime_context().get_node_id()`)
  - `PipelineDriver.node_of(actor) -> str`
  - `PipelineDriver._pool_index_for(episode_id: str) -> int | None` (returns an index into `self.gpu_actors`; `None` = no available node)
  - the `driver` writes `assigned_node` into `EpisodeRecord` before dispatching a GPU task

- [ ] **Step 1: write failing tests**

Append to `tests/unit/test_actors.py`:

```python
def test_gpu_actor_reports_its_node_id():
    """The driver builds the "Episode → node" mapping from the actor's self-reported node_id.

    An actor may be created without affinity (new pool), with the landing node
    decided by Ray + the autoscaler, so it can only be asked **after** creation.
    """
    from roboray.ray_runtime.actors import GpuModelActor

    actor = GpuModelActor.remote()
    node_id = ray.get(actor.node_id.remote())
    assert isinstance(node_id, str) and node_id


def test_cpu_actor_reports_its_node_id():
    from roboray.ray_runtime.actors import CpuStageActor

    actor = CpuStageActor.remote()
    assert isinstance(ray.get(actor.node_id.remote()), str)
```

- [ ] **Step 2: run the tests and confirm they fail**

```bash
RAY_ADDRESS=local python -m pytest \
  tests/unit/test_actors.py -q -o addopts="" -k reports_its_node_id
```
Expected: FAIL, `AttributeError: 'GpuModelActor' object has no attribute 'node_id'`

- [ ] **Step 3: add `node_id()` to both Actors**

`src/roboray/ray_runtime/actors.py`:

After `GpuModelActor.status()` add:

```python
    def node_id(self) -> str:
        """The Ray node this Actor is on.

        The driver uses it to build the "Episode → node" mapping: a GPU actor in a
        newly created pool is created **without** affinity (the landing node is
        decided by Ray + the autoscaler), so it can only be asked after creation.
        """
        return ray.get_runtime_context().get_node_id()
```

After `CpuStageActor.registered_models()` add the same method (same implementation; comment says "same as GpuModelActor, used to verify the CPU actor is on the same node as this Episode's GPU actor").

- [ ] **Step 4: driver builds the node mapping and wires in selection**

`src/roboray/runtime/driver.py`:

Add to the top-of-file imports:

```python
from roboray.ray_runtime.nodes import NodePool, NodePoolRegistry, discover_gpu_nodes
```

In `__init__`, **before** `self.slots = ...` insert:

```python
        # the Episode → node mapping. Affinity is enforced by actor placement
        # (see build_node_pools); the driver only routes each Episode to the
        # actor at the right index.
        self._actor_node: dict[int, str] = {}
        self.node_registry = NodePoolRegistry(
            live_nodes=discover_gpu_nodes(),
            cpu_actors_per_node=self.request.execution.cpu_actors_per_node,
            create_pool=self._create_pool,
        )
        self._register_existing_pools(gpu_actors, cpu_actors)
```

After `__init__` add three methods:

```python
    def _register_existing_pools(self, gpu_actors: list, cpu_actors: list) -> None:
        """Register the actors created at startup as node pools.

        With no GPU nodes (local test machine) the actors have no affinity and
        ``node_id()`` still returns the local node id; all actors then share one
        node, which behaves like the pre-change flat pool.
        """
        import ray as _ray

        node_ids = _ray.get([a.node_id.remote() for a in gpu_actors]) if gpu_actors else []
        for idx, node_id in enumerate(node_ids):
            self._actor_node[idx] = node_id
            self.node_registry.register(
                NodePool(node_id=node_id, gpu_actor=gpu_actors[idx], cpu_actors=list(cpu_actors))
            )

    def _create_pool(self, node_id: str) -> NodePool:
        """Create a new node pool (called by the registry when all existing pools are full).

        Empty ``node_id`` = "a new node is needed": the GPU actor is created
        **without** affinity, the landing node is decided by Ray + the autoscaler;
        after creation ask it for its own node_id and build CPU actors accordingly.
        """
        from roboray.ray_runtime.actors import CpuStageActor, GpuModelActor

        if node_id:
            from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

            strategy = NodeAffinitySchedulingStrategy(node_id=node_id, soft=False)
            gpu = GpuModelActor.options(scheduling_strategy=strategy).remote()
        else:
            gpu = GpuModelActor.remote()
        actual = ray.get(gpu.node_id.remote())
        strategy = NodeAffinitySchedulingStrategy(node_id=actual, soft=False)
        cpus = [
            CpuStageActor.options(scheduling_strategy=strategy).remote()
            for _ in range(self.request.execution.cpu_actors_per_node)
        ]
        self.gpu_actors.append(gpu)
        self.cpu_actors.extend(cpus)
        self._actor_node[len(self.gpu_actors) - 1] = actual
        self.slots.append(GpuSlot(len(self.gpu_actors) - 1))
        return NodePool(node_id=actual, gpu_actor=gpu, cpu_actors=cpus)

    def _pool_index_for(self, episode_id: str) -> int | None:
        """Which GPU actor this Episode should use; ``None`` = no available node.

        After selection, writes ``assigned_node`` into the record for checkpoint
        resume (back to the original node) and dead-node detection (Task 7).
        """
        rec = self._ws(episode_id).load()
        assigned = {episode_id: rec.assigned_node} if rec.assigned_node else {}
        pool = self.node_registry.pool_for_episode(episode_id, assigned)
        if pool is None:
            return None
        if rec.assigned_node != pool.node_id:
            rec.assigned_node = pool.node_id
            self._ws(episode_id).save_record(rec)
        for idx, node_id in self._actor_node.items():
            if node_id == pool.node_id and self.gpu_actors[idx] is pool.gpu_actor:
                return idx
        return None
```

- [ ] **Step 5: dispatch loop filters slots by node**

`src/roboray/runtime/driver.py`, replace the dispatch section of `_run_pipeline` (`driver.py:223-244`) with:

```python
        while scene_q or hand_q or cpu_q or in_flight:
            # 1) dispatch GPU tasks (filter eligible slots by the Episode's node)
            while free_slots:
                head_item = scene_q[0] if scene_q else (hand_q[0] if hand_q else None)
                if head_item is None:
                    break
                want_idx = self._pool_index_for(head_item.episode_id)
                eligible = [
                    (s, a) for s, a in free_slots
                    if want_idx is not None and s.slot_id == want_idx
                ]
                if not eligible:
                    # this Episode's node has no free slot: skip it this round and
                    # let Episodes on other nodes run first (the queues may hold
                    # tasks from other nodes).
                    moved = scene_q.pop(0) if scene_q else hand_q.pop(0)
                    (scene_q if moved.kind == "scene" else hand_q).append(moved)
                    if len(scene_q) + len(hand_q) <= 1:
                        break
                    continue
                slot, actor = eligible[0]
                if self.config.adaptive:
                    a = self.scheduler.assign(scene_q, hand_q, slot)
                else:
                    a = self._static_assign(scene_q, hand_q, slot)
                if a is None:
                    break
                free_slots.remove((slot, actor))
                q = scene_q if a.kind == "scene" else hand_q
                item = q.pop(0)
                spec = self.specs[item.episode_id]
                ws = self._ws(item.episode_id)
                stage = Stage.SCENE if a.kind == "scene" else Stage.HAND
                ws.mark_running(stage)
                self._event(item.episode_id, stage, "start")
                payload = self._gpu_payload(a.kind, spec, item, ws)
                slot.busy = True
                ref = actor.run_task.remote(payload)
                in_flight[ref] = (item.episode_id, a.kind, item.model_name, slot, actor)
```

Also release the pool's occupancy count in the reap section (`driver.py:270-277`):

```python
            for ref in done:
                ep, kind, model_name, slot, actor = in_flight.pop(ref)
                res = self._get_result(ref)
                self._complete((ep, kind, model_name, slot), res)
                if slot is not None:
                    free_slots.append((slot, actor))
                    node = self._actor_node.get(slot.slot_id)
                    if node and node in self.node_registry.pools:
                        self.node_registry.pools[node].release_gpu()
                else:
                    free_cpu.append(actor)
```

And after a successful dispatch (after `in_flight[ref] = ...`) record the occupancy:

```python
                node = self._actor_node.get(slot.slot_id)
                if node and node in self.node_registry.pools:
                    self.node_registry.pools[node].take_gpu()
```

In `_execute_one` (serial mode), change `self.gpu_actors[0]` to select by pool:

```python
            idx = self._pool_index_for(spec.episode_id)
            if idx is None:
                ws.mark_failed(stage, "no available GPU node")
                return False
            res = self._get_result(self.gpu_actors[idx].run_task.remote(payload))
```

- [ ] **Step 6: run the tests and confirm they pass**

```bash
RAY_ADDRESS=local python -m pytest \
  tests/unit/test_actors.py tests/e2e/test_public_api.py -q -o addopts="" --timeout=600
```
Expected: all PASS

- [ ] **Step 7: run the full regression**

```bash
RAY_ADDRESS=local python -m pytest -q --tb=short
```
Expected: `>= 217 passed`, 0 failed

- [ ] **Step 8: commit**

```bash
git add src/roboray/ray_runtime/actors.py src/roboray/runtime/driver.py tests/unit/test_actors.py
git commit -m "feat(driver): integrate the node pool -- Episode selection, assigned_node persisted, per-node dispatch

Affinity is enforced by actor placement; the driver only routes each Episode to
the actor at the right index. With no GPU nodes NodePoolRegistry returns None and
it falls back to single-pool flat behavior, so all e2e tests are unaffected."
```

---

### Task 7: dead node → rerun the whole Episode

**Files:**
- Modify: `src/roboray/runtime/episode.py:127-143` (`reconcile_for_new_run`), `src/roboray/runtime/driver.py:131-142` (`_reconcile_states`)
- Test: `tests/unit/test_episode.py`

**Interfaces:**
- Consumes: `EpisodeRecord.assigned_node` written by Task 6
- Produces: `EpisodeRecord.reconcile_for_new_run(live_nodes: set | None = None) -> list`

- [ ] **Step 1: write failing tests**

Append to `tests/unit/test_episode.py`:

```python
def test_reconcile_resets_all_stages_when_assigned_node_is_gone():
    """Dead node -> rerun the whole thing; must not just flip RUNNING back to PENDING.

    The workspace disappears with the node: resetting only RUNNING would leave a
    fake "prepare DONE" and the next round would crash trying to read a
    non-existent prepared/.
    """
    from roboray.runtime.episode import EpisodeRecord, Stage, StageStatus

    rec = EpisodeRecord.new("ep_0001")
    rec.stages[Stage.PREPARE] = StageStatus.DONE
    rec.stages[Stage.SCENE] = StageStatus.DONE
    rec.stages[Stage.HAND] = StageStatus.DONE
    rec.assigned_node = "node-gone"

    changes = rec.reconcile_for_new_run(live_nodes={"node-alive"})
    assert rec.stages[Stage.PREPARE] is StageStatus.PENDING
    assert rec.stages[Stage.SCENE] is StageStatus.PENDING
    assert rec.stages[Stage.HAND] is StageStatus.PENDING
    assert rec.assigned_node is None
    assert len(changes) == 3


def test_reconcile_keeps_done_stages_when_node_still_live():
    """Node still live -> keep completed stages (checkpoint-resume semantics unchanged)."""
    from roboray.runtime.episode import EpisodeRecord, Stage, StageStatus

    rec = EpisodeRecord.new("ep_0001")
    rec.stages[Stage.PREPARE] = StageStatus.DONE
    rec.assigned_node = "node-1"

    changes = rec.reconcile_for_new_run(live_nodes={"node-1"})
    assert rec.stages[Stage.PREPARE] is StageStatus.DONE
    assert rec.assigned_node == "node-1"
    assert changes == []


def test_reconcile_without_live_nodes_does_not_touch_assignment():
    """Without live_nodes (old callers) -> no node check, behavior unchanged."""
    from roboray.runtime.episode import EpisodeRecord, Stage, StageStatus

    rec = EpisodeRecord.new("ep_0001")
    rec.stages[Stage.PREPARE] = StageStatus.DONE
    rec.assigned_node = "whatever"

    changes = rec.reconcile_for_new_run()
    assert rec.stages[Stage.PREPARE] is StageStatus.DONE
    assert rec.assigned_node == "whatever"
    assert changes == []
```

- [ ] **Step 2: run the tests and confirm they fail**

```bash
RAY_ADDRESS=local python -m pytest \
  tests/unit/test_episode.py -q -o addopts="" -k "assigned_node or live_nodes"
```
Expected: FAIL, `TypeError: reconcile_for_new_run() got an unexpected keyword argument 'live_nodes'`

- [ ] **Step 3: implement**

`src/roboray/runtime/episode.py`, replace the whole of `reconcile_for_new_run` with:

```python
    def reconcile_for_new_run(self, live_nodes: set | None = None) -> list:
        """Align state for a new run: crashes/failures across runs return to a resumable
        semantic.

        - RUNNING → PENDING: crash residue; no in-flight task can exist when a run starts.
        - FAILED → PENDING: the cross-run retry semantic of spec §5, "rerun resumes from
          the first incomplete stage".
        - **assigned_node no longer in live_nodes -> all stages back to PENDING**: the
          workspace is node-local, so a gone node means a gone workspace. At that point
          **only** flipping RUNNING back to PENDING is wrong -- it would leave a fake
          "prepare DONE" and the next round would read a non-existent prepared/.
        (Within a run, FAILED stays terminal; the Driver keeps it as-is for the current run.)

        ``live_nodes=None`` means no node check (old callers keep their behavior).
        Returns the list of changes [(stage, from_status)] so the caller can persist as needed.
        """
        changes = []

        node_lost = (
            live_nodes is not None
            and self.assigned_node is not None
            and self.assigned_node not in live_nodes
        )
        if node_lost:
            for stage in STAGE_ORDER:
                if self.stages[stage] is not StageStatus.PENDING:
                    changes.append((stage, self.stages[stage]))
                    self.stages[stage] = StageStatus.PENDING
            self.assigned_node = None
            self.error = None
            return changes

        for stage in STAGE_ORDER:
            cur = self.stages[stage]
            if cur in (StageStatus.RUNNING, StageStatus.FAILED):
                self.stages[stage] = StageStatus.PENDING
                changes.append((stage, cur))
        if changes:
            self.error = None
        return changes
```

- [ ] **Step 4: driver passes the live nodes**

`src/roboray/runtime/driver.py`, replace the whole of `_reconcile_states` with:

```python
    def _reconcile_states(self) -> None:
        """Align all Episode states at run start (cross-run semantics, see
        EpisodeRecord.reconcile_for_new_run).

        Additionally checks ``assigned_node``: the workspace is node-local, so a
        gone node means a gone workspace and the Episode must be rerun in full
        (not just reset RUNNING).

        With no GPU nodes (local test machine) pass ``None`` to skip the node check --
        there is nothing to check, and the workspace is on the local machine with no
        failure risk.
        """
        nodes = discover_gpu_nodes()
        live_nodes = {n.node_id for n in nodes} if nodes else None

        for ep in self.specs:
            ws = self._ws(ep)
            rec = ws.load()
            changes = rec.reconcile_for_new_run(live_nodes=live_nodes)
            if changes:
                ws.save_record(rec)
```

- [ ] **Step 5: run the tests and confirm they pass**

```bash
RAY_ADDRESS=local python -m pytest \
  tests/unit/test_episode.py -q -o addopts=""
```
Expected: all PASS

- [ ] **Step 6: run the full regression**

```bash
RAY_ADDRESS=local python -m pytest -q --tb=short
```
Expected: `>= 220 passed`, 0 failed

- [ ] **Step 7: commit**

```bash
git add src/roboray/runtime/episode.py src/roboray/runtime/driver.py tests/unit/test_episode.py
git commit -m "feat(driver): rerun the whole Episode when its node dies

The workspace is node-local, so a gone node means a gone workspace. Only
flipping RUNNING back to PENDING would leave a fake "prepare DONE" and the next
round would read a non-existent prepared/. live_nodes=None means no check
(old callers and GPU-less environments keep their behavior)."
```

---

### Task 8: hard-invariant acceptance -- driver has zero workspace access

**Files:**
- Create: `tests/e2e/test_driver_workspace_isolation.py`

**Interfaces:**
- Consumes: all the changes from the previous seven tasks
- Produces: one end-to-end invariant test that pins E1's core promise

- [ ] **Step 1: write the test**

Create `tests/e2e/test_driver_workspace_isolation.py`:

```python
"""E1 hard invariant: **the driver process never accesses any workspace file**.

This invariant is E1's core promise -- under multi-node the driver runs on the
head/CPU worker and cannot read a node-local workspace. Tasks 1-6 all serve it;
this is the end-to-end acceptance.

Approach: wrap ``Path.read_bytes`` / ``Path.read_text`` / ``np.load`` inside the
driver process, recording every access that lands under ``workspace_root``; run a
complete fake e2e; assert the access count is 0.

Note: Ray actors run in **separate processes**, so their disk reads do not pass
through this patch -- which is exactly what we want to verify: all disk reads
should happen inside actor processes.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from roboray.api.config import ReconstructionRequest
from roboray.api.reconstruction import EgoReconstruction

pytestmark = pytest.mark.slow


def _make_input(root: Path, n_episodes: int = 2, n_frames: int = 4) -> Path:
    """Build a minimal input: only the mp4 + calibration + meta that prepare needs."""
    import cv2

    root.mkdir(parents=True, exist_ok=True)
    for e in range(n_episodes):
        ep = root / f"ep_{e + 1:04d}"
        ep.mkdir(parents=True, exist_ok=True)
        w, h = 64, 48
        writer = cv2.VideoWriter(
            str(ep / "rgb_left.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h)
        )
        for i in range(n_frames):
            writer.write(np.full((h, w, 3), 40 + i, np.uint8))
        writer.release()
        (ep / "calibration.json").write_text(json.dumps({
            "K": [[51.2, 0.0, w / 2], [0.0, 51.2, h / 2], [0.0, 0.0, 1.0]],
            "dist": [0.0] * 5, "T_left_to_right": np.eye(4).tolist(),
            "fps": 30.0, "width": w, "height": h,
        }))
        (ep / "meta.json").write_text(json.dumps(
            {"episode_id": f"ep_{e + 1:04d}", "device": "test", "n_frames": n_frames}
        ))
    return root


def test_driver_never_reads_workspace_files(tmp_path, monkeypatch):
    """The driver process's actual access count for files under workspace_root must be 0."""
    storage_root = _make_input(tmp_path / "input")
    workspace_root = tmp_path / "run"
    output_root = tmp_path / "out"

    hits: list[str] = []
    ws_prefix = str(workspace_root.resolve())

    real_read_bytes = Path.read_bytes
    real_read_text = Path.read_text
    real_np_load = np.load

    def _note(p) -> None:
        try:
            resolved = str(Path(p).resolve())
        except OSError:
            return
        if resolved.startswith(ws_prefix):
            hits.append(resolved)

    def _read_bytes(self):
        _note(self)
        return real_read_bytes(self)

    def _read_text(self, *a, **k):
        _note(self)
        return real_read_text(self, *a, **k)

    def _np_load(file, *a, **k):
        _note(file)
        return real_np_load(file, *a, **k)

    monkeypatch.setattr(Path, "read_bytes", _read_bytes)
    monkeypatch.setattr(Path, "read_text", _read_text)
    monkeypatch.setattr(np, "load", _np_load)

    request = ReconstructionRequest.from_dict({
        "input": {"uri": str(storage_root)},
        "output": {"uri": str(output_root), "format": "lerobot_v3"},
        "execution": {"max_gpus": 1, "num_cpu_actors": 2,
                      "max_episodes_in_flight": 2, "backend": "test"},
        "stages": {
            "scene_reconstruction": {"model": "fake_scene"},
            "hand_reconstruction": {"model": "fake_hand"},
            "action_understanding": {"model": "fake_vlm"},
            "visualization": {"camera_view": True, "world_view": True},
        },
        "models": {
            "lerobot_python": "python",
            "lerobot_script": "tests/fakes/modelenv_scripts/fake_lerobot_export.py",
        },
    })
    result = EgoReconstruction(request, run_root=str(workspace_root)).run()
    assert not result.failed, f"e2e failed: {result.failed}"

    assert hits == [], (
        "the driver process accessed workspace files (E1 hard invariant broken):\n  "
        + "\n  ".join(hits[:10])
    )
```

- [ ] **Step 2: run the test**

```bash
RAY_ADDRESS=local python -m pytest \
  tests/e2e/test_driver_workspace_isolation.py -q -o addopts="" --timeout=600
```
Expected: PASS

If it FAILs with non-empty `hits`, some driver-side disk-read path is still un-cleaned -- **do not relax the assertion**; go back to the Task at the failure point and fix the read.

- [ ] **Step 3: run the full regression**

```bash
RAY_ADDRESS=local python -m pytest -q --tb=short
```
Expected: `>= 221 passed`, 0 failed

- [ ] **Step 4: commit**

```bash
git add tests/e2e/test_driver_workspace_isolation.py
git commit -m "test(e2e): pin E1 hard invariant -- driver process has zero workspace access

Under multi-node the driver runs on the head/CPU worker and cannot read a
node-local workspace. Patch Path.read_bytes/read_text and np.load in the driver
process to record accesses, run a full fake e2e, then assert the access count is
0. Ray actors run in separate processes so their reads bypass this patch -- which
is exactly what we verify: all reads should happen inside actor processes."
```

---

## Self-Review

**1. Spec coverage (item by item against spec section 4)**

| spec item | covering task |
|---|---|
| 4.1 driver never touches any workspace file | Task 1 (return values) + Task 2 (routing metadata) + Task 8 (acceptance) |
| 4.2 payload construction moves down | Task 3 |
| 4.3 the last two driver-side disk reads removed | Task 2 |
| 4.4 node pools and Episode selection | Task 4 (rules) + Task 5 (pool building) + Task 6 (driver integration: selection, `assigned_node` persisted, per-node dispatch) |
| 4.5 node failure reruns the whole Episode | Task 7 |
| 4.6 storage in three layers | **not in this plan** (pure deployment artifacts, belongs to E3) |

**2. Placeholder scan**: no TBD / TODO / "like Task N"; every code step gives complete code.

**3. Type consistency**: `run_cpu_stage` / `read_prepared_frames` / `scene_payload_args` /
`hand_payload_args` / `build_node_pools` / `NodePoolRegistry.pool_for_episode` /
`EpisodeRecord.record_prepare` / `reconcile_for_new_run(live_nodes=)` / `node_id()` /
`_pool_index_for` -- definitions and references agree across tasks. Specifically checked:
- Task 4's `NodePool(node_id, gpu_actor, cpu_actors, in_flight)` is constructed by Task 6's `_create_pool` / `_register_existing_pools` with the same keyword names;
- Task 4's `pool_for_episode(episode_id, assigned)` signature matches Task 6's `_pool_index_for` call;
- Task 1's `stage_result` key is consumed by Task 2's `_complete` backfill logic.

**4. Review Focus**: all five land in concrete tasks (#1→Task 2's legacy state.json test, #2→Task 4's
`empty_live_nodes` test and Task 5's fallback test, #3→Task 4's `single_node_cluster` test,
#4→Task 5's `node_wait_timeout_s` config option + Task 6's `_pool_index_for` None path and
`_execute_one`'s "no available GPU node" failure branch, #5→Task 3 Step 9 full regression).

**Known uncovered (explicitly recorded, out of scope for this plan)**:
- `node_wait_timeout_s`'s **actual timeout enforcement**: the current dispatch loop skips-and-rotates
  on "this node's slot is busy" rather than blocking, so there is no infinite hang; the option's
  enforcement semantics (timeout counts as failure) only need implementing once a **blocking wait**
  appears (E2 scheduler rework). Registered in spec §10's risk table.
- **Runtime dead-node detection** (an actor dies mid-run): Task 7 only covers the reconcile at run
  start; a mid-run death is converted to that stage FAILED by `_get_result`'s exception handling,
  and the next run reruns it wholesale via Task 7.
