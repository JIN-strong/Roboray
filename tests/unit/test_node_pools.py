"""Node-pool placement rules (pure logic, no Ray runtime dependency)."""

import pytest

from roboray.ray_runtime.nodes import NodeInfo, NodePool, NodePoolRegistry


def _pool(node_id: str, cpu_actors: int = 2) -> NodePool:
    return NodePool(node_id=node_id, gpu_actors=[object()], cpu_actors=[object()] * cpu_actors)


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
    """Resumable run: the episode already has an assigned_node and that node is still live -> must return to the original node.

    If we cannot return to the original node, the workspace is not on that machine and the
    episode must be re-run in full.
    """
    reg, _ = _registry([NodeInfo("n1", 1, "h1"), NodeInfo("n2", 1, "h2")])
    reg.register(_pool("n1"))
    reg.register(_pool("n2"))
    chosen = reg.pool_for_episode("ep_0001", {"ep_0001": "n2"})
    assert chosen.node_id == "n2"


def test_assigned_node_no_longer_live_returns_none():
    """Original node no longer in the live list -> return None (the caller triggers a full re-run)."""
    reg, _ = _registry([NodeInfo("n1", 1, "h1")])
    reg.register(_pool("n1"))
    assert reg.pool_for_episode("ep_0001", {"ep_0001": "gone"}) is None


def test_picks_least_loaded_live_node():
    """With no assigned_node, pick the node with the fewest in-flight episodes."""
    reg, _ = _registry([NodeInfo("n1", 1, "h1"), NodeInfo("n2", 1, "h2")])
    p1, p2 = _pool("n1"), _pool("n2")
    reg.register(p1)
    reg.register(p2)
    p1.take_gpu()
    assert reg.pool_for_episode("ep_x", {}).node_id == "n2"


def test_creates_new_pool_when_all_full():
    """All pools full -> create a new one (no affinity, so the autoscaler gets a chance to scale out a node)."""
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
    """A CPU-only machine / no GPU nodes -> return None, must not raise.

    All existing tests run on GPU-less machines (Ray uses virtual num_gpus), so an
    empty node discovery is the norm, not an exception.
    """
    reg, _ = _registry([])
    assert reg.pool_for_episode("ep_x", {}) is None
    assert reg.live_node_ids() == set()


def test_single_node_cluster_all_episodes_land_there():
    """On a single-node cluster, all episodes land on the same node."""
    reg, _ = _registry([NodeInfo("only", 1, "h")])
    reg.register(_pool("only"))
    assert reg.pool_for_episode("ep_a", {}).node_id == "only"
    assert reg.pool_for_episode("ep_b", {}).node_id == "only"


def test_drop_removes_pool_and_node_id():
    """After node loss, drop it: no longer in the live set, and never chosen again."""
    reg, _ = _registry([NodeInfo("n1", 1, "h1"), NodeInfo("n2", 1, "h2")])
    reg.register(_pool("n1"))
    reg.register(_pool("n2"))
    reg.drop("n1")
    assert "n1" not in reg.live_node_ids()
    assert reg.pool_for_episode("ep_x", {}).node_id == "n2"


def test_gpu_slot_is_exclusive():
    """A node pool runs only one episode at a time (the reuse criterion)."""
    p = _pool("n1")
    assert p.has_free_gpu() is True
    p.take_gpu()
    assert p.has_free_gpu() is False
    p.release_gpu()
    assert p.has_free_gpu() is True


def test_run_pipeline_new_pool_slot_enters_dispatch_view(tmp_path, monkeypatch):
    """A new pool's slot must enter the dispatch view, otherwise multi-node livelock.

    Repro scenario: single-node cluster A already has 1 slot and is fully in flight
    (``in_flight=1``); two episodes - ep_old is assigned to A (GPU ready, queued),
    ep_new unassigned. ep_new goes through ``pool_for_episode``'s "all full -> create
    new pool": ``_create_pool`` appends a new slot to ``self.slots`` (the fake actor
    lands on node B, simulating a new node scaled out by the autoscaler) and pins
    ep_new to B. Before the fix, ``free_slots`` was a **pre-loop snapshot** that never
    saw this new slot - ep_new is only eligible for node B's slot (hard affinity
    constraint) and no slot could dispatch it: with the queue holding only it and
    in_flight empty, `push_ready(); continue` spins without sleep (in this scenario A's
    pool `in_flight` never returns to 0, so ep_new cannot reroute back to A).
    After the fix, the idle view is rebuilt from ``self.slots`` every round
    (``slot.busy`` is the sole authority); a new pool's slot (busy=False) created
    mid-dispatch enters the view on the next inner iteration and the pipeline finishes.
    Regression direction: reverting ``_run_pipeline``'s dispatch view to an
    "incremental free_slots snapshot + `while free_slots`" would hang this test
    (pytest's default 300s timeout as the safety net).
    """
    import ray as ray_mod

    from roboray.api.config import ReconstructionRequest
    from roboray.ray_runtime.nodes import NodeInfo, NodePool, NodePoolRegistry
    from roboray.runtime.driver import DriverConfig, PipelineDriver
    from roboray.runtime.episode import (
        EpisodeRecord,
        EpisodeSpec,
        EpisodeWorkspace,
        Stage,
        StageStatus,
    )
    from roboray.scheduling.cost_model import CostModel
    from roboray.scheduling.scheduler import AdaptiveScheduler, GpuSlot

    class _FakeRef:
        """Play an ObjectRef: the return of `actor.run_task.remote(payload)`, immediately
        yielding a success result by payload kind; `ray.wait` is replaced with
        "all immediately reclaimable". """
        def remote(self, payload):
            self.result = {
                "episode_id": payload["episode_id"],
                "kind": payload["kind"],
                "ok": True,
                "error": None,
                "seconds": 0.01,
                "resolution_key": "64x64",
                "was_load": False,
            }
            return self

    class _FakeActor:
        def __init__(self):
            self.run_task = _FakeRef()

    ws_root = tmp_path / "ws"
    req = ReconstructionRequest.from_dict({
        "input": {"uri": str(tmp_path / "in")},
        "output": {"uri": str(tmp_path / "out"), "format": "lerobot_v3"},
        "execution": {"backend": "test", "max_episodes_in_flight": 2,
                      "max_gpus": 2, "num_cpu_actors": 2},
        "stages": {"scene_reconstruction": {"model": "auto"},
                   "hand_reconstruction": {"model": "auto"}},
    })

    # driver built via __new__: skip __init__ to avoid discover_gpu_nodes / real Ray actors.
    d = PipelineDriver.__new__(PipelineDriver)
    d.config = DriverConfig(
        storage_root=str(tmp_path / "in"), workspace_root=str(ws_root),
        output_root=str(tmp_path / "out"), serial_baseline=False, adaptive=True,
    )
    d.request = req
    d.cost_model = CostModel()
    d.specs = {}
    d.records = {}
    d.timeline = []
    d._t0 = 0.0
    d.slots = [GpuSlot(0)]                     # existing 1 slot
    d.gpu_actors = [_FakeActor()]              # slot 0's actor (node A)
    d.cpu_actors = [_FakeActor()]              # for CPU stages (prepare/fusion/...)
    d._actor_node = {0: "A"}
    d._cpu_by_node = {"A": [d.cpu_actors[0]]}  # CPU grouped by node
    d._cpu_busy = set()                        # id(actor) accounting: set on dispatch
    d._cpu_stall = 0                           # guaranteed-convergence counter

    def _fake_create_pool(node_id: str) -> NodePool:
        """Simulate _create_pool's "create a new node pool": append a slot/actor (landing on node B).

        In the real implementation the GPU actor is created without affinity and placement
        is decided by Ray + the autoscaler; here it is fixed to node B - i.e. the bug
        scenario of "a new pool landing on a node with no other slots". Consistent with
        the real _create_pool: the new pool's CPU actors also join the group by actual
        node_id (ep_new's later fusion/action/publish all use group B).
        """
        if node_id:
            raise AssertionError("this test should only take the empty-node_id new-pool path")
        idx = len(d.slots)
        actor = _FakeActor()
        cpu = _FakeActor()
        d.slots.append(GpuSlot(idx))
        d.gpu_actors.append(actor)
        d._actor_node[idx] = "B"
        d.cpu_actors.append(cpu)
        d._cpu_by_node.setdefault("B", []).append(cpu)
        return NodePool(node_id="B", gpu_actors=[actor], cpu_actors=[cpu])

    d.node_registry = NodePoolRegistry(
        live_nodes=[NodeInfo("A", 1, "host-a")],
        cpu_actors_per_node=2,
        create_pool=_fake_create_pool,
    )
    # existing pool A: in_flight=1 means the pool "has an in-flight task", so
    # pool_for_episode sees it as full -> the unassigned ep_new takes the new-pool path.
    pool_a = NodePool(node_id="A", gpu_actors=[d.gpu_actors[0]], cpu_actors=[])
    pool_a.take_gpu()
    d.node_registry.register(pool_a)
    d.scheduler = AdaptiveScheduler(d.slots, d.cost_model)

    # two episodes: ep_old assigned to A with prepare DONE (GPU ready); ep_new unassigned
    # (pool_for_episode: unassigned and all pools full -> create a new pool pinning ep_new to node B).
    for ep, rel, assigned in (("ep_old", "ep_0001", "A"), ("ep_new", "ep_0002", None)):
        d.specs[ep] = EpisodeSpec(episode_id=ep, rel_dir=rel, has_stereo=False)
        rec = EpisodeRecord.new(ep)
        rec.assigned_node = assigned
        d.records[ep] = rec
        EpisodeWorkspace(ws_root, ep).save_record(rec)
    d.records["ep_old"].stages[Stage.PREPARE] = StageStatus.DONE
    d.records["ep_old"].has_stereo = False
    d.records["ep_old"].resolution_key = "64x64"
    EpisodeWorkspace(ws_root, "ep_old").save_record(d.records["ep_old"])

    monkeypatch.setattr(ray_mod, "wait", lambda refs, **kw: (list(refs), []))
    monkeypatch.setattr(d, "_get_result", lambda ref: ref.result)
    d.in_flight = {}

    # before the fix this looped forever (pytest 300s timeout as the safety net);
    # after the fix both episodes complete.
    result = d._run_pipeline()
    assert result.failed == []
    for ep in ("ep_old", "ep_new"):
        assert d.records[ep].is_terminal() is True, f"{ep} did not complete: new-slot regression"
    # the new pool was indeed created and assigned (ep_new landed on new node B's slot)
    assert d.records["ep_new"].assigned_node == "B"
    assert len(d.slots) == 2


# =====================================================================
# CPU-stage node routing
#
# Pure-logic tests, no Ray runtime: `pick_idle` is a driver top-level pure function;
# `_cpu_actor_for` / `_group_cpu_actors` / `_register_existing_pools` are driver glue,
# tested directly via __new__ construction + fake actors + monkeypatch.
# =====================================================================


def _driver_shell(monkeypatch=None):
    """Build a PipelineDriver shell skipping __init__ (only the routing fields needed)."""
    from roboray.runtime.driver import DriverConfig, PipelineDriver

    d = PipelineDriver.__new__(PipelineDriver)
    d.config = DriverConfig(storage_root="/s", workspace_root="/w", output_root="/o")
    d.request = None
    d.cpu_actors = []
    d.gpu_actors = []
    d.slots = []
    d.specs = {}
    d.records = {}
    d.timeline = []
    d._t0 = 0.0
    d._actor_node = {}
    d._cpu_by_node = {}
    d._cpu_busy = set()
    d._cpu_stall = 0
    return d


def test_pick_idle_returns_first_non_busy():
    """Pure function: pick the first non-busy actor in the group in order; None if all busy / empty."""
    from roboray.runtime.driver import pick_idle

    a, b, c = object(), object(), object()
    assert pick_idle([a, b, c], set()) is a
    assert pick_idle([a, b, c], {id(a)}) is b
    assert pick_idle([a, b, c], {id(a), id(b)}) is c
    assert pick_idle([a, b, c], {id(a), id(b), id(c)}) is None
    assert pick_idle([], set()) is None


def test_group_cpu_actors_by_node_id(monkeypatch):
    """`_group_cpu_actors` groups by node_id; an empty list returns {} (skipping remote calls)."""
    import ray as ray_mod

    from roboray.runtime.driver import PipelineDriver

    class _Ref:
        def remote(self):
            return self

    class _NodeActor:
        def __init__(self):
            self.node_id = _Ref()

    a1, a2, b1 = _NodeActor(), _NodeActor(), _NodeActor()
    monkeypatch.setattr(ray_mod, "get", lambda refs, **kw: ["A", "A", "B"])
    d = _driver_shell()
    by_node = d._group_cpu_actors([a1, a2, b1])
    assert list(by_node) == ["A", "B"]
    assert by_node["A"] == [a1, a2]
    assert by_node["B"] == [b1]
    assert d._group_cpu_actors([]) == {}


def test_register_existing_pools_gives_each_pool_its_own_cpu_actors(monkeypatch):
    """Shared-reference fix: each pool gets its **own node's** CPU actor list, not one shared copy."""
    import ray as ray_mod

    from roboray.ray_runtime.nodes import NodePoolRegistry
    from roboray.runtime.driver import PipelineDriver

    class _GpuActor:
        class _Ref:
            def remote(self):
                return self

        def __init__(self):
            self.node_id = self._Ref()  # only consumed by ray.get; no real call after monkeypatch

    d = _driver_shell()
    cpu_a1, cpu_a2, cpu_b1 = object(), object(), object()
    d._cpu_by_node = {"A": [cpu_a1, cpu_a2], "B": [cpu_b1]}
    d.node_registry = NodePoolRegistry(
        live_nodes=[], cpu_actors_per_node=2, create_pool=lambda nid: _pool(nid)
    )
    monkeypatch.setattr(ray_mod, "get", lambda refs, **kw: ["A", "B"])
    gpu_a, gpu_b = _GpuActor(), _GpuActor()
    d._register_existing_pools([gpu_a, gpu_b], [cpu_a1, cpu_a2, cpu_b1])
    assert d._actor_node == {0: "A", 1: "B"}
    assert d.node_registry.pools["A"].cpu_actors == [cpu_a1, cpu_a2]
    assert d.node_registry.pools["B"].cpu_actors == [cpu_b1]
    assert d.node_registry.pools["A"].cpu_actors is not d.node_registry.pools["B"].cpu_actors


def test_cpu_actor_for_routes_to_episode_node_group_only():
    """Assigned to node B -> only pick from group B; A's idle actor is not borrowed (the workspace is node-local)."""
    from roboray.runtime.episode import EpisodeRecord
    from roboray.runtime.driver import PipelineDriver

    d = _driver_shell()
    d.node_registry = object()  # non-None: node routing enabled
    a = object()
    b1, b2 = object(), object()
    d.cpu_actors = [a, b1, b2]
    d._cpu_by_node = {"A": [a], "B": [b1, b2]}
    rec = EpisodeRecord.new("ep_0001")
    rec.assigned_node = "B"
    d.records = {"ep_0001": rec}
    assert d._cpu_actor_for("ep_0001") is b1
    d._cpu_busy = {id(b1)}
    assert d._cpu_actor_for("ep_0001") is b2  # still only from group B, never touches A


def test_cpu_actor_for_all_busy_returns_none_no_borrow():
    """Group B all busy -> None (this episode's CPU stages wait for the next round); A's idle actor is not borrowed."""
    from roboray.runtime.episode import EpisodeRecord
    from roboray.runtime.driver import PipelineDriver

    d = _driver_shell()
    d.node_registry = object()
    a, b1 = object(), object()
    d.cpu_actors = [a, b1]
    d._cpu_by_node = {"A": [a], "B": [b1]}
    rec = EpisodeRecord.new("ep_0001")
    rec.assigned_node = "B"
    d.records = {"ep_0001": rec}
    d._cpu_busy = {id(b1)}
    assert d._cpu_actor_for("ep_0001") is None


def test_cpu_actor_for_node_group_gone_returns_none():
    """Owning node's group does not exist (node gone) -> None; must not fall through to another node."""
    from roboray.runtime.episode import EpisodeRecord
    from roboray.runtime.driver import PipelineDriver

    d = _driver_shell()
    d.node_registry = object()
    a = object()
    d.cpu_actors = [a]
    d._cpu_by_node = {"A": [a]}
    rec = EpisodeRecord.new("ep_0001")
    rec.assigned_node = "gone"
    d.records = {"ep_0001": rec}
    assert d._cpu_actor_for("ep_0001") is None


def test_cpu_actor_for_no_gpu_env_global_idle():
    """No GPU environment (registry None) -> any globally idle CPU actor."""
    from roboray.runtime.episode import EpisodeRecord
    from roboray.runtime.driver import PipelineDriver

    d = _driver_shell()
    d.node_registry = None
    a, b = object(), object()
    d.cpu_actors = [a, b]
    d._cpu_by_node = {"A": [a]}  # group exists but not enabled: scan cpu_actors globally
    rec = EpisodeRecord.new("ep_0001")
    d.records = {"ep_0001": rec}
    assert d._cpu_actor_for("ep_0001") is a
    d._cpu_busy = {id(a), id(b)}
    assert d._cpu_actor_for("ep_0001") is None


def test_cpu_actor_for_unassigned_episode_global_fallback():
    """Routing enabled but the episode not yet assigned a node (prepare runs before any assignment) -> global idle."""
    from roboray.runtime.episode import EpisodeRecord
    from roboray.runtime.driver import PipelineDriver

    d = _driver_shell()
    d.node_registry = object()  # non-None: node routing enabled
    a = object()
    d.cpu_actors = [a]
    d._cpu_by_node = {"B": []}  # no node group yet
    rec = EpisodeRecord.new("ep_0001")  # assigned_node stays None
    d.records = {"ep_0001": rec}
    assert d._cpu_actor_for("ep_0001") is a


def test_run_pipeline_cpu_stall_timeout_fails_episode_and_clears_queue(tmp_path, monkeypatch):
    """Guaranteed convergence: the owning node's CPU group stays free-less (no-progress rounds mapped from
    node_wait_timeout_s) until the threshold -> the head CPU task is marked FAILED and dequeued; the pipeline stops spinning.

    Repro scenario: a single-node cluster has a GPU slot, but some episode is pinned to
    node B which has vanished - `_cpu_actor_for` always returns None; cpu_q non-empty,
    in_flight empty, and `push_ready(); continue` would spin without sleep (the queued
    set blocks re-enqueue). This test uses a tiny node_wait_timeout_s (0.15s -> 3
    rounds) and verifies convergence once the threshold is reached.
    """
    import ray as ray_mod

    from roboray.api.config import ReconstructionRequest
    from roboray.ray_runtime.nodes import NodeInfo, NodePoolRegistry
    from roboray.runtime.driver import DriverConfig, PipelineDriver
    from roboray.runtime.episode import (
        EpisodeRecord,
        EpisodeSpec,
        EpisodeWorkspace,
        Stage,
        StageStatus,
    )
    from roboray.scheduling.cost_model import CostModel
    from roboray.scheduling.scheduler import AdaptiveScheduler

    ws_root = tmp_path / "ws"
    req = ReconstructionRequest.from_dict({
        "input": {"uri": str(tmp_path / "in")},
        "output": {"uri": str(tmp_path / "out"), "format": "lerobot_v3"},
        "execution": {"backend": "test", "node_wait_timeout_s": 0.15},
        "stages": {"scene_reconstruction": {"model": "auto"},
                   "hand_reconstruction": {"model": "auto"}},
    })

    d = PipelineDriver.__new__(PipelineDriver)
    d.config = DriverConfig(
        storage_root=str(tmp_path / "in"), workspace_root=str(ws_root),
        output_root=str(tmp_path / "out"), serial_baseline=False, adaptive=True,
    )
    d.request = req
    d.cost_model = CostModel()
    d.slots = []          # no GPU slots: _scan_idle is a no-op, avoids remote calls
    d.gpu_actors = []
    d.cpu_actors = [object()]
    d._cpu_by_node = {}   # owning node B's group does not exist (node gone)
    d._cpu_busy = set()
    d._cpu_stall = 0
    # real registry (non-None): B not in live, no pool -> pool_for_episode returns None,
    # `_ensure_assigned` returns False -> CPU dispatch skipped -> stall counter accumulates
    # (reproducing the original spin scenario).
    d.node_registry = NodePoolRegistry(
        live_nodes=[NodeInfo("A", 1, "host-a")],
        cpu_actors_per_node=1,
        create_pool=lambda nid: (_ for _ in ()).throw(AssertionError("no pool should be created")),
    )
    d.scheduler = AdaptiveScheduler(d.slots, d.cost_model)
    d.timeline = []
    d._t0 = 0.0

    d.specs = {"ep_0001": EpisodeSpec(episode_id="ep_0001", rel_dir="ep_0001",
                                      has_stereo=False)}
    rec = EpisodeRecord.new("ep_0001")
    rec.assigned_node = "B"  # pinned to a vanished node
    d.records = {"ep_0001": rec}
    EpisodeWorkspace(ws_root, "ep_0001").save_record(rec)

    monkeypatch.setattr(ray_mod, "wait", lambda refs, **kw: ([], []))
    assert d._cpu_stall_limit() == 3  # 0.15 / 0.05 = 3 rounds

    result = d._run_pipeline()
    # guaranteed convergence: ep_0001's prepare is marked FAILED, cpu_q cleared, pipeline exits normally
    assert d.records["ep_0001"].is_terminal() is True
    assert d.records["ep_0001"].stages[Stage.PREPARE] is StageStatus.FAILED
    assert "node_wait_timeout_s" in (d.records["ep_0001"].error or "")
    assert result.failed == ["ep_0001"]


def test_ensure_assigned_writes_assigned_node_before_prepare_dispatch(tmp_path, monkeypatch):
    """Regression: assigned_node must be persisted before the prepare dispatch.

    Under multi-node, if prepare took the global fallback and landed on an arbitrary node X,
    a later scene picking Y via pool_for_episode could not read the prepared/ on X (the loop
    breaks at the first stage). `_ensure_assigned` calls pool_for_episode for the unassigned
    episode before the CPU dispatch and persists assigned_node - the first choice is stable,
    and later scene/hand are forced back to the same node.
    Verified with a fake registry: when the registry is non-None and the episode is
    unassigned, after push_ready (before CPU dispatch) ``rec.assigned_node`` is already
    non-None and equals pool_for_episode's choice.
    """
    import ray as ray_mod

    from roboray.api.config import ReconstructionRequest
    from roboray.ray_runtime.nodes import NodeInfo, NodePool, NodePoolRegistry
    from roboray.runtime.driver import DriverConfig, PipelineDriver
    from roboray.runtime.episode import (
        EpisodeRecord,
        EpisodeSpec,
        EpisodeWorkspace,
    )
    from roboray.scheduling.cost_model import CostModel
    from roboray.scheduling.scheduler import AdaptiveScheduler, GpuSlot

    class _FakeRef:
        """Play an ObjectRef: run_task.remote returns success immediately, ray.wait reclaims immediately."""

        def remote(self, payload):
            self.result = {
                "episode_id": payload["episode_id"],
                "kind": payload["kind"],
                "ok": True,
                "error": None,
                "seconds": 0.01,
                "resolution_key": "64x64",
                "was_load": False,
            }
            return self

    class _FakeActor:
        def __init__(self):
            self.run_task = _FakeRef()

    ws_root = tmp_path / "ws"
    req = ReconstructionRequest.from_dict({
        "input": {"uri": str(tmp_path / "in")},
        "output": {"uri": str(tmp_path / "out"), "format": "lerobot_v3"},
        "execution": {"backend": "test", "max_episodes_in_flight": 2,
                      "max_gpus": 2, "num_cpu_actors": 2},
        "stages": {"scene_reconstruction": {"model": "auto"},
                   "hand_reconstruction": {"model": "auto"}},
    })

    d = PipelineDriver.__new__(PipelineDriver)
    d.config = DriverConfig(
        storage_root=str(tmp_path / "in"), workspace_root=str(ws_root),
        output_root=str(tmp_path / "out"), serial_baseline=False, adaptive=True,
    )
    d.request = req
    d.cost_model = CostModel()
    d.specs = {}
    d.records = {}
    d.timeline = []
    d._t0 = 0.0
    d.slots = [GpuSlot(0)]
    d.gpu_actors = [_FakeActor()]
    d.cpu_actors = [_FakeActor()]
    d._actor_node = {0: "A"}
    d._cpu_by_node = {"A": [d.cpu_actors[0]]}
    d._cpu_busy = set()
    d._cpu_stall = 0
    d.scheduler = AdaptiveScheduler(d.slots, d.cost_model)

    d.node_registry = NodePoolRegistry(
        live_nodes=[NodeInfo("A", 1, "host-a")],
        cpu_actors_per_node=1,
        create_pool=lambda nid: (_ for _ in ()).throw(AssertionError("no pool should be created")),
    )
    pool_a = NodePool(node_id="A", gpu_actors=[d.gpu_actors[0]], cpu_actors=d.cpu_actors)
    d.node_registry.register(pool_a)

    d.specs["ep_0001"] = EpisodeSpec(episode_id="ep_0001", rel_dir="ep_0001",
                                     has_stereo=False)
    rec = EpisodeRecord.new("ep_0001")  # assigned_node stays None
    d.records["ep_0001"] = rec
    EpisodeWorkspace(ws_root, "ep_0001").save_record(rec)

    monkeypatch.setattr(ray_mod, "wait", lambda refs, **kw: (list(refs), []))
    monkeypatch.setattr(d, "_get_result", lambda ref: ref.result)

    result = d._run_pipeline()
    assert result.failed == []
    # critical assertion: after the prepare dispatch (pipeline finished), assigned_node is written and stable
    assert d.records["ep_0001"].assigned_node == "A"
    assert d.records["ep_0001"].is_terminal() is True


def test_create_pool_with_empty_node_id_does_not_crash_and_registers_cpu_actors(monkeypatch):
    """Regression: all pools full -> _create_pool("") takes the "new node" branch.

    Before the fix this branch did not import NodeAffinitySchedulingStrategy and raised
    UnboundLocalError; tested directly with fake actors + fake ray.get, no real Ray cluster.
    """
    import ray as ray_mod

    from roboray.api.config import ReconstructionRequest
    from roboray.ray_runtime import actors as actors_mod
    from roboray.runtime.driver import PipelineDriver

    node_id = "a" * 56  # Ray node ids are 56-bit hex (validated by the NodeAffinity strategy)
    created_cpu = []

    class _Ref:
        def remote(self):
            return self

    class _Gpu:
        def __init__(self):
            self.node_id = _Ref()

        @classmethod
        def options(cls, **kwargs):
            return cls

        @classmethod
        def remote(cls):
            return cls()

    class _Cpu:
        @classmethod
        def options(cls, **kwargs):
            return cls

        @classmethod
        def remote(cls):
            created_cpu.append(object())
            return created_cpu[-1]

    req = ReconstructionRequest.from_dict({
        "input": {"uri": "/in"},
        "output": {"uri": "/out", "format": "lerobot_v3"},
        "execution": {"cpu_actors_per_node": 3},
    })
    d = _driver_shell()
    d.request = req
    monkeypatch.setattr(actors_mod, "GpuModelActor", _Gpu)
    monkeypatch.setattr(actors_mod, "CpuStageActor", _Cpu)
    monkeypatch.setattr(ray_mod, "get", lambda refs, **kw: node_id)

    pool = d._create_pool("")

    assert isinstance(pool, NodePool)
    assert pool.node_id == node_id
    assert len(created_cpu) == req.execution.cpu_actors_per_node == 3
    assert d.gpu_actors == pool.gpu_actors
    assert d.cpu_actors == created_cpu
    assert d._cpu_by_node[node_id] == created_cpu
    assert d._actor_node == {0: node_id}
    assert len(d.slots) == 1


def test_node_pool_multi_gpu_slots():
    from roboray.ray_runtime.nodes import NodePool

    a1, a2, cpu = object(), object(), object()
    pool = NodePool(node_id="n1", gpu_actors=[a1], cpu_actors=[cpu])
    assert pool.has_free_gpu()
    pool.take_gpu()
    pool.add_gpu(a2)
    assert pool.has_free_gpu()  # 2 slots, only 1 occupied
    pool.take_gpu()
    assert not pool.has_free_gpu()  # 2 slots, 2 occupied
    pool.release_gpu()
    pool.release_gpu()
    pool.release_gpu()  # never goes below zero
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
    reg.register(extra)  # second pool on the same node: merge, not overwrite
    merged = reg.pools["n1"]
    assert merged is p1
    assert len(merged.gpu_actors) == 2 and len(merged.cpu_actors) == 2


# =====================================================================
# multi-GPU slots + _create_pool("") landing back on an existing node -> append/merge
# =====================================================================


def test_multi_gpu_slots_dispatchable():
    """2 actors on the same node must each own an independent slot: the dispatch view zip(slots, gpu_actors) is lossless."""
    from roboray.scheduling.scheduler import GpuSlot

    slots = [GpuSlot(i) for i in range(2)]
    gpu_actors = [object(), object()]
    actor_node = {0: "A", 1: "A"}
    # simulate dispatch: both slots can be set busy/reclaimed independently
    slots[0].busy = True
    free = [s for s in slots if not s.busy]
    assert len(free) == 1 and free[0].slot_id == 1
    assert set(actor_node) == {0, 1}  # indices contiguous and aligned


def test_create_pool_empty_lands_on_existing_node_appends():
    """Scenario: an affinity-less creation lands back on an existing node's pool -> append and return the existing pool, no overwrite."""
    from roboray.ray_runtime.nodes import NodeInfo, NodePool, NodePoolRegistry

    existing = NodePool("A", gpu_actors=[object()], cpu_actors=[object()])
    reg = NodePoolRegistry(
        live_nodes=[NodeInfo("A", 2, "h")], cpu_actors_per_node=1,
        create_pool=lambda nid: existing,
    )
    reg.register(existing)
    # driver._create_pool("") actual==A branch semantics: returns existing and the actor is in the pool
    new_actor = object()
    existing.add_gpu(new_actor)
    assert reg.pools["A"] is existing and len(existing.gpu_actors) == 2


def test_create_pool_on_existing_node_merges(monkeypatch):
    """_create_pool("") lands on a node that already has a pool: the new actor is appended to the existing pool and returned.

    Reuses the _Gpu/_Cpu/ray.get stub technique from
    test_create_pool_with_empty_node_id_does_not_crash_and_registers_cpu_actors; the only difference
    is that the new actor's node_id() returns the existing node A.
    """
    import ray as ray_mod

    from roboray.api.config import ReconstructionRequest
    from roboray.ray_runtime import actors as actors_mod
    from roboray.ray_runtime.nodes import NodeInfo, NodePool, NodePoolRegistry
    from roboray.runtime.driver import PipelineDriver

    class _Ref:
        def remote(self):
            return self

    class _Gpu:
        def __init__(self):
            self.node_id = _Ref()

        @classmethod
        def options(cls, **kwargs):
            return cls

        @classmethod
        def remote(cls):
            return cls()

    class _Cpu:
        @classmethod
        def options(cls, **kwargs):
            return cls

        @classmethod
        def remote(cls):
            return object()

    node_id = "a" * 56  # Ray node ids are 56-bit hex (validated by the NodeAffinity strategy)
    req = ReconstructionRequest.from_dict({
        "input": {"uri": "/in"},
        "output": {"uri": "/out", "format": "lerobot_v3"},
        "execution": {"cpu_actors_per_node": 2},
    })
    d = _driver_shell()
    d.request = req
    monkeypatch.setattr(actors_mod, "GpuModelActor", _Gpu)
    monkeypatch.setattr(actors_mod, "CpuStageActor", _Cpu)
    monkeypatch.setattr(ray_mod, "get", lambda refs, **kw: node_id)

    # an existing node-A pool (1 GPU actor) is already registered
    existing_actor = object()
    d.node_registry = NodePoolRegistry(
        live_nodes=[NodeInfo(node_id, 2, "host-a")],
        cpu_actors_per_node=req.execution.cpu_actors_per_node,
        create_pool=d._create_pool,
    )
    existing_pool = NodePool(node_id=node_id, gpu_actors=[existing_actor], cpu_actors=[])
    d.node_registry.register(existing_pool)

    pool = d._create_pool("")

    # the returned pool is the existing pool; the new actor is appended; the registry still has only 1 A entry
    assert pool is existing_pool
    assert len(existing_pool.gpu_actors) == 2
    assert existing_pool.gpu_actors[1] is d.gpu_actors[-1]
    assert len(d.node_registry.pools) == 1
    assert d.node_registry.pools[node_id] is existing_pool
