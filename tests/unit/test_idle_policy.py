"""Three-level idle policy pure function: compute_idle_actions tested directly with a fake clock (no Ray dependency).

Policy and mechanism are separated: idle detection is a pure function, and the
threshold semantics (>=, exit before unload, busy/unloaded never act, 0=disabled)
are tested directly against a fake IdleState, without real actors or a cluster.
"""
from roboray.scheduling.idle import (
    EXIT_DEFAULT,
    UNLOAD_DEFAULT,
    IdleState,
    compute_idle_actions,
)


def _state(idle, loaded=True, busy=False, actor="a", slot_id=0):
    """Build a fake slot + IdleState. The slot only exposes busy (busy is decided solely by slot.busy)."""

    class _Slot:
        def __init__(self, busy, slot_id):
            self.busy = busy
            self.slot_id = slot_id

    return IdleState(
        actor=actor,
        slot=_Slot(busy, slot_id),
        idle_seconds=idle,
        loaded=loaded,
    )


def test_empty_states_no_actions():
    assert compute_idle_actions([], now=0.0, unload_s=30.0, exit_s=240.0) == []


def test_below_unload_threshold_no_action():
    st = _state(idle=29.0)
    assert compute_idle_actions([st], now=0.0, unload_s=30.0, exit_s=240.0) == []


def test_at_unload_threshold_emits_unload():
    st = _state(idle=30.0)
    assert compute_idle_actions([st], now=0.0, unload_s=30.0, exit_s=240.0) == [
        ("unload", st)
    ]


def test_at_exit_threshold_emits_exit_not_unload():
    st = _state(idle=240.0)
    actions = compute_idle_actions([st], now=0.0, unload_s=30.0, exit_s=240.0)
    # exit is checked before unload and implies unload first: one state yields one action, no duplicate unload
    assert [a for a, _ in actions] == ["exit"]


def test_busy_slot_never_emits_action():
    st = _state(idle=10_000.0, busy=True)
    assert compute_idle_actions([st], now=0.0, unload_s=30.0, exit_s=240.0) == []


def test_unloaded_actor_emits_exit_but_not_unload():
    """An unloaded actor **does not unload** (nothing to unload), but **must exit at the
    exit threshold** - it also times out idle and wastes a slot/node; when 240s arrive
    it should exit, otherwise the node never returns to the cluster.
    """
    st = _state(idle=10_000.0, loaded=False)
    actions = compute_idle_actions([st], now=0.0, unload_s=30.0, exit_s=240.0)
    assert actions == [("exit", st)]


def test_unloaded_actor_below_exit_no_action():
    """An unloaded actor only acts at the exit threshold: between 30s and 240s there is neither unload nor exit."""
    st = _state(idle=100.0, loaded=False)
    assert compute_idle_actions([st], now=0.0, unload_s=30.0, exit_s=240.0) == []


def test_unload_s_zero_disables_unload():
    # unload_s=0 = never unload proactively (only role switch / teardown), not "unload immediately"
    st = _state(idle=30.0)
    assert compute_idle_actions([st], now=0.0, unload_s=0.0, exit_s=240.0) == []


def test_exit_s_zero_disables_exit():
    # exit_s=0 = never exit proactively; the unload layer still applies
    st = _state(idle=240.0)
    actions = compute_idle_actions([st], now=0.0, unload_s=30.0, exit_s=0.0)
    assert [a for a, _ in actions] == ["unload"]


def test_default_thresholds_are_increasing():
    # must increase with 240 < 300 (node reclamation), otherwise nodes never see an actor exit
    assert UNLOAD_DEFAULT < EXIT_DEFAULT


# =====================================================================
# driver._apply_idle_actions exit branch (previously untested - the exit branch is where the
# critical behavior hides)
# =====================================================================


def _exit_driver(actor, created):
    """Build a skip-__init__ driver shell with only the fields the exit branch needs."""
    from roboray.api.config import ReconstructionRequest
    from roboray.ray_runtime.nodes import NodeInfo, NodePool, NodePoolRegistry
    from roboray.runtime.driver import DriverConfig, PipelineDriver
    from roboray.scheduling.scheduler import GpuSlot

    d = PipelineDriver.__new__(PipelineDriver)
    d.config = DriverConfig(storage_root="/s", workspace_root="/w", output_root="/o")
    d.request = ReconstructionRequest.from_dict({
        "input": {"uri": "/in"},
        "output": {"uri": "/out", "format": "lerobot_v3"},
        "execution": {"backend": "test"},
        "stages": {"scene_reconstruction": {"model": "auto"},
                   "hand_reconstruction": {"model": "auto"}},
    })
    d.slots = [GpuSlot(0)]
    d.gpu_actors = [actor]
    d._actor_node = {0: "A"}
    d.timeline = []
    d._t0 = 0.0

    def _create_pool(node_id):
        created.append(node_id)
        return NodePool(node_id=node_id, gpu_actors=[actor], cpu_actors=[])

    d.node_registry = NodePoolRegistry(
        live_nodes=[NodeInfo("A", 1, "host-a")],
        cpu_actors_per_node=1,
        create_pool=_create_pool,
    )
    d.node_registry.register(NodePool(node_id="A", gpu_actors=[actor], cpu_actors=[]))
    return d


def test_exit_branch_removes_actor_keeps_multi_gpu_pool(monkeypatch):
    """Removal semantics (two actors in one node pool): exit only removes **this actor**; the pool is popped only when empty.

    Mutation guard: if the implementation regressed to "exit pops the whole pool", this test must
    fail - after one actor on a multi-GPU node exits, the pool for the **other live actors** on
    the same node must survive (if the pool vanished, the remaining slots on that node could
    never match and would stall). First actor exit -> pool survives, `gpu_actors==1`
    (the live actor is not lost); second actor exit -> the pool is popped.
    """
    import ray as ray_mod

    from roboray.api.config import ReconstructionRequest
    from roboray.ray_runtime.nodes import NodeInfo, NodePool, NodePoolRegistry
    from roboray.runtime.driver import DriverConfig, PipelineDriver
    from roboray.scheduling.scheduler import GpuSlot

    calls: list[str] = []

    class _Unload:
        def remote(self):
            calls.append("unload")
            return "unload-ref"

    class _FakeActor:
        def __init__(self):
            self.unload = _Unload()

    actor1, actor2 = _FakeActor(), _FakeActor()

    d = PipelineDriver.__new__(PipelineDriver)
    d.config = DriverConfig(storage_root="/s", workspace_root="/w", output_root="/o")
    d.request = ReconstructionRequest.from_dict({
        "input": {"uri": "/in"},
        "output": {"uri": "/out", "format": "lerobot_v3"},
        "execution": {"backend": "test"},
        "stages": {"scene_reconstruction": {"model": "auto"},
                   "hand_reconstruction": {"model": "auto"}},
    })
    d.slots = [GpuSlot(0), GpuSlot(1)]
    d.gpu_actors = [actor1, actor2]
    d._actor_node = {0: "A", 1: "A"}
    d.timeline = []
    d._t0 = 0.0
    d.node_registry = NodePoolRegistry(
        live_nodes=[NodeInfo("A", 2, "host-a")],
        cpu_actors_per_node=1,
        create_pool=lambda nid: NodePool(node_id=nid, gpu_actors=[], cpu_actors=[]),
    )
    d.node_registry.register(
        NodePool(node_id="A", gpu_actors=[actor1, actor2], cpu_actors=[]))

    monkeypatch.setattr(ray_mod, "get", lambda ref, **kw: {})
    monkeypatch.setattr(ray_mod, "kill", lambda a, **kw: calls.append("kill"))

    # first actor exits: remove actor1, pool survives with only actor2 (the live actor is not lost)
    d._apply_idle_actions([("exit", IdleState(
        actor=actor1, slot=d.slots[0], idle_seconds=240.0, loaded=True))])
    assert "A" in d.node_registry.pools
    assert d.node_registry.pools["A"].gpu_actors == [actor2]
    assert d.slots[0].busy is True

    # second actor exits: pool empty -> whole pool popped (the keep-_live self-healing semantics are unchanged)
    d._apply_idle_actions([("exit", IdleState(
        actor=actor2, slot=d.slots[1], idle_seconds=240.0, loaded=True))])
    assert "A" not in d.node_registry.pools
    assert "A" in d.node_registry._live
    assert calls.count("kill") == 2


def test_exit_branch_unloads_before_kill_busy_and_pops_pool_keeps_live(monkeypatch):
    """Exit-branch regression:

    (a) **unload before ray.kill** (ray.kill is SIGKILL and skips cleanup, so the unload
        must come first to close the resident session);
    (b) the killed slot becomes **permanently busy=True** (the dead slot is never dispatched);
    (c) only the pool is removed from the registry (**pools.pop**), **_live is kept** - an
        episode assigned to that node self-heals on the next pool_for_episode via
        `node_id in _live -> _create_pool` affinity rebuild. Misusing drop() (removing _live
        too) would leave that episode unable to match a slot, and scene_q/hand_q would
        spin forever.
    """
    import ray as ray_mod

    calls: list[str] = []

    class _Unload:
        def remote(self):
            calls.append("unload")
            return "unload-ref"

    class _FakeActor:
        def __init__(self):
            self.unload = _Unload()

    actor = _FakeActor()
    created: list[str] = []
    d = _exit_driver(actor, created)

    monkeypatch.setattr(ray_mod, "get", lambda ref, **kw: {})
    monkeypatch.setattr(ray_mod, "kill", lambda a, **kw: calls.append("kill"))

    st = IdleState(actor=actor, slot=d.slots[0], idle_seconds=240.0, loaded=True)
    d._apply_idle_actions([("exit", st)])

    # (a) call order: unload before kill
    assert calls == ["unload", "kill"]
    # (b) slot permanently busy
    assert d.slots[0].busy is True
    # (c) only the pool removed, _live kept
    assert "A" not in d.node_registry.pools
    assert "A" in d.node_registry._live
    assert "A" in d.node_registry.live_node_ids()
    # an assigned episode's next placement triggers the affinity rebuild (_create_pool is called)
    pool = d.node_registry.pool_for_episode("ep_0001", {"ep_0001": "A"})
    assert created == ["A"]
    assert pool.node_id == "A"
    assert d.node_registry.pools["A"] is pool
