"""Driver main loop: dispatch planner tasks to Ray actors and feed back scheduling signals.

Three modes (for side-by-side comparison):
- serial_baseline=True: run each episode stage-by-stage in series (baseline)
- serial_baseline=False, adaptive=False: cross-stage pipeline + static pool split
- serial_baseline=False, adaptive=True: cross-stage pipeline + adaptive scheduling (target form)
"""

import time
from dataclasses import dataclass, field
from pathlib import Path

import ray

from roboray.api.config import ReconstructionRequest
from roboray.ray_runtime.nodes import NodePool, NodePoolRegistry, discover_gpu_nodes
from roboray.scheduling.cost_model import DEFAULT_PRIOR_SECONDS, CostModel
from roboray.scheduling.idle import IdleState, compute_idle_actions
from roboray.scheduling.scheduler import AdaptiveScheduler, GpuSlot, QueueItem
from roboray.stages.scene import resolve_scene_model

from .episode import (
    EpisodeRecord,
    EpisodeSpec,
    EpisodeWorkspace,
    Stage,
    StageStatus,
    discover_episodes,
)
from .planner import DEFAULT_RESOLUTION_KEY, build_cpu_kwargs, episode_ready_tasks
from .storage import make_storage

def pick_idle(group: list, busy_ids: set) -> object | None:
    """Return the first non-busy actor in a CPU group; None if all are busy (caller retries next round).

    Pure function with no Ray dependency - same decoupled design as the
    NodePoolRegistry placement rule, unit-testable directly. Busy accounting uses
    ``id(actor)`` (``_cpu_busy``) to avoid any remote calls on actors.
    """
    for actor in group:
        if id(actor) not in busy_ids:
            return actor
    return None


__all__ = ["DriverConfig", "RunResult", "PipelineDriver", "pick_idle"]


@dataclass
class DriverConfig:
    storage_root: str
    workspace_root: str
    output_root: str
    serial_baseline: bool = False
    adaptive: bool = True


@dataclass
class RunResult:
    total_seconds: float
    per_episode: dict = field(default_factory=dict)
    scheduler_snapshots: list = field(default_factory=list)
    timeline: list = field(default_factory=list)
    failed: list = field(default_factory=list)
    manifest_path: str | None = None


class PipelineDriver:
    def __init__(
        self,
        config: DriverConfig,
        request: ReconstructionRequest,
        gpu_actors: list,
        cpu_actors: list,
    ):
        self.config = config
        self.request = request
        self.gpu_actors = gpu_actors
        self.cpu_actors = cpu_actors
        # CPU actors grouped by node_id. All CPU stages
        # (prepare/fusion/action/publish) and GPU stages must land on the same
        # node - the workspace is node-local. When there are no GPU nodes
        # (registry None) every CPU actor has a real node_id (same node = same
        # group) and `_cpu_actor_for` degrades to global-idle selection (all e2e
        # tests exercise this path).
        self._cpu_by_node: dict[str, list] = self._group_cpu_actors(cpu_actors)
        self._cpu_busy: set[int] = set()      # id(actor) accounting: set on dispatch, cleared on reclaim
        self._cpu_stall: int = 0               # consecutive no-progress rounds (guaranteed-convergence guard)
        self.storage = make_storage(config.storage_root)
        self.specs: dict[str, EpisodeSpec] = {}
        # The authoritative in-memory record for each episode. Hard invariant:
        # the driver process never reads workspace files - state.json is only
        # cross-run persistence for resumable runs; the driver loads it into
        # memory once at run start, and all state advancement during the run
        # happens in memory, never re-reading the disk.
        self.records: dict[str, EpisodeRecord] = {}
        self.cost_model = CostModel()
        self.slots = [GpuSlot(i) for i in range(len(gpu_actors))]
        # Episode -> node mapping. Affinity is enforced by actor placement (see
        # build_node_pools); the driver only routes each episode to the actor at
        # the matching index.
        self._actor_node: dict[int, str] = {}
        live_nodes = discover_gpu_nodes()
        if live_nodes:
            self.node_registry = NodePoolRegistry(
                live_nodes=live_nodes,
                cpu_actors_per_node=self.request.execution.cpu_actors_per_node,
                create_pool=self._create_pool,
            )
            self._register_existing_pools(gpu_actors, cpu_actors)
        else:
            # No GPU nodes (local test machine / CPU-only cluster): disable node
            # routing. NodePoolRegistry returns None when there are no live nodes
            # and no pools, which would hang every e2e test; fall back to the
            # original flat "use the first actor" behavior (see _pool_index_for).
            self.node_registry = None
        self.scheduler = AdaptiveScheduler(self.slots, self.cost_model)
        self.timeline: list = []
        self._t0 = 0.0

    def _group_cpu_actors(self, cpu_actors: list) -> dict[str, list]:
        """Ask each CPU actor for its node_id and group by node.

        ``node_id()`` is a plain method on CpuStageActor: it still returns the
        local node id when created without affinity (single node = single
        group). An empty list returns ``{}`` (unit tests that build the driver
        directly without real actors skip the remote calls).
        """
        if not cpu_actors:
            return {}
        node_ids = ray.get([a.node_id.remote() for a in cpu_actors])
        by_node: dict[str, list] = {}
        for actor, nid in zip(cpu_actors, node_ids):
            by_node.setdefault(nid, []).append(actor)
        return by_node

    def _register_existing_pools(self, gpu_actors: list, cpu_actors: list) -> None:
        """Register the actors built at startup as node pools.

        Fixes a shared-reference bug: previously every pool received the same
        ``list(cpu_actors)`` (one shared list of all CPU actors), so under
        multi-node fusion could be dispatched to a CPU actor on a different node
        that cannot see the node-local prepared/. Now each pool gets its own CPU
        actor list from ``_cpu_by_node`` (grouped by actual node_id in
        ``__init__``).
        """
        node_ids = ray.get([a.node_id.remote() for a in gpu_actors]) if gpu_actors else []
        by_node: dict[str, list] = {}
        for idx, node_id in enumerate(node_ids):
            self._actor_node[idx] = node_id
            by_node.setdefault(node_id, []).append(gpu_actors[idx])
        for node_id, actors in by_node.items():
            self.node_registry.register(
                NodePool(
                    node_id=node_id,
                    gpu_actors=actors,
                    cpu_actors=list(self._cpu_by_node.get(node_id, [])),
                )
            )

    def _create_pool(self, node_id: str) -> NodePool:
        """Create a new node pool (called by the registry when all existing pools are full).

        An empty ``node_id`` means "a new node is needed": the GPU actor is
        created **without affinity**, so its placement is decided by Ray + the
        autoscaler; once created, we ask it for its own node_id and create the
        CPU actors accordingly. If that placement already has a pool (an
        affinity-less creation landed back on an existing node), the new actor
        is appended to the existing pool and that pool is returned - no new
        pool, no overwrite, and no duplicate CPU actors.
        """
        from roboray.ray_runtime.actors import CpuStageActor, GpuModelActor
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

        if node_id:
            strategy = NodeAffinitySchedulingStrategy(node_id=node_id, soft=False)
            gpu = GpuModelActor.options(scheduling_strategy=strategy).remote()
        else:
            gpu = GpuModelActor.remote()
        actual = ray.get(gpu.node_id.remote())

        # Landing back on an existing node -> append to the existing pool, no
        # new pool and no overwrite. The existing pool's in_flight>0 does not
        # affect correctness: take/release follow the caller's cadence, this
        # only adds capacity to the pool (has_free_gpu uses the actor count as
        # capacity).
        reg = getattr(self, "node_registry", None)
        existing = reg.pools.get(actual) if reg is not None else None
        if existing is not None:
            existing.add_gpu(gpu)
            self._register_actor(gpu, actual)
            return existing

        strategy = NodeAffinitySchedulingStrategy(node_id=actual, soft=False)
        cpus = [
            CpuStageActor.options(scheduling_strategy=strategy).remote()
            for _ in range(self.request.execution.cpu_actors_per_node)
        ]
        self.cpu_actors.extend(cpus)
        self._cpu_by_node.setdefault(actual, []).extend(cpus)  # new pool CPU actors join their group too
        self._register_actor(gpu, actual)
        return NodePool(node_id=actual, gpu_actors=[gpu], cpu_actors=cpus)

    def _register_actor(self, gpu, node_id: str) -> None:
        """Register a newly created GPU actor in the driver's bookkeeping (gpu_actors/slots/_actor_node).

        Shared by both branches: a brand-new pool and "landed back on an
        existing node -> appended to the existing pool". slot_id is the
        gpu_actors index (the _pool_index_for contract: append-only, never
        remove, since removal would shift indices).
        """
        idx = len(self.gpu_actors)
        self.gpu_actors.append(gpu)
        self._actor_node[idx] = node_id
        self.slots.append(GpuSlot(idx))

    def _pool_index_for(self, episode_id: str) -> int | None:
        """Which GPU actor this episode should use; ``None`` = no usable node.

        Once chosen, ``assigned_node`` is written into the record so resumable
        runs come back to the same node, and so node loss can be detected.
        """
        if self.node_registry is None:
            # No GPU node environment (local test machine / CPU-only cluster):
            # node routing is disabled, fall back to the original flat "use the
            # first GPU actor" behavior.
            return 0 if self.gpu_actors else None
        rec = self.records[episode_id]  # authoritative in-memory record, never re-read from disk
        assigned = {episode_id: rec.assigned_node} if rec.assigned_node else {}
        pool = self.node_registry.pool_for_episode(episode_id, assigned)
        if pool is None:
            return None
        if rec.assigned_node != pool.node_id:
            rec.assigned_node = pool.node_id
            self._ws(episode_id).save_record(rec)
        for idx, node_id in self._actor_node.items():
            if node_id == pool.node_id and any(
                self.gpu_actors[idx] is a for a in pool.gpu_actors
            ):
                return idx
        return None

    def _ensure_assigned(self, episode_id: str) -> bool:
        """Ensure this episode has an ``assigned_node`` chosen and persisted; return whether it is dispatchable.

        Critical fix: the only writer of ``assigned_node`` used to be
        ``_pool_index_for`` (at scene/hand GPU dispatch time). But prepare is
        the first stage, and at its dispatch ``assigned_node`` is always None,
        so ``_cpu_actor_for`` took the global fallback and landed on an
        arbitrary node X; a later scene picked Y via ``pool_for_episode``, and
        the GPU actor on Y could not read the prepared/ sitting on X - the
        multi-node loop broke at the first stage. Choose and persist the node
        *before* the CPU (prepare) dispatch: ``pool_for_episode`` forces already
        assigned episodes back to their original node, so once the first choice
        is stable every later scene/hand returns to the same node and the loop
        closes.

        - No GPU nodes (``node_registry is None``): nothing to assign, always
          returns True (prepare uses the global fallback; single-node e2e
          behavior is unchanged).
        - Returns False = no usable node (``_pool_index_for`` also returns
          None): the caller skips this task this round and retries next round,
          same semantics as a busy GPU node.
        """
        if self.node_registry is None:
            return True
        # _pool_index_for already encapsulates "choose + persist assigned_node":
        # for an unassigned episode it calls pool_for_episode and writes the
        # record; for an assigned one it forces a return to the original node.
        return self._pool_index_for(episode_id) is not None

    def _cpu_actor_for(self, episode_id: str) -> object | None:
        """Pick an actor for this episode's CPU stages; ``None`` = nothing dispatchable this round.

        Routing rule: an assigned episode only picks from the ``assigned_node``
        group; if the group is fully busy -> return None (caller skips this
        round and retries next), and **never borrow from another node** - the
        workspace is node-local, so a CPU actor on another node cannot read the
        node-local prepared/. Unassigned (should not happen anymore, since
        ``_ensure_assigned`` runs before dispatch; this is purely defensive) and
        no-GPU-node (registry None) degrade to any globally idle CPU actor.
        """
        if self.node_registry is None:
            return pick_idle(self.cpu_actors, self._cpu_busy)
        rec = self.records.get(episode_id)  # authoritative in-memory record, never re-read from disk
        node = rec.assigned_node if rec is not None else None
        if not node:
            return pick_idle(self.cpu_actors, self._cpu_busy)
        return pick_idle(self._cpu_by_node.get(node, []), self._cpu_busy)

    def _cpu_stall_limit(self) -> int:
        """Guaranteed-convergence threshold: map node_wait_timeout_s to a number of no-progress rounds.

        The threshold semantics were remapped from "seconds" to "rounds with no
        progress" (default 600s, main loop ray.wait ticks every 0.05s -> 600/0.05
        = 12000 rounds). After the remap, the config field name is unchanged and
        only the semantic comment is updated.
        """
        secs = float(self.request.execution.node_wait_timeout_s)
        return max(1, int(round(secs / 0.05)))

    def _node_eligible_queues(self, slot, scene_q: list, hand_q: list):
        """Return the subset of items dispatchable from this slot's node (references unchanged, view slices).

        All stages of an episode must land on the same node (the workspace is
        node-local), so only tasks whose episode's node equals the slot's node
        may be dispatched to this slot. The owning node is decided by
        ``_pool_index_for`` - a side effect is that it chooses and persists
        ``assigned_node`` for unassigned episodes; assigned ones just read the
        persisted value (back to the original node).
        """
        if self.node_registry is None:
            return list(scene_q), list(hand_q)

        def match(item) -> bool:
            idx = self._pool_index_for(item.episode_id)
            return idx is not None and self._actor_node.get(slot.slot_id) == self._actor_node.get(idx)

        return (
            [it for it in scene_q if match(it)],
            [it for it in hand_q if match(it)],
        )

    # ---------- internal helpers ----------
    def _ws(self, episode_id: str) -> EpisodeWorkspace:
        return EpisodeWorkspace(Path(self.config.workspace_root), episode_id)

    def _event(self, episode_id: str, stage: Stage, event: str) -> None:
        self.timeline.append(
            {
                "t": round(time.perf_counter() - self._t0, 6),
                "episode_id": episode_id,
                "stage": stage.value,
                "event": event,
            }
        )

    # ---------- three-level idle (spec: mechanism in the actor, policy in the driver) ----------
    def _idle_timeline_event(self, event: str) -> None:
        """Timeline event for unload/exit: no episode owner, so append the dict directly.

        Does not reuse ``_event`` - that one forces episode_id/stage. Idle
        actions are operational actions the driver takes on actors and are
        unrelated to any episode. The t field format matches ``_event``.
        """
        self.timeline.append(
            {
                "t": round(time.perf_counter() - self._t0, 6),
                "episode_id": None,
                "stage": event,
                "event": event,
            }
        )

    def _scan_idle(self) -> list[tuple[str, IdleState]]:
        """Scan once per main loop iteration: snapshot idle slots and hand them to the pure function for actions.

        The scan only touches ``not slot.busy`` slots - slot.busy is the single
        authority for busy (the fixed semantics): an actor running a task must
        never be unloaded/killed (hard red line). idle_seconds is a **remote
        call**; exceptions/death are treated as 0 and never raised - dead actors
        are converged by the exit branch's ray.kill, and timeout=2 bounds it.
        """
        exec_cfg = self.request.execution
        states: list[IdleState] = []
        for slot, actor in zip(self.slots, self.gpu_actors):
            if slot.busy:
                continue
            try:
                idle = float(ray.get(actor.idle_seconds.remote(), timeout=2))
            except Exception:
                idle = 0.0  # dead/timed out: not determinable this round; the exit threshold's kill converges it
            states.append(
                IdleState(
                    actor=actor,
                    slot=slot,
                    idle_seconds=idle,
                    loaded=self._actor_loaded(actor),
                )
            )
        return compute_idle_actions(
            states,
            now=time.perf_counter() - self._t0,
            unload_s=float(exec_cfg.actor_idle_unload_s),
            exit_s=float(exec_cfg.actor_idle_exit_s),
        )

    def _actor_loaded(self, actor) -> bool:
        """Whether this actor has a model loaded (actor.loaded_kind is not None).

        A dead actor's status call raises/times out, so it is treated as
        unloaded: no model to unload, the idle scan only produces exit actions
        for dead actors.
        """
        try:
            return bool(ray.get(actor.status.remote(), timeout=2).get("loaded_kind"))
        except Exception:
            return False

    def _apply_idle_actions(self, actions: list[tuple[str, IdleState]]) -> None:
        """Apply idle actions: unload frees the model; exit unloads then ray.kills, and the slot becomes permanently busy.

        A killed slot is **not removed from the list** - slot_id is the
        gpu_actors index and removal would shift indices (the _pool_index_for
        contract); marking busy=True permanently keeps the dead slot from ever
        being dispatched again. On the registry side, only the pool is removed
        (``pools.pop``), never moved out of ``_live``: the pool is gone but the
        node is still alive (node-level resources are reclaimed by the
        autoscaler's idleTimeoutSeconds). Semantic distinction: ``drop()`` =
        the node is really gone (removed from _live too); ``pools.pop`` = only
        the pool is rebuilt. **drop() must not be used** - episodes assigned to
        that node self-heal through ``pool_for_episode``'s ``node_id in self._live
        -> _create_pool(node_id)`` affinity rebuild (the rebuilt GPU actor is
        pinned back to the original node where the workspace still lives); if
        _live is removed too, that episode can never match a slot and
        scene_q/hand_q spin forever.
        """
        exec_cfg = self.request.execution
        for action, st in actions:
            if action == "unload":
                try:
                    ray.get(st.actor.unload.remote(), timeout=5)
                except Exception:
                    pass  # best effort: a dead actor has no model to unload; the exit branch converges it
                self._idle_timeline_event("unload")
            elif action == "exit":
                try:
                    ray.get(st.actor.unload.remote(), timeout=5)
                except Exception:
                    pass  # best-effort graceful cleanup; kill is the fallback
                self._idle_timeline_event("unload")
                try:
                    ray.kill(st.actor)
                except Exception:
                    pass
                st.slot.busy = True  # permanently busy: the dead slot is never dispatched again
                self._idle_timeline_event("exit")
                # Removal semantics: take **this actor** out of its pool, and
                # only pop the pool when it becomes empty - after one actor on a
                # multi-GPU node exits, the pool for the other actors on the
                # same node must survive (if the pool vanished, the remaining
                # slots on that node could never match and would stall). **Keep
                # _live**: the pool is gone but the node is alive, so episodes
                # assigned to that node self-heal via pool_for_episode's
                # `node_id in _live -> _create_pool(node_id)` affinity rebuild.
                # Using drop() (removing _live too) would block that
                # self-healing path and the episode would never match a slot.
                if self.node_registry is not None:
                    for idx, actor in enumerate(self.gpu_actors):
                        if actor is st.actor:
                            node_id = self._actor_node.get(idx)
                            if node_id:
                                pool = self.node_registry.pools.get(node_id)
                                if pool is not None:
                                    pool.gpu_actors = [
                                        a for a in pool.gpu_actors if a is not st.actor
                                    ]
                                    if not pool.gpu_actors:
                                        self.node_registry.pools.pop(node_id, None)
                            break

    def _scene_model(self, episode_id: str) -> str:
        """The real model name for this episode's scene path.

        Prefers the has_stereo backfilled into the record by prepare; before
        prepare has run (cold-start prior phase) it falls back to the input-side
        guess ``spec.has_stereo`` - this only affects the initial scheduling
        prior, never correctness. **Never reads workspace files**: under
        multi-node the driver cannot read node-local workspaces.
        """
        rec = self.records[episode_id]  # authoritative in-memory record, never re-read from disk
        has_stereo = rec.has_stereo
        if has_stereo is None:
            spec = self.specs.get(episode_id)
            has_stereo = spec.has_stereo if spec is not None else False
        return resolve_scene_model(
            self.request.stages.scene_reconstruction.model, has_stereo
        )

    def _resolution_key(self, episode_id: str) -> str:
        """This episode's resolution tier; falls back to the placeholder tier until prepare finishes.

        The placeholder only affects the cold-start scheduling prior (same as
        planner.DEFAULT_RESOLUTION_KEY); at execution time the actor reports the
        real tier.
        """
        rec = self.records[episode_id]  # authoritative in-memory record, never re-read from disk
        return rec.resolution_key or DEFAULT_RESOLUTION_KEY

    def _seed_priors(self) -> None:
        """Seed cold-start priors from each model's self-reported cost_hint so the scheduler is not fully blind.

        Skipped for unregistered models (e.g. missing GPU dependency): a missing
        prior only degrades early decision quality, never correctness - CostModel
        falls back to DEFAULT_PRIOR_SECONDS.
        """
        from roboray.models import registry

        sc = self.request.stages.scene_reconstruction
        hc = self.request.stages.hand_reconstruction
        # Seed the prior only once, using the first episode's prepared calib to
        # decide the scene-path model name - same source as the push_ready /
        # _gpu_payload decision (both: prepared calib -> model name).
        first = next(iter(self.specs.values()), None)
        scene_model = self._scene_model(first.episode_id) if first else sc.model
        for res in ("320x240", "640x480"):
            for kind, model in (("scene", scene_model), ("hand", hc.model)):
                if self.cost_model.predict(model, res) != DEFAULT_PRIOR_SECONDS:
                    continue  # prior already present
                try:
                    factory = (
                        registry.get_scene_model_class(model)
                        if kind == "scene"
                        else registry.get_hand_model_class(model)
                    )
                    m = factory()
                except Exception:
                    continue
                self.cost_model.seed_prior(model, res, m.cost_hint(24, res))

    # ---------- three modes ----------
    def run(self) -> RunResult:
        self._t0 = time.perf_counter()
        specs = discover_episodes(self.storage)
        self.specs = {s.episode_id: s for s in specs}
        # Load state.json into memory once at run start (resumable runs).
        # Afterwards any state advancement during the run only touches memory
        # plus writes to disk, **never re-reads** - hard invariant: the driver
        # process does not read workspace files.
        self.records = {ep: self._ws(ep).load() for ep in self.specs}
        self._reconcile_states()
        # Teardown unload lives in try/finally: even an unexpected exception in
        # the main loop must return VRAM. Otherwise the driver process exiting
        # -> Ray kills the actor without running __ray_terminate__ (SIGKILL) ->
        # resident child processes are orphaned while holding VRAM.
        try:
            if self.config.serial_baseline:
                result = self._run_serial()
            else:
                result = self._run_pipeline()
        finally:
            self._teardown_unload_all()
        result.total_seconds = time.perf_counter() - self._t0
        return result

    def _teardown_unload_all(self) -> None:
        """run teardown: best-effort unload() on every GPU actor - swallow exceptions, teardown must not raise.

        If run ends without unload, the Ray actor stays alive and the resident
        child process hangs around holding VRAM. A dead actor's unload call
        raises; treat it as "no model to unload" and skip silently.
        """
        for actor in self.gpu_actors:
            try:
                ray.get(actor.unload.remote(), timeout=5)
            except Exception:
                pass

    def _reconcile_states(self) -> None:
        """Align all episode states at run start (cross-run semantics, see EpisodeRecord.reconcile_for_new_run).

        Additionally validates ``assigned_node``: the workspace is node-local, so
        if the node is gone the workspace is gone too and the episode must be
        re-run in full (not just reset RUNNING).

        With no GPU nodes (local test machine) pass ``None`` to skip node
        validation - there is nothing to validate, and since the workspace is on
        this machine there is no loss risk.
        """
        nodes = discover_gpu_nodes()
        live_nodes = {n.node_id for n in nodes} if nodes else None

        for ep in self.specs:
            ws = self._ws(ep)
            rec = self.records[ep]  # authoritative in-memory record, loaded at run start
            changes = rec.reconcile_for_new_run(live_nodes=live_nodes)
            if changes:
                ws.save_record(rec)

    def _run_serial(self) -> RunResult:
        """Baseline: run one episode through all stages before starting the next, with no parallelism."""
        per_episode, failed = {}, []
        for spec in self.specs.values():
            for stage in [Stage.PREPARE, Stage.SCENE, Stage.HAND, Stage.FUSION,
                          Stage.ACTION, Stage.PUBLISH]:
                ok = self._execute_one(spec, stage)
                if not ok:
                    failed.append(spec.episode_id)
                    break
            per_episode[spec.episode_id] = self.records[spec.episode_id]
        failed = self._collect_failed(failed)
        return RunResult(
            total_seconds=0.0,
            per_episode={k: v.to_json() for k, v in per_episode.items()},
            scheduler_snapshots=[],
            timeline=self.timeline,
            failed=failed,
        )

    def _run_pipeline(self) -> RunResult:
        """Cross-stage pipeline + (optional) adaptive scheduling."""
        self._seed_priors()
        scene_q: list = []
        hand_q: list = []
        cpu_q: list = []
        in_flight: dict = {}     # ray.ObjectRef -> (ep_id, kind, model_name, slot_or_None, actor)
        queued: set = set()      # queued but unfinished (episode_id, stage); prevents double enqueue

        def push_ready() -> None:
            """Read each episode's current state from the **authoritative in-memory record** and enqueue its ready stages.

            State advancement (``_complete`` / ``_execute_one``) mutates
            ``self.records`` in place; this method reads that same in-memory
            snapshot - **it never re-reads the workspace**. Hard invariant: the
            driver process does not read workspace files; state.json is only
            cross-run persistence, and within a run all state lives in memory.

            In static pool-split mode, max_episodes_in_flight gates admission:
            once the quota of in-progress episodes is reached no new episode is
            enqueued; adaptive mode sets no gate - the scheduler's queue-backlog
            balancing is itself the concurrency control.
            """
            busy_eps = {
                ep for ep, _kind, _model, _slot, _actor in in_flight.values()
            } | {ep for ep, _stage, _spec, _t in cpu_q} | {
                it.episode_id for it in scene_q
            } | {it.episode_id for it in hand_q}
            max_in_flight = max(1, self.request.execution.max_episodes_in_flight)
            for ep in list(self.specs):
                rec = self.records[ep]  # authoritative in-memory record
                if rec.is_terminal():
                    continue
                if not self.config.adaptive and len(busy_eps) >= max_in_flight and ep not in busy_eps:
                    continue  # static mode: do not enqueue episodes beyond the quota
                spec = self.specs[ep]
                for t in episode_ready_tasks(rec, spec, self.request):
                    if (ep, t.stage) in queued:
                        continue
                    queued.add((ep, t.stage))
                    if t.stage in (Stage.SCENE, Stage.HAND):
                        # The real resolution tier is read from the record
                        # (prepare is guaranteed DONE before reaching here); the
                        # scene model name follows the record-backfilled stereo
                        # decision, same source as _gpu_payload.
                        res_key = self._resolution_key(ep)
                        model_name = (
                            self._scene_model(ep)
                            if t.stage is Stage.SCENE
                            else t.model_name
                        )
                        item = QueueItem(t.kind, ep, res_key, model_name)
                        (scene_q if t.stage is Stage.SCENE else hand_q).append(item)
                    else:
                        cpu_q.append((ep, t.stage, spec, t))
                # Tasks enqueued for this episode this round count toward the
                # quota (so the gate also holds within a single round).
                busy_eps.add(ep)

        push_ready()
        # GPU slots use slot.busy as the single authority (set True on dispatch,
        # back to False on reclaim, see _complete / scheduler.finish). The idle
        # view is rebuilt every round from self.slots/self.gpu_actors; no
        # incremental free_slots list is kept: _create_pool (triggered through
        # _pool_index_for -> pool_for_episode's create/replenish path) appends
        # new slots to self.slots (no usable slot -> create -> autoscaler pulls
        # a new node), which the incremental list would never see - if a new
        # pool lands on a node with no other slots, that episode could not match
        # any slot, and with the queue holding only it and in_flight empty,
        # `push_ready(); continue` would spin without sleep forever. Rebuilding
        # the view lets the new pool's GpuSlot (busy=False) enter the next round
        # naturally.
        while scene_q or hand_q or cpu_q or in_flight:
            # 1) Dispatch GPU tasks (slot-driven + node-constrained)
            #
            # Dispatch in reverse: iterate every **idle slot** and filter the
            # **items belonging to that slot's node** out of the two global
            # queues (references unchanged, view only), hand the subset to the
            # scheduler/static allocator to choose, then remove the chosen item
            # from the global queue by identity (the item is not necessarily the
            # queue head, so remove is used instead of pop(0); episode_id is
            # unique within a queue, so remove is unambiguous).
            #
            # Why not filter slots by queue head (two Important findings):
            #  a) rotation deadlock: node A's head task is ineligible for node
            #     B's idle slot, and rotating the head stops at the "queue
            #     length <= 1 then break" guard, so B never gets its own task
            #     (happens with interleaved multi-node queues and with
            #     multi-GPU single nodes);
            #  b) affinity violation: assign/_static_assign reads the global
            #     queues and may pick a task from **another node** for this
            #     slot, breaking "the same episode always lands on the same
            #     node".
            #
            # With no GPU nodes (node_registry is None), node routing is
            # disabled: the subset is the whole queue, byte-for-byte equivalent
            # to the pre-change behavior (see the flat fallback in
            # _pool_index_for).
            # This round's GPU dispatch is driven by the idle view rebuilt from
            # self.slots/self.gpu_actors (new pool slots enter the view too);
            # the same slot is not dispatched twice in one round because
            # slot.busy=True is set on dispatch (see below), and busy slots are
            # dropped from the view next round.
            while True:
                dispatched_any = False
                free_slots = [
                    (slot, actor)
                    for slot, actor in zip(self.slots, self.gpu_actors)
                    if not slot.busy
                ]
                for slot, actor in free_slots:
                    if self.node_registry is None:
                        sub_scene, sub_hand = list(scene_q), list(hand_q)
                    else:
                        sub_scene, sub_hand = self._node_eligible_queues(
                            slot, scene_q, hand_q
                        )
                    if not sub_scene and not sub_hand:
                        # This slot has nothing dispatchable: look at the next idle slot.
                        continue
                    if self.config.adaptive:
                        a = self.scheduler.assign(sub_scene, sub_hand, slot)
                    else:
                        a = self._static_assign(sub_scene, sub_hand, slot)
                    if a is None:
                        continue
                    q = scene_q if a.kind == "scene" else hand_q
                    item = a.item
                    q.remove(item)  # remove from the global queue by identity (subset refs map back to global positions)
                    spec = self.specs[item.episode_id]
                    ws = self._ws(item.episode_id)
                    rec = self.records[item.episode_id]
                    stage = Stage.SCENE if a.kind == "scene" else Stage.HAND
                    ws.apply_and_save(rec, stage, StageStatus.RUNNING)
                    self._event(item.episode_id, stage, "start")
                    payload = self._gpu_payload(a.kind, spec, item, ws)
                    slot.busy = True
                    ref = actor.run_task.remote(payload)
                    in_flight[ref] = (item.episode_id, a.kind, item.model_name, slot, actor)
                    if self.node_registry is not None:
                        node = self._actor_node.get(slot.slot_id)
                        if node and node in self.node_registry.pools:
                            self.node_registry.pools[node].take_gpu()
                    self._cpu_stall = 0  # any GPU dispatch is progress: clear the stall counter
                    dispatched_any = True
                # No idle slot has anything to dispatch: end this round of GPU
                # dispatch and move to ray.wait reclaim. Never rotate the queue
                # head - rotation deadlocks.
                if not dispatched_any:
                    break

            # 2) Dispatch CPU tasks
            #
            # CPU stages are routed by the episode's owning node. Scan cpu_q one
            # by one; ineligible entries (`_cpu_actor_for` returns None) are
            # **skipped without popping** (continue to the tail, then rescan),
            # so a blocked head cannot stall later ready tasks. On dispatch the
            # actor is recorded in `_cpu_busy` (accounted by id(actor)) and
            # cleared on reclaim (see reclaim below). Any successful dispatch
            # (GPU or CPU) clears `_cpu_stall` - the convergence counter only
            # counts rounds with no dispatch at all.
            cpu_dispatch_any = False
            idx = 0
            while idx < len(cpu_q):
                ep, stage, spec, _t = cpu_q[idx]
                # Critical fix: ensure assigned_node is persisted before
                # dispatch - prepare is the first stage and assigned_node is
                # always None at its dispatch; the global fallback would land on
                # an arbitrary node X, and a later scene picking Y via
                # pool_for_episode could not read the prepared/ on X (the
                # multi-node loop breaks at the first stage). _ensure_assigned
                # chooses and persists the node for unassigned episodes and
                # forces assigned ones back to their original node, so the
                # result is stable.
                # False = no usable node: skip this round (same semantics as a
                # busy GPU node) and retry next round.
                if not self._ensure_assigned(ep):
                    idx += 1
                    continue
                actor = self._cpu_actor_for(ep)
                if actor is None:
                    idx += 1  # not eligible this round: skip and retry next round (no head blocking)
                    continue
                fn_name, kwargs = build_cpu_kwargs(
                    stage, ep, spec, self.request,
                    self.config.storage_root, self.config.workspace_root, self.config.output_root,
                )
                ws = self._ws(ep)
                rec = self.records[ep]
                ws.apply_and_save(rec, stage, StageStatus.RUNNING)
                self._event(ep, stage, "start")
                self._cpu_busy.add(id(actor))
                ref = actor.run_task.remote(
                    {"fn_name": fn_name, "kwargs": kwargs, "episode_id": ep, "kind": fn_name}
                )
                in_flight[ref] = (ep, fn_name, "", None, actor)
                cpu_q.pop(idx)  # dispatched: remove from queue, keep scanning in place
                cpu_dispatch_any = True
            if cpu_dispatch_any:
                self._cpu_stall = 0

            # 2.5) three-level idle scan: unload returns VRAM / exit returns the
            # node. Scanned every main-loop round (ray.wait timeout=0.05 already
            # spins at high frequency, so the scan cost is bounded -
            # idle_seconds/status are light remote calls and there are usually
            # 0-2 non-busy slots). Only loaded and non-busy slots produce
            # actions; 0 = layer disabled (see idle.py).
            self._apply_idle_actions(self._scan_idle())

            if not in_flight:
                if scene_q or hand_q or cpu_q:
                    # Guaranteed convergence: when cpu_q is non-empty and
                    # in_flight is empty, if nothing can be dispatched
                    # (`_cpu_actor_for` all None, e.g. the owning node vanished)
                    # and the queued set blocks re-enqueue, `push_ready();
                    # continue` would spin without sleep. Per the ruling,
                    # node_wait_timeout_s is mapped to consecutive no-progress
                    # rounds; once the threshold is hit, the head CPU task is
                    # marked FAILED and dequeued - leaving a searchable error
                    # instead of hanging silently.
                    if cpu_q:
                        self._cpu_stall = getattr(self, "_cpu_stall", 0) + 1
                        if self._cpu_stall >= self._cpu_stall_limit():
                            ep, stage, _spec, _t = cpu_q.pop(0)
                            queued.discard((ep, stage))
                            rec = self.records[ep]  # authoritative in-memory record
                            self._ws(ep).apply_and_save(
                                rec, stage, StageStatus.FAILED,
                                error="timed out waiting for a free CPU on the owning node (node_wait_timeout_s)",
                            )
                            self._event(ep, stage, "timeout")
                            self._cpu_stall = 0
                    push_ready()
                    continue
                break

            # 3) Reclaim finished tasks
            done, _ = ray.wait(list(in_flight.keys()), num_returns=1, timeout=0.05)
            for ref in done:
                ep, kind, model_name, slot, actor = in_flight.pop(ref)
                res = self._get_result(ref)
                self._complete((ep, kind, model_name, slot), res)
                if slot is not None:
                    # The slot idle state is registered authoritatively as
                    # slot.busy=False by _complete / scheduler.finish; the next
                    # round's idle view rediscovers it (no incremental
                    # free_slots list is kept).
                    node = self._actor_node.get(slot.slot_id)
                    if node and node in self.node_registry.pools:
                        self.node_registry.pools[node].release_gpu()
                else:
                    # CPU task finished: clear the actor from the busy set so it
                    # can be dispatched again next round.
                    self._cpu_busy.discard(id(actor))

            push_ready()

        failed = self._collect_failed()
        return RunResult(
            total_seconds=0.0,
            per_episode={ep: self.records[ep].to_json() for ep in self.specs},
            scheduler_snapshots=self.scheduler.stats_obj.snapshots,
            timeline=self.timeline,
            failed=failed,
        )

    # ---------- helpers ----------
    def _backfill_prepare(self, ep: str, res: dict) -> None:
        """Backfill prepare's routing metadata into the in-memory record and persist it.

        Shared by the parallel path (``_complete``) and the serial baseline
        (``_execute_one``): the driver does not read workspace files (under
        multi-node it cannot read prepared/summary.json), so has_stereo /
        resolution / degraded info must come back with the actor's return value
        (``stage_result``). Without this step, the serial baseline would keep
        ``rec.has_stereo`` None and ``_scene_model`` would fall back to the
        input-side guess - picking the wrong model when the input is stereo but
        prepare degraded to monocular.
        """
        sr = res.get("stage_result") or {}
        if "width" in sr and "height" in sr:
            rec = self.records.get(ep)  # authoritative in-memory record; skip if absent (abnormal scenario)
            if rec is not None:
                rec.record_prepare(
                    has_stereo=bool(sr.get("has_stereo", False)),
                    width=int(sr["width"]),
                    height=int(sr["height"]),
                    degraded=bool(sr.get("degraded", False)),
                    degrade_reason=sr.get("degrade_reason"),
                    alignment=sr.get("alignment"),
                    align_dropped=int(sr.get("align_dropped", 0) or 0),
                )
                ws = self._ws(ep) if ep else None
                if ws is not None:
                    ws.save_record(rec)

    def _collect_failed(self, already: list | None = None) -> list:
        """Collect the failed list: existing failures + silently unfinished (no FAILED stage but not all DONE).

        The latter (e.g. a stage never submitted because of an orchestration
        gap) cannot be detected through error, so it must be marked explicitly:
        the first unfinished stage is marked FAILED with "incomplete after run",
        so rc/manifest reflect the true result and the record carries a
        searchable error.
        """
        failed = list(already) if already else []
        for ep in self.specs:
            ws = self._ws(ep)
            rec = self.records[ep]  # authoritative in-memory record
            if rec.error:
                if ep not in failed:
                    failed.append(ep)
                continue
            if rec.is_terminal():
                continue
            # non-terminal and no failure: silently unfinished -> mark and count as failed
            first = rec.undone_stages()
            if not first:
                continue
            ws.apply_and_save(rec, Stage(first[0]), StageStatus.FAILED,
                              error="incomplete after run")
            if ep not in failed:
                failed.append(ep)
        return failed

    def _static_assign(self, scene_q, hand_q, slot):
        """Static pool-split assignment - **only operates on the node-eligible subset passed by the caller**.

        The caller (_run_pipeline) passes in the subset of items belonging to
        this slot's node; this method never touches the global queues, so
        node-affinity constraints are satisfied by construction.
        """
        from roboray.scheduling.scheduler import Assignment

        if slot.busy:
            return None
        # static pool split: even slot ids always run scene, odd ones always run hand
        preferred = "scene" if slot.slot_id % 2 == 0 else "hand"
        other = "hand" if preferred == "scene" else "scene"
        for kind in (preferred, other):
            q = scene_q if kind == "scene" else hand_q
            if q:
                return Assignment(
                    slot.slot_id, kind, q[0], slot.loaded_model != q[0].model_name
                )
        return None

    def _gpu_payload(self, kind, spec, item, ws) -> dict:
        """Build the GPU payload - **paths and scalars only** (the actor reads frames on its own node)."""
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
            inference_fps=self.request.stages.hand_reconstruction.inference_fps,
            max_hands=self.request.stages.hand_reconstruction.max_hands,
        )

    def _get_result(self, ref) -> dict:
        try:
            return ray.get(ref)
        except Exception as exc:  # actor crashed
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "episode_id": None,
                    "kind": None, "seconds": 0.0}

    def _complete(self, info, res) -> None:
        ep, kind, model_name, slot = info
        # kind is "scene"/"hand" for GPU tasks and the stage function name for
        # CPU tasks ("prepare"/"fusion"/"action"/"publish") - both share the
        # Stage value name.
        try:
            stage = Stage(kind)
        except ValueError:
            stage = None
        ws = self._ws(ep) if ep else None
        rec = self.records.get(ep)  # authoritative in-memory record; skip if absent (abnormal scenario)
        if res.get("ok"):
            seconds = res.get("infer_seconds") or res.get("seconds") or 0.0
            if kind == "prepare":
                self._backfill_prepare(ep, res)
            if slot is not None:
                self.scheduler.finish(
                    slot,
                    model_name,
                    res.get("resolution_key", "640x480"),
                    seconds,
                    was_load=bool(res.get("was_load")),
                )
            if ws is not None and stage is not None and rec is not None:
                ws.apply_and_save(rec, stage, StageStatus.DONE, seconds=seconds)
                self._event(ep, stage, "end")
                # Cold-start evidence: model load counts are countable from the
                # timeline across all tiers. The "load" event type does not pair
                # with start/end, and overlap_evidence ignores it.
                if res.get("was_load"):
                    self._event(ep, stage, "load")
        else:
            if slot is not None:
                slot.busy = False
            if ws is not None and stage is not None and rec is not None:
                ws.apply_and_save(rec, stage, StageStatus.FAILED, error=str(res.get("error")))

    def _execute_one(self, spec, stage: Stage) -> bool:
        """Serial mode: execute a single stage once."""
        ws = self._ws(spec.episode_id)
        rec = self.records[spec.episode_id]  # authoritative in-memory record
        ws.apply_and_save(rec, stage, StageStatus.RUNNING)
        self._event(spec.episode_id, stage, "start")
        if stage in (Stage.SCENE, Stage.HAND):
            kind = stage.value
            model_name = (
                self._scene_model(spec.episode_id)
                if stage is Stage.SCENE
                else self.request.stages.hand_reconstruction.model
            )
            res_key = self._resolution_key(spec.episode_id)
            payload = self._gpu_payload(
                kind, spec, QueueItem(kind, spec.episode_id, res_key, model_name), ws
            )
            idx = self._pool_index_for(spec.episode_id)
            if idx is None:
                ws.apply_and_save(rec, stage, StageStatus.FAILED, error="no usable GPU node")
                return False
            res = self._get_result(self.gpu_actors[idx].run_task.remote(payload))
            if res.get("ok") and res.get("was_load"):
                self._event(spec.episode_id, stage, "load")
        else:
            # Critical fix: same source as _run_pipeline - ensure assigned_node
            # is persisted before dispatch so prepare routes to the same node as
            # later scene.
            if not self._ensure_assigned(spec.episode_id):
                ws.apply_and_save(
                    rec, stage, StageStatus.FAILED, error="no usable CPU node (no usable GPU node)"
                )
                return False
            fn_name, kwargs = build_cpu_kwargs(
                stage, spec.episode_id, spec, self.request,
                self.config.storage_root, self.config.workspace_root, self.config.output_root,
            )
            actor = self._cpu_actor_for(spec.episode_id)
            if actor is None:
                ws.apply_and_save(
                    rec, stage, StageStatus.FAILED,
                    error="no usable CPU node (owning node's CPU group all busy or node gone)",
                )
                return False
            self._cpu_busy.add(id(actor))
            res = self._get_result(
                actor.run_task.remote(
                    {"fn_name": fn_name, "kwargs": kwargs,
                     "episode_id": spec.episode_id, "kind": fn_name}
                )
            )
            self._cpu_busy.discard(id(actor))
        if res.get("ok"):
            if stage is Stage.PREPARE:
                # Same source as the parallel _complete path (same helper):
                # backfill prepare's routing metadata. Without it, rec.has_stereo
                # stays None, _scene_model falls back to the input-side guess,
                # and the wrong model is chosen when input is stereo but prepare
                # degraded to monocular.
                self._backfill_prepare(spec.episode_id, res)
            ws.apply_and_save(
                rec, stage, StageStatus.DONE,
                seconds=res.get("infer_seconds") or res.get("seconds") or 0.0,
            )
            self._event(spec.episode_id, stage, "end")
            return True
        ws.apply_and_save(rec, stage, StageStatus.FAILED, error=str(res.get("error")))
        return False
