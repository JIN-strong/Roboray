"""Node discovery and node-pool placement.

**Why this is needed**: episode workspaces live on the **node-local** filesystem
(hostPath). Therefore all stages of the same episode must land on the same node
- otherwise ``prepared/`` and ``artifacts/`` cannot be read. This module owns
the "which episode goes to which node" rule.

**Why it is decoupled from Ray**: placement is the part of this design most
prone to error and most worth pinning down with unit tests. Ray calls are
isolated in ``discover_gpu_nodes()`` and the ``create_pool`` callback, so the
rule itself can be tested directly against a fake node table without starting a
cluster.
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

    Returns ``[]`` when Ray is not initialized or the cluster has no GPU nodes -
    **this is the norm, not an exception**: local test machines have no GPU (Ray
    runs with virtual ``num_gpus``) and CPU-only clusters are the same. Callers
    fall back to the old no-pool behavior accordingly.
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
    """An actor pool on one node: K GPU actors + N CPU actors, all on the same node.

    GPU slots are exclusive (K = the node's GPU count): a multi-GPU node's
    second card is no longer idled. ``in_flight`` is the concurrent-occupancy
    counter and ``has_free_gpu`` uses the actor count as capacity.
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


class NodePoolRegistry:
    """The episode -> node-pool placement registry.

    ``live_nodes`` is a **snapshot at discovery time**; ``register`` / ``drop``
    maintain the actual pools. The pool count can exceed ``live_nodes`` (a newly
    created pool can land on a new node not yet in the snapshot).
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
        existing = self.pools.get(pool.node_id)
        if existing is None:
            self.pools[pool.node_id] = pool
            return
        # A second pool on the same node (a naive overwrite would drop actors
        # and misalign in_flight) - merge instead.
        existing.gpu_actors.extend(
            a for a in pool.gpu_actors if not any(a is b for b in existing.gpu_actors))
        existing.cpu_actors.extend(
            a for a in pool.cpu_actors if not any(a is b for b in existing.cpu_actors))

    def drop(self, node_id: str) -> None:
        # Node loss: remove both the pool and the live snapshot - otherwise
        # live_node_ids() would still contain it.
        self.pools.pop(node_id, None)
        self._live.pop(node_id, None)

    def live_node_ids(self) -> set[str]:
        """Current usable nodes = live at discovery time union pools (including newly scaled-out nodes)."""
        return set(self._live) | set(self.pools)

    def pool_for_episode(self, episode_id: str, assigned: dict) -> NodePool | None:
        """Pick a node pool for this episode; None when no node is usable and none can be created.

        Priority:
        1. If this episode's ``assigned_node`` is still in the live nodes ->
           **must return to the original node** (otherwise the workspace is not
           on this machine and the episode must be re-run in full)
        2. The idle pool with the fewest in-flight episodes
        3. All full -> create a new pool (``create_pool`` creates the GPU actor
           without affinity; placement is decided by Ray + the autoscaler)
        """
        node_id = assigned.get(episode_id)
        if node_id:
            if node_id in self.pools:
                return self.pools[node_id]
            if node_id in self._live:
                pool = self._create_pool(node_id)
                self.register(pool)
                return pool
            return None  # original node gone: the caller must re-run in full

        if not self._live and not self.pools:
            return None  # no GPU nodes (local test machine / CPU-only cluster)

        free = [p for p in self.pools.values() if p.has_free_gpu()]
        if free:
            return min(free, key=lambda p: (p.in_flight, p.node_id))

        # All full: create new. node_id is left empty first; create_pool fills
        # it in after creating the actor (the GPU actor has no affinity, so
        # placement is decided by Ray + the autoscaler).
        pool = self._create_pool("")
        self.register(pool)
        return pool
