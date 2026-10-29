"""Three-level idle policy: compute_idle_actions pure function (testable with a fake clock, no Ray dependency).

Policy and mechanism are separated: idle detection is a **pure function** and
the driver only applies the actions (unload / unload-then-ray.kill). This module
does not import ray - the threshold semantics can be tested directly against a
fake IdleState without starting a cluster.

Three levels (thresholds must increase; node reclamation at 300s):
- actor_idle_unload_s (default 30s): idle >= threshold -> unload (VRAM returned,
  actor stays alive)
- actor_idle_exit_s (default 240s): idle >= threshold -> unload then kill the
  actor (node returned)
- Node reclamation at 300s is the RayCluster autoscaler's idleTimeoutSeconds,
  outside this module's responsibility

**0 means "disabled", not "immediate"**: unload_s=0 = never unload proactively
(only on role switch / teardown); exit_s=0 = never exit proactively. Threshold
comparisons use **>=**; exit is checked **before** unload - exit implies unload
first, and the same state yields only one action, never two.
"""

from dataclasses import dataclass

__all__ = [
    "IdleState",
    "UNLOAD_DEFAULT",
    "EXIT_DEFAULT",
    "compute_idle_actions",
]

# Same source as the ExecutionConfig defaults (actor_idle_unload_s=30.0 /
# 240.0). Tests explicitly assert UNLOAD_DEFAULT < EXIT_DEFAULT to pin the
# increasing relationship.
UNLOAD_DEFAULT = 30.0
EXIT_DEFAULT = 240.0


@dataclass
class IdleState:
    """A snapshot of one idle slot **at scan time**."""

    actor: object       # GpuModelActor (the driver calls idle_seconds/unload on it directly)
    slot: object        # GpuSlot - busy is decided solely by slot.busy (the fixed semantics)
    idle_seconds: float  # seconds since the last successful inference (already fetched by the caller's ray.get; exceptions count as 0)
    loaded: bool         # corresponds to actor.loaded_kind is not None: nothing to unload if no model is loaded


def compute_idle_actions(
    states: list[IdleState],
    now: float,
    unload_s: float,
    exit_s: float,
) -> list[tuple[str, IdleState]]:
    """Decide actions for each idle-slot snapshot, returning the action sequence ``[("unload"|"exit", st), ...]``.

    **A busy slot never produces an action**: it is running a task (slot.busy is
    the sole authority, set at dispatch/reclaim) and must never be unloaded or
    killed - "the idle scan hurting an in-flight actor" is a hard red line.

    The loaded gate applies only to **unload**: an actor with no model loaded
    has nothing to unload; but the **exit check is not limited by the loaded
    gate** - an unloaded actor also times out idle (it still occupies a slot /
    node; when 240s arrive it should exit, otherwise the node never returns to
    the cluster). The driver-side unload on an unloaded actor is a no-op, so
    exit's implied unload-first semantics are unaffected.

    ``now`` is a fake-clock baseline, kept for a future change to "pass
    last_used rather than absolute idle seconds"; the current decision uses the
    snapshot's absolute ``idle_seconds`` directly (see the module docstring for
    threshold semantics).
    """
    actions: list[tuple[str, IdleState]] = []
    for st in states:
        if getattr(st.slot, "busy", False):
            continue  # never touch an in-flight slot (red line), before any threshold check
        idle = st.idle_seconds
        # exit is checked before unload: exit implies unload first (gracefully
        # close the session, then ray.kill - otherwise SIGKILL skips cleanup and
        # orphaned child processes keep holding VRAM). One state yields only one
        # action. exit ignores loaded: an unloaded actor also times out idle.
        if exit_s and idle >= exit_s:
            actions.append(("exit", st))
        elif st.loaded and unload_s and idle >= unload_s:
            actions.append(("unload", st))
        # 0 = disabled (see module docstring); if no threshold is met, no action this round
    return actions
