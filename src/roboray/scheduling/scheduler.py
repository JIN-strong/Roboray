"""Adaptive scheduler: assigns GPU actor roles from four signals (queue backlog / predicted latency / available resources / loaded model).

Pure Python with no Ray dependency - easy to unit test; the Ray runtime is just
its execution shell.
"""

from dataclasses import dataclass, field
from typing import Optional

from .cost_model import CostModel

__all__ = ["QueueItem", "GpuSlot", "Assignment", "SchedulingStats", "AdaptiveScheduler"]


@dataclass(frozen=True)
class QueueItem:
    kind: str  # "scene" | "hand"
    episode_id: str
    resolution_key: str
    model_name: str


@dataclass
class GpuSlot:
    slot_id: int
    loaded_model: Optional[str] = None
    busy: bool = False


@dataclass(frozen=True)
class Assignment:
    slot_id: int
    kind: str
    item: QueueItem
    needs_load: bool


@dataclass
class SchedulingStats:
    snapshots: list = field(default_factory=list)

    def to_json(self) -> dict:
        return {"snapshots": list(self.snapshots)}


class AdaptiveScheduler:
    def __init__(
        self,
        slots: list,
        cost_model: CostModel,
        scene_bias: float = 1.0,
        hand_bias: float = 1.0,
        reuse_tolerance: float = 1.0,
    ):
        self.slots = list(slots)
        self.cost_model = cost_model
        self.scene_bias = float(scene_bias)
        self.hand_bias = float(hand_bias)
        self.reuse_tolerance = float(reuse_tolerance)
        self.stats_obj = SchedulingStats()

    def stats(self) -> SchedulingStats:
        return self.stats_obj

    def _backlog(self, kind: str, queue: list) -> float:
        """Total estimated time for a kind's queue (by each task's real model + resolution tier)."""
        bias = self.scene_bias if kind == "scene" else self.hand_bias
        return sum(
            self.cost_model.predict(i.model_name, i.resolution_key) for i in queue
        ) * bias

    def choose_kind(self, scene_q: list, hand_q: list) -> Optional[str]:
        if not scene_q and not hand_q:
            return None
        scene_est = self._backlog("scene", scene_q)
        hand_est = self._backlog("hand", hand_q)
        if scene_est <= 0.0 and hand_est <= 0.0:
            return None
        return "scene" if scene_est >= hand_est else "hand"

    def _queue_for(self, kind: str, scene_q: list, hand_q: list) -> list:
        return scene_q if kind == "scene" else hand_q

    def assign(self, scene_q: list, hand_q: list, slot: GpuSlot) -> Optional[Assignment]:
        if slot.busy:
            return None
        wanted = self.choose_kind(scene_q, hand_q)
        if wanted is None:
            return None

        # Reuse decision (the design article: prefer reusing the loaded model).
        # ``loaded_model`` is the **model name**; first locate which kind queue
        # that model falls in (judged by the head item's model_name), then
        # trade backlog vs load cost for a role switch; needs_load compares
        # model names exactly.
        chosen = wanted
        if slot.loaded_model is not None:
            loaded_kind = None
            for kind in ("scene", "hand"):
                q = self._queue_for(kind, scene_q, hand_q)
                if q and q[0].model_name == slot.loaded_model:
                    loaded_kind = kind
                    break
            if loaded_kind is not None and loaded_kind != wanted:
                reuse_q = self._queue_for(loaded_kind, scene_q, hand_q)
                want_q = self._queue_for(wanted, scene_q, hand_q)
                if reuse_q and want_q:
                    extra_backlog = (
                        self._backlog(wanted, want_q)
                        - self._backlog(loaded_kind, reuse_q)
                    )
                    want_model = want_q[0].model_name
                    if (
                        extra_backlog
                        <= self.cost_model.predict_load(want_model) * self.reuse_tolerance
                    ):
                        chosen = loaded_kind

        queue = self._queue_for(chosen, scene_q, hand_q)
        if not queue:
            return None
        item = queue[0]
        chosen_model = item.model_name
        assignment = Assignment(
            slot_id=slot.slot_id,
            kind=chosen,
            item=item,
            needs_load=slot.loaded_model != chosen_model,
        )
        self.stats_obj.snapshots.append(
            {
                "slot_id": slot.slot_id,
                "chosen": chosen,
                "wanted": wanted,
                "needs_load": assignment.needs_load,
                "scene_q": len(scene_q),
                "hand_q": len(hand_q),
                "predict": round(
                    self.cost_model.predict(chosen_model, item.resolution_key), 4
                ),
                "loaded": slot.loaded_model,
            }
        )
        return assignment

    def finish(
        self,
        slot: GpuSlot,
        model_name: str,
        resolution_key: str,
        seconds: float,
        was_load: bool,
    ) -> None:
        """Feed measured latency back: inference into the (model, resolution tier) EMA, loading into the model-level load EMA.

        ``model_name`` is the real model name that ran the task; the slot is
        then recorded as having that model loaded.
        """
        if was_load:
            self.cost_model.load_cost(model_name, seconds)
        else:
            self.cost_model.update(model_name, resolution_key, seconds)
        slot.loaded_model = model_name
        slot.busy = False
