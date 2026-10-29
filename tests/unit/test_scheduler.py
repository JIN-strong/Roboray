import pytest

from roboray.scheduling.cost_model import CostModel
from roboray.scheduling.scheduler import AdaptiveScheduler, GpuSlot, QueueItem

# Semantics: the scene queue uses the real model name vipe, the hand queue uses hamer.
SCENE_MODEL = "vipe"
HAND_MODEL = "hamer"


def _item(kind, ep="ep_0001", res="640x480", model=None):
    if model is None:
        model = SCENE_MODEL if kind == "scene" else HAND_MODEL
    return QueueItem(kind=kind, episode_id=ep, resolution_key=res, model_name=model)


def test_choose_kind_picks_bigger_backlog():
    cm = CostModel()
    cm.seed_prior(SCENE_MODEL, "640x480", 4.0)
    cm.seed_prior(HAND_MODEL, "640x480", 1.0)
    sch = AdaptiveScheduler([GpuSlot(0)], cm)
    scene_q = [_item("scene"), _item("scene")]
    hand_q = [_item("hand")]
    # scene backlog 8s vs hand 1s -> choose scene
    assert sch.choose_kind(scene_q, hand_q) == "scene"
    # hand stacked to 10 -> 10s > 8s -> choose hand
    assert sch.choose_kind(scene_q, [_item("hand")] * 10) == "hand"


def test_choose_kind_returns_none_on_empty_queues():
    sch = AdaptiveScheduler([GpuSlot(0)], CostModel())
    assert sch.choose_kind([], []) is None


def test_assign_marks_load_when_model_not_loaded():
    cm = CostModel()
    cm.load_prior(SCENE_MODEL, 5.0)
    sch = AdaptiveScheduler([GpuSlot(0, loaded_model=None)], cm)
    a = sch.assign([_item("scene")], [], sch.slots[0])
    assert a.kind == "scene" and a.needs_load is True


def test_assign_reuses_loaded_model_when_switching_is_not_worth_it():
    cm = CostModel()
    cm.seed_prior(SCENE_MODEL, "640x480", 4.0)
    cm.seed_prior(HAND_MODEL, "640x480", 3.0)
    cm.load_prior(HAND_MODEL, 30.0)      # switching to hand is very expensive
    slot = GpuSlot(0, loaded_model=SCENE_MODEL)
    sch = AdaptiveScheduler([slot], cm)
    # hand backlog 6s > scene 4s, normally hand; but extra only 2s <= 30s load cost -> reuse scene
    a = sch.assign([_item("scene")], [_item("hand")] * 2, slot)
    assert a.kind == "scene" and a.needs_load is False


def test_assign_switches_when_extra_backlog_exceeds_load_cost():
    cm = CostModel()
    cm.seed_prior(SCENE_MODEL, "640x480", 3.0)
    cm.seed_prior(HAND_MODEL, "640x480", 3.0)
    cm.load_prior(HAND_MODEL, 1.0)
    slot = GpuSlot(0, loaded_model=SCENE_MODEL)
    sch = AdaptiveScheduler([slot], cm)
    # hand backlog 30s, scene backlog 3s -> extra 27s > 1s load cost -> switch role
    a = sch.assign([_item("scene")], [_item("hand")] * 10, slot)
    assert a.kind == "hand" and a.needs_load is True


def test_busy_slot_gets_no_assignment():
    sch = AdaptiveScheduler([GpuSlot(0, busy=True)], CostModel())
    assert sch.assign([_item("scene")], [], sch.slots[0]) is None


def test_finish_feeds_back_and_frees_slot():
    cm = CostModel(alpha=0.5)
    cm.seed_prior(SCENE_MODEL, "640x480", 4.0)
    slot = GpuSlot(0, loaded_model=SCENE_MODEL, busy=True)
    sch = AdaptiveScheduler([slot], cm)
    sch.finish(slot, SCENE_MODEL, "640x480", seconds=8.0, was_load=False)
    assert slot.busy is False
    assert cm.predict(SCENE_MODEL, "640x480") == pytest.approx(6.0)


def test_finish_records_load_cost_separately():
    cm = CostModel(alpha=1.0)
    slot = GpuSlot(0, loaded_model=None, busy=True)
    sch = AdaptiveScheduler([slot], cm)
    sch.finish(slot, SCENE_MODEL, "640x480", seconds=12.0, was_load=True)
    assert cm.predict_load(SCENE_MODEL) == pytest.approx(12.0)
    assert cm.stats()["samples"].get(f"{SCENE_MODEL}/640x480", 0) == 0  # loading is not counted in the inference EMA


def test_division_of_labour_follows_skewed_load_without_thrashing():
    """When the hand queue is clearly longer: the actor switches to hand and keeps reusing it (no thrashing)."""
    cm = CostModel(alpha=0.5)
    cm.seed_prior(SCENE_MODEL, "640x480", 4.0)
    cm.seed_prior(HAND_MODEL, "640x480", 1.0)
    slot = GpuSlot(0, loaded_model=SCENE_MODEL)
    sch = AdaptiveScheduler([slot], cm)
    kinds, loads = [], 0
    for _ in range(20):
        a = sch.assign([_item("scene")], [_item("hand")] * 14, slot)
        kinds.append(a.kind)
        if a.needs_load:
            loads += 1
        # measured latency fed back (current prediction used as the observation, simulating steady state)
        sch.finish(slot, HAND_MODEL if a.kind == "hand" else SCENE_MODEL, "640x480",
                   seconds=cm.predict(HAND_MODEL if a.kind == "hand" else SCENE_MODEL, "640x480"),
                   was_load=a.needs_load)
    assert kinds.count("hand") == len(kinds)             # all routed to hand
    assert loads <= 1                                    # loaded only once, then reused
    assert cm.samples(HAND_MODEL, "640x480") == len(kinds) - loads  # measured latency really fed back


def test_ema_loop_closes_on_real_resolution_axis():
    """EMA loop closes on the real resolution axis: finish feeds back the real model name + resolution tier, and decisions converge.

    Seed model "m" with a 1s prior at 640x480 and a 2s prior at 96x64; execution returns a 10s
    observation at 96x64 -> that tier's EMA(2->6). The next 96x64 decision must use the
    converged 6s (not the other tier's 1s, nor the unconverged 2s prior).
    """
    cm = CostModel(alpha=0.5)
    model = "m"
    res_a, res_b = "640x480", "96x64"
    cm.seed_prior(model, res_a, 1.0)
    cm.seed_prior(model, res_b, 2.0)
    slot = GpuSlot(0, loaded_model=model, busy=True)
    sch = AdaptiveScheduler([slot], cm)

    # first execution: observed 10s at the 96x64 tier -> EMA(2->6)
    sch.finish(slot, model, res_b, seconds=10.0, was_load=False)
    assert cm.predict(model, res_b) == pytest.approx(6.0)
    # the queued task is exactly that tier -> the decision must use the converged new estimate
    a = sch.assign([_item("scene", res=res_b, model=model)], [], slot)
    assert a is not None
    snap = sch.stats().snapshots[-1]
    assert snap["predict"] == pytest.approx(6.0)


def test_stats_snapshot_is_json_serialisable():
    import json

    cm = CostModel()
    sch = AdaptiveScheduler([GpuSlot(0)], cm)
    sch.assign([_item("scene")], [], sch.slots[0])
    json.dumps(sch.stats().to_json())
