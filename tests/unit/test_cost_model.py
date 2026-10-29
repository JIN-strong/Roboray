import pytest

from roboray.scheduling.cost_model import DEFAULT_PRIOR_SECONDS, CostModel


def test_unknown_key_returns_default_prior():
    cm = CostModel()
    assert cm.predict("scene", "640x480") == DEFAULT_PRIOR_SECONDS


def test_seed_prior_sets_initial_estimate():
    cm = CostModel()
    cm.seed_prior("scene", "640x480", 4.0)
    assert cm.predict("scene", "640x480") == pytest.approx(4.0)


def test_update_moves_estimate_toward_observation():
    cm = CostModel(alpha=0.5)
    cm.seed_prior("scene", "640x480", 4.0)
    cm.update("scene", "640x480", 8.0)
    assert cm.predict("scene", "640x480") == pytest.approx(6.0)
    cm.update("scene", "640x480", 8.0)
    assert cm.predict("scene", "640x480") == pytest.approx(7.0)


def test_updates_converge_to_true_cost():
    cm = CostModel(alpha=0.3)
    for _ in range(80):
        cm.update("hand", "320x240", 2.5)
    assert cm.predict("hand", "320x240") == pytest.approx(2.5, abs=0.05)


def test_keys_are_independent():
    cm = CostModel(alpha=0.5)
    cm.seed_prior("scene", "640x480", 4.0)
    cm.seed_prior("hand", "640x480", 1.0)
    cm.update("scene", "640x480", 10.0)
    assert cm.predict("hand", "640x480") == pytest.approx(1.0)


def test_load_cost_tracked_separately():
    cm = CostModel(alpha=0.5)
    cm.load_prior("scene", 5.0)
    cm.load_cost("scene", 15.0)
    assert cm.predict_load("scene") == pytest.approx(10.0)


def test_snapshot_is_plain_dict():
    cm = CostModel()
    cm.seed_prior("scene", "640x480", 3.0)
    snap = cm.snapshot()
    assert snap["infer"]["scene/640x480"] == pytest.approx(3.0)
