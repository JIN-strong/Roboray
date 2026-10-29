import pytest

from roboray.models import registry
from roboray.models.protocol import resolution_key


def test_resolution_key_is_stable():
    assert resolution_key(640, 480) == "640x480"
    assert resolution_key(641, 480) == "640x480"  # bucketed by 64px so jitter does not fragment the EMA


def test_register_and_lookup(monkeypatch):
    registry.reset_registry_for_tests()
    sentinel = object()
    registry.register_scene_model("fake_scene", lambda: sentinel)
    assert registry.get_scene_model_class("fake_scene")() is sentinel
    assert registry.known_scene_models() == ["fake_scene"]


def test_unknown_model_raises_with_available_names():
    registry.reset_registry_for_tests()
    registry.register_hand_model("fake_hand", lambda: object())
    with pytest.raises(registry.UnknownModelError) as exc:
        registry.get_hand_model_class("nope")
    assert "fake_hand" in str(exc.value)


def test_duplicate_registration_without_override_is_rejected():
    registry.reset_registry_for_tests()
    registry.register_scene_model("m", lambda: object())
    with pytest.raises(ValueError):
        registry.register_scene_model("m", lambda: object())


def test_load_builtin_models_registers_production_names():
    registry.reset_registry_for_tests()
    registry.load_builtin_models()
    assert set(registry.known_scene_models()) >= {"vipe", "fast_foundation_stereo"}
    assert "hamer" in registry.known_hand_models()
    assert "openai_vlm" in registry.known_semantic_models()
