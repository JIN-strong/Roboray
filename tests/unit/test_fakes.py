import numpy as np
import pytest

from roboray.models import registry
from tests.fakes.models import install_fake_models


@pytest.fixture(autouse=True)
def _clean_registry():
    registry.reset_registry_for_tests()
    yield
    registry.reset_registry_for_tests()


def test_install_fake_models_registers_all_three():
    names = install_fake_models()
    assert names == ("fake_scene", "fake_hand", "fake_vlm")
    assert registry.get_scene_model_class("fake_scene")() is not None


def test_fake_scene_writes_valid_artifact(tmp_path):
    install_fake_models()
    model = registry.get_scene_model_class("fake_scene")()
    model.load("cpu")
    out = tmp_path / "scene.npz"
    model.infer({"frames": np.zeros((3, 8, 8, 3), np.uint8)}, {"fps": 30}, out)
    z = np.load(out)
    assert z["camera_poses"].shape == (3, 4, 4)
    assert "depth" in z.files
    assert z["depth_source"].item() in ("stereo", "monocular")


def test_fake_hand_writes_valid_artifact(tmp_path):
    install_fake_models()
    model = registry.get_hand_model_class("fake_hand")()
    model.load("cpu")
    out = tmp_path / "hand.npz"
    model.infer({"frames": np.zeros((3, 8, 8, 3), np.uint8)}, {"fps": 30}, out)
    z = np.load(out)
    assert z["joints_2d"].shape[0:2] == (3, 2)
    assert z["mano_pose"].shape[0:2] == (3, 2)


def test_fake_failure_is_scoped_to_one_episode(tmp_path):
    install_fake_models(fail_on="ep_0002")
    model = registry.get_scene_model_class("fake_scene")()
    model.load("cpu")
    with pytest.raises(RuntimeError, match="ep_0002"):
        model.infer(
            {"frames": np.zeros((2, 8, 8, 3), np.uint8), "episode_id": "ep_0002"},
            {"fps": 30, "fail_on": "ep_0002"},
            tmp_path / "x.npz",
        )
