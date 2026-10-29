import json
import sys
from pathlib import Path

import numpy as np
import pytest

from roboray.models import registry
from roboray.models.hand_models import HamerHandModel
from roboray.models.modelenv import ModelEnvError

FAKES = Path(__file__).resolve().parents[1] / "fakes" / "modelenv_scripts"


@pytest.fixture(autouse=True)
def _registry():
    registry.reset_registry_for_tests()
    registry.load_builtin_models()
    yield
    registry.reset_registry_for_tests()


def test_hamer_runs_fake_script(tmp_path, monkeypatch):
    weights = tmp_path / "weights"
    (weights / "mano").mkdir(parents=True)
    (weights / "mano" / "MANO_RIGHT.pkl").write_bytes(b"fake")
    monkeypatch.setenv("ROBORAY_HAMER_PYTHON", sys.executable)
    monkeypatch.setenv("ROBORAY_HAMER_WEIGHTS", str(weights))
    model = HamerHandModel(script=str(FAKES / "fake_hamer.py"))
    model.load("cpu")
    out = tmp_path / "hand.npz"
    model.infer(
        {"frames": np.zeros((5, 8, 8, 3), np.uint8), "episode_id": "e"},
        {"fps": 30.0},
        out,
    )
    z = np.load(out, allow_pickle=False)
    assert z["hand_present"].shape == (5, 2)
    assert z["mano_pose"].shape == (5, 2, 48)
    assert z["joints_3d_cam"].shape == (5, 2, 21, 3)  # required key hard-validated by the adapter
    assert int(z["__schema_version__"]) == 1


def test_hamer_missing_weights_gives_registration_hint(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBORAY_HAMER_PYTHON", sys.executable)
    monkeypatch.setenv("ROBORAY_HAMER_WEIGHTS", str(tmp_path / "nope"))
    model = HamerHandModel(script=str(FAKES / "fake_hamer.py"))
    with pytest.raises(ModelEnvError) as exc:
        model.load("cpu")
    assert "MANO_RIGHT.pkl" in str(exc.value)
    assert "mano.is.tue.mpg.de" in str(exc.value)


def test_hamer_weights_without_mano_rejected(tmp_path, monkeypatch):
    weights = tmp_path / "w2"
    weights.mkdir()
    monkeypatch.setenv("ROBORAY_HAMER_PYTHON", sys.executable)
    monkeypatch.setenv("ROBORAY_HAMER_WEIGHTS", str(weights))
    model = HamerHandModel(script=str(FAKES / "fake_hamer.py"))
    with pytest.raises(ModelEnvError) as exc:
        model.load("cpu")
    assert "MANO_RIGHT.pkl" in str(exc.value)


def test_registry_resolves_builtin_hand_model():
    assert hasattr(registry.get_hand_model_class("hamer")(), "infer")


# --------------------------------------------------------------------------
# Last mile of inference_fps: payload -> calib -> script extra
# --------------------------------------------------------------------------

def test_hamer_extra_forwards_inference_fps_from_calib(tmp_path, monkeypatch):
    """When calib carries inference_fps/max_hands, the script extra contains those
    keys; when absent, the extra omits them (byte-for-byte default behavior:
    max_hands falls back to the script's DEFAULT_MAX_HANDS=4)."""
    weights = tmp_path / "weights"
    (weights / "mano").mkdir(parents=True)
    (weights / "mano" / "MANO_RIGHT.pkl").write_bytes(b"fake")
    monkeypatch.setenv("ROBORAY_HAMER_WEIGHTS", str(weights))
    model = HamerHandModel()
    model.loaded_on = "cpu"
    extra = model._extra({"fps": 30.0, "inference_fps": 15, "max_hands": 6})
    assert extra["inference_fps"] == 15
    assert extra["max_hands"] == 6
    assert "checkpoint" not in extra
    extra_default = model._extra({"fps": 30.0})
    assert "inference_fps" not in extra_default
    assert "max_hands" not in extra_default


def test_hamer_oneshot_infers_with_stride_when_inference_fps_in_calib(tmp_path, monkeypatch):
    """Oneshot path: calib.inference_fps=15 -> forwarded via --extra -> the stand-in
    script samples with stride=2 (odd frames get hand_present=False). Proves the
    payload field really reaches run_hamer rather than being dead config."""
    weights = tmp_path / "weights"
    (weights / "mano").mkdir(parents=True)
    (weights / "mano" / "MANO_RIGHT.pkl").write_bytes(b"fake")
    monkeypatch.setenv("ROBORAY_HAMER_PYTHON", sys.executable)
    monkeypatch.setenv("ROBORAY_HAMER_WEIGHTS", str(weights))
    model = HamerHandModel(script=str(FAKES / "fake_hamer.py"))
    model.load("cpu")
    out = tmp_path / "hand.npz"
    model.infer(
        {"frames": np.zeros((8, 8, 8, 3), np.uint8), "episode_id": "e"},
        {"fps": 30.0, "inference_fps": 15},
        out,
    )
    z = np.load(out, allow_pickle=False)
    present = z["hand_present"]
    # stride=2: all odd frames unobserved; the even frames keep the fixture's
    # missed-detection frames (3:5) in slot 0, slot 1 observed
    assert not present[1::2].any()
    assert present[0].all()
    assert not present[4, 0] and present[4, 1]


def test_hamer_oneshot_default_has_no_stride_masking(tmp_path, monkeypatch):
    """Without inference_fps: the oneshot output only has the fixture's
    missed detections (frames 3:5 in slot 0), with no stride sampling — a default or
    empty ``--extra`` must not change the default behavior."""
    weights = tmp_path / "weights"
    (weights / "mano").mkdir(parents=True)
    (weights / "mano" / "MANO_RIGHT.pkl").write_bytes(b"fake")
    monkeypatch.setenv("ROBORAY_HAMER_PYTHON", sys.executable)
    monkeypatch.setenv("ROBORAY_HAMER_WEIGHTS", str(weights))
    model = HamerHandModel(script=str(FAKES / "fake_hamer.py"))
    model.load("cpu")
    out = tmp_path / "hand.npz"
    model.infer(
        {"frames": np.zeros((8, 8, 8, 3), np.uint8), "episode_id": "e"},
        {"fps": 30.0},
        out,
    )
    z = np.load(out, allow_pickle=False)
    present = z["hand_present"]
    assert np.all(present[[0, 1, 2, 5, 6, 7]])      # only the fixture frames 3,4 are missing in slot 0
    assert not present[3, 0] and not present[4, 0]
    assert present[3, 1] and present[4, 1]


def test_hamer_session_forwards_inference_fps_in_extra(tmp_path, monkeypatch):
    """resident/session path: the extra the session receives must contain
    inference_fps=15."""
    weights = tmp_path / "weights"
    (weights / "mano").mkdir(parents=True)
    (weights / "mano" / "MANO_RIGHT.pkl").write_bytes(b"fake")
    monkeypatch.setenv("ROBORAY_HAMER_PYTHON", sys.executable)
    monkeypatch.setenv("ROBORAY_HAMER_WEIGHTS", str(weights))
    model = HamerHandModel(script=str(FAKES / "fake_hamer.py"))
    model.load("cpu")

    seen = {}

    class _FakeSession:
        def infer(self, input_path, output_path, weights="", extra="{}", timeout_s=None):
            seen["extra"] = extra
            import subprocess as sp
            p = sp.run(
                [sys.executable, str(FAKES / "fake_hamer.py"),
                 "--input", input_path, "--output", output_path,
                 "--weights", weights, "--extra", extra],
                capture_output=True, text=True, timeout=120,
            )
            assert p.returncode == 0, p.stderr
            return {"ok": True, "model": "fake_hamer", "n_frames": 8}

    model._session = _FakeSession()
    out = tmp_path / "hand.npz"
    model.infer(
        {"frames": np.zeros((8, 8, 8, 3), np.uint8), "episode_id": "e"},
        {"fps": 30.0, "inference_fps": 15},
        out,
    )
    assert "inference_fps" in json.loads(seen["extra"])
    assert json.loads(seen["extra"])["inference_fps"] == 15
    z = np.load(out, allow_pickle=False)
    assert not z["hand_present"][1::2].any()  # the session path also applied stride
