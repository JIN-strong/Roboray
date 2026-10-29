import cv2
import numpy as np
import pytest

from roboray.runtime.episode import EpisodeWorkspace, Stage
from roboray.runtime.storage import PosixStorage
from roboray.runtime.episode import discover_episodes


def test_generate_dataset_layout_and_readability(tmp_path):
    from scripts.make_synthetic_ego import generate_dataset

    generate_dataset(tmp_path, n_episodes=2, n_frames=12, size=(96, 72), stereo=True, seed=1)
    specs = discover_episodes(PosixStorage(tmp_path))
    assert [s.episode_id for s in specs] == ["ep_0001", "ep_0002"]
    assert all(s.has_stereo for s in specs)

    cap = cv2.VideoCapture(str(tmp_path / "ep_0001" / "rgb_left.mp4"))
    count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        count += 1
        assert frame.shape[:2] == (72, 96)
    cap.release()
    assert count == 12


def test_ground_truth_is_present_and_shaped(tmp_path):
    from scripts.make_synthetic_ego import generate_dataset

    generate_dataset(tmp_path, n_episodes=1, n_frames=10, size=(96, 72), seed=2)
    gt = np.load(tmp_path / "ep_0001" / "ground_truth.npz")
    assert gt["joints_world"].shape == (10, 2, 21, 3)
    assert gt["depth"].shape[0] == 10


def test_monocular_dataset_has_no_right_video(tmp_path):
    from scripts.make_synthetic_ego import generate_dataset
    from roboray.runtime.storage import PosixStorage as PS

    generate_dataset(tmp_path, n_episodes=1, n_frames=6, stereo=False, seed=3)
    specs = discover_episodes(PS(tmp_path))
    assert specs[0].has_stereo is False


def test_calibration_is_loadable(tmp_path):
    import json

    from scripts.make_synthetic_ego import generate_dataset

    generate_dataset(tmp_path, n_episodes=1, n_frames=6, seed=4)
    calib = json.loads((tmp_path / "ep_0001" / "calibration.json").read_text())
    assert np.array(calib["K"]).shape == (3, 3)
    assert "fps" in calib


def test_stereo_baseline_encodes_exact_disparity(tmp_path):
    """Calibration is consistent with pixels: f*B/Z = disparity. The default fixture has 5px disparity and constant depth Z=0.5."""
    import json

    from scripts.make_synthetic_ego import generate_dataset

    generate_dataset(tmp_path, n_episodes=1, n_frames=6, seed=5)
    calib = json.loads((tmp_path / "ep_0001" / "calibration.json").read_text())
    k00 = np.array(calib["K"])[0, 0]
    t03 = np.array(calib["T_left_to_right"])[0, 3]
    assert abs(k00 * t03 - 5 * 0.5) < 1e-6
