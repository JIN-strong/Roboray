"""Real-model GPU smoke tests. Excluded by default via pyproject's addopts
`-m "not gpu"`; run on a GPU node with `pytest -m gpu`.

Prerequisites: the three model envs are built (deploy/modelenvs/*.sh), weights are
downloaded, and the ROBORAY_*_PYTHON/WEIGHTS environment variables are set (see
docs/models.md).

Running: the CLI `-m gpu` overrides the `-m "not gpu"` in addopts (pytest CLI
options override rather than AND-combine), so **no** extra options are needed:

    ROBORAY_GPU_SMOKE=1 conda run -n roboray python -m pytest -m gpu

Optional (defensively clearing addopts; not required, the command above works):

    ROBORAY_GPU_SMOKE=1 conda run -n roboray python -m pytest -o addopts="" -m gpu
"""

import os

import cv2
import numpy as np
import pytest

from roboray.models.scene_models import FastFoundationStereoModel, VipeSceneModel
from roboray.models.hand_models import HamerHandModel

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not os.environ.get("ROBORAY_GPU_SMOKE"),
        reason="ROBORAY_GPU_SMOKE is not set",
    ),
]


def _frames(t=4, h=288, w=512):
    """Synthetic frames with **visible structure** (make_synthetic_ego style:
    gradient + checkerboard texture + hand disks).

    Pure random noise is featureless for ViPE SLAM and could legitimately fail the
    smoke test while wasting a GPU run; gradient, checkerboard edges and disk
    contours provide trackable features, so the smoke test exercises "the wiring is
    connected" rather than noise.
    """
    yy, xx = np.mgrid[0:h, 0:w]
    grad = ((xx / max(1, w - 1) + yy / max(1, h - 1)) * 0.5 * 200).astype(np.int32)
    checker = (((xx // 24) + (yy // 24)) % 2).astype(np.int32) * 60
    tex = np.clip(grad + checker, 0, 255).astype(np.uint8)
    base = np.repeat(np.stack([tex, tex, tex], axis=-1)[None], t, axis=0)
    ts = np.linspace(0, 2 * np.pi, t)
    cx = (w * (0.5 + 0.25 * np.sin(ts))).astype(int)
    cy = (h * (0.55 + 0.15 * np.cos(ts))).astype(int)
    for i in range(t):
        cv2.circle(base[i], (int(cx[i]), int(cy[i])), max(3, w // 16), (0, 255, 0), -1)
    right = np.roll(base, -6, axis=2)
    return base, right


def test_vipe_gpu_smoke(tmp_path):
    frames, _ = _frames()
    model = VipeSceneModel()
    model.load("cuda")
    out = tmp_path / "scene.npz"
    model.infer({"frames": frames, "episode_id": "smoke"},
                {"fps": 30.0, "has_stereo": False, "T_left_to_right": np.eye(4).tolist()}, out)
    z = np.load(out, allow_pickle=False)
    assert z["depth"].shape == frames.shape[:3]
    assert np.isfinite(z["depth"]).all() and (z["depth"] > 0).any()
    assert np.isfinite(z["camera_poses"]).all()


@pytest.mark.skipif(not os.environ.get("ROBORAY_FFS_WEIGHTS"), reason="FFS weights are not set")
def test_ffs_gpu_smoke(tmp_path):
    frames, right = _frames()
    calib = {"fps": 30.0, "has_stereo": True,
             "T_left_to_right": np.eye(4).tolist(), "K": np.diag([512.0, 512, 1, 1]).tolist()}
    calib["T_left_to_right"][0][3] = 0.06
    model = FastFoundationStereoModel()
    model.load("cuda")
    out = tmp_path / "scene.npz"
    model.infer({"frames": frames, "frames_right": right, "episode_id": "smoke"}, calib, out)
    z = np.load(out, allow_pickle=False)
    assert z["depth"].shape == frames.shape[:3]
    assert np.isfinite(z["depth"]).all() and (z["depth"] > 0).any()
    assert z["camera_poses"].shape == (frames.shape[0], 4, 4)
    assert z["intrinsics"].shape == (frames.shape[0], 3, 3)


@pytest.mark.skipif(not os.environ.get("ROBORAY_HAMER_WEIGHTS"), reason="HaMeR weights are not set")
def test_hamer_gpu_smoke(tmp_path):
    frames, _ = _frames()
    model = HamerHandModel()
    model.load("cuda")
    out = tmp_path / "hand.npz"
    model.infer({"frames": frames, "episode_id": "smoke"}, {"fps": 30.0}, out)
    z = np.load(out, allow_pickle=False)
    assert z["hand_present"].shape == (4, 2)
    # Contract: camera-frame MANO forward-kinematics joints (t, 2, 21, 3), the input
    # of the mano association path.
    assert z["joints_3d_cam"].shape == (4, 2, 21, 3)
    assert np.isfinite(z["mano_pose"]).all()
