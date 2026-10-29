"""Test doubles: implement the models.protocol protocols, simulating GPU cost with sleep.

Test-only; production code never imports this module.
"""

import time
from pathlib import Path

import numpy as np

from roboray.models import registry
from roboray.stages.artifacts import save_artifact

__all__ = [
    "FakeSceneModel",
    "FakeHandModel",
    "FakeSemanticModel",
    "install_fake_models",
]

N_JOINTS = 21
HANDS = 2


def fake_joints_3d_cam(t: int, mano_trans: np.ndarray) -> np.ndarray:
    """Stand-in MANO forward kinematics: root = mano_trans, joints = root + deterministic small offsets.

    Left and right hands use fixed mirrored-x / scaled-y offsets so L/R are distinguishable;
    no randomness, fully determined by mano_trans (so the fusion mano path reproduces
    stably in tests).
    """
    base = np.zeros((N_JOINTS, 3), np.float64)
    for j in range(N_JOINTS):
        base[j] = (0.012 * j, 0.006 * (j % 3), 0.004 * j)
    mirror = np.array([-1.0, 0.5, 1.0], np.float64)  # left hand = right hand mirrored x, scaled y
    offs = np.stack([base, base * mirror])  # (2,21,3); j=0 all zero -> root = mano_trans
    return (np.asarray(mano_trans, np.float64)[:, :, None, :] + offs[None]).astype(np.float32)


class _FakeBase:
    def __init__(self, name: str, seconds: float = 0.05, fail_on: str | None = None):
        self.name = name
        self.seconds = float(seconds)
        self.fail_on = fail_on
        self.loaded_on = None

    def load(self, device: str) -> None:
        self.loaded_on = device

    def cost_hint(self, n_frames: int, resolution_key: str) -> float:
        return self.seconds * max(1, n_frames) / 8.0

    def _maybe_fail(self, frames, calib: dict) -> None:
        # Through the driver path, calib is the prepared calibration (no fail_on key);
        # matching relies on the episode_id in the frames dict passed by the actor
        # ({"frames": arr, "episode_id": ep}).
        target = calib.get("fail_on") if isinstance(calib, dict) else None
        if not target and isinstance(frames, dict):
            target = frames.get("episode_id")
        if target and self.fail_on == target:
            raise RuntimeError(f"simulated inference failure: episode {target}")

    def _sleep(self) -> None:
        time.sleep(self.seconds)


class FakeSceneModel(_FakeBase):
    def __init__(self, seconds: float = 0.05, fail_on: str | None = None):
        super().__init__("fake_scene", seconds, fail_on)

    def infer(self, frames: dict, calib: dict, out_path: Path) -> None:
        self._maybe_fail(frames, calib)
        arr = np.asarray(frames["frames"])
        t, h, w = arr.shape[0], arr.shape[1], arr.shape[2]
        self._sleep()
        depth_source = "stereo" if calib.get("has_stereo") else "monocular"
        save_artifact(
            out_path,
            camera_poses=np.tile(np.eye(4, dtype=np.float32), (t, 1, 1)),
            intrinsics=np.tile(np.eye(3, dtype=np.float32), (t, 1, 1)),
            depth=np.full((t, h, w), 0.5, np.float32),
            depth_source=np.array(depth_source),
            n_frames=np.int32(t),
            fps=np.int32(int(calib.get("fps", 30))),
        )


_FAKE_CAM_DEPTH = 2.0  # fake camera depth pad: stand-in z may be negative (rng); dividing by it directly would blow up
_WEARER_SCORE = 0.9


def _fake_joints_2d(j3d: np.ndarray, score: float, width: int, height: int) -> np.ndarray:
    """Fake-camera projection to full-image pixel coordinates: (u,v) = (0.5 + xy/(z+depth))*(w,h), 3rd column = score.

    Isomorphic to tests/fakes/modelenv_scripts/fake_hamer.py (the candidates protocol
    requires pixel units).
    """
    z = j3d[..., 2] + _FAKE_CAM_DEPTH
    u = (0.5 + j3d[..., 0] / z) * float(width)
    v = (0.5 + j3d[..., 1] / z) * float(height)
    return np.stack([u, v, np.full(u.shape, score)], axis=-1).astype(np.float32)


def _fake_bbox(joints_2d: np.ndarray) -> np.ndarray:
    """bbox = joint-2D envelope padded by 15% (same pad ratio as run_hamer/_fake_hamer)."""
    u, v = joints_2d[:, 0], joints_2d[:, 1]
    pad = 0.15 * max(float(u.max() - u.min()), float(v.max() - v.min()), 1e-3)
    return np.asarray([u.min() - pad, v.min() - pad, u.max() + pad, v.max() + pad], np.float32)


class FakeHandModel(_FakeBase):
    def __init__(self, seconds: float = 0.05, fail_on: str | None = None):
        super().__init__("fake_hand", seconds, fail_on)

    def infer(self, frames: dict, calib: dict, out_path: Path) -> None:
        self._maybe_fail(frames, calib)
        arr = np.asarray(frames["frames"])
        t = arr.shape[0]
        self._sleep()
        rng = np.random.default_rng(0)
        present = np.ones((t, HANDS), bool)
        # manufacture a missed-detection segment for the gap-fill logic tests
        if t >= 8:
            present[3:5, 0] = False
        mano_trans = rng.normal(0, 0.1, size=(t, HANDS, 3)).astype(np.float32)
        # candidates protocol (isomorphic to fake_hamer.py): per frame, both of the
        # wearer's hands, pixel coordinates (derived from the fake_joints_3d_cam fake
        # camera projection), bbox = envelope padded 15%, score 0.9; aligned with the
        # main present array (the missed-detection side has no candidate that frame).
        # e2e fusion thus goes through the association path (candidates are the wearer
        # themselves -> association matches x-rank/main arrays, existing assertions hold).
        j3d = fake_joints_3d_cam(t, mano_trans)
        img_h, img_w = int(arr.shape[1]), int(arr.shape[2])
        cands = np.empty(t, dtype=object)
        for i in range(t):
            row = []
            for s in (0, 1):
                if not present[i, s]:
                    continue
                j2d = _fake_joints_2d(j3d[i, s], _WEARER_SCORE, img_w, img_h)
                row.append({"bbox": _fake_bbox(j2d), "side": s, "score": _WEARER_SCORE,
                            "joints_2d": j2d, "joints_3d_cam": j3d[i, s]})
            cands[i] = row
        save_artifact(
            out_path,
            hand_present=present,
            joints_2d=rng.normal(0, 1, size=(t, HANDS, N_JOINTS, 3)).astype(np.float32),
            # camera-frame MANO 3D joints (input to the mano path; HaMeR contract names this field)
            joints_3d_cam=j3d,
            mano_pose=rng.normal(0, 0.1, size=(t, HANDS, 48)).astype(np.float32),
            mano_shape=rng.normal(0, 0.1, size=(t, HANDS, 10)).astype(np.float32),
            mano_trans=mano_trans,
            confidence=np.where(present, 0.9, 0.0).astype(np.float32),
            candidates=cands,
            n_frames=np.int32(t),
            fps=np.int32(int(calib.get("fps", 30))),
        )


class FakeSemanticModel:
    def __init__(self, seconds: float = 0.02):
        self.name = "fake_vlm"
        self.seconds = float(seconds)

    def label(self, prompt: str, images: list, out_path: Path, n_segments: int = 0) -> None:
        time.sleep(self.seconds)
        save_artifact(
            out_path,
            labels=np.array(["reach", "grasp", "lift"], dtype=object),
            segment_start=np.array([0, 8, 16], dtype=np.int32),
            segment_end=np.array([8, 16, 24], dtype=np.int32),
        )


def install_fake_models(
    scene_seconds: float = 0.05,
    hand_seconds: float = 0.05,
    fail_on: str | None = None,
) -> tuple:
    """Register the doubles into the registry, returning (scene_name, hand_name, semantic_name)."""
    registry.register_scene_model(
        "fake_scene", lambda: FakeSceneModel(seconds=scene_seconds, fail_on=fail_on)
    )
    registry.register_hand_model(
        "fake_hand", lambda: FakeHandModel(seconds=hand_seconds, fail_on=fail_on)
    )
    registry.register_semantic_model("fake_vlm", lambda: FakeSemanticModel())
    return ("fake_scene", "fake_hand", "fake_vlm")
