import json
import sys
from pathlib import Path

import numpy as np
import pytest

from roboray.stages.fusion import UNTRUSTED_REPROJ_ERROR, backproject_to_world

_FAKE_EXPORT = (
    Path(__file__).resolve().parents[1] / "fakes" / "modelenv_scripts" / "fake_lerobot_export.py"
)


@pytest.fixture(autouse=True)
def _lerobot_export_env(monkeypatch):
    """publish now calls the LeRobot export in a subprocess (the real script needs the
    ego-lerobot env).

    This module only verifies QC semantics, so an export stand-in suffices — following the
    real contract (NPZ in / __RESULT__ out).
    """
    monkeypatch.setenv("ROBORAY_LEROBOT_PYTHON", sys.executable)
    monkeypatch.setenv("ROBORAY_LEROBOT_SCRIPT", str(_FAKE_EXPORT))
    yield


# -- A1 Kabsch rigid alignment --
def _rot(axis, deg):
    axis = np.asarray(axis, float)
    axis /= np.linalg.norm(axis)
    a = np.deg2rad(deg)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * (K @ K)


def test_rigid_align_recovers_rotation_translation():
    from roboray.stages.fusion import rigid_align

    rng = np.random.default_rng(7)
    base = rng.normal(0, 0.05, (21, 3))
    R, t = _rot([0.3, 1.0, -0.2], 25.0), np.array([0.02, -0.03, 0.4])
    mano = base + rng.normal(0, 1e-4, base.shape)          # MANO metric skeleton (≈ground truth)
    target = (R @ base.T).T + t + rng.normal(0, 2e-3, base.shape)  # depth backprojection (noisy)
    valid = np.ones(21, bool)
    aligned, method = rigid_align(mano, target, valid)
    assert method == "kabsch"
    # Recovery quality: max deviation from the noiseless truth after alignment < 5mm
    # (Kabsch closed form is robust to 2mm noise)
    assert np.linalg.norm(aligned - ((R @ base.T).T + t), axis=-1).max() < 5e-3


def test_rigid_align_partial_valid_points():
    from roboray.stages.fusion import rigid_align

    rng = np.random.default_rng(8)
    base = rng.normal(0, 0.05, (21, 3))
    R, t = _rot([1, 0, 0], 10.0), np.array([0.0, 0.0, 0.35])
    mano = base.copy()
    target = (R @ base.T).T + t
    valid = np.zeros(21, bool)
    valid[[0, 4, 8, 12, 16, 20]] = True   # 6 depth-support points
    aligned, method = rigid_align(mano, target, valid)
    assert method == "kabsch"
    assert np.linalg.norm(aligned - target, axis=-1).max() < 1e-6


def test_rigid_align_degrades_to_root_only():
    from roboray.stages.fusion import rigid_align

    pts = np.ones((21, 3)) * np.arange(21)[:, None] * 0.01   # all collinear
    valid = np.ones(21, bool)
    aligned, method = rigid_align(pts, pts + 0.3, valid)
    assert method == "root_only"                            # collinear -> singular covariance -> fallback
    assert abs(aligned[0, 2] - (pts[0, 2] + 0.3)) < 1e-9    # root-translation semantics preserved
    valid3 = np.zeros(21, bool)
    valid3[:3] = True
    _a, m3 = rigid_align(pts, pts, valid3)
    assert m3 == "root_only"                                # valid points < 4 -> fallback


def test_mano_world_kabsch_beats_root_translation():
    """In a synthetic scene Kabsch alignment's reprojection error is <= root translation.

    Note (necessary correction to the original construction, verified numerically): the
    original moved the hand to z≈0.5 m then rotated the whole hand 20° about the y axis —
    the palm swung to x≈+0.17 m, so at f=60 px all 21 joints project to u≈23~28, and within
    a 6x8 image **0/21 fall in-bounds**; with the depth map NaN except the forced root point,
    the root's bilinear depth sample (2x2 neighborhood must be all-finite) is necessarily
    NaN -> the function early-returns independent of the implementation and the triple
    unpacking necessarily fails. Instead this test uses "rotate about its own origin first,
    then translate to z=0.5" (the hand stays in view) and writes a **3x3 depth
    neighborhood** for every in-bounds joint (satisfying the bilinear 2x2 all-finite
    requirement); the MANO side gets a real rigid misalignment (30° about y, i.e. the
    "overall orientation" divergence) + perturbation, so the kabsch <= root comparison has
    real separation (root_only retains the whole-hand rotation error). The z jitter (0.4 mm)
    is far smaller than the in-plane scale (8 mm): non-coplanarity is enough to pass Kabsch's
    condition-number degeneracy check (measured singular-value ratio ~1e-3, 3 orders above
    the 1e-6 threshold), while keeping the adjacent-joint depth-block aliasing error far
    below the misalignment residual.
    """
    from roboray.stages.fusion import _mano_hand_to_world, _reproject_px, rigid_align

    rng = np.random.default_rng(9)
    K = np.array([[60.0, 0, 4.0], [0, 60.0, 3.0], [0, 0, 1]])
    h, w = 6, 8
    base = rng.normal(0, 0.008, (21, 3))
    base[:, 2] = rng.normal(0, 4e-4, 21)               # near-coplanar (rank check still safe)
    cam = (_rot([0, 1, 0], 20.0) @ base.T).T          # camera-frame truth (20° about own orientation)
    cam[:, 2] += 0.5                                   # translate 0.5 m in front of the camera
    depth = np.full((h, w), np.nan, np.float32)
    joints_2d = np.zeros((21, 3))
    for j, (x, y, z) in enumerate(cam):
        u = K[0, 0] * x / z + K[0, 2]
        v = K[1, 1] * y / z + K[1, 2]
        joints_2d[j, :2] = (u, v)
        ui, vi = int(round(u)), int(round(v))
        if 0 <= ui < w and 0 <= vi < h:
            depth[max(vi - 1, 0):vi + 2, max(ui - 1, 0):ui + 2] = z  # 3x3 neighborhood -> 2x2 all-finite
    # Root-node depth point must be valid (clamp into bounds if out of range)
    r_v, r_u = int(round(joints_2d[0, 1])), int(round(joints_2d[0, 0]))
    r_v, r_u = min(max(r_v, 0), h - 1), min(max(r_u, 0), w - 1)
    depth[r_v, r_u] = float(cam[0, 2])
    # MANO with a rigid misalignment ("overall orientation" divergence) + perturbation
    mano = (_rot([0, 1, 0], 30.0) @ cam.T).T
    mano = mano + rng.normal(0, 1e-3, cam.shape)
    c2w = np.eye(4); w2c = np.eye(4)
    world, reproj, method = _mano_hand_to_world(mano, joints_2d, depth, K, w2c, c2w)
    assert method == "kabsch"
    # Contrast: only the root valid with the same input -> rigid_align takes root translation
    # (fallback path)
    valid_root = np.zeros(21, bool); valid_root[0] = True
    root_pts, root_method = rigid_align(mano, cam, valid_root)
    assert root_method == "root_only"
    fin = np.isfinite(joints_2d[:, :2]).all(axis=1)
    root_reproj = _reproject_px(root_pts[fin], joints_2d[fin, :2], K, w2c)
    # Kabsch removes the overall orientation divergence -> reprojection strictly no worse
    # than root translation (measured ~0.17 vs ~0.32 px)
    assert reproj <= root_reproj + 1e-6
    assert reproj < 0.5  # after kabsch, roughly aligned with the image evidence (rules out
    #                     the "both equally bad" false contrast)


def _pinhole_setup(n=3, h=48, w=64, depth_m=2.0):
    """Synthetic ground truth: a 3D point moving along x, K/w2c known, the depth map holds
    the true depth at the projection."""
    K = np.tile(np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], np.float64), (n, 1, 1))
    w2c = np.tile(np.eye(4, dtype=np.float64), (n, 1, 1))
    pts_cam = np.zeros((n, 3), np.float64)
    pts_2d = np.zeros((n, 2), np.float64)
    depth = np.full((n, h, w), depth_m, np.float32)
    return K, w2c, pts_cam, pts_2d, depth, depth_m


def test_backproject_reprojection_roundtrip_is_exact():
    """Backprojection -> reprojection round-trip error < 1e-3."""
    n, depth_m = 3, 2.0
    K, w2c, pts_cam, pts_2d, depth, _ = _pinhole_setup(n)
    # Camera-frame 3D point: (0.1, -0.05, 2.0)
    pts_cam[:, 0] = 0.1
    pts_cam[:, 1] = -0.05
    pts_cam[:, 2] = depth_m
    # Project to pixels (K identical per frame; frame 0's K suffices — K is already (n,3,3) stacked)
    uvw = K[0] @ pts_cam.T  # (3,n)
    pts_2d = (uvw[:2] / uvw[2]).T
    # Single-joint input: joints_2d (T,2,21,3) with joints only on hand 0; dim 3 unused
    joints_2d = np.zeros((n, 2, 21, 3), np.float64)
    joints_2d[:, 0, 0, :2] = pts_2d
    mano_trans = np.zeros((n, 2, 3), np.float64)
    mano_pose = np.zeros((n, 2, 48), np.float64)

    joints_world, reproj, _alignment = backproject_to_world(
        mano_trans, mano_pose, joints_2d, depth, K, w2c,
        joint_source="joints_2d",
    )
    # Hand-0 joint-0 backprojects to camera frame == pts_cam
    assert np.allclose(joints_world[:, 0, 0, :], pts_cam, atol=1e-6)
    # Reprojection round-trip error < 1e-3 px
    assert np.all(reproj[:, 0] < 1e-3)
    # The joints_2d fallback path has no alignment step: alignment is always "none"
    assert (_alignment == "none").all()


def test_backproject_invalid_depth_is_nan_and_flagged():
    """In-view valid points vs zeroed depth: the former backproject normally, the latter
    NaN + large-error sentinel.

    Note (minimal correction to the task): the original only zeroed the depth of "a single
    joint", but that hand still has 20 valid joints — under the pinned semantics (untrusted
    only when valid joints < 3) that hand would not be flagged, so ``reproj[0, 0] >= 1e6``
    could never hold. Here the whole frame-0 depth is zeroed so that hand, that frame is
    genuinely untrusted; the four assertions stay as-is (the finer single-joint-failure
    semantics live in the next test).
    """
    n, h, w = 2, 48, 64
    K = np.tile(np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], np.float64), (n, 1, 1))
    w2c = np.tile(np.eye(4, dtype=np.float64), (n, 1, 1))
    depth = np.full((n, h, w), 2.0, np.float32)
    # In-view pixel: u = w/2 + w*0.1/2, v = h/2 + w*(-0.05)/2
    u, v = w / 2 + w * 0.1 / 2.0, h / 2 + w * (-0.05) / 2.0
    joints_2d = np.zeros((n, 2, 21, 3), np.float64)
    joints_2d[:, 0, 0, 0] = u
    joints_2d[:, 0, 0, 1] = v
    depth[0] = 0.0  # frame-0 depth entirely invalid (valid joints < 3 -> that hand, that frame untrusted)
    joints_world, reproj, _alignment = backproject_to_world(
        np.zeros((n, 2, 3)), np.zeros((n, 2, 48)), joints_2d, depth, K, w2c,
        joint_source="joints_2d",
    )
    assert np.isnan(joints_world[0, 0, 0, 0])    # invalid depth -> NaN
    assert not np.isnan(joints_world[1, 0, 0, 0])  # valid depth normal
    assert reproj[0, 0] >= 1e6  # flagged as large error
    assert reproj[1, 0] < 1e-3


def test_backproject_single_invalid_joint_is_nan_but_hand_still_trusted():
    """Fine-grained contract for a single joint's depth failure: that joint is NaN, the rest
    stay trusted (the hand is not frame-flagged)."""
    n, h, w = 2, 48, 64
    K = np.tile(np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], np.float64), (n, 1, 1))
    w2c = np.tile(np.eye(4, dtype=np.float64), (n, 1, 1))
    depth = np.full((n, h, w), 2.0, np.float32)
    u, v = w / 2 + w * 0.1 / 2.0, h / 2 + w * (-0.05) / 2.0
    joints_2d = np.zeros((n, 2, 21, 3), np.float64)
    joints_2d[:, 0, 0, 0] = u
    joints_2d[:, 0, 0, 1] = v
    # Zero only the depth at joint-0's sampling site (the bilinear 2x2 neighborhood)
    depth[:, int(v):int(v) + 2, int(u):int(u) + 2] = 0.0
    joints_world, reproj, _alignment = backproject_to_world(
        np.zeros((n, 2, 3)), np.zeros((n, 2, 48)), joints_2d, depth, K, w2c,
        joint_source="joints_2d",
    )
    assert np.isnan(joints_world[:, 0, 0, :]).all()      # that joint's depth invalid -> NaN
    assert not np.isnan(joints_world[:, 0, 5, :]).any()  # the rest stay valid
    assert (reproj[:, 0] < 1e-3).all()                   # hand still trusted -> real residual ≈ 0


def test_backproject_roundtrip_with_nonidentity_camera_pose():
    """Non-trivial pose: the round trip stays < 1e-3 px with a rotation+translation w2c."""
    n, h, w = 1, 48, 64
    K = np.tile(np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], np.float64), (n, 1, 1))
    ang = 0.35
    rot = np.array([[np.cos(ang), 0, np.sin(ang)], [0, 1, 0], [-np.sin(ang), 0, np.cos(ang)]])
    w2c = np.tile(np.eye(4, dtype=np.float64), (n, 1, 1))
    w2c[0, :3, :3] = rot
    w2c[0, :3, 3] = np.array([0.2, -0.1, 0.5])
    c2w = np.linalg.inv(w2c[0])
    # Known camera-frame point -> known world-frame point (ground truth)
    pts_cam = np.array([0.3, -0.2, 3.0], np.float64)
    world_true = (c2w @ np.append(pts_cam, 1.0))[:3]
    uvw = K[0] @ pts_cam
    uv = (uvw[:2] / uvw[2])
    # The depth map holds the true depth at that pixel (constant map -> bilinear sample = value)
    depth = np.full((n, h, w), pts_cam[2], np.float32)
    joints_2d = np.zeros((n, 2, 21, 3), np.float64)
    joints_2d[0, 0, 0, :2] = uv
    joints_world, reproj, _alignment = backproject_to_world(
        np.zeros((n, 2, 3)), np.zeros((n, 2, 48)), joints_2d, depth, K, w2c,
        joint_source="joints_2d",
    )
    assert np.allclose(joints_world[0, 0, 0, :], world_true, atol=1e-6)
    assert reproj[0, 0] < 1e-3


def test_run_fusion_reprojection_error_is_real(tmp_path):
    """End to end: run_fusion's reprojection_error is no longer an all-zero placeholder.

    Note (minimal correction to the task): in synthetic data with self-consistent
    K/w2c/depth, "backprojection -> reprojection" is a pixel-level identity round trip
    (K·K⁻¹·(u,v,1)·z/(z) ≡ (u,v), independent of z and pose), so the error is always 0 under
    any self-consistent geometry — the original assertion ``reproj[:, 0].max() > 1.0`` could
    not hold against the actual algorithm. Here hand 0's joints fall outside the image
    (depth sampling entirely invalid) to genuinely trigger the "untrusted frame" output path
    (writing 1e6, not placeholder 0), and a new hand 1 (an off-center in-view point) asserts
    that the written value is a genuinely computed round-trip residual.
    """
    from roboray.runtime.episode import EpisodeWorkspace

    ws = EpisodeWorkspace(tmp_path, "ep_0001")
    n, h, w, depth_m = 4, 48, 64, 2.0
    K = np.tile(np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], np.float64), (n, 1, 1))
    ws.write_artifact(
        "scene.npz",
        {
            "camera_poses": np.tile(np.eye(4, dtype=np.float64), (n, 1, 1)),
            "intrinsics": K,
            "depth": np.full((n, h, w), depth_m, np.float32),
            "depth_source": np.array("monocular"),
            "n_frames": np.int32(n),
            "fps": np.int32(30),
        },
    )
    present = np.ones((n, 2), bool)
    joints_2d = np.zeros((n, 2, 21, 3), np.float64)
    # hand 0: all joints outside the image -> depth invalid -> untrusted (must write the 1e6
    # sentinel, not placeholder 0)
    joints_2d[:, 0, :, 0] = w + 50.0
    joints_2d[:, 0, :, 1] = h / 2
    # hand 1: an off-center in-view point -> trusted (must write a genuinely computed
    # round-trip residual)
    joints_2d[:, 1, 0, 0] = w / 2 + 10
    joints_2d[:, 1, 0, 1] = h / 2
    ws.write_artifact(
        "hand.npz",
        {
            "hand_present": present,
            "joints_2d": joints_2d,
            "mano_pose": np.zeros((n, 2, 48), np.float32),
            "mano_shape": np.zeros((n, 2, 10), np.float32),
            "mano_trans": np.zeros((n, 2, 3), np.float32),
            "confidence": np.where(present, 0.9, 0.0).astype(np.float32),
            "n_frames": np.int32(n),
            "fps": np.int32(30),
        },
    )
    from roboray.stages.fusion import run_fusion

    run_fusion(workspace_root=str(tmp_path), episode_id="ep_0001",
               wearer_association=True, unobserved_gap_s=0.2, min_supporting_joints=8)
    z = np.load(ws.artifact("fused.npz"), allow_pickle=True)
    reproj = z["reprojection_error"]
    assert reproj.shape == (n, 2)
    assert np.abs(reproj[:, 0]).max() > 1.0  # untrusted hand flagged (not an all-zero placeholder)
    assert reproj[:, 1].max() < 1e-3  # trusted hand writes the genuinely computed round-trip residual


# ---------------------------------------------------------------------------
# mano path (informative reprojection error): 3D joints from the hand model, 2D detections
# from image evidence
# ---------------------------------------------------------------------------


def _mano_analytic_setup(n=1, h=48, w=64, depth_m=2.0, offset_m=0.05):
    """Build an analytically solvable example for the mano path.

    Detected 2D joints (u_j, v_j) fall in view; MANO camera-frame joints = 3D points
    consistent with the depth, translated as a whole by ``mano_origin_shift`` (simulating a
    real hand model whose origin does not lie on the depth plane, making "translation
    alignment" a genuinely active step), then a fixed camera-frame x offset ``offset_m`` is
    applied to j>=1 (structural divergence). After alignment, hand 1 (no structural
    divergence) agrees with the image evidence; hand 0's pixel deviation is solvable
    analytically: ``err_j = f_x * offset_m / z`` (exact under identity pose and constant depth).
    """
    K = np.tile(np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], np.float64), (n, 1, 1))
    w2c = np.tile(np.eye(4, dtype=np.float64), (n, 1, 1))
    depth = np.full((n, h, w), depth_m, np.float32)
    uv = np.zeros((n, 2, 21, 2), np.float64)
    for j in range(21):
        uv[:, :, j, 0] = 16.0 + 1.5 * j  # 16..46, all in [0, w-1)
        uv[:, :, j, 1] = 20.0 + 0.5 * j  # 20..30, all in [0, h-1)
    joints_2d = np.zeros((n, 2, 21, 3), np.float64)
    joints_2d[:, :, :, :2] = uv
    Kinv = np.linalg.inv(K[0])
    consistent = np.stack(
        [Kinv @ np.array([uv[0, 0, j, 0], uv[0, 0, j, 1], 1.0]) * depth_m for j in range(21)]
    )
    shift = np.array([0.7, -0.4, 0.3], np.float64)  # MANO origin ≠ depth-plane origin
    joints_3d_cam = np.tile(consistent + shift, (n, 2, 1, 1))
    joints_3d_cam[:, 0, 1:, 0] += offset_m  # hand 0: camera-frame x offset on j>=1
    mano_trans = joints_3d_cam[:, :, 0].copy()  # hand-model contract: root = mano_trans
    mano_pose = np.zeros((n, 2, 48), np.float64)
    return K, w2c, depth, joints_2d, joints_3d_cam, mano_trans, mano_pose, consistent, offset_m


def test_backproject_mano_source_reproj_equals_analytic_pixel_error():
    """mano path: reprojection error = analytically computable pixel deviation (non-zero),
    and translation alignment anchors the root node."""
    n, w, depth_m, offset = 1, 64, 2.0, 0.05
    K, w2c, depth, joints_2d, joints_3d_cam, mano_trans, mano_pose, consistent, _ = (
        _mano_analytic_setup(n, offset_m=offset)
    )
    joints_world, reproj, _alignment = backproject_to_world(
        mano_trans, mano_pose, joints_2d, depth, K, w2c,
        joint_source="mano", joints_3d_cam=joints_3d_cam,
    )
    # Precondition: the MANO origin indeed differs from the depth-plane origin (the alignment
    # step is load-bearing)
    assert not np.allclose(joints_3d_cam[0, 0, 0], consistent[0], atol=1e-3)
    # Translation alignment: root lands on the metric position given by detected 2D + depth
    # (identity pose -> camera frame = world frame)
    assert np.allclose(joints_world[0, 0, 0], consistent[0], atol=1e-9)
    # Other joints carry a known camera-frame offset e: pixel deviation = f_x * e / z (f_x = w)
    expected_px = w * offset / depth_m
    expected_mean = 20 * expected_px / 21  # joint 0 has no deviation, the other 20 each expected_px
    assert abs(reproj[0, 0] - expected_mean) < 1e-6
    assert reproj[0, 0] > 1.0  # informative: divergence shows up directly in pixels
    # Joint 1's world position = consistent point + offset (alignment moves the MANO origin
    # onto the depth plane)
    assert np.allclose(joints_world[0, 0, 1], consistent[1] + [offset, 0.0, 0.0], atol=1e-9)
    # Hand 1 is structurally consistent -> after alignment it agrees with the image evidence:
    # reprojection error ≈ 0
    assert np.allclose(joints_world[0, 1, 5], consistent[5], atol=1e-9)
    assert reproj[0, 1] < 1e-3
    # Alignment method exposed: constant depth plane -> target points coplanar -> Kabsch
    # rank-deficient -> per-hand fallback to root translation (the pre-change behavior, so
    # the analytic assertions above still hold unchanged)
    assert list(_alignment[0]) == ["root_only", "root_only"]


def test_backproject_mano_untrusted_when_root_depth_invalid():
    """mano-path untrusted semantics: invalid root depth -> all-NaN + sentinel (NaN ->
    untrusted, as before)."""
    n, depth_m = 1, 2.0
    K, w2c, depth, joints_2d, joints_3d_cam, mano_trans, mano_pose, _, _ = _mano_analytic_setup(n)
    depth[0] = 0.0  # depth invalid at the root-node pixel
    joints_world, reproj, _alignment = backproject_to_world(
        mano_trans, mano_pose, joints_2d, depth, K, w2c,
        joint_source="mano", joints_3d_cam=joints_3d_cam,
    )
    assert np.isnan(joints_world[0, 0]).all()  # cannot be metrically anchored -> that hand, that frame untrusted
    assert reproj[0, 0] == UNTRUSTED_REPROJ_ERROR

    # 3D joints containing NaN (the model produced no such hand) is likewise untrusted
    joints_3d_cam[0, 1] = np.nan
    joints_world2, reproj2, _alignment2 = backproject_to_world(
        mano_trans, mano_pose, joints_2d, np.full((n, 48, 64), depth_m, np.float32), K, w2c,
        joint_source="mano", joints_3d_cam=joints_3d_cam,
    )
    assert np.isnan(joints_world2[0, 1]).all()
    assert reproj2[0, 1] == UNTRUSTED_REPROJ_ERROR


def _write_fusion_inputs(ws, n, joints_2d, joints_3d_cam=None, h=48, w=64, depth_m=2.0):
    """Write scene.npz + hand.npz (with joints_3d_cam=None, use the fallback contract)."""
    K = np.tile(np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], np.float64), (n, 1, 1))
    ws.write_artifact(
        "scene.npz",
        {
            "camera_poses": np.tile(np.eye(4, dtype=np.float64), (n, 1, 1)),
            "intrinsics": K,
            "depth": np.full((n, h, w), depth_m, np.float32),
            "depth_source": np.array("monocular"),
            "n_frames": np.int32(n),
            "fps": np.int32(30),
        },
    )
    present = np.ones((n, 2), bool)
    hand = {
        "hand_present": present,
        "joints_2d": joints_2d,
        "mano_pose": np.zeros((n, 2, 48), np.float32),
        "mano_shape": np.zeros((n, 2, 10), np.float32),
        "mano_trans": np.zeros((n, 2, 3), np.float32),
        "confidence": np.where(present, 0.9, 0.0).astype(np.float32),
        "n_frames": np.int32(n),
        "fps": np.int32(30),
    }
    if joints_3d_cam is not None:
        hand["joints_3d_cam"] = joints_3d_cam
    ws.write_artifact("hand.npz", hand)
    return ws


def test_run_fusion_uses_mano_source_and_reports_metric(tmp_path):
    """With joints_3d_cam -> run_fusion takes the mano path, and records/returns the metric
    source and statistics."""
    from roboray.runtime.episode import EpisodeWorkspace
    from roboray.stages.fusion import run_fusion

    n = 3
    ws = EpisodeWorkspace(tmp_path, "ep_0001")
    _, _, _, joints_2d, joints_3d_cam, _, _, _, _ = _mano_analytic_setup(n)
    _write_fusion_inputs(ws, n, joints_2d, joints_3d_cam)

    info = run_fusion(workspace_root=str(tmp_path), episode_id="ep_0001",
                      wearer_association=True, unobserved_gap_s=0.2, min_supporting_joints=8)
    assert info["joint_source"] == "mano"
    z = np.load(ws.artifact("fused.npz"), allow_pickle=True)
    assert str(z["joint_source"]) == "mano"
    # Per-hand alignment method in the npz (optional key). Constant depth plane -> Kabsch
    # rank-deficient -> all root_only
    alignment = z["alignment"]
    assert alignment.shape == (n, 2)
    assert (alignment == "root_only").all()
    reproj = z["reprojection_error"]
    assert np.isfinite(reproj).all()  # sentinel is a finite large number, not NaN
    # Hand 0 has real divergence (> 1 px), hand 1 is self-consistent with the evidence (≈0)
    assert reproj[:, 0].min() > 1.0
    assert reproj[:, 1].max() < 1e-3
    # Statistics include only trusted hand frames, and agree with an independent recompute
    trusted = reproj[reproj < UNTRUSTED_REPROJ_ERROR]
    assert info["flagged_hands"] == int((reproj >= UNTRUSTED_REPROJ_ERROR).sum())
    assert info["reprojection_error_mean_px"] == pytest.approx(float(trusted.mean()))
    assert info["reprojection_error_max_px"] == pytest.approx(float(trusted.max()))


def test_run_fusion_falls_back_to_joints_2d_and_flags_are_counted(tmp_path):
    """Missing joints_3d_cam -> fallback round-trip path (source labeled joints_2d), with
    untrusted frames counted into flagged_hands."""
    from roboray.runtime.episode import EpisodeWorkspace
    from roboray.stages.fusion import run_fusion

    n, h, w = 2, 48, 64
    ws = EpisodeWorkspace(tmp_path, "ep_0002")
    joints_2d = np.zeros((n, 2, 21, 3), np.float64)
    joints_2d[:, 0, :, 0] = w + 50.0  # hand 0 entirely outside the image -> untrusted
    joints_2d[:, 0, :, 1] = h / 2
    joints_2d[:, 1, 0, 0] = w / 2 + 10
    joints_2d[:, 1, 0, 1] = h / 2
    _write_fusion_inputs(ws, n, joints_2d)

    info = run_fusion(workspace_root=str(tmp_path), episode_id="ep_0002",
                      wearer_association=True, unobserved_gap_s=0.2, min_supporting_joints=8)
    assert info["joint_source"] == "joints_2d"
    z = np.load(ws.artifact("fused.npz"), allow_pickle=True)
    assert str(z["joint_source"]) == "joints_2d"
    # The fallback path has no alignment step: alignment is always "none"
    assert (z["alignment"] == "none").all()
    reproj = z["reprojection_error"]
    assert (reproj[:, 0] == UNTRUSTED_REPROJ_ERROR).all()  # hand 0 untrusted
    assert reproj[:, 1].max() < 1e-3  # hand 1 round trip -> always zero (structural fact of that path)
    assert info["flagged_hands"] == n
    assert info["reprojection_error_max_px"] < 1e-3


def test_run_fusion_with_fake_hand_model_reports_mano_source(tmp_path):
    """Stand-in chain: FakeHandModel produces joints_3d_cam -> fusion automatically takes
    the mano path and reports honestly."""
    from roboray.runtime.episode import EpisodeWorkspace
    from tests.fakes.models import FakeHandModel, FakeSceneModel
    from roboray.stages.fusion import run_fusion

    n = 8
    ws = EpisodeWorkspace(tmp_path, "ep_0003")
    frames = np.zeros((n, 48, 64, 3), np.uint8)
    calib = {"fps": 30, "has_stereo": False}
    FakeSceneModel(seconds=0.0).infer({"frames": frames, "episode_id": "ep_0003"}, calib,
                                      ws.artifact("scene.npz"))
    FakeHandModel(seconds=0.0).infer({"frames": frames, "episode_id": "ep_0003"}, calib,
                                     ws.artifact("hand.npz"))
    # Stand-in contract: 3D joints present with root = mano_trans, left/right distinguishable
    with np.load(ws.artifact("hand.npz"), allow_pickle=False) as z:
        j3d = z["joints_3d_cam"]
        mano_trans = z["mano_trans"]
    assert j3d.shape == (n, 2, 21, 3)
    assert np.allclose(j3d[:, :, 0], mano_trans, atol=1e-6)
    assert not np.allclose(j3d[:, 0], j3d[:, 1])

    info = run_fusion(workspace_root=str(tmp_path), episode_id="ep_0003",
                      wearer_association=True, unobserved_gap_s=0.2, min_supporting_joints=8)
    assert info["joint_source"] == "mano"
    z = np.load(ws.artifact("fused.npz"), allow_pickle=True)
    reproj = z["reprojection_error"]
    assert np.isfinite(reproj).all()
    # Trusted + untrusted = all hand frames (the stand-in's 2D detections are mostly off-frame
    # -> untrusted frames occur)
    trusted = int((reproj < UNTRUSTED_REPROJ_ERROR).sum())
    assert trusted + info["flagged_hands"] == n * 2
    assert info["flagged_hands"] > 0
    # Filled-frame joint interpolation does not crash under NaN semantics and length stays
    assert z["joints_world"].shape == (n, 2, 21, 3)


def test_publish_qc_stats_exclude_flagged_hands(tmp_path):
    """QC: mean/max only cover trusted hand frames; untrusted frames are excluded by the
    sentinel and reported separately."""
    from roboray.runtime.episode import EpisodeWorkspace, Stage
    from roboray.stages.publish import run_publish
    from roboray.stages.artifacts import save_artifact

    ws = EpisodeWorkspace(tmp_path / "ws", "ep_0001")
    ws.mark_done(Stage.ACTION, 1.0)
    save_artifact(
        ws.artifact("fused.npz"),
        joints_world=np.zeros((3, 2, 21, 3), np.float32),
        reprojection_error=np.array(
            [[0.5, UNTRUSTED_REPROJ_ERROR], [1.5, 2.0], [UNTRUSTED_REPROJ_ERROR, 0.0]], np.float32
        ),
        hand_ids=np.zeros((3, 2), np.int32),
        joint_source=np.array("mano"),
        depth_source=np.array(["monocular"] * 3, dtype=object),
        identity_switches=np.zeros((0, 2), np.int32),
        gapfill_filled=np.int32(0),
        gapfill_rejected=np.int32(0),
        n_frames=np.int32(3),
        fps=np.int32(30),
    )
    save_artifact(
        ws.artifact("action.npz"),
        labels=np.array(["reach"], dtype=object),
        segment_start=np.array([0], np.int32),
        segment_end=np.array([3], np.int32),
        state=np.zeros((3, 2, 21, 3), np.float32),
        action=np.zeros((3, 2, 21, 3), np.float32),
        n_frames=np.int32(3),
        fps=np.int32(30),
    )
    info = run_publish(
        workspace_root=str(tmp_path / "ws"), output_root=str(tmp_path / "out"),
        episode_id="ep_0001", make_visualization=False,
    )
    # Trusted values = [0.5, 1.5, 2.0, 0.0] -> mean 1.0, max 2.0; the two 1e6 sentinels excluded
    assert info["flagged_hands"] == 2
    assert info["reprojection_error_mean_px"] == pytest.approx(1.0)
    assert info["reprojection_error_max_px"] == pytest.approx(2.0)
    assert info["joint_source"] == "mano"
    html = (tmp_path / "out" / "qc" / "ep_0001" / "report.html").read_text()
    # Statistics exclude sentinel values (mean/max rows only show trusted values), and the
    # metric source + untrusted count are explicitly labeled
    assert "mean 1.0000, max 2.0000" in html
    assert "untrusted hand frames 2" in html
    assert "Metric source: mano" in html


def test_publish_qc_report_renders_gapfill_audit(tmp_path):
    """The QC report's "gap-fill audit" section renders the A3 rejection-reason
    distribution and trajectory-check skipped counts.

    Previously these existed only in run_fusion's return value (discarded by the Actor) and
    never landed on any persistent surface; they are now persisted via fused.npz and
    readable in QC.
    """
    from roboray.runtime.episode import EpisodeWorkspace, Stage
    from roboray.stages.publish import run_publish
    from roboray.stages.artifacts import save_artifact

    ws = EpisodeWorkspace(tmp_path / "ws", "ep_0001")
    ws.mark_done(Stage.ACTION, 1.0)
    save_artifact(
        ws.artifact("fused.npz"),
        joints_world=np.zeros((3, 2, 21, 3), np.float32),
        reprojection_error=np.zeros((3, 2), np.float32),
        hand_ids=np.zeros((3, 2), np.int32),
        joint_source=np.array("mano"),
        depth_source=np.array(["monocular"] * 3, dtype=object),
        identity_switches=np.zeros((0, 2), np.int32),
        gapfill_filled=np.int32(2),
        gapfill_rejected=np.int32(3),
        gapfill_reasons=np.array(
            json.dumps({"insufficient_keypoint_depth": 2, "out_of_view": 1})
        ),
        gapfill_skipped=np.array(json.dumps({"sequence_boundary": 2})),
        n_frames=np.int32(3),
        fps=np.int32(30),
    )
    save_artifact(
        ws.artifact("action.npz"),
        labels=np.array(["reach"], dtype=object),
        segment_start=np.array([0], np.int32),
        segment_end=np.array([3], np.int32),
        state=np.zeros((3, 2, 21, 3), np.float32),
        action=np.zeros((3, 2, 21, 3), np.float32),
        n_frames=np.int32(3),
        fps=np.int32(30),
    )
    run_publish(
        workspace_root=str(tmp_path / "ws"), output_root=str(tmp_path / "out"),
        episode_id="ep_0001", make_visualization=False,
    )
    html = (tmp_path / "out" / "qc" / "ep_0001" / "report.html").read_text()
    assert "filled 2 frames, dropped 3 frames" in html
    assert "insufficient_keypoint_depth=2" in html and "out_of_view=1" in html
    assert "sequence_boundary=2" in html


# ---------------------------------------------------------------------------
# A3 four post-fill checks (depth / view / wrist displacement / trajectory deviation) +
# supporting_joints wiring
# ---------------------------------------------------------------------------


def test_a3_all_four_checks_via_run_fusion(tmp_path):
    """A3 refinement: construct a scenario where the filled segment's trajectory deviates
    strongly -> fill rejected (audit rejected increases)."""
    from roboray.runtime.episode import EpisodeWorkspace
    from roboray.stages.fusion import run_fusion

    ws = EpisodeWorkspace(tmp_path, "ep_0001")
    n, h, w = 12, 48, 64
    ws.write_artifact(
        "scene.npz",
        {
            "camera_poses": np.tile(np.eye(4, dtype=np.float64), (n, 1, 1)),
            "intrinsics": np.tile(np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], np.float64), (n, 1, 1)),
            "depth": np.full((n, h, w), 2.0, np.float32),
            "depth_source": np.array("monocular"),
            "n_frames": np.int32(n),
            "fps": np.int32(30),
        },
    )
    present = np.ones((n, 2), bool)
    present[5:7, 0] = False  # 2-frame gap, 0.067s < 0.2s at 30fps -> duration condition met
    joints_2d = np.zeros((n, 2, 21, 3), np.float64)
    # Both ends at the same position but in a depth-invalid zone (zero the gap segment's
    # pixel depths -> the depth check rejects)
    ws_arrays = dict(np.load(ws.artifact("scene.npz"), allow_pickle=False))
    ws_arrays["depth"][5:7, int(h / 2):, :] = 0.0
    np.savez(ws.artifact("scene.npz"), **ws_arrays)
    joints_2d[:, 0, :, 0] = w / 2
    joints_2d[:, 0, :, 1] = h - 2  # lands in the lower half whose depth was zeroed
    ws.write_artifact(
        "hand.npz",
        {
            "hand_present": present,
            "joints_2d": joints_2d,
            "mano_pose": np.zeros((n, 2, 48), np.float32),
            "mano_shape": np.zeros((n, 2, 10), np.float32),
            "mano_trans": np.zeros((n, 2, 3), np.float32),
            "confidence": np.where(present, 0.9, 0.0).astype(np.float32),
            "n_frames": np.int32(n),
            "fps": np.int32(30),
        },
    )
    info = run_fusion(workspace_root=str(tmp_path), episode_id="ep_0001",
                      wearer_association=True, unobserved_gap_s=0.2, min_supporting_joints=8)
    # The depth check rejects the fill: the gap is not filled
    assert info["gapfill_filled"] == 0
    assert info["gapfill_rejected"] >= 1


def _a3_fixture(tmp_path, joints_2d, *, depth=None, confidence=None, present=None, n=12, h=48,
                w=64, gap=(5, 7), min_supporting_joints=8, unobserved_gap_s=0.2):
    """Standard A3 scenario: hand 0 missed in the gap segment; scene/hand artifacts written
    from the given depth map and 2D detections.

    ``present`` overrides the default "hand 0 missed in the gap segment" mask (for adjacent-gap
    shapes etc.); ``unobserved_gap_s=0`` disables filling (max_gap degenerates to 1), used as
    the "no-fill baseline".
    """
    from roboray.runtime.episode import EpisodeWorkspace
    from roboray.stages.fusion import run_fusion

    ws = EpisodeWorkspace(tmp_path, "ep_0001")
    if depth is None:
        depth = np.full((n, h, w), 2.0, np.float32)
    ws.write_artifact(
        "scene.npz",
        {
            "camera_poses": np.tile(np.eye(4, dtype=np.float64), (n, 1, 1)),
            "intrinsics": np.tile(
                np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], np.float64), (n, 1, 1)),
            "depth": depth,
            "depth_source": np.array("monocular"),
            "n_frames": np.int32(n),
            "fps": np.int32(30),
        },
    )
    if present is None:
        present = np.ones((n, 2), bool)
        present[gap[0]:gap[1], 0] = False
    if confidence is None:
        confidence = np.where(present, 0.9, 0.0).astype(np.float32)
    ws.write_artifact(
        "hand.npz",
        {
            "hand_present": present,
            "joints_2d": joints_2d,
            "mano_pose": np.zeros((n, 2, 48), np.float32),
            "mano_shape": np.zeros((n, 2, 10), np.float32),
            "mano_trans": np.zeros((n, 2, 3), np.float32),
            "confidence": confidence,
            "n_frames": np.int32(n),
            "fps": np.int32(30),
        },
    )
    return run_fusion(workspace_root=str(tmp_path), episode_id="ep_0001",
                      wearer_association=True, unobserved_gap_s=unobserved_gap_s,
                      min_supporting_joints=min_supporting_joints), ws


def _joints_at(n, u, v, w=64, h=48):
    joints_2d = np.zeros((n, 2, 21, 3), np.float64)
    joints_2d[..., 0] = u
    joints_2d[..., 1] = v
    return joints_2d


def _wrist_track(wrist_x_m, *, n=12, h=48, w=64, depth_m=2.0):
    """Generate hand 0's 2D detections from the given per-frame wrist world x (meters)
    (f=w, z=depth_m, cx=w/2).

    Under identity pose ``u = cx + x·f/z``; the whole hand shares one pixel (degenerate but
    sufficient to drive the checks).
    """
    u = w / 2 + np.asarray(wrist_x_m, np.float64) * w / depth_m
    joints_2d = np.zeros((n, 2, 21, 3), np.float64)
    joints_2d[..., 1] = h / 2
    joints_2d[:, 0, :, 0] = u[:, None]
    return joints_2d


def test_a3_depth_check_records_reason(tmp_path):
    """Zeroed depth in the fill region -> the depth check rejects, with the reason counted
    by name and returned via gapfill_reasons."""
    n, h, w = 12, 48, 64
    depth = np.full((n, h, w), 2.0, np.float32)
    depth[5:7, int(h / 2):, :] = 0.0  # lower half of the gap segment's depth invalid (fill pixels land there)
    info, _ = _a3_fixture(tmp_path, _joints_at(n, w / 2, h - 2), depth=depth)
    assert info["gapfill_filled"] == 0
    assert info["gapfill_reasons"] == {"insufficient_keypoint_depth": 2}


def test_run_fusion_persists_gapfill_audit_in_artifact(tmp_path):
    """gapfill_reasons / gapfill_skipped must persist into fused.npz (np.load readable)."""
    n, h, w = 12, 48, 64
    depth = np.full((n, h, w), 2.0, np.float32)
    depth[5:7, int(h / 2):, :] = 0.0
    info, ws = _a3_fixture(tmp_path, _joints_at(n, w / 2, h - 2), depth=depth)
    assert info["gapfill_reasons"] == {"insufficient_keypoint_depth": 2}
    with np.load(ws.artifact("fused.npz"), allow_pickle=False) as z:
        assert json.loads(str(z["gapfill_reasons"])) == {"insufficient_keypoint_depth": 2}
        assert json.loads(str(z["gapfill_skipped"])) == {}

    # The trajectory-check "unable to evaluate and passed" skip counts also persist
    # (sequence-boundary scenario)
    x = 0.02 * np.arange(12)
    _info2, ws2 = _a3_fixture(tmp_path / "skip", _wrist_track(x), gap=(1, 3))
    with np.load(ws2.artifact("fused.npz"), allow_pickle=False) as z:
        assert json.loads(str(z["gapfill_skipped"])) == {"sequence_boundary": 2}


def test_a3_min_supporting_joints_gates_via_confidence(tmp_path):
    """supporting_joints wiring: endpoint confidence < 0.5 -> insufficient supporting
    joints, the candidate does not enter the four checks.

    Without this wiring the scenario would be filled normally (depth and position are
    self-consistent), so this test pins the wiring itself.
    """
    n = 12
    confidence = np.ones((n, 2), np.float32) * 0.9
    confidence[[5, 6], 0] = 0.0  # the gap frames have no detection anyway
    confidence[4, 0] = 0.1       # left-endpoint observation confidence too low -> supporting joints 0 < 8
    info, _ = _a3_fixture(tmp_path, _joints_at(n, 64 / 2, 48 / 2), confidence=confidence)
    assert info["gapfill_filled"] == 0
    assert info["gapfill_reasons"].get("insufficient_supporting_joints") == 1


def test_a3_out_of_view_rejects_fill(tmp_path):
    """Interpolation-basis 2D joints out of bounds -> the view check rejects (depth support
    is still sufficient, so the reason is not depth)."""
    joints_2d = _joints_at(12, 64 / 2, 48 / 2)
    joints_2d[:, 0, 20, 0] = 64 + 5.0  # joint 20 out of bounds (the other 20 joints have full depth support)
    info, _ = _a3_fixture(tmp_path, joints_2d)
    assert info["gapfill_filled"] == 0
    assert info["gapfill_reasons"].get("out_of_view") == 2


def test_a3_wrist_step_rejects_fill(tmp_path):
    """Endpoint wrist world positions 0.875 m apart (f=64, z=2 m) -> the filled frame's
    wrist displacement exceeds the 0.25 m cap."""
    joints_2d = _joints_at(12, 64 / 2, 48 / 2)
    joints_2d[7, 0, :, 0] = 64 - 4.0  # right-endpoint x displacement (60-32)/64*2 = 0.875 m
    info, _ = _a3_fixture(tmp_path, joints_2d)
    assert info["gapfill_filled"] == 0
    assert info["gapfill_reasons"].get("wrist_step_exceeded") == 2


def test_a3_check4_rejects_on_velocity_discontinuity(tmp_path):
    """4th check (trajectory deviation = velocity continuity): slow into the gap, fast out
    -> the bridge velocity deviates from the endpoints'

    measured velocity mean beyond the cap -> rejected, and filled drops accordingly.

    gap = [5,7), endpoints 4 (left), 7 (right): v_before = x[4]-x[3] = 0.01 m/frame,
    v_after = x[8]-x[7] = 0.12 m/frame (≈3.6 m/s fast flick, still physical),
    v_bridge = (x[7]-x[4])/3 = 0 (the wrist almost stops inside the gap),
    deviation = |0 − 0.065| = 0.065 > _MAX_WRIST_VEL_MISMATCH_M_PER_FRAME (0.05).
    """
    from roboray.stages.fusion import _MAX_WRIST_VEL_MISMATCH_M_PER_FRAME as limit

    x = np.zeros(12)
    x[4] = 0.01
    x[7] = 0.01
    x[8:] = 0.13
    info, _ = _a3_fixture(tmp_path, _wrist_track(x))
    assert 0.065 > limit  # precondition: this scenario indeed crosses the threshold (else the test is meaningless)
    assert info["gapfill_filled"] == 0
    assert info["gapfill_rejected"] == 2
    assert info["gapfill_reasons"].get("trajectory_deviation_exceeded") == 2


def test_a3_check4_passes_on_constant_velocity(tmp_path):
    """Control group: constant velocity throughout -> all three velocities agree ->
    deviation 0 -> filling happens normally."""
    x = 0.02 * np.arange(12)
    info, ws = _a3_fixture(tmp_path, _wrist_track(x))
    assert info["gapfill_filled"] == 2
    assert "trajectory_deviation_exceeded" not in info["gapfill_reasons"]
    assert np.isfinite(np.load(ws.artifact("fused.npz"), allow_pickle=False)["joints_world"][5, 0]).all()


def test_a3_check4_skipped_at_sequence_boundary(tmp_path):
    """Gap touching the sequence start (gap = [1,3), no s−2 neighbor frame) -> the check
    cannot be evaluated: passed but counted by category."""
    x = 0.02 * np.arange(12)
    info, _ = _a3_fixture(tmp_path, _wrist_track(x), gap=(1, 3))
    assert info["gapfill_filled"] == 2
    assert info["gapfill_skipped"] == {"sequence_boundary": 2}
    # Non-rejection semantics: must not leak into the rejection-reason counts
    assert "trajectory_check_skipped" not in info["gapfill_reasons"]
    assert info["gapfill_reasons"] == {}


def test_a3_in_view_bound_matches_sampleable_region():
    """The view bound must match ``_sample_depth``'s sampleable region (``u < w-1``).

    Otherwise there is a seam where a joint "passes the view check yet necessarily fails at
    the depth sample": boundary pixels are judged in view but the depth sample is always NaN.
    """
    from roboray.stages.fusion import _all_in_view, _sample_depth

    w, h = 64, 48
    depth = np.full((h, w), 2.0, np.float32)
    joints = np.zeros((1, 21, 3), np.float64)
    joints[..., 1] = h / 2
    joints[..., 0] = w - 2  # the last sampleable pixel
    assert _all_in_view(joints, w, h)
    assert np.isfinite(_sample_depth(depth, w - 2, h / 2))
    joints[..., 0] = w - 1  # boundary pixel: sampling must be NaN -> the view check must also judge it out
    assert not _all_in_view(joints, w, h)
    assert not np.isfinite(_sample_depth(depth, w - 1, h / 2))


def test_a3_check4_skips_on_unobserved_neighbour(tmp_path):
    """Adjacent gaps (observation column = …F T F…): the checked gap's neighbor frame falls
    inside another gap (unobserved).

    That neighbor has no detection (zero/garbage in real data) -> it must never be treated
    as measured displacement: the check must **skip** and count ``neighbour_unobserved``,
    not pass judgment on a fake velocity.

    Construction: present[3]=False, present[5]=False (frame 4 is an isolated observation) ->
    gap [3,4)'s next_i+1 = 5 is unobserved, gap [5,6)'s prev_i−1 = 3 is unobserved.
    """
    n = 12
    present = np.ones((n, 2), bool)
    present[3, 0] = False
    present[5, 0] = False
    info, _ = _a3_fixture(tmp_path, _wrist_track(0.02 * np.arange(n)), present=present)
    assert info["gapfill_filled"] == 2  # check skipped -> passed (not a rejection)
    assert info["gapfill_skipped"] == {"neighbour_unobserved": 2}
    assert info["gapfill_reasons"] == {}


def test_a3_rejected_fill_frames_are_not_observed_for_wearer_association(tmp_path):
    """Rejected filled frames must not participate in wearer association (``effective``
    must be rebuilt after the rejection loop).

    Scenario: hand 0's detections at x=40 (right), hand 1 at x=10 (left); the gap frames'
    hand-0 2D is zero (garbage in real data), and the fill region's depth is zeroed -> the
    fill is rejected by the depth check. If ``effective`` still treated the gap frames as
    "observed", identity ranking would use the zero x to rank hand 0 leftmost, flipping
    left/right identity and fabricating identity-switch events. Assertions: the gap frames'
    hand 0 has no identity (-1), no identity switch, and is element-wise identical to the
    "filling disabled" baseline.
    """
    n, h, w = 12, 48, 64
    joints_2d = np.zeros((n, 2, 21, 3), np.float64)
    joints_2d[:, 0, :, 0] = 40.0          # hand 0 always on the right
    joints_2d[5:7, 0, :, 0] = 0.0         # gap frames: zero-value 2D (if treated as observed, ranks leftmost)
    joints_2d[:, 0, :, 1] = h - 2
    joints_2d[:, 1, :, 0] = 10.0          # hand 1 always on the left
    joints_2d[:, 1, :, 1] = h / 2
    depth = np.full((n, h, w), 2.0, np.float32)
    depth[5:7, int(h / 2):, :] = 0.0      # depth invalid at the fill pixels -> the depth check rejects

    info, ws = _a3_fixture(tmp_path, joints_2d, depth=depth)
    assert info["gapfill_filled"] == 0 and info["gapfill_rejected"] == 2
    hand_ids = np.load(ws.artifact("fused.npz"), allow_pickle=False)["hand_ids"]
    assert list(hand_ids[5]) == [-1, 0]   # hand 0 unobserved -> no identity assigned; hand 1 stays 0
    assert info["identity_switches"] == 0

    # Baseline: same input but filling disabled (max_gap degenerates to 1) -> identity
    # assignment must be identical
    base_info, base_ws = _a3_fixture(
        tmp_path / "base", joints_2d, depth=depth, unobserved_gap_s=0.0)
    assert base_info["gapfill_filled"] == 0
    base_ids = np.load(base_ws.artifact("fused.npz"), allow_pickle=False)["hand_ids"]
    assert np.array_equal(hand_ids, base_ids)


def test_a3_accepted_fill_frames_keep_stable_wearer_identity(tmp_path):
    """**Accepted** filled frames must not use zero-value 2D in wearer association
    (otherwise identity switches are fabricated).

    Scenario: hand 0 still at x=40 (right), hand 1 at x=10 (left); hand 0 missed in [5,7)
    with its detections written as zero by the script, but the fill region's depth is valid
    -> the fill is **accepted** (``gapfill_filled==2``). If identity ranking read the zero
    value 2D directly, hand 0 would temporarily rank leftmost (x=0) -> two fabricated
    identity switches. Assertions: 2 frames filled, ``identity_switches==0``, ``hand_ids``
    stable as ``[1, 0]`` throughout.
    """
    n, h, w = 12, 48, 64
    joints_2d = np.zeros((n, 2, 21, 3), np.float64)
    joints_2d[:, 0, :, 0] = 40.0        # hand 0 still on the right
    joints_2d[5:7, 0, :, 0] = 0.0       # gap frames: zero-value 2D (garbage in real data)
    joints_2d[:, 0, :, 1] = h / 2
    joints_2d[:, 1, :, 0] = 10.0        # hand 1 still on the left
    joints_2d[:, 1, :, 1] = h / 2
    info, ws = _a3_fixture(tmp_path, joints_2d)  # depth valid -> fill accepted
    assert info["gapfill_filled"] == 2
    assert info["identity_switches"] == 0
    hand_ids = np.load(ws.artifact("fused.npz"), allow_pickle=False)["hand_ids"]
    assert np.array_equal(hand_ids, np.tile([1, 0], (n, 1)))


def test_a3_filled_frames_reprojection_error_is_a_flag(tmp_path):
    """Filled frames have no observation to compare against -> the reprojection error is a
    sentinel (1e6); observed frames keep the real (round-trip) error."""
    info, ws = _a3_fixture(tmp_path, _joints_at(12, 64 / 2, 48 / 2))
    assert info["gapfill_filled"] == 2
    reproj = np.load(ws.artifact("fused.npz"), allow_pickle=False)["reprojection_error"]
    assert (reproj[5:7, 0] == UNTRUSTED_REPROJ_ERROR).all()  # the two filled frames are sentinel, not measurement
    assert reproj[4, 0] < 1e-3                               # observed frames still carry the real (≈0) error
    assert info["flagged_hands"] >= 2                        # sentinel merges into the untrusted-hand count


def test_a3_trajectory_check_helper_rejects_bridge_velocity_mismatch():
    """4th check's criteria and skip reasons (helper level): velocity discontinuity / constant
    velocity / the three skip cases."""
    from roboray.stages.fusion import (
        _MAX_WRIST_VEL_MISMATCH_M_PER_FRAME as limit,
        _trajectory_deviation,
    )

    observed = np.ones(12, bool)
    traj = np.zeros((12, 3), np.float64)
    traj[:, 2] = 2.0
    traj[4, 0], traj[7, 0], traj[8:, 0] = 0.01, 0.01, 0.13  # 0.01 in / 0 bridge / 0.12 out
    dev, skip = _trajectory_deviation(traj, 4, 7, observed)
    assert dev > limit and skip is None

    smooth = np.zeros((12, 3), np.float64)
    smooth[:, 0] = 0.02 * np.arange(12)
    smooth[:, 2] = 2.0
    assert _trajectory_deviation(smooth, 4, 7, observed)[0] < 1e-9
    # Three skip states: out of range / neighbor unobserved / wrist untrusted
    assert _trajectory_deviation(smooth, 0, 3, observed) == (None, "sequence_boundary")
    gap_after = observed.copy()
    gap_after[8] = False
    assert _trajectory_deviation(smooth, 4, 7, gap_after) == (None, "neighbour_unobserved")
    untrusted = smooth.copy()
    untrusted[3] = np.nan
    assert _trajectory_deviation(untrusted, 4, 7, observed) == (None, "wrist_untrusted")
