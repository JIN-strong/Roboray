import json
import sys
from pathlib import Path

import numpy as np
import pytest

from roboray.viz.camera_view import render_camera_view
from roboray.viz.world_view import render_world_view


def _fixture_hands(t=6, h=48, w=64):
    joints_world = np.zeros((t, 2, 21, 3), np.float64)
    joints_world[:, 0, :, 0] = 0.1
    joints_world[:, 0, :, 1] = np.linspace(0, 0.2, t)[:, None]
    joints_world[:, 1, :, 0] = -0.1
    joints_world[2, 1, :, :] = np.nan  # one untrusted frame
    cam = np.tile(np.eye(4), (t, 1, 1))
    depth = np.full((t, h, w), 0.5, np.float32)
    return joints_world, cam, depth


def _playable_frames(path):
    import cv2

    cap = cv2.VideoCapture(str(path))
    n = 0
    while cap.read()[0]:
        n += 1
    cap.release()
    return n


def _synthetic_cloud_scene(t=2, h=4, w=4, f=100.0, cx=2.0, cy=2.0, z=2.0, u=1, v=1):
    """Synthetic scene with a single pixel of valid depth: known (u, v, z, K, c2w)
    -> a known world point."""
    K = np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]])
    depth = np.zeros((t, h, w), np.float32)
    depth[:, v, u] = z
    c2w = np.eye(4)
    c2w[:3, 3] = (0.5, -0.25, 0.0)
    w2c = np.tile(np.linalg.inv(c2w), (t, 1, 1))
    joints = np.zeros((t, 2, 21, 3), np.float64)
    expected = c2w[:3, :3] @ (np.linalg.inv(K) @ np.array([u, v, 1.0]) * z) + c2w[:3, 3]
    return joints, w2c, depth, np.tile(K, (t, 1, 1)), expected


def test_world_view_writes_playable_video(tmp_path):
    joints, cam, depth = _fixture_hands()
    out = tmp_path / "world.mp4"
    info = render_world_view(joints, cam, depth, 30, out)
    assert info["frames"] == 6
    assert out.exists() and out.stat().st_size > 0
    assert _playable_frames(out) == 6


def test_world_view_all_nan_is_reported_not_faked(tmp_path):
    joints = np.full((4, 2, 21, 3), np.nan)
    cam = np.tile(np.eye(4), (4, 1, 1))
    depth = np.full((4, 8, 8), 0.5, np.float32)
    out = tmp_path / "world.mp4"
    info = render_world_view(joints, cam, depth, 30, out)
    assert info["frames"] == 0
    assert info.get("skipped_reason")
    assert not out.exists()


def test_world_view_all_nan_removes_stale_output(tmp_path):
    """When everything is untrusted, stale files must be removed: a previous round's
    output must not pass as this round's success."""
    out = tmp_path / "world.mp4"
    out.write_bytes(b"stale-video-from-previous-run")
    joints = np.full((4, 2, 21, 3), np.nan)
    cam = np.tile(np.eye(4), (4, 1, 1))
    depth = np.full((4, 8, 8), 0.5, np.float32)
    info = render_world_view(joints, cam, depth, 30, out)
    assert info["frames"] == 0
    assert info.get("skipped_reason")
    assert not out.exists()


def test_world_view_camera_trail_uses_c2w_not_w2c(tmp_path):
    """scene.npz stores w2c: the camera trail must be the translation of inv(w2c)
    (world-frame camera centers).

    Plotting the w2c translation directly would give a mirrored/inverted path; a
    unit-pose fixture would not reveal that.
    """
    t = 4
    joints = np.zeros((t, 2, 21, 3), np.float64)
    c2w = np.tile(np.eye(4), (t, 1, 1))
    c2w[:, :3, 3] = (1.0, 2.0, 3.0)  # the camera's true position in the world frame
    w2c = np.linalg.inv(c2w)         # run_vipe.py's on-disk form
    depth = np.full((t, 4, 4), 0.5, np.float32)
    info = render_world_view(joints, w2c, depth, 30, tmp_path / "world.mp4")
    assert info["frames"] == t
    centers = info["camera_centers"]
    assert np.allclose(centers, [[1.0, 2.0, 3.0]] * t)
    # Regression guard: the w2c translation is the negation of c2w's translation,
    # never the camera center
    assert not np.allclose(centers[-1], w2c[-1, :3, 3])


def test_world_view_pointcloud_is_backprojected_to_world(tmp_path):
    """Point cloud = K⁻¹·[u,v,1]·z then transformed to the world frame via the first
    frame's c2w (not pixel index times a constant)."""
    joints, w2c, depth, K, expected = _synthetic_cloud_scene()
    info = render_world_view(joints, w2c, depth, 30, tmp_path / "world.mp4",
                             intrinsics=K, pointcloud_stride=1)
    cloud = info["pointcloud"]
    assert cloud.shape == (1, 3)  # only 1 valid depth pixel; the rest are 0-depth and dropped
    assert np.allclose(cloud[0], expected, atol=1e-9)


def test_world_view_axis_limits_contain_pointcloud(tmp_path):
    """Axis limits are data-driven (not hardcoded): the point cloud must fit entirely
    within the reported range, not be cropped."""
    joints, w2c, depth, K, expected = _synthetic_cloud_scene()
    info = render_world_view(joints, w2c, depth, 30, tmp_path / "world.mp4",
                             intrinsics=K, pointcloud_stride=1)
    cloud, lim = info["pointcloud"], info["axis_limits"]
    assert cloud.shape[0] > 0
    for i, axis in enumerate("xyz"):
        lo, hi = lim[axis]
        assert lo <= expected[i] <= hi, f"{axis} axis cropped the point cloud: {lo} > {cloud[0, i]} > {hi}"


def test_world_view_without_intrinsics_skips_only_pointcloud(tmp_path):
    """Missing intrinsics drop only the point cloud: the skeleton/camera trail still
    render, and the point-cloud skip reason is recorded honestly."""
    joints, cam, depth = _fixture_hands()
    out = tmp_path / "world.mp4"
    info = render_world_view(joints, cam, depth, 30, out)
    assert info["frames"] == 6 and _playable_frames(out) == 6
    assert info["pointcloud"].shape[0] == 0
    assert info["pointcloud_skipped"]
    assert "skipped_reason" not in info  # this is a world-view "degradation", not an "overall skip"


def test_world_view_draws_future_wrist_trail(tmp_path):
    """The future 1s wrist trail needs a long-enough clip to appear (a short fixture
    never triggers it)."""
    t, fps, future_s = 40, 30, 1.0
    joints = np.zeros((t, 2, 21, 3), np.float64)
    joints[:, 0, :, 1] = np.linspace(0, 0.4, t)[:, None]
    cam = np.tile(np.eye(4), (t, 1, 1))
    depth = np.full((t, 8, 8), 0.5, np.float32)
    info = render_world_view(joints, cam, depth, fps, tmp_path / "world.mp4",
                             future_s=future_s)
    k = int(round(future_s * fps))
    assert info["future_trail_drawn"] == t - k > 0


def test_camera_view_overlays_and_keeps_aspect(tmp_path):
    import cv2

    t, h, w = 4, 48, 64
    frames = np.zeros((t, h, w, 3), np.uint8)
    joints_2d = np.zeros((t, 2, 21, 3), np.float64)
    joints_2d[:, 0, :, 0] = w / 2
    joints_2d[:, 0, :, 1] = h / 2
    present = np.ones((t, 2), bool)
    present[:, 1] = False
    out = tmp_path / "cam.mp4"
    info = render_camera_view(frames, joints_2d, present, 30, out)
    assert info["frames"] == t
    cap = cv2.VideoCapture(str(out))
    ok, frame = cap.read()
    cap.release()
    assert ok and frame.shape[:2] == (h, w)
    assert frame.sum() > 0  # keypoints were overlaid


def test_camera_view_no_frames_removes_stale_output(tmp_path):
    """With no frames nothing is produced and stale files are removed (otherwise an
    empty video would be treated as a successful product)."""
    out = tmp_path / "cam.mp4"
    out.write_bytes(b"stale-video-from-previous-run")
    info = render_camera_view(np.zeros((0, 8, 8, 3), np.uint8),
                              np.zeros((0, 2, 21, 3), np.float64),
                              np.zeros((0, 2), bool), 30, out)
    assert info["frames"] == 0
    assert info.get("skipped_reason")
    assert not out.exists()


# ---------------------------------------------------------------------------
# publish seam: views/QC "degrade but do not crash" contract when artifacts are missing
# ---------------------------------------------------------------------------
_FAKE_EXPORT = (Path(__file__).resolve().parents[1] / "fakes" / "modelenv_scripts"
                / "fake_lerobot_export.py")


@pytest.fixture
def _lerobot_export_env(monkeypatch):
    """publish calls the LeRobot export in a subprocess: the stand-in script follows
    the real contract (NPZ in / __RESULT__ out)."""
    monkeypatch.setenv("ROBORAY_LEROBOT_PYTHON", sys.executable)
    monkeypatch.setenv("ROBORAY_LEROBOT_SCRIPT", str(_FAKE_EXPORT))


def _write_min_workspace(tmp_path, *, frames_t=4, fused=True, scene=True, hand=True,
                         scene_intrinsics=True):
    """Minimal workspace: state.json + action.npz (required) + optional viz/QC
    artifacts."""
    from roboray.stages.artifacts import save_artifact

    t, h, w = 4, 16, 16
    ws = tmp_path / "ws" / "ep_0001"
    (ws / "artifacts").mkdir(parents=True)
    (ws / "prepared").mkdir(parents=True)
    (ws / "state.json").write_text(json.dumps({"timings": {"scene": 0.1, "publish": 0.2}}))
    np.save(ws / "prepared" / "frames_left.npy", np.zeros((frames_t, h, w, 3), np.uint8))
    if hand:
        joints_2d = np.zeros((t, 2, 21, 3), np.float64)
        joints_2d[..., 0] = 8.0
        joints_2d[..., 1] = 8.0
        save_artifact(ws / "artifacts" / "hand.npz", joints_2d=joints_2d,
                      hand_present=np.ones((t, 2), bool))
    if scene:
        scene_arrays = {}
        if scene_intrinsics:
            scene_arrays["intrinsics"] = np.tile(np.array([[8.0, 0.0, 8.0],
                                                           [0.0, 8.0, 8.0],
                                                           [0.0, 0.0, 1.0]]), (t, 1, 1))
        save_artifact(ws / "artifacts" / "scene.npz",
                      camera_poses=np.tile(np.eye(4), (t, 1, 1)),
                      depth=np.full((t, h, w), 0.5, np.float32),
                      **scene_arrays)
    if fused:
        save_artifact(ws / "artifacts" / "fused.npz",
                      joints_world=np.zeros((t, 2, 21, 3), np.float32),
                      reprojection_error=np.array([[0.5, 1.0]] * t, np.float32),
                      identity_switches=np.zeros((0, 2), np.int32),
                      gapfill_filled=np.int32(0), gapfill_rejected=np.int32(0),
                      joint_source=np.array("mano"))
    save_artifact(ws / "artifacts" / "action.npz",
                  labels=np.array(["reach"], dtype=object),
                  segment_start=np.array([0], np.int32),
                  segment_end=np.array([t], np.int32),
                  state=np.zeros((t, 2, 21, 3), np.float32),
                  action=np.zeros((t, 2, 21, 3), np.float32))
    return tmp_path / "ws", tmp_path / "out"


def test_publish_writes_both_views_and_qc(tmp_path, _lerobot_export_env):
    """happy path: both views and the QC report are produced, with no skipped
    records."""
    from roboray.stages.publish import run_publish

    ws, out = _write_min_workspace(tmp_path)
    info = run_publish(workspace_root=str(ws), output_root=str(out),
                       episode_id="ep_0001", fps=30)
    outputs = info["outputs"]
    assert (out / "viz" / "ep_0001" / "camera.mp4").exists()
    assert (out / "viz" / "ep_0001" / "world.mp4").exists()
    assert outputs["camera_view"].endswith("camera.mp4")
    assert outputs["world_view"].endswith("world.mp4")
    assert (out / "qc" / "ep_0001" / "report.html").exists()
    assert not {"camera_view_skipped", "world_view_skipped", "qc_skipped"} & outputs.keys()
    # When the data is complete, the statistics sections must render normally
    html = (out / "qc" / "ep_0001" / "report.html").read_text()
    assert "Gap-fill audit" in html and "Reprojection error" in html and "Identity switches" in html
    assert "mean 0.7500, max 1.0000" in html
    assert "no measured data" not in html


def test_publish_degrades_without_crashing_when_artifacts_missing(tmp_path, _lerobot_export_env):
    """Missing hand/scene/fused: views and QC all degrade and record the reasons,
    the episode does not crash.

    When both views are missing several artifacts, **all** reasons must be kept (not
    overwrite each other).
    """
    from roboray.stages.publish import run_publish

    ws, out = _write_min_workspace(tmp_path, fused=False, scene=False, hand=False)
    info = run_publish(workspace_root=str(ws), output_root=str(out),
                       episode_id="ep_0001", fps=30)
    outputs = info["outputs"]
    assert "camera_view" not in outputs and "hand.npz" in outputs["camera_view_skipped"]
    assert "world_view" not in outputs
    # Both world-view reasons must be present (one for scene.npz and one for
    # fused.npz; not just the last one)
    assert "scene.npz" in outputs["world_view_skipped"]
    assert "fused.npz" in outputs["world_view_skipped"]
    assert "qc_skipped" in outputs
    assert not (out / "viz" / "ep_0001" / "camera.mp4").exists()
    assert not (out / "viz" / "ep_0001" / "world.mp4").exists()
    # Degraded: the report is still written (including stage timings) but clearly
    # states there is no measured data; zeros are not presented as measurements
    html = (out / "qc" / "ep_0001" / "report.html").read_text()
    assert "no measured data" in html
    assert "scene" in html  # stage timings still render
    assert info["reprojection_error_mean_px"] == 0.0


def test_publish_degraded_qc_report_has_no_fabricated_stats(tmp_path, _lerobot_export_env):
    """When fused.npz is missing, the report must not show any measurement values
    fabricated from the missing data.

    "mean 0.0000 / untrusted hand frames 0 / 0 total" looks like measurements but is
    just the default of empty data: these sections must be omitted entirely, leaving
    only the missing-data note and the stage timings.
    """
    from roboray.stages.publish import run_publish

    ws, out = _write_min_workspace(tmp_path, fused=False)
    info = run_publish(workspace_root=str(ws), output_root=str(out),
                       episode_id="ep_0001", fps=30)
    assert "qc_skipped" in info["outputs"]
    html = (out / "qc" / "ep_0001" / "report.html").read_text()
    # Required: missing-data note + stage timings
    assert "no measured data" in html and "fused.npz" in html
    assert "Stage timings" in html and "publish" in html and "0.200" in html
    # Forbidden: fabricated statistics numbers and statistics sections
    assert "0.0000" not in html
    assert "untrusted hand frames" not in html
    assert "0 total" not in html
    assert "Gap-fill audit" not in html and "Reprojection error" not in html and "Identity switches" not in html


def test_publish_world_view_survives_missing_intrinsics(tmp_path, _lerobot_export_env):
    """scene.npz missing intrinsics: only the point cloud degrades, the world view
    still renders and records it honestly."""
    from roboray.stages.publish import run_publish

    ws, out = _write_min_workspace(tmp_path, scene_intrinsics=False)
    info = run_publish(workspace_root=str(ws), output_root=str(out),
                       episode_id="ep_0001", fps=30)
    outputs = info["outputs"]
    assert outputs["world_view"].endswith("world.mp4")
    assert (out / "viz" / "ep_0001" / "world.mp4").exists()
    assert "world_view_skipped" not in outputs
    assert "intrinsics" in outputs["pointcloud_skipped"]


def test_publish_records_skip_when_camera_view_has_no_frames(tmp_path, _lerobot_export_env):
    """frames_left.npy has 0 frames: the empty video is not recorded as a product,
    only the reason is; the world view is unaffected."""
    from roboray.stages.publish import run_publish

    ws, out = _write_min_workspace(tmp_path, frames_t=0)
    info = run_publish(workspace_root=str(ws), output_root=str(out),
                       episode_id="ep_0001", fps=30)
    assert "camera_view" not in info["outputs"]
    assert info["outputs"]["camera_view_skipped"] == "no frames"
    assert not (out / "viz" / "ep_0001" / "camera.mp4").exists()
    assert "world_view" in info["outputs"]
