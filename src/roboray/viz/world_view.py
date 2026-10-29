"""World view: hand skeleton + camera trajectory + downsampled point cloud + future
1s wrist trail.

matplotlib Agg (no EGL dependency); frames rendered one by one then written to mp4
via cv2.

**Coordinate-frame contract (all three data sources must be in the same frame to be
shown together)**:
- ``scene.npz``'s ``camera_poses`` are **w2c** (world->camera, see the output
  contract of ``run_vipe.py``); the camera center is the translation of
  ``inv(w2c)``. Plotting the w2c translation directly as the trajectory gives a
  mirrored/inverted path for non-identity poses.
- ``joints_world`` (fusion output) and the depth-backprojected point cloud are both
  in the **world frame**: the cloud is computed as ``K⁻¹·[u, v, 1]·z`` in the
  camera frame, then transformed to the world frame via the first frame's ``c2w``.
"""

from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

__all__ = ["render_world_view"]

_HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
               (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16),
               (0, 17), (17, 18), (18, 19), (19, 20)]


def _draw_hand(ax, pts, color):
    for a, b in _HAND_EDGES:
        if np.isfinite(pts[a]).all() and np.isfinite(pts[b]).all():
            ax.plot([pts[a][0], pts[b][0]], [pts[a][1], pts[b][1]], [pts[a][2], pts[b][2]],
                    color=color, linewidth=1.0)
    # Joint scatter: only draw joints with finite coordinates (NaN joints skipped,
    # positions are never fabricated).
    for p in pts:
        if np.isfinite(p).all():
            ax.scatter(p[0], p[1], p[2], color=color, s=4)


def _first_intrinsics(intrinsics) -> np.ndarray | None:
    """Take the first-frame intrinsics (3,3); supports both per-frame (T,3,3) and
    single-frame (3,3) layouts."""
    if intrinsics is None:
        return None
    k = np.asarray(intrinsics, np.float64)
    if k.ndim == 3:
        if k.shape[0] == 0:
            return None
        k = k[0]
    if k.shape != (3, 3) or not np.isfinite(k).all():
        return None
    return k


def _backproject_pointcloud(depth_frame, intrinsics, c2w, stride: int) -> np.ndarray:
    """Downsample-backproject the first-frame depth to the world frame (missing depth
    never fabricates 3D points).

    ``X_cam = K⁻¹ · [u, v, 1]ᵀ · z`` (a true pinhole backprojection, not pixel index
    times a constant), then transformed to the world frame via the first frame's
    ``c2w`` so it is shown together with the skeleton/camera trajectory.
    Non-finite or non-positive depths are always dropped.
    """
    depth_frame = np.asarray(depth_frame, np.float64)
    if depth_frame.ndim != 2 or depth_frame.size == 0 or intrinsics is None:
        return np.zeros((0, 3), np.float64)
    dh, dw = depth_frame.shape
    ys, xs = np.mgrid[0:dh:stride, 0:dw:stride]
    zs = depth_frame[::stride, ::stride]
    valid = np.isfinite(zs) & (zs > 0)
    if not valid.any():
        return np.zeros((0, 3), np.float64)
    u = xs[valid].astype(np.float64)
    v = ys[valid].astype(np.float64)
    z = zs[valid]
    rays = np.vstack([u, v, np.ones_like(u)])          # (3, N)
    cam_pts = (np.linalg.inv(intrinsics) @ rays) * z[None, :]  # (3, N) camera frame
    c2w = np.asarray(c2w, np.float64)
    world = (c2w[:3, :3] @ cam_pts) + c2w[:3, 3:4]     # (3, N) world frame
    return world.T


def _robust_limits(reference, cloud, margin: float = 0.1) -> dict:
    """Set the axis limits (meters) from the actual data, so a hardcoded range never
    crops a real point cloud out of the frame.

    Reference data (skeleton / camera centers) uses the 2-98th percentiles to resist
    outlier bad points; the point cloud is dense reference geometry, so its full
    extent is merged in as well — otherwise "there is data but you cannot see it".
    """
    refs = [np.asarray(a, np.float64).reshape(-1, 3) for a in reference]
    refs = [p[np.isfinite(p).all(axis=1)] for p in refs if p.size]
    ref = np.concatenate(refs, axis=0) if refs else np.zeros((0, 3), np.float64)
    cloud = np.asarray(cloud, np.float64).reshape(-1, 3)
    cloud = cloud[np.isfinite(cloud).all(axis=1)]
    if ref.shape[0] == 0 and cloud.shape[0] == 0:
        return {k: (-0.5, 0.5) for k in "xyz"}
    base = ref if ref.shape[0] else cloud
    lo = np.percentile(base, 2, axis=0)
    hi = np.percentile(base, 98, axis=0)
    if cloud.shape[0]:
        lo = np.minimum(lo, cloud.min(axis=0))
        hi = np.maximum(hi, cloud.max(axis=0))
    span = np.maximum(hi - lo, 1e-3)  # constant points (span=0) still get a readable cubic view
    lo, hi = lo - margin * span, hi + margin * span
    return {axis: (float(lo[i]), float(hi[i])) for i, axis in enumerate("xyz")}


def render_world_view(joints_world, camera_poses, depth, fps, out_path, *,
                      intrinsics=None, pointcloud_stride: int = 8,
                      future_s: float = 1.0) -> dict:
    joints_world = np.asarray(joints_world, np.float64)
    camera_poses = np.asarray(camera_poses, np.float64)
    depth = np.asarray(depth)
    t = joints_world.shape[0]
    if t == 0 or not np.isfinite(joints_world).any():
        # All untrusted: do not produce a misleading video, report the reason (and
        # remove any stale file that may exist)
        stale = Path(out_path)
        if stale.exists():
            stale.unlink()
        return {"frames": 0, "out_path": str(out_path),
                "skipped_reason": "no trusted hand joints",
                "camera_centers": np.zeros((0, 3)), "pointcloud": np.zeros((0, 3)),
                "pointcloud_skipped": "no trusted hand joints",
                "axis_limits": {}, "future_trail_drawn": 0}

    # w2c -> c2w: the trail plots the camera centers in the **world frame**; using
    # the w2c translation directly would mirror/invert the path
    if camera_poses.ndim == 3 and camera_poses.shape[0] > 0:
        c2w_all = np.linalg.inv(camera_poses)
        centers = c2w_all[:, :3, 3]
    else:
        c2w_all = None
        centers = np.zeros((0, 3), np.float64)

    # Point cloud: first-frame depth downsampled and backprojected to the world
    # frame. It needs the intrinsics (when intrinsics are missing only the cloud is
    # dropped; the skeleton/camera/future trail still render — an optional input
    # must not sink the whole view).
    cloud = np.zeros((0, 3), np.float64)
    pointcloud_skipped = None
    if c2w_all is None:
        pointcloud_skipped = "no camera poses: point cloud cannot be transformed to the world frame"
    elif depth.ndim < 3 or depth.shape[0] == 0:
        pointcloud_skipped = "no depth frames"
    elif _first_intrinsics(intrinsics) is None:
        pointcloud_skipped = "no usable intrinsics (point cloud only)"
    else:
        cloud = _backproject_pointcloud(
            depth[0], _first_intrinsics(intrinsics), c2w_all[0], pointcloud_stride
        )
        if cloud.shape[0] == 0:
            pointcloud_skipped = "first frame has no valid depth (non-finite or non-positive)"

    limits = _robust_limits([joints_world, centers], cloud)

    writer = None
    frames_written = 0
    future_trail_drawn = 0
    k = max(1, int(round(future_s * fps)))
    fig = plt.figure(figsize=(4, 3), dpi=100)
    try:
        for i in range(t):
            fig.clf()
            ax = fig.add_subplot(111, projection="3d")
            ax.set_xlim(*limits["x"])
            ax.set_ylim(*limits["y"])
            ax.set_zlim(*limits["z"])
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
            ax.set_zlabel("z (m)")
            _draw_hand(ax, joints_world[i, 0], "tab:blue")
            _draw_hand(ax, joints_world[i, 1], "tab:red")
            # Camera trajectory (centers of c2w, world frame), including the first
            # frame; a single frame is still visible via the marker
            if centers.shape[0]:
                pos = centers[:i + 1]
                ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], color="black", linewidth=1.0,
                        marker="o", markersize=2)
            # Point cloud: static world structure, drawn only on the first frame
            if i == 0 and cloud.shape[0]:
                ax.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2],
                           c=cloud[:, 2], cmap="viridis", s=1, alpha=0.4)
            # Future 1s wrist trail
            if i + k < t:
                w0 = joints_world[i:i + k + 1, 0, 0]
                if np.isfinite(w0).all():
                    ax.plot(w0[:, 0], w0[:, 1], w0[:, 2], "b--", linewidth=0.8)
                    future_trail_drawn += 1
            fig.canvas.draw()
            buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
            frame = cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)
            if writer is None:
                fh, fw = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(out_path), fourcc, fps, (fw, fh))
                if not writer.isOpened():
                    raise RuntimeError(f"cannot open VideoWriter for {out_path}")
            writer.write(frame)
            frames_written += 1
    finally:
        plt.close(fig)
        if writer is not None:
            writer.release()
    return {"frames": frames_written, "out_path": str(out_path),
            "camera_centers": centers, "pointcloud": cloud,
            "pointcloud_skipped": pointcloud_skipped,
            "axis_limits": limits, "future_trail_drawn": future_trail_drawn}
