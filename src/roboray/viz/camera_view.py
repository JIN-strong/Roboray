"""Camera view: 2D keypoints and skeleton lines overlaid on the original frames
(original aspect ratio preserved)."""

from pathlib import Path

import cv2
import numpy as np

__all__ = ["render_camera_view"]

_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
          (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16),
          (0, 17), (17, 18), (18, 19), (19, 20)]
_COLORS = {0: (255, 128, 0), 1: (0, 128, 255)}


def render_camera_view(frames, joints_2d, present, fps, out_path) -> dict:
    frames = np.asarray(frames)
    joints_2d = np.asarray(joints_2d)
    present = np.asarray(present, bool)
    t = frames.shape[0]
    if t == 0:
        # No frames: produce nothing (and remove any stale file) so a previous round's
        # result is not mistaken for this one
        stale = Path(out_path)
        if stale.exists():
            stale.unlink()
        return {"frames": 0, "out_path": str(out_path), "skipped_reason": "no frames"}
    h, w = frames.shape[1], frames.shape[2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"cannot open VideoWriter for {out_path}")
    try:
        for i in range(t):
            frame = frames[i].copy()
            for hand in range(joints_2d.shape[1]):
                if not present[i, hand]:
                    continue
                pts = joints_2d[i, hand][:, :2]
                color = _COLORS.get(hand, (0, 255, 0))
                for a, b in _EDGES:
                    pa, pb = pts[a], pts[b]
                    if np.isfinite(pa).all() and np.isfinite(pb).all():
                        cv2.line(frame, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), color, 1)
                for p in pts:
                    if np.isfinite(p).all():
                        cv2.circle(frame, (int(p[0]), int(p[1])), 2, color, -1)
            writer.write(frame)
    finally:
        writer.release()
    return {"frames": t, "out_path": str(out_path)}
