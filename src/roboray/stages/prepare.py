"""Stage 1: unified ingestion and data preparation (decode, timeline alignment, calibration normalization)."""

import json
from pathlib import Path

import cv2
import numpy as np

__all__ = ["decode_video", "load_calibration", "run_prepare", "align_by_timestamps"]

_REQUIRED_CALIB_KEYS = ("K", "dist", "fps")


def align_by_timestamps(
    ts_l: np.ndarray, ts_r: np.ndarray, fps: float
) -> tuple[np.ndarray, np.ndarray, int]:
    """Nearest-neighbor timestamp pairing: each left frame picks its closest right frame,
    paired only when |Δt| ≤ 0.5/fps.

    Returns (left_idx, right_idx, dropped_left). Mutual nearest-neighbor is not enforced
    (repeated pairing is allowed — stereo alignment only requires matching timestamps).
    """
    tol = 0.5 / float(fps)
    order = np.argsort(np.asarray(ts_r, np.float64))
    rs = np.asarray(ts_r, np.float64)[order]
    keep_l, keep_r = [], []
    for i, t in enumerate(np.asarray(ts_l, np.float64)):
        j = np.searchsorted(rs, t)
        best, best_d = None, tol + 1.0
        for k in (j - 1, j):
            if 0 <= k < len(rs):
                d = abs(rs[k] - t)
                if d < best_d:
                    best, best_d = k, d
        if best is not None and best_d <= tol:
            keep_l.append(i)
            keep_r.append(int(order[best]))
    return (
        np.asarray(keep_l, np.int64),
        np.asarray(keep_r, np.int64),
        len(ts_l) - len(keep_l),
    )


def decode_video(path: Path) -> np.ndarray:
    """Decode an mp4 into (T,H,W,3) uint8. Empty or unreadable video raises."""
    cap = cv2.VideoCapture(str(path))
    frames = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        cap.release()
    if not frames:
        raise ValueError(f"cannot decode {path.name}: video is empty or codec unsupported ({path})")
    return np.stack(frames, axis=0)


def load_calibration(path: Path) -> dict:
    """Read and normalize calibration. Missing/incomplete files raise readable errors."""
    if not path.exists():
        raise ValueError(f"calibration.json missing: {path}")
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"calibration.json is not valid JSON ({path}): {exc}") from exc
    missing = [k for k in _REQUIRED_CALIB_KEYS if k not in raw]
    if missing:
        raise ValueError(f"calibration.json missing required fields {missing} ({path})")
    K = np.asarray(raw["K"], dtype=np.float64).reshape(3, 3)
    dist = np.asarray(raw["dist"], dtype=np.float64).ravel()
    T = np.asarray(raw.get("T_left_to_right", np.eye(4)), dtype=np.float64).reshape(4, 4)
    return {
        "K": K,
        "dist": dist,
        "T_left_to_right": T,
        "fps": float(raw["fps"]),
        "width": int(raw.get("width", 0)),
        "height": int(raw.get("height", 0)),
    }


def _read_timestamps(src: Path, left_n: int, right_n: int) -> tuple | None:
    """Read the optional ``timestamps_left/right`` (seconds, float arrays) from ``src / "meta.json"``.

    Returns None if anything is off (file/key missing, lengths != frame counts, non-numeric),
    in which case the caller falls back to the truncation path and records
    ``alignment="truncation"`` (fault-tolerant semantics).
    """
    meta_path = src / "meta.json"
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(meta, dict):
        return None
    for key in ("timestamps_left", "timestamps_right"):
        if key not in meta:
            return None
    ts_l, ts_r = meta["timestamps_left"], meta["timestamps_right"]
    if not isinstance(ts_l, list) or not isinstance(ts_r, list):
        return None
    if len(ts_l) != left_n or len(ts_r) != right_n:
        return None
    try:
        tl, tr = np.asarray(ts_l, np.float64), np.asarray(ts_r, np.float64)
    except (TypeError, ValueError):
        return None
    if tl.ndim != 1 or tr.ndim != 1:
        return None  # nested lists pass length/isfinite but searchsorted raises "object too deep"
    if not np.isfinite(tl).all() or not np.isfinite(tr).all():
        return None
    return tl, tr


def run_prepare(
    storage_root: str,
    workspace_root: str,
    episode_rel_dir: str,
    episode_id: str,
) -> dict:
    """Decode raw media and normalize it into the Episode workspace. Returns a preparation summary.

    - No timestamps: on frame-count mismatch truncate to the shorter stream and record the
      degradation (falls back to monocular when stereo is unusable),
      ``alignment="truncation"`` (current behavior).
    - meta.json timestamps: after nearest-neighbor pairing, equal left/right indices are
      synchronized pairs (transparent downstream); ``alignment="timestamps"`` with
      ``align_dropped`` recorded separately. Timestamp-dropped frames do not count as stereo
      degradation (has_stereo stays True).
    """
    src = Path(storage_root) / episode_rel_dir
    ws = Path(workspace_root) / episode_id
    prepared = ws / "prepared"
    prepared.mkdir(parents=True, exist_ok=True)

    calib = load_calibration(src / "calibration.json")
    left = decode_video(src / "rgb_left.mp4")

    degraded = False
    degrade_reason = None
    alignment = "truncation"
    align_dropped = 0
    right = None
    right_path = src / "rgb_right.mp4"
    if right_path.exists():
        right = decode_video(right_path)
        ts = _read_timestamps(src, left.shape[0], right.shape[0])
        if ts is not None:
            il, ir, n_dropped = align_by_timestamps(ts[0], ts[1], calib["fps"])
            if len(il) == 0:
                # Zero-pair guard: e.g. an offset between the two camera clocks leaves the
                # views unalignable as a stereo pair. Fall back to truncation semantics,
                # record the timestamp_no_pairs degradation, and do not emit an empty frame table.
                n = min(left.shape[0], right.shape[0])
                left, right = left[:n], right[:n]
                degraded = True
                degrade_reason = "timestamp_no_pairs"
            else:
                left, right = left[il], right[ir]
                alignment = "timestamps"
                align_dropped = int(n_dropped)
        elif right.shape[0] != left.shape[0]:
            n = min(left.shape[0], right.shape[0])
            left, right = left[:n], right[:n]
            degraded = True
            degrade_reason = "stereo_frame_mismatch"

    # Stereo is usable only when the right stream exists and no frame-count degradation
    # occurred; on frame mismatch fall back to monocular. Timestamp-dropped frames are not a
    # degradation (has_stereo stays True; only align_dropped is recorded).
    has_stereo = right is not None and not degraded

    n_frames, h, w = left.shape[0], left.shape[1], left.shape[2]
    calib.update(
        {
            "width": int(w),
            "height": int(h),
            "has_stereo": has_stereo,
            "n_frames": int(n_frames),
        }
    )

    np.save(prepared / "frames_left.npy", left)
    # Write right frames only when stereo is finally usable (no orphan frames_right.npy after degradation).
    if right is not None and has_stereo:
        np.save(prepared / "frames_right.npy", right)
    (prepared / "calib.json").write_text(json.dumps({k: v.tolist() if isinstance(v, np.ndarray) else v
                                                     for k, v in calib.items()}, ensure_ascii=False))
    summary = {
        "episode_id": episode_id,
        "n_frames": int(n_frames),
        "has_stereo": has_stereo,
        "degraded": degraded,
        "degrade_reason": degrade_reason,
        # Alignment semantics flow through stage_result/record/manifest consistently.
        # Timestamp-dropped frames are not stereo degradation (align_dropped is separate).
        "alignment": alignment,
        "align_dropped": align_dropped,
        # For driver routing: the driver does not read workspace files (unreadable on
        # multi-node setups), so these values must come back via the Actor return value.
        "width": int(w),
        "height": int(h),
    }
    (prepared / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)
    )
    return summary
