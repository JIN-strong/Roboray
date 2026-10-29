"""HaMeR fake inference script: contract identical to run_hamer.py (slots 0=left 1=right, includes a missed-detection span).

candidates protocol (isomorphic to run_hamer.py): npz key ``candidates`` = an object array (T,)
whose elements are list[dict] (bbox/side/score/joints_2d/joints_3d_cam).
**bbox and joints_2d u,v are full-image pixel coordinates** (the w/h of the frame shape (t,h,w,3);
same units as the real protocol, so the association layer's pixel thresholds stay aligned on the
fake e2e).
- Default (no extra): 2 per frame, the wearer's left and right - joints_3d_cam use the same offset
  logic as the main array, joints_2d are derived by fake-camera projection (normalized
  0.5+xy/(z+2) then multiplied by w/h), bbox = joint 2D envelope padded by 15% (same pad ratio as
  run_hamer's boxes), score 0.9 (same as the main-array confidence); consistent with the main
  array's present/stride masks - unobserved/missed frames have no candidate on that side (the
  first left/right candidate is the main-array source).
- extra ``bystander=True``: each observed frame gains a 3rd "bystander hand" (side=0, score 0.8,
  joints = wearer's left-hand joints plus a translation offset, smaller bbox with smaller y = higher
  up) - candidates only, the main array stays (T,2) untouched.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "modelenvs"))

from _common import compute_stride, load_input, parse_args, save_with_result  # noqa: E402

N_JOINTS = 21

# Bystander-hand translation offset (camera-frame meters): raise y + push z away -> projects higher (smaller v) and smaller.
_BYSTANDER_SHIFT = np.array([0.45, -0.30, 0.60], np.float64)
_BYSTANDER_SCORE = 0.8
_FAKE_CAM_DEPTH = 2.0  # fake-camera depth lift: fake z can be negative (rng); dividing directly by it would blow up


def _fake_joints_3d_cam(t: int, mano_trans: np.ndarray) -> np.ndarray:
    """Fake MANO forward kinematics: root = mano_trans, joints = root + deterministic small offsets (L/R distinguishable)."""
    base = np.zeros((N_JOINTS, 3), np.float64)
    for j in range(N_JOINTS):
        base[j] = (0.012 * j, 0.006 * (j % 3), 0.004 * j)
    mirror = np.array([-1.0, 0.5, 1.0], np.float64)  # left hand = mirror of right in x, scaled in y
    offs = np.stack([base, base * mirror])  # (2,21,3); j=0 all-zero -> root = mano_trans
    return (np.asarray(mano_trans, np.float64)[:, :, None, :] + offs[None]).astype(np.float32)


def _fake_joints_2d(j3d: np.ndarray, score: float, width: int, height: int) -> np.ndarray:
    """Fake-camera projection to **full-image pixel coordinates**: (u,v) = (0.5 + xy/(z+depth)) * (w,h),
    3rd column = score (same convention as the main array's joints_2d 3rd dim = confidence)."""
    z = j3d[..., 2] + _FAKE_CAM_DEPTH
    u = (0.5 + j3d[..., 0] / z) * float(width)
    v = (0.5 + j3d[..., 1] / z) * float(height)
    return np.stack([u, v, np.full(u.shape, score)], axis=-1).astype(np.float32)


def _fake_bbox(joints_2d: np.ndarray) -> np.ndarray:
    """bbox = joint 2D envelope + 15% pad (same pad ratio as run_hamer._boxes_from_mediapipe)."""
    u, v = joints_2d[:, 0], joints_2d[:, 1]
    pad = 0.15 * max(float(u.max() - u.min()), float(v.max() - v.min()), 1e-3)
    return np.asarray([u.min() - pad, v.min() - pad, u.max() + pad, v.max() + pad], np.float32)


def _fake_candidate(j3d: np.ndarray, side: int, score: float,
                    width: int, height: int) -> dict:
    j2d = _fake_joints_2d(j3d, score, width, height)
    return {"bbox": _fake_bbox(j2d), "side": int(side), "score": float(score),
            "joints_2d": j2d, "joints_3d_cam": j3d.astype(np.float32)}


def main() -> None:
    args = parse_args("fake hamer")
    data = load_input(args.input)
    extra = json.loads(args.extra) if args.extra else {}
    t = int(data["n_frames"])
    rng = np.random.default_rng(0)
    stride = compute_stride(
        int(data.get("fps") or 0),
        extra.get("inference_fps") or data.get("inference_fps"),
    )
    idx = np.arange(t)
    present = np.ones((t, 2), bool)
    if t >= 8:
        present[3:5, 0] = False
    if stride > 1:
        present[idx % stride != 0] = False   # non-sampled frames are not observed (short-gap completion)
    mano_trans = rng.normal(0, 0.1, (t, 2, 3)).astype(np.float32)
    joints_3d_cam = _fake_joints_3d_cam(t, mano_trans)
    frames = data["frames"]                  # (t,h,w,3): candidates project with the real frame width/height
    img_w, img_h = int(frames.shape[2]), int(frames.shape[1])
    observed = idx % stride == 0             # always true for stride=1: all-frame inference (same as the run_hamer loop)
    candidates_rows: list = []
    for i in range(t):
        frame: list = []
        if not observed[i]:
            candidates_rows.append(frame)    # non-sampled frames skip HaMeR: no candidates for the whole frame
            continue
        if present[i, 0]:                    # wearer's left hand (same source as the main array, score = confidence)
            frame.append(_fake_candidate(joints_3d_cam[i, 0], side=0, score=0.9,
                                         width=img_w, height=img_h))
        if present[i, 1]:
            frame.append(_fake_candidate(joints_3d_cam[i, 1], side=1, score=0.9,
                                         width=img_w, height=img_h))
        if extra.get("bystander"):
            frame.append(_fake_candidate(joints_3d_cam[i, 0] + _BYSTANDER_SHIFT,
                                         side=0, score=_BYSTANDER_SCORE,
                                         width=img_w, height=img_h))
        candidates_rows.append(frame)
    # object array (T,): np.array(list-of-lists) collapses into 2-D when the sublists are
    # equal-length, so assign per frame.
    candidates = np.empty(t, dtype=object)
    for i, row in enumerate(candidates_rows):
        candidates[i] = row
    save_with_result(
        args.output,
        {
            "hand_present": present,
            "joints_2d": rng.normal(0, 1, (t, 2, 21, 3)).astype(np.float32),
            # camera-frame MANO 3D joints (same field name as run_hamer.py, the A1 mano-path input)
            "joints_3d_cam": joints_3d_cam,
            "mano_pose": rng.normal(0, 0.1, (t, 2, 48)).astype(np.float32),
            "mano_shape": rng.normal(0, 0.1, (t, 2, 10)).astype(np.float32),
            "mano_trans": mano_trans,
            "confidence": np.where(present, 0.9, 0.0).astype(np.float32),
            "candidates": candidates,
            "n_frames": np.int32(t),
            "fps": data["fps"],
        },
        {"ok": True, "model": "fake_hamer", "n_frames": t,
         "n_candidates": int(sum(len(r) for r in candidates_rows))},
    )


if __name__ == "__main__":
    main()
