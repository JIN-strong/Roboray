"""Generate a synthetic ego dataset with known ground truth for L1/L3 validation.

Usage:
    python scripts/make_synthetic_ego.py --out data/synthetic --episodes 3 --frames 60
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

N_JOINTS = 21
HANDS = 2
DEPTH_VALUE = 0.5  # fixed depth (meters) of the synthetic depth ground truth; see _ground_truth
STEREO_DISPARITY_PX = 5  # right-camera horizontal disparity (px) relative to left; see _episode_frames


def _write_video(path: Path, frames: list, fps: int) -> None:
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():  # pragma: no cover - depends on local codecs
        raise RuntimeError(f"cannot open VideoWriter for {path} (check opencv mp4v support)")
    try:
        for f in frames:
            writer.write(f)
    finally:
        writer.release()


def _episode_frames(n_frames: int, size: tuple, seed: int, hand_in_view: bool = True):
    rng = np.random.default_rng(seed)
    h, w = size[1], size[0]
    left, right = [], []
    # 2D hand trajectory across the frame: sinusoidal drift, mimicking a first-person hand
    t = np.linspace(0, 2 * np.pi, n_frames)
    cx = (w * (0.5 + 0.25 * np.sin(t))).astype(int)
    cy = (h * (0.55 + 0.15 * np.cos(t))).astype(int)
    for i in range(n_frames):
        base = rng.integers(30, 90, size=(h, w, 3), dtype=np.uint8)
        if hand_in_view:
            cv2.circle(base, (int(cx[i]), int(cy[i])), max(3, w // 16), (0, 255, 0), -1)
        left.append(base)
        # right eye = whole left frame shifted horizontally (simulated baseline disparity),
        # giving stereo algorithms real structure to work with
        shift = STEREO_DISPARITY_PX
        right.append(np.roll(base, -shift, axis=1))
    return left, right, (cx, cy)


def _ground_truth(n_frames: int, seed: int, size: tuple) -> dict:
    rng = np.random.default_rng(seed + 1000)
    t = np.linspace(0, 2 * np.pi, n_frames)
    joints = np.zeros((n_frames, HANDS, N_JOINTS, 3), np.float32)
    for i in range(n_frames):
        for hand in range(HANDS):
            base = np.array([0.1 * (-1) ** hand, 0.3 * np.sin(t[i]), -0.4], np.float32)
            joints[i, hand] = base + rng.normal(0, 0.004, size=(N_JOINTS, 3)).astype(np.float32)
    depth = np.full((n_frames, size[1], size[0]), DEPTH_VALUE, np.float32)
    return {"joints_world": joints, "depth": depth, "fps": np.int32(30)}


def generate_dataset(
    out_dir,
    n_episodes: int = 2,
    n_frames: int = 24,
    fps: int = 30,
    size: tuple = (96, 72),
    stereo: bool = True,
    seed: int = 0,
    hand_jitter: float = 0.0,
) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    w, h = size
    for e in range(n_episodes):
        ep_id = f"ep_{e + 1:04d}"
        ep_dir = out / ep_id
        ep_dir.mkdir(parents=True, exist_ok=True)
        left, right, _ = _episode_frames(n_frames, size, seed + e)
        _write_video(ep_dir / "rgb_left.mp4", left, fps)
        if stereo:
            _write_video(ep_dir / "rgb_right.mp4", right, fps)
        if stereo:
            # Right eye = left frame shifted horizontally by STEREO_DISPARITY_PX px; depth ground
            # truth is constant DEPTH_VALUE (Z=0.5 m), focal length f = K[0][0]. The disparity
            # equation f*B/Z = disparity yields baseline B = disparity * Z / f, so calibration and
            # pixels are exactly consistent (consumers can go disparity -> depth directly).
            f = float(w) * 0.8
            b = STEREO_DISPARITY_PX * DEPTH_VALUE / f
            t_left_to_right = np.eye(4)
            t_left_to_right[0, 3] = b
        else:
            # Monocular: no right camera, calibration needs no baseline (identity).
            t_left_to_right = np.eye(4)
        calib = {
            "K": [[float(w) * 0.8, 0.0, w / 2.0], [0.0, float(h) * 0.8, h / 2.0], [0.0, 0.0, 1.0]],
            "dist": [0.0, 0.0, 0.0, 0.0, 0.0],
            "T_left_to_right": t_left_to_right.tolist(),
            "fps": float(fps),
            "width": int(w),
            "height": int(h),
        }
        (ep_dir / "calibration.json").write_text(json.dumps(calib, ensure_ascii=False, indent=2))
        (ep_dir / "meta.json").write_text(
            json.dumps(
                {"episode_id": ep_id, "device": "synthetic", "n_frames": int(n_frames)},
                ensure_ascii=False,
                indent=2,
            )
        )
        gt = _ground_truth(n_frames, seed + e, size)
        if hand_jitter:
            gt["joints_world"] += np.random.default_rng(seed).normal(
                0, hand_jitter, gt["joints_world"].shape
            ).astype(np.float32)
        np.savez(ep_dir / "ground_truth.npz", **gt)
    return out


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=96)
    ap.add_argument("--height", type=int, default=72)
    ap.add_argument("--no-stereo", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    path = generate_dataset(
        args.out,
        n_episodes=args.episodes,
        n_frames=args.frames,
        fps=args.fps,
        size=(args.width, args.height),
        stereo=not args.no_stereo,
        seed=args.seed,
    )
    print(f"wrote synthetic dataset: {path}")


if __name__ == "__main__":  # pragma: no cover
    main()
