"""FFS fake inference script: contract identical to run_ffs.py. Fails when frames_right is missing."""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "modelenvs"))

from _common import load_input, parse_args, save_with_result  # noqa: E402


def main() -> None:
    args = parse_args("fake ffs")
    data = load_input(args.input)
    if "frames_right" not in data:
        print("missing frames_right: FFS needs stereo input", file=sys.stderr)
        sys.exit(2)
    t = int(data["n_frames"])
    # Original frame geometry (caller grid): the final depth and intrinsics geometry is always this.
    orig_h, orig_w = int(data["frames"].shape[1]), int(data["frames"].shape[2])
    h, w = orig_h, orig_w
    # Read --extra with the same structure as run_ffs.py: baseline/fx are computed by the adapter and
    # passed via --extra, so the depth values are really decided by the calib->extra->script chain
    # (not hardcoded inside the fake).
    extra = json.loads(args.extra)
    baseline = float(extra.get("baseline_m", 0.0325))
    fx = float(extra.get("fx", float(w)))
    fx_orig = fx  # unscaled intrinsics: output intrinsics use the original geometry
    # Input-resolution control isomorphic to run_ffs.py: stereo_max_image_side_px scales the stereo
    # pair and scales fx proportionally; the fake's **baseline disparity scales with it** (disparity ∝
    # resolution), so depth = fx*B/d is invariant to scaling - making run_ffs's depth invariance
    # observable on the fake. 0/missing = no scaling.
    limit = int(extra.get("stereo_max_image_side_px", 0) or 0)
    disparity_px = 5.0
    if limit > 0 and max(h, w) > limit:
        scale = limit / float(max(h, w))
        w, h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        fx = fx * w / float(orig_w)
        disparity_px = disparity_px * w / float(orig_w)
    # Model-input grid used for inference after scaling (for __RESULT__/QC verification)
    model_h, model_w = int(h), int(w)
    disparity = np.full((t, h, w), disparity_px, np.float32)
    depth = (fx * baseline / disparity).astype(np.float32)
    # Depth returns to the original frame geometry (same semantics as run_ffs.py): the model computes
    # on the scaled grid, the output must return to the caller's grid.
    if (model_h, model_w) != (orig_h, orig_w):
        depth = np.stack([
            cv2.resize(d, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            for d in depth
        ]).astype(np.float32)
    poses = np.tile(np.eye(4, dtype=np.float64), (t, 1, 1))
    # intrinsics are always the original-geometry intrinsics (consistent with the depth grid)
    intr = np.tile(np.array([[fx_orig, 0, orig_w / 2], [0, fx_orig, orig_h / 2], [0, 0, 1]], np.float64), (t, 1, 1))
    save_with_result(
        args.output,
        {
            "camera_poses": poses,
            "intrinsics": intr,
            "depth": depth,
            "depth_source": np.array("stereo"),
            "n_frames": np.int32(t),
            "fps": data["fps"],
        },
        {"ok": True, "model": "fake_ffs", "n_frames": t, "input_size": [int(model_w), int(model_h)],
         "model_input_size": [int(model_h), int(model_w)],
         "depth_grid": [int(depth.shape[1]), int(depth.shape[2])]},
    )


if __name__ == "__main__":
    main()
