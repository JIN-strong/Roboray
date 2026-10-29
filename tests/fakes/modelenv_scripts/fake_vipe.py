"""ViPE fake inference script: CLI/NPZ contract identical to run_vipe.py, for local orchestration tests."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "modelenvs"))

from _common import load_input, parse_args, save_with_result  # noqa: E402


def main() -> None:
    args = parse_args("fake vipe")
    data = load_input(args.input)
    t = int(data["n_frames"])
    h, w = int(data["frames"].shape[1]), int(data["frames"].shape[2])
    poses = np.tile(np.eye(4, dtype=np.float64), (t, 1, 1))  # w2c identity
    intr = np.tile(np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], np.float64), (t, 1, 1))
    depth = np.full((t, h, w), 0.5, np.float32)
    save_with_result(
        args.output,
        {
            "camera_poses": poses,
            "intrinsics": intr,
            "depth": depth,
            "depth_source": np.array("monocular"),
            "n_frames": np.int32(t),
            "fps": data["fps"],
        },
        {"ok": True, "model": "fake_vipe", "n_frames": t},
    )


if __name__ == "__main__":
    main()
