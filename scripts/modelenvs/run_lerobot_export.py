"""LeRobot v3 export script (runs in the ego-lerobot env).

Uses the lerobot public API (spec §13 requirement); no handwritten file formats:
  LeRobotDataset.create(repo_id, fps, features, root=..., use_videos=False)
  -> add_frame({...,"task": ...}) -> save_episode() -> finalize()
Features: observation.state / action, both float32 flattened joint coordinates (dynamic width =
state.shape[1]; since 2026-10-26: 126 joints + 1 gripper column per hand x 2 hands = 128).
use_videos=False: this pipeline's training tensors are numerical features, no ffmpeg needed.

Verify on GPU / first real run: [verify on GPU] lerobot 0.6.1 create() argument names and root
semantics (if root must be the dataset root rather than its parent, adjust per the actual error
and record it in docs/models.md).
"""

import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import load_input, parse_args, save_with_result  # noqa: E402


def main() -> None:
    args = parse_args("LeRobot v3 export")
    data = load_input(args.input)
    extra = json.loads(args.extra)
    state = np.asarray(data["state"], np.float32)
    action = np.asarray(data["action"], np.float32)
    task = str(extra.get("task", "unknown"))
    fps = int(extra.get("fps", 30))
    root = Path(args.output)

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset  # [verify on GPU: import path]
    except ImportError as exc:
        raise RuntimeError(
            f"lerobot env unavailable (missing package {exc}); build it with deploy/modelenvs/build_lerobot.sh"
        ) from exc

    # If publish fails, the Driver re-runs this stage (spec §5 resume-after-breakpoint), but
    # lerobot's create() internally uses mkdir(exist_ok=False): a leftover directory from the
    # previous attempt makes the retry fail permanently (FileExistsError, observed). This directory
    # is owned exclusively by this stage, so clear it before writing.
    if root.exists():
        shutil.rmtree(root)

    dim = state.shape[1]
    features = {
        "observation.state": {"dtype": "float32", "shape": (dim,),
                              "names": [f"j{i}" for i in range(dim)]},
        "action": {"dtype": "float32", "shape": (dim,),
                   "names": [f"a{i}" for i in range(dim)]},
    }
    ds = LeRobotDataset.create(  # [verify on GPU: argument names (repo_id/root/use_videos)]
        repo_id=str(extra.get("repo_id", "roboray_episode")),
        fps=fps,
        features=features,
        root=root,
        use_videos=False,
    )
    for i in range(state.shape[0]):
        ds.add_frame({  # [verify on GPU: frame key names and task field requirements]
            "observation.state": state[i],
            "action": action[i],
            "task": task,
        })
    ds.save_episode()
    ds.finalize()

    save_with_result(
        args.output + ".summary.npz",
        {"n_frames": np.int32(state.shape[0])},
        {"ok": True, "dataset_dir": str(root), "n_frames": int(state.shape[0]), "n_episodes": 1},
    )


if __name__ == "__main__":
    main()
