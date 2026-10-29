"""LeRobot export fake: writes the same directory skeleton as the real script, for local orchestration tests."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "modelenvs"))

from _common import load_input, parse_args, save_with_result  # noqa: E402


def main() -> None:
    args = parse_args("fake lerobot export")
    data = load_input(args.input)
    extra = json.loads(args.extra)
    root = Path(args.output)
    (root / "meta" / "episodes").mkdir(parents=True, exist_ok=True)
    (root / "data").mkdir(parents=True, exist_ok=True)
    n = int(data["state"].shape[0])
    (root / "meta" / "info.json").write_text(json.dumps(
        {"codebase_version": "v3.0", "fps": int(extra.get("fps", 30)),
         "features": {"observation.state": {"dtype": "float32", "shape": list(data["state"].shape[1:])},
                      "action": {"dtype": "float32", "shape": list(data["action"].shape[1:])}},
         "total_episodes": 1, "total_frames": n}, ensure_ascii=False))
    (root / "data" / "chunk-000.parquet").write_bytes(b"fake-parquet")
    (root / "meta" / "episodes" / "chunk-000.parquet").write_bytes(b"fake-parquet")
    (root / "meta" / "tasks.jsonl").write_text(json.dumps({"task": extra.get("task", "unknown")}) + "\n")
    save_with_result(args.output + ".summary.npz",
                     {"n_frames": __import__("numpy").int32(n)},
                     {"ok": True, "dataset_dir": str(root), "n_frames": n, "n_episodes": 1})


if __name__ == "__main__":
    main()
