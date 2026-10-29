"""Build the shared input subset (3-5 short clips) for the L3 three-mode timing comparison.

Reuses `scripts/make_synthetic_ego.py::generate_dataset` to produce the **minimal
ego layout** (`rgb_left.mp4` / `rgb_right.mp4` / `calibration.json` / `meta.json`)
and also writes a run config pointing at it, so all three modes run on the
**same inputs**.

Usage:
    python scripts/make_l3_dataset.py --out /work/ego-l3 --episodes 3 --frames 30
    # also writes a config (default: <out>/l3-config.yaml)
    python scripts/make_l3_dataset.py --out /work/ego-l3 --config-out configs/l3.yaml

Honest scope: this is a **self-built synthetic subset**, not HOT3D. It is only
for **relative** timing comparisons across the three modes and for relative
improvements on this subset; it is not for reproducing paper-absolute metrics.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_synthetic_ego import generate_dataset  # noqa: E402


def _write_config(path: Path, dataset: Path, run_root: Path, extra: dict) -> None:
    cfg = {
        "input": {"uri": str(dataset)},
        "output": {"uri": str(run_root / "out"), "format": "lerobot_v3"},
        "execution": {
            "max_gpus": 1,
            "num_cpu_actors": 4,
            "max_episodes_in_flight": 2,
            "backend": "real",
        },
    }
    for key, value in extra.items():
        cfg[key] = value
    path.parent.mkdir(parents=True, exist_ok=True)
    import yaml

    path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")


#: With backend=test the three stages must explicitly point at fakes: they are
#: registered as `fake_scene`/`fake_hand`/`fake_vlm` (see tests/fakes/models.py),
#: while the stage defaults are `auto`/`hamer`/`openai_vlm` — without overriding,
#: real models would be loaded (and it crashes on ROBORAY_HAMER_WEIGHTS).
FAKE_STAGES = {
    "scene_reconstruction": {"model": "fake_scene"},
    "hand_reconstruction": {"model": "fake_hand"},
    "action_understanding": {"model": "fake_vlm"},
    "visualization": {"camera_view": True, "world_view": True},
}
FAKE_MODELS = {
    "lerobot_python": "python",
    "lerobot_script": "tests/fakes/modelenv_scripts/fake_lerobot_export.py",
}


def main() -> None:  # pragma: no cover - needs the filesystem and cv2
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="dataset output directory")
    ap.add_argument("--episodes", type=int, default=3, help="number of clips (3-5 recommended for L3)")
    ap.add_argument("--frames", type=int, default=30, help="frames per clip")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--height", type=int, default=240)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-stereo", action="store_true", help="skip the right camera (monocular path)")
    ap.add_argument("--config-out", default=None, help="where to write the run config (default <out>/l3-config.yaml)")
    ap.add_argument("--run-root", default=None, help="run_root inside the config (default <out>/../run)")
    ap.add_argument("--backend", default="real", choices=["real", "test"], help="execution.backend")
    # On the cluster these paths **must** be pinned in the config models: section:
    # resolve_env_request flattens them into ROBORAY_* env vars shipped via
    # runtime_env, and they override the Pod env.
    ap.add_argument("--vipe-python", default=None)
    ap.add_argument("--ffs-python", default=None)
    ap.add_argument("--hammer-python", default=None)
    ap.add_argument("--lerobot-python", default=None)
    ap.add_argument("--vipe-weights", default=None)
    ap.add_argument("--ffs-weights", default=None)
    ap.add_argument("--hammer-weights", default=None)
    args = ap.parse_args()

    out = Path(args.out)
    dataset = generate_dataset(
        out,
        n_episodes=args.episodes,
        n_frames=args.frames,
        fps=args.fps,
        size=(args.width, args.height),
        stereo=not args.no_stereo,
        seed=args.seed,
    )
    run_root = Path(args.run_root) if args.run_root else out.parent / "run"
    cfg_path = Path(args.config_out) if args.config_out else out / "l3-config.yaml"
    extra = {
        "execution": {"max_gpus": 1, "num_cpu_actors": 4,
                      "max_episodes_in_flight": 2, "backend": args.backend},
    }
    models = {
        key: value
        for key, value in {
            "vipe_python": args.vipe_python,
            "ffs_python": args.ffs_python,
            "hammer_python": args.hammer_python,
            "lerobot_python": args.lerobot_python,
            "vipe_weights": args.vipe_weights,
            "ffs_weights": args.ffs_weights,
            "hammer_weights": args.hammer_weights,
        }.items()
        if value
    }
    if args.backend == "test":
        # Fake path: point at fake models explicitly and let the LeRobot export run the fake script
        extra["stages"] = FAKE_STAGES
        models = {**FAKE_MODELS, **models}
    if models:
        extra["models"] = models
    _write_config(cfg_path, dataset, run_root, extra)

    n_bytes = sum(p.stat().st_size for p in dataset.rglob("*") if p.is_file())
    print(json.dumps({
        "dataset": str(dataset),
        "episodes": args.episodes,
        "frames_per_episode": args.frames,
        "size": [args.width, args.height],
        "stereo": not args.no_stereo,
        "total_bytes": n_bytes,
        "config": str(cfg_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
