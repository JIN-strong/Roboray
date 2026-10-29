"""roboray command-line entry point."""

import argparse
import json
import sys
from pathlib import Path

import yaml

from roboray.api.config import ConfigError, ReconstructionRequest
from roboray.api.reconstruction import EgoReconstruction

__all__ = ["main"]


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(prog="roboray", description="Ego video to training-ready dataset pipeline")
    sub = ap.add_subparsers(dest="cmd", required=True)
    run_p = sub.add_parser("run", help="Run one data production")
    run_p.add_argument("--config", required=True, help="Path to the YAML/JSON config file")
    run_p.add_argument("--run-root", default=None, help="Workspace root directory")
    run_p.add_argument("--serial-baseline", action="store_true", help="Serial baseline mode (for comparison runs)")
    run_p.add_argument("--no-adaptive", action="store_true", help="Disable adaptive scheduling (static pool)")
    run_p.add_argument("--ray-address", default=None, help="Address of an existing Ray cluster, e.g. auto")
    args = ap.parse_args(argv)

    if args.cmd == "run":
        path = Path(args.config)
        try:
            if path.suffix in (".yaml", ".yml"):
                raw = yaml.safe_load(path.read_text())
            else:
                raw = json.loads(path.read_text())
            request = ReconstructionRequest.from_dict(raw)
        except (ConfigError, OSError, ValueError, yaml.YAMLError) as exc:
            # Config-error contract: any read/parse/validation failure is printed to
            # stderr and returns 2. json.JSONDecodeError is a ValueError subclass;
            # FileNotFoundError is an OSError subclass; yaml.YAMLError is its own
            # class, so all four types must be listed.
            print(f"config error: {exc}", file=sys.stderr)
            return 2
        result = EgoReconstruction(
            request,
            run_root=args.run_root,
            serial_baseline=args.serial_baseline,
            adaptive=not args.no_adaptive,
            ray_address=args.ray_address,
        ).run()
        print(json.dumps(
            {
                "total_seconds": round(result.total_seconds, 3),
                "failed": result.failed,
                "manifest": getattr(result, "manifest_path", None),
            },
            ensure_ascii=False,
            indent=2,
        ))
        return 1 if result.failed else 0
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
