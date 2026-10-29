"""Local fake-model end-to-end run: install fakes, then run once against the config.

Usage:
    python scripts/run_local_fake.py --config configs/local-fake.yaml
"""

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from roboray.api.config import ReconstructionRequest
from roboray.api.reconstruction import EgoReconstruction
from tests.fakes.models import install_fake_models


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/local-fake.yaml")
    ap.add_argument("--run-root", default=None)
    ap.add_argument("--serial-baseline", action="store_true")
    args = ap.parse_args()
    install_fake_models()
    request = ReconstructionRequest.from_dict(yaml.safe_load(Path(args.config).read_text()))
    result = EgoReconstruction(
        request, run_root=args.run_root, serial_baseline=args.serial_baseline
    ).run()
    print(f"total {result.total_seconds:.3f}s, failed {result.failed}, manifest at {result.manifest_path}")


if __name__ == "__main__":
    main()
