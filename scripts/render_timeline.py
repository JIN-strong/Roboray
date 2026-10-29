#!/usr/bin/env python
"""Render cross-stage overlap evidence from manifest.json (PNG + verdict JSON + one-line summary).

Usage: python scripts/render_timeline.py <manifest.json> --out-dir <dir>
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from roboray.viz.timeline_view import render_from_manifest, summarize  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifest", help="path to the run artifact manifest.json")
    ap.add_argument("--out-dir", required=True, help="evidence output directory")
    args = ap.parse_args(argv)
    ev = render_from_manifest(args.manifest, args.out_dir)
    print(summarize(ev))
    print(f"PNG: {Path(args.out_dir) / 'timeline.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
