#!/usr/bin/env python
"""Real-model three-mode benchmark (spec G1): runs serial -> pipeline -> adaptive in order on one cluster.

Each mode gets its own run-root and subprocess isolation (cold-start fairness for
driver/CostModel); after the adaptive run it auto-renders the
cross-stage overlap evidence (spec G2, wired in the CLI section below).
This file has two layers: pure reporting functions (testable) + CLI orchestration (see main).
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

MODES = ("serial", "pipeline", "adaptive")
MODE_FLAGS = {
    "serial": ["--serial-baseline"],
    "pipeline": ["--no-adaptive"],
    "adaptive": [],
}


def load_run(run_root) -> dict:
    """Read <run_root>/out/manifest.json -> a uniform per-mode result; missing/corrupt -> ok=False, no raise."""
    manifest = Path(run_root) / "out" / "manifest.json"
    try:
        data = json.loads(manifest.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return {"ok": False, "error": f"manifest unreadable: {exc}", "total_seconds": None,
                "failed": None, "n_episodes": None, "snapshots": 0, "load_events": 0,
                "timeline": []}
    summary = data.get("summary", {})
    snaps = data.get("scheduler_snapshots") or []
    return {"ok": True, "error": None,
            "total_seconds": summary.get("total_seconds"),
            "failed": summary.get("failed"),
            "n_episodes": summary.get("n_episodes"),
            "snapshots": len(snaps),
            # timeline "load" event count = model loads for this mode (spec §3.3 cold-start evidence).
            # serial/static-pipeline have no scheduler snapshots; only timeline holds for all three modes.
            "load_events": sum(1 for e in (data.get("timeline") or [])
                               if e.get("event") == "load"),
            "timeline": data.get("timeline") or []}


def aggregate(runs: dict) -> dict:
    """Per-mode raw results -> report dict; serial is its own baseline (ratio None) and unavailable is always None (no fake ratios)."""
    out = {"modes": {}}
    serial = runs.get("serial") or {}
    base = serial.get("total_seconds") if serial.get("ok") else None
    for mode in MODES:
        r = runs.get(mode) or {"ok": False, "error": "not run"}
        entry = dict(r)
        if mode == "serial" or not (base and r.get("ok") and r.get("total_seconds")):
            entry["speedup_vs_serial"] = None
        else:
            entry["speedup_vs_serial"] = round(base / r["total_seconds"], 4)
        out["modes"][mode] = entry
    return out


def render_report(agg: dict, action_model: str = "none") -> str:
    """Markdown three-mode table + methodology note (recorded verbatim)."""
    lines = [
        "| mode | total_seconds | vs serial | failed | episodes | scheduler snapshots | load events |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for mode in MODES:
        m = agg["modes"].get(mode) or {}
        ts = m.get("total_seconds")
        lines.append(
            f"| {mode} | {ts if ts is not None else 'error'} "
            f"| {m.get('speedup_vs_serial') if m.get('speedup_vs_serial') is not None else '—'} "
            f"| {m.get('failed')} | {m.get('n_episodes')} | {m.get('snapshots')} "
            f"| {m.get('load_events')} |")
    lines += [
        "",
        f"action model: {action_model}",
        "",
        "Methodology alignment: serial ~ 902.499 / pipeline ~ 644.439 / adaptive ~ 536.550 (checklist doc); "
        "absolute values are not comparable - different hardware and data, "
        "only the three-mode ratios are comparable within this batch.",
    ]
    return "\n".join(lines) + "\n"


def write_report(agg: dict, bench_root, action_model: str = "none") -> tuple:
    root = Path(bench_root)
    root.mkdir(parents=True, exist_ok=True)
    jp = root / "bench_report.json"
    mp = root / "bench_report.md"
    payload = {**agg, "action_model": action_model}
    jp.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    mp.write_text(render_report(agg, action_model))
    return jp, mp


def mode_config(config: str, run_root) -> str:
    """Derive a per-mode config: output.uri points at <run_root>/out.

    `--run-root` alone cannot isolate artifacts - EgoReconstruction's manifest/export
    directories come from config.output.uri (reconstruction.py), so sharing one uri
    across modes would overwrite each other. Each mode writes its own derived config so
    the path assumptions of `load_run(<run_root>)` and timeline rendering hold.
    Returns the original path unchanged when the config file does not exist
    (unit-test / pure-orchestration scenarios do not write files).
    """
    src = Path(config)
    if not src.exists():
        return str(config)
    import yaml

    cfg = yaml.safe_load(src.read_text()) or {}
    cfg.setdefault("output", {})["uri"] = str(Path(run_root) / "out")
    dst = Path(run_root) / "bench-config.yaml"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
    return str(dst)


def run_mode(mode: str, config: str, run_root, ray_address=None) -> int:
    """Run one mode in a subprocess for isolation (fresh driver/CostModel per mode)."""
    cmd = [sys.executable, "-m", "roboray.cli", "run",
           "--config", mode_config(config, run_root),
           "--run-root", str(run_root), *MODE_FLAGS[mode]]
    if ray_address is not None:
        cmd += ["--ray-address", ray_address]
    print(f"[bench] {mode}: {' '.join(cmd)}", flush=True)
    return subprocess.call(cmd)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="run config yaml (shared by all three modes)")
    ap.add_argument("--bench-root", required=True, help="benchmark artifacts root directory")
    ap.add_argument("--modes", default=",".join(MODES),
                    help=f"comma-separated, default {'.'.join(MODES)}")
    ap.add_argument("--keep-going", action="store_true", help="continue with remaining modes after one fails")
    ap.add_argument("--action-model", default="none",
                    help="action model to annotate in the report (pass openai_vlm once VLM is wired)")
    ap.add_argument("--ray-address", default=None,
                    help="existing Ray cluster address (e.g. auto); if omitted each mode starts its own local cluster")
    args = ap.parse_args(argv)
    bench_root = Path(args.bench_root)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    for m in modes:
        if m not in MODES:
            print(f"unknown mode: {m} (choose from {'/'.join(MODES)})", file=sys.stderr)
            return 2
    runs: dict = {}
    aborted = None
    for mode in modes:
        run_root = bench_root / f"run-{mode}"
        rc = run_mode(mode, args.config, run_root, ray_address=args.ray_address)
        runs[mode] = load_run(run_root)
        if rc != 0 or not runs[mode]["ok"]:
            aborted = mode
            print(f"[bench] {mode} failed (rc={rc}); report keeps the modes that completed", flush=True)
            if not args.keep_going:
                break
    agg = aggregate(runs)
    # adaptive succeeded -> auto-render the cross-stage overlap evidence (spec G2)
    overlap_line = None
    ad = runs.get("adaptive") or {}
    if ad.get("ok") and ad.get("timeline"):
        from roboray.viz.timeline_view import render_from_manifest, summarize

        ev = render_from_manifest(
            bench_root / "run-adaptive" / "out" / "manifest.json",
            bench_root / "run-adaptive" / "timeline")
        overlap_line = summarize(ev)
        agg["overlap"] = ev
        print(f"[bench] overlap evidence: {overlap_line}", flush=True)
    jp, mp = write_report(agg, bench_root, action_model=args.action_model)
    print(json.dumps({"report": str(mp), "overlap": overlap_line,
                      "aborted_at": aborted}, ensure_ascii=False, indent=2))
    return 1 if aborted else 0


if __name__ == "__main__":
    raise SystemExit(main())
