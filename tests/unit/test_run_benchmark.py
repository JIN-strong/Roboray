"""run_benchmark reporting logic, driven by fake manifests (spec §3.1)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import run_benchmark as rb


def _write_manifest(run_root, total, failed=0, n=6, snapshots=3, timeline=None, loads=1):
    out = Path(run_root) / "out"
    out.mkdir(parents=True, exist_ok=True)
    snaps = [{}] * max(0, snapshots - loads) + [{"was_load": True}] * min(loads, snapshots)
    tl = list(timeline or [])
    # load count is taken from the timeline's "load" events (recorded for all three modes; snapshots exist only for adaptive)
    tl += [{"t": 0.0, "episode_id": f"ep{i}", "stage": "scene", "event": "load"}
           for i in range(max(0, loads))]
    (out / "manifest.json").write_text(json.dumps({
        "summary": {"n_episodes": n, "failed": failed, "total_seconds": total},
        "scheduler_snapshots": snaps,
        "timeline": tl,
    }))


def test_load_run_reads_manifest(tmp_path):
    _write_manifest(tmp_path / "run-a", 10.5, snapshots=4, loads=1)
    r = rb.load_run(tmp_path / "run-a")
    assert {k: r[k] for k in ("ok", "error", "total_seconds", "failed", "n_episodes",
                              "snapshots", "load_events")} == {
        "ok": True, "error": None, "total_seconds": 10.5, "failed": 0,
        "n_episodes": 6, "snapshots": 4, "load_events": 1}
    assert [e["event"] for e in r["timeline"]] == ["load"]


def test_load_run_missing_and_corrupt(tmp_path):
    # missing/corrupt manifest -> ok=False, no raise
    r = rb.load_run(tmp_path / "nope")
    assert r["ok"] is False and "manifest" in r["error"]
    bad = tmp_path / "bad" / "out"
    bad.mkdir(parents=True)
    (bad / "manifest.json").write_text("{oops")
    r = rb.load_run(tmp_path / "bad")
    assert r["ok"] is False and "manifest" in r["error"]


def test_aggregate_speedups(tmp_path):
    runs = {
        "serial": {"ok": True, "error": None, "total_seconds": 10.0, "failed": 0,
                   "n_episodes": 6, "snapshots": 0, "timeline": []},
        "pipeline": {"ok": True, "error": None, "total_seconds": 8.0, "failed": 0,
                     "n_episodes": 6, "snapshots": 0, "timeline": []},
        "adaptive": {"ok": True, "error": None, "total_seconds": 5.0, "failed": 0,
                     "n_episodes": 6, "snapshots": 6, "timeline": [{"t": 0}]},
    }
    agg = rb.aggregate(runs)
    assert agg["modes"]["pipeline"]["speedup_vs_serial"] == 1.25
    assert agg["modes"]["adaptive"]["speedup_vs_serial"] == 2.0
    assert agg["modes"]["serial"]["speedup_vs_serial"] is None


def test_aggregate_survives_missing_mode():
    agg = rb.aggregate({"serial": {"ok": True, "error": None, "total_seconds": 10.0,
                                   "failed": 0, "n_episodes": 6, "snapshots": 0,
                                   "timeline": []}})
    assert agg["modes"]["adaptive"]["ok"] is False
    assert agg["modes"]["adaptive"]["speedup_vs_serial"] is None


def test_render_report_has_methodology_note():
    agg = rb.aggregate({m: {"ok": False, "error": "not run"} for m in rb.MODES})
    md = rb.render_report(agg, action_model="none")
    assert "| serial |" in md and "| pipeline |" in md and "| adaptive |" in md
    assert "902.499" in md and "644.439" in md and "536.550" in md  # methodology note
    assert "action model: none" in md


def test_write_report_emits_both_files(tmp_path):
    agg = rb.aggregate({m: {"ok": False, "error": "not run"} for m in rb.MODES})
    jp, mp = rb.write_report(agg, tmp_path)
    assert jp.name == "bench_report.json" and mp.name == "bench_report.md"
    assert json.loads(jp.read_text())["modes"].keys() >= {"serial", "pipeline", "adaptive"}


# --- CLI orchestration (subprocess stubbed; no real runs) ---
def _ev(t, ep, stage, event):
    return {"t": t, "episode_id": ep, "stage": stage, "event": event}


def test_main_runs_modes_in_order_and_renders_overlap(monkeypatch, tmp_path):
    cmds = []

    def fake_call(cmd):
        cmds.append(list(cmd))
        rr = Path(cmd[cmd.index("--run-root") + 1])
        _write_manifest(rr, 10.0 if "serial" in str(rr) else 5.0,
                        timeline=[_ev(0.0, "ep1", "prepare", "start"),
                                  _ev(1.0, "ep1", "prepare", "end")])
        return 0

    monkeypatch.setattr(rb.subprocess, "call", fake_call)
    rc = rb.main(["--config", "x.yaml", "--bench-root", str(tmp_path)])
    assert rc == 0
    assert len(cmds) == 3  # serial -> pipeline -> adaptive order
    assert cmds[0][:4] == [sys.executable, "-m", "roboray.cli", "run"]
    assert cmds[0][-1] == "--serial-baseline"       # flag drift guard: serial mode carries its flag
    assert cmds[1][-1] == "--no-adaptive"           # pipeline mode carries its flag
    assert "--serial-baseline" not in cmds[2] and "--no-adaptive" not in cmds[2]
    assert "--ray-address" not in cmds[0]           # not forwarded when unspecified
    assert (tmp_path / "run-adaptive" / "timeline" / "timeline.png").exists()
    assert (tmp_path / "bench_report.md").exists()


def test_keep_going_continues_after_failure(monkeypatch, tmp_path):
    n = {"i": 0}

    def fake_call(cmd):
        n["i"] += 1
        rr = Path(cmd[cmd.index("--run-root") + 1])
        if n["i"] == 1:  # serial mode fails: no manifest written
            return 1
        _write_manifest(rr, 5.0, timeline=[])
        return 0

    monkeypatch.setattr(rb.subprocess, "call", fake_call)
    rc = rb.main(["--config", "x.yaml", "--bench-root", str(tmp_path), "--keep-going"])
    assert rc == 1  # a mode failed
    assert n["i"] == 3  # the remaining two modes still ran
    md = (tmp_path / "bench_report.md").read_text()
    assert "error" in md  # serial mode is marked error


def test_abort_stops_without_keep_going(monkeypatch, tmp_path):
    def fake_call(cmd):
        return 1  # the first mode fails

    monkeypatch.setattr(rb.subprocess, "call", fake_call)
    rc = rb.main(["--config", "x.yaml", "--bench-root", str(tmp_path)])
    assert rc == 1
    agg = json.loads((tmp_path / "bench_report.json").read_text())
    assert agg["modes"]["pipeline"]["error"] == "not run"


def test_unknown_mode_rejected(tmp_path, capsys):
    rc = rb.main(["--config", "x.yaml", "--bench-root", str(tmp_path), "--modes", "bogus"])
    assert rc == 2


def test_mode_config_isolates_output_uri(tmp_path):
    """When a real config exists, each mode's derived config points output.uri at run-<mode>/out."""
    import yaml

    base = tmp_path / "bench.yaml"
    base.write_text(yaml.safe_dump({"input": {"uri": "/in"},
                                    "output": {"uri": "/shared/out"}}))
    run_root = tmp_path / "run-serial"
    derived = Path(rb.mode_config(str(base), run_root))
    data = yaml.safe_load(derived.read_text())
    assert data["output"]["uri"] == str(run_root / "out")
    assert data["input"]["uri"] == "/in"  # other keys preserved as-is
    # missing config file -> returned unchanged (pure-orchestration/unit-test paths write nothing)
    assert rb.mode_config("missing.yaml", run_root) == "missing.yaml"


def test_ray_address_passed_to_subprocess(monkeypatch, tmp_path):
    cmds = []

    def fake_call(cmd):
        cmds.append(list(cmd))
        rr = Path(cmd[cmd.index("--run-root") + 1])
        _write_manifest(rr, 5.0, timeline=[])
        return 0

    monkeypatch.setattr(rb.subprocess, "call", fake_call)
    rc = rb.main(["--config", "x.yaml", "--bench-root", str(tmp_path),
                  "--modes", "pipeline", "--ray-address", "auto"])
    assert rc == 0
    assert cmds[0][-2:] == ["--ray-address", "auto"]


def test_action_model_reaches_report(monkeypatch, tmp_path):
    def fake_call(cmd):
        rr = Path(cmd[cmd.index("--run-root") + 1])
        _write_manifest(rr, 5.0, timeline=[])
        return 0

    monkeypatch.setattr(rb.subprocess, "call", fake_call)
    rc = rb.main(["--config", "x.yaml", "--bench-root", str(tmp_path),
                  "--modes", "adaptive", "--action-model", "openai_vlm"])
    assert rc == 0
    md = (tmp_path / "bench_report.md").read_text()
    assert "action model: openai_vlm" in md
    assert json.loads((tmp_path / "bench_report.json").read_text())["action_model"] == "openai_vlm"
