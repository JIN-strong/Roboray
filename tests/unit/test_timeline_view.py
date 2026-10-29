"""overlap_evidence: programmatic cross-stage overlap determination (spec §4.1)."""
from roboray.viz.timeline_view import overlap_evidence


def _ev(t, ep, stage, event):
    return {"t": t, "episode_id": ep, "stage": stage, "event": event}


def test_three_episode_cross_stage_overlap():
    tl = [
        _ev(0.0, "ep1", "prepare", "start"), _ev(1.0, "ep1", "prepare", "end"),
        _ev(1.0, "ep1", "action", "start"), _ev(2.5, "ep1", "action", "end"),
        _ev(0.5, "ep2", "scene", "start"), _ev(3.0, "ep2", "scene", "end"),
        _ev(0.6, "ep2", "hand", "start"), _ev(2.5, "ep2", "hand", "end"),
        _ev(1.5, "ep3", "prepare", "start"), _ev(2.0, "ep3", "prepare", "end"),
        # Idle/ops events (episode_id=None) must be skipped
        {"t": 2.0, "episode_id": None, "stage": "unload", "event": "start"},
    ]
    ev = overlap_evidence(tl)
    assert ev["max_concurrent_episodes"] == 3
    assert ev["max_concurrent_stages"] == 4
    assert ev["open_intervals"] == 0
    snap = ev["evidence"][0]
    assert snap["t"] == 1.5
    assert snap["episodes"] == {
        "ep1": ["action"], "ep2": ["scene", "hand"], "ep3": ["prepare"],
    }


def test_serial_single_episode_max_is_one():
    tl = [
        _ev(0.0, "ep1", "prepare", "start"), _ev(1.0, "ep1", "prepare", "end"),
        _ev(1.0, "ep1", "scene", "start"), _ev(2.0, "ep1", "scene", "end"),
    ]
    ev = overlap_evidence(tl)
    assert ev["max_concurrent_episodes"] == 1
    assert ev["max_concurrent_stages"] == 1
    assert ev["evidence"][0]["episodes"] == {"ep1": ["prepare"]}


def test_open_interval_truncated_and_counted():
    # ep1's action has a start but no end (crash scenario). The first 2-concurrency
    # peak time (t=1.0) coincides with the action interval being open -> the snapshot
    # shows the truncated interval.
    tl = [
        _ev(0.0, "ep1", "prepare", "start"), _ev(1.0, "ep1", "prepare", "end"),
        _ev(1.0, "ep1", "action", "start"),
        _ev(1.0, "ep2", "scene", "start"), _ev(3.0, "ep2", "scene", "end"),
        _ev(1.0, "ep2", "hand", "start"), _ev(2.5, "ep2", "hand", "end"),
    ]
    ev = overlap_evidence(tl)
    assert ev["open_intervals"] == 1
    assert ev["max_concurrent_episodes"] == 2  # the truncated interval still counts toward concurrency
    assert ev["evidence"][0]["episodes"]["ep1"] == ["action"]


def test_illegal_pairs_ignored():
    # Duplicate start, and an end without a start
    tl = [
        _ev(0.0, "ep1", "scene", "start"),
        _ev(0.5, "ep1", "scene", "start"),   # duplicate start: ignored
        _ev(0.6, "ep1", "hand", "end"),      # end without start: ignored
        _ev(1.0, "ep1", "scene", "end"),
    ]
    ev = overlap_evidence(tl)
    assert ev["max_concurrent_episodes"] == 1
    assert ev["max_concurrent_stages"] == 1
    assert ev["open_intervals"] == 0


def test_load_events_ignored_even_with_valid_episode_and_stage():
    """load is a cold-start event, not a stage interval: it must not count toward
    concurrency/open_intervals."""
    tl = [
        _ev(0.0, "ep1", "prepare", "start"), _ev(1.0, "ep1", "prepare", "end"),
        _ev(0.5, "ep2", "scene", "load"),  # valid episode_id/stage, but not start/end
    ]
    ev = overlap_evidence(tl)
    assert ev["max_concurrent_episodes"] == 1
    assert ev["max_concurrent_stages"] == 1
    assert ev["open_intervals"] == 0
    assert ev["evidence"][0]["episodes"] == {"ep1": ["prepare"]}


def test_empty_and_idle_only():
    assert overlap_evidence([]) == {
        "max_concurrent_episodes": 0, "max_concurrent_stages": 0,
        "evidence": [], "open_intervals": 0,
    }
    ev = overlap_evidence([{"t": 1.0, "episode_id": None, "stage": "exit", "event": "start"}])
    assert ev["max_concurrent_episodes"] == 0 and ev["evidence"] == []


# -- rendering and one-stop entry --
import json as _json


def _manifest_with(timeline, tmp_path):
    p = tmp_path / "manifest.json"
    p.write_text(_json.dumps({"timeline": timeline, "summary": {"n_episodes": 1}}))
    return p


def test_render_from_manifest_writes_png_and_evidence(tmp_path):
    from roboray.viz.timeline_view import render_from_manifest
    tl = [
        _ev(0.0, "ep1", "prepare", "start"), _ev(1.0, "ep1", "prepare", "end"),
        _ev(1.0, "ep1", "action", "start"), _ev(2.0, "ep1", "action", "end"),
    ]
    mp = _manifest_with(tl, tmp_path)
    out_dir = tmp_path / "timeline"
    ev = render_from_manifest(mp, out_dir)
    assert (out_dir / "timeline.png").stat().st_size > 0
    assert _json.loads((out_dir / "timeline_evidence.json").read_text()) == ev
    assert ev["max_concurrent_episodes"] == 1


def test_summarize_formats_evidence_line():
    from roboray.viz.timeline_view import summarize
    ev = {"max_concurrent_episodes": 3, "max_concurrent_stages": 4,
          "evidence": [{"t": 1.5, "episodes": {"ep1": ["action"], "ep2": ["hand", "scene"]}}],
          "open_intervals": 0}
    line = summarize(ev)
    assert line.startswith("max_concurrent_episodes=3")
    assert "ep1: action" in line and "ep2: hand, scene" in line
    assert summarize({"max_concurrent_episodes": 0, "evidence": []}) == "max_concurrent_episodes=0"
