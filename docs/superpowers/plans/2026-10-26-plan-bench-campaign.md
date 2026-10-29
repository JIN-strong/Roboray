# Benchmark Closed Loop (real-model three-tier benchmark + overlap evidence + VLM probe) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** turn the two new sections of the checklist doc into deliverable evidence: a real-model three-tier benchmark report, cross-stage overlap evidence (PNG + programmatic determination), and a VLM reachability probe with both branches wired through.

**Architecture:** first persist the timeline into the manifest (today `RunResult.timeline` is dropped at the write_manifest call site); then `viz/timeline_view.py` provides pure-function determination + Gantt rendering + one-stop manifest rendering; `scripts/probe_vlm.py` probes the VLM with four-level tags; `scripts/run_benchmark.py` orchestrates the three tiers via subprocess (serial → pipeline → adaptive) and aggregates into bench_report.md/json. Finally Task 7 runs the campaign on the A800 real cluster (GATE-gated).

**Tech Stack:** Python 3.12 (roboray conda env), Ray 2.59, matplotlib (Agg, existing dependency), cv2 (existing dependency), openai SDK (lazy import, existing dependency), pytest.

**Spec:** `docs/superpowers/specs/2026-10-26-bench-vlm-overlap-design.md`

## Global Constraints

- Test command (repo root): `RAY_ADDRESS=local python -m pytest -q --tb=short`
- Commit messages: English conventional commits (`feat:`/`test:`/`docs:`), ending with `Co-Authored-By: Claude Code <noreply@anthropic.com>`
- No new third-party dependencies; `openai` is only allowed as a lazy import (same style as `semantic_models.py:83`)
- matplotlib only uses the Agg backend (`matplotlib.use("Agg")` before importing pyplot, same as `viz/qc_plots.py:9-13`)
- timeline event schema (`driver.py:_event`): `{"t": float, "episode_id": str|None, "stage": str, "event": str}`; idle events have `episode_id=None`
- Unit tests forbid real network/real subprocess: the VLM uses an injected `client_factory`, and the benchmark orchestration monkeypatches `subprocess.call`
- Scripts add a bootstrap before importing roboray: `sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))`
- Manifest changes are **additive only**: add the `timeline` key, touch no existing keys
- Comments/docs in English, consistent with the repo convention

## Review Focus

1. The timeline mixes in idle/ops events (`episode_id=None`, see `_idle_timeline_event`) → `overlap_evidence` must skip them, neither crashing nor counting them as concurrency (Task 2's three-episode overlap case includes an idle-event assertion).
2. In a crash scenario an interval has no end event → truncate at the last event time, count `open_intervals`, and still participate in concurrency statistics -- neither undercounting nor raising (Task 2 `test_open_interval_truncated_and_counted`).
3. After a tier crashes, the manifest is missing or a half-finished JSON → `load_run` returns `ok=False` with a reason, the report marks that tier error, and `--keep-going` continues the remaining tiers (Task 5 `test_load_run_missing_and_corrupt`, Task 6 `test_keep_going_continues_after_failure`).
4. Abnormal timelines (duplicate start for the same (ep,stage), an end without a start) → ignore illegal pairs without raising; `open_intervals` only counts genuinely unclosed ones (Task 2 `test_illegal_pairs_ignored`).
5. The platform vLLM raises non-openai exception types (ConnectionError, timeout, arbitrary ValueError) → `classify_exc` bottoms out at `unreachable`, and any level's failure in `probe` returns a tag rather than raising (Task 4 `test_classify_exc_fallback`, `test_probe_never_raises`).

---

### Task 1: persist the timeline into the manifest

**Files:**
- Modify: `src/roboray/manifest.py` (`write_manifest` signature + manifest dict)
- Modify: `src/roboray/api/reconstruction.py:149-156` (call site passes `result.timeline`)
- Test: `tests/unit/test_manifest.py` (new)

**Interfaces:**
- Consumes: `RunResult.timeline: list` (`runtime/driver.py:62`, already present)
- Produces: `write_manifest(output_root, request, per_episode, scheduler_snapshots, total_seconds=0.0, timeline=()) -> Path`; manifest.json's top level gains `"timeline": [...]`. Task 6's `load_run` depends on this key.

- [ ] **Step 1: write failing tests**

```python
"""manifest timeline persistence (precursor to spec 2026-10-26 §4.2)."""
import json
from pathlib import Path

from roboray.api.config import ReconstructionRequest
from roboray.manifest import write_manifest

_MIN_CONFIG = {
    "input": {"uri": "/tmp/in"},
    "output": {"uri": "/tmp/out", "format": "lerobot_v3"},
}


def _request():
    return ReconstructionRequest.from_dict(_MIN_CONFIG)


def test_timeline_roundtrips_into_manifest(tmp_path):
    timeline = [
        {"t": 0.1, "episode_id": "ep_0001", "stage": "prepare", "event": "start"},
        {"t": 0.2, "episode_id": "ep_0001", "stage": "prepare", "event": "end"},
    ]
    path = write_manifest(str(tmp_path), _request(), {}, [], 1.0, timeline)
    data = json.loads(Path(path).read_text())
    assert data["timeline"] == timeline
    # additive invariant: the existing keys are preserved as-is
    assert data["summary"]["n_episodes"] == 0
    assert data["scheduler_snapshots"] == []


def test_timeline_defaults_to_empty(tmp_path):
    path = write_manifest(str(tmp_path), _request(), {}, [], 1.0)
    data = json.loads(Path(path).read_text())
    assert data["timeline"] == []
```

- [ ] **Step 2: run the tests and confirm failure**

Run: `RAY_ADDRESS=local python -m pytest tests/unit/test_manifest.py -q`
Expected: FAIL — `write_manifest() takes from 5 to 6 positional arguments but 7 were given`

- [ ] **Step 3: minimal implementation**

`src/roboray/manifest.py`: add `timeline: list = ()` to the signature; in the manifest dict, right after `"scheduler_snapshots": list(scheduler_snapshots),` add a line:

```python
        "timeline": list(timeline),
```

`src/roboray/api/reconstruction.py`'s call site (the `write_manifest(...)` at lines 149-156) appends the 6th argument `result.timeline,`.

- [ ] **Step 4: run the tests and confirm pass (incl. full regression)**

Run: `RAY_ADDRESS=local python -m pytest tests/unit/test_manifest.py -q` → PASS
Run: `RAY_ADDRESS=local python -m pytest -q --tb=short` → all green (284+2)

- [ ] **Step 5: commit**

```bash
git add src/roboray/manifest.py src/roboray/api/reconstruction.py tests/unit/test_manifest.py
git commit -m "feat(manifest): persist the timeline into manifest.json -- the data source for cross-stage overlap evidence"
```

---

### Task 2: `overlap_evidence` pure function

**Files:**
- Create: `src/roboray/viz/timeline_view.py`
- Modify: `src/roboray/viz/__init__.py` (export)
- Test: `tests/unit/test_timeline_view.py` (new)

**Interfaces:**
- Consumes: the timeline event schema (Global Constraints)
- Produces:
  - `_intervals(timeline: list) -> tuple[list[tuple[float, float, str, str]], int]` -- (interval list `(t0, t1, ep, stage)`, unclosed count); Task 3's `render_gantt` reuses it
  - `overlap_evidence(timeline: list) -> dict`, keys: `max_concurrent_episodes: int`, `max_concurrent_stages: int`, `evidence: list[{"t": float, "episodes": dict[str, list[str]]}]`, `open_intervals: int`

- [ ] **Step 1: write failing tests**

```python
"""overlap_evidence: programmatic determination of cross-stage overlap (spec §4.1)."""
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
        # Review Focus 1: idle/ops events (episode_id=None) must be skipped
        {"t": 2.0, "episode_id": None, "stage": "unload", "event": "unload"},
    ]
    ev = overlap_evidence(tl)
    assert ev["max_concurrent_episodes"] == 3
    assert ev["max_concurrent_stages"] == 4
    assert ev["open_intervals"] == 0
    snap = ev["evidence"][0]
    assert snap["t"] == 1.5
    assert snap["episodes"] == {
        "ep1": ["action"], "ep2": ["hand", "scene"], "ep3": ["prepare"],
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
    # ep1's action has only a start, no end (crash scenario) -> Review Focus 2
    tl = [
        _ev(0.0, "ep1", "prepare", "start"), _ev(1.0, "ep1", "prepare", "end"),
        _ev(1.0, "ep1", "action", "start"),
        _ev(0.5, "ep2", "scene", "start"), _ev(3.0, "ep2", "scene", "end"),
    ]
    ev = overlap_evidence(tl)
    assert ev["open_intervals"] == 1
    assert ev["max_concurrent_episodes"] == 2  # the truncated interval still counts toward concurrency
    assert ev["evidence"][0]["episodes"]["ep1"] == ["action"]


def test_illegal_pairs_ignored():
    # Review Focus 4: duplicate start, end without a start
    tl = [
        _ev(0.0, "ep1", "scene", "start"),
        _ev(0.5, "ep1", "scene", "start"),   # duplicate start: ignored
        _ev(0.6, "ep1", "hand", "end"),      # end without a start: ignored
        _ev(1.0, "ep1", "scene", "end"),
    ]
    ev = overlap_evidence(tl)
    assert ev["max_concurrent_episodes"] == 1
    assert ev["max_concurrent_stages"] == 1
    assert ev["open_intervals"] == 0


def test_empty_and_idle_only():
    assert overlap_evidence([]) == {
        "max_concurrent_episodes": 0, "max_concurrent_stages": 0,
        "evidence": [], "open_intervals": 0,
    }
    ev = overlap_evidence([{"t": 1.0, "episode_id": None, "stage": "exit", "event": "exit"}])
    assert ev["max_concurrent_episodes"] == 0 and ev["evidence"] == []
```

- [ ] **Step 2: run the tests and confirm failure**

Run: `RAY_ADDRESS=local python -m pytest tests/unit/test_timeline_view.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'roboray.viz.timeline_view'`

- [ ] **Step 3: minimal implementation**

```python
"""Cross-stage overlap evidence: determine the max concurrent Episode count and the
concurrency shape from the timeline event stream.

spec: docs/superpowers/specs/2026-10-26-bench-vlm-overlap-design.md §4.1.
Pure functions, no matplotlib dependency -- easy to unit-test; rendering lives in
render_gantt (same file, lazily loaded).
"""

import json
from pathlib import Path

__all__ = ["overlap_evidence", "render_gantt", "render_from_manifest", "summarize"]

_STAGE_ORDER = ("prepare", "scene", "hand", "fusion", "action", "publish")


def _intervals(timeline: list) -> tuple[list, int]:
    """Pair start/end into (ep, stage) intervals; returns (interval list, unclosed count).

    - Idle/ops events (episode_id None) are skipped (Review Focus 1).
    - Duplicate starts and ends without a start are ignored, not raised (Review Focus 4).
    - Unclosed intervals are truncated at the last event time and counted (Review Focus 2).
    """
    intervals: list[tuple[float, float, str, str]] = []
    if not timeline:
        return intervals, 0
    last_t = max(float(e["t"]) for e in timeline)
    open_set: dict[tuple[str, str], float] = {}
    for e in sorted(timeline, key=lambda x: float(x["t"])):
        ep, stage, event = e.get("episode_id"), e.get("stage"), e.get("event")
        if ep is None or stage is None:
            continue
        key = (ep, stage)
        if event == "start":
            if key not in open_set:
                open_set[key] = float(e["t"])
        elif event == "end":
            t0 = open_set.pop(key, None)
            if t0 is not None:
                intervals.append((t0, float(e["t"]), ep, stage))
    open_count = len(open_set)
    for (ep, stage), t0 in open_set.items():
        intervals.append((t0, last_t, ep, stage))
    return intervals, open_count


def overlap_evidence(timeline: list) -> dict:
    """Concurrency-peak determination. The peak is the **first** start-time snapshot
    with the largest concurrent Episode count; max_concurrent_stages is the max number
    of parallel intervals across any boundary time in the whole run."""
    intervals, open_count = _intervals(timeline)
    max_eps, max_stages, snapshot, snap_t = 0, 0, None, None
    for t in sorted({t0 for t0, _, _, _ in intervals}):
        live = [(a, b, e, s) for a, b, e, s in intervals if a <= t < b]
        eps = {e for _, _, e, _ in live}
        if len(live) > max_stages:
            max_stages = len(live)
        if len(eps) > max_eps:
            max_eps = len(eps)
            snap_t = t
            tmp: dict[str, list[str]] = {}
            for _, _, e, s in live:
                tmp.setdefault(e, []).append(s)
            snapshot = {
                e: sorted(v, key=_STAGE_ORDER.index) for e, v in tmp.items()
            }
    evidence = []
    if snapshot is not None:
        evidence.append({"t": round(snap_t, 6), "episodes": snapshot})
    return {
        "max_concurrent_episodes": max_eps,
        "max_concurrent_stages": max_stages,
        "evidence": evidence,
        "open_intervals": open_count,
    }
```

Meanwhile `render_gantt` / `render_from_manifest` / `summarize` are placeholders to be added in Task 3 -- this task **only** implements the three functions above and `__all__ = ["overlap_evidence"]`, to avoid half-finished output.

- [ ] **Step 4: run the tests and confirm pass**

Run: `RAY_ADDRESS=local python -m pytest tests/unit/test_timeline_view.py -q` → 5 PASS

- [ ] **Step 5: commit**

```bash
git add src/roboray/viz/timeline_view.py tests/unit/test_timeline_view.py
git commit -m "feat(viz): overlap_evidence pure function -- programmatic determination of cross-stage overlap"
```

---

### Task 3: Gantt rendering + one-stop manifest rendering + render_timeline CLI

**Files:**
- Modify: `src/roboray/viz/timeline_view.py` (append three functions)
- Modify: `src/roboray/viz/__init__.py` (export the four names)
- Create: `scripts/render_timeline.py`
- Test: `tests/unit/test_timeline_view.py` (append)

**Interfaces:**
- Consumes: Task 2's `_intervals`, `overlap_evidence`
- Produces:
  - `render_gantt(timeline: list, out_png, title: str = "Episode × Stage timeline") -> Path`
  - `render_from_manifest(manifest_path, out_dir, title: str | None = None) -> dict` (returns the overlap_evidence result; writes `<out_dir>/timeline.png` + `<out_dir>/timeline_evidence.json`) -- Task 6's run_benchmark calls it directly
  - `summarize(evidence: dict) -> str` (a one-line markdown summary)

- [ ] **Step 1: append failing tests**

```python
# -- Task 3: rendering and the one-stop entry point --
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
```

- [ ] **Step 2: run the tests and confirm failure**

Run: `RAY_ADDRESS=local python -m pytest tests/unit/test_timeline_view.py -q`
Expected: FAIL — `ImportError: cannot import name 'render_from_manifest'`

- [ ] **Step 3: implement**

Append to `timeline_view.py` (matplotlib lazily loaded inside the function so the pure functions import without matplotlib):

```python
_STAGE_COLORS = {
    "prepare": "#7f7f7f", "scene": "#1f77b4", "hand": "#d62728",
    "fusion": "#ff7f0e", "action": "#2ca02c", "publish": "#9467bd",
}


def render_gantt(timeline: list, out_png, title: str = "Episode × Stage timeline") -> Path:
    """Horizontal bar chart with one row per Episode; stages color-coded, x axis seconds.
    Agg backend."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    intervals, _open = _intervals(timeline)
    eps = sorted({e for _, _, e, _ in intervals})
    fig, ax = plt.subplots(figsize=(10, max(1.5, 0.6 * len(eps) + 1)))
    for row, ep in enumerate(eps):
        for t0, t1, e, s in intervals:
            if e != ep:
                continue
            ax.broken_barh(
                [(t0, max(t1 - t0, 1e-6))], (row - 0.35, 0.7),
                facecolors=_STAGE_COLORS.get(s, "#bbbbbb"),
            )
    ax.set_yticks(range(len(eps)))
    ax.set_yticklabels(eps)
    ax.set_xlabel("t (s)")
    ax.set_title(title)
    ax.invert_yaxis()
    fig.tight_layout()
    out = Path(out_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def render_from_manifest(manifest_path, out_dir, title: str | None = None) -> dict:
    """Read the timeline from manifest.json -> write PNG + determination JSON, return
    the evidence."""
    data = json.loads(Path(manifest_path).read_text())
    tl = data.get("timeline") or []
    ev = overlap_evidence(tl)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if title is None:
        title = f"manifest: {Path(manifest_path).parent.parent.name}"
    render_gantt(tl, out / "timeline.png", title=title)
    (out / "timeline_evidence.json").write_text(
        json.dumps(ev, ensure_ascii=False, indent=2))
    return ev


def summarize(evidence: dict) -> str:
    """One-line markdown summary: max_concurrent_episodes=N (EP a: s1; EP b: s2)."""
    if not evidence.get("evidence"):
        return "max_concurrent_episodes=0"
    snap = evidence["evidence"][0]
    parts = "; ".join(f"{ep}: {', '.join(stages)}"
                      for ep, stages in snap["episodes"].items())
    return (f"max_concurrent_episodes={evidence['max_concurrent_episodes']}"
            f" ({parts})")
```

Update `__all__ = ["overlap_evidence", "render_gantt", "render_from_manifest", "summarize"]`.
Append to `viz/__init__.py`:

```python
from .timeline_view import (  # noqa: F401
    overlap_evidence, render_from_manifest, render_gantt, summarize,
)
```

(Add the four names to `__all__` as well.)

`scripts/render_timeline.py`:

```python
#!/usr/bin/env python
"""Render cross-stage overlap evidence from a manifest.json (PNG + determination JSON +
one-line summary).

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
```

- [ ] **Step 4: run the tests and confirm pass + CLI smoke test**

Run: `RAY_ADDRESS=local python -m pytest tests/unit/test_timeline_view.py -q` → 7 PASS
Run: `RAY_ADDRESS=local python scripts/render_timeline.py --help` → exits 0 and prints usage

- [ ] **Step 5: commit**

```bash
git add src/roboray/viz/timeline_view.py src/roboray/viz/__init__.py scripts/render_timeline.py tests/unit/test_timeline_view.py
git commit -m "feat(viz): timeline Gantt rendering + one-stop manifest evidence export + render_timeline CLI"
```

---

### Task 4: probe_vlm probe script

**Files:**
- Create: `scripts/probe_vlm.py`
- Test: `tests/unit/test_probe_vlm.py` (new)

**Interfaces:**
- Consumes: openai SDK (lazy import); cv2/numpy (existing dependencies)
- Produces: `probe(base_url: str, model: str, api_key: str, timeout: float = 30.0, client_factory=None) -> tuple[str, list[str]]`; `classify_exc(exc: Exception) -> str`; the tag set `{"reachable_multimodal", "reachable_text_only", "auth_failed", "unreachable"}`; CLI exit codes: `reachable_multimodal → 0`, everything else `→ 1`; stdout's first line is `PROBE_RESULT=<tag>`

- [ ] **Step 1: write failing tests**

```python
"""probe_vlm: inject a fake client, cover all 4 tag branches + dummy key (Review Focus 5)."""
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import probe_vlm


class _AuthErr(Exception):
    status_code = 401


def _client(models_ids=("m1",), text_fail=None, mm_fail=None):
    """Build a fake OpenAI client. text_fail/mm_fail are the exception to raise, or None."""
    def create(model, messages, **kw):
        is_mm = isinstance(messages[0]["content"], list)
        fail = mm_fail if is_mm else text_fail
        if fail is not None:
            raise fail
        return types.SimpleNamespace(choices=[types.SimpleNamespace(
            message=types.SimpleNamespace(content="pong"))])

    completions = types.SimpleNamespace(create=create)
    models = types.SimpleNamespace(list=lambda: types.SimpleNamespace(
        data=[types.SimpleNamespace(id=i) for i in models_ids]))
    chat = types.SimpleNamespace(completions=completions)
    return types.SimpleNamespace(models=models, chat=chat)


def _factory(client):
    return lambda **kw: client


def test_all_ok_returns_reachable_multimodal():
    tag, details = probe_vlm.probe("http://x", "m", "k", client_factory=_factory(_client()))
    assert tag == "reachable_multimodal"
    assert any("models" in d for d in details)


def test_multimodal_rejected_returns_text_only():
    c = _client(mm_fail=ValueError("image not supported"))
    tag, _ = probe_vlm.probe("http://x", "m", "k", client_factory=_factory(c))
    assert tag == "reachable_text_only"


def test_auth_failed_on_text():
    c = _client(text_fail=_AuthErr("bad key"))
    tag, _ = probe_vlm.probe("http://x", "m", "k", client_factory=_factory(c))
    assert tag == "auth_failed"


def test_unreachable_on_models_and_classify_fallback():
    # a non-typed exception (no status_code, name without auth) must bottom out at unreachable
    assert probe_vlm.classify_exc(ConnectionError("refused")) == "unreachable"
    assert probe_vlm.classify_exc(ValueError("boom")) == "unreachable"
    c = _client(text_fail=ConnectionError("refused"))
    models = types.SimpleNamespace(list=lambda: (_ for _ in ()).throw(ConnectionError("refused")))
    c = types.SimpleNamespace(models=models, chat=c.chat)
    tag, _ = probe_vlm.probe("http://x", "m", "k", client_factory=_factory(c))
    assert tag == "unreachable"


def test_probe_never_raises_on_arbitrary_failure():
    # Review Focus 5: any exception becomes a tag, never raised upward
    def bad_factory(**kw):
        raise RuntimeError("anything")
    tag, _ = probe_vlm.probe("http://x", "m", "k", client_factory=bad_factory)
    assert tag == "unreachable"


def test_resolve_key_dummy_and_env(monkeypatch):
    monkeypatch.delenv("VLM_API_KEY", raising=False)
    key, lines = probe_vlm.resolve_key("VLM_API_KEY")
    assert key == "EMPTY" and any("dummy" in l for l in lines)
    monkeypatch.setenv("VLM_API_KEY", "sk-real")
    key, lines = probe_vlm.resolve_key("VLM_API_KEY")
    assert key == "sk-real"
```

- [ ] **Step 2: run the tests and confirm failure**

Run: `RAY_ADDRESS=local python -m pytest tests/unit/test_probe_vlm.py -q`
Expected: FAIL — `No module named 'probe_vlm'`

- [ ] **Step 3: implement**

```python
#!/usr/bin/env python
"""VLM reachability probe: can the platform vLLM be used by the openai_vlm path
(spec §5.1).

Three-level probe: list models (best-effort) -> minimal text chat -> 8×8 PNG
multimodal chat.
stdout's first line outputs the machine-readable PROBE_RESULT=<tag>, tag ∈
{reachable_multimodal, reachable_text_only, auth_failed, unreachable}.
Exit codes: reachable_multimodal → 0, everything else → 1.

Must run inside a cluster pod (login-node network reachability does not imply pod
network reachability):
  python scripts/probe_vlm.py --base-url http://<host>:<port>/v1 --model <name>
"""
import argparse
import base64
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

__all__ = ["probe", "classify_exc", "resolve_key"]


def resolve_key(api_key_env: str) -> tuple[str, list[str]]:
    """Take the key from an environment variable; if missing, use the dummy 'EMPTY'
    (vLLM convention, spec decision D-B7)."""
    lines = []
    key = os.environ.get(api_key_env, "")
    if key:
        lines.append(f"auth: using the environment variable {api_key_env}")
    else:
        key = "EMPTY"
        lines.append(f"warning: {api_key_env} is empty, using the dummy key 'EMPTY' (vLLM convention)")
    return key, lines


def classify_exc(exc: Exception) -> str:
    """Exception -> tag. 401/403 or a type name containing auth -> auth_failed;
    otherwise -> unreachable."""
    status = getattr(exc, "status_code", None)
    if status in (401, 403):
        return "auth_failed"
    if "auth" in type(exc).__name__.lower():
        return "auth_failed"
    return "unreachable"


def _tiny_png_data_uri() -> str:
    """8×8 grayscale PNG -> data URI (cv2 encoded, within the main dependencies)."""
    import cv2
    import numpy as np

    ok, enc = cv2.imencode(".png", np.full((8, 8), 128, np.uint8))
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return "data:image/png;base64," + base64.b64encode(enc.tobytes()).decode()


def _content(prompt: str, image_uri: str | None) -> object:
    """Same multimodal format as semantic_models._build_content."""
    if image_uri is None:
        return prompt
    return [{"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_uri}}]


def probe(base_url: str, model: str, api_key: str, timeout: float = 30.0,
          client_factory=None) -> tuple[str, list[str]]:
    """Three-level probe; returns (tag, human-readable detail lines). client_factory is
    injectable for unit tests."""
    details: list[str] = []
    if client_factory is None:
        from openai import OpenAI  # lazy: same style as semantic_models (Global Constraints)
        client_factory = OpenAI
    try:
        client = client_factory(base_url=base_url, api_key=api_key, timeout=timeout)
    except Exception as exc:
        return classify_exc(exc), [f"client construction failed: {type(exc).__name__}: {exc}"]
    # 1) list models (best-effort)
    try:
        names = [m.id for m in client.models.list().data]
        details.append(f"models: {names}")
    except Exception as exc:
        details.append(f"models listing failed: {type(exc).__name__}: {exc}")
        return classify_exc(exc), details
    # 2) minimal text chat
    try:
        resp = client.chat.completions.create(
            model=model, max_tokens=8, timeout=timeout,
            messages=[{"role": "user", "content": "ping, answer only pong"}])
        details.append(f"text: {repr((resp.choices[0].message.content or '')[:40])}")
    except Exception as exc:
        details.append(f"text chat failed: {type(exc).__name__}: {exc}")
        return classify_exc(exc), details
    # 3) multimodal
    try:
        resp = client.chat.completions.create(
            model=model, max_tokens=8, timeout=timeout,
            messages=[{"role": "user",
                       "content": _content("What is in the image? Answer in one word.", _tiny_png_data_uri())}])
        details.append(f"multimodal: {repr((resp.choices[0].message.content or '')[:40])}")
    except Exception as exc:
        details.append(f"multimodal failed: {type(exc).__name__}: {exc}")
        return "reachable_text_only", details
    return "reachable_multimodal", details


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", required=True, help="OpenAI-compatible endpoint, e.g. http://host:8000/v1")
    ap.add_argument("--model", required=True, help="chat model name")
    ap.add_argument("--api-key-env", default="VLM_API_KEY")
    ap.add_argument("--timeout", type=float, default=30.0)
    args = ap.parse_args(argv)
    key, lines = resolve_key(args.api_key_env)
    for line in lines:
        print(line)
    tag, details = probe(args.base_url, args.model, key, timeout=args.timeout)
    print(f"PROBE_RESULT={tag}")
    for line in details:
        print(f"  {line}")
    return 0 if tag == "reachable_multimodal" else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: run the tests and confirm pass**

Run: `RAY_ADDRESS=local python -m pytest tests/unit/test_probe_vlm.py -q` → 6 PASS

- [ ] **Step 5: commit**

```bash
git add scripts/probe_vlm.py tests/unit/test_probe_vlm.py
git commit -m "feat(vlm): probe_vlm three-level reachability probe -- the basis for deciding between a real hookup and an honest fallback"
```

---

### Task 5: run_benchmark reporting logic (pure functions)

**Files:**
- Create: `scripts/run_benchmark.py` (this task only contains the reporting logic + MODES constant; the CLI orchestration is Task 6)
- Test: `tests/unit/test_run_benchmark.py` (new)

**Interfaces:**
- Consumes: Task 1's manifest `"timeline"` key
- Produces (depended on by Task 6, signatures verbatim):
  - `MODES = ("serial", "pipeline", "adaptive")`; `MODE_FLAGS = {"serial": ["--serial-baseline"], "pipeline": ["--no-adaptive"], "adaptive": []}`
  - `load_run(run_root) -> dict`, keys: `ok/error/total_seconds/failed/n_episodes/snapshots/load_events/timeline` ([implementation-time correction]`load_events` is finally counted as timeline `event=="load"` events, not the `was_load` snapshots -- since campaign commit 4eaa0f1 all three tiers emit load events, so cold-start evidence is countable in all three; see docs/bench/2026-10-26-bench-results.md §4)
  - `aggregate(runs: dict) -> dict` (each tier gains `speedup_vs_serial`)
  - `render_report(agg: dict, action_model: str = "none") -> str`
  - `write_report(agg, bench_root) -> tuple[Path, Path]`

- [ ] **Step 1: write failing tests**

```python
"""run_benchmark reporting logic: driven by fake manifests (spec §3.1)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import run_benchmark as rb


def _write_manifest(run_root, total, failed=0, n=6, snapshots=3, timeline=None, loads=1):
    out = Path(run_root) / "out"
    out.mkdir(parents=True, exist_ok=True)
    snaps = [{}] * max(0, snapshots - loads) + [{"was_load": True}] * min(loads, snapshots)
    (out / "manifest.json").write_text(json.dumps({
        "summary": {"n_episodes": n, "failed": failed, "total_seconds": total},
        "scheduler_snapshots": snaps,
        "timeline": timeline or [],
    }))


def test_load_run_reads_manifest(tmp_path):
    _write_manifest(tmp_path / "run-a", 10.5, snapshots=4, loads=1)
    r = rb.load_run(tmp_path / "run-a")
    assert r == {"ok": True, "error": None, "total_seconds": 10.5, "failed": 0,
                 "n_episodes": 6, "snapshots": 4, "load_events": 1, "timeline": []}


def test_load_run_missing_and_corrupt(tmp_path):
    # Review Focus 3: a missing/corrupt manifest -> ok=False, not raised
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
    assert "902.499" in md and "644.439" in md and "536.550" in md  # decision D-B3 note
    assert "action model: none" in md


def test_write_report_emits_both_files(tmp_path):
    agg = rb.aggregate({m: {"ok": False, "error": "not run"} for m in rb.MODES})
    jp, mp = rb.write_report(agg, tmp_path)
    assert jp.name == "bench_report.json" and mp.name == "bench_report.md"
    assert json.loads(jp.read_text())["modes"].keys() >= {"serial", "pipeline", "adaptive"}
```

- [ ] **Step 2: run the tests and confirm failure**

Run: `RAY_ADDRESS=local python -m pytest tests/unit/test_run_benchmark.py -q`
Expected: FAIL — `No module named 'run_benchmark'`

- [ ] **Step 3: implement**

```python
#!/usr/bin/env python
"""Real-model three-tier benchmark (spec G1): on the same cluster run serial →
pipeline → adaptive in sequence.

Each tier has its own run-root and subprocess isolation (fair cold start for the
driver/CostModel, decision D-B1); after the adaptive tier finishes, the
cross-stage overlap evidence renders automatically (spec G2, wired in Task 6).
This file has two layers: reporting pure functions (this section, unit-testable)
+ CLI orchestration (see main below).
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
    """Read <run_root>/out/manifest.json -> a unified tier result; missing/corrupt ->
    ok=False, not raised."""
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
            # the was_load snapshot count = that tier's model-load count
            # (spec §3.3 cold-start evidence)
            "load_events": sum(1 for s in snaps if s.get("was_load")),
            "timeline": data.get("timeline") or []}


def aggregate(runs: dict) -> dict:
    """Per-tier raw results -> report dict; if serial is unavailable, all speedups are
    None (no fabricated ratios)."""
    out = {"modes": {}}
    serial = runs.get("serial") or {}
    base = serial.get("total_seconds") if serial.get("ok") else None
    for mode in MODES:
        r = runs.get(mode) or {"ok": False, "error": "not run"}
        entry = dict(r)
        entry["speedup_vs_serial"] = (
            round(base / r["total_seconds"], 4)
            if base and r.get("ok") and r.get("total_seconds") else None)
        out["modes"][mode] = entry
    return out


def render_report(agg: dict, action_model: str = "none") -> str:
    """markdown three-tier table + methodology note (decision D-B3's wording archived)."""
    lines = [
        "| tier | total_seconds | vs serial | failed | episodes | sched snapshots | load events |",
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
        "methodology alignment: serial↔902.499 / pipeline↔644.439 / adaptive↔536.550 "
        "(checklist doc); absolute values are not comparable -- different hardware and "
        "data (spec decision D-B3); only the three-tier ratios are comparable within this batch.",
    ]
    return "\n".join(lines) + "\n"


def write_report(agg: dict, bench_root) -> tuple:
    root = Path(bench_root)
    root.mkdir(parents=True, exist_ok=True)
    jp = root / "bench_report.json"
    mp = root / "bench_report.md"
    jp.write_text(json.dumps(agg, ensure_ascii=False, indent=2))
    mp.write_text(render_report(agg))
    return jp, mp
```

(`main` is left to Task 6; this task's file must **not** end with an `if __name__` block.)

- [ ] **Step 4: run the tests and confirm pass**

Run: `RAY_ADDRESS=local python -m pytest tests/unit/test_run_benchmark.py -q` → 6 PASS

- [ ] **Step 5: commit**

```bash
git add scripts/run_benchmark.py tests/unit/test_run_benchmark.py
git commit -m "feat(bench): run_benchmark reporting pure functions -- three-tier aggregation/speedups/methodology note"
```

---

### Task 6: run_benchmark CLI orchestration + flag drift prevention

**Files:**
- Modify: `scripts/run_benchmark.py` (append `run_mode` + `main` + `__main__`)
- Test: `tests/unit/test_run_benchmark.py` (append orchestration cases)
- Test: `tests/unit/test_cli.py` (append flag-pass-through assertions; read the existing cases first to avoid name clashes)

**Interfaces:**
- Consumes: all of Task 5's signatures; Task 3's `render_from_manifest` / `summarize`; the CLI flags `--serial-baseline` / `--no-adaptive` (`cli.py:22-23`)
- Produces: `python scripts/run_benchmark.py --config <yaml> --bench-root <dir> [--modes serial,pipeline,adaptive] [--keep-going] [--action-model none|openai_vlm]`; exit codes: 0 all succeed, 1 some tier failed, 2 unknown tier name; artifacts `<bench-root>/bench_report.{json,md}` + `<bench-root>/run-adaptive/timeline/{timeline.png,timeline_evidence.json}`

- [ ] **Step 1: append failing tests**

```python
# -- Task 6: CLI orchestration (subprocess stubbed, no real runs) --
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
    assert cmds[0][-1] == "--serial-baseline"       # flag drift prevention: the serial tier carries the flag
    assert cmds[1][-1] == "--no-adaptive"           # the pipeline tier carries the flag
    assert "--serial-baseline" not in cmds[2] and "--no-adaptive" not in cmds[2]
    assert (tmp_path / "run-adaptive" / "timeline" / "timeline.png").exists()
    assert (tmp_path / "bench_report.md").exists()


def test_keep_going_continues_after_failure(monkeypatch, tmp_path):
    n = {"i": 0}

    def fake_call(cmd):
        n["i"] += 1
        rr = Path(cmd[cmd.index("--run-root") + 1])
        if n["i"] == 1:  # the serial tier crashes: no manifest written
            return 1
        _write_manifest(rr, 5.0, timeline=[])
        return 0

    monkeypatch.setattr(rb.subprocess, "call", fake_call)
    rc = rb.main(["--config", "x.yaml", "--bench-root", str(tmp_path), "--keep-going"])
    assert rc == 1  # some tier failed
    assert n["i"] == 3  # the remaining two tiers still run
    md = (tmp_path / "bench_report.md").read_text()
    assert "error" in md  # the serial tier is marked error


def test_abort_stops_without_keep_going(monkeypatch, tmp_path):
    def fake_call(cmd):
        return 1  # the first tier crashes

    monkeypatch.setattr(rb.subprocess, "call", fake_call)
    rc = rb.main(["--config", "x.yaml", "--bench-root", str(tmp_path)])
    assert rc == 1
    agg = json.loads((tmp_path / "bench_report.json").read_text())
    assert agg["modes"]["pipeline"]["error"] == "not run"


def test_unknown_mode_rejected(tmp_path, capsys):
    rc = rb.main(["--config", "x.yaml", "--bench-root", str(tmp_path), "--modes", "bogus"])
    assert rc == 2
```

Append to `tests/unit/test_cli.py` (read the file first to confirm there is no same-named case; `_MIN_CONFIG` uses the same dict as Task 1, or inline it):

```python
def test_mode_flags_reach_reconstruction(monkeypatch, tmp_path):
    """Three-tier flag drift prevention (spec §7): --serial-baseline/--no-adaptive must
    pass through to the API."""
    import types

    import yaml

    from roboray.cli import main

    captured = {}

    class _FakeRun:
        def __init__(self, request, run_root=None, serial_baseline=False,
                     adaptive=True, ray_address=None):
            captured["serial_baseline"] = serial_baseline
            captured["adaptive"] = adaptive

        def run(self):
            return types.SimpleNamespace(total_seconds=0.0, failed=[],
                                         manifest_path=None, timeline=[])

    monkeypatch.setattr("roboray.cli.EgoReconstruction", _FakeRun)
    cfg = tmp_path / "c.yaml"
    cfg.write_text(yaml.safe_dump({
        "input": {"uri": "/tmp/in"},
        "output": {"uri": "/tmp/out", "format": "lerobot_v3"},
    }))
    rc = main(["run", "--config", str(cfg), "--serial-baseline"])
    assert rc == 0
    assert captured == {"serial_baseline": True, "adaptive": True}
```

- [ ] **Step 2: run the tests and confirm failure**

Run: `RAY_ADDRESS=local python -m pytest tests/unit/test_run_benchmark.py -q`
Expected: the new cases FAIL — `AttributeError: module 'run_benchmark' has no attribute 'main'`

- [ ] **Step 3: implement (append to run_benchmark.py)**

```python
def run_mode(mode: str, config: str, run_root) -> int:
    """Run one tier in subprocess isolation (decision D-B1: a fresh driver/CostModel per tier)."""
    cmd = [sys.executable, "-m", "roboray", "run", "--config", str(config),
           "--run-root", str(run_root), *MODE_FLAGS[mode]]
    print(f"[bench] {mode}: {' '.join(cmd)}", flush=True)
    return subprocess.call(cmd)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="run config yaml (shared by all three tiers)")
    ap.add_argument("--bench-root", required=True, help="benchmark artifact root directory")
    ap.add_argument("--modes", default=",".join(MODES),
                    help=f"comma-separated, default {'.'.join(MODES)}")
    ap.add_argument("--keep-going", action="store_true", help="continue the remaining tiers after a tier fails")
    ap.add_argument("--action-model", default="none",
                    help="the action model the report is labeled with (pass openai_vlm once the VLM is wired)")
    args = ap.parse_args(argv)
    bench_root = Path(args.bench_root)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    for m in modes:
        if m not in MODES:
            print(f"unknown tier: {m} (options {'/'.join(MODES)})", file=sys.stderr)
            return 2
    runs: dict = {}
    aborted = None
    for mode in modes:
        run_root = bench_root / f"run-{mode}"
        rc = run_mode(mode, args.config, run_root)
        runs[mode] = load_run(run_root)
        if rc != 0 or not runs[mode]["ok"]:
            aborted = mode
            print(f"[bench] {mode} failed (rc={rc}); the report keeps the completed tiers", flush=True)
            if not args.keep_going:
                break
    agg = aggregate(runs)
    # the adaptive tier succeeded -> auto-render the cross-stage overlap evidence (spec G2)
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
    jp, mp = write_report(agg, bench_root)
    print(json.dumps({"report": str(mp), "overlap": overlap_line,
                      "aborted_at": aborted}, ensure_ascii=False, indent=2))
    return 1 if aborted else 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: run the tests and confirm pass (incl. full regression)**

Run: `RAY_ADDRESS=local python -m pytest tests/unit/test_run_benchmark.py tests/unit/test_cli.py -q` → PASS
Run: `RAY_ADDRESS=local python -m pytest -q --tb=short` → all green
Run: `RAY_ADDRESS=local python scripts/run_benchmark.py --help` → exits 0

- [ ] **Step 5: commit**

```bash
git add scripts/run_benchmark.py tests/unit/test_run_benchmark.py tests/unit/test_cli.py
git commit -m "feat(bench): run_benchmark CLI orchestration -- three-tier order/keep-going/adaptive auto-renders overlap evidence"
```

---

### Task 7: bench config sample + A800 real-cluster campaign

**Files:**
- Create: `deploy/bench/bench-l3.yaml` (structure sample; node concrete values are derived at execution time from `/ego/l3-final.yaml`)
- Create: `docs/bench/2026-10-26-bench-results.md` (campaign results, written after execution)

**Interfaces:**
- Consumes: Task 4's `probe_vlm.py` (inside a pod), Task 6's `run_benchmark.py`, `deploy/k8s/{status,apply,teardown}.sh` (E3-verified), `scripts/make_l3_dataset.py --episodes 6 --frames 60`
- Produces: `bench_report.md/json`, `run-adaptive/timeline/*`, the results doc, the `action_model` branch conclusion

- [ ] **Step 1: commit the config sample**

```bash
mkdir -p deploy/bench docs/bench
```

`deploy/bench/bench-l3.yaml` (structure sample; the `<>` places are replaced at execution time with the real values from the cluster node's `/ego/l3-final.yaml` -- that file is the real config used by E3's final validation, so its field names are authoritative):

```yaml
# Real-model three-tier benchmark config sample (spec §3.2). The three tiers are
# switched by run_benchmark.py flags; this file does not change with the tier.
# Value source: the A800 node's /ego/l3-final.yaml (the real-model config already
# validated by E3's final validation), only input.uri changes.
input: {uri: /ego/bench-data}
output: {uri: /ego/bench/out, format: lerobot_v3}
execution:
  max_gpus: 1
  num_cpu_actors: 4
  max_episodes_in_flight: 2
  backend: real
  model_worker: resident
stages:
  scene_reconstruction: {model: auto, stereo_max_image_side_px: 640}
  hand_reconstruction: {model: hamer}
  action_understanding: {model: none}   # probe reachable_multimodal -> change to openai_vlm + vlm_*
  visualization: {camera_view: true, world_view: true}
models:   # key names below match api/config.py's ModelEnvsConfig; values follow the actual node
  vipe_python: /ego/vipe-venv/bin/python
  ffs_python: /ego/roboray-venv/bin/python
  hammer_python: /ego/roboray-venv/bin/python
  vipe_weights: /ego/vipe
  ffs_weights: /ego/weights/ffs
  hammer_weights: /ego/ego-models/mano
  lerobot_python: /ego/vipe-venv/bin/python
```

```bash
git add deploy/bench/bench-l3.yaml
git commit -m "feat(bench): benchmark config sample -- real models resident, action branch decided by the probe"
```

- [ ] **Step 2: GATE gate (stop truthfully if not passed)**

Run: `bash deploy/k8s/status.sh`
Expected: `GATE=OPEN`. If `GATE=CLOSED`/`UNKNOWN`: **stop**, and truthfully write "campaign gated: no free GPU" in the final report; the first 6 completed tasks are delivered as normal -- do not fake a deployment (the repo's hard convention, rule #28).

- [ ] **Step 3: bring up the cluster + dataset + config + VLM probe (head-exec, following the E3 approach)**

```bash
bash deploy/k8s/apply.sh   # standalone cluster (spec §6: no RayJob; the pitfall is already registered)
```

Inside the cluster (head-exec, the same execution method as E3's final validation), in order:

```bash
# 1) dataset: 6 items × 60 frames (decision D-B2; on timeout reduce to 4×45 and record it)
python scripts/make_l3_dataset.py --out /ego/bench-data --episodes 6 --frames 60
# 2) config: derive /ego/bench.yaml from /ego/l3-final.yaml (input.uri -> /ego/bench-data,
#    output.uri -> /ego/bench/out; models.* values kept as-is)
# 3) VLM probe (spec §5.1; address/model name from the platform side -- the svc of the
#    qwen38 pod in the existing platform namespace)
python scripts/probe_vlm.py --base-url <VLM_BASE_URL> --model <VLM_MODEL> --api-key-env VLM_API_KEY
#    PROBE_RESULT=reachable_multimodal -> change /ego/bench.yaml's action_understanding to
#    {model: openai_vlm, vlm_base_url: <URL>, vlm_model: <NAME>} and export VLM_API_KEY in the run env;
#    any other tag -> keep model: none, and record the tag + reason in one line
```

- [ ] **Step 4: run the three-tier benchmark**

```bash
# on the same cluster run the three tiers in sequence (decisions D-B1/D-B4); expect
# 30-40 minutes of GPU time
python scripts/run_benchmark.py --config /ego/bench.yaml --bench-root /ego/bench \
    --action-model none   # pass openai_vlm if the VLM is wired
```

Expected: exit 0; `bench_report.md` shows failed=0 for all three tiers; `run-adaptive/timeline/timeline_evidence.json`'s `max_concurrent_episodes >= 3` (if it only reaches 2: truthfully record the actual value + bottleneck analysis; spec §9 does not pad numbers).

- [ ] **Step 5: collect artifacts + write the results doc + commit**

```bash
# bring back bench_report.md/json, timeline.png, timeline_evidence.json
# docs/bench/2026-10-26-bench-results.md contents:
#   1. three-tier table (bench_report.md in full or excerpted)
#   2. overlap evidence: the summarize one-liner + PNG relative link + the measured
#      max_concurrent_episodes value
#   3. VLM branch conclusion: reachable_multimodal + model name, or none + probe tag + reason
#   4. confounders and environment (USED/FREE GPUS, nodes, batch, timeout/down-batch records);
#      the load-events column = each tier's model-load count (spec §3.3 cold-start
#      evidence: all three tiers should be >0)
#   5. honest boundaries (spec §9's four items copied verbatim + newly measured items)
git add docs/bench/ deploy/bench/
git commit -m "docs(bench): real-model three-tier benchmark results -- numbers/overlap evidence/VLM branch conclusion/honest boundaries"
```

- [ ] **Step 6: teardown + GPU-return check**

```bash
bash deploy/k8s/teardown.sh
bash deploy/k8s/status.sh   # USED_GPUS should return to the platform baseline (was 2 at E3 close, both platform vLLM)
```

Expected: no residual roboray RayCluster/Pod; the results doc appends the check line.
