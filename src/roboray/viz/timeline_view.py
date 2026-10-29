"""Cross-stage overlap evidence: derive the max concurrent episode count and the
concurrency shape from the timeline event stream (spec §4.1).

Pure functions, no matplotlib dependency — easy to unit test; rendering lives in
render_gantt (same file, lazily loaded).
"""

import json
from pathlib import Path

__all__ = ["overlap_evidence", "render_gantt", "render_from_manifest", "summarize"]

_STAGE_ORDER = ("prepare", "scene", "hand", "fusion", "action", "publish")


def _intervals(timeline: list) -> tuple[list, int]:
    """Pair start/end into (ep, stage) intervals; returns (intervals, open count).

    - Idle/ops events (episode_id is None) are skipped.
    - Duplicate starts and ends without a start are ignored, not raised.
    - Unclosed intervals are truncated at the last event time and counted as open.
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
    """Concurrency-peak determination. The peak is the snapshot at the **first**
    start time with the max concurrent episode count; max_concurrent_stages is the
    max number of parallel intervals over any boundary time in the run."""
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


_STAGE_COLORS = {
    "prepare": "#7f7f7f", "scene": "#1f77b4", "hand": "#d62728",
    "fusion": "#ff7f0e", "action": "#2ca02c", "publish": "#9467bd",
}


def render_gantt(timeline: list, out_png, title: str = "Episode x Stage timeline") -> Path:
    """Horizontal bar chart with one row per episode; stages color-coded, x axis in
    seconds. Agg backend."""
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
    """Read the timeline from manifest.json -> write a PNG + the evidence JSON,
    return the evidence."""
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
