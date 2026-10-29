"""Batch run index: manifest.json."""

import json
from datetime import datetime, timezone
from pathlib import Path

__all__ = ["write_manifest"]


def write_manifest(
    output_root: str,
    request,
    per_episode: dict,
    scheduler_snapshots: list,
    total_seconds: float = 0.0,
    timeline: list = (),
) -> Path:
    out = Path(output_root)
    out.mkdir(parents=True, exist_ok=True)
    episodes = []
    for ep_id, rec in sorted(per_episode.items()):
        payload = rec if isinstance(rec, dict) else rec.to_json()
        entry = {
            "episode_id": ep_id,
            "stages": payload.get("stages", {}),
            "timings": payload.get("timings", {}),
            "error": payload.get("error"),
        }
        # Merge prepare's degradation info — **read from the record, not the workspace
        # summary.json**. Hard invariant: the driver process does not read workspace files
        # (unreadable on multi-node setups); the degradation info is returned via the Actor
        # return value (record_prepare) and the manifest only consumes it.
        # `degraded` defaults to False: an old record (without the field) counts as
        # non-degraded, no failure on a missing key.
        entry["degraded"] = bool(payload.get("degraded", False))
        entry["degrade_reason"] = payload.get("degrade_reason")
        # Prepare's alignment and timestamp-dropped frame count (default None/0, compatible
        # with old records).
        entry["alignment"] = payload.get("alignment")
        entry["align_dropped"] = payload.get("align_dropped", 0)
        episodes.append(entry)
    failed = [e for e in episodes if e["error"]]
    manifest = {
        "format": request.output.format,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input": request.input.uri,
        "output": request.output.uri,
        "execution": {
            "max_gpus": request.execution.max_gpus,
            "backend": request.execution.backend,
        },
        "summary": {
            "n_episodes": len(episodes),
            "failed": len(failed),
            "total_seconds": round(float(total_seconds), 6),
        },
        "episodes": episodes,
        "scheduler_snapshots": list(scheduler_snapshots),
        "timeline": list(timeline),
    }
    path = out / "manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    return path
