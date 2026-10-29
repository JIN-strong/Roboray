"""manifest timeline persistence (spec 2026-10-26 §4.2 prerequisite)."""
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
    # additive invariant: existing keys preserved unchanged
    assert data["summary"]["n_episodes"] == 0
    assert data["scheduler_snapshots"] == []


def test_timeline_defaults_to_empty(tmp_path):
    path = write_manifest(str(tmp_path), _request(), {}, [], 1.0)
    data = json.loads(Path(path).read_text())
    assert data["timeline"] == []


def test_manifest_carries_alignment_semantics(tmp_path):
    """Manifest entries carry prepare's alignment/align_dropped.

    Backfilled via record_prepare (the driver does not read the workspace); default None/0
    (old-record compatible).
    """
    from roboray.runtime.episode import EpisodeRecord

    rec = EpisodeRecord.new("ep_0001")
    rec.record_prepare(has_stereo=True, width=64, height=64,
                       alignment="timestamps", align_dropped=4)
    rec_old = EpisodeRecord.new("ep_0002")
    rec_old.record_prepare(has_stereo=False, width=64, height=64)

    path = write_manifest(str(tmp_path), _request(),
                          {"ep_0001": rec.to_json(), "ep_0002": rec_old.to_json()}, [])
    data = json.loads(Path(path).read_text())
    entries = {e["episode_id"]: e for e in data["episodes"]}
    assert entries["ep_0001"]["alignment"] == "timestamps"
    assert entries["ep_0001"]["align_dropped"] == 4
    # Old record (new params not passed) defaults None/0, no failure on a missing key
    assert entries["ep_0002"]["alignment"] is None
    assert entries["ep_0002"]["align_dropped"] == 0
