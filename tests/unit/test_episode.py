import json

import numpy as np
import pytest

from roboray.runtime.episode import (
    ARTIFACT_SCHEMA_VERSION,
    EpisodeDiscoveryError,
    EpisodeRecord,
    EpisodeWorkspace,
    Stage,
    StageStatus,
    discover_episodes,
)
from roboray.runtime.storage import PosixStorage


def _make_episode(root, ep_id, *, stereo=True, meta=True):
    d = root / ep_id
    d.mkdir(parents=True)
    (d / "rgb_left.mp4").write_bytes(b"fake")
    if stereo:
        (d / "rgb_right.mp4").write_bytes(b"fake")
    (d / "calibration.json").write_text("{}")
    if meta:
        (d / "meta.json").write_text(json.dumps({"episode_id": ep_id}))
    return d


def test_discover_finds_episodes_with_meta(tmp_path):
    _make_episode(tmp_path, "ep_0001")
    _make_episode(tmp_path, "ep_0002", stereo=False)
    specs = discover_episodes(PosixStorage(tmp_path))
    assert [s.episode_id for s in specs] == ["ep_0001", "ep_0002"]
    assert specs[0].has_stereo is True
    assert specs[1].has_stereo is False


def test_discover_rejects_missing_meta(tmp_path):
    _make_episode(tmp_path, "ep_0001", meta=False)
    with pytest.raises(EpisodeDiscoveryError) as exc:
        discover_episodes(PosixStorage(tmp_path))
    assert "meta.json" in str(exc.value)


def test_discover_skips_stray_files(tmp_path):
    _make_episode(tmp_path, "ep_0001")
    (tmp_path / "README.md").write_text("notes")
    (tmp_path / ".DS_Store").write_bytes(b"\x00\x00")
    specs = discover_episodes(PosixStorage(tmp_path))
    assert [s.episode_id for s in specs] == ["ep_0001"]


def test_discover_rejects_stray_dir_without_meta(tmp_path):
    _make_episode(tmp_path, "ep_0001")
    (tmp_path / "orphan").mkdir()  # a directory without meta.json: must raise
    with pytest.raises(EpisodeDiscoveryError) as exc:
        discover_episodes(PosixStorage(tmp_path))
    assert "orphan" in str(exc.value)


def test_discover_rejects_duplicate_episode_id(tmp_path):
    _make_episode(tmp_path, "ep_0001")
    d = tmp_path / "ep_0001_copy"
    d.mkdir()
    (d / "rgb_left.mp4").write_bytes(b"x")
    (d / "meta.json").write_text(json.dumps({"episode_id": "ep_0001"}))
    with pytest.raises(EpisodeDiscoveryError) as exc:
        discover_episodes(PosixStorage(tmp_path))
    assert "ep_0001" in str(exc.value)


def test_record_next_runnable_follows_state_machine():
    rec = EpisodeRecord.new("ep_0001")
    assert rec.next_runnable() == Stage.PREPARE
    rec.stages[Stage.PREPARE] = StageStatus.DONE
    # SCENE and HAND can both run after PREPARE
    assert rec.next_runnable() == Stage.SCENE
    rec.stages[Stage.SCENE] = StageStatus.DONE
    assert rec.next_runnable() == Stage.HAND
    rec.stages[Stage.HAND] = StageStatus.DONE
    assert rec.next_runnable() == Stage.FUSION
    rec.stages[Stage.FUSION] = StageStatus.DONE
    assert rec.next_runnable() == Stage.ACTION
    rec.stages[Stage.ACTION] = StageStatus.DONE
    assert rec.next_runnable() == Stage.PUBLISH
    rec.stages[Stage.PUBLISH] = StageStatus.DONE
    assert rec.next_runnable() is None
    assert rec.is_terminal()


def test_running_stage_blocks_next_runnable():
    rec = EpisodeRecord.new("ep_0001")
    rec.stages[Stage.PREPARE] = StageStatus.RUNNING
    assert rec.next_runnable() is None


def test_workspace_state_persists_and_resumes(tmp_path):
    ws = EpisodeWorkspace(tmp_path, "ep_0001")
    assert ws.load().stages[Stage.PREPARE] == StageStatus.PENDING
    ws.mark_done(Stage.PREPARE, seconds=1.5)
    ws.mark_done(Stage.SCENE, seconds=2.0)
    again = EpisodeWorkspace(tmp_path, "ep_0001").load()
    assert again.stages[Stage.PREPARE] == StageStatus.DONE
    assert again.stages[Stage.SCENE] == StageStatus.DONE
    assert again.next_runnable() == Stage.HAND
    assert again.timings["prepare"] == pytest.approx(1.5)


def test_failure_is_recorded_and_blocks_downstream(tmp_path):
    ws = EpisodeWorkspace(tmp_path, "ep_0001")
    ws.mark_failed(Stage.PREPARE, "calibration.json missing")
    rec = ws.load()
    assert rec.stages[Stage.PREPARE] == StageStatus.FAILED
    assert "calibration.json" in rec.error
    assert rec.is_terminal()          # failure terminates the episode; nothing downstream runs
    assert rec.next_runnable() is None


def test_corrupt_artifact_is_detected(tmp_path):
    ws = EpisodeWorkspace(tmp_path, "ep_0001")
    ws.write_artifact("scene.npz", {"camera_poses": np.zeros((2, 4, 4), np.float32)})
    assert ws.check_artifact("scene.npz") is True
    # corrupt file
    ws.artifact("scene.npz").write_bytes(b"not an npz")
    assert ws.check_artifact("scene.npz") is False


def test_artifact_schema_mismatch_is_detected(tmp_path):
    ws = EpisodeWorkspace(tmp_path, "ep_0001")
    np.savez(ws.artifact("scene.npz"), __schema_version__=ARTIFACT_SCHEMA_VERSION + 1)
    ws.check_artifact("scene.npz")
    assert ws.check_artifact("scene.npz") is False


def test_workspace_rejects_path_traversal_episode_id(tmp_path):
    with pytest.raises(ValueError):
        EpisodeWorkspace(tmp_path, "../evil")


# ---------- cross-run state reconcile ----------

def test_reconcile_resets_crash_running_to_pending(tmp_path):
    ws = EpisodeWorkspace(tmp_path, "ep_0001")
    ws.mark_running(Stage.SCENE)
    ws.mark_done(Stage.HAND, seconds=1.0)
    rec = ws.load()
    changes = rec.reconcile_for_new_run()
    assert (Stage.SCENE, StageStatus.RUNNING) in changes
    assert rec.stages[Stage.SCENE] == StageStatus.PENDING
    assert rec.stages[Stage.HAND] == StageStatus.DONE   # DONE is unaffected
    assert rec.error is None


def test_reconcile_resets_failed_to_pending_and_clears_error(tmp_path):
    ws = EpisodeWorkspace(tmp_path, "ep_0001")
    ws.mark_failed(Stage.PREPARE, "calibration.json missing")
    ws.mark_done(Stage.HAND, seconds=2.0)
    rec = ws.load()
    changes = rec.reconcile_for_new_run()
    assert (Stage.PREPARE, StageStatus.FAILED) in changes
    assert rec.stages[Stage.PREPARE] == StageStatus.PENDING
    assert rec.stages[Stage.HAND] == StageStatus.DONE
    assert rec.error is None


def test_reconcile_is_idempotent_on_clean_record():
    rec = EpisodeRecord.new("ep_0001")
    rec.stages[Stage.PREPARE] = StageStatus.DONE
    assert rec.reconcile_for_new_run() == []


def test_reconcile_persists_through_save_record(tmp_path):
    ws = EpisodeWorkspace(tmp_path, "ep_0001")
    ws.mark_running(Stage.SCENE)
    rec = ws.load()
    changes = rec.reconcile_for_new_run()
    assert changes
    ws.save_record(rec)
    again = ws.load()
    assert again.stages[Stage.SCENE] == StageStatus.PENDING


def test_undone_stages_lists_non_done_stages():
    rec = EpisodeRecord.new("ep_0001")
    rec.stages[Stage.PREPARE] = StageStatus.DONE
    rec.stages[Stage.SCENE] = StageStatus.RUNNING
    assert set(rec.undone_stages()) == {"scene", "hand", "fusion", "action", "publish"}


# ---------- fold-in: state schema version stamp ----------

def test_state_schema_version_mismatch_is_rejected(tmp_path):
    ws = EpisodeWorkspace(tmp_path, "ep_0001")
    ws.save_record(EpisodeRecord.new("ep_0001"))
    raw = json.loads(ws.state_file.read_text())
    raw["state_schema_version"] = 999
    ws.state_file.write_text(json.dumps(raw))
    with pytest.raises(ValueError) as exc:
        ws.load()
    assert "schema" in str(exc.value)


def test_state_without_version_still_loads_as_v1(tmp_path):
    """Backward compatible: an old state.json with no version stamp is treated as v1."""
    ws = EpisodeWorkspace(tmp_path, "ep_0001")
    raw = EpisodeRecord.new("ep_0001").to_json()
    del raw["state_schema_version"]
    ws.state_file.write_text(json.dumps(raw))
    assert ws.load().episode_id == "ep_0001"


def test_state_missing_keys_is_rejected(tmp_path):
    ws = EpisodeWorkspace(tmp_path, "ep_0001")
    ws.state_file.write_text(json.dumps({"episode_id": "ep_0001"}))  # missing stages
    with pytest.raises(ValueError) as exc:
        ws.load()
    assert "state.json" in str(exc.value)


# ---------- prepare metadata round-tripped through EpisodeRecord ----------

def test_record_loads_legacy_state_json_without_new_fields(tmp_path):
    """Re-running an **existing workspace** after an upgrade: the old state.json lacks the new fields.

    Expected: it loads fine, new fields take their defaults, no exception, and completed stage state
    is not lost.
    """
    from roboray.runtime.episode import EpisodeRecord, EpisodeWorkspace, Stage, StageStatus

    ws = EpisodeWorkspace(tmp_path, "ep_0001")
    # hand-write an "old" state.json: no has_stereo / resolution_key
    ws.state_file.write_text(json.dumps({
        "episode_id": "ep_0001",
        "state_schema_version": 1,
        "stages": {"prepare": "done", "scene": "pending", "hand": "pending",
                   "fusion": "pending", "action": "pending", "publish": "pending"},
        "timings": {"prepare": 0.5},
        "error": None,
    }))

    rec = ws.load()
    assert rec.is_done(Stage.PREPARE)
    assert rec.stages[Stage.SCENE] is StageStatus.PENDING
    assert rec.has_stereo is None
    assert rec.resolution_key is None


def test_record_prepare_persists_metadata(tmp_path):
    """record_prepare writes has_stereo / resolution_key and they round-trip through serialization."""
    from roboray.runtime.episode import EpisodeWorkspace

    ws = EpisodeWorkspace(tmp_path, "ep_0002")
    rec = ws.load()
    rec.record_prepare(has_stereo=True, width=320, height=240)
    ws.save_record(rec)

    again = ws.load()
    assert again.has_stereo is True
    assert again.resolution_key == "320x240"


def test_record_prepare_persists_degraded_info(tmp_path):
    """Degraded info is returned via record_prepare (the driver no longer reads summary.json).

    ``degraded`` / ``degrade_reason`` are backfilled into the EpisodeRecord together with
    prepare's stage_result (consumed by the manifest), instead of the driver reading the
    workspace ``prepared/summary.json``. Old records missing these fields must tolerate None.
    """
    from roboray.runtime.episode import EpisodeWorkspace

    ws = EpisodeWorkspace(tmp_path, "ep_0002")
    rec = ws.load()
    rec.record_prepare(
        has_stereo=False, width=64, height=64,
        degraded=True, degrade_reason="stereo_frame_mismatch",
    )
    ws.save_record(rec)

    again = ws.load()
    assert again.degraded is True
    assert again.degrade_reason == "stereo_frame_mismatch"

    # old state.json (no degraded field) tolerates None without raising
    legacy = ws.state_file.read_text()
    import json as _json

    raw = _json.loads(legacy)
    raw.pop("degraded", None)
    raw.pop("degrade_reason", None)
    ws.state_file.write_text(_json.dumps(raw))
    old = ws.load()
    assert old.degraded is None
    assert old.degrade_reason is None


def test_record_prepare_persists_alignment_info(tmp_path):
    """alignment/align_dropped are returned via record_prepare and round-trip through serialization.

    ``alignment`` defaults to None (old callers not passing it -> no new semantics written),
    ``align_dropped`` defaults to 0; old state.json missing these fields tolerates None/0
    without raising.
    """
    from roboray.runtime.episode import EpisodeWorkspace

    ws = EpisodeWorkspace(tmp_path, "ep_0002")
    rec = ws.load()
    rec.record_prepare(
        has_stereo=True, width=64, height=64,
        alignment="timestamps", align_dropped=4,
    )
    ws.save_record(rec)

    again = ws.load()
    assert again.alignment == "timestamps"
    assert again.align_dropped == 4

    # old state.json (no alignment field) tolerates None/0 without raising
    raw = json.loads(ws.state_file.read_text())
    raw.pop("alignment", None)
    raw.pop("align_dropped", None)
    ws.state_file.write_text(json.dumps(raw))
    old = ws.load()
    assert old.alignment is None
    assert old.align_dropped == 0


# ---------- node loss -> the episode re-runs in full ----------

def test_reconcile_resets_all_stages_when_assigned_node_is_gone():
    """Node loss -> re-run in full; must not only flip RUNNING back to PENDING.

    The workspace vanished with the node: only resetting RUNNING would leave a false
    "prepare DONE" and the next run would crash reading a prepared/ that no longer exists.
    """
    from roboray.runtime.episode import EpisodeRecord, Stage, StageStatus

    rec = EpisodeRecord.new("ep_0001")
    rec.stages[Stage.PREPARE] = StageStatus.DONE
    rec.stages[Stage.SCENE] = StageStatus.DONE
    rec.stages[Stage.HAND] = StageStatus.DONE
    rec.assigned_node = "node-gone"

    changes = rec.reconcile_for_new_run(live_nodes={"node-alive"})
    assert rec.stages[Stage.PREPARE] is StageStatus.PENDING
    assert rec.stages[Stage.SCENE] is StageStatus.PENDING
    assert rec.stages[Stage.HAND] is StageStatus.PENDING
    assert rec.assigned_node is None
    assert len(changes) == 3


def test_reconcile_keeps_done_stages_when_node_still_live():
    """Node still live -> keep completed stages (the resume semantics are unchanged)."""
    from roboray.runtime.episode import EpisodeRecord, Stage, StageStatus

    rec = EpisodeRecord.new("ep_0001")
    rec.stages[Stage.PREPARE] = StageStatus.DONE
    rec.assigned_node = "node-1"

    changes = rec.reconcile_for_new_run(live_nodes={"node-1"})
    assert rec.stages[Stage.PREPARE] is StageStatus.DONE
    assert rec.assigned_node == "node-1"
    assert changes == []


def test_reconcile_without_live_nodes_does_not_touch_assignment():
    """Not passing live_nodes (old caller) -> no node validation, behavior unchanged."""
    from roboray.runtime.episode import EpisodeRecord, Stage, StageStatus

    rec = EpisodeRecord.new("ep_0001")
    rec.stages[Stage.PREPARE] = StageStatus.DONE
    rec.assigned_node = "whatever"

    changes = rec.reconcile_for_new_run()
    assert rec.stages[Stage.PREPARE] is StageStatus.DONE
    assert rec.assigned_node == "whatever"
    assert changes == []
