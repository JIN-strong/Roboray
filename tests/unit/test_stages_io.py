import json

import numpy as np
import pytest

from roboray.stages.prepare import decode_video, load_calibration, run_prepare
from scripts.make_synthetic_ego import generate_dataset


def test_decode_video_returns_frames(tmp_path):
    generate_dataset(tmp_path, n_episodes=1, n_frames=7, size=(64, 48), seed=0)
    frames = decode_video(tmp_path / "ep_0001" / "rgb_left.mp4")
    assert frames.shape == (7, 48, 64, 3)
    assert frames.dtype == np.uint8


def test_load_calibration_normalises_fields(tmp_path):
    generate_dataset(tmp_path, n_episodes=1, n_frames=4, size=(64, 48), seed=0)
    calib = load_calibration(tmp_path / "ep_0001" / "calibration.json")
    assert calib["K"].shape == (3, 3)
    assert calib["T_left_to_right"].shape == (4, 4)
    assert calib["fps"] == 30.0


def test_load_calibration_rejects_missing_file(tmp_path):
    with pytest.raises(ValueError) as exc:
        load_calibration(tmp_path / "nope.json")
    assert "calibration.json" in str(exc.value)


def test_load_calibration_rejects_incomplete_file(tmp_path):
    p = tmp_path / "calibration.json"
    p.write_text(json.dumps({"K": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}))  # missing fps/dist
    with pytest.raises(ValueError) as exc:
        load_calibration(p)
    assert "calibration.json" in str(exc.value)


def test_run_prepare_writes_frames_and_calib(tmp_path):
    src = tmp_path / "src"
    ws_root = tmp_path / "ws"
    generate_dataset(src, n_episodes=1, n_frames=6, size=(64, 48), seed=0)
    info = run_prepare(
        storage_root=str(src),
        workspace_root=str(ws_root),
        episode_rel_dir="ep_0001",
        episode_id="ep_0001",
    )
    assert info["n_frames"] == 6
    assert info["has_stereo"] is True
    assert info["degraded"] is False
    assert np.load(ws_root / "ep_0001" / "prepared" / "frames_left.npy").shape[0] == 6
    calib = json.loads((ws_root / "ep_0001" / "prepared" / "calib.json").read_text())
    assert calib["has_stereo"] is True and calib["n_frames"] == 6


def test_run_prepare_flags_stereo_frame_mismatch(tmp_path):
    """Right-eye frame count less than left-eye -> take the shorter and record the
    degradation without crashing."""
    import cv2

    src = tmp_path / "src"
    ws_root = tmp_path / "ws"
    generate_dataset(src, n_episodes=1, n_frames=6, size=(64, 48), seed=0)
    # rewrite the right eye to 4 frames
    ep = src / "ep_0001"
    ep.joinpath("rgb_right.mp4").unlink()
    frames = [np.zeros((48, 64, 3), np.uint8)] * 4
    w = cv2.VideoWriter(str(ep / "rgb_right.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 48))
    for f in frames:
        w.write(f)
    w.release()
    info = run_prepare(
        storage_root=str(src), workspace_root=str(ws_root),
        episode_rel_dir="ep_0001", episode_id="ep_0001",
    )
    assert info["n_frames"] == 4
    assert info["degraded"] is True
    assert info["degrade_reason"] == "stereo_frame_mismatch"
    assert info["has_stereo"] is False  # stereo unusable, fall back to monocular


def test_run_prepare_missing_calibration_raises_readable_error(tmp_path):
    """calibration.json missing -> readable error (the Driver marks it FAILED)."""
    src = tmp_path / "src"
    generate_dataset(src, n_episodes=1, n_frames=4, size=(64, 48), seed=0)
    (src / "ep_0001" / "calibration.json").unlink()
    with pytest.raises(ValueError) as exc:
        run_prepare(
            storage_root=str(src), workspace_root=str(tmp_path / "ws"),
            episode_rel_dir="ep_0001", episode_id="ep_0001",
        )
    assert "calibration.json" in str(exc.value)


def test_run_prepare_writes_summary_json(tmp_path):
    """prepare writes summary.json, the single source of degradation info."""
    src = tmp_path / "src"
    ws_root = tmp_path / "ws"
    generate_dataset(src, n_episodes=1, n_frames=5, size=(64, 48), seed=0)
    info = run_prepare(
        storage_root=str(src), workspace_root=str(ws_root),
        episode_rel_dir="ep_0001", episode_id="ep_0001",
    )
    summary = json.loads((ws_root / "ep_0001" / "prepared" / "summary.json").read_text())
    assert summary == info
    assert summary["has_stereo"] is True and summary["degraded"] is False


def test_run_prepare_writes_degraded_summary_and_no_orphan_right_frames(tmp_path):
    """When stereo frame-count mismatch degrades, the summary records the degradation,
    and no orphan frames_right.npy is written."""
    import cv2

    src = tmp_path / "src"
    ws_root = tmp_path / "ws"
    generate_dataset(src, n_episodes=1, n_frames=6, size=(64, 48), seed=0)
    ep = src / "ep_0001"
    ep.joinpath("rgb_right.mp4").unlink()
    frames = [np.zeros((48, 64, 3), np.uint8)] * 3
    w = cv2.VideoWriter(str(ep / "rgb_right.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 48))
    for f in frames:
        w.write(f)
    w.release()
    info = run_prepare(
        storage_root=str(src), workspace_root=str(ws_root),
        episode_rel_dir="ep_0001", episode_id="ep_0001",
    )
    assert info["degraded"] is True and info["degrade_reason"] == "stereo_frame_mismatch"
    summary = json.loads((ws_root / "ep_0001" / "prepared" / "summary.json").read_text())
    assert summary["degraded"] is True
    # after degradation frames_right.npy must not exist (no orphan files left behind)
    assert not (ws_root / "ep_0001" / "prepared" / "frames_right.npy").exists()
    calib = json.loads((ws_root / "ep_0001" / "prepared" / "calib.json").read_text())
    assert calib["has_stereo"] is False


def test_prepared_has_stereo_matches_calib(tmp_path):
    """prepared_has_stereo / prepared_resolution_key share the same source as the prepared
    calib.

    With stereo has_stereo=True, and the resolution tier matches the calib width/height
    ("64x64").
    """
    from roboray.stages.scene import prepared_has_stereo, prepared_resolution_key

    src = tmp_path / "src"
    ws_root = tmp_path / "ws"
    generate_dataset(src, n_episodes=1, n_frames=4, size=(64, 48), seed=0)
    run_prepare(
        storage_root=str(src), workspace_root=str(ws_root),
        episode_rel_dir="ep_0001", episode_id="ep_0001",
    )
    assert prepared_has_stereo(str(ws_root), "ep_0001") is True
    assert prepared_resolution_key(str(ws_root), "ep_0001") == "64x64"


def test_prepared_single_eye_has_no_frames_right(tmp_path):
    """Monocular input (no right eye) -> has_stereo=False, the Actor does not attach
    frames_right."""
    from roboray.ray_runtime.actors import read_prepared_frames
    from roboray.stages.scene import prepared_has_stereo, prepared_resolution_key

    src = tmp_path / "src"
    ws_root = tmp_path / "ws"
    generate_dataset(src, n_episodes=1, n_frames=4, size=(64, 48), seed=0)
    (src / "ep_0001" / "rgb_right.mp4").unlink()
    run_prepare(
        storage_root=str(src), workspace_root=str(ws_root),
        episode_rel_dir="ep_0001", episode_id="ep_0001",
    )
    assert prepared_has_stereo(str(ws_root), "ep_0001") is False
    assert prepared_resolution_key(str(ws_root), "ep_0001") == "64x64"
    frames, calib = read_prepared_frames(str(ws_root), "ep_0001")
    assert calib["has_stereo"] is False
    assert "frames_right" not in frames


def test_gpu_payload_hand_carries_max_hands_scalar(tmp_path):
    """Config surface end to end (driver segment): HandStageConfig.max_hands enters the
    hand payload as a scalar (same style as inference_fps — _gpu_payload carries only paths
    and scalars; the Actor run_task then injects calib, and the adapter's _extra forwards it
    into the script's extra)."""
    from roboray.api.config import ReconstructionRequest
    from roboray.runtime.driver import DriverConfig, PipelineDriver
    from roboray.scheduling.scheduler import QueueItem

    def _driver(hand_cfg):
        req = ReconstructionRequest.from_dict({
            "input": {"uri": str(tmp_path)},
            "output": {"uri": str(tmp_path / "out")},
            "execution": {"backend": "test"},
            "stages": {"hand_reconstruction": hand_cfg},
        })
        return PipelineDriver(
            DriverConfig(storage_root=str(tmp_path), workspace_root=str(tmp_path / "ws"),
                         output_root=str(tmp_path / "out")),
            req, gpu_actors=[], cpu_actors=[],
        )

    class _Ws:  # _gpu_payload only uses artifact() to get the path string
        def artifact(self, name):
            return tmp_path / "ws" / "artifacts" / name

    item = QueueItem("hand", "ep_0001", "64x64", "hamer")
    payload = _driver({"max_hands": 6})._gpu_payload("hand", None, item, _Ws())
    assert payload["max_hands"] == 6
    assert payload["inference_fps"] == 30  # existing scalars on the same chain unaffected
    # The default 4 is also passed through (same passthrough semantics as inference_fps's
    # default 30; script-side behavior unchanged)
    payload_default = _driver({})._gpu_payload("hand", None, item, _Ws())
    assert payload_default["max_hands"] == 4
    # Hard constraint: only paths and scalars
    for key, value in payload.items():
        assert not isinstance(value, np.ndarray), f"payload[{key!r}] must not be an array"


def test_gpu_payload_for_stereo_carries_resize_knob_only_scalars(tmp_path):
    """Seam: the Driver's GPU payload construction (the shared `_gpu_payload` used by both
    the serial `_execute_one` and the pipeline loop) **carries only paths and scalars** —
    frame arrays are read from disk by the Actor on its own node; the configured
    `stereo_max_image_side_px` travels as a scalar in the payload, and `auto` is resolved to
    the stereo model via the record. Stereo frame promotion is preserved by
    `read_prepared_frames`."""
    from roboray.api.config import ReconstructionRequest
    from roboray.runtime.driver import DriverConfig, PipelineDriver
    from roboray.ray_runtime.actors import read_prepared_frames
    from roboray.scheduling.scheduler import QueueItem

    src = tmp_path / "src"
    ws_root = tmp_path / "ws"
    generate_dataset(src, n_episodes=1, n_frames=4, size=(64, 48), seed=0)
    run_prepare(
        storage_root=str(src), workspace_root=str(ws_root),
        episode_rel_dir="ep_0001", episode_id="ep_0001",
    )
    req = ReconstructionRequest.from_dict(
        {
            "input": {"uri": str(src)},
            "output": {"uri": str(tmp_path / "out")},
            "execution": {"backend": "test"},
            "stages": {"scene_reconstruction": {"model": "auto",
                                                "stereo_max_image_side_px": 32}},
        }
    )
    # Driver routing only reads the record — prepare backfills metadata via `_complete`, it
    # no longer reads prepared/calib.json. Here the production path writes the prepare
    # result into the record and into the driver's **in-memory authoritative record**
    # (run() loads from state.json; this test builds the driver directly, so the records are
    # filled manually, consistent with the production contract: no workspace re-read).
    from roboray.runtime.episode import EpisodeWorkspace

    ws = EpisodeWorkspace(ws_root, "ep_0001")
    rec = ws.load()
    rec.record_prepare(has_stereo=True, width=64, height=64)
    ws.save_record(rec)
    driver = PipelineDriver(
        DriverConfig(storage_root=str(src), workspace_root=str(ws_root),
                     output_root=str(tmp_path / "out")),
        req, gpu_actors=[], cpu_actors=[],
    )
    driver.records = {"ep_0001": rec}
    payload = driver._gpu_payload(
        "scene", None, QueueItem("scene", "ep_0001", "64x64", "auto"), ws
    )
    assert payload["model_name"] == "fast_foundation_stereo"  # auto -> stereo resolution
    assert payload["kind"] == "scene"
    assert payload["episode_id"] == "ep_0001"
    assert payload["workspace_root"] == str(ws_root)
    assert payload["out_path"] == str(ws.artifact("scene.npz"))
    # Hard constraint: payload carries only paths and scalars
    for key, value in payload.items():
        assert not isinstance(value, np.ndarray), f"payload[{key!r}] must not be an array"
    # stereo knob travels as a payload scalar (run_task then writes it into calib)
    assert payload["stereo_max_image_side_px"] == 32
    # Stereo frame promotion is preserved by the Actor's disk read
    frames, calib = read_prepared_frames(str(ws_root), "ep_0001")
    assert calib["has_stereo"] is True
    assert "frames_right" in frames


def test_run_prepare_zero_frame_video_raises(tmp_path):
    """Zero-frame mp4 -> explicit error, no half-baked artifacts."""
    import cv2

    src = tmp_path / "src" / "ep_0001"
    src.mkdir(parents=True)
    w = cv2.VideoWriter(str(src / "rgb_left.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 48))
    w.release()
    (src / "calibration.json").write_text(json.dumps(
        {"K": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "dist": [0] * 5, "fps": 30.0}
    ))
    (src / "meta.json").write_text(json.dumps({"episode_id": "ep_0001"}))
    with pytest.raises(ValueError) as exc:
        run_prepare(
            storage_root=str(tmp_path / "src"), workspace_root=str(tmp_path / "ws"),
            episode_rel_dir="ep_0001", episode_id="ep_0001",
        )
    assert "rgb_left.mp4" in str(exc.value)


def test_align_by_timestamps_rules():
    from roboray.stages.prepare import align_by_timestamps

    fps = 30.0
    ts_l = np.arange(10) / fps                       # 0.0 … 0.3
    ts_r = np.arange(10) / fps + 0.3                 # right eye lags 0.3s = 9 frames as a whole
    il, ir, dropped = align_by_timestamps(ts_l, ts_r, fps)
    assert len(il) == len(ir) == 1                   # only the t≈0.3 pair falls in tolerance
    assert il[0] == 9 and ir[0] == 0
    assert dropped == 9
    # No drift: everything pairs
    il, ir, dropped = align_by_timestamps(ts_l, ts_l.copy(), fps)
    assert len(il) == 10 and dropped == 0


def test_prepare_timestamp_alignment_end_to_end(tmp_path):
    """Timestamp drift between the two eyes -> after alignment left/right share indices;
    manifest semantics alignment=timestamps.

    The right eye drops 1 of every 3 frames (12 left frames vs 10 right), meta.json carries
    the corresponding timestamps -> after pairing prepared/frames_left.npy.shape[0] == pair
    count (8), summary records alignment="timestamps", align_dropped=4. has_stereo stays
    True: timestamp-dropped frames are not stereo degradation.
    """
    import cv2

    fps = 30.0
    src = tmp_path / "src"
    ws_root = tmp_path / "ws"
    ep = src / "ep_0001"
    ep.mkdir(parents=True)
    # 12 left frames; 10 right frames (the whole timeline drops 1 of every 3: take left
    # times 0,1,3,4,6,7,9,10,12,13)
    left_frames = [np.full((48, 64, 3), i, np.uint8) for i in range(12)]
    kept = [i for i in range(15) if i % 3 != 2]
    right_frames = [np.full((48, 64, 3), 100 + i, np.uint8) for i in range(10)]

    def _write_video(path, frames):
        w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 48))
        for f in frames:
            w.write(f)
        w.release()

    _write_video(ep / "rgb_left.mp4", left_frames)
    _write_video(ep / "rgb_right.mp4", right_frames)
    (ep / "calibration.json").write_text(json.dumps(
        {"K": [[51.2, 0, 32], [0, 38.4, 24], [0, 0, 1]], "dist": [0] * 5,
         "T_left_to_right": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
         "fps": fps, "width": 64, "height": 48}
    ))
    (ep / "meta.json").write_text(json.dumps({
        "episode_id": "ep_0001",
        "timestamps_left": [i / fps for i in range(12)],
        "timestamps_right": [kept[i] / fps for i in range(10)],
    }))
    info = run_prepare(
        storage_root=str(src), workspace_root=str(ws_root),
        episode_rel_dir="ep_0001", episode_id="ep_0001",
    )
    assert info["alignment"] == "timestamps"
    assert info["align_dropped"] == 4
    # timestamp-dropped frames are not stereo degradation: has_stereo stays True,
    # degraded=False
    assert info["has_stereo"] is True
    assert info["degraded"] is False
    assert info["n_frames"] == 8
    # pair count == prepared frame count; equal left/right indices are synchronized pairs
    assert np.load(ws_root / "ep_0001" / "prepared" / "frames_left.npy").shape[0] == 8
    assert np.load(ws_root / "ep_0001" / "prepared" / "frames_right.npy").shape[0] == 8
    summary = json.loads((ws_root / "ep_0001" / "prepared" / "summary.json").read_text())
    assert summary["alignment"] == "timestamps"
    assert summary["align_dropped"] == 4
    calib = json.loads((ws_root / "ep_0001" / "prepared" / "calib.json").read_text())
    assert calib["has_stereo"] is True and calib["n_frames"] == 8


def test_prepare_timestamp_alignment_falls_back_to_truncation(tmp_path):
    """Regression: no meta timestamps -> the current truncation path, alignment="truncation"."""
    import cv2

    src = tmp_path / "src"
    ws_root = tmp_path / "ws"
    ep = src / "ep_0001"
    ep.mkdir(parents=True)
    left_frames = [np.full((48, 64, 3), i, np.uint8) for i in range(12)]
    right_frames = [np.full((48, 64, 3), 100 + i, np.uint8) for i in range(10)]

    def _write_video(path, frames):
        w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 48))
        for f in frames:
            w.write(f)
        w.release()

    _write_video(ep / "rgb_left.mp4", left_frames)
    _write_video(ep / "rgb_right.mp4", right_frames)
    (ep / "calibration.json").write_text(json.dumps(
        {"K": [[51.2, 0, 32], [0, 38.4, 24], [0, 0, 1]], "dist": [0] * 5,
         "T_left_to_right": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
         "fps": 30.0, "width": 64, "height": 48}
    ))
    # meta.json only carries episode_id, no timestamps (the generator's default shape)
    (ep / "meta.json").write_text(json.dumps({"episode_id": "ep_0001"}))
    info = run_prepare(
        storage_root=str(src), workspace_root=str(ws_root),
        episode_rel_dir="ep_0001", episode_id="ep_0001",
    )
    assert info["alignment"] == "truncation"
    assert info["align_dropped"] == 0
    assert info["degraded"] is True
    assert info["degrade_reason"] == "stereo_frame_mismatch"
    assert info["n_frames"] == 10
    summary = json.loads((ws_root / "ep_0001" / "prepared" / "summary.json").read_text())
    assert summary["alignment"] == "truncation"


def test_prepare_timestamp_nested_list_falls_back_to_truncation(tmp_path):
    """Hardening: meta.json timestamps as a nested list -> fall back to truncation without
    raising.

    [[0.0], [0.033], ...] passes the length/isfinite checks but has ndim=2, and searchsorted
    would blow up ("object too deep") — the ndim check must explicitly fall back, not leave
    the crash to downstream.
    """
    import cv2

    fps = 30.0
    src = tmp_path / "src"
    ws_root = tmp_path / "ws"
    ep = src / "ep_0001"
    ep.mkdir(parents=True)
    left_frames = [np.full((48, 64, 3), i, np.uint8) for i in range(12)]
    kept = [i for i in range(15) if i % 3 != 2]
    right_frames = [np.full((48, 64, 3), 100 + i, np.uint8) for i in range(10)]

    def _write_video(path, frames):
        w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 48))
        for f in frames:
            w.write(f)
        w.release()

    _write_video(ep / "rgb_left.mp4", left_frames)
    _write_video(ep / "rgb_right.mp4", right_frames)
    (ep / "calibration.json").write_text(json.dumps(
        {"K": [[51.2, 0, 32], [0, 38.4, 24], [0, 0, 1]], "dist": [0] * 5,
         "T_left_to_right": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
         "fps": fps, "width": 64, "height": 48}
    ))
    # Nested list: passes the length checks, but ndim=2 must not be treated as 1D timestamps
    (ep / "meta.json").write_text(json.dumps({
        "episode_id": "ep_0001",
        "timestamps_left": [[i / fps] for i in range(12)],
        "timestamps_right": [[kept[i] / fps] for i in range(10)],
    }))
    info = run_prepare(
        storage_root=str(src), workspace_root=str(ws_root),
        episode_rel_dir="ep_0001", episode_id="ep_0001",
    )
    assert info["alignment"] == "truncation"
    assert info["align_dropped"] == 0
    assert info["degraded"] is True
    assert info["degrade_reason"] == "stereo_frame_mismatch"
    assert np.load(ws_root / "ep_0001" / "prepared" / "frames_left.npy").shape[0] == 10


def test_prepare_timestamp_zero_pairs_falls_back_to_truncation(tmp_path):
    """Hardening: dual-clock base offset of +5s -> 0 pairs -> fall back to truncation with
    monocular degradation.

    With zero pairs the two views cannot be aligned as stereo at all: alignment="truncation",
    truncation to the current frame counts, degrade_reason="timestamp_no_pairs",
    degraded=True (has_stereo=False), and no empty frames_left.npy is produced.
    """
    import cv2

    fps = 30.0
    src = tmp_path / "src"
    ws_root = tmp_path / "ws"
    ep = src / "ep_0001"
    ep.mkdir(parents=True)
    left_frames = [np.full((48, 64, 3), i, np.uint8) for i in range(12)]
    right_frames = [np.full((48, 64, 3), 100 + i, np.uint8) for i in range(10)]

    def _write_video(path, frames):
        w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 48))
        for f in frames:
            w.write(f)
        w.release()

    _write_video(ep / "rgb_left.mp4", left_frames)
    _write_video(ep / "rgb_right.mp4", right_frames)
    (ep / "calibration.json").write_text(json.dumps(
        {"K": [[51.2, 0, 32], [0, 38.4, 24], [0, 0, 1]], "dist": [0] * 5,
         "T_left_to_right": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
         "fps": fps, "width": 64, "height": 48}
    ))
    # Right-eye clock base offset +5s -> 5s away from every left time >> tolerance -> 0 pairs
    (ep / "meta.json").write_text(json.dumps({
        "episode_id": "ep_0001",
        "timestamps_left": [i / fps for i in range(12)],
        "timestamps_right": [5.0 + i / fps for i in range(10)],
    }))
    info = run_prepare(
        storage_root=str(src), workspace_root=str(ws_root),
        episode_rel_dir="ep_0001", episode_id="ep_0001",
    )
    assert info["alignment"] == "truncation"
    assert info["align_dropped"] == 0
    assert info["degraded"] is True
    assert info["degrade_reason"] == "timestamp_no_pairs"
    assert info["has_stereo"] is False
    assert info["n_frames"] == 10
    # no empty frame table: truncate to the shorter stream by the current frame counts
    assert np.load(ws_root / "ep_0001" / "prepared" / "frames_left.npy").shape[0] == 10
    assert not (ws_root / "ep_0001" / "prepared" / "frames_right.npy").exists()


def _minimal_request():
    from roboray.api.config import ReconstructionRequest

    return ReconstructionRequest.from_dict({
        "input": {"uri": "data/synthetic"},
        "output": {"uri": "output/run-test", "format": "lerobot_v3"},
        "execution": {"backend": "test"},
    })


def test_driver_does_not_read_prepared_calib(tmp_path, monkeypatch):
    """Sibling of the "readable error / missing calibration" tests: driver routing must not
    read prepared/calib.json.

    The hard constraint is "the driver does not touch the workspace"; `_scene_model` and
    `_resolution_key` must depend only on the EpisodeRecord.
    """
    import roboray.runtime.driver as drv
    from roboray.runtime.episode import EpisodeRecord, EpisodeWorkspace

    ws = EpisodeWorkspace(tmp_path, "ep_0001")
    rec = EpisodeRecord.new("ep_0001")
    rec.record_prepare(has_stereo=True, width=320, height=240)
    ws.save_record(rec)

    def _forbidden(*a, **k):
        raise AssertionError("driver must not read prepared/calib.json")

    monkeypatch.setattr(drv, "prepared_resolution_key", _forbidden, raising=False)

    d = drv.PipelineDriver.__new__(drv.PipelineDriver)
    d.config = drv.DriverConfig(
        storage_root=str(tmp_path), workspace_root=str(tmp_path), output_root=str(tmp_path)
    )
    d.specs = {}
    d.request = _minimal_request()
    # Driver routing reads the **in-memory authoritative record** (run() loads from
    # state.json once and does not re-read during the run). Here the records are filled
    # manually, consistent with the production contract.
    d.records = {"ep_0001": rec}
    assert d._scene_model("ep_0001") == "fast_foundation_stereo"
    assert d._resolution_key("ep_0001") == "320x240"


def test_serial_execute_one_backfills_prepare_metadata(tmp_path, monkeypatch):
    """After the serial baseline `_execute_one` runs prepare successfully, it backfills
    has_stereo/resolution/degradation metadata (same source as the `_complete` parallel
    path).

    Without this step, `rec.has_stereo` stays None and `_scene_model` falls back to guessing
    from the input: with stereo input but prepare already degraded to monocular it would
    wrongly pick fast_foundation_stereo (the actor reads calib has_stereo=False -> no
    frames_right -> FFS fails -> the scene stage fails).
    """
    from roboray.api.config import ReconstructionRequest
    from roboray.runtime.driver import DriverConfig, PipelineDriver
    from roboray.runtime.episode import EpisodeSpec, EpisodeWorkspace, Stage

    src = tmp_path / "src"
    ws_root = tmp_path / "ws"
    generate_dataset(src, n_episodes=1, n_frames=4, size=(64, 48), seed=0)  # stereo input

    ws = EpisodeWorkspace(ws_root, "ep_0001")
    rec = ws.load()
    assert rec.has_stereo is None  # prepare not yet run, metadata not yet backfilled

    req = ReconstructionRequest.from_dict({
        "input": {"uri": str(src)},
        "output": {"uri": str(tmp_path / "out"), "format": "lerobot_v3"},
        "execution": {"backend": "test"},
        "stages": {"scene_reconstruction": {"model": "auto"}},
    })
    spec = EpisodeSpec(episode_id="ep_0001", rel_dir="ep_0001", has_stereo=True)

    d = PipelineDriver(
        DriverConfig(storage_root=str(src), workspace_root=str(ws_root),
                     output_root=str(tmp_path / "out"), serial_baseline=True),
        req, gpu_actors=[], cpu_actors=[],
    )
    d.records = {"ep_0001": rec}
    d.specs = {"ep_0001": spec}
    d._t0 = 0.0

    class _FakeRemote:
        """Plays the ObjectRef: the return value of `actor.run_task.remote(payload)`."""

        def __init__(self, result):
            self.result = result

        def remote(self, payload):
            return self

    class _FakeCpu:
        def __init__(self, result):
            self.run_task = _FakeRemote(result)

    result = {
        "episode_id": "ep_0001", "kind": "prepare", "seconds": 0.01, "ok": True,
        "error": None,
        "stage_result": {
            "episode_id": "ep_0001", "n_frames": 4, "has_stereo": False,
            "degraded": True, "degrade_reason": "stereo_frame_mismatch",
            "width": 64, "height": 48,
        },
    }
    monkeypatch.setattr(d, "_get_result", lambda ref: ref.result)
    d.cpu_actors = [_FakeCpu(result)]

    assert d._execute_one(spec, Stage.PREPARE) is True
    # A successful serial prepare must backfill (same source as the `_complete` parallel path)
    assert d.records["ep_0001"].has_stereo is False
    assert d.records["ep_0001"].degraded is True
    assert d.records["ep_0001"].resolution_key is not None
    # Degraded monocular -> pick vipe (guessing has_stereo=True from the input would wrongly
    # pick FFS)
    assert d._scene_model("ep_0001") == "vipe"


def test_publish_accepts_legacy_4d_state(tmp_path):
    """An action.npz with old 4D state can still be published (the publish reshape branch is
    kept).

    New products have 2D state (T, 126+hands); an old artifact's 4D (T,2,21,3) must be
    flattened through the reshape branch and exported as usual — an upgrade must not refuse
    to publish existing data.
    """
    from pathlib import Path

    from roboray.runtime.episode import EpisodeWorkspace, Stage
    from roboray.stages.artifacts import save_artifact
    from roboray.stages.publish import run_publish

    t = 8
    fakes = Path(__file__).resolve().parents[1] / "fakes" / "modelenv_scripts"
    ws = EpisodeWorkspace(tmp_path / "ws", "ep_0001")
    ws.mark_done(Stage.ACTION, 1.0)
    save_artifact(
        ws.artifact("fused.npz"),
        joints_world=np.zeros((t, 2, 21, 3), np.float32),
        reprojection_error=np.zeros((t, 2), np.float32),
        hand_ids=np.zeros((t, 2), np.int32),
        joint_source=np.array("mano"),
        depth_source=np.array(["monocular"] * t, dtype=object),
        identity_switches=np.zeros((0, 2), np.int32),
        gapfill_filled=np.int32(0),
        gapfill_rejected=np.int32(0),
        n_frames=np.int32(t),
        fps=np.int32(30),
    )
    save_artifact(
        ws.artifact("action.npz"),
        labels=np.array(["reach"], dtype=object),
        segment_start=np.array([0], np.int32),
        segment_end=np.array([t], np.int32),
        state=np.zeros((t, 2, 21, 3), np.float32),   # old artifact: 4D state
        action=np.zeros((t, 2, 21, 3), np.float32),  # action same shape
        n_frames=np.int32(t),
        fps=np.int32(30),
    )
    info = run_publish(
        workspace_root=str(tmp_path / "ws"), output_root=str(tmp_path / "out"),
        episode_id="ep_0001", make_visualization=False,
        export_script=str(fakes / "fake_lerobot_export.py"),
    )
    assert info["outputs"]["dataset"]
    ds = tmp_path / "out" / "lerobot_v3" / "ep_0001"
    assert (ds / "meta" / "info.json").exists()
    shape = json.loads((ds / "meta" / "info.json").read_text())[
        "features"]["observation.state"]["shape"]
    assert shape == [126]  # 4D (t,2,21,3) flattened -> 126: the old artifact has no gripper column
