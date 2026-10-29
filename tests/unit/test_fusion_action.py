import json

import numpy as np
import pytest

from roboray.runtime.episode import EpisodeWorkspace
from roboray.stages.action import (
    run_action,
    segment_motions,
    select_keyframe_indices,
)
from roboray.stages.fusion import associate_wearer, fill_short_gaps, run_fusion


def _fake_present(n=20, gap_start=5, gap_len=2):
    present = np.ones((n, 2), bool)
    present[gap_start : gap_start + gap_len, 0] = False
    return present


def test_fill_short_gaps_fills_when_all_conditions_met():
    present = _fake_present(n=20, gap_start=5, gap_len=3)
    filled, audit = fill_short_gaps(present, max_gap=6, min_supporting_joints=8)
    assert filled[5:8, 0].all()
    assert audit["filled"] == 3
    assert audit["rejected"] == 0


def test_fill_short_gaps_refuses_when_gap_too_long():
    """Prefer keeping the gap over hallucinating: oversized gaps are never filled."""
    present = _fake_present(n=30, gap_start=5, gap_len=10)
    filled, audit = fill_short_gaps(present, max_gap=6, min_supporting_joints=8)
    assert not filled[5:15, 0].any()
    assert audit["filled"] == 0
    assert audit["rejected"] >= 1


def test_fill_short_gaps_refuses_without_two_sided_observation():
    present = np.ones((10, 2), bool)
    present[0:3, 0] = False  # gap at the start: only one-side observation
    filled, audit = fill_short_gaps(present, max_gap=6, min_supporting_joints=8)
    assert not filled[0:3, 0].any()
    assert audit["filled"] == 0


def test_fill_short_gaps_refuses_below_min_supporting_joints():
    present = _fake_present()
    filled, audit = fill_short_gaps(
        present, max_gap=6, min_supporting_joints=8, supporting_joints=np.array([[8, 8], [7, 8]] * 10)
    )
    assert audit["filled"] == 0
    assert audit["rejected"] >= 1


def test_associate_wearer_keeps_identity_continuous():
    joints_2d = np.zeros((6, 2, 21, 3), np.float32)
    joints_2d[:, 0, :, 0] = 0.3   # left hand always on the left
    joints_2d[:, 1, :, 0] = 0.7   # right hand always on the right
    present = np.ones((6, 2), bool)
    hand_ids, switches = associate_wearer(joints_2d, present)
    assert hand_ids.shape == (6, 2)
    assert list(hand_ids[:, 0]) == [0] * 6
    assert list(hand_ids[:, 1]) == [1] * 6
    assert len(switches) == 0


def test_associate_wearer_detects_identity_switch():
    joints_2d = np.zeros((4, 2, 21, 3), np.float32)
    joints_2d[:, 0, :, 0] = 0.3
    joints_2d[:, 1, :, 0] = 0.7
    joints_2d[2:, :, :, 0] = joints_2d[2:, ::-1, :, 0]  # left/right swap after frame 2
    present = np.ones((4, 2), bool)
    _, switches = associate_wearer(joints_2d, present)
    assert len(switches) >= 1


def test_run_fusion_writes_artifact_with_audit(tmp_path):
    ws = EpisodeWorkspace(tmp_path, "ep_0001")
    n = 12
    ws.write_artifact(
        "scene.npz",
        {
            "camera_poses": np.tile(np.eye(4, dtype=np.float32), (n, 1, 1)),
            "intrinsics": np.tile(np.eye(3, dtype=np.float32), (n, 1, 1)),
            "depth": np.full((n, 8, 8), 0.5, np.float32),
            "depth_source": np.array("monocular"),
            "n_frames": np.int32(n),
            "fps": np.int32(30),
        },
    )
    present = np.ones((n, 2), bool)
    present[4:6, 0] = False
    ws.write_artifact(
        "hand.npz",
        {
            "hand_present": present,
            "joints_2d": np.zeros((n, 2, 21, 3), np.float32),
            "mano_pose": np.zeros((n, 2, 48), np.float32),
            "mano_shape": np.zeros((n, 2, 10), np.float32),
            "mano_trans": np.zeros((n, 2, 3), np.float32),
            "confidence": np.where(present, 0.9, 0.0).astype(np.float32),
            "n_frames": np.int32(n),
            "fps": np.int32(30),
        },
    )
    info = run_fusion(
        workspace_root=str(tmp_path),
        episode_id="ep_0001",
        wearer_association=True,
        unobserved_gap_s=0.2,
        min_supporting_joints=8,
    )
    assert info["gapfill_filled"] == 2
    z = np.load(ws.artifact("fused.npz"), allow_pickle=False)
    assert z["joints_world"].shape == (n, 2, 21, 3)
    assert z["reprojection_error"].shape == (n, 2)


def test_run_fusion_rejects_corrupt_input_artifact(tmp_path):
    """Corrupt artifacts must be detected, not silently consumed."""
    ws = EpisodeWorkspace(tmp_path, "ep_0001")
    ws.artifact("scene.npz").write_bytes(b"corrupt")
    ws.write_artifact("hand.npz", {"hand_present": np.ones((4, 2), bool)})
    with pytest.raises(ValueError) as exc:
        run_fusion(workspace_root=str(tmp_path), episode_id="ep_0001",
                   wearer_association=True, unobserved_gap_s=0.2, min_supporting_joints=8)
    assert "scene" in str(exc.value)


def test_run_fusion_rejects_unstamped_artifact(tmp_path):
    """Artifacts written with a bare np.savez (no schema version stamp) must also be
    rejected.

    Validation must be structural (version stamp) rather than content readability —
    otherwise old toolchain products would be silently consumed.
    """
    ws = EpisodeWorkspace(tmp_path, "ep_0001")
    # Write a content-valid scene.npz with a bare np.savez but no version stamp
    np.savez(
        ws.artifact("scene.npz"),
        camera_poses=np.tile(np.eye(4, dtype=np.float32), (2, 1, 1)),
        intrinsics=np.tile(np.eye(3, dtype=np.float32), (2, 1, 1)),
        depth=np.full((2, 8, 8), 0.5, np.float32),
        depth_source=np.array("monocular"),
        n_frames=np.int32(2),
        fps=np.int32(30),
    )
    ws.write_artifact("hand.npz", {"hand_present": np.ones((2, 2), bool)})
    with pytest.raises(ValueError) as exc:
        run_fusion(workspace_root=str(tmp_path), episode_id="ep_0001",
                   wearer_association=True, unobserved_gap_s=0.2, min_supporting_joints=8)
    assert "schema" in str(exc.value)


def _assoc_cand(cx, cy, size, side, score=0.9, seed=0):
    """Build a candidates-protocol detection (same approach as tests/unit/test_association.py)."""
    rng = np.random.default_rng(seed)
    j2 = np.zeros((21, 3), np.float32)
    j2[:, 0] = cx + rng.normal(0, 0.01, 21) * size
    j2[:, 1] = cy + rng.normal(0, 0.01, 21) * size
    j2[:, 2] = score
    return {"bbox": np.array([cx - size / 2, cy - size / 2,
                              cx + size / 2, cy + size / 2], np.float32),
            "side": int(side), "score": float(score), "joints_2d": j2,
            "joints_3d_cam": np.zeros((21, 3), np.float32)}


def _write_assoc_fixture(ws, n=10, with_candidates=True):
    """Minimal scene.npz + hand.npz: wearer hands on a crossing trajectory + a bystander
    hand (candidates only).

    Wearer left 200->416, right 430->214 (x-order crosses around frame 5: x-rank would
    report a false switch; the association layer keeps the entity continuous); bystander is
    small and high, only in candidates. Main-array slots = wearer hands (isomorphic to
    run_hamer's "first-left/first-right" contract). depth (n,480,640)@0.5 + identity K/pose
    -> joints_2d path backprojects world = (u,v,1)·0.5 (bystander v=100 -> world y=50).
    """
    w, h = 640, 480
    ws.write_artifact(
        "scene.npz",
        {
            "camera_poses": np.tile(np.eye(4, dtype=np.float32), (n, 1, 1)),
            "intrinsics": np.tile(np.eye(3, dtype=np.float32), (n, 1, 1)),
            "depth": np.full((n, h, w), 0.5, np.float32),
            "depth_source": np.array("monocular"),
            "n_frames": np.int32(n),
            "fps": np.int32(30),
        },
    )
    frames = []
    for t in range(n):
        frames.append([
            _assoc_cand(200 + 24 * t, 400, 120, 0, seed=t),
            _assoc_cand(430 - 24 * t, 390, 120, 1, seed=t + 1),
        ])
    present = np.ones((n, 2), bool)
    joints_2d = np.zeros((n, 2, 21, 3), np.float32)
    for i, fr in enumerate(frames):
        for s in (0, 1):
            joints_2d[i, s] = fr[s]["joints_2d"]
    arrays = {
        "hand_present": present,
        "joints_2d": joints_2d,
        "mano_pose": np.zeros((n, 2, 48), np.float32),
        "mano_shape": np.zeros((n, 2, 10), np.float32),
        "mano_trans": np.zeros((n, 2, 3), np.float32),
        "confidence": np.where(present, 0.9, 0.0).astype(np.float32),
        "n_frames": np.int32(n),
        "fps": np.int32(30),
    }
    if with_candidates:
        cands = np.empty(n, dtype=object)
        for i, fr in enumerate(frames):
            cands[i] = fr + [_assoc_cand(320, 100, 50, 0, score=0.8, seed=100 + i)]
        arrays["candidates"] = cands
    ws.write_artifact("hand.npz", arrays)
    return present, joints_2d, frames


def test_fusion_with_candidates_uses_association(tmp_path):
    """hand.npz with candidates -> joints_2d/present come from the association layer, switch
    semantics upgraded.

    On the crossing-hand sequence x-rank would report ordering changes, the association
    layer keeps entities continuous: identity_switches must equal the association layer's
    entity switches (0 for this fixture), hand_ids semantics upgrade to "candidate index
    bound in that frame"; world joints reflect only the wearer candidates (bystander
    coordinates never appear).
    """
    from roboray.stages.association import associate_candidates

    ws = EpisodeWorkspace(tmp_path, "ep_0001")
    n = 10
    _present, _joints_2d, frames = _write_assoc_fixture(ws, n=n, with_candidates=True)
    info = run_fusion(
        workspace_root=str(tmp_path),
        episode_id="ep_0001",
        wearer_association=True,
        unobserved_gap_s=0.2,
        min_supporting_joints=8,
    )
    assoc = associate_candidates(frames, 640, 480)
    assert assoc["switches"] == []  # baseline truth: entities continuous (fixture geometry guard)
    assert info["identity_switches"] == len(assoc["switches"]) == 0
    with np.load(ws.artifact("fused.npz"), allow_pickle=False) as z:
        assert z["hand_ids"].shape == (n, 2)
        assert np.array_equal(z["hand_ids"], assoc["labels"])
        assert np.array_equal(
            z["identity_switches"],
            np.asarray(assoc["switches"], np.int32).reshape(-1, 2),
        )
        world = z["joints_world"]
    # World joints match the wearer candidates the association layer selected
    # (world = (u,v,1)·0.5, frame by frame, slot by slot)
    sel = assoc["present"]
    assoc_xy = assoc["joints_2d"][..., :2].astype(np.float64).mean(axis=2)  # (n,2,2)
    world_xy = world[..., :2].astype(np.float64).mean(axis=2)               # (n,2,2)
    assert np.allclose(world_xy[sel], assoc_xy[sel] / 2, atol=1e-3)
    # The bystander (v=100 -> world y=50) must not appear in any frame/slot
    # (wearer y≈390 -> world y≈195)
    assert (np.abs(world_xy[..., 1] - 50.0) > 10.0).all()


def test_fusion_without_candidates_falls_back(tmp_path):
    """No candidates key -> x-rank legacy path, results identical to before the change.

    Contrast test: same fixture without the candidates key; hand_ids/identity_switches must
    equal the direct associate_wearer (x-rank) output — on a crossing sequence x-rank counts
    ordering changes (a different semantic from the association layer's entity continuity,
    and here that old semantic must still hold).
    """
    ws = EpisodeWorkspace(tmp_path, "ep_0001")
    n = 10
    present, joints_2d, _frames = _write_assoc_fixture(ws, n=n, with_candidates=False)
    info = run_fusion(
        workspace_root=str(tmp_path),
        episode_id="ep_0001",
        wearer_association=True,
        unobserved_gap_s=0.2,
        min_supporting_joints=8,
    )
    xr_ids, xr_switches = associate_wearer(joints_2d, present)
    assert len(xr_switches) >= 1  # fixture x-order crosses around frame 5, x-rank counts truthfully
    assert info["identity_switches"] == len(xr_switches)
    with np.load(ws.artifact("fused.npz"), allow_pickle=False) as z:
        assert np.array_equal(z["hand_ids"], xr_ids)
        assert np.array_equal(
            z["identity_switches"],
            np.asarray(xr_switches, np.int32).reshape(-1, 2),
        )


def test_fusion_candidates_reorder_main_array_bystander_first(tmp_path):
    """Review counter-example: main-array slot 0 = bystander (first in detection order),
    the wearer's left only in candidates.

    Under run_hamer's "first-left/first-right" contract, an early-detected bystander
    pollutes the main-array slots — if fusion's fill checks and world projection consumed
    the main array directly, the geometry would come from the bystander. Association must
    run first and reorder the main arrays by its labels: joints_world[i,0] must be the
    wearer-left candidate's backprojection (y≈200), not the bystander's (y≈50); the
    identity plane (hand_ids/identity_switches) comes from the association layer.
    """
    from roboray.stages.association import associate_candidates

    ws = EpisodeWorkspace(tmp_path, "ep_0001")
    n, w, h = 6, 640, 480
    ws.write_artifact(
        "scene.npz",
        {
            "camera_poses": np.tile(np.eye(4, dtype=np.float32), (n, 1, 1)),
            "intrinsics": np.tile(np.eye(3, dtype=np.float32), (n, 1, 1)),
            "depth": np.full((n, h, w), 0.5, np.float32),
            "depth_source": np.array("monocular"),
            "n_frames": np.int32(n),
            "fps": np.int32(30),
        },
    )
    frames = []
    for t in range(n):
        # Detection order: bystander first -> main-array "first-left" = bystander; wearer
        # left exists only in candidates
        frames.append([
            _assoc_cand(320, 100, 50, 0, score=0.8, seed=100 + t),
            _assoc_cand(200, 400, 120, 0, seed=t),
            _assoc_cand(430, 390, 120, 1, seed=t + 1),
        ])
    present = np.ones((n, 2), bool)
    joints_2d = np.zeros((n, 2, 21, 3), np.float32)
    for i, fr in enumerate(frames):
        joints_2d[i, 0] = fr[0]["joints_2d"]  # slot 0 = bystander (first-left)
        joints_2d[i, 1] = fr[2]["joints_2d"]  # slot 1 = wearer right (first-right)
    cands = np.empty(n, dtype=object)
    for i, fr in enumerate(frames):
        cands[i] = fr
    ws.write_artifact(
        "hand.npz",
        {
            "hand_present": present,
            "joints_2d": joints_2d,
            "mano_pose": np.zeros((n, 2, 48), np.float32),
            "mano_shape": np.zeros((n, 2, 10), np.float32),
            "mano_trans": np.zeros((n, 2, 3), np.float32),
            "confidence": np.array([[0.8, 0.9]] * n, np.float32),
            "candidates": cands,
            "n_frames": np.int32(n),
            "fps": np.int32(30),
        },
    )
    info = run_fusion(
        workspace_root=str(tmp_path),
        episode_id="ep_0001",
        wearer_association=True,
        unobserved_gap_s=0.2,
        min_supporting_joints=8,
    )
    assoc = associate_candidates(frames, w, h)
    # The association layer selects the wearer: the bystander (index 0) is bound to no slot
    assert np.array_equal(assoc["labels"], np.tile([1, 2], (n, 1)))
    assert assoc["switches"] == []
    assert info["identity_switches"] == 0
    with np.load(ws.artifact("fused.npz"), allow_pickle=False) as z:
        assert np.array_equal(z["hand_ids"], assoc["labels"])
        world_xy = z["joints_world"][..., :2].astype(np.float64).mean(axis=2)
    # Slot 0 = wearer-left candidate backprojection (world=(u,v,1)·0.5 -> (100,200)),
    # not the bystander (160,50)
    assert np.allclose(world_xy[:, 0, 0], 100.0, atol=10.0)
    assert np.allclose(world_xy[:, 0, 1], 200.0, atol=10.0)
    # Slot 1 = wearer-right candidate backprojection (215,195)
    assert np.allclose(world_xy[:, 1, 0], 215.0, atol=10.0)
    assert np.allclose(world_xy[:, 1, 1], 195.0, atol=10.0)


def test_segment_motions_finds_moving_interval():
    joints = np.zeros((20, 2, 21, 3), np.float32)
    joints[8:14, 0, 0, 0] = np.linspace(0.0, 0.5, 6)  # frames 8-13 are moving
    segments = segment_motions(joints, fps=30, speed_thresh=0.01, min_len=2)
    assert (8, 14, "motion") in segments  # triple: kind ∈ motion|contact


def test_run_action_uses_registered_semantic_model(tmp_path):
    from roboray.models import registry
    from tests.fakes.models import install_fake_models

    registry.reset_registry_for_tests()
    install_fake_models()
    try:
        ws = EpisodeWorkspace(tmp_path, "ep_0001")
        ws.write_artifact(
            "fused.npz",
            {
                "joints_world": np.zeros((24, 2, 21, 3), np.float32),
                "reprojection_error": np.zeros((24, 2), np.float32),
                "depth_source": np.array(["monocular"] * 24, dtype=object),
                "identity_switches": np.zeros((0, 2), np.int32),
                "gapfill_filled": np.int32(0),
                "gapfill_rejected": np.int32(0),
                "n_frames": np.int32(24),
                "fps": np.int32(30),
            },
        )
        info = run_action(
            workspace_root=str(tmp_path), episode_id="ep_0001",
            semantic_model_name="fake_vlm", keyframe_stride=30,
        )
        assert info["n_segments"] >= 1
        assert info["labels"]
        assert ws.artifact("action.npz").exists()
    finally:
        registry.reset_registry_for_tests()


class _RecordingSemantic:
    """Records the images handed to the action stage, to verify keyframes actually reach
    the VLM request."""

    def __init__(self):
        self.name = "recording_vlm"
        self.images = None
        self.n_segments = None

    def label(self, prompt, images, out_path, n_segments=0):
        self.images = list(images)
        self.n_segments = n_segments
        np.savez(out_path, labels=np.array(["reach"], dtype=object),
                 segment_start=np.zeros(1, np.int32), segment_end=np.zeros(1, np.int32))


def _write_fused(ws, n_frames=24, fps=30):
    ws.write_artifact(
        "fused.npz",
        {
            "joints_world": np.zeros((n_frames, 2, 21, 3), np.float32),
            "reprojection_error": np.zeros((n_frames, 2), np.float32),
            "depth_source": np.array(["monocular"] * n_frames, dtype=object),
            "identity_switches": np.zeros((0, 2), np.int32),
            "gapfill_filled": np.int32(0),
            "gapfill_rejected": np.int32(0),
            "n_frames": np.int32(n_frames),
            "fps": np.int32(fps),
        },
    )


def test_select_keyframe_indices_keeps_first_last_and_caps():
    assert select_keyframe_indices(20, 5) == [0, 5, 10, 15, 19]
    assert select_keyframe_indices(0, 5) == []
    assert select_keyframe_indices(1, 5) == [0]
    capped = select_keyframe_indices(500, 1, max_frames=16)
    assert len(capped) == 16
    assert capped[0] == 0 and capped[-1] == 499
    assert capped == sorted(set(capped))


def test_run_action_passes_encoded_keyframes_to_semantic_model(tmp_path):
    """spec §9: action understanding must see keyframes + a trajectory summary; stride
    determines sampling density."""
    from roboray.models import registry

    registry.reset_registry_for_tests()
    rec = _RecordingSemantic()
    registry.register_semantic_model("recording_vlm", lambda: rec)
    try:
        ws = EpisodeWorkspace(tmp_path, "ep_0001")
        _write_fused(ws)
        frames = np.zeros((20, 8, 8, 3), np.uint8)
        frames[:, :, :, 0] = np.arange(20, dtype=np.uint8)[:, None, None]  # distinguishable
        np.save(ws.prepared_dir / "frames_left.npy", frames)
        info = run_action(
            workspace_root=str(tmp_path), episode_id="ep_0001",
            semantic_model_name="recording_vlm", keyframe_stride=5,
        )
        assert rec.images, "keyframes must be passed to the semantic model (else the VLM sees no frames)"
        assert len(rec.images) == 5  # 0/5/10/15/19
        assert all(u.startswith("data:image/jpeg;base64,") for u in rec.images)
        # Decode back to confirm real frames, not placeholder strings
        import base64 as _b64

        import cv2

        head = rec.images[0].split(",", 1)[1]
        img = cv2.imdecode(np.frombuffer(_b64.b64decode(head), np.uint8), cv2.IMREAD_COLOR)
        assert img.shape == (8, 8, 3) and img[0, 0, 2] == 0
        assert rec.n_segments == len(info["labels"]) == info["n_segments"]
    finally:
        registry.reset_registry_for_tests()


def test_run_action_tolerates_missing_prepared_frames(tmp_path):
    """A missing frames_left.npy only drops the visual input; it must not crash the episode."""
    from roboray.models import registry

    registry.reset_registry_for_tests()
    rec = _RecordingSemantic()
    registry.register_semantic_model("recording_vlm", lambda: rec)
    try:
        ws = EpisodeWorkspace(tmp_path, "ep_0001")
        _write_fused(ws)  # do not write prepared/frames_left.npy
        info = run_action(
            workspace_root=str(tmp_path), episode_id="ep_0001",
            semantic_model_name="recording_vlm", keyframe_stride=5,
        )
        assert rec.images == []
        assert info["labels"]
    finally:
        registry.reset_registry_for_tests()


def test_segment_motions_contact_kind():
    """Low-speed dwell segments are marked contact and coexist with motion segments in the
    triple return value."""
    T, fps = 30, 30
    jw = np.zeros((T, 2, 21, 3))
    jw[:10, :, :, 0] = np.linspace(0, 0.6, 10)[:, None, None]   # motion (0.067/frame > 0.05)
    jw[10:20, :, :, 0] = 0.6                                     # dwell 10 frames (contact)
    jw[20:, :, :, 0] = np.linspace(0.6, 1.2, 10)[:, None, None]  # motion again
    segs = segment_motions(jw, fps)
    kinds = {k for _s, _e, k in segs}
    assert kinds == {"motion", "contact"}
    contact = [s for s in segs if s[2] == "contact"]
    assert contact and contact[0][0] >= 9 and contact[0][1] <= 20


def test_segment_motions_legacy_shape_gone_but_labels_ok():
    """Triple return: label count matches segment count (run_action call-site compatibility
    pinned by the full regression)."""
    jw = np.zeros((20, 2, 21, 3))
    jw[:10, :, :, 0] = np.linspace(0, 0.5, 10)[:, None, None]
    segs = segment_motions(jw, 30)
    assert all(len(s) == 3 for s in segs) and segs


def test_gripper_channel_range_and_nan(tmp_path):
    """run_action product: gripper ∈ [0,1], missing-hand frames NaN, state includes the
    gripper column.

    gripper = thumb(4)-index(8) tip distance / batch max — distance-ratio semantics, not
    real gripper state.
    """
    ws = EpisodeWorkspace(tmp_path, "ep_0001")
    t = 24
    jw = np.zeros((t, 2, 21, 3), np.float32)
    # Tip distance varies per frame (0.1->0.2): finite values feed the batch-max normalization
    jw[:, :, 8, 0] = np.linspace(0.1, 0.2, t)[:, None]
    jw[6:9, 1] = np.nan  # missing-hand frames: hand 1 joints absent -> tip_dist NaN -> gripper NaN
    ws.write_artifact(
        "fused.npz",
        {
            "joints_world": jw,
            "reprojection_error": np.zeros((t, 2), np.float32),
            "depth_source": np.array(["monocular"] * t, dtype=object),
            "identity_switches": np.zeros((0, 2), np.int32),
            "gapfill_filled": np.int32(0),
            "gapfill_rejected": np.int32(0),
            "n_frames": np.int32(t),
            "fps": np.int32(30),
        },
    )
    run_action(
        workspace_root=str(tmp_path), episode_id="ep_0001",
        semantic_model_name="none",
    )
    with np.load(ws.artifact("action.npz"), allow_pickle=True) as z:
        gripper = z["gripper"]
        state = z["state"]
        action = z["action"]
        # Triple-segment persistence: segment_kind aligns per segment with segment_start/end
        assert z["segment_kind"].shape == z["segment_start"].shape
        assert {str(k) for k in z["segment_kind"]} <= {"motion", "contact"}
    assert gripper.shape == (t, 2)
    # state is 2D (T, 126+2): flattened joints + one gripper column per hand
    assert state.ndim == 2 and state.shape == (t, 126 + 2)
    assert action.shape == state.shape
    # action = state shifted one frame (last frame copied)
    assert np.allclose(action[:-1, :126], state[1:, :126], equal_nan=True)
    # Missing-hand frames -> gripper NaN, and state's gripper column matches the gripper key
    # frame by frame
    assert np.isnan(gripper[6:9, 1]).all()
    assert np.array_equal(np.isnan(state[:, 126:]), np.isnan(gripper))
    ok = ~np.isnan(gripper)
    assert np.allclose(state[:, 126:][ok], gripper[ok])
    # All finite values fall in [0,1] (distance ratio clipped)
    finite = gripper[np.isfinite(gripper)]
    assert finite.size and (finite >= 0.0).all() and (finite <= 1.0).all()
