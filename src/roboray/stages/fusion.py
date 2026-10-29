"""A1 metric fusion + A2 wearer association + A3 short-gap filling (CPU intensive)."""

import json
from pathlib import Path

import numpy as np

from roboray.runtime.episode import ARTIFACT_SCHEMA_VERSION

from .artifacts import check_artifact_file, save_artifact
from .association import associate_candidates

__all__ = [
    "fill_short_gaps",
    "associate_wearer",
    "backproject_to_world",
    "rigid_align",
    "run_fusion",
    "UNTRUSTED_REPROJ_ERROR",
]

# The four post-fill checks' thresholds (spec §7, A3)
_MAX_WRIST_STEP_M = 0.25      # max per-frame wrist displacement (meters)
# Upper bound for the 4th check (trajectory deviation = velocity continuity): the gap
# between the per-frame displacement implied by the bridge and the mean of the observed
# per-frame displacements at the gap ends (m/frame). In ego manipulation scenes wrist speed
# is roughly 0.1~2 m/s (≈0.003~0.067 m/frame at 30fps); its upper part 0.05 m/frame
# (≈1.5 m/s) is used as a tunable safety bound — currently an empirical value, to be
# re-validated with real data.
_MAX_WRIST_VEL_MISMATCH_M_PER_FRAME = 0.05

# Reprojection-error sentinel for untrusted hand frames (frame, hand): a numeric marker
# that QC/downstream filters on.
UNTRUSTED_REPROJ_ERROR = 1e6


def fill_short_gaps(
    present: np.ndarray,
    max_gap: int,
    min_supporting_joints: int,
    supporting_joints: np.ndarray | None = None,
) -> tuple:
    """Short-gap filling. All three conditions must hold to fill, otherwise record rejected.

    Conditions: (a) both ends observed (b) same hand on both sides (caller passes per-hand
    index) (c) gap length <= max_gap. Returns (filled_mask, audit).
    """
    present = np.asarray(present, bool)
    t, n_hands = present.shape
    filled = np.zeros_like(present)
    audit = {"filled": 0, "rejected": 0, "reasons": {}}

    def _reject(reason: str) -> None:
        audit["rejected"] += 1
        audit["reasons"][reason] = audit["reasons"].get(reason, 0) + 1

    if max_gap <= 0:
        return filled, audit

    for hand in range(n_hands):
        col = present[:, hand]
        i = 0
        while i < t:
            if col[i]:
                i += 1
                continue
            start = i
            while i < t and not col[i]:
                i += 1
            end = i  # gap is [start, end)
            gap_len = end - start
            if gap_len > max_gap:
                _reject("gap_too_long")
                continue
            if start == 0 or end >= t:
                _reject("missing_one_side_observation")
                continue
            if supporting_joints is not None:
                support = min(
                    int(supporting_joints[start - 1, hand]),
                    int(supporting_joints[end, hand]),
                )
                if support < min_supporting_joints:
                    _reject("insufficient_supporting_joints")
                    continue
            filled[start:end, hand] = True
            audit["filled"] += gap_len
    return filled, audit


def associate_wearer(joints_2d: np.ndarray, present: np.ndarray) -> tuple:
    """A2 wearer association: bind detections to the wearer's left/right hands by
    frame-position prior + temporal continuity.

    Returns (hand_ids (T,2) int32, switches: list[(frame, hand)]).
    """
    t, n_hands = present.shape
    hand_ids = np.full((t, n_hands), -1, np.int32)
    switches = []
    prev = None
    for i in range(t):
        xs = []
        for h in range(n_hands):
            if present[i, h]:
                xs.append(float(np.mean(joints_2d[i, h, :, 0])))
            else:
                xs.append(np.nan)
        order = np.argsort([x if not np.isnan(x) else np.inf for x in xs])
        cur = {}
        for rank, h in enumerate(order):
            if not np.isnan(xs[h]):
                cur[h] = int(rank)
        if prev and cur != prev:
            for h, v in cur.items():
                if h in prev and prev[h] != v:
                    switches.append((i, int(h)))
        hand_ids[i] = [cur.get(h, -1) for h in range(n_hands)]
        if cur:
            prev = cur
    return hand_ids, switches


def _sample_depth(depth: np.ndarray, u: float, v: float) -> float:
    """Bilinear depth sampling; out-of-range or non-positive returns NaN."""
    h, w = depth.shape
    if not (0 <= u < w - 1 and 0 <= v < h - 1):
        return float("nan")
    x0, y0 = int(u), int(v)
    fx, fy = u - x0, v - y0
    vals = [depth[y0, x0], depth[y0, x0 + 1], depth[y0 + 1, x0], depth[y0 + 1, x0 + 1]]
    if any(val <= 0 or not np.isfinite(val) for val in vals):
        return float("nan")
    return float(
        (1 - fx) * (1 - fy) * vals[0] + fx * (1 - fy) * vals[1]
        + (1 - fx) * fy * vals[2] + fx * fy * vals[3]
    )


def _project_to_px(world_pts: np.ndarray, K: np.ndarray, w2c: np.ndarray) -> np.ndarray:
    """World points -> camera frame -> pixels (N,2). Denominator clipped to 1e-9 to avoid
    division by zero (same convention as reprojection)."""
    cam = (
        np.concatenate([np.asarray(world_pts, np.float64), np.ones((len(world_pts), 1))], axis=1)
        @ np.asarray(w2c, np.float64).T
    )[:, :3]
    uvw = (np.asarray(K, np.float64) @ cam.T).T
    return uvw[:, :2] / np.clip(uvw[:, 2:3], 1e-9, None)


def _reproject_px(world_pts: np.ndarray, src_uv: np.ndarray, K: np.ndarray, w2c: np.ndarray) -> float:
    """World joints -> w2c -> pixels, mean per-joint Euclidean distance to the detected 2D (pixels)."""
    uv = _project_to_px(world_pts, K, w2c)
    return float(np.linalg.norm(uv - np.asarray(src_uv, np.float64), axis=1).mean())


def _depth_supported_count(
    joints_world_hand: np.ndarray, depth_frame: np.ndarray, K: np.ndarray, w2c: np.ndarray
) -> int:
    """After projecting the filled joints back to that frame's pixels, the number of joints
    with valid (finite and > 0) depth samples.

    A filled frame has no 2D detections of its own, so the (interpolated) world joints are
    **reprojected** to that frame's pixels and sampled against its depth map as "depth
    support" evidence — depth failing in the filled region means the fill position has no
    image evidence behind it.
    """
    pts = np.asarray(joints_world_hand, np.float64)
    finite = np.isfinite(pts).all(axis=1)
    if not finite.any():
        return 0
    uv = _project_to_px(pts[finite], K, w2c)
    count = 0
    for u, v in uv:
        if not (np.isfinite(u) and np.isfinite(v)):
            continue
        z = _sample_depth(depth_frame, float(u), float(v))
        if np.isfinite(z) and z > 0:
            count += 1
    return count


def _all_in_view(joints_2d: np.ndarray, width: int, height: int) -> bool:
    """Whether all 2D joints fall inside the **sampleable region**.

    The bounds match ``_sample_depth`` (``0 <= u < width-1``, ``0 <= v < height-1``) — the
    sampleable region is the authoritative definition, so "in view" implies "depth
    sampleable" and no seam exists where a joint passes the view check yet is guaranteed to
    fail at the depth sample. Non-finite detections do not count as out-of-bounds (that is
    a different failure mode, depth/backprojection), but at least one finite detection is
    required, otherwise there is no usable 2D evidence.
    """
    uv = np.asarray(joints_2d, np.float64)[..., :2]
    finite = np.isfinite(uv).all(axis=-1)
    if not finite.any():
        return False
    inside = (
        (uv[..., 0] >= 0) & (uv[..., 0] < width - 1)
        & (uv[..., 1] >= 0) & (uv[..., 1] < height - 1)
    )
    return bool((inside | ~finite).all())


def _trajectory_deviation(
    wrist_traj: np.ndarray, prev_i: int, next_i: int, observed: np.ndarray
) -> tuple:
    """Deviation of the fill bridge's implied velocity from the mean of the observed
    velocities at the gap ends (m/frame).

    Returns ``(deviation | None, skip_reason | None)``; when deviation is None the second
    element says why it could not be evaluated:

    - ``"sequence_boundary"``: neighbor index out of range (the gap touches the sequence boundary);
    - ``"neighbour_unobserved"``: the neighbor frame has **no observation** (that hand was
      missed in that frame) — its ``joints_world`` must never be used as a measured
      displacement (unobserved joints may be all-zero/NaN garbage);
    - ``"wrist_untrusted"``: the wrist of an involved frame is not finite (observed, but
      that hand is untrusted).

    ``wrist_traj`` (T,3) is the world trajectory of the wrist (joint 0), ``observed`` (T,)
    is that hand's ``hand_present`` column, and ``prev_i``/``next_i`` are the observed
    endpoints flanking the gap (gap = ``[prev_i+1, next_i)``):

    - ``v_before = P[prev_i] − P[prev_i−1]``: single-frame **measured** displacement into the gap;
    - ``v_after = P[next_i+1] − P[next_i]``: single-frame **measured** displacement out of the gap;
    - ``v_bridge = (P[next_i] − P[prev_i]) / (next_i − prev_i)``: per-frame displacement of the bridge.

    When both are available, returns ``‖v_bridge − (v_before + v_after)/2‖``.
    """
    t = len(wrist_traj)
    if prev_i - 1 < 0 or next_i + 1 >= t:
        return None, "sequence_boundary"
    if not (bool(observed[prev_i - 1]) and bool(observed[next_i + 1])):
        return None, "neighbour_unobserved"
    span = next_i - prev_i
    if span <= 0:
        return None, "sequence_boundary"
    w = np.asarray(wrist_traj, np.float64)
    p_start, p_before = w[prev_i], w[prev_i - 1]
    p_end, p_after = w[next_i], w[next_i + 1]
    if not all(np.isfinite(p).all() for p in (p_start, p_before, p_end, p_after)):
        return None, "wrist_untrusted"
    v_before = p_start - p_before
    v_after = p_after - p_end
    v_bridge = (p_end - p_start) / span
    return float(np.linalg.norm(v_bridge - 0.5 * (v_before + v_after))), None


def _fill_reject_reason(
    candidate: np.ndarray,
    prev_j: np.ndarray,
    *,
    min_supporting_joints: int,
    depth_frame: np.ndarray,
    intrinsics_frame: np.ndarray,
    camera_pose_frame: np.ndarray,
    joints_2d_used: np.ndarray,
    image_hw: tuple,
    wrist_traj: np.ndarray,
    prev_i: int,
    next_i: int,
    observed: np.ndarray,
) -> tuple:
    """The four post-fill checks (spec §7, A3). Returns ``(first_failed_reason | None, trajectory_skip_reason | None)``.

    The second element is meaningful only when the fill is **not** rejected (None when
    rejected; the caller does not read it).

    ``candidate`` (21,3, world, NaN allowed) is the candidate fill pose, ``prev_j`` the
    trusted left endpoint, ``joints_2d_used`` (2,21,3) the two endpoint 2D detections the
    interpolation is based on, ``wrist_traj`` (T,3) the wrist world trajectory, and
    ``observed`` (T,) that hand's ``hand_present`` column. The check order (depth -> view ->
    wrist displacement -> trajectory deviation) determines which reason lands in
    ``gapfill_reasons``; NaN endpoints are caught by the first check (supporting joints = 0).

    The 4th **trajectory deviation** check measures velocity continuity: the gap between the
    bridge's implied per-frame displacement and the mean measured displacement at the gap
    ends (``_trajectory_deviation``). The earlier "candidate pose vs linear interpolation of
    the endpoints" formulation was degenerate — the fill itself is that linear interpolation,
    so the deviation was identically 0 and could never trigger. The genuinely untrustworthy
    case for linear bridging is "non-uniform surrounding motion" (slow in, fast out, or vice
    versa), exactly what this check catches. When it cannot be evaluated the fill passes, but
    the caller counts by ``sequence_boundary`` / ``neighbour_unobserved`` /
    ``wrist_untrusted`` (see ``_trajectory_deviation``).
    """
    # 1) Keypoint depth: joints still supported by that frame's depth map at the fill
    #    position >= min_supporting_joints
    if _depth_supported_count(
        candidate, depth_frame, intrinsics_frame, camera_pose_frame
    ) < min_supporting_joints:
        return "insufficient_keypoint_depth", None
    # 2) View: the two endpoint 2D detections used for interpolation must fall inside the
    #    image bounds (the sampleable region)
    height, width = image_hw
    if not _all_in_view(joints_2d_used, width, height):
        return "out_of_view", None
    # 3) Wrist displacement: per-frame displacement upper bound (relative to the trusted left endpoint)
    if np.isfinite(candidate[0]).all() and np.isfinite(prev_j[0]).all():
        if float(np.linalg.norm(candidate[0] - prev_j[0])) > _MAX_WRIST_STEP_M:
            return "wrist_step_exceeded", None
    # 4) Trajectory deviation: bridge velocity vs mean **measured** velocity at the gap ends
    #    (unable to evaluate -> pass and count by category)
    dev, skip = _trajectory_deviation(wrist_traj, prev_i, next_i, observed)
    if skip is not None:
        return None, skip
    if dev > _MAX_WRIST_VEL_MISMATCH_M_PER_FRAME:
        return "trajectory_deviation_exceeded", None
    return None, None


def rigid_align(
    mano_pts: np.ndarray,
    target_pts: np.ndarray,
    valid: np.ndarray,
    min_points: int = 4,
) -> tuple:
    """A1 upgrade: Kabsch unscaled rigid alignment — rotation + translation, **no scaling**.

    ``target_pts`` are the depth-backprojected joint 3D (trusted only where ``valid``).
    Too few valid points, or a singular centered covariance (collinear/coincident) -> fall
    back to root translation (pre-change behavior), method="root_only"; otherwise
    method="kabsch". No scaling preserves MANO bone lengths.
    """
    pts = np.asarray(mano_pts, np.float64)
    tgt = np.asarray(target_pts, np.float64)
    m = np.asarray(valid, bool) & np.isfinite(pts).all(axis=1) & np.isfinite(tgt).all(axis=1)
    if int(m.sum()) < min_points or not np.isfinite(pts[0]).all():
        shifted = pts + (tgt[0] - pts[0])
        return shifted, "root_only"
    p, q = pts[m], tgt[m]
    pc, qc = p.mean(axis=0), q.mean(axis=0)
    H = (p - pc).T @ (q - qc)
    try:
        U, S, Vt = np.linalg.svd(H)
    except np.linalg.LinAlgError:
        return pts + (tgt[0] - pts[0]), "root_only"
    # Collinear/coincident points make the centered covariance rank-deficient:
    # ``np.linalg.svd`` does not raise on rank-deficient matrices, it just drives the
    # corresponding singular value down to ~1e-17 — degeneracy must be judged explicitly
    # by the condition number, otherwise Kabsch returns numerically garbage rotations
    # (mirror/global flip) instead of a legal fallback. 1e-6 sits ~7 orders of magnitude
    # below the full-rank case (min/max singular-value ratio ≈ 0.3+, measured), so it only
    # blocks genuine degeneracy.
    if S[0] <= 0 or S[-1] / S[0] < 1e-6:
        return pts + (tgt[0] - pts[0]), "root_only"
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])          # no scaling (|singular values|=1), only prevents reflection
    R = Vt.T @ D @ U.T
    return (R @ (pts - pc).T).T + qc, "kabsch"


def _mano_hand_to_world(
    mano_pts_cam: np.ndarray,
    joints_2d_hand: np.ndarray,
    depth_frame: np.ndarray,
    K: np.ndarray,
    camera_poses_i: np.ndarray,
    c2w: np.ndarray,
) -> tuple:
    """MANO camera-frame 3D joints -> world + that hand's reprojection error + alignment method.

    A1 upgrade: alignment goes from "single-point translation anchoring" to **Kabsch rigid
    alignment preferred, root translation fallback** (``rigid_align``). MANO output is
    already metric (meters) with bone lengths calibrated by MANO, so the alignment does
    **not scale**; depth-map evidence no longer anchors only the root node — every valid
    detected joint is backprojected by its own depth (``K⁻¹·(u,v,1)·z``) as a target point,
    and ``rigid_align`` solves for the rotation + translation that fits the hand as a
    **whole**, correcting the "overall orientation" divergence between MANO and the image
    evidence. With too few or degenerate (collinear/coincident) target points it falls back
    to root translation (pre-change behavior; the root depth is validated first, so the
    fallback anchor is always valid). ``method`` records the truth: ``"kabsch"`` /
    ``"root_only"``.

    The reprojection error measures the **consistency** between the MANO structure and the
    image evidence (+depth): rigid divergence is absorbed by Kabsch, the residual comes from
    non-rigid structural divergence and depth noise, and is nonzero when divergence exists.

    Missing root detection, non-finite MANO joints, or invalid root depth -> cannot be
    metrically anchored -> that hand, that frame is untrusted (returns (None, None, None)).
    """
    src = np.asarray(joints_2d_hand, np.float64)[:, :2]
    valid_src = np.isfinite(src).all(axis=1)
    pts_cam = np.asarray(mano_pts_cam, np.float64)
    if not valid_src[0] or not np.isfinite(pts_cam).all():
        return None, None, None
    z_root = _sample_depth(depth_frame, float(src[0, 0]), float(src[0, 1]))
    if not np.isfinite(z_root) or z_root <= 0:
        return None, None, None
    # Depth-backprojected target: valid detections backprojected by their own depth (K⁻¹·(u,v,1)·z)
    uv = src[valid_src]
    z_pts = np.array([_sample_depth(depth_frame, float(u), float(v))
                      for u, v in uv])
    ok = np.isfinite(z_pts) & (z_pts > 0)
    target = pts_cam.copy()
    tgt_valid = np.zeros(len(pts_cam), bool)
    if ok.any():
        idxs = np.where(valid_src)[0][ok]
        uv_h = np.stack([uv[ok, 0], uv[ok, 1], np.ones(int(ok.sum()))])  # (3,N)
        target[idxs] = (np.linalg.inv(K) @ uv_h * z_pts[ok]).T          # (N,3)
        tgt_valid[idxs] = True
    pts_cam, method = rigid_align(pts_cam, target, tgt_valid)
    hom = np.concatenate([pts_cam, np.ones((21, 1))], axis=1)
    world = (hom @ c2w.T)[:, :3]
    # Only finite detections participate in the error (missing-detection joints are excluded)
    return world, _reproject_px(world[valid_src], src[valid_src], K, camera_poses_i), method


def backproject_to_world(
    mano_trans: np.ndarray,
    mano_pose: np.ndarray,
    joints_2d: np.ndarray,
    depth: np.ndarray,
    intrinsics: np.ndarray,
    camera_poses: np.ndarray,
    joint_source: str = "joints_2d",
    joints_3d_cam: np.ndarray | None = None,
) -> tuple:
    """A1 metric fusion: joints 3D -> world; also computes the reprojection error (pixels)
    and alignment method.

    Two source paths (``joint_source``):

    - ``"mano"`` (requires ``joints_3d_cam``: the hand model's **camera-frame** MANO
      joints): per-joint depth backprojection yields target points, ``rigid_align`` applies
      Kabsch rigid alignment (rotation + translation, no scaling) then ``c2w`` -> world,
      correcting the "overall orientation" divergence between MANO and the image evidence;
      with too few/degenerate target points it falls back to root translation (pre-change
      behavior). The reprojection error is the consistency of the MANO structure against the
      image evidence (+depth), **informative** (larger non-rigid divergence, larger error).
      Per-hand alignment method (``"kabsch"`` / ``"root_only"``) is exposed via ``alignment``
      for npz/QC consumption.
    - ``"joints_2d"`` (fallback when 3D joints are absent): per-joint depth sampling +
      ``K⁻¹`` backprojection. On this path the reprojection is a pixel-level identity
      round-trip (``K·K⁻¹·(u,v,1)·z/z ≡ (u,v)``, and ``c2w·w2c ≡ I``) — **consistency check
      only, structurally always 0/sentinel** — it must not be treated as a quality metric
      (neither depth nor pose error shows up in this number). ``alignment`` is always
      ``"none"`` (no alignment step).

    Untrusted semantics (identical on both paths): that hand, that frame gets all-NaN +
    reprojection error ``UNTRUSTED_REPROJ_ERROR`` (1e6) + ``alignment = "none"``. On the
    ``mano`` path the criteria are root detection present, root depth valid, MANO joints all
    finite; on the ``joints_2d`` path, valid joints < 3.

    ``mano_trans`` / ``mano_pose`` stay in the signature (API stability, and the canonical
    source of the MANO root/pose used for root anchoring and future extensions; the ``mano``
    path carries equivalent information via ``joints_3d_cam``).

    Returns ``(joints_world (T,2,21,3) float64 — NaN where untrusted,
    reprojection_error (T,2) float64 — flagged 1e6 where untrusted,
    alignment (T,2) per-hand alignment-method string — "kabsch"/"root_only"/"none")``.
    """
    t, n_hands = joints_2d.shape[0], joints_2d.shape[1]
    joints_world = np.full((t, n_hands, 21, 3), np.nan, np.float64)
    reproj = np.full((t, n_hands), UNTRUSTED_REPROJ_ERROR, np.float64)
    alignment = np.full((t, n_hands), "none", dtype=object)
    use_mano = joint_source == "mano" and joints_3d_cam is not None
    for i in range(t):
        c2w = np.linalg.inv(np.asarray(camera_poses[i], np.float64))
        K = np.asarray(intrinsics[i], np.float64)
        Kinv = np.linalg.inv(K)
        for h_idx in range(n_hands):
            if use_mano:
                world, err, method = _mano_hand_to_world(
                    joints_3d_cam[i, h_idx], joints_2d[i, h_idx], depth[i], K, camera_poses[i], c2w
                )
                if world is None:
                    continue  # cannot be metrically anchored: that hand, that frame is untrusted
                joints_world[i, h_idx] = world
                reproj[i, h_idx] = err
                alignment[i, h_idx] = method
                continue
            pts_cam = np.full((21, 3), np.nan, np.float64)
            for j in range(21):
                u, v = float(joints_2d[i, h_idx, j, 0]), float(joints_2d[i, h_idx, j, 1])
                z = _sample_depth(depth[i], u, v)
                if not np.isfinite(z) or z <= 0:
                    continue
                pts_cam[j] = Kinv @ np.array([u, v, 1.0]) * z
            valid = np.isfinite(pts_cam).all(axis=1)
            if valid.sum() < 3:
                continue  # keypoint depth almost entirely invalid: mark the whole hand untrusted
            # Camera frame -> world frame
            hom = np.concatenate([pts_cam, np.ones((21, 1))], axis=1)
            joints_world[i, h_idx] = (hom @ c2w.T)[:, :3]
            # Reprojection: world -> camera -> pixels, compared to detected 2D (trusted joints only)
            src = joints_2d[i, h_idx][valid][:, :2]
            reproj[i, h_idx] = _reproject_px(joints_world[i, h_idx][valid], src, K, camera_poses[i])
    return joints_world, reproj, alignment


def run_fusion(
    workspace_root: str,
    episode_id: str,
    wearer_association: bool = True,
    unobserved_gap_s: float = 0.2,
    min_supporting_joints: int = 8,
) -> dict:
    ws = Path(workspace_root) / episode_id
    scene_path = ws / "artifacts" / "scene.npz"
    hand_path = ws / "artifacts" / "hand.npz"
    # Structural validation: missing/mismatched/corrupt schema version stamp -> readable ValueError.
    for p in (scene_path, hand_path):
        if not check_artifact_file(p):
            raise ValueError(f"fusion artifact missing or corrupt (schema version {ARTIFACT_SCHEMA_VERSION}): {p.name}")

    with np.load(scene_path, allow_pickle=True) as z:
        camera_poses = z["camera_poses"]
        intrinsics = z["intrinsics"]
        depth = z["depth"]
        depth_source = z["depth_source"]
        fps = int(z["fps"])
    # The A2 association layer's motion threshold is in pixels (0.15·width), so it needs the
    # real frame width/height: same source as the fill view check (the depth grid; scene.npz
    # carries no separate width/height keys); degenerate shapes without spatial dims fall
    # back to defaults.
    if depth.ndim >= 3:
        img_height, img_width = int(depth.shape[1]), int(depth.shape[2])
    else:
        img_width, img_height = 640, 480
    with np.load(hand_path, allow_pickle=True) as z:
        present = z["hand_present"]
        joints_2d = z["joints_2d"]
        mano_pose = z["mano_pose"]
        mano_trans = z["mano_trans"]
        # Camera-frame MANO joints from the hand model: if present use the informative mano
        # path, otherwise fall back to the round-trip path
        joints_3d_cam = z["joints_3d_cam"] if "joints_3d_cam" in z.files else None
        confidence = z["confidence"] if "confidence" in z.files else None
        # Optional A2 multi-person association input: object array (T,), each element
        # list[dict] (bbox/side/score/joints_2d/joints_3d_cam, full-image pixel coords).
        # Missing key -> fall back to x-rank.
        candidates = z["candidates"] if "candidates" in z.files else None

    joint_source = "mano" if joints_3d_cam is not None else "joints_2d"
    n_frames = int(present.shape[0])
    n_joints = int(joints_2d.shape[2])
    max_gap = max(1, int(round(unobserved_gap_s * fps)))

    # A2 loop — association first: reorder the main arrays by the association result
    # **before** fill_short_gaps and world projection. The main arrays are filled as
    # "first-left/first-right" (run_hamer contract), so an early-detected bystander pollutes
    # the slots; without reordering the fill checks and backprojection would use the
    # bystander's geometry as the wearer's. Reordering touches only
    # present/joints_2d/joints_3d_cam/confidence (scalar form); mano_trans/mano_pose are
    # numerically unconsumed (signature placeholders for the fallback path) and stay as is.
    # Unobserved slots (labels=-1) are zeroed in the three arrays + present=False
    # (consistent with run_hamer writing zero-value 2D for undetected frames).
    assoc_labels = None
    assoc_switches: list = []
    if wearer_association and candidates is not None:
        if len(candidates) != n_frames:
            raise ValueError(
                f"candidates frame count {len(candidates)} != main array frame count {n_frames} (hand.npz)"
            )
        assoc = associate_candidates(list(candidates), img_width, img_height)
        assoc_labels = np.asarray(assoc["labels"], np.int32)
        assoc_switches = [(int(f), int(s)) for f, s in assoc["switches"]]
        present = np.array(assoc["present"], bool)
        reorder_j2 = np.zeros_like(joints_2d)
        reorder_j3 = np.zeros_like(joints_3d_cam) if joints_3d_cam is not None else None
        # confidence is rebuildable from candidate score only in scalar (T,2) form;
        # per-joint (T,2,J) form has no protocol source, keep original values (slots may not
        # line up after reorder — a known limitation; no current writer produces that shape).
        reorder_conf = (
            np.zeros_like(confidence)
            if confidence is not None and np.asarray(confidence).ndim < 3
            else None
        )
        for i in range(n_frames):
            row = list(candidates[i] or [])
            for s in range(present.shape[1]):
                j = int(assoc_labels[i, s])
                if j < 0 or j >= len(row):
                    continue  # unobserved slot: three arrays stay zero, present already False
                c = row[j]
                reorder_j2[i, s] = c["joints_2d"]
                if reorder_j3 is not None:
                    j3 = c.get("joints_3d_cam")
                    if j3 is not None:
                        reorder_j3[i, s] = j3
                if reorder_conf is not None:
                    reorder_conf[i, s] = float(c.get("score", 0.0))
        joints_2d = reorder_j2
        if reorder_j3 is not None:
            joints_3d_cam = reorder_j3
        if reorder_conf is not None:
            confidence = reorder_conf

    # confidence -> per-(frame, hand) supporting-joint count, activating fill_short_gaps'
    # min_supporting_joints gate: per-joint confidence (T,2,J) counts joints with conf >= 0.5;
    # per-hand scalar (T,2) counts the whole hand's 21 joints as support when trusted.
    if confidence is None:
        supporting = np.full(present.shape, n_joints, np.int32)
    else:
        conf = np.asarray(confidence, np.float64)
        if conf.ndim >= 3:
            supporting = (conf >= 0.5).sum(axis=-1).astype(np.int32)
        else:
            supporting = np.where(conf >= 0.5, n_joints, 0).astype(np.int32)
    filled, audit = fill_short_gaps(present, max_gap=max_gap,
                                    min_supporting_joints=min_supporting_joints,
                                    supporting_joints=supporting)
    # Per-category counts for the 4th check being "unable to evaluate" (not rejections;
    # independent of audit["reasons"])
    traj_skipped: dict = {}

    joints_world, reproj, alignment = backproject_to_world(
        mano_trans, mano_pose, joints_2d, depth, intrinsics, camera_poses,
        joint_source=joint_source, joints_3d_cam=joints_3d_cam,
    )

    def _reject_fill(reason: str) -> None:
        """Filled frame rejected: decrement the audit counts (filled counts frames, rejected
        counts rejections) and accumulate the reason."""
        audit["rejected"] += 1
        audit["filled"] -= 1
        audit["reasons"][reason] = audit["reasons"].get(reason, 0) + 1

    # Filled-frame joints are linearly interpolated from the adjacent trusted observations —
    # no hallucination (no new structure, only bridging); the interpolation must then pass
    # the four checks (depth/view/wrist displacement/trajectory deviation), otherwise the
    # whole frame is dropped.
    for i in range(n_frames):
        for h in range(present.shape[1]):
            if not filled[i, h]:
                continue
            prev_i = i - 1
            next_i = i + 1
            while prev_i >= 0 and not present[prev_i, h]:
                prev_i -= 1
            while next_i < n_frames and not present[next_i, h]:
                next_i += 1
            if prev_i < 0 or next_i >= n_frames:
                # Endpoint observation is guaranteed by fill_short_gaps; this is just a
                # defensive guard (handled consistently)
                _reject_fill("missing_one_side_observation")
                filled[i, h] = False
                joints_world[i, h] = np.nan
                reproj[i, h] = UNTRUSTED_REPROJ_ERROR
                continue
            alpha = (i - prev_i) / (next_i - prev_i)
            prev_j, next_j = joints_world[prev_i, h], joints_world[next_i, h]
            # NaN-safe: only joints trusted at both ends participate in the interpolation,
            # the rest stay NaN (no hallucinated backfill); an endpoint whole-hand untrusted
            # (NaN semantics) -> candidate all-NaN -> the depth check necessarily rejects.
            both = np.isfinite(prev_j).all(axis=1) & np.isfinite(next_j).all(axis=1)
            candidate = np.full_like(prev_j, np.nan)
            candidate[both] = (1 - alpha) * prev_j[both] + alpha * next_j[both]
            reason, traj_skip = _fill_reject_reason(
                candidate, prev_j,
                min_supporting_joints=min_supporting_joints,
                depth_frame=depth[i],
                intrinsics_frame=intrinsics[i],
                camera_pose_frame=camera_poses[i],
                joints_2d_used=joints_2d[[prev_i, next_i], h],
                image_hw=depth.shape[1:],
                wrist_traj=joints_world[:, h, 0],
                prev_i=prev_i,
                next_i=next_i,
                observed=present[:, h],
            )
            if reason is None and traj_skip is not None:
                # The 4th check cannot be evaluated -> pass, but count by category so QC can
                # see it (a non-rejection count):
                #   "sequence_boundary"    the gap touches the sequence boundary, no neighbor frames;
                #   "neighbour_unobserved" the neighbor frame missed that hand (no observation, cannot
                #                         stand in for measured displacement);
                #   "wrist_untrusted"      the wrist of an involved frame is not finite (observed but
                #                         that hand untrusted).
                traj_skipped[traj_skip] = traj_skipped.get(traj_skip, 0) + 1
            if reason is not None:
                _reject_fill(reason)
                filled[i, h] = False
                # Rejected fills keep untrusted semantics: that frame, that hand has NaN
                # joints + 1e6 error sentinel
                joints_world[i, h] = np.nan
                reproj[i, h] = UNTRUSTED_REPROJ_ERROR
                continue
            joints_world[i, h] = candidate
            # A filled frame has no observation to compare against -> the reprojection error
            # is a sentinel, not a measurement (written NaN -> 1e6)
            reproj[i, h] = np.nan

    # Wearer association uses the **final** valid observation mask (present ∪ accepted
    # fills). It must be rebuilt after the rejection loop: rejected filled frames have no
    # usable joints (NaN/zero), and if they were still treated as "observed" in identity
    # ranking, garbage 2D positions would flip the left/right identity (and fabricate
    # identity-switch events).
    effective = present.copy()
    effective[filled] = True
    # **Accepted** filled frames likewise have no real detections (in real data run_hamer.py
    # writes zero-value 2D for undetected frames); feeding those zeros straight into the
    # ranking puts their x-mean at 0 -> always leftmost -> fabricated identity switches
    # (earlier work only fixed rejected fills). Here "effective 2D" is constructed: filled
    # frames get a substitute x position by linearly interpolating the endpoint observations'
    # x-means. Why interpolate the x-mean (rather than reproject the interpolated world
    # wrist to pixels): `associate_wearer` ranks by per-hand 2D x-mean, so interpolating
    # exactly this quantity matches the ranking key, and it does not rely on K/w2c still
    # being available at the filled frame — enough to yield the correct left/right rank.
    joints_2d_eff = np.asarray(joints_2d).copy()
    for i in range(n_frames):
        for h in range(present.shape[1]):
            if not filled[i, h]:
                continue
            prev_i = i - 1
            next_i = i + 1
            while prev_i >= 0 and not present[prev_i, h]:
                prev_i -= 1
            while next_i < n_frames and not present[next_i, h]:
                next_i += 1
            if prev_i < 0 or next_i >= n_frames:
                continue  # defensive: cannot happen (fill_short_gaps guarantees observations at both ends)
            alpha = (i - prev_i) / (next_i - prev_i)
            x_eff = (1 - alpha) * float(np.mean(joints_2d[prev_i, h, :, 0])) + alpha * float(
                np.mean(joints_2d[next_i, h, :, 0])
            )
            joints_2d_eff[i, h, :, 0] = x_eff
    if assoc_labels is not None:
        # A2: association already ran up front and reordered the main arrays; only the
        # identity plane is written here. hand_ids semantics upgrade from x-order rank to
        # "candidate index bound to each slot in that frame"; switches upgrade from x-order
        # changes to entity re-assignment events (associate_wearer's old semantics are kept
        # for the no-candidates fallback) — npz key shapes are unchanged (hand_ids (T,2),
        # identity_switches (N,2)), publish only reads the switch count.
        hand_ids = assoc_labels
        switches = assoc_switches
    elif wearer_association:
        hand_ids, switches = associate_wearer(joints_2d_eff, effective)
    else:
        hand_ids = np.tile(np.arange(present.shape[1], dtype=np.int32), (n_frames, 1))
        switches = []

    # Untrusted frames keep the large-error sentinel: NaN -> replaced by 1e6 before writing
    # (stays numeric, directly usable by QC statistics). Filled frames (accepted or
    # rejected) have no observation to compare against -> their error is a "sentinel", not a
    # "measurement", same convention as untrusted hand frames (1e6), removed together by the
    # QC sentinel filter — never passed off as low-error measurements.
    reproj = np.where(np.isfinite(reproj), reproj, UNTRUSTED_REPROJ_ERROR).astype(np.float32)
    # Sentinel values must not participate in quality statistics (or mean/max get polluted
    # by 1e6); report only trusted-frame statistics.
    trusted = reproj[reproj < UNTRUSTED_REPROJ_ERROR]
    flagged_hands = int((reproj >= UNTRUSTED_REPROJ_ERROR).sum())
    reproj_mean = float(trusted.mean()) if trusted.size else 0.0
    reproj_max = float(trusted.max()) if trusted.size else 0.0
    src = np.full((n_frames,), str(depth_source), dtype=object)

    save_artifact(
        ws / "artifacts" / "fused.npz",
        joints_world=joints_world,
        hand_ids=hand_ids,
        reprojection_error=reproj,
        # Per-hand alignment method ("kabsch"/"root_only"/"none", an object string array
        # like depth_source). Optional key: older consumers (QC/publish) are unaffected if
        # they don't read it; QC report presentation is a later item.
        alignment=alignment,
        joint_source=np.array(joint_source),
        depth_source=src,
        identity_switches=np.asarray(switches, np.int32).reshape(-1, 2),
        gapfill_filled=np.int32(audit["filled"]),
        gapfill_rejected=np.int32(audit["rejected"]),
        # Fill audit persistence: rejection-reason distribution + per-category counts of the
        # 4th check being "unable to evaluate and passed", stored as JSON strings (np.load
        # readable, allow_pickle=False safe) for QC report rendering. run_fusion's return
        # value already carries them, but the Actor discards return values
        # (actors.CpuStageActor) and the manifest does not record them — hence they must be
        # persisted in the artifact (explicitly deferred at the manifest level).
        gapfill_reasons=np.array(json.dumps(audit["reasons"], ensure_ascii=False)),
        gapfill_skipped=np.array(json.dumps(traj_skipped, ensure_ascii=False)),
        n_frames=np.int32(n_frames),
        fps=np.int32(fps),
    )
    return {
        "episode_id": episode_id,
        "n_frames": n_frames,
        "gapfill_filled": audit["filled"],
        "gapfill_rejected": audit["rejected"],
        "identity_switches": len(switches),
        # Rejection reasons counted by name (sum always equals gapfill_rejected; pure
        # rejection semantics).
        "gapfill_reasons": audit["reasons"],
        # Per-category counts of the 4th check being "unable to evaluate and passed" (not
        # rejections): sequence_boundary / neighbour_unobserved / wrist_untrusted, in their
        # own key so as not to pollute rejection-reason semantics.
        "gapfill_skipped": traj_skipped,
        # Metric source of the reprojection error ("mano" is informative; "joints_2d" is a
        # round-trip consistency check only).
        "joint_source": joint_source,
        "reprojection_error_mean_px": reproj_mean,
        "reprojection_error_max_px": reproj_max,
        "flagged_hands": flagged_hands,
    }
