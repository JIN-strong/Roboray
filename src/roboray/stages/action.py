"""Stage 3: action timeline + AI Function (pluggable VLM) + state/action tensors."""

import base64
from pathlib import Path

import cv2
import numpy as np

from roboray.models import registry

from .artifacts import save_artifact

__all__ = [
    "segment_motions",
    "run_action",
    "select_keyframe_indices",
    "load_keyframe_images",
]

# Keyframe cap per request: even long segments must not blow up the request body/context.
MAX_KEYFRAMES = 16


def segment_motions(
    joints_world: np.ndarray,
    fps: int,
    speed_thresh: float = 0.05,
    min_len: int = 3,
    contact_thresh: float = 0.02,
    min_contact: int = 3,
) -> list:
    """Speed-threshold segmentation + low-speed dwell = contact candidate. Returns [(start, end, kind)].

    kind ∈ ``"motion"|"contact"``: consecutive spans with wrist speed > ``speed_thresh``
    are motion segments (kept when >= ``min_len`` frames), consecutive dwell spans
    < ``contact_thresh`` are contact candidates (kept when >= ``min_contact`` frames),
    and transition frames between the two thresholds belong to no segment.
    With no segments at all, falls back to the whole span ``[(0, T, "motion")]``
    (same semantics as the old pair form).
    """
    wrist = joints_world[:, :, 0, :]  # (T, hands, 3)
    speed = np.linalg.norm(np.diff(wrist, axis=0), axis=-1).max(axis=1)  # (T-1,)
    active = speed > speed_thresh
    idle = speed < contact_thresh
    segments = []
    i = 0
    n = len(active)
    while i < n:
        if active[i]:
            start = i
            while i < n and active[i]:
                i += 1
            if i - start >= min_len:
                segments.append((start, i, "motion"))  # half-open; frame i-1 still belongs
        elif idle[i]:
            start = i
            while i < n and idle[i]:
                i += 1
            if i - start >= min_contact:
                segments.append((start, i, "contact"))
        else:
            i += 1  # transition frame (between the two thresholds): no segment
    if not segments:
        segments = [(0, joints_world.shape[0], "motion")]
    return segments


def select_keyframe_indices(
    n_frames: int, stride: int, max_frames: int = MAX_KEYFRAMES
) -> list:
    """Pick keyframe indices by stride: always include first/last; over the cap, thin
    uniformly (first/last still kept)."""
    if n_frames <= 0:
        return []
    step = max(1, int(stride))
    idx = list(range(0, n_frames, step))
    if idx[-1] != n_frames - 1:
        idx.append(n_frames - 1)
    if len(idx) > max_frames:
        picks = np.linspace(0, len(idx) - 1, max_frames)
        idx = sorted({idx[int(round(p))] for p in picks})
    return idx


def load_keyframe_images(
    ws: Path, stride: int, max_frames: int = MAX_KEYFRAMES
) -> list:
    """Encode prepared keyframes as data URIs (spec §9 visual input).

    Missing/corrupt frames_left.npy falls back to an empty list — visual input is optional
    and must not interrupt an entire episode on its account (the decision against VLM
    unreachability only concerns auth/address issues).
    """
    try:
        frames = np.ascontiguousarray(np.load(ws / "prepared" / "frames_left.npy"))
    except (OSError, ValueError, EOFError):
        return []
    images = []
    for i in select_keyframe_indices(len(frames), stride, max_frames):
        try:
            ok, buf = cv2.imencode(".jpg", frames[i])
        except cv2.error:  # frame is not an encodable image: skip it, keep the rest
            continue
        if not ok:
            continue
        images.append(
            "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")
        )
    return images


def run_action(
    workspace_root: str,
    episode_id: str,
    semantic_model_name: str = "openai_vlm",
    keyframe_stride: int = 30,
    **semantic_kwargs,
) -> dict:
    """Action timeline + semantic annotation + state/action tensors (with gripper channel).

    action.npz keys: labels / segment_start / segment_end / segment_kind (triple
    segmentation, kind ∈ motion|contact) / gripper (T, n_hands) / state (T, D+n_hands) /
    action (state shifted one frame) / n_frames / fps.

    **gripper = thumb(4)-index(8) tip distance / batch max, NOT real gripper state** — it
    only carries distance-ratio semantics ([0,1], clip-normalized). Missing-hand frames
    (NaN joints) -> tip_dist NaN -> gripper NaN; max_d uses nanmax and observed.any() to
    avoid all-NaN division by zero.
    """
    ws = Path(workspace_root) / episode_id
    fused_path = ws / "artifacts" / "fused.npz"
    if not fused_path.exists():
        raise ValueError(f"action understanding artifact missing: {fused_path.name}")
    try:
        with np.load(fused_path, allow_pickle=True) as z:
            joints_world = z["joints_world"]
            fps = int(z["fps"])
    except (OSError, ValueError, EOFError) as exc:
        raise ValueError(f"fusion artifact corrupt: {fused_path.name} ({exc})") from exc

    segments = segment_motions(joints_world, fps)

    semantic_path = ws / "artifacts" / "semantic_raw.npz"
    if semantic_model_name not in (None, "", "none"):
        factory = registry.get_semantic_model_class(semantic_model_name)
        model = factory()
        for attr, key in (("_base_url", "vlm_base_url"), ("_model", "vlm_model"),
                          ("_api_key_env", "vlm_api_key_env")):
            value = semantic_kwargs.get(key)
            if value is not None and hasattr(model, attr):
                setattr(model, attr, value)
        images = load_keyframe_images(ws, keyframe_stride)
        summary = (
            f"episode={episode_id} segments={segments} fps={fps} "
            f"keyframe_stride={keyframe_stride}"
        )
        model.label(summary, images, semantic_path, n_segments=len(segments))
        with np.load(semantic_path, allow_pickle=True) as z:
            labels = [str(x) for x in z["labels"]]
    else:
        labels = [f"segment_{i}" for i in range(len(segments))]

    if len(labels) < len(segments):
        labels = labels + [f"segment_{i}" for i in range(len(labels), len(segments))]
    labels = labels[: len(segments)]

    t, n_hands = joints_world.shape[0], joints_world.shape[1]
    # Tensor flattening + gripper channel column: state becomes 2D (T, hands*21*3 + hands).
    # gripper = thumb-index tip distance / batch max, not real gripper state (distance-ratio
    # semantics); missing-hand frames pass NaN through (publish reshapes old 4D artifacts,
    # 2D passes through directly).
    flat = joints_world.reshape(t, -1).astype(np.float32)  # (T, hands*21*3)
    tip_dist = np.linalg.norm(
        joints_world[:, :, 4, :3] - joints_world[:, :, 8, :3], axis=-1
    )  # (T, hands)
    observed = np.isfinite(tip_dist).all(axis=1)
    max_d = float(np.nanmax(tip_dist)) if observed.any() else 1.0
    gripper = np.where(
        np.isfinite(tip_dist), np.clip(tip_dist / (max_d or 1.0), 0.0, 1.0), np.nan
    ).astype(np.float32)  # distance ratio, not real gripper state
    state = np.concatenate([flat, gripper], axis=1)
    action = np.empty_like(state)
    action[:-1] = state[1:]
    action[-1] = state[-1]

    save_artifact(
        ws / "artifacts" / "action.npz",
        labels=np.array(labels, dtype=object),
        segment_start=np.array([s for s, _e, _k in segments], np.int32),
        segment_end=np.array([e for _s, e, _k in segments], np.int32),
        segment_kind=np.array([k for _s, _e, k in segments], dtype=object),
        gripper=gripper,
        state=state,
        action=action,
        n_frames=np.int32(t),
        fps=np.int32(fps),
    )
    return {"episode_id": episode_id, "n_segments": len(segments), "labels": labels}
