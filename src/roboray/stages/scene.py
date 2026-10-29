"""Stage 2 Scene path: build the payload (actual inference happens inside the GPU actor).

Routing rules (spec §6):
- With stereo -> depth via fast_foundation_stereo, pose via vipe;
- Without stereo -> everything via vipe.
``model_name == "auto"`` is resolved here into a concrete model name.
"""

import json
from pathlib import Path

from roboray.models.protocol import resolution_key

__all__ = [
    "resolve_scene_model",
    "scene_payload_args",
    "prepared_has_stereo",
    "prepared_resolution_key",
]


def resolve_scene_model(config_model: str, has_stereo: bool) -> str:
    if config_model != "auto":
        return config_model
    return "fast_foundation_stereo" if has_stereo else "vipe"


def _load_prepared_calib(workspace_root: str, episode_id: str) -> dict:
    """Read prepared/calib.json; missing/invalid file returns an empty dict (caller decides the default)."""
    p = Path(workspace_root) / episode_id / "prepared" / "calib.json"
    try:
        raw = json.loads(p.read_text())
        return raw if isinstance(raw, dict) else {}
    except (OSError, ValueError):
        return {}


def prepared_has_stereo(workspace_root: str, episode_id: str) -> bool:
    """Read the stereo flag from prepared calib.json (prepare's single authoritative source).

    Missing file/field -> False (monocular decision).
    """
    calib = _load_prepared_calib(workspace_root, episode_id)
    return bool(calib.get("has_stereo", False))


def prepared_resolution_key(workspace_root: str, episode_id: str) -> str:
    """Read the real frame resolution tier from prepared calib.json.

    prepare always completes before SCENE/HAND, so calib.json exists on the normal path;
    if the file is unreadable, fall back to the placeholder tier DEFAULT_RESOLUTION_KEY
    (only affects the cold-start prior).
    """
    calib = _load_prepared_calib(workspace_root, episode_id)
    width, height = calib.get("width"), calib.get("height")
    if not width or not height:
        # Local import avoids a circular dependency: planner imports this module at top level.
        from roboray.runtime.planner import DEFAULT_RESOLUTION_KEY

        return DEFAULT_RESOLUTION_KEY
    return resolution_key(int(width), int(height))


def scene_payload_args(
    workspace_root: str,
    episode_id: str,
    model_name: str,
    out_path: str,
    stereo_max_image_side_px: int = 640,
) -> dict:
    """Build the Scene GPU payload — **paths and scalars only, no arrays**.

    Frame arrays are read by ``GpuModelActor`` on **its own node**. On a multi-node cluster
    the driver runs on the head/CPU worker and **cannot read** node-local workspaces; and
    passing arrays across nodes through the Ray object store is exactly the "large-array
    cross-node transfer" the spec eliminates.

    ``stereo_max_image_side_px`` (spec §6 FFS input resolution control) flows through
    Actor -> adapter ``--extra`` -> ``run_ffs.py``: the stereo pair is scaled so
    ``max(H, W) <= that cap``, ``fx`` scaled proportionally; since disparity scales with
    resolution, ``fx'·B/d' == fx·B/d``, so **depth is invariant to scaling** — this knob
    trades accuracy for speed. ``0`` means no scaling.
    """
    return {
        "kind": "scene",
        "episode_id": episode_id,
        "model_name": model_name,
        "workspace_root": str(workspace_root),
        "out_path": str(out_path),
        "stereo_max_image_side_px": int(stereo_max_image_side_px),
    }
