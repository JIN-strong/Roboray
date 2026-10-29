"""Stage 2 Hand path: build the payload."""

__all__ = ["hand_payload_args"]


def hand_payload_args(
    workspace_root: str,
    episode_id: str,
    model_name: str,
    out_path: str,
    inference_fps: int | None = None,
    max_hands: int | None = None,
) -> dict:
    """Build the Hand GPU payload — **paths and scalars only, no arrays** (same as Scene).

    When ``inference_fps`` (HandStageConfig, same name) is set, it is passed down: the
    model side samples inference at round(fps/inference_fps), and skipped frames get
    present=False, handed to A3 gap filling.
    When ``max_hands`` is not None (HandStageConfig, same name), it is passed down: the
    detection-count cap (defaulted by the script to DEFAULT_MAX_HANDS=4 when absent).
    """
    payload = {
        "kind": "hand",
        "episode_id": episode_id,
        "model_name": model_name,
        "workspace_root": str(workspace_root),
        "out_path": str(out_path),
    }
    if inference_fps:
        payload["inference_fps"] = int(inference_fps)
    if max_hands is not None:
        payload["max_hands"] = int(max_hands)
    return payload
