"""Pure-function planner: translate episode state into submittable tasks. No Ray dependency, easy to unit test."""

from dataclasses import dataclass

from roboray.api.config import ReconstructionRequest
from roboray.stages.scene import resolve_scene_model

from .episode import PREREQS, STAGE_ORDER, EpisodeRecord, EpisodeSpec, Stage, StageStatus

__all__ = ["StageTask", "episode_ready_tasks", "build_cpu_kwargs"]

_CPU_STAGES = {Stage.PREPARE, Stage.FUSION, Stage.ACTION, Stage.PUBLISH}

# Placeholder resolution tier used at planning time when the actual frame size
# is unknown; the actor reports the real tier at execution time, and this
# placeholder only affects the cold-start scheduling prior, never correctness.
DEFAULT_RESOLUTION_KEY = "640x480"


@dataclass(frozen=True)
class StageTask:
    episode_id: str
    stage: Stage
    kind: str          # "scene" | "hand" | "cpu"
    model_name: str    # only meaningful for GPU tasks
    resolution_key: str
    needs_gpu: bool


def episode_ready_tasks(
    record: EpisodeRecord, spec: EpisodeSpec, config: ReconstructionRequest
) -> list:
    """Return **all** stages this episode can currently submit (SCENE and HAND are returned together)."""
    if record.is_terminal():
        return []
    tasks = []
    # Check each PENDING stage one by one for all-DONE prerequisites (same rule
    # as next_runnable, but here we return several at once so SCENE/HAND can be
    # submitted in parallel).
    for stage in STAGE_ORDER:
        if record.stages[stage] is not StageStatus.PENDING:
            continue
        if not all(record.stages[p] is StageStatus.DONE for p in PREREQS[stage]):
            continue
        if stage in _CPU_STAGES:
            tasks.append(StageTask(spec.episode_id, stage, "cpu", "", DEFAULT_RESOLUTION_KEY, False))
        elif stage is Stage.SCENE:
            model = resolve_scene_model(config.stages.scene_reconstruction.model, spec.has_stereo)
            tasks.append(StageTask(spec.episode_id, stage, "scene", model, DEFAULT_RESOLUTION_KEY, True))
        elif stage is Stage.HAND:
            model = config.stages.hand_reconstruction.model
            tasks.append(StageTask(spec.episode_id, stage, "hand", model, DEFAULT_RESOLUTION_KEY, True))
    return tasks


def build_cpu_kwargs(
    stage: Stage,
    episode_id: str,
    spec: EpisodeSpec,
    config: ReconstructionRequest,
    storage_root: str,
    workspace_root: str,
    output_root: str,
) -> tuple:
    """Return (fn_name, kwargs) for CpuStageActor to call."""
    if stage is Stage.PREPARE:
        return (
            "prepare",
            {
                "storage_root": storage_root,
                "workspace_root": workspace_root,
                "episode_rel_dir": spec.rel_dir,
                "episode_id": episode_id,
            },
        )
    if stage is Stage.FUSION:
        h = config.stages.hand_reconstruction
        return (
            "fusion",
            {
                "workspace_root": workspace_root,
                "episode_id": episode_id,
                "wearer_association": h.wearer_association,
                "unobserved_gap_s": h.unobserved_gap_s,
                "min_supporting_joints": h.min_supporting_joints,
            },
        )
    if stage is Stage.ACTION:
        a = config.stages.action_understanding
        return (
            "action",
            {
                "workspace_root": workspace_root,
                "episode_id": episode_id,
                "semantic_model_name": a.model,
                "keyframe_stride": a.keyframe_stride,
                "vlm_base_url": a.vlm_base_url,
                "vlm_model": a.vlm_model,
                "vlm_api_key_env": a.vlm_api_key_env,
            },
        )
    if stage is Stage.PUBLISH:
        v = config.stages.visualization
        return (
            "publish",
            {
                "workspace_root": workspace_root,
                "output_root": output_root,
                "episode_id": episode_id,
                "make_visualization": bool(v.camera_view or v.world_view),
                "camera_view": v.camera_view,
                "world_view": v.world_view,
                "fps": v.fps,
            },
        )
    raise ValueError(f"stage {stage} is not a CPU stage")
