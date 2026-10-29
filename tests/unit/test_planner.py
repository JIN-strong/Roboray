from roboray.api.config import ReconstructionRequest
from roboray.runtime.episode import EpisodeRecord, EpisodeSpec, Stage, StageStatus
from roboray.runtime.planner import build_cpu_kwargs, episode_ready_tasks

CONFIG = ReconstructionRequest.from_dict(
    {
        "input": {"uri": "/in"},
        "output": {"uri": "/out"},
        "execution": {"max_gpus": 2},
        "stages": {},
    }
)
SPEC = EpisodeSpec(episode_id="ep_0001", rel_dir="ep_0001", has_stereo=True)


def test_fresh_episode_only_offers_prepare():
    rec = EpisodeRecord.new("ep_0001")
    tasks = episode_ready_tasks(rec, SPEC, CONFIG)
    assert [t.stage for t in tasks] == [Stage.PREPARE]
    assert tasks[0].needs_gpu is False


def test_after_prepare_both_scene_and_hand_are_offered():
    rec = EpisodeRecord.new("ep_0001")
    rec.stages[Stage.PREPARE] = StageStatus.DONE
    tasks = episode_ready_tasks(rec, SPEC, CONFIG)
    assert {t.stage for t in tasks} == {Stage.SCENE, Stage.HAND}
    assert all(t.needs_gpu for t in tasks)


def test_fusion_requires_both_scene_and_hand():
    rec = EpisodeRecord.new("ep_0001")
    rec.stages[Stage.PREPARE] = StageStatus.DONE
    rec.stages[Stage.SCENE] = StageStatus.DONE
    tasks = episode_ready_tasks(rec, SPEC, CONFIG)
    assert [t.stage for t in tasks] == [Stage.HAND]


def test_stereo_episode_resolves_scene_model_to_stereo():
    rec = EpisodeRecord.new("ep_0001")
    rec.stages[Stage.PREPARE] = StageStatus.DONE
    tasks = {t.stage: t for t in episode_ready_tasks(rec, SPEC, CONFIG)}
    assert tasks[Stage.SCENE].resolution_key  # non-empty


def test_monocular_episode_scene_task_present():
    spec = EpisodeSpec(episode_id="ep_0002", rel_dir="ep_0002", has_stereo=False)
    rec = EpisodeRecord.new("ep_0002")
    rec.stages[Stage.PREPARE] = StageStatus.DONE
    tasks = {t.stage: t for t in episode_ready_tasks(rec, spec, CONFIG)}
    assert Stage.SCENE in tasks


def test_failed_episode_offers_nothing():
    rec = EpisodeRecord.new("ep_0001")
    rec.stages[Stage.PREPARE] = StageStatus.FAILED
    assert episode_ready_tasks(rec, SPEC, CONFIG) == []


def test_build_cpu_kwargs_for_publish_contains_outputs_root():
    fn_name, kwargs = build_cpu_kwargs(
        Stage.PUBLISH, "ep_0001", SPEC, CONFIG,
        storage_root="/in", workspace_root="/ws", output_root="/out",
    )
    assert fn_name == "publish"
    assert kwargs["output_root"] == "/out"
    assert kwargs["episode_id"] == "ep_0001"


def test_build_cpu_kwargs_for_fusion_passes_gap_config():
    _, kwargs = build_cpu_kwargs(
        Stage.FUSION, "ep_0001", SPEC, CONFIG,
        storage_root="/in", workspace_root="/ws", output_root="/out",
    )
    assert kwargs["unobserved_gap_s"] == 0.2
    assert kwargs["min_supporting_joints"] == 8
    assert kwargs["wearer_association"] is True
