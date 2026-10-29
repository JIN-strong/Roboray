import pytest

from roboray.api.config import ConfigError, ReconstructionRequest

ARTICLE_CONFIG = {
    "input": {"uri": "oss://bucket/raw"},
    "output": {"uri": "oss://bucket/results/run-unique", "format": "lerobot_v3"},
    "execution": {"max_gpus": 8},
    "stages": {
        "scene_reconstruction": {"stereo_max_image_side_px": 640},
        "hand_reconstruction": {
            "inference_fps": 30,
            "max_hands": 6,
            "wearer_association": True,
            "unobserved_gap_s": 0.2,
            "min_supporting_joints": 8,
        },
        "visualization": {"camera_view": True, "world_view": True},
    },
}


def test_article_config_parses_with_defaults_filled():
    req = ReconstructionRequest.from_dict(ARTICLE_CONFIG)
    assert req.input.uri == "oss://bucket/raw"
    assert req.output.format == "lerobot_v3"
    assert req.execution.max_gpus == 8
    assert req.execution.backend == "real"  # default is real models
    assert req.stages.scene_reconstruction.stereo_max_image_side_px == 640
    assert req.stages.hand_reconstruction.max_hands == 6
    assert req.stages.hand_reconstruction.unobserved_gap_s == 0.2
    assert req.stages.hand_reconstruction.min_supporting_joints == 8
    assert req.stages.action_understanding.keyframe_stride == 30  # default when not given
    assert req.stages.visualization.camera_view is True


def test_hand_stage_max_hands_defaults_to_4():
    """Config surface: max_hands is explicitly configurable; default 4 (same source
    as the script's DEFAULT_MAX_HANDS)."""
    req = ReconstructionRequest.from_dict({
        "input": {"uri": "oss://bucket/raw"},
        "output": {"uri": "oss://bucket/results/run-unique", "format": "lerobot_v3"},
    })
    assert req.stages.hand_reconstruction.max_hands == 4
    assert req.stages.hand_reconstruction.inference_fps == 30  # existing default on the same chain unchanged


def test_unknown_key_is_rejected_with_path():
    bad = {**ARTICLE_CONFIG, "stages": {"scene_reconstruction": {"nope": 1}}}
    with pytest.raises(ConfigError) as exc:
        ReconstructionRequest.from_dict(bad)
    assert "nope" in str(exc.value)
    assert "scene_reconstruction" in str(exc.value)


def test_missing_required_uri_is_rejected():
    with pytest.raises(ConfigError) as exc:
        ReconstructionRequest.from_dict({"output": {"uri": "x"}})
    assert "input" in str(exc.value)


def test_wrong_type_is_rejected():
    bad = {**ARTICLE_CONFIG, "execution": {"max_gpus": "eight"}}
    with pytest.raises(ConfigError) as exc:
        ReconstructionRequest.from_dict(bad)
    assert "max_gpus" in str(exc.value)


def test_bool_is_not_accepted_as_int():
    bad = {**ARTICLE_CONFIG, "execution": {"max_gpus": True}}
    with pytest.raises(ConfigError):
        ReconstructionRequest.from_dict(bad)


def test_roundtrip_to_dict():
    req = ReconstructionRequest.from_dict(ARTICLE_CONFIG)
    again = ReconstructionRequest.from_dict(req.to_dict())
    assert again == req


def test_models_env_config_roundtrip():
    from roboray.api.config import ModelEnvsConfig

    raw = {
        "input": {"uri": "a"},
        "output": {"uri": "b"},
        "models": {
            "vipe_python": "/opt/vipe/bin/python",
            "ffs_weights": "/data/ffs",
            "lerobot_python": "/opt/ego-lerobot/bin/python",
            "lerobot_script": "/tmp/fake_export.py",
        },
    }
    req = ReconstructionRequest.from_dict(raw)
    assert req.models.vipe_python == "/opt/vipe/bin/python"
    assert req.models.ffs_python is None
    assert req.models.hammer_weights is None or req.models.hammer_weights == ""
    assert req.models.lerobot_python == "/opt/ego-lerobot/bin/python"
    assert req.models.lerobot_script == "/tmp/fake_export.py"
    again = ReconstructionRequest.from_dict(req.to_dict())
    assert again.models == req.models


def test_models_env_unknown_key_rejected():
    bad = {
        "input": {"uri": "a"},
        "output": {"uri": "b"},
        "models": {"nope": 1},
    }
    import pytest
    from roboray.api.config import ConfigError

    with pytest.raises(ConfigError) as exc:
        ReconstructionRequest.from_dict(bad)
    assert "nope" in str(exc.value)
