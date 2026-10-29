import json
import sys
from pathlib import Path

import pytest

ray = pytest.importorskip("ray")

from roboray import EgoReconstruction, ReconstructionRequest
from roboray.models import registry
from scripts.make_synthetic_ego import generate_dataset
from tests.fakes.models import install_fake_models

pytestmark = pytest.mark.slow

# publish calls the LeRobot export in a subprocess: the real script needs the
# ego-lerobot env, which production points to via ROBORAY_LEROBOT_PYTHON; e2e only
# verifies orchestration and on-disk layout, so it uses a stand-in script following
# the real contract (NPZ in / __RESULT__ out / same directory layout).
_FAKE_EXPORT = Path(__file__).resolve().parents[1] / "fakes" / "modelenv_scripts" / "fake_lerobot_export.py"


@pytest.fixture(autouse=True)
def _ray():
    """Function-level cluster + worker-side stand-in hook (same deadlock guard as
    tests/unit/test_actors.py)."""
    if ray.is_initialized():
        ray.shutdown()
    ray.init(
        # num_gpus=4 (not 2): GpuModelActor persists for the cluster lifetime. After
        # slots are sized by GPU count, max_gpus=2 builds 2 GPU actors on a single
        # node; test_resume... runs run() twice on the same cluster (2 actors each),
        # for 4 resident actors total. With only 2 virtual GPUs the first run fills
        # them and the second run's new pool stays PENDING forever (same precedent as
        # test_driver_parallel's num_gpus=4).
        num_gpus=4,
        num_cpus=8,
        include_dashboard=False,
        log_to_driver=False,
        runtime_env={
            "working_dir": ".",
            "worker_process_setup_hook": "tests.fakes.worker_setup.install_for_ray",
            "env_vars": {
                "ROBORAY_FAKE_SCENE_SECONDS": "0.02",
                "ROBORAY_FAKE_HAND_SECONDS": "0.02",
                "ROBORAY_LEROBOT_PYTHON": sys.executable,
                "ROBORAY_LEROBOT_SCRIPT": str(_FAKE_EXPORT),
            },
        },
    )
    yield
    ray.shutdown()


@pytest.fixture
def _fakes():
    registry.reset_registry_for_tests()
    install_fake_models(scene_seconds=0.02, hand_seconds=0.02)
    yield
    registry.reset_registry_for_tests()


def test_public_api_runs_and_writes_manifest(tmp_path):
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    generate_dataset(in_dir, n_episodes=2, n_frames=10, size=(96, 72), seed=0)
    config = {
        "input": {"uri": str(in_dir)},
        "output": {"uri": str(out_dir), "format": "lerobot_v3"},
        "execution": {"max_gpus": 2, "num_cpu_actors": 4, "backend": "test"},
        "stages": {
            "scene_reconstruction": {"model": "fake_scene"},
            "hand_reconstruction": {"model": "fake_hand"},
            "action_understanding": {"model": "fake_vlm"},
            "visualization": {"camera_view": True, "world_view": True},
        },
    }
    request = ReconstructionRequest.from_dict(config)
    result = EgoReconstruction(request, run_root=str(tmp_path / "ws")).run()
    assert result.failed == []
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["format"] == "lerobot_v3"
    assert {e["episode_id"] for e in manifest["episodes"]} == {"ep_0001", "ep_0002"}
    assert manifest["summary"]["failed"] == 0
    assert "scheduler_snapshots" in manifest


def test_one_bad_episode_does_not_kill_the_batch(tmp_path):
    """A single episode failing -> the batch continues and the rest produce
    normally."""
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    generate_dataset(in_dir, n_episodes=2, n_frames=10, size=(96, 72), seed=0)
    # break the second episode's calibration
    (in_dir / "ep_0002" / "calibration.json").unlink()
    config = {
        "input": {"uri": str(in_dir)},
        "output": {"uri": str(out_dir)},
        "execution": {"max_gpus": 2, "num_cpu_actors": 4, "backend": "test"},
        "stages": {
            "scene_reconstruction": {"model": "fake_scene"},
            "hand_reconstruction": {"model": "fake_hand"},
            "action_understanding": {"model": "fake_vlm"},
        },
    }
    result = EgoReconstruction(
        ReconstructionRequest.from_dict(config), run_root=str(tmp_path / "ws")
    ).run()
    assert result.failed == ["ep_0002"]
    assert (out_dir / "lerobot_v3" / "ep_0001" / "meta" / "info.json").exists()
    assert not (out_dir / "lerobot_v3" / "ep_0002").exists()
    manifest = json.loads((out_dir / "manifest.json").read_text())
    failed = [e for e in manifest["episodes"] if e["episode_id"] == "ep_0002"][0]
    assert "calibration.json" in failed["error"]


def test_resume_skips_completed_stages(tmp_path):
    """Resume: after one full run, re-running must not re-execute stages already
    DONE."""
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    ws_dir = tmp_path / "ws"
    generate_dataset(in_dir, n_episodes=1, n_frames=8, size=(96, 72), seed=0)
    config = {
        "input": {"uri": str(in_dir)},
        "output": {"uri": str(out_dir)},
        "execution": {"max_gpus": 2, "num_cpu_actors": 4, "backend": "test"},
        "stages": {
            "scene_reconstruction": {"model": "fake_scene"},
            "hand_reconstruction": {"model": "fake_hand"},
            "action_understanding": {"model": "fake_vlm"},
        },
    }
    req = ReconstructionRequest.from_dict(config)
    EgoReconstruction(req, run_root=str(ws_dir)).run()
    first = json.loads((ws_dir / "ep_0001" / "state.json").read_text())
    second_result = EgoReconstruction(req, run_root=str(ws_dir)).run()
    assert second_result.failed == []
    # The second run should not produce new stage-start events (all already DONE)
    assert [e for e in second_result.timeline if e["event"] == "start"] == []
    assert json.loads((ws_dir / "ep_0001" / "state.json").read_text())["timings"] == first["timings"]
