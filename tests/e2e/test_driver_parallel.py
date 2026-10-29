import json
import sys
from pathlib import Path

import pytest

ray = pytest.importorskip("ray")

from roboray.api.config import ReconstructionRequest
from roboray.models import registry
from roboray.ray_runtime.actors import build_actor_pool
from roboray.runtime.driver import DriverConfig, PipelineDriver
from scripts.make_synthetic_ego import generate_dataset
from tests.fakes.models import install_fake_models

pytestmark = pytest.mark.slow

# Same as tests/e2e/test_public_api.py: publish calls the LeRobot export via a
# subprocess, and the stand-in script follows the real contract (NPZ in /
# __RESULT__ out / meta/info.json + data/ layout).
_FAKE_EXPORT = Path(__file__).resolve().parents[1] / "fakes" / "modelenv_scripts" / "fake_lerobot_export.py"


@pytest.fixture(autouse=True)
def _ray(request):
    """Function-level cluster + worker-side double hook (same reasoning as the
    analogous comment in tests/unit/test_actors.py: GpuModelActor occupies a
    virtual GPU for its lifetime, so a module-level cluster + per-test new pools would deadlock).
    ``@pytest.mark.fake_fail_on(ep_id)`` configures the double's fail_on into the
    worker env (same as tests/unit/test_actors.py).
    """
    if ray.is_initialized():
        ray.shutdown()
    env_vars = {
        "ROBORAY_FAKE_SCENE_SECONDS": "0.05",
        "ROBORAY_FAKE_HAND_SECONDS": "0.05",
        "ROBORAY_LEROBOT_PYTHON": sys.executable,
        "ROBORAY_LEROBOT_SCRIPT": str(_FAKE_EXPORT),
    }
    marker = request.node.get_closest_marker("fake_fail_on")
    if marker:
        env_vars["ROBORAY_FAKE_FAIL_ON"] = marker.args[0]
    ray.init(
        # num_gpus=4 (not 2): GpuModelActor lives for the cluster's lifetime, and
        # test_adaptive_is_faster_than_serial_baseline builds two 2-actor pools
        # sequentially on the same cluster; with only 2 virtual GPUs, the second
        # pool would be permanently PENDING once the first fills them.
        num_gpus=4,
        num_cpus=8,
        include_dashboard=False,
        log_to_driver=False,
        runtime_env={
            "working_dir": ".",
            "worker_process_setup_hook": "tests.fakes.worker_setup.install_for_ray",
            "env_vars": env_vars,
        },
    )
    yield
    ray.shutdown()


@pytest.fixture
def _fakes():
    registry.reset_registry_for_tests()
    install_fake_models(scene_seconds=0.05, hand_seconds=0.05)
    yield
    registry.reset_registry_for_tests()


def _request(in_dir, out_dir, max_gpus=2, backend="test"):
    return ReconstructionRequest.from_dict(
        {
            "input": {"uri": str(in_dir)},
            "output": {"uri": str(out_dir), "format": "lerobot_v3"},
            "execution": {"max_gpus": max_gpus, "num_cpu_actors": 4,
                          "max_episodes_in_flight": 3, "backend": backend},
            "stages": {
                "scene_reconstruction": {"model": "fake_scene"},
                "hand_reconstruction": {"model": "fake_hand", "unobserved_gap_s": 0.2},
                "action_understanding": {"model": "fake_vlm"},
                "visualization": {"camera_view": True, "world_view": True},
            },
        }
    )


def test_pipeline_run_produces_all_outputs(tmp_path):
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    ws_dir = tmp_path / "ws"
    generate_dataset(in_dir, n_episodes=2, n_frames=12, size=(96, 72), seed=0)
    gpu, cpu = build_actor_pool(num_gpus=2, num_cpu_actors=4)
    driver = PipelineDriver(
        DriverConfig(storage_root=str(in_dir), workspace_root=str(ws_dir),
                     output_root=str(out_dir), serial_baseline=False, adaptive=True),
        _request(in_dir, out_dir),
        gpu,
        cpu,
    )
    result = driver.run()
    assert result.failed == []
    for ep in ("ep_0001", "ep_0002"):
        ds = out_dir / "lerobot_v3" / ep
        assert (ds / "meta" / "info.json").exists()
        assert (ds / "data").is_dir()
        assert (ds / "meta" / "episodes").is_dir()
        assert (out_dir / "qc" / ep / "report.html").exists()
        assert (out_dir / "viz" / ep / "camera.mp4").exists()
    info = json.loads((out_dir / "lerobot_v3" / "ep_0001" / "meta" / "info.json").read_text())
    assert info["codebase_version"] == "v3.0"
    # left/right hand 21 joints x 3 coords flattened (126) + one gripper column per
    # hand (thumb-index fingertip distance / batch max; ratio, not a real gripper) -> 2 hands = 128
    assert info["features"]["observation.state"]["shape"] == [128]
    assert info["features"]["action"]["shape"] == [128]
    assert info["total_frames"] == 12
    assert info["fps"] == 30


def test_pipeline_overlaps_stages_across_episodes(tmp_path):
    """Cross-stage parallelism: there must be a time window where one episode is in its
    later stages while another is preparing/inferring."""
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    ws_dir = tmp_path / "ws"
    generate_dataset(in_dir, n_episodes=3, n_frames=16, size=(96, 72), seed=1)
    gpu, cpu = build_actor_pool(num_gpus=2, num_cpu_actors=4)
    driver = PipelineDriver(
        DriverConfig(storage_root=str(in_dir), workspace_root=str(ws_dir),
                     output_root=str(out_dir), serial_baseline=False, adaptive=True),
        _request(in_dir, out_dir),
        gpu,
        cpu,
    )
    result = driver.run()
    tl = result.timeline
    events = {}
    for e in tl:
        events.setdefault((e["episode_id"], e["stage"]), []).append(e)
    # at least one instant with two different episodes running concurrently
    starts = sorted(e["t"] for e in tl if e["event"] == "start")
    ends = sorted(e["t"] for e in tl if e["event"] == "end")
    concurrent = 0
    for s in starts:
        concurrent = max(concurrent, sum(1 for s2 in starts if s2 <= s) - sum(1 for e2 in ends if e2 <= s))
    assert concurrent >= 2, f"no cross-stage parallelism observed; timeline={tl}"


def test_adaptive_is_faster_than_serial_baseline(tmp_path):
    """Behavior reproduction: the adaptive pipeline should be faster than the pure serial baseline.

    Each mode builds fresh actors so the serial run does not carry model-loaded state or
    EMA priors into the pipeline run; the comparison then reflects "orchestration" rather
    than "which ran first".
    """
    in_dir = tmp_path / "in"
    generate_dataset(in_dir, n_episodes=3, n_frames=12, size=(96, 72), seed=2)
    req_serial = _request(in_dir, tmp_path / "out_serial")
    req_pipe = _request(in_dir, tmp_path / "out_pipe")

    gpu_s, cpu_s = build_actor_pool(num_gpus=2, num_cpu_actors=4)
    serial = PipelineDriver(
        DriverConfig(storage_root=str(in_dir), workspace_root=str(tmp_path / "ws_serial"),
                     output_root=str(tmp_path / "out_serial"), serial_baseline=True, adaptive=False),
        req_serial, gpu_s, cpu_s,
    ).run()

    gpu_p, cpu_p = build_actor_pool(num_gpus=2, num_cpu_actors=4)
    pipeline = PipelineDriver(
        DriverConfig(storage_root=str(in_dir), workspace_root=str(tmp_path / "ws_pipe"),
                     output_root=str(tmp_path / "out_pipe"), serial_baseline=False, adaptive=True),
        req_pipe, gpu_p, cpu_p,
    ).run()

    assert pipeline.total_seconds < serial.total_seconds, (
        f"pipeline parallelism produced no gain: serial={serial.total_seconds:.3f}s "
        f"pipeline={pipeline.total_seconds:.3f}s"
    )


def test_adaptive_scheduler_snapshots_record_decisions(tmp_path):
    in_dir = tmp_path / "in"
    generate_dataset(in_dir, n_episodes=3, n_frames=12, size=(96, 72), seed=3)
    gpu, cpu = build_actor_pool(num_gpus=2, num_cpu_actors=4)
    driver = PipelineDriver(
        DriverConfig(storage_root=str(in_dir), workspace_root=str(tmp_path / "ws"),
                     output_root=str(tmp_path / "out"), serial_baseline=False, adaptive=True),
        _request(in_dir, tmp_path / "out"),
        gpu, cpu,
    )
    result = driver.run()
    assert len(result.scheduler_snapshots) >= 2
    snaps = result.scheduler_snapshots
    assert {"chosen", "scene_q", "hand_q", "predict", "loaded"} <= set(snaps[0])


@pytest.mark.fake_fail_on("ep_0002")
def test_bad_episode_does_not_kill_batch(tmp_path):
    """A bad episode (SCENE inference fails) must not sink the batch: the other episodes still produce output."""
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    ws_dir = tmp_path / "ws"
    generate_dataset(in_dir, n_episodes=3, n_frames=12, size=(96, 72), seed=4)
    gpu, cpu = build_actor_pool(num_gpus=2, num_cpu_actors=4)
    driver = PipelineDriver(
        DriverConfig(storage_root=str(in_dir), workspace_root=str(ws_dir),
                     output_root=str(out_dir), serial_baseline=False, adaptive=True),
        _request(in_dir, out_dir),
        gpu, cpu,
    )
    result = driver.run()
    assert result.failed == ["ep_0002"]
    for ep in ("ep_0001", "ep_0003"):
        assert (out_dir / "lerobot_v3" / ep / "meta" / "info.json").exists()
        assert (out_dir / "qc" / ep / "report.html").exists()
    assert not (out_dir / "lerobot_v3" / "ep_0002").exists()


def _run_once(in_dir, out_dir, ws_dir):
    gpu, cpu = build_actor_pool(num_gpus=2, num_cpu_actors=4)
    driver = PipelineDriver(
        DriverConfig(storage_root=str(in_dir), workspace_root=str(ws_dir),
                     output_root=str(out_dir), serial_baseline=False, adaptive=True),
        _request(in_dir, out_dir),
        gpu,
        cpu,
    )
    return driver.run()


def test_reconcile_crash_running_scene_reruns_on_new_run(tmp_path):
    """Cross-run: SCENE=RUNNING (crash residue) and HAND=DONE at run start -> reconcile resets
    SCENE to PENDING and re-runs it; HAND stays done. The episode finally completes fully.

    First run once so artifacts are in place, then manually set SCENE to RUNNING (simulated
    crash residue); the second run should only re-run SCENE (and its downstream
    fusion/action/publish).
    """
    import json

    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    ws_dir = tmp_path / "ws"
    generate_dataset(in_dir, n_episodes=1, n_frames=10, size=(96, 72), seed=0)
    _run_once(in_dir, out_dir, ws_dir)
    # simulate crash residue: SCENE=RUNNING (interrupted before power loss), HAND stays DONE
    state = json.loads((ws_dir / "ep_0001" / "state.json").read_text())
    state["stages"]["scene"] = "running"
    state["stages"]["publish"] = "pending"
    del state["timings"]["scene"]
    (ws_dir / "ep_0001" / "state.json").write_text(json.dumps(state))
    out_dir2 = tmp_path / "out2"
    result = _run_once(in_dir, out_dir2, ws_dir)
    assert result.failed == []
    final = json.loads((ws_dir / "ep_0001" / "state.json").read_text())
    assert final["stages"]["scene"] == "done"
    assert final["stages"]["hand"] == "done"
    assert final["stages"]["publish"] == "done"
    # SCENE really re-ran: the new run's timeline has a scene start
    scene_starts = [e for e in result.timeline
                    if e["episode_id"] == "ep_0001" and e["stage"] == "scene" and e["event"] == "start"]
    assert len(scene_starts) == 1
    # if reconcile had not reset RUNNING to PENDING, SCENE would not re-run -> would fail
    assert final["stages"]["scene"] == "done"


def test_reconcile_failed_prepare_retries_and_completes(tmp_path):
    """Cross-run: PREPARE=FAILED -> the new run resets it to PENDING and retries successfully,
    and the episode completes fully (re-run resumes from the first unfinished stage)."""
    import json

    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    ws_dir = tmp_path / "ws"
    generate_dataset(in_dir, n_episodes=1, n_frames=8, size=(96, 72), seed=0)
    # hand-write a PREPARE=FAILED state directly to simulate the previous run failing
    (ws_dir / "ep_0001").mkdir(parents=True, exist_ok=True)
    for sub in ("prepared", "artifacts", "media"):
        (ws_dir / "ep_0001" / sub).mkdir(exist_ok=True)
    raw = {
        "episode_id": "ep_0001",
        "state_schema_version": 1,
        "stages": {"prepare": "failed", "scene": "pending", "hand": "pending",
                   "fusion": "pending", "action": "pending", "publish": "pending"},
        "timings": {},
        "error": "previous run failed",
    }
    (ws_dir / "ep_0001" / "state.json").write_text(json.dumps(raw))
    result = _run_once(in_dir, out_dir, ws_dir)
    assert result.failed == []
    final = json.loads((ws_dir / "ep_0001" / "state.json").read_text())
    assert final["stages"]["prepare"] == "done"
    assert final["stages"]["publish"] == "done"
    assert (out_dir / "lerobot_v3" / "ep_0001" / "meta" / "info.json").exists()


def test_static_mode_respects_max_episodes_in_flight(tmp_path):
    """Static pool split: with max_episodes_in_flight=1, at most 1 episode is in flight at any time."""
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    ws_dir = tmp_path / "ws"
    generate_dataset(in_dir, n_episodes=3, n_frames=12, size=(96, 72), seed=5)
    gpu, cpu = build_actor_pool(num_gpus=2, num_cpu_actors=4)
    req = ReconstructionRequest.from_dict(
        {
            "input": {"uri": str(in_dir)},
            "output": {"uri": str(out_dir), "format": "lerobot_v3"},
            "execution": {"max_gpus": 2, "num_cpu_actors": 4,
                          "max_episodes_in_flight": 1, "backend": "test"},
            "stages": {
                "scene_reconstruction": {"model": "fake_scene"},
                "hand_reconstruction": {"model": "fake_hand", "unobserved_gap_s": 0.2},
                "action_understanding": {"model": "fake_vlm"},
                "visualization": {"camera_view": True, "world_view": True},
            },
        }
    )
    driver = PipelineDriver(
        DriverConfig(storage_root=str(in_dir), workspace_root=str(ws_dir),
                     output_root=str(out_dir), serial_baseline=False, adaptive=False),
        req, gpu, cpu,
    )
    result = driver.run()
    assert result.failed == []
    # the timeline is recorded synchronously by the driver thread, so the unclosed-start
    # count gives an upper bound on concurrent episodes
    starts = [e for e in result.timeline if e["event"] == "start"]
    ends = [e for e in result.timeline if e["event"] == "end"]
    open_eps = set()
    max_concurrent = 0
    for e in sorted(starts + ends, key=lambda x: (x["t"], x["event"])):
        if e["event"] == "start":
            open_eps.add(e["episode_id"])
        else:
            open_eps.discard(e["episode_id"])
        max_concurrent = max(max_concurrent, len(open_eps))
    assert max_concurrent <= 1, (
        f"max_episodes_in_flight=1 violated: max_concurrent={max_concurrent}"
    )
