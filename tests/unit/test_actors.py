import json
import time

import numpy as np
import pytest

ray = pytest.importorskip("ray")

from roboray.models import registry
from roboray.ray_runtime.actors import build_actor_pool
from roboray.stages.artifacts import save_artifact
from tests.fakes.models import install_fake_models

pytestmark = pytest.mark.slow


def _write_prepared(ws_root, episode_id="ep_0001", n_frames=2, stereo=False,
                    calib_extra=None):
    """Write prepared frames and calibration to disk (the actor now reads from **its own node**)."""
    from pathlib import Path

    prepared = Path(ws_root) / episode_id / "prepared"
    prepared.mkdir(parents=True, exist_ok=True)
    np.save(prepared / "frames_left.npy", np.zeros((n_frames, 8, 8, 3), np.uint8))
    if stereo:
        np.save(prepared / "frames_right.npy", np.zeros((n_frames, 8, 8, 3), np.uint8))
    calib = {"fps": 30, "has_stereo": bool(stereo)}
    calib.update(calib_extra or {})
    (prepared / "calib.json").write_text(json.dumps(calib))
    return ws_root


@pytest.fixture(autouse=True)
def _ray_cluster(request):
    """Function-level Ray cluster: each test owns its virtual GPU and releases it at teardown.

    - This machine has no GPU: num_gpus is a virtual scheduling resource, so the GPU
      actor scheduling semantics can be validated for real.
    - Must be function-level: GpuModelActor occupies 1 virtual GPU for its lifetime; if the
      cluster were module-shared, a second test creating a new actor pool would be
      permanently PENDING (deadlock) because the resource is never released.
    - worker_process_setup_hook: a Ray worker is a separate process, so the driver-side
      registry registration does not propagate; the test-side hook registers doubles in
      each worker at startup (production code never imports tests/).
    - ``@pytest.mark.fake_fail_on(ep_id)`` configures the double's fail_on into the worker env.
    """
    env_vars = {
        "ROBORAY_FAKE_SCENE_SECONDS": "0.05",
        "ROBORAY_FAKE_HAND_SECONDS": "0.05",
    }
    marker = request.node.get_closest_marker("fake_fail_on")
    if marker:
        env_vars["ROBORAY_FAKE_FAIL_ON"] = marker.args[0]
    if ray.is_initialized():
        ray.shutdown()
    ray.init(
        num_gpus=1,
        num_cpus=4,
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


@pytest.fixture(autouse=True)
def _fakes():
    registry.reset_registry_for_tests()
    install_fake_models()
    yield
    registry.reset_registry_for_tests()


def test_gpu_actor_runs_task_and_reports_timings(tmp_path):
    gpu_actors, _ = build_actor_pool(num_gpus=1, num_cpu_actors=1)
    actor = gpu_actors[0]
    load_seconds = ray.get(actor.load_model.remote("scene", "fake_scene", "cpu"))
    assert load_seconds >= 0.0
    ws_root = str(_write_prepared(tmp_path, n_frames=3))
    res = ray.get(
        actor.run_task.remote(
            {
                "kind": "scene",
                "episode_id": "ep_0001",
                "model_name": "fake_scene",
                "workspace_root": ws_root,
                "out_path": str(tmp_path / "scene.npz"),
            }
        )
    )
    assert res["ok"] is True, res["error"]
    assert res["infer_seconds"] >= 0.0
    assert res["was_load"] is False
    assert res["resolution_key"] == "64x64"  # real frames 8x8 -> quantized to the 64 tier
    assert np.load(res["out_path"])["depth"].shape[0] == 3


def test_repeated_task_with_same_model_does_not_need_reload(tmp_path):
    gpu_actors, _ = build_actor_pool(num_gpus=1, num_cpu_actors=1)
    actor = gpu_actors[0]
    ray.get(actor.load_model.remote("scene", "fake_scene", "cpu"))
    ws_root = str(_write_prepared(tmp_path, n_frames=2))
    payload = {
        "kind": "scene",
        "episode_id": "ep_0001",
        "model_name": "fake_scene",
        "workspace_root": ws_root,
        "out_path": str(tmp_path / "s.npz"),
    }
    ray.get(actor.run_task.remote(payload))
    res2 = ray.get(actor.run_task.remote(payload))
    assert res2["was_load"] is False


def test_hand_payload_inference_fps_reaches_calib(tmp_path):
    """Last mile: payload.inference_fps must be injected into calib via run_task,
    so the adapter can forward it into the script's extra (the same path as scene's
    stereo_max_image_side_px).
    ``_raw_actor()`` bypasses the Ray decorator and instantiates directly - a Ray
    worker is a separate process, so the driver-side monkeypatch does not propagate;
    only calling the method body directly covers this seam (same pure-function
    precedent as this file's ``test_run_cpu_stage_*``). The model is an in-process
    probe double that records calib, so no real registration is needed.
    """
    seen = {}

    class _ProbeModel:
        name = "probe_hand"

        def load(self, device):
            self.loaded_on = device

        def close_session(self):
            pass

        def infer(self, frames, calib, out_path):
            seen["calib"] = dict(calib)
            t = np.asarray(frames["frames"]).shape[0]
            save_artifact(
                out_path,
                hand_present=np.ones((t, 2), bool),
                joints_2d=np.zeros((t, 2, 21, 3), np.float32),
                joints_3d_cam=np.zeros((t, 2, 21, 3), np.float32),
                mano_pose=np.zeros((t, 2, 48), np.float32),
                mano_shape=np.zeros((t, 2, 10), np.float32),
                mano_trans=np.zeros((t, 2, 3), np.float32),
                confidence=np.ones((t, 2), np.float32),
                n_frames=np.int32(t),
                fps=np.int32(30),
            )

    actor = _raw_actor()
    actor.loaded_kind = "hand"
    actor.model_name = "probe_hand"
    actor.model = _ProbeModel()
    ws_root = str(_write_prepared(tmp_path, n_frames=2))
    res = actor.run_task(
        {
            "kind": "hand",
            "episode_id": "ep_0001",
            "model_name": "probe_hand",
            "workspace_root": ws_root,
            "out_path": str(tmp_path / "h.npz"),
            "inference_fps": 15,
        }
    )
    assert res["ok"] is True, res["error"]
    assert seen["calib"]["inference_fps"] == 15
    # converse: without the key, calib must not gain inference_fps out of nowhere (default behavior unchanged)
    seen.clear()
    res2 = actor.run_task(
        {
            "kind": "hand",
            "episode_id": "ep_0001",
            "model_name": "probe_hand",
            "workspace_root": ws_root,
            "out_path": str(tmp_path / "h2.npz"),
        }
    )
    assert res2["ok"] is True, res2["error"]
    assert "inference_fps" not in seen["calib"]
    assert "max_hands" not in seen["calib"]
    # payload.max_hands is injected into calib the same way (the third config-side segment)
    seen.clear()
    res3 = actor.run_task(
        {
            "kind": "hand",
            "episode_id": "ep_0001",
            "model_name": "probe_hand",
            "workspace_root": ws_root,
            "out_path": str(tmp_path / "h3.npz"),
            "max_hands": 6,
        }
    )
    assert res3["ok"] is True, res3["error"]
    assert seen["calib"]["max_hands"] == 6


def test_switching_model_kind_reports_was_load(tmp_path):
    gpu_actors, _ = build_actor_pool(num_gpus=1, num_cpu_actors=1)
    actor = gpu_actors[0]
    ray.get(actor.load_model.remote("scene", "fake_scene", "cpu"))
    ws_root = str(_write_prepared(tmp_path, n_frames=2))
    res = ray.get(
        actor.run_task.remote(
            {
                "kind": "hand",
                "episode_id": "ep_0001",
                "model_name": "fake_hand",
                "workspace_root": ws_root,
                "out_path": str(tmp_path / "h.npz"),
            }
        )
    )
    assert res["was_load"] is True
    assert res["load_seconds"] >= 0.0


@pytest.mark.fake_fail_on("ep_0002")
def test_actor_failure_is_returned_not_raised(tmp_path):
    registry.reset_registry_for_tests()
    install_fake_models(fail_on="ep_0002")
    gpu_actors, _ = build_actor_pool(num_gpus=1, num_cpu_actors=1)
    actor = gpu_actors[0]
    # the double hits fail_on via frames["episode_id"] (read_prepared_frames keeps that key
    # in the frames dict, see `_maybe_fail` in tests/fakes/models.py).
    ws_root = str(_write_prepared(tmp_path, episode_id="ep_0002", n_frames=2))
    res = ray.get(
        actor.run_task.remote(
            {
                "kind": "scene",
                "episode_id": "ep_0002",
                "model_name": "fake_scene",
                "workspace_root": ws_root,
                "out_path": str(tmp_path / "x.npz"),
            }
        )
    )
    assert res["ok"] is False
    assert "ep_0002" in res["error"]


def test_status_reports_loaded_model(tmp_path):
    gpu_actors, _ = build_actor_pool(num_gpus=1, num_cpu_actors=1)
    actor = gpu_actors[0]
    assert ray.get(actor.status.remote())["loaded_kind"] is None
    ray.get(actor.load_model.remote("hand", "fake_hand", "cpu"))
    assert ray.get(actor.status.remote())["loaded_kind"] == "hand"


def test_cpu_actor_registers_builtin_models_for_action_stage():
    """Regression: the CPU actor previously did not register builtin models, so the action

    stage's VLM could never be resolved.
    A Ray worker is a **separate process**, so the driver-side ``load_builtin_models()``
    does not propagate. ``GpuModelActor.load_model`` registers itself, but
    ``CpuStageActor`` did not, so under a real backend the action stage inevitably failed
    with ``UnknownModelError: unregistered semantic model 'openai_vlm'; registered: []``.
    This path was previously only exercised under backend=test (the worker_process_setup_hook
    registers doubles), so it never surfaced until real models ran on the GPU cluster.
    A payload with a deliberately nonexistent ``fn_name`` triggers ``run_task``
    (registration happens before the lookup), then the **read-only** ``registered_models``
    checks the worker-side real registration state.
    """
    from roboray.ray_runtime.actors import CpuStageActor

    actor = CpuStageActor.remote()
    ray.get(actor.run_task.remote({
        "episode_id": "ep_probe", "kind": "action", "fn_name": "__probe__", "kwargs": {},
    }))
    models = ray.get(actor.registered_models.remote())
    assert "openai_vlm" in models["semantic"], models


def test_run_cpu_stage_returns_stage_function_result(monkeypatch):
    """Regression: the CPU actor previously dropped the stage function's return value.

    prepare's width/height/has_stereo are the driver's only source for later routing
    (scene stereo decision, resolution tier); the driver cannot read the workspace
    itself (unreadable under multi-node), so these values must come back via the actor.
    Calling the **module-level pure function body** directly rather than ``.remote()``:
    Ray actors run in separate processes, so the driver-side monkeypatch does not
    propagate and `.remote()` cannot test this.
    """
    from roboray import stages as stages_mod
    from roboray.ray_runtime.actors import run_cpu_stage

    monkeypatch.setitem(
        stages_mod.STAGE_FUNCTIONS, "__probe__", lambda **kw: {"width": 320, "height": 240}
    )
    res = run_cpu_stage(
        {"episode_id": "ep_probe", "kind": "prepare", "fn_name": "__probe__", "kwargs": {}}
    )
    assert res["ok"] is True, res["error"]
    assert res["stage_result"] == {"width": 320, "height": 240}


def test_run_cpu_stage_stage_result_is_none_on_failure(monkeypatch):
    """When the stage function raises, stage_result must be None (no stale value leaks through)."""
    from roboray import stages as stages_mod
    from roboray.ray_runtime.actors import run_cpu_stage

    def _boom(**kw):
        raise ValueError("intentional failure")

    monkeypatch.setitem(stages_mod.STAGE_FUNCTIONS, "__boom__", _boom)
    res = run_cpu_stage(
        {"episode_id": "ep_probe", "kind": "fusion", "fn_name": "__boom__", "kwargs": {}}
    )
    assert res["ok"] is False
    assert res["stage_result"] is None
    assert "ValueError" in res["error"]


def test_build_node_pools_fills_by_gpu_count(monkeypatch):
    """Build slots by GPU count rather than node count: a single node with 2 cards no longer idles its second.

    Assertion semantics: num_gpus=3 -> node A gets 2 + node B gets 1; num_gpus=2 -> exactly 2
    (A fills its 2, B is not built). A's two actors must both be created with affinity
    pinned to node A.
    """
    from roboray.ray_runtime import actors as mod
    from roboray.ray_runtime import nodes as nodes_mod
    from roboray.ray_runtime.nodes import NodeInfo

    # Ray node ids are 56-bit hex (validated by the NodeAffinitySchedulingStrategy
    # constructor), so a realistic id is needed to reach the strategy construction
    # (same node_id convention as the existing stub technique).
    node_a, node_b = "a" * 56, "b" * 56
    nodes = [NodeInfo(node_a, 2, "ha"), NodeInfo(node_b, 1, "hb")]
    # build_node_pools does `from ...nodes import discover_gpu_nodes` inside the function
    # body, so the patch must be applied on the **defining module** (nodes) (same
    # monkeypatch technique as the rest of this file).
    monkeypatch.setattr(nodes_mod, "discover_gpu_nodes", lambda: nodes)
    made = []

    class _Fake:
        @staticmethod
        def options(**kw):
            class _B:
                @staticmethod
                def remote():
                    made.append(kw.get("scheduling_strategy"))
                    return object()
            return _B

    monkeypatch.setattr(mod, "GpuModelActor", _Fake)
    monkeypatch.setattr(mod, "CpuStageActor", _Fake)
    gpu, _cpu = mod.build_node_pools(num_gpus=3, num_cpu_actors=1)
    assert len(gpu) == 3                      # A×2 + B×1
    # A's two actors each own an independent slot: both go through independent remotes,
    # and both affinity strategies are pinned to node A
    assert made[1].node_id == node_a
    # num_gpus=2: A fills its 2, B is not built; both GPU actors' affinity strategies
    # are pinned to node A (if the distribution regressed to 1+1 this would go red -
    # locking down the "fill A's two cards first" distribution).
    # _Fake stands in for both GpuModelActor/CpuStageActor, so `made` records CPU actors too:
    # A fills 2 GPUs + 1 CPU (num_cpu_actors=1 -> per_node_cpu=1), all pinned to A, B untouched.
    made.clear()
    gpu2, _ = mod.build_node_pools(num_gpus=2, num_cpu_actors=1)
    assert len(gpu2) == 2
    assert len(made) == 3  # 2 GPU + 1 CPU, all on A
    assert all(s.node_id == node_a for s in made)  # B not built; all pinned to A


def test_build_node_pools_falls_back_on_gpu_less_machine():
    """With no GPU nodes, fall back to the original single-pool behavior.

    This machine has no GPU, so Ray uses virtual num_gpus - node discovery necessarily
    returns empty. build_node_pools must still return usable actor lists, otherwise every
    e2e test would break.
    """
    from roboray.ray_runtime.actors import build_node_pools

    gpu_actors, cpu_actors = build_node_pools(num_gpus=1, num_cpu_actors=2)
    assert len(gpu_actors) == 1
    assert len(cpu_actors) == 2


def test_gpu_actor_reports_its_node_id():
    """The driver builds the "episode -> node" mapping from the actor's self-reported node_id.

    An actor may be created without affinity (new pool), with placement decided by
    Ray + the autoscaler, so it can only be asked **after** creation.
    """
    from roboray.ray_runtime.actors import GpuModelActor

    actor = GpuModelActor.remote()
    node_id = ray.get(actor.node_id.remote())
    assert isinstance(node_id, str) and node_id


def test_cpu_actor_reports_its_node_id():
    from roboray.ray_runtime.actors import CpuStageActor

    actor = CpuStageActor.remote()
    assert isinstance(ray.get(actor.node_id.remote()), str)


# --------------------------------------------------------------------------
# resident switch / unload / idle_seconds / touch (method-level direct tests)
#
# GpuModelActor is a @ray.remote class - retrieve the original class via
# ``__ray_actor_class__`` and instantiate with ``__new__``, bypassing the decorator to
# test the method bodies directly (same pure-function precedent as test_ray_init.py,
# no cluster needed); "working through a real actor" is covered by the e2e resident test.

def _raw_actor():
    from roboray.ray_runtime.actors import GpuModelActor

    actor = GpuModelActor.__ray_actor_class__.__new__(GpuModelActor.__ray_actor_class__)
    actor.loaded_kind = None
    actor.model = None
    actor.model_name = None
    actor.mode = "resident"
    actor._last_used = None
    return actor


def test_unload_clears_model_and_returns_kind():
    """unload clears loaded_kind/model/model_name and returns the unloaded kind;
    **`_last_used` is not reset** (the idle clock counts from the last use - otherwise
    the 30s unload would permanently push back the 240s exit layer and nodes would never
    return to the cluster)."""

    actor = _raw_actor()
    actor.loaded_kind = "scene"
    actor.model = object()  # a double with no close_session: unload must safely skip session cleanup
    actor.model_name = "fake_scene"
    t = time.monotonic()
    actor._last_used = t
    assert actor.unload() == {"unloaded": "scene"}
    assert actor.loaded_kind is None
    assert actor.model is None
    assert actor.model_name is None
    assert actor._last_used == t  # clock preserved: the exit layer judges idle timeout from it
    # when nothing is loaded: unload is idempotent, returns {"unloaded": None}
    assert _raw_actor().unload() == {"unloaded": None}


def test_unload_calls_close_session_when_present():
    """unload must drive the adapter's close_session (graceful shutdown of the resident session)."""
    class _M:
        def __init__(self):
            self.closed = False

        def close_session(self):
            self.closed = True

    actor = _raw_actor()
    m = _M()
    actor.loaded_kind = "scene"
    actor.model = m
    actor.model_name = "x"
    actor.unload()
    assert m.closed is True


def test_load_model_resident_mode_fake_without_open_session_is_safe():
    """A fake adapter (no open_session) must not crash load_model in mode=resident.

    The hasattr guard is what keeps fakes zero-change: an in-process double has no
    open_session/close_session, and resident mode must not crash by calling them.
    """
    actor = _raw_actor()
    secs = actor.load_model("scene", "fake_scene", "cpu")
    assert secs >= 0.0
    assert actor.loaded_kind == "scene"
    assert actor.model_name == "fake_scene"
    assert not hasattr(actor.model, "open_session")  # the fake indeed lacks this interface


def test_idle_seconds_and_touch():
    """idle_seconds: never touched -> 0.0; after touch it counts from then; manually winding the
    clock back makes it measurable."""

    actor = _raw_actor()
    assert actor.idle_seconds() == 0.0  # never ran a task
    actor.touch()
    assert 0.0 <= actor.idle_seconds() < 0.5
    actor._last_used = time.monotonic() - 3.0
    assert 2.5 < actor.idle_seconds() < 3.5


def test_switching_kind_closes_previous_resident_session(monkeypatch):
    """A scene<->hand switch must close_session first, otherwise every switch leaks one set of weights."""
    closed = []

    class _M:
        def load(self, device):
            pass

        def open_session(self):
            pass

        def close_session(self):
            closed.append(True)

    monkeypatch.setattr(registry, "get_scene_model_class", lambda name: _M)
    monkeypatch.setattr(registry, "get_hand_model_class", lambda name: _M)
    actor = _raw_actor()
    actor.load_model("scene", "m1", "cpu")
    assert closed == []
    actor.load_model("hand", "m2", "cpu")
    assert closed == [True]
