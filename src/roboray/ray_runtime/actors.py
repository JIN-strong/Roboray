"""Ray actors: the GPU model actor (role-switchable) and the CPU stage actor.

Key design: failures are not raised but returned as ``ok=False``, so the driver
can mark that episode FAILED and keep advancing the remaining episodes.
"""

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import ray

from roboray.models import registry
from roboray.stages.scene import resolution_key

__all__ = ["GpuModelActor", "CpuStageActor", "read_prepared_frames", "build_actor_pool",
           "build_node_pools", "run_cpu_stage"]


def read_prepared_frames(workspace_root: str, episode_id: str) -> tuple:
    """Read the prepared frames and calibration **on the actor's own node**, returning ``(frames, calib)``.

    ``frames`` is the dict expected by the model adapter
    ``infer(frames, calib, out_path)``: ``frames_right`` is only included when
    the calibration marks ``has_stereo`` (not for monocular). This semantics was
    originally implemented by ``build_infer_frames`` lifting from
    ``calib["frames_right"]``; it is now read directly from disk, semantics
    unchanged.
    """
    prepared = Path(workspace_root) / episode_id / "prepared"
    calib = json.loads((prepared / "calib.json").read_text(encoding="utf-8"))
    frames = {"frames": np.load(prepared / "frames_left.npy"),
              "episode_id": episode_id}
    if calib.get("has_stereo"):
        frames["frames_right"] = np.load(prepared / "frames_right.npy")
    return frames, calib


def run_cpu_stage(payload: dict) -> dict:
    """The CPU stage executor (a module-level pure function so unit tests can call it directly).

    Extracted as a module-level function rather than an actor method because Ray
    actors run in separate processes: a test monkeypatch of ``STAGE_FUNCTIONS``
    does not propagate to the driver side; only calling the function body
    directly covers this path.
    """
    from roboray.stages import STAGE_FUNCTIONS

    # A Ray worker is a **separate process**, so the driver-side registry
    # registration does not propagate. The action stage (VLM) runs on a CPU
    # actor; without registering here it would fail with
    # `UnknownModelError: unregistered semantic model 'openai_vlm'; registered: []`.
    # Registration is idempotent (guarded internally by `if name not in ...`).
    registry.load_builtin_models()

    result: dict[str, Any] = {
        "episode_id": payload.get("episode_id"),
        "kind": payload.get("kind"),
        "seconds": 0.0,
        "ok": True,
        "error": None,
        "stage_result": None,
    }
    t0 = time.perf_counter()
    try:
        fn = STAGE_FUNCTIONS[payload["fn_name"]]
        out = fn(**payload["kwargs"])
        result["seconds"] = time.perf_counter() - t0
        # The stage function's return value must be passed back: prepare's
        # width/height/has_stereo are the driver's only source for later routing
        # (the driver cannot read the workspace itself - unreadable under
        # multi-node).
        result["stage_result"] = out if isinstance(out, dict) else None
    except Exception as exc:
        result["ok"] = False
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


@ray.remote(num_gpus=1)
class GpuModelActor:
    def __init__(self):
        self.loaded_kind: str | None = None
        self.model = None
        self.model_name: str | None = None
        # Model worker-process mode (the worker reads ROBORAY_MODEL_WORKER from
        # runtime_env): "resident" = the adapter holds a long-lived
        # ModelEnvSession (weights loaded once); "oneshot" = a subprocess per
        # inference (byte-for-byte equivalent to the pre-change behavior).
        self.mode = os.environ.get("ROBORAY_MODEL_WORKER", "resident")
        self._last_used: float | None = None  # time.monotonic timestamp (refreshed on touch)

    def load_model(self, kind: str, model_name: str, device: str = "cuda") -> float:
        # A Ray worker is a separate process: the driver-side registration does
        # not propagate here. Idempotently self-register the production models
        # (factories are lazily imported, repeated calls have no side effects)
        # so production paths (vipe/hamer, etc.) can be resolved by name inside
        # the worker.
        registry.load_builtin_models()
        factory = (
            registry.get_scene_model_class(model_name)
            if kind == "scene"
            else registry.get_hand_model_class(model_name)
        )
        # A role switch (scene<->hand) replaces self.model: close the old
        # adapter's resident session first, otherwise every scene/hand round
        # trip leaks one set of weights.
        if self.model is not None and hasattr(self.model, "close_session"):
            self.model.close_session()
        t0 = time.perf_counter()
        model = factory()
        model.load(device)
        self.model = model
        self.model_name = model_name
        self.loaded_kind = kind
        # resident mode: loading also starts the long-lived session (weights are
        # built lazily on the first infer). In-process fake adapters have no
        # open_session; the hasattr guard keeps them zero-change. Exceptions
        # propagate up - "return failures as data" is run_task's job, not
        # load_model's.
        if self.mode == "resident" and hasattr(self.model, "open_session"):
            self.model.open_session()
        return time.perf_counter() - t0

    def unload(self) -> dict:
        """Unload the current model: close the resident session and clear state, returning the unloaded model kind.

        **Does not reset ``_last_used``**: the idle clock counts from the last
        successful use; if unload zeroed it, the exit layer (240s) would be
        permanently pushed back by the 30s unload, so nodes would never return
        to the cluster. ``loaded=False`` is enough to prevent duplicate unloads.
        """
        if self.model is not None and hasattr(self.model, "close_session"):
            self.model.close_session()
        unloaded = self.loaded_kind
        self.loaded_kind = None
        self.model = None
        self.model_name = None
        return {"unloaded": unloaded}

    def idle_seconds(self) -> float:
        """Seconds since the last successful inference (time.monotonic); never ran -> 0.0."""
        if self._last_used is None:
            return 0.0
        return time.monotonic() - self._last_used

    def touch(self) -> None:
        """Refresh the last-used time (called at the end of the run_task success path)."""
        self._last_used = time.monotonic()

    def __ray_terminate__(self) -> None:
        """Runs when Ray terminates the actor normally: gracefully close the resident session.

        Note ``ray.kill`` is SIGKILL and **does not** run this method - which is
        why the three-level idle policy has the driver unload before kill.
        """
        try:
            self.unload()
        except Exception:
            pass  # the terminate path must not raise - best-effort cleanup

    def status(self) -> dict:
        return {"loaded_kind": self.loaded_kind, "model_name": self.model_name}

    def node_id(self) -> str:
        """The Ray node this actor is on.

        The driver uses it to build the "episode -> node" mapping: a new pool's
        GPU actor is created **without affinity** (placement decided by Ray +
        the autoscaler), so the actor can only be asked after creation.
        """
        return ray.get_runtime_context().get_node_id()

    def run_task(self, payload: dict) -> dict:
        kind = payload["kind"]
        model_name = payload["model_name"]
        result: dict[str, Any] = {
            "episode_id": payload["episode_id"],
            "kind": kind,
            "out_path": payload.get("out_path"),
            "resolution_key": payload.get("resolution_key", "640x480"),
            "infer_seconds": 0.0,
            "load_seconds": 0.0,
            "was_load": False,
            "ok": True,
            "error": None,
        }
        try:
            if self.loaded_kind != kind or self.model_name != model_name:
                result["load_seconds"] = self.load_model(kind, model_name)
                result["was_load"] = True
            # Frames are read on the actor's own node (the payload only carries
            # paths): under multi-node the driver cannot read node-local
            # workspaces, and shipping arrays across nodes through the object
            # store is exactly what we want to avoid.
            frames, calib = read_prepared_frames(payload["workspace_root"], payload["episode_id"])
            if kind == "scene":
                calib["stereo_max_image_side_px"] = int(payload.get("stereo_max_image_side_px", 0))
            if kind == "hand":
                # inference_fps flows calib -> adapter extra -> script (the same
                # path as scene's stereo_max_image_side_px). Default 30 keeps the
                # stride=1 behavior byte-for-byte unchanged.
                if payload.get("inference_fps"):
                    calib["inference_fps"] = int(payload["inference_fps"])
                # max_hands is injected the same way (payload.get truthiness
                # check; a default of 4 passes through too - same semantics as
                # the inference_fps default of 30, script-side behavior unchanged).
                if payload.get("max_hands"):
                    calib["max_hands"] = int(payload["max_hands"])
            # The resolution tier is decided by the **real frames** and reported
            # back to the driver for the cost model.
            result["resolution_key"] = resolution_key(frames["frames"].shape[2],
                                                      frames["frames"].shape[1])
            t0 = time.perf_counter()
            self.model.infer(frames, calib, Path(payload["out_path"]))
            result["infer_seconds"] = time.perf_counter() - t0
            # Refresh the idle clock at the end of the success path (failures
            # do not touch: the idle policy counts successful activity).
            self.touch()
        except Exception as exc:  # return failures as data; the driver decides next steps
            result["ok"] = False
            result["error"] = f"{type(exc).__name__}: {exc}"
        return result


@ray.remote(num_cpus=0.5)
class CpuStageActor:
    """Executes the CPU-heavy stages (prepare/fusion/action/publish).

    ``fn_name`` is resolved to a stage function on the driver side, so the actor
    never holds a mutable closure.
    """

    def run_task(self, payload: dict) -> dict:
        return run_cpu_stage(payload)

    def registered_models(self) -> dict:
        """Diagnostics: model names registered **in this actor's process** (read-only, does not trigger registration).

        A Ray worker is a separate process, so the driver-side registration does
        not propagate; when debugging "stage cannot find the model in the
        worker", asking the worker directly is the simplest path. Because it
        does not trigger registration, its return value also verifies that
        ``run_task`` really did register.
        """
        return {
            "scene": registry.known_scene_models(),
            "hand": registry.known_hand_models(),
            "semantic": registry.known_semantic_models(),
        }

    def node_id(self) -> str:
        """The Ray node this actor is on.

        Same as GpuModelActor, used to verify that a CPU actor and its
        episode's GPU actor are on the same node.
        """
        return ray.get_runtime_context().get_node_id()


def build_node_pools(num_gpus: int, num_cpu_actors: int) -> tuple:
    """Create the GPU/CPU actor pools, returning ``(gpu_actors, cpu_actors)``.

    **The return structure is unchanged** (existing callers and tests need no
    changes). When GPU nodes exist, GPU slots are built by **GPU count** rather
    than node count: a multi-GPU single node no longer idles its second card -
    with ``num_gpus=3`` and nodes A (2 cards) / B (1 card), A gets 2 and B gets
    1. Every GPU actor and ``num_cpu_actors`` CPU actors are pinned to that node
    with ``NodeAffinitySchedulingStrategy(soft=False)`` - **affinity is enforced
    by actor placement**, so calling a method on an actor always executes on its
    node, with no per-task strategy needed.

    When there are no GPU nodes (local test machine / CPU-only cluster) this
    falls back to the original behavior: build pools flat by count, no affinity.
    This is the **norm, not an exception** - all e2e tests run in this
    environment.
    """
    from roboray.ray_runtime.nodes import discover_gpu_nodes

    nodes = discover_gpu_nodes()
    if not nodes:
        gpu_actors = [GpuModelActor.remote() for _ in range(max(0, num_gpus))]
        cpu_actors = [CpuStageActor.remote() for _ in range(max(1, num_cpu_actors))]
        return gpu_actors, cpu_actors

    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    remaining = int(num_gpus)
    per_node_cpu = max(1, int(num_cpu_actors))
    gpu_actors, cpu_actors = [], []
    for node in nodes:
        if remaining <= 0:
            break
        k = min(int(node.gpu_count), remaining)  # fill slots by GPU count
        strategy = NodeAffinitySchedulingStrategy(node_id=node.node_id, soft=False)
        for _ in range(k):
            gpu_actors.append(
                GpuModelActor.options(scheduling_strategy=strategy).remote())
        cpu_actors.extend(
            CpuStageActor.options(scheduling_strategy=strategy).remote()
            for _ in range(per_node_cpu)
        )
        remaining -= k
    return gpu_actors, cpu_actors


# Compatibility alias: existing tests and callers still reference the old name.
build_actor_pool = build_node_pools
