"""Public API: EgoReconstruction(request).run()."""

import os
from pathlib import Path

from .config import ReconstructionRequest

__all__ = ["EgoReconstruction"]


def _vlm_key_env_vars(request: ReconstructionRequest, environ=None) -> dict:
    """Forward the VLM credential to workers under the configured env name; skip when
    unset.

    Workers do not inherit the driver's process environment (multi-node / separate
    processes), yet ``openai_vlm`` reads ``vlm_api_key_env`` inside the worker.
    Without forwarding, the action stage always fails with "credentials missing".

    Forwarding condition: an env name is configured and the driver env has a value;
    it is not conditional on action model=openai_vlm (model selection happens on the
    worker side), and the value does not go into the manifest.
    """
    env = os.environ if environ is None else environ
    name = request.stages.action_understanding.vlm_api_key_env
    value = env.get(name, "")
    return {name: value} if name and value else {}


def _resolve(path: str) -> str:
    """Resolve a relative path to absolute (relative to the driver process CWD).

    A Ray worker's CWD is a packaged copy of working_dir (files in .gitignore are
    not in it), so a relative path would resolve to the wrong place inside the
    worker; resolving to an absolute path on the driver side avoids that.
    """
    return str(Path(path).expanduser().resolve())


def ray_init_kwargs(
    *,
    ray_address: str | None,
    backend: str,
    max_gpus: int,
    runtime_env: dict,
    num_cpus: int | None = None,
) -> dict:
    """Build the ``ray.init`` keyword arguments. Extracted so the parameter contract
    can be unit-tested directly.

    Contract (three paths, learned the hard way):
    - **Connecting to an existing cluster** (``--ray-address auto``, e.g. a KubeRay
      head): only ``address`` may be passed; adding ``num_cpus``/``num_gpus`` is
      rejected by Ray outright:
      ``ValueError: When connecting to an existing cluster, num_cpus and num_gpus
      must not be provided.`` Resource sizing is decided on the cluster side.
    - **``RAY_ADDRESS=local`` (start a new local cluster but through the existing
      cluster's ``address`` contract)**: ``address="local"`` has "connect to
      cluster" semantics — does Ray reject ``num_cpus``/``num_gpus`` alongside
      ``address``? **No** — in practice ``ray.init(address="local", num_cpus=...,
      num_gpus=...)`` is legal, and ``num_gpus`` is merged into the new cluster's
      resource table. So local must also carry the resource sizes: otherwise the
      local cluster has GPU=0, ``GpuModelActor(num_gpus=1)`` can never be
      scheduled, and ``ray.wait`` hangs forever (a deadlock caught by the e2e
      acceptance test).
    - **Starting a local cluster** (no ``RAY_ADDRESS``): pass ``num_cpus``/
      ``num_gpus``, no ``address``.
    """
    common = {"include_dashboard": False, "log_to_driver": False, "runtime_env": runtime_env}
    if ray_address == "local":
        # Start a local cluster with explicit resource sizes. **num_gpus is
        # required**: otherwise the cluster has GPU=0, GPU actors never get
        # scheduled, and the driver's ray.wait hangs forever.
        return {
            "address": ray_address,
            "num_cpus": os.cpu_count() if num_cpus is None else num_cpus,
            "num_gpus": max(0, max_gpus),
            **common,
        }
    if ray_address:
        return {"address": ray_address, **common}
    return {
        "num_cpus": os.cpu_count() if num_cpus is None else num_cpus,
        "num_gpus": max(0, max_gpus),
        **common,
    }


class EgoReconstruction:
    def __init__(
        self,
        request: ReconstructionRequest,
        *,
        run_root: str | None = None,
        serial_baseline: bool = False,
        adaptive: bool = True,
        ray_address: str | None = None,
    ):
        self.request = request
        self.serial_baseline = serial_baseline
        self.adaptive = adaptive
        self.ray_address = ray_address or os.environ.get("RAY_ADDRESS")
        base = Path(run_root or os.environ.get("ROBORAY_RUN_ROOT", "./run"))
        # run_root is the workspace root: checkpoint state is written to
        # run_root/<episode_id>/state.json (consistent with the test_public_api.py
        # assertions). Resolved to an absolute path on the driver side so relative
        # paths do not break inside workers (packaged copies of working_dir).
        self.run_root = Path(_resolve(str(base)))
        self.workspace_root = str(self.run_root)

    def run(self):
        import ray

        from roboray.models import registry
        from roboray.models.modelenv import resolve_env_request
        from roboray.ray_runtime.actors import build_actor_pool
        from roboray.runtime.driver import DriverConfig, PipelineDriver
        from roboray.manifest import write_manifest

        if self.request.execution.backend == "real":
            registry.load_builtin_models()

        # Relative paths are resolved to absolute on the driver side (a worker's CWD
        # is a packaged copy of working_dir, so relative paths fail inside workers).
        storage_root = _resolve(self.request.input.uri)
        output_root = _resolve(self.request.output.uri)

        # Model-env interpreter/weights paths flattened into env_vars and shipped to
        # workers via runtime_env.
        env_vars = {
            "ROBORAY_FAKE_SCENE_SECONDS": os.environ.get("ROBORAY_FAKE_SCENE_SECONDS", "0.05"),
            "ROBORAY_FAKE_HAND_SECONDS": os.environ.get("ROBORAY_FAKE_HAND_SECONDS", "0.05"),
        }
        env_vars.update(resolve_env_request(self.request.models))
        # VLM credential likewise: workers do not inherit the driver's process
        # environment, forward it under the configured env name.
        env_vars.update(_vlm_key_env_vars(self.request))
        # Model worker-process mode: the worker's GpuModelActor.__init__ reads it to
        # decide whether to open_session (resident) on the adapter. Under multi-node
        # it must be shipped via runtime_env — workers do not inherit the driver's
        # process environment.
        env_vars["ROBORAY_MODEL_WORKER"] = self.request.execution.model_worker

        if not ray.is_initialized():
            if self.request.execution.backend == "test":
                # Local fake-model end-to-end: Ray workers are separate processes, so
                # driver-side registration does not propagate; worker_process_setup_hook
                # registers the stand-ins inside the worker (same as the test fixture).
                runtime_env = {
                    "working_dir": ".",
                    "worker_process_setup_hook": "tests.fakes.worker_setup.install_for_ray",
                    "env_vars": env_vars,
                }
            else:
                # Real models: workers need the model env's python/weights paths.
                runtime_env = {"env_vars": env_vars}
            ray.init(**ray_init_kwargs(
                ray_address=self.ray_address,
                backend=self.request.execution.backend,
                max_gpus=self.request.execution.max_gpus,
                runtime_env=runtime_env,
            ))

        gpu_actors, cpu_actors = build_actor_pool(
            num_gpus=max(0, self.request.execution.max_gpus),
            num_cpu_actors=self.request.execution.num_cpu_actors,
        )
        driver = PipelineDriver(
            DriverConfig(
                storage_root=storage_root,
                workspace_root=self.workspace_root,
                output_root=output_root,
                serial_baseline=self.serial_baseline,
                adaptive=self.adaptive,
            ),
            self.request,
            gpu_actors,
            cpu_actors,
        )
        result = driver.run()
        result.manifest_path = str(
            write_manifest(
                output_root,
                self.request,
                result.per_episode,
                result.scheduler_snapshots,
                result.total_seconds,
                result.timeline,
            )
        )
        return result
