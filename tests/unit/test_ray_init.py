"""Contract for `ray.init` kwargs (`roboray.api.reconstruction.ray_init_kwargs`).

Regression source: running on a KubeRay cluster with `--ray-address auto` **crashes** -

    ValueError: When connecting to an existing cluster, num_cpus and num_gpus
    must not be provided.

The reason is that connecting to an existing cluster still unconditionally passes
`num_cpus=os.cpu_count()`. This path previously had no test coverage (the existing
e2e tests all start a local cluster), hence these two mutual-exclusion contracts.
"""

from roboray.api.reconstruction import ray_init_kwargs


def test_connecting_to_existing_cluster_omits_cpu_and_gpu():
    """Connecting to an existing cluster: only pass address, never num_cpus / num_gpus."""
    kw = ray_init_kwargs(
        ray_address="auto", backend="test", max_gpus=1, runtime_env={"env_vars": {}}
    )
    assert kw["address"] == "auto"
    assert "num_cpus" not in kw
    assert "num_gpus" not in kw
    assert kw["include_dashboard"] is False
    assert kw["log_to_driver"] is False


def test_local_cluster_sets_resources_and_no_address():
    """Starting a local cluster: pass the resource scale, no address."""
    kw = ray_init_kwargs(
        ray_address=None, backend="test", max_gpus=1, runtime_env={"env_vars": {}}, num_cpus=7
    )
    assert "address" not in kw
    assert kw["num_cpus"] == 7
    assert kw["num_gpus"] == 1


def test_ray_address_local_must_also_carry_resources():
    """``RAY_ADDRESS=local`` must also carry the resource scale - otherwise the local cluster has GPU=0,
    the GPU actor can never be scheduled, and the driver's ``ray.wait`` hangs forever.

    Regression source: when the acceptance test ran with ``RAY_ADDRESS=local``,
    the whole pipeline hung in ``ray.wait`` for the full 600s. Testing showed
    ``ray.init(address="local", num_cpus=.., num_gpus=..)`` is legal and num_gpus
    merges into the new cluster's resource table; without it the cluster has GPU=0,
    so ``GpuModelActor(num_gpus=1)`` never gets scheduled.
    """
    kw = ray_init_kwargs(
        ray_address="local", backend="test", max_gpus=1, runtime_env={"env_vars": {}}, num_cpus=7
    )
    assert kw["address"] == "local"
    assert kw["num_cpus"] == 7
    assert kw["num_gpus"] == 1


def test_ray_address_local_clamps_negative_max_gpus_to_zero():
    """``RAY_ADDRESS=local`` clamps a negative max_gpus to 0 too (same as a local cluster)."""
    kw = ray_init_kwargs(
        ray_address="local", backend="real", max_gpus=-1, runtime_env={}, num_cpus=2
    )
    assert kw["address"] == "local"
    assert kw["num_cpus"] == 2
    assert kw["num_gpus"] == 0


def test_local_cluster_negative_max_gpus_clamped_to_zero():
    """A negative max_gpus is clamped to 0 (a GPU-less machine should not request GPU resources)."""
    kw = ray_init_kwargs(
        ray_address=None, backend="real", max_gpus=-1, runtime_env={}, num_cpus=2
    )
    assert kw["num_gpus"] == 0


def test_runtime_env_is_passed_through_in_both_paths():
    """runtime_env (including the model env's env_vars) must pass through on both paths."""
    re = {"env_vars": {"ROBORAY_FFS_PYTHON": "/x/python"}}
    local = ray_init_kwargs(ray_address=None, backend="real", max_gpus=1, runtime_env=re)
    remote = ray_init_kwargs(ray_address="auto", backend="real", max_gpus=1, runtime_env=re)
    assert local["runtime_env"] is re
    assert remote["runtime_env"] is re


def test_vlm_api_key_forwarded_to_workers_by_configured_env_name():
    """VLM auth must be sent to workers via runtime_env: workers do not inherit the driver environment."""
    from roboray.api.config import ReconstructionRequest
    from roboray.api.reconstruction import _vlm_key_env_vars

    req = ReconstructionRequest.from_dict({
        "input": {"uri": "/in"},
        "output": {"uri": "/out"},
        "stages": {"action_understanding": {"model": "openai_vlm",
                                            "vlm_api_key_env": "VLM_API_KEY"}},
    })
    assert _vlm_key_env_vars(req, {"VLM_API_KEY": "sk-x"}) == {"VLM_API_KEY": "sk-x"}
    assert _vlm_key_env_vars(req, {}) == {}  # not set -> not sent (dummy/error contract)
