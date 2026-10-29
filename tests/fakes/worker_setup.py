"""Ray worker-side fake registration hook (test-only).

A Ray worker is a separate process, so the driver-side ``install_fake_models()``
registration does not propagate. The test fixture uses
``runtime_env={"worker_process_setup_hook": ...}`` so every worker runs
``install_for_ray()`` at startup, reading the fake parameters from env vars.
"""

import os


def install_for_ray() -> None:
    from roboray.models import registry

    from tests.fakes.models import install_fake_models

    install_fake_models(
        scene_seconds=float(os.environ.get("ROBORAY_FAKE_SCENE_SECONDS", "0.05")),
        hand_seconds=float(os.environ.get("ROBORAY_FAKE_HAND_SECONDS", "0.05")),
        fail_on=os.environ.get("ROBORAY_FAKE_FAIL_ON") or None,
    )
    # resident e2e: the worker registers the echo scene adapter (real subprocess
    # bridge) on demand. The gate is passed via runtime_env env_vars; the
    # EchoSceneModel import path must be resolvable in the worker - runtime_env
    # working_dir already contains tests/.
    if os.environ.get("ROBORAY_TEST_ECHO_SCENE") == "1":
        from tests.fakes.echo_scene_adapter import EchoSceneModel

        registry.register_scene_model("echo_scene", lambda: EchoSceneModel())
