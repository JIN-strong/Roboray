"""Core acceptance: resident mode = weights loaded only once (end to end).

No torch dependency. Uses the **echo worker subprocess + real subprocess bridge
adapter** (EchoSceneModel, isomorphic to VipeSceneModel, inheriting _SceneBase for
open_session/close_session and the session-branch infer) running a backend=test e2e:
2 episodes share the same GPU actor -> the same ModelEnvSession -> the echo worker's
build_model is called only once.
Evidence chain (**asserted on the artifact side, outside the response side**): when
``extra={"contract": "scene"}`` the echo worker writes the build evidence as
``builds`` (cumulative build count) and ``built_flag`` (whether this call rebuilt)
arrays into scene.npz (see infer_once in tests/fakes/serve_worker.py). The e2e reads
the workspace scene.npz directly after the driver finishes - no actor introspection
(ray.get cannot see actor-internal state).
**The ``ROBORAY_TEST_ECHO_SCENE`` gate relies on driver->worker env inheritance**:
the monkeypatch only sets the env in the driver process, and Ray workers are forked
after the driver starts, inheriting ``os.environ`` - valid in local mode. On a real
cluster workers do not inherit the driver env (must be sent via runtime_env) -
this gate is **test-only**; production code (outside worker_setup) never reads it.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from roboray.api.config import ReconstructionRequest
from roboray.api.reconstruction import EgoReconstruction

pytestmark = pytest.mark.slow


def _make_input(root: Path, n_episodes: int = 2, n_frames: int = 4) -> Path:
    """Build a minimal input (same construction style as the isolation test).

    Monocular (stereo=False): explicitly setting ``scene_reconstruction.model="echo_scene"``
    bypasses auto stereo routing, ensuring the scene stage always goes through the echo adapter.
    """
    import cv2

    root.mkdir(parents=True, exist_ok=True)
    for e in range(n_episodes):
        ep = root / f"ep_{e + 1:04d}"
        ep.mkdir(parents=True, exist_ok=True)
        w, h = 64, 48
        writer = cv2.VideoWriter(
            str(ep / "rgb_left.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h)
        )
        for i in range(n_frames):
            writer.write(np.full((h, w, 3), 40 + i, np.uint8))
        writer.release()
        (ep / "calibration.json").write_text(json.dumps({
            "K": [[51.2, 0.0, w / 2], [0.0, 51.2, h / 2], [0.0, 0.0, 1.0]],
            "dist": [0.0] * 5, "T_left_to_right": np.eye(4).tolist(),
            "fps": 30.0, "width": w, "height": h,
        }))
        (ep / "meta.json").write_text(json.dumps(
            {"episode_id": f"ep_{e + 1:04d}", "device": "test", "n_frames": n_frames}
        ))
    return root


@pytest.fixture
def _echo_env(monkeypatch):
    """Worker-side registration of the echo scene adapter + resident already provided by config."""
    monkeypatch.setenv("ROBORAY_TEST_ECHO_SCENE", "1")


def test_resident_two_episodes_build_model_once(tmp_path, monkeypatch, _echo_env):
    storage_root = _make_input(tmp_path / "input")
    output_root = tmp_path / "out"
    workspace_root = tmp_path / "run"
    # minimal dataset (n_frames=4) for speed; monocular input ensures the scene model hits echo_scene explicitly.
    request = ReconstructionRequest.from_dict({
        "input": {"uri": str(storage_root)},
        "output": {"uri": str(output_root), "format": "lerobot_v3"},
        "execution": {"max_gpus": 1, "num_cpu_actors": 2,
                      "max_episodes_in_flight": 2, "backend": "test",
                      "model_worker": "resident"},
        "stages": {
            "scene_reconstruction": {"model": "echo_scene"},
            "hand_reconstruction": {"model": "fake_hand"},
            "action_understanding": {"model": "fake_vlm"},
            "visualization": {"camera_view": False, "world_view": False},
        },
        "models": {
            "lerobot_python": "python",
            "lerobot_script": "tests/fakes/modelenv_scripts/fake_lerobot_export.py",
        },
    })
    result = EgoReconstruction(request, run_root=str(workspace_root)).run()
    assert result.failed == [], f"e2e failed: {result.failed}"

    # evidence chain (artifact side): each of the 2 episodes runs scene once, sharing the same GPU
    # actor -> the same ModelEnvSession -> the echo worker's build_model is called only once in total
    # (builds==1); exactly **one** of the two has built_flag=True (first inference rebuilds),
    # the other False (reused within the same session, not rebuilt).
    flags = {}
    for ep in ("ep_0001", "ep_0002"):
        scene = workspace_root / ep / "artifacts" / "scene.npz"
        assert scene.exists(), f"missing scene artifact: {scene}"
        with np.load(scene, allow_pickle=False) as z:
            assert int(z["builds"]) == 1, (
                f"{ep}: echo worker cumulative build count must be exactly 1 (resident loads once), "
                f"got {int(z['builds'])}"
            )
            assert str(z["depth_source"]) == "monocular"
            assert z["depth"].shape == (4, 48, 64)
            flags[ep] = bool(z["built_flag"])
    # only one of the two inferences rebuilds (order decided by scheduling: ep_0002's prepare
    # finishes first, so its scene runs first with built_flag=True). Assert "exactly one True /
    # exactly one False" - order-independent, and it simultaneously proves "build happened only
    # once" and "the second time was not a rebuild".
    assert sorted(flags.values()) == [False, True], (
        f"the two scene inferences should rebuild exactly once (exactly one built_flag True): {flags}"
    )
