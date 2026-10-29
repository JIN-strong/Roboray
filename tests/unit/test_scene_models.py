import json
import sys
from pathlib import Path

import numpy as np
import pytest

from roboray.models import registry
from roboray.models.modelenv import ModelEnvError
from roboray.models.scene_models import (_SceneBase, FastFoundationStereoModel,
                                          VipeSceneModel)

FAKES = Path(__file__).resolve().parents[1] / "fakes" / "modelenv_scripts"


@pytest.fixture(autouse=True)
def _registry():
    registry.reset_registry_for_tests()
    registry.load_builtin_models()
    yield
    registry.reset_registry_for_tests()


def _calib(tmp_path, fake_script_name, extra_env=None):
    calib = {
        "fps": 30.0,
        "has_stereo": False,
        "T_left_to_right": np.eye(4).tolist(),
    }
    env = {"ROBORAY_VIPE_PYTHON": sys.executable}
    env.update(extra_env or {})
    return calib, env


def test_vipe_adapter_runs_fake_script(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBORAY_VIPE_PYTHON", sys.executable)
    model = VipeSceneModel(script=str(FAKES / "fake_vipe.py"))
    model.load("cpu")
    out = tmp_path / "scene.npz"
    model.infer(
        {"frames": np.zeros((3, 8, 8, 3), np.uint8), "episode_id": "ep_0001"},
        {"fps": 30.0, "has_stereo": False, "T_left_to_right": np.eye(4).tolist()},
        out,
    )
    z = np.load(out, allow_pickle=False)
    assert z["camera_poses"].shape == (3, 4, 4)
    assert str(z["depth_source"]) == "monocular"
    assert int(z["__schema_version__"]) == 1


def test_vipe_adapter_missing_env_python_gives_readable_error(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBORAY_VIPE_PYTHON", "/nonexistent/python")
    model = VipeSceneModel(script=str(FAKES / "fake_vipe.py"))
    with pytest.raises(ModelEnvError) as exc:
        model.load("cpu")
    assert "ROBORAY_VIPE_PYTHON" in str(exc.value)
    assert "build_vipe" in str(exc.value)


def test_ffs_open_session_cascades_to_vipe_model(monkeypatch):
    """The nested vipe_model on the FFS stereo route holds a session too.

    ``FastFoundationStereoModel.infer`` gets poses from ``self.vipe_model.infer`` —
    vipe_model is an independently constructed VipeSceneModel, so without the
    cascade, nested calls in session mode would go through oneshot and ViPE weights
    would be rebuilt on every stereo scene inference ("weights loaded once" would
    not hold for the stereo route). Asserts that after open_session both hold a
    session and after close_session both are cleared.

    No real subprocess: ``_open_own_session`` is patched (on the **base class**, so
    the outer FFS and the nested vipe_model share the same replacement); only the
    propagation logic itself is verified.
    """

    class _FakeSession:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    opened = []
    monkeypatch.setattr(_SceneBase, "_open_own_session",
                        lambda self: (opened.append(self), setattr(self, "_session", _FakeSession()))[1])

    ffs = FastFoundationStereoModel(script="/dev/null", vipe_script="/dev/null")
    assert ffs._session is None
    assert ffs.vipe_model._session is None

    ffs.open_session()
    assert ffs._session is not None, "outer FFS should open a session"
    assert ffs.vipe_model._session is not None, "nested vipe_model should also open a session (stereo route: weights load once)"
    assert opened == [ffs, ffs.vipe_model]

    ffs.close_session()
    assert ffs._session is None
    assert ffs.vipe_model._session is None


def test_ffs_adapter_requires_stereo_input(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBORAY_FFS_PYTHON", sys.executable)
    model = FastFoundationStereoModel(script=str(FAKES / "fake_ffs.py"),
                                      vipe_script=str(FAKES / "fake_vipe.py"))
    model.load("cpu")
    with pytest.raises(ModelEnvError) as exc:
        model.infer(
            {"frames": np.zeros((2, 8, 8, 3), np.uint8), "episode_id": "e"},
            {"fps": 30.0, "has_stereo": False, "T_left_to_right": np.eye(4).tolist()},
            tmp_path / "s.npz",
        )
    assert "frames_right" in str(exc.value)


def test_ffs_adapter_computes_depth_from_baseline(tmp_path, monkeypatch):
    """depth = fx*B/disparity; the stand-in has disparity=5, B=0.0325, fx=8 -> 0.052."""
    monkeypatch.setenv("ROBORAY_FFS_PYTHON", sys.executable)
    calib = {
        "fps": 30.0,
        "has_stereo": True,
        "T_left_to_right": np.diag([1.0, 1, 1, 1]).tolist(),
    }
    calib["T_left_to_right"][0][3] = 0.0325
    model = FastFoundationStereoModel(script=str(FAKES / "fake_ffs.py"),
                                      vipe_script=str(FAKES / "fake_vipe.py"))
    model.load("cpu")
    out = tmp_path / "s.npz"
    model.infer(
        {"frames": np.zeros((2, 8, 8, 3), np.uint8),
         "frames_right": np.zeros((2, 8, 8, 3), np.uint8), "episode_id": "e"},
        calib,
        out,
    )
    z = np.load(out, allow_pickle=False)
    assert str(z["depth_source"]) == "stereo"
    assert z["depth"].shape == (2, 8, 8)
    assert np.isfinite(z["depth"]).all()
    # Depth value pinned: fx=8 (w=8), B=0.0325, disparity=5 -> 8*0.0325/5 = 0.052.
    assert np.allclose(z["depth"], 0.052)


def test_ffs_depth_flows_from_calib_baseline_via_extra(tmp_path, monkeypatch):
    """Counter-check the --extra path: perturbing calib's baseline changes the depth.

    B=0.065 should give 8*0.065/5 = 0.104; if the adapter did not forward baseline
    to the script via --extra (the script falls back to the hardcoded 0.0325), the
    result would be 0.052 and this test would fail.
    """
    monkeypatch.setenv("ROBORAY_FFS_PYTHON", sys.executable)
    calib = {
        "fps": 30.0,
        "has_stereo": True,
        "T_left_to_right": np.eye(4).tolist(),
    }
    calib["T_left_to_right"][0][3] = 0.065
    model = FastFoundationStereoModel(script=str(FAKES / "fake_ffs.py"),
                                      vipe_script=str(FAKES / "fake_vipe.py"))
    model.load("cpu")
    out = tmp_path / "s_b2.npz"
    model.infer(
        {"frames": np.zeros((2, 8, 8, 3), np.uint8),
         "frames_right": np.zeros((2, 8, 8, 3), np.uint8), "episode_id": "e"},
        calib,
        out,
    )
    z = np.load(out, allow_pickle=False)
    assert np.allclose(z["depth"], 0.104)


def test_stereo_route_reaches_ffs_adapter_end_to_end(tmp_path, monkeypatch):
    """C1 regression: stereo prepared workspace -> read_prepared_frames (actor reads
    disk) -> FFS adapter (pointing at the fake script) -> scene.npz's
    depth_source == "stereo".

    This is the end-to-end regression for the seam that was fixed: the old plan put
    the right frames in ``calib["frames_right"]``, but the FFS adapter requires them
    inside the **frames dict**; the old actor only passed ``{"frames":...,
    "episode_id":...}``, so every stereo episode's Scene stage failed. The test uses
    exactly the frames construction the actor uses (``read_prepared_frames``).
    """
    from scripts.make_synthetic_ego import generate_dataset
    from roboray.stages.prepare import run_prepare
    from roboray.ray_runtime.actors import read_prepared_frames

    monkeypatch.setenv("ROBORAY_FFS_PYTHON", sys.executable)
    monkeypatch.setenv("ROBORAY_VIPE_PYTHON", sys.executable)
    src, ws_root = tmp_path / "src", tmp_path / "ws"
    generate_dataset(src, n_episodes=1, n_frames=4, size=(64, 48), seed=0)  # stereo by default
    run_prepare(storage_root=str(src), workspace_root=str(ws_root),
                episode_rel_dir="ep_0001", episode_id="ep_0001")

    # The actor reads disk on **its own node** (payload only carries paths) — same
    # construction as run_task. C1 contract: with stereo, frames_right is hoisted
    # into the frames dict.
    frames, calib = read_prepared_frames(str(ws_root), "ep_0001")
    assert calib["has_stereo"] is True
    assert "frames_right" in frames

    model = FastFoundationStereoModel(script=str(FAKES / "fake_ffs.py"),
                                      vipe_script=str(FAKES / "fake_vipe.py"))
    model.load("cpu")
    out = tmp_path / "scene.npz"
    model.infer(frames, calib, out)
    z = np.load(out, allow_pickle=False)
    assert str(z["depth_source"]) == "stereo"
    assert z["depth"].shape == frames["frames"].shape[:3]
    assert np.isfinite(z["depth"]).all()

    # Counter-check: a frames dict without frames_right makes the FFS adapter fail
    # directly — proving the call above really exercised the fixed seam rather than
    # unrelated code.
    old_frames = {"frames": frames["frames"], "episode_id": "ep_0001"}
    with pytest.raises(ModelEnvError):
        model.infer(old_frames, calib, tmp_path / "old.npz")


def test_ffs_adapter_resize_keeps_depth_invariant(tmp_path, monkeypatch):
    """stereo_max_image_side_px goes calib -> --extra -> run_ffs, shrinking the
    **model input** geometry, but the output depth returns to the original frame
    geometry and the values are invariant to the resize (fx and disparity scale
    together)."""
    monkeypatch.setenv("ROBORAY_FFS_PYTHON", sys.executable)
    frames = np.zeros((2, 8, 8, 3), np.uint8)
    calib = {"fps": 30.0, "has_stereo": True, "K": np.eye(3).tolist(),
             "T_left_to_right": np.eye(4).tolist()}
    calib["T_left_to_right"][0][3] = 0.0325
    model = FastFoundationStereoModel(script=str(FAKES / "fake_ffs.py"),
                                      vipe_script=str(FAKES / "fake_vipe.py"))
    model.load("cpu")
    out_full = tmp_path / "full.npz"
    model.infer({"frames": frames, "frames_right": frames, "episode_id": "e"}, calib, out_full)
    # limit=4 < max(8,8): the model input geometry shrinks but the depth values are
    # unchanged (and the output returns to the original 8x8)
    out_small = tmp_path / "small.npz"
    model.infer({"frames": frames, "frames_right": frames, "episode_id": "e"},
                dict(calib, stereo_max_image_side_px=4), out_small)
    with np.load(out_full, allow_pickle=False) as zf, np.load(out_small, allow_pickle=False) as zs:
        assert zf["depth"].shape == (2, 8, 8)
        # The knob only shrinks the **model input** geometry; the output depth always
        # returns to the original frame geometry (residual #1 fix)
        assert zs["depth"].shape == (2, 8, 8)
        assert np.allclose(zf["depth"][:, 0, 0], zs["depth"][:, 0, 0])  # depth values unchanged


def test_ffs_depth_returns_to_original_grid_after_resize(tmp_path, monkeypatch):
    """Regression gate: when the knob is active, depth must return to the original
    frame geometry (otherwise fusion sampling misaligns)."""
    import json as _json
    import subprocess as _sp

    monkeypatch.setenv("ROBORAY_FFS_PYTHON", sys.executable)
    # Original 16x16, capped at max_side=8 -> model input 8x8, but the output depth
    # must be 16x16
    calib = {"fps": 30.0, "has_stereo": True, "T_left_to_right": np.eye(4).tolist(),
             "K": np.diag([16.0, 16.0, 1.0, 1.0]).tolist(),
             "stereo_max_image_side_px": 8}
    calib["T_left_to_right"][0][3] = 0.0325
    frames = np.zeros((2, 16, 16, 3), np.uint8)
    right = np.zeros((2, 16, 16, 3), np.uint8)
    model = FastFoundationStereoModel(script=str(FAKES / "fake_ffs.py"),
                                      vipe_script=str(FAKES / "fake_vipe.py"))
    model.load("cpu")
    out = tmp_path / "s.npz"
    model.infer({"frames": frames, "frames_right": right, "episode_id": "e"}, calib, out)
    z = np.load(out, allow_pickle=False)
    assert z["depth"].shape == (2, 16, 16)          # original geometry
    assert z["intrinsics"].shape == (2, 3, 3)


def test_fake_ffs_reports_both_grids(tmp_path):
    """The script's __RESULT__ records the model-input grid and the output depth
    grid, which differ under resizing."""
    import json as _json
    import subprocess as _sp

    inp = tmp_path / "in.npz"
    np.savez(inp, frames=np.zeros((2, 16, 16, 3), np.uint8),
             frames_right=np.zeros((2, 16, 16, 3), np.uint8),
             n_frames=np.int32(2), fps=np.float32(30.0))
    out = tmp_path / "o.npz"
    proc = _sp.run([sys.executable, str(FAKES / "fake_ffs.py"),
                    "--input", str(inp), "--output", str(out),
                    "--extra", _json.dumps({"baseline_m": 0.0325, "fx": 16.0, "stereo_max_image_side_px": 8})],
                   capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    res = _json.loads([l for l in proc.stdout.splitlines() if l.startswith("__RESULT__")][-1][10:])
    assert res["model_input_size"] == [8, 8]
    assert res["depth_grid"] == [16, 16]
    assert np.load(out, allow_pickle=False)["depth"].shape == (2, 16, 16)


def test_registry_resolves_builtin_scene_models():
    assert isinstance(registry.get_scene_model_class("vipe")(), VipeSceneModel) or hasattr(
        registry.get_scene_model_class("vipe")(), "infer"
    )
    assert hasattr(registry.get_scene_model_class("fast_foundation_stereo")(), "infer")


def test_cost_hint_scales_with_frames():
    m = VipeSceneModel(script="/dev/null")
    assert m.cost_hint(10, "640x480") == pytest.approx(m.cost_hint(5, "640x480") * 2, rel=1e-6)


def test_gpu_payload_carries_no_arrays(tmp_path):
    """Hard constraint: the GPU payload may only carry paths and scalars.

    Sending frame arrays through the Ray object store is shared memory on a single
    node but travels over the network across nodes — exactly the "large arrays
    moved across nodes" that the spec eliminates. An ndarray appearing in the
    payload is a regression.
    """
    from roboray.stages.scene import scene_payload_args

    payload = scene_payload_args(
        workspace_root=str(tmp_path), episode_id="ep_0001",
        model_name="vipe", out_path=str(tmp_path / "ep_0001" / "artifacts" / "scene.npz"),
    )
    for key, value in payload.items():
        assert not isinstance(value, np.ndarray), f"payload[{key!r}] must not be an array"
    assert payload["workspace_root"] == str(tmp_path)
    assert payload["episode_id"] == "ep_0001"


def test_hand_payload_carries_no_arrays(tmp_path):
    """Hard constraint (hand route): the GPU payload may only carry paths and scalars.

    scene and hand share the same payload contract — arrays are always read from
    disk by the actor on its own node; an ndarray in the payload is a regression.
    """
    from roboray.stages.hand import hand_payload_args

    payload = hand_payload_args(
        workspace_root=str(tmp_path), episode_id="ep_0001",
        model_name="hamer", out_path=str(tmp_path / "ep_0001" / "artifacts" / "hand.npz"),
    )
    for key, value in payload.items():
        assert not isinstance(value, np.ndarray), f"payload[{key!r}] must not be an array"
    assert payload["kind"] == "hand"
    assert payload["workspace_root"] == str(tmp_path)
    assert payload["episode_id"] == "ep_0001"
    assert payload["model_name"] == "hamer"
    # max_hands is absent unless explicitly requested (the caller forwards it only
    # when passed; the script falls back to 4 by default)
    assert "max_hands" not in payload
    payload_max = hand_payload_args(
        workspace_root=str(tmp_path), episode_id="ep_0001",
        model_name="hamer", out_path=str(tmp_path / "x.npz"), max_hands=6,
    )
    assert payload_max["max_hands"] == 6


def test_read_prepared_frames_hoists_right_only_when_stereo(tmp_path):
    """C1 companion semantics unchanged: frames_right is attached only for stereo.

    Previously hoisted by ``build_infer_frames`` from ``calib["frames_right"]``; the
    actor now reads disk directly, but the "no frames_right for monocular"
    semantics must be preserved.
    """
    from roboray.ray_runtime.actors import read_prepared_frames

    ws = tmp_path / "ep_0001" / "prepared"
    ws.mkdir(parents=True)
    np.save(ws / "frames_left.npy", np.zeros((4, 8, 8, 3), np.uint8))

    (ws / "calib.json").write_text(json.dumps(
        {"K": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "dist": [0] * 5, "fps": 30,
         "width": 8, "height": 8, "has_stereo": False}
    ))
    frames, calib = read_prepared_frames(str(tmp_path), "ep_0001")
    assert "frames_right" not in frames
    assert calib["has_stereo"] is False
    # episode_id travels with the frames dict (the stand-in _maybe_fail matches
    # fail_on on it; e2e depends on this)
    assert frames["episode_id"] == "ep_0001"

    np.save(ws / "frames_right.npy", np.ones((4, 8, 8, 3), np.uint8))
    (ws / "calib.json").write_text(json.dumps(
        {"K": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "dist": [0] * 5, "fps": 30,
         "width": 8, "height": 8, "has_stereo": True}
    ))
    frames, calib = read_prepared_frames(str(tmp_path), "ep_0001")
    assert "frames_right" in frames
    assert frames["frames_right"].shape == (4, 8, 8, 3)
