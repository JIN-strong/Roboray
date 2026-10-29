import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

FAKES = Path(__file__).resolve().parents[1] / "fakes" / "modelenv_scripts"
REPO = Path(__file__).resolve().parents[2]


def _run(script: Path, out: Path, *, stereo: bool = False, extra: dict | None = None) -> dict:
    inp = out.parent / "in.npz"
    arrays = {"frames": np.zeros((4, 8, 8, 3), np.uint8), "n_frames": np.int32(4), "fps": np.int32(30)}
    if stereo:
        arrays["frames_right"] = np.zeros((4, 8, 8, 3), np.uint8)
    np.savez(inp, **arrays)
    cmd = [sys.executable, str(script), "--input", str(inp), "--output", str(out)]
    if extra is not None:
        cmd += ["--extra", json.dumps(extra)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    result_lines = [l for l in proc.stdout.splitlines() if l.startswith("__RESULT__")]
    assert result_lines
    return json.loads(result_lines[-1][len("__RESULT__"):])


def test_fake_vipe_contract(tmp_path):
    out = tmp_path / "scene.npz"
    summary = _run(FAKES / "fake_vipe.py", out)
    assert summary["ok"] is True
    z = np.load(out, allow_pickle=False)
    assert z["camera_poses"].shape == (4, 4, 4)
    assert z["intrinsics"].shape == (4, 3, 3)
    assert z["depth"].shape == (4, 8, 8)
    assert str(z["depth_source"]) == "monocular"


def test_fake_ffs_contract(tmp_path):
    out = tmp_path / "scene.npz"
    summary = _run(FAKES / "fake_ffs.py", out, stereo=True)
    assert summary["ok"] is True
    z = np.load(out, allow_pickle=False)
    assert z["depth"].shape == (4, 8, 8)
    assert str(z["depth_source"]) == "stereo"


def test_fake_ffs_requires_right_frames(tmp_path):
    out = tmp_path / "scene.npz"
    inp = out.parent / "in.npz"
    np.savez(inp, frames=np.zeros((4, 8, 8, 3), np.uint8), n_frames=np.int32(4), fps=np.int32(30))
    proc = subprocess.run(
        [sys.executable, str(FAKES / "fake_ffs.py"), "--input", str(inp), "--output", str(out)],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode != 0
    assert "frames_right" in proc.stderr


def test_fake_ffs_resize_scales_disparity_depth_invariant(tmp_path):
    """I3: when stereo_max_image_side_px shrinks the stereo pair, the fake disparity scales equally
    (same semantics as run_ffs), so depth is unchanged; summary reports the model input size and the
    output depth grid so geometric semantics can be asserted."""
    base_extra = {"fx": 8.0, "baseline_m": 0.0325}
    full = _run(FAKES / "fake_ffs.py", tmp_path / "full.npz", stereo=True, extra=base_extra)
    small = _run(FAKES / "fake_ffs.py", tmp_path / "small.npz", stereo=True,
                 extra={**base_extra, "stereo_max_image_side_px": 4})
    assert full["input_size"] == [8, 8]
    assert small["input_size"] == [4, 4]  # model input geometry shrinks
    assert small["model_input_size"] == [4, 4] and small["depth_grid"] == [8, 8]
    zf = np.load(tmp_path / "full.npz", allow_pickle=False)["depth"]
    zs = np.load(tmp_path / "small.npz", allow_pickle=False)["depth"]
    # depth always returns to the original frame geometry (scaling only affects the model input)
    assert zf.shape[-2:] == (8, 8) and zs.shape[-2:] == (8, 8)
    assert np.allclose(zf[:, 0, 0], zs[:, 0, 0])  # depth invariant to scaling


def test_fake_ffs_no_resize_when_limit_zero_or_absent(tmp_path):
    """I3: limit=0 or absent means no scaling (input size and depth unchanged)."""
    base_extra = {"fx": 8.0, "baseline_m": 0.0325}
    absent = _run(FAKES / "fake_ffs.py", tmp_path / "absent.npz", stereo=True, extra=base_extra)
    zero = _run(FAKES / "fake_ffs.py", tmp_path / "zero.npz", stereo=True,
                extra={**base_extra, "stereo_max_image_side_px": 0})
    assert absent["input_size"] == [8, 8]
    assert zero["input_size"] == [8, 8]


def test_fake_hamer_contract(tmp_path):
    out = tmp_path / "hand.npz"
    summary = _run(FAKES / "fake_hamer.py", out)
    assert summary["ok"] is True
    z = np.load(out, allow_pickle=False)
    assert z["hand_present"].shape == (4, 2)
    assert z["mano_pose"].shape == (4, 2, 48)
    assert z["joints_2d"].shape == (4, 2, 21, 3)
    # camera-frame MANO 3D joints (the A1 mano-path input): root = mano_trans, left/right distinguishable
    assert z["joints_3d_cam"].shape == (4, 2, 21, 3)
    assert np.allclose(z["joints_3d_cam"][:, :, 0], z["mano_trans"], atol=1e-6)
    assert not np.allclose(z["joints_3d_cam"][:, 0], z["joints_3d_cam"][:, 1])


def test_fake_hamer_candidates_contract(tmp_path):
    out = tmp_path / "hand.npz"
    summary = _run(FAKES / "fake_hamer.py", out, extra={"max_hands": 4, "bystander": True})
    with np.load(out, allow_pickle=True) as z:
        assert "candidates" in z.files
        cands = z["candidates"]
    assert cands.shape == (4,)                       # T=4 (_run frame count)
    assert summary["n_candidates"] == 4 * 3          # wearer 2 + bystander 1, per frame
    for frame in cands:
        assert isinstance(frame, list) and len(frame) >= 2
        for c in frame:
            assert set(c) >= {"bbox", "side", "score", "joints_2d", "joints_3d_cam"}
            assert c["bbox"].shape == (4,)
            assert c["joints_2d"].shape == (21, 3)
            assert c["joints_3d_cam"].shape == (21, 3)
            # pixel units (same as the real protocol): projections land inside _run's 8x8 fake frame
            assert np.all(c["joints_2d"][:, :2] >= 0) and np.all(c["joints_2d"][:, :2] <= 8)
            assert np.all(c["bbox"] >= 0) and np.all(c["bbox"] <= 8)
    # bystander frames: >=3 candidates with the bystander bbox higher/smaller (association-layer
    # distinguishability is guaranteed by the data shape)
    for frame in cands:
        lefts = sorted((c for c in frame if c["side"] == 0), key=lambda c: c["bbox"][1])
        assert len(lefts) >= 2                       # bystander side=0, coexisting with the wearer's left hand
        byst, wearer = lefts[0], lefts[-1]
        assert byst["bbox"][1] < wearer["bbox"][1]   # higher up (smaller bbox y)
        assert (byst["bbox"][2] - byst["bbox"][0]) < (wearer["bbox"][2] - wearer["bbox"][0])
        assert (byst["bbox"][3] - byst["bbox"][1]) < (wearer["bbox"][3] - wearer["bbox"][1])
        # bystander joints = wearer's left-hand joints plus a translation offset (same shape, distinguishable)
        assert byst["joints_3d_cam"].shape == wearer["joints_3d_cam"].shape
        assert not np.allclose(byst["joints_3d_cam"], wearer["joints_3d_cam"])


def test_fake_hamer_candidates_default_wearer_only(tmp_path):
    """Default (no extra) also produces candidates: 2 per frame, the wearer's left and right, from the
    same source as the main array (root joint = the corresponding mano_trans slot; present mask
    consistent) - no effect downstream until consumed."""
    out = tmp_path / "hand.npz"
    summary = _run(FAKES / "fake_hamer.py", out)
    with np.load(out, allow_pickle=True) as z:
        cands = z["candidates"]
        mano_trans = z["mano_trans"]
        present = z["hand_present"]
    assert cands.shape == (4,)
    assert summary["n_candidates"] == 8
    for i, frame in enumerate(cands):
        assert isinstance(frame, list)
        assert {c["side"] for c in frame} == {0, 1}  # first left/right, same source as the main-array slots
        assert all(c["score"] == 0.9 for c in frame)
        for c in frame:                              # candidate and main array share source: root = mano_trans
            assert np.allclose(c["joints_3d_cam"][0], mano_trans[i, c["side"]], atol=1e-6)
            # unit pin: joints_2d are **pixel** coordinates - recompute the fake-camera projection
            # (x 8 frame width/height); if the fake forgot to multiply by w/h (stuck at the ~0.5
            # normalized scale), this assertion turns red.
            j3d = c["joints_3d_cam"]
            z = j3d[:, 2] + 2.0
            assert np.allclose(c["joints_2d"][:, 0], (0.5 + j3d[:, 0] / z) * 8.0, atol=1e-5)
            assert np.allclose(c["joints_2d"][:, 1], (0.5 + j3d[:, 1] / z) * 8.0, atol=1e-5)
    # t=4 has no fixture miss and stride=1: both hands present on all frames
    assert present.all()


def test_run_hamer_max_hands_and_candidates_static():
    """The real env is not executable locally (same as test_real_scripts_importable_and_have_serve_symbols):
    run_hamer's max_hands/candidates wiring is pinned statically with AST on the **call chain**
    (not just existence): infer_once -> _boxes_from_mediapipe(max_num_hands=max_hands) ->
    Hands(max_num_hands=<param>), and every detection goes into candidates_rows[...].append(...).
    Behavior is backed by the isomorphic fake (the two cases above) plus real-cluster regression."""
    import ast

    src = (REPO / "scripts" / "modelenvs" / "run_hamer.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    boxes_fn = next(n for n in ast.walk(tree)
                    if isinstance(n, ast.FunctionDef) and n.name == "_boxes_from_mediapipe")
    assert "max_num_hands" in {a.arg for a in boxes_fn.args.args}
    infer_fn = next(n for n in ast.walk(tree)
                    if isinstance(n, ast.FunctionDef) and n.name == "infer_once")
    assert {"max_hands", "candidates_rows"} <= _bound_names(infer_fn)
    assert "DEFAULT_MAX_HANDS" in src
    assert '"candidates"' in src                     # the save_arrays dict carries the new key

    def kw_is_name(call, kw: str, name: str) -> bool:
        val = next((k.value for k in call.keywords if k.arg == kw), None)
        return isinstance(val, ast.Name) and val.id == name

    # (1) call-site forwarding: infer_once -> _boxes_from_mediapipe(..., max_num_hands=max_hands)
    box_calls = [n for n in ast.walk(infer_fn) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name)
                 and n.func.id == "_boxes_from_mediapipe"]
    assert box_calls and any(kw_is_name(c, "max_num_hands", "max_hands")
                             for c in box_calls), (
        "infer_once does not forward max_hands to _boxes_from_mediapipe via max_num_hands=")
    # (2) detector wiring: Hands(..., max_num_hands=<param>) inside _boxes_from_mediapipe
    param = "max_num_hands"
    hands_calls = [n for n in ast.walk(boxes_fn) if isinstance(n, ast.Call)
                   and isinstance(n.func, ast.Attribute) and n.func.attr == "Hands"]
    assert hands_calls and any(kw_is_name(c, "max_num_hands", param)
                               for c in hands_calls), (
        "Hands(...) max_num_hands is not wired to the parameter (falling back to the HANDS constant turns this red)")
    # (3) all detections kept: infer_once must contain candidates_rows[...].append(...)
    appends = [n for n in ast.walk(infer_fn) if isinstance(n, ast.Call)
               and isinstance(n.func, ast.Attribute) and n.func.attr == "append"
               and isinstance(n.func.value, ast.Subscript)
               and isinstance(n.func.value.value, ast.Name)
               and n.func.value.value.id == "candidates_rows"]
    assert appends, "infer_once has no candidates_rows[...].append - are same-side detections dropped?"


# Names each script's infer_once **must bind locally** - a defect class found in review: the binding
# lived in build_model while the use was in infer_once, NameError on the first inference in a real
# env (unobservable locally without the real env).
_REQUIRED_INFER_BINDINGS = {
    "run_ffs.py": ("InputPadder",),
    "run_hamer.py": ("mano",),
}


def _bound_names(fn) -> set:
    """Collect names bound inside a function body: params, import aliases, assign/for/with/except/comprehension targets."""
    import ast

    bound = set()
    for arg in list(fn.args.posonlyargs) + list(fn.args.args) + list(fn.args.kwonlyargs):
        bound.add(arg.arg)
    if fn.args.vararg:
        bound.add(fn.args.vararg.arg)
    if fn.args.kwarg:
        bound.add(fn.args.kwarg.arg)

    def add_target(t):
        if t is None:
            return
        if isinstance(t, ast.Name):
            bound.add(t.id)
        elif isinstance(t, (ast.Tuple, ast.List)):
            for e in t.elts:
                add_target(e)

    for node in ast.walk(fn):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                bound.add(a.asname or a.name.split(".")[0])
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                add_target(t)
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            add_target(node.target)
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            add_target(node.target)
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                add_target(item.optional_vars)
        elif isinstance(node, ast.comprehension):
            add_target(node.target)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
    return bound


def test_real_scripts_importable_and_have_serve_symbols():
    """The three real scripts must parse as AST and contain build_model/infer_once/main symbols.

    The real env (torch/hamer) is not available locally, so they cannot be executed; but
    "syntax-valid + symbols present + --serve branch present + infer_once local bindings" can be
    locked via ast/text checks in the main env - E2 once hid a resident-worker-killing SystemExit
    left in the serve loop, and review found the NameError class where the binding stayed in
    build_model while the use was in infer_once.
    """
    import ast

    for name in ("run_vipe.py", "run_ffs.py", "run_hamer.py"):
        src = (REPO / "scripts" / "modelenvs" / name).read_text(encoding="utf-8")
        tree = ast.parse(src)  # syntax errors blow up here
        funcs = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        assert {"build_model", "infer_once", "main"} <= funcs, name
        assert '"--serve" in sys.argv' in src, name
        # the infer path must not use SystemExit (it would kill the resident process)
        infer_fn = next(n for n in ast.walk(tree)
                        if isinstance(n, ast.FunctionDef) and n.name == "infer_once")
        for node in ast.walk(infer_fn):
            if isinstance(node, ast.Raise):
                exc = node.exc
                called = getattr(exc, "func", None)
                assert getattr(called, "id", None) != "SystemExit", (
                    f"{name}: infer_once must not raise SystemExit (would kill the resident worker)"
                )
        # binding-scope assertion: names used in infer_once must be bound inside it (locking the review defect class).
        bindings = _bound_names(infer_fn)
        for required in _REQUIRED_INFER_BINDINGS.get(name, ()):
            assert required in bindings, (
                f"{name}: infer_once does not bind {required} - binding in build_model, use in "
                f"infer_once would NameError on the first inference in a real env"
            )
