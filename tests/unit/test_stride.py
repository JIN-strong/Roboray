"""inference_fps -> stride sampling: default behavior byte-identical."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "modelenvs"))
from _common import compute_stride  # noqa: E402

FAKES = Path(__file__).resolve().parents[1] / "fakes" / "modelenv_scripts"


def _run(out, extra=None):
    inp = out.parent / "in.npz"
    np.savez(inp, frames=np.zeros((8, 8, 8, 3), np.uint8),
             n_frames=np.int32(8), fps=np.int32(30))
    cmd = [sys.executable, str(FAKES / "fake_hamer.py"),
           "--input", str(inp), "--output", str(out)]
    if extra is not None:
        cmd += ["--extra", json.dumps(extra)]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    assert p.returncode == 0, p.stderr
    return json.loads([l for l in p.stdout.splitlines()
                       if l.startswith("__RESULT__")][-1][10:])


def test_compute_stride_rules():
    assert compute_stride(30, 30) == 1      # same rate: all frames (default)
    assert compute_stride(30, 15) == 2
    assert compute_stride(30, None) == 1    # not configured
    assert compute_stride(None, 15) == 1    # fps unknown: fall back to all frames
    assert compute_stride(30, 100) == 1     # above source rate: never below all frames


def test_stride_two_marks_unobserved(tmp_path):
    """stride=2: odd frames unobserved (both slots False); the fixture's missed frames 3:5
    (slot 0) are not covered by the stride."""
    out = tmp_path / "hand.npz"
    _run(out, extra={"inference_fps": 15})
    with np.load(out) as z:
        present = z["hand_present"]
    assert present[0].all() and not present[1].any()   # 0,2,4,6 infer; 1,3,5,7 unobserved
    assert not present[1::2].any()
    # The fixture's missed frames 3:5 (slot 0) remain missed at even frame 4 — the stride
    # mask stacks on top, it does not overwrite them
    assert not present[4, 0] and present[4, 1]
    assert np.all(present[[0, 2, 6]])


def _deep_equal(a, b) -> bool:
    """Element-wise deep comparison: object arrays (candidates, dicts containing ndarrays)
    cannot go through np.array_equal — dict == dict triggers ndarray's ambiguous truth value."""
    if isinstance(a, dict):
        return set(a) == set(b) and all(_deep_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_deep_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, np.ndarray):
        if a.dtype == object:
            return a.shape == b.shape and all(
                _deep_equal(x, y) for x, y in zip(a.tolist(), b.tolist()))
        return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
    return bool(a == b)


def test_stride_default_identical_output(tmp_path):
    """With inference_fps=30 (the default) vs no extra, the output is byte-identical."""
    a, b = tmp_path / "a.npz", tmp_path / "b.npz"
    _run(a, extra={"inference_fps": 30})
    _run(b)
    # allow_pickle=True: the fake carries the candidates (object array) key
    za, zb = np.load(a, allow_pickle=True), np.load(b, allow_pickle=True)
    try:
        assert za.files == zb.files
        for k in za.files:
            assert _deep_equal(za[k], zb[k]), f"key {k} drifted"
    finally:
        za.close(), zb.close()
