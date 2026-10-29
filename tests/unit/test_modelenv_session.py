"""ModelEnvSession: lifecycle contract of the resident subprocess session (tested
with the echo worker)."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from roboray.models.modelenv import ModelEnvError, ModelEnvSession

REPO = Path(__file__).resolve().parents[2]
WORKER = str(REPO / "tests" / "fakes" / "serve_worker.py")


def _inp(path: Path, value: float) -> None:
    np.savez(path, x=np.full((4,), value, np.float32))


def _session(tmp_path: Path) -> ModelEnvSession:
    s = ModelEnvSession(sys.executable, WORKER, timeout_s=30.0)
    s.start()
    return s


def test_start_infer_close_roundtrip(tmp_path):
    s = _session(tmp_path)
    try:
        assert s.alive
        _inp(tmp_path / "in.npz", 5.0)
        r = s.infer(str(tmp_path / "in.npz"), str(tmp_path / "out.npz"))
        assert r["ok"] is True
        with np.load(tmp_path / "out.npz") as z:
            assert z["y"].tolist() == [10.0, 10.0, 10.0, 10.0]
    finally:
        s.close()
    assert not s.alive


def test_close_is_idempotent(tmp_path):
    s = _session(tmp_path)
    s.close()
    s.close()  # the second close must not raise


def test_start_handshake_failure_cleans_up_process_and_err_file(monkeypatch):
    """A handshake failure (``_read_response`` raising timeout/EOF) must go through
    ``close()`` cleanup, leaving no ``_proc``/``_err_file`` residue.

    monkeypatch makes ``_read_response`` raise to simulate the handshake failure — a
    real timeout takes 60s, and the real EOF branch already closes internally, which
    would mask this path. Before the fix, start() had no try/except and the
    exception propagated directly, leaking the Popen handle and the temp file; the
    fix routes everything through ``self.close(); raise``.
    """
    s = ModelEnvSession(sys.executable, WORKER, timeout_s=30.0)

    def _boom(self, *a, **kw):
        raise ModelEnvError("handshake failed (simulated timeout/EOF)")

    monkeypatch.setattr(ModelEnvSession, "_read_response", _boom)
    with pytest.raises(ModelEnvError, match="handshake failed"):
        s.start()
    assert s._proc is None     # close() cleaned up the process handle
    assert s._err_file is None  # close() closed the temp file


def test_infer_failure_returns_error_and_session_survives(tmp_path):
    s = _session(tmp_path)
    try:
        r = s.infer(str(tmp_path / "missing.npz"), str(tmp_path / "o.npz"))
        assert r["ok"] is False and "error" in r
        assert s.alive
    finally:
        s.close()


def test_infer_timeout_raises_and_kills(tmp_path):
    """Deadline exhausted: ModelEnvError + the process is cleaned up (no orphans)."""
    s = ModelEnvSession(sys.executable, WORKER, timeout_s=1.0)
    s.start()
    try:
        _inp(tmp_path / "in.npz", 1.0)
        with pytest.raises(ModelEnvError, match="timed out"):
            # sleep_s: makes the echo worker infer slowly to hit the 0.2s deadline
            # (the worker has an extra["sleep_s"] test hook, see
            # tests/fakes/serve_worker.py)
            s.infer(str(tmp_path / "in.npz"), str(tmp_path / "o.npz"),
                    timeout_s=0.2, extra='{"sleep_s": 0.5}')
        assert not s.alive
    finally:
        s.close()


def test_crash_then_next_infer_auto_restarts(tmp_path):
    """Child killed -> alive False; the next infer auto-restarts successfully
    (spec §5.4)."""
    s = _session(tmp_path)
    try:
        s._proc.kill()  # simulate OOM/segfault
        s._proc.wait(timeout=5)
        assert not s.alive
        _inp(tmp_path / "in.npz", 2.0)
        r = s.infer(str(tmp_path / "in.npz"), str(tmp_path / "out.npz"))
        assert r["ok"] is True
        assert s.alive
    finally:
        s.close()


def test_ping_dead_process_returns_false(tmp_path):
    s = _session(tmp_path)
    try:
        assert s.ping() is True
        s._proc.kill()
        s._proc.wait(timeout=5)
        assert s.ping() is False
    finally:
        s.close()
