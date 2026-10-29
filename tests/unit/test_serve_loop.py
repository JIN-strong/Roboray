"""Contract for `_common.run_serve_loop` (tested via the echo worker subprocess, no real model needed)."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
WORKER = REPO / "tests" / "fakes" / "serve_worker.py"


class _Client:
    """Minimal test client: write request lines, receive __RESULT__ responses by id."""

    def __init__(self, proc):
        self.proc = proc

    def send(self, obj: dict) -> dict:
        assert self.proc.poll() is None, "worker exited"
        self.proc.stdin.write((json.dumps(obj) + "\n").encode())
        self.proc.stdin.flush()
        while True:
            line = self.proc.stdout.readline().decode()
            assert line, "worker stdout closed unexpectedly"
            if line.startswith("__RESULT__"):
                return json.loads(line[len("__RESULT__"):])


def _start() -> tuple:
    proc = subprocess.Popen(
        [sys.executable, "-u", str(WORKER), "--serve"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    client = _Client(proc)
    hello = client.send({"id": 0, "cmd": "hello"})
    assert hello == {"id": 0, "ready": True}
    return proc, client


def _inp(path: Path, value: float) -> None:
    np.savez(path, x=np.full((4,), value, np.float32))


def test_serve_infer_builds_once_and_reuses():
    """Two infers build the model only once (when weights/extra are unchanged) - the core benefit of residency."""
    import tempfile

    proc, client = _start()
    try:
        with tempfile.TemporaryDirectory() as td:
            _inp(f"{td}/in1.npz", 1.0)
            _inp(f"{td}/in2.npz", 2.0)
            r1 = client.send({"id": 1, "cmd": "infer", "input": f"{td}/in1.npz",
                              "output": f"{td}/out1.npz", "weights": "", "extra": "{}"})
            r2 = client.send({"id": 2, "cmd": "infer", "input": f"{td}/in2.npz",
                              "output": f"{td}/out2.npz", "weights": "", "extra": "{}"})
        assert r1["ok"] and r2["ok"]
        assert r1["builds"] == 1 and r2["builds"] == 1  # echo worker reports the build count
        assert r2["built_this_call"] is False
    finally:
        client.send({"id": 9, "cmd": "shutdown"})
        proc.wait(timeout=10)


def test_serve_infer_failure_does_not_kill_worker():
    """An infer failure only returns an error line; the process must stay alive and keep serving."""
    import tempfile

    proc, client = _start()
    try:
        with tempfile.TemporaryDirectory() as td:
            r_bad = client.send({"id": 1, "cmd": "infer", "input": f"{td}/missing.npz",
                                 "output": f"{td}/o.npz", "weights": "", "extra": "{}"})
            assert r_bad["ok"] is False
            assert "error" in r_bad
            _inp(f"{td}/in.npz", 3.0)
            r_good = client.send({"id": 2, "cmd": "infer", "input": f"{td}/in.npz",
                                  "output": f"{td}/o.npz", "weights": "", "extra": "{}"})
            assert r_good["ok"] is True  # process is still alive
    finally:
        client.send({"id": 9, "cmd": "shutdown"})
        proc.wait(timeout=10)


def test_serve_weights_change_triggers_rebuild():
    """weights/extra changes must trigger a rebuild (different weights have different cache keys)."""
    import tempfile

    proc, client = _start()
    try:
        with tempfile.TemporaryDirectory() as td:
            _inp(f"{td}/in.npz", 1.0)
            r1 = client.send({"id": 1, "cmd": "infer", "input": f"{td}/in.npz",
                              "output": f"{td}/o1.npz", "weights": "", "extra": "{}"})
            r2 = client.send({"id": 2, "cmd": "infer", "input": f"{td}/in.npz",
                              "output": f"{td}/o2.npz", "weights": "other", "extra": "{}"})
        assert r1["builds"] == 1 and r2["builds"] == 2
        assert r2["built_this_call"] is True
    finally:
        client.send({"id": 9, "cmd": "shutdown"})
        proc.wait(timeout=10)


def test_serve_ping_and_unknown_cmd():
    proc, client = _start()
    try:
        p = client.send({"id": 3, "cmd": "ping"})
        assert p["ok"] is True and "uptime_s" in p
        u = client.send({"id": 4, "cmd": "wat"})
        assert u["ok"] is False and "wat" in u["error"]
    finally:
        client.send({"id": 9, "cmd": "shutdown"})
        proc.wait(timeout=10)
