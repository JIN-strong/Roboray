# E2 Actor Lifecycle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]` syntax for tracking.

**Goal:** Load the three real models' weights only once in resident worker subprocesses (reused across Episodes), implement unload()/three-tier idling (30s unload model / 240s actor exit / 300s node reclaim) and CPU-stage node routing (the Important-2 handoff from E1).

**Architecture:** the service loop and IPC protocol go in `_common.py` (shared by the three model envs); `ModelEnvSession` (modelenv.py) implements infer timeouts with select+deadline and writes stderr to a temp file to avoid pipe deadlock; the session is **held by the adapter** (payload-construction knowledge lives in the adapter), the actor only drives `open_session()`/`close_session()`; the idle policy is extracted into a pure function `compute_idle_actions` (testable with a fake clock), and the driver only applies actions. CPU routing: CPU actors are grouped by node_id and dispatched by the Episode's assigned_node.

**Tech Stack:** Python 3.10 / 3.12 (the two model envs), subprocess + selectors (IPC), Ray 2.59.0, pytest

**Spec:** `docs/superpowers/specs/2026-10-21-ray-runtime-completion-design.md` sections 5, 6, 8 + E1 handoff (Important-2)

## Global Constraints

- Test command: `RAY_ADDRESS=local python -m pytest -q --tb=short`.
- **Regression baseline: 238 passed / 0 failed** (E1's final number); each task only grows it, no failures.
- **oneshot compatibility**: the `execution.model_worker="oneshot"` path must be **verbatim equivalent** to before; the `run_*.py` CLI behavior is unchanged without `--serve` (L2 smoke tests and the 236 existing tests use it).
- **Protocol** (spec §5.2): JSON-lines over stdin/stdout, reusing the `__RESULT__{json}` marker; an infer failure returns `{"ok":false,"error":...}` **without killing the process**; the subprocess stdout must use `-u`/explicit flush.
- **Exit path**: `unload()` first (graceful session close) then `ray.kill` -- `ray.kill` is SIGKILL and runs no cleanup, leaving orphan child processes that keep holding VRAM (spec §5.6 states this explicitly).
- **The driver does not read the workspace** (E1 hard invariant, still in force): idle scanning uses only in-memory/actor return values.
- Resident subprocesses use `start_new_session=True` to create a process group; cleanup kills the whole group.
- Comments/docstrings explain "why".

## Review Focus

(each has a test in the task in parentheses)

1. **`SystemExit` in the serve loop kills the worker**: `SystemExit` does not inherit `Exception`, so a leftover `raise SystemExit("...")` from the old script code on the infer path would take down the whole resident process -- validation moved into `infer_once` must be changed to `ValueError`, and the oneshot `main()` catches it and converts to `SystemExit` to preserve CLI behavior. (Task 3)
2. **The chumpy/numpy shim and `os.chdir` are lost in serve mode**: run_hamer's `_install_numpy_shim()` and `os.chdir(weights)` (`CACHE_DIR_HAMER="./_DATA"` relative to CWD) must be done in `build_model`, not only in main. (Task 3)
3. **The next request after a session's process dies**: after a subprocess crash (OOM/segfault), `infer` must report an error and **not silently retry that same request**; the next `infer` auto-restarts and reloads. (Task 2)
4. **The idle scan must not hit in-flight actors**: an actor running a task must not be unloaded/killed -- the scan only acts on "non-busy slot" actors; `slot.busy` is the single authority for busy-ness (the semantic fixed in E1). (Task 6)
5. **CPU routing in a GPU-less environment**: this machine has no GPU (registry=None, flat pool), so CPU routing must degrade to "any free CPU actor" and no e2e regresses. (Task 7)

---

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `scripts/modelenvs/_common.py` | Add `run_serve_loop` (protocol core, shared by the three scripts) + `save_arrays` | Modify |
| `scripts/modelenvs/run_vipe.py` / `run_ffs.py` / `run_hamer.py` | Split `build_model` / `infer_once`; `main()` stays oneshot; `--serve` entry point | Modify |
| `src/roboray/models/modelenv.py` | Add `ModelEnvSession` | Modify |
| `src/roboray/models/scene_models.py` / `hand_models.py` | Adapter holds session: `open_session`/`close_session`, `infer` branch | Modify |
| `src/roboray/ray_runtime/actors.py` | GpuModelActor: `unload`/`idle_seconds`/`last_used`/`__ray_terminate__`; helper to group CPU actors by node | Modify |
| `src/roboray/scheduling/idle.py` | **New**: `compute_idle_actions` pure function (testable with a fake clock) | Create |
| `src/roboray/runtime/driver.py` | Apply the idle scan, unload at run end, dispatch CPU by node | Modify |
| `src/roboray/api/config.py` | `model_worker` / `actor_idle_unload_s` / `actor_idle_exit_s` | Modify |
| `src/roboray/api/reconstruction.py` | env_vars hands down `ROBORAY_MODEL_WORKER` etc.; unload at run end | Modify |
| `tests/fakes/serve_worker.py` | **New**: echo worker (for session/protocol tests) | Create |
| `tests/unit/test_serve_loop.py`, `tests/unit/test_modelenv_session.py`, `tests/unit/test_idle_policy.py`, `tests/e2e/test_resident_mode.py` | **New** tests | Create |

---

### Task 1: `_common.run_serve_loop` — protocol core

**Files:**
- Modify: `scripts/modelenvs/_common.py`
- Create: `tests/fakes/serve_worker.py`
- Test: `tests/unit/test_serve_loop.py`

**Interfaces:**
- Produces:
  - `save_arrays(path: str, arrays: dict) -> None` (npz written to disk + `__schema_version__=1`, **no printing**)
  - `run_serve_loop(build_model: Callable, infer_once: Callable) -> None`
    Protocol: `{"id":0,"cmd":"hello"}` → `__RESULT__{"id":0,"ready":true}`;
    `{"id":N,"cmd":"infer","input":...,"output":...,"weights":...,"extra":"{json}"}` → `__RESULT__{"id":N,"ok":true,...summary}` or `{"ok":false,"error":...}`;
    `{"id":N,"cmd":"ping"}` → `{"id":N,"ok":true,"uptime_s":...}`;
    `{"id":N,"cmd":"shutdown"}` → the process exits.
    Models are built lazily once keyed by `(weights, extra)`; a key change rebuilds. An infer exception (including `ValueError`) only returns an error line, **without killing the process**.

- [ ] **Step 1: write failing tests** (`tests/unit/test_serve_loop.py`)

```python
"""`_common.run_serve_loop` protocol contract (tested through the echo worker
subprocess, no real models needed)."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
WORKER = REPO / "tests" / "fakes" / "serve_worker.py"


class _Client:
    """Minimal test client: write a request line, collect the __RESULT__ response by id."""

    def __init__(self, proc):
        self.proc = proc

    def send(self, obj: dict) -> dict:
        assert self.proc.poll() is None, "worker already exited"
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
    """Two infers build the model only once (unchanged weights/extra) -- the core
    benefit of resident mode."""
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
        assert r1["builds"] == 1 and r2["builds"] == 1  # the echo worker reports build counts
        assert r2["built_this_call"] is False
    finally:
        client.send({"id": 9, "cmd": "shutdown"})
        proc.wait(timeout=10)


def test_serve_infer_failure_does_not_kill_worker():
    """An infer failure only returns an error line; the process must stay alive and keep
    serving (spec §5.2)."""
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
            assert r_good["ok"] is True  # the process is still alive
    finally:
        client.send({"id": 9, "cmd": "shutdown"})
        proc.wait(timeout=10)


def test_serve_weights_change_triggers_rebuild():
    """A weights/extra change must rebuild the model (different weights mean a different
    cache key)."""
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
```

- [ ] **Step 2: run and confirm failure**

```bash
cd <repo-root>
RAY_ADDRESS=local python -m pytest tests/unit/test_serve_loop.py -q -o addopts=""
```
Expected: FAIL (`serve_worker.py` does not exist → Popen reports a file error)

- [ ] **Step 3: write the echo worker (`tests/fakes/serve_worker.py`)**

```python
"""Echo worker for serve-protocol tests: implements the minimal build_model/infer_once
contract.

- build_model increments a counter on each call; infer responses carry the cumulative
  build count and whether this call rebuilt (for the "built only once" assertion).
- infer: read x from the input npz, output y = x * factor (extra may pass factor,
  default 2).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "modelenvs"))

from _common import load_input, run_serve_loop, save_arrays  # noqa: E402

_LAST_SEEN = {"builds": 0}


def build_model(weights: str, extra: dict) -> dict:
    return {"builds": _LAST_SEEN["builds"] + 1, "weights": weights,
            "factor": float(extra.get("factor", 2.0))}


def infer_once(model: dict, input_path: str, output_path: str, extra: dict) -> dict:
    data = load_input(input_path)
    y = data["x"] * model["factor"]
    save_arrays(output_path, {"y": y})
    summary = {"n": int(y.shape[0])}
    # built_this_call: whether a model build happened on this call (build count
    # advancing = built new this time)
    summary["built_this_call"] = model["builds"] != _LAST_SEEN["builds"]
    summary["builds"] = model["builds"]
    _LAST_SEEN["builds"] = model["builds"]
    return summary


if __name__ == "__main__":
    assert "--serve" in sys.argv, "the echo worker only supports --serve mode"
    run_serve_loop(build_model, infer_once)
```

- [ ] **Step 4: implement `run_serve_loop` and `save_arrays` (append to `_common.py`)**

```python
def save_arrays(path: str, arrays: dict) -> None:
    """Write npz to disk (with a schema stamp) but **do not print** __RESULT__ -- the
    response is returned uniformly by the serve loop."""
    np.savez(path, __schema_version__=1, **arrays)


def run_serve_loop(build_model, infer_once) -> None:
    """Resident service loop (--serve mode). Protocol: see the module docstring and
    spec §5.2.

    - Models are built lazily keyed by ``(weights, extra)``, built once; only a key
      change rebuilds.
    - Any Exception from infer (including ValueError) only returns an error line --
      one bad input must not drop an already-loaded model. Note that SystemExit is
      NOT caught: code moved onto the infer path must no longer use SystemExit (it
      does not inherit Exception and would kill the resident process).
    """
    import time

    start = time.time()
    cache: dict = {"key": None, "model": None}

    def _emit(obj: dict) -> None:
        print(RESULT_MARKER + json.dumps(obj, ensure_ascii=False, default=str))
        sys.stdout.flush()

    _emit({"id": 0, "ready": True})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            _emit({"id": -1, "ok": False, "error": f"request line is not valid JSON: {line[:120]}"})
            continue
        cmd, req_id = req.get("cmd"), req.get("id", -1)
        if cmd == "shutdown":
            return
        if cmd == "ping":
            _emit({"id": req_id, "ok": True, "uptime_s": round(time.time() - start, 3)})
            continue
        if cmd != "infer":
            _emit({"id": req_id, "ok": False, "error": f"unknown command: {cmd!r}"})
            continue
        try:
            weights = req.get("weights", "")
            extra_raw = req.get("extra", "{}")
            try:
                extra = json.loads(extra_raw) if extra_raw else {}
            except json.JSONDecodeError as exc:
                raise ValueError(f"--extra is not valid JSON: {exc}") from exc
            key = (weights, extra_raw)
            if cache["model"] is None or cache["key"] != key:
                cache["model"] = build_model(weights, extra)
                cache["key"] = key
            summary = infer_once(cache["model"], req["input"], req["output"], extra)
            _emit({"id": req_id, "ok": True, **(summary or {})})
        except Exception as exc:  # noqa: BLE001 -- see docstring: failure must not kill
            _emit({"id": req_id, "ok": False,
                   "error": f"{type(exc).__name__}: {exc}"})
```

- [ ] **Step 5: run the tests and confirm they pass**

```bash
RAY_ADDRESS=local python -m pytest tests/unit/test_serve_loop.py -q -o addopts=""
```
Expected: 4 passed

- [ ] **Step 6: commit**

```bash
git add scripts/modelenvs/_common.py tests/fakes/serve_worker.py tests/unit/test_serve_loop.py
git commit -m "feat(modelenvs): serve protocol core run_serve_loop + echo worker tests

JSON-lines over stdin/stdout, reusing the __RESULT__ marker; models built lazily
once keyed by (weights, extra); an infer failure only returns an error line and
does not kill the process.
serve_worker.py lets the protocol be fully tested on a GPU-less local machine."
```

---

### Task 2: `ModelEnvSession` — long-lived subprocess session

**Files:**
- Modify: `src/roboray/models/modelenv.py` (add `ModelEnvSession` to `__all__`)
- Test: `tests/unit/test_modelenv_session.py`

**Interfaces:**
- Produces:
  - `ModelEnvSession(python: str, script: str, timeout_s: float = 3600.0)`
  - `.start() -> None` (Popen `[python, "-u", script, "--serve"]`, `start_new_session=True`, stderr→temp file; wait for the `ready` line)
  - `.infer(input_path, output_path, weights="", extra="{}", timeout_s=None) -> dict` (the full payload is spec 5.2's infer request; select+deadline reads the response; timeout/dead process → `ModelEnvError` and mark dead; the **next** infer auto-restarts)
  - `.ping() -> bool` (a dead process returns False, does not raise)
  - `.alive -> bool`
  - `.close() -> None` (close stdin → wait(5s) → kill the process group as a fallback; idempotent)

**Implementation notes (write into the code comments):**
- stdout uses a **binary** pipe + `selectors.DefaultSelector` + manual line buffering -- `text=True`'s internal buffering mixed with select drops lines/hangs.
- stderr goes to `tempfile.TemporaryFile()` (not PIPE): not reading it does not block the subprocess on a full pipe; on error, read the tail.
- Response-read loop: `selector.select(remaining time)` → append read bytes to the buffer → parse each complete line; a line containing `__RESULT__` whose `id` matches → return. When the deadline runs out → kill the process group + `ModelEnvError(timeout)`.
- Auto-restart: at the `infer` entry, if `self._proc is None or poll() is not None` → `start()` first (`restarted=True` does not affect this request's semantics -- the request still runs on the new process).

- [ ] **Step 1: write failing tests** (`tests/unit/test_modelenv_session.py`)

```python
"""ModelEnvSession: lifecycle contract of the long-lived subprocess session (tested with
the echo worker)."""

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
    s.close()  # the second call must not raise


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
        with pytest.raises(ModelEnvError, match="timeout"):
            s.infer(str(tmp_path / "in.npz"), str(tmp_path / "o.npz"),
                    timeout_s=0.2)
        assert not s.alive
    finally:
        s.close()


def test_crash_then_next_infer_auto_restarts(tmp_path):
    """Subprocess killed -> alive False; the next infer auto-restarts successfully
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
```

- [ ] **Step 2: run and confirm failure** (`ImportError: cannot import name 'ModelEnvSession'`)

- [ ] **Step 3: implement** (append to `modelenv.py`; add `import os`, `import selectors`, `import signal`, `import tempfile`, `import time` at the top)

```python
class ModelEnvSession:
    """Session for the long-lived worker subprocess in a model env (spec §5.4).

    Why stderr goes to a temp file instead of PIPE: nobody reads stderr while the
    process is resident, so once the subprocess fills the pipe buffer it blocks
    entirely; a temp file has no such risk, and we read the tail on error.
    Why stdout uses binary+select: text=True's internal buffering mixed with select
    drops lines/hangs; manual line buffering keeps the "select until deadline"
    semantic clean.
    """

    def __init__(self, python: str, script: str, timeout_s: float = 3600.0):
        self.python = python
        self.script = script
        self.timeout_s = timeout_s
        self._proc: subprocess.Popen | None = None
        self._err_file = None
        self._buf = b""
        self._next_id = 1

    # -- lifecycle -------------------------------------------------------
    def start(self) -> None:
        self._err_file = tempfile.TemporaryFile()
        self._proc = subprocess.Popen(
            [self.python, "-u", self.script, "--serve"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._err_file,
            start_new_session=True,  # own process group: cleanup kills the whole group, no orphans
        )
        self._buf = b""
        ready = self._request({"cmd": "hello"}, timeout_s=60.0)
        if ready.get("ready") is not True:
            self.close()
            raise ModelEnvError(f"serve worker handshake failed: {ready!r} ({self.script})")

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def close(self) -> None:
        """Graceful shutdown: closing stdin lets the serve loop exit naturally; if it
        does not exit within 5s, kill the process group."""
        if self._proc is None:
            return
        try:
            if self._proc.stdin and not self._proc.stdin.closed:
                self._proc.stdin.close()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(self._proc.pid), signal.SIGKILL)
                self._proc.wait(timeout=5)
        except (OSError, ProcessLookupError):
            pass  # the process group is already gone
        finally:
            if self._proc.stdout:
                self._proc.stdout.close()
            if self._err_file:
                self._err_file.close()
                self._err_file = None
            self._proc = None
            self._buf = b""

    # -- requests --------------------------------------------------------
    def ping(self) -> bool:
        if not self.alive:
            return False
        try:
            return bool(self._request({"cmd": "ping"}, timeout_s=10.0).get("ok"))
        except (ModelEnvError, OSError):
            return False

    def infer(self, input_path: str, output_path: str, weights: str = "",
              extra: str = "{}", timeout_s: float | None = None) -> dict:
        """Run one inference. If the subprocess is dead, **restart first** and then run
        (this request is not silently retried: the request that crashed was already
        reported as an error by the previous call)."""
        if not self.alive:
            self.close()  # clean up the wreckage (err file etc.)
            self.start()
        req = {"cmd": "infer", "input": str(input_path), "output": str(output_path),
               "weights": weights, "extra": extra}
        return self._request(req, timeout_s=timeout_s if timeout_s is not None else self.timeout_s)

    # -- internal --------------------------------------------------------
    def _request(self, payload: dict, timeout_s: float) -> dict:
        if self._proc is None or self._proc.poll() is not None:
            raise ModelEnvError(f"serve worker is not running: {self.script}")
        req_id = self._next_id
        self._next_id += 1
        payload = {"id": req_id, **payload}
        try:
            self._proc.stdin.write((json.dumps(payload) + "\n").encode())
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            self.close()
            raise ModelEnvError(f"serve worker stdin is broken: {exc} ({self.script})") from exc
        resp = self._read_response(req_id, deadline=time.monotonic() + timeout_s)
        if not resp.get("ok", True):
            # protocol-level failure (exception inside the script): the process is
            # still alive and the error comes back as data
            return resp
        return resp

    def _read_response(self, req_id: int, deadline: float) -> dict:
        sel = selectors.DefaultSelector()
        try:
            sel.register(self._proc.stdout, selectors.EVENT_READ)
            while True:
                while b"\n" in self._buf:
                    line, self._buf = self._buf.split(b"\n", 1)
                    text = line.decode("utf-8", errors="replace").strip()
                    if not text.startswith(_RESULT_MARKER):
                        continue  # other subprocess prints (progress bars etc.): skip and keep waiting
                    try:
                        obj = json.loads(text[len(_RESULT_MARKER):])
                    except json.JSONDecodeError as exc:
                        raise ModelEnvError(f"__RESULT__ line is invalid JSON: {text[:200]}") from exc
                    if obj.get("id") == req_id:
                        return obj
                    # late response for an old id (a request abandoned after a
                    # timeout): drop it and keep waiting
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._kill_group()
                    raise ModelEnvError(
                        f"serve worker response timeout (>{timeout_s}s, deadline reached): {self.script}\n"
                        f"stderr tail: {self._stderr_tail()}"
                    )
                for key, _ in sel.select(remaining):
                    chunk = os.read(key.fd, 65536)
                    if not chunk:
                        tail = self._stderr_tail()
                        self.close()
                        raise ModelEnvError(
                            f"serve worker stdout closed unexpectedly (process exited): {self.script}\n"
                            f"stderr tail: {tail}"
                        )
                    self._buf += chunk
        finally:
            sel.close()

    def _kill_group(self) -> None:
        if self._proc is None:
            return
        try:
            os.killpg(os.getpgid(self._proc.pid), signal.SIGKILL)
        except (OSError, ProcessLookupError):
            pass
        try:
            self._proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass

    def _stderr_tail(self) -> str:
        if self._err_file is None:
            return ""
        try:
            self._err_file.seek(0)
            return self._err_file.read().decode("utf-8", errors="replace")[-_STDERR_TAIL:]
        except (OSError, ValueError):
            return ""
```

Note: the `timeout_s` variable referenced in `_read_response` comes from `_request`'s argument -- pass it into `_read_response(req_id, deadline, timeout_s)` or use `round(timeout,1)` in the error message; keep the seconds in the message correct when implementing.

- [ ] **Step 4: run the tests and confirm they pass**

```bash
RAY_ADDRESS=local python -m pytest tests/unit/test_modelenv_session.py -q -o addopts=""
```
Expected: 6 passed

- [ ] **Step 5: full regression ≥244 passed, 0 failed, commit**

```bash
git add src/roboray/models/modelenv.py tests/unit/test_modelenv_session.py
git commit -m "feat(modelenv): ModelEnvSession -- long-lived worker subprocess session

select+deadline reads the response (infer can time out), stderr goes to a temp
file to avoid pipe deadlock, start_new_session process group (cleanup kills the
whole group, no orphans), and the next infer auto-restarts after a crash (this
request is not silently retried)."
```

---

### Task 3: split the three model scripts into `build_model` / `infer_once` + `--serve`

**Files:**
- Modify: `scripts/modelenvs/run_vipe.py`, `scripts/modelenvs/run_ffs.py`, `scripts/modelenvs/run_hamer.py`
- Test: `tests/unit/test_modelenv_scripts.py` (the existing contract tests must pass unchanged -- they pin the oneshot CLI behavior)

**Interfaces:**
- Produces (same shape across the three scripts):
  - `build_model(weights: str, extra: dict) -> model`
  - `infer_once(model, input_path: str, output_path: str, extra: dict) -> dict summary` (npz written with `save_arrays`, **no** `__RESULT__` printing)
  - `main()`: the oneshot CLI behavior is unchanged (parse_args → build → infer_once → print `__RESULT__`)
  - `__main__`: `"--serve" in sys.argv` → `run_serve_loop(build_model, infer_once)`, otherwise `main()`

**Refactor rules (same shape across the three files; mostly verbatim moves):**

1. Move the whole **model-loading block** of `main()` into `build_model(weights, extra)`:
   - vipe: `DefaultAnnotationPipeline` construction (the hydra compose + make_pipeline block inside `_run_vipe` -- split `_run_vipe` into `build_model` (returns the pipeline) and `infer_once` (writes the video + run + reads payload + project_map depth + returns the summary dict))
   - ffs: `torch.load` + cfg.yaml merge + `model.cuda().eval()` block
   - hamer: `_install_numpy_shim()` + **`os.chdir(weights_dir)`** (Review Focus #2: must not stay in main) + `load_hamer(ckpt)` + `model.to(device).eval()` + fetch `model.mano`
2. Move the whole **inference block** of `main()` into `infer_once(model, input_path, output_path, extra)`:
   - input reading becomes `load_input(input_path)`; `args.extra` becomes the parameter `extra` (already a dict)
   - the trailing `save_with_result(...)` splits into `save_arrays(output, arrays)` + `return summary`
3. **`raise SystemExit(...)` → `raise ValueError(...)`** (Review Focus #1: SystemExit is not caught by the serve loop and would kill the resident process); if `main()` needs to keep the CLI exit-code semantics, wrap with `try: ... except ValueError as exc: raise SystemExit(str(exc))`
4. `parse_args` must tolerate `--serve`: it is an existing public function; change to `ap.add_argument("--serve", action="store_true", help="resident service mode")` -- oneshot calls are unaffected (default False)
5. hamer-specific: pure functions like `_boxes_from_mediapipe` / `_rotmat_to_axis_angle` stay; the `extra` key `build_model` needs is `checkpoint` (optional); the per-frame loop in `infer_once` is unchanged

**Test additions** (append to `tests/unit/test_modelenv_scripts.py`; pin that in serve mode the fakes are unusable but the **real scripts' syntax** imports cleanly):

```python
def test_real_scripts_importable_and_have_serve_symbols():
    """The three real scripts must be ast-parseable and contain the
    build_model/infer_once/main symbols.

    The real envs (torch/hamer) are not on this machine, so they cannot run; but
    "syntax correct + symbols present + --serve branch present" can be pinned with
    ast/text checks in the main env -- E2 once had a SystemExit left in the serve
    loop that killed the resident process, so this pins the minimal guarantee.
    """
    import ast

    for name in ("run_vipe.py", "run_ffs.py", "run_hamer.py"):
        src = (REPO / "scripts" / "modelenvs" / name).read_text(encoding="utf-8")
        tree = ast.parse(src)  # syntax errors explode here
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
```

(If `REPO` is not already defined in that file, add `REPO = Path(__file__).resolve().parents[2]`.)

- [ ] **Steps**: TDD -- first add the symbol test above (FAIL: symbols missing), then refactor each script (vipe → ffs → hamer, running `test_modelenv_scripts.py` after each to confirm the existing contract stays green), then full regression ≥244 passed, 0 failed, commit:

```bash
git add scripts/modelenvs/run_vipe.py scripts/modelenvs/run_ffs.py scripts/modelenvs/run_hamer.py \
        scripts/modelenvs/_common.py tests/unit/test_modelenv_scripts.py
git commit -m "refactor(modelenvs): split the three scripts into build_model/infer_once and support --serve

The oneshot CLI behavior is unchanged (pinned by the L2 smoke tests and the
existing contract tests). Every SystemExit moved onto the infer path became
ValueError -- SystemExit does not inherit Exception, so it bypasses the serve
loop's exception handling and kills the resident worker (Review Focus #1).
run_hamer's numpy shim and os.chdir(weights) moved into build_model --
CACHE_DIR_HAMER is relative to CWD, so it must also hold in serve mode
(Review Focus #2)."
```

---

### Task 4: the adapter holds the session + resident switch + e2e (echo worker through the real adapter)

**Files:**
- Modify: `src/roboray/models/scene_models.py`, `src/roboray/models/hand_models.py`, `src/roboray/ray_runtime/actors.py`, `src/roboray/api/config.py`, `src/roboray/api/reconstruction.py`
- Create: `tests/e2e/test_resident_mode.py`
- Test: same + `tests/unit/test_actors.py`

**Interfaces:**
- Produces:
  - `_SceneBase` / `HamerHandModel`: `open_session() -> float` (build a ModelEnvSession and start it, returning the elapsed time), `close_session() -> None` (idempotent), an internal `infer()` branch: `self._session is not None` → `_write_input_npz` + `self._session.infer(...)` (weights/extra built **from the same source as the oneshot argv**); otherwise the original `run_in_modelenv` path
  - `ExecutionConfig.model_worker: str = "resident"`; `reconstruction.py`'s env_vars add `ROBORAY_MODEL_WORKER`
  - GpuModelActor: `__init__` reads `os.environ.get("ROBORAY_MODEL_WORKER", "resident")` into `self.mode`; at the end of `load_model`, `if self.mode == "resident" and hasattr(self.model, "open_session"): self._load_seconds_session = self.model.open_session()`; new `unload() -> dict` (close_session + clear loaded_kind/model_name, returning `{"unloaded": kind or None}`), `idle_seconds(now) -> float`, `touch()` (update last_used), `__ray_terminate__` (calls unload); `run_task` calls `self.touch()` on the success path
  - in-process fakes (Fake* in tests/fakes/models.py): **no** `open_session` -- the actor guards with `hasattr`, so the fake path needs zero changes
- **e2e (the core acceptance of this task)**: register an echo scene adapter pointing at the **echo worker script** (a subprocess bridge, same shape as VipeSceneModel but with python=sys.executable, script=serve_worker.py, output conforming to the scene.npz contract); in a backend=test e2e with `model_worker="resident"`, run 2 Episodes × scene stage and assert **build_model is called only once** (the echo worker's builds count comes back via stage_result/summary). This proves the "weights loaded only once" end-to-end promise.

- [ ] **Steps** (TDD):
1. First write the failing test in `tests/e2e/test_resident_mode.py` (echo adapter + worker_setup adds an env-gated registration `ROBORAY_TEST_ECHO_SCENE=1` → the echo scene model registers inside the worker; the test sets that env + `model_worker="resident"`, runs 2 Episodes, and asserts both scenes' `stage_result`/summary have `builds == 1` and the second has `built_this_call is False` -- follow E1's `test_driver_workspace_isolation.py` input construction and `tests/fakes/worker_setup.py` registration pattern)
2. Run and confirm FAIL (no open_session / no model_worker field)
3. Implement the config option, env hand-down, adapter session branch, actor mode/unload/touch
4. e2e GREEN + full regression ≥246 passed, 0 failed
5. Commit:

```bash
git commit -m "feat(models): the adapter holds the ModelEnvSession -- resident mode loads weights only once

The session is held by the adapter: the payload-construction knowledge
(_write_input_npz, argv-source weights/extra) already lives in the adapter, and
the actor only drives open/close.
The in-process fakes have no open_session; the hasattr guard keeps them at zero
changes.
The e2e uses the echo worker through the real adapter subprocess bridge and
asserts builds==1 across two inferences."
```

---

### Task 5: idle policy pure function + driver three-tier idling

**Files:**
- Create: `src/roboray/scheduling/idle.py`
- Modify: `src/roboray/runtime/driver.py`, `src/roboray/api/config.py`
- Test: `tests/unit/test_idle_policy.py`

**Interfaces:**
- Produces:
  - `@dataclass IdleState: actor: object; slot: object; idle_seconds: float; loaded: bool`
  - `compute_idle_actions(states: list[IdleState], now: float, unload_s: float, exit_s: float) -> list[tuple[str, IdleState]]`
    Returns `[("unload", st), ...]` and `[("exit", st), ...]`; **only `loaded` non-busy slots**; threshold semantics: `idle_seconds >= unload_s` → unload; `idle_seconds >= exit_s` → exit (exit implies unload first); `unload_s=0` means "never unload proactively" (only on role switch/run end)
  - `ExecutionConfig.actor_idle_unload_s: float = 30.0`, `actor_idle_exit_s: float = 240.0`
- driver application:
  - `_run_pipeline` main loop, each round: for `zip(self.slots, self.gpu_actors)` where `not slot.busy`, `ray.get(actor.idle_seconds.remote(), timeout=2)` (treat exception/death as 0) → `compute_idle_actions` → unload actions `ray.get(actor.unload.remote())` (timeline records `{"event":"unload",...}`); exit actions unload first then `ray.kill(actor)`, and the slot is permanently marked `busy=True` (**not** removed from the list -- slot_id is the index into gpu_actors, so removing would misalign; a dead slot is never dispatched again)
  - at the end of `run()` (before both the serial and pipeline paths return): best-effort `unload()` on all GPU actors (swallow exceptions; cleanup must not raise)
- [ ] **Steps** (TDD):
1. `tests/unit/test_idle_policy.py`: test the pure function directly with fake `now` values -- empty list → []; idle=29s/unload=30 → no action; idle=30 → unload; idle=240 → exit (and no repeated unload); a busy slot never appears; unload_s=0 → no unload; explicitly assert the ascending-threshold relation 30<240 in the test (`assert UNLOAD_DEFAULT < EXIT_DEFAULT`)
2. Run FAIL → implement `idle.py` (pure function, no ray dependency) → GREEN
3. Wire up the driver (scan + end-of-run unload + timeline events) + config options
4. Full regression ≥247 passed, 0 failed (e2e does not regress: with no local GPU the scan idles out)
5. Commit:

```bash
git commit -m "feat(scheduling): three-tier idling -- compute_idle_actions pure function + driver application

Policy and mechanism are separated: idle determination is extracted into a pure
function testable with a fake clock; the driver only applies actions.
exit uses ray.kill but unloads first (ray.kill is SIGKILL and runs no cleanup,
leaving orphan children that keep holding VRAM -- spec §5.6); a killed slot stays
busy permanently and is not removed from the list (slot_id is the index into
gpu_actors; removing would misalign)."
```

---

### Task 6: CPU-stage node routing (E1 handoff Important-2)

**Files:**
- Modify: `src/roboray/runtime/driver.py`
- Test: `tests/unit/test_node_pools.py` (append), `tests/e2e/test_public_api.py` (regression)

**Interfaces:**
- Produces:
  - driver `__init__`: `_register_existing_pools` changes to **ask each CPU actor for its node_id** (`ray.get([a.node_id.remote() for a in cpu_actors])`) and group into `self._cpu_by_node: dict[str, list[actor]]` -- fixes the E1 leftover of "the same shared CPU-actor reference stuffed into every pool"; CPU actors created by `_create_pool` also join their group by actual node_id
  - `_cpu_actor_for(episode_id: str) -> actor | None`: take the Episode's `assigned_node` (record) → look for a free one in `self._cpu_by_node.get(node)` (tracked via `self._cpu_busy: set[int]`, keyed by id(actor)); in a GPU-less environment (registry None) it degrades to any globally free CPU actor (Review Focus #5)
  - the two CPU dispatch points (`_run_pipeline` section 2, `_execute_one`'s else branch) go through `_cpu_actor_for`
- [ ] **Steps** (TDD):
1. Append to `test_node_pools.py`: with a fake CPU actor list, `_cpu_by_node` groups correctly; when the Episode belongs to node B only the B group is used; when B's group is fully busy → return None (that Episode's CPU stage waits for the next round, **must not** borrow from another node -- the workspace is node-local); GPU-less mode returns any free actor
2. FAIL → implement → GREEN; full regression ≥248 passed, 0 failed
3. Commit:

```bash
git commit -m "feat(driver): route CPU stages by the Episode's node (closing E1 Important-2)

E1 only delivered GPU-side pinning; CPU popped from the global free_cpu, so under
multi-node fusion could run on an off-node actor that cannot read the node-local
prepared/. Now CPU actors are grouped by node_id (fixing the shared-reference
problem) and dispatch is constrained to the Episode's assigned_node;
GPU-less environments degrade to a globally free actor (e2e does not regress)."
```

---

### Task 7: regression wrap-up + docs

**Files:**
- Modify: `docs/models.md`, `deploy/modelenvs/build_*.sh` (trailing hint text), `README.md` (add `ROBORAY_MODEL_WORKER` to the env-var table)
- Test: full

- [ ] **Step 1**: full `RAY_ADDRESS=local ... -m pytest -q --tb=short` ≥248 passed, 0 failed
- [ ] **Step 2**: docs, three places -- `docs/models.md` gets a "resident worker mode" section (protocol, resident default, oneshot fallback, the relation between serve and the L2 smoke CLI); the three build scripts' trailing echo hints gain a line "the script supports --serve resident mode, controlled by ROBORAY_MODEL_WORKER"; `README.md`'s env-var table adds `ROBORAY_MODEL_WORKER` (resident|oneshot, default resident)
- [ ] **Step 3**: commit `docs(e2): resident worker mode documentation`

---

## Self-Review

**1. Spec coverage (spec §5 item by item)**

| spec item | covering task |
|---|---|
| 5.2 protocol (hello/infer/ping/shutdown, failure does not kill the process) | Task 1 |
| 5.3 three-way script split | Task 3 |
| 5.4 ModelEnvSession (incl. crash restart, timeout) | Task 2 |
| 5.5 actor unload/idle_seconds/load_seconds/touch/__ray_terminate__ | Task 4 |
| 5.6 three-tier idling + driver scan + end-of-run unload + timeline | Task 5 |
| 5.7 model_worker switch + oneshot equivalence | Task 4 (switch) + regression in every task |
| §8 config options model_worker/actor_idle_unload_s/actor_idle_exit_s | Task 4/5 |
| E1 handoff Important-2 (CPU node routing) | Task 6 |
| 9.1 verification (fake-worker protocol tests, fake clock, oneshot regression) | Task 1/2/5 + regression in every task |

**2. Placeholder scan**: Task 3's three-file refactor is expressed as "move rules + precise source/target" (the inference bodies move verbatim; it is neither possible nor desirable to repeat 600 lines of inference code in this plan); the other tasks give complete code.

**3. Type consistency**: `run_serve_loop(build_model, infer_once)` / `ModelEnvSession.infer(input_path, output_path, weights, extra, timeout_s)` / `open_session() -> float` / `unload() -> dict` / `compute_idle_actions(states, now, unload_s, exit_s)` / `_cpu_actor_for(episode_id)` -- definitions and consumers agree across tasks.

**4. Review Focus**: the five map to Task 1 (SystemExit catch boundary in the serve-loop comment + Task 3 AST test), Task 3 (shim/chdir), Task 2 (crash restart), Task 5 (busy is the single authority), Task 7 (GPU-less degradation).

**Known uncovered (registered)**:
- Measuring the resident benefit on the real models (builds==1 on real ViPE/HaMeR) needs an L2 run on the A800 cluster -- this plan only does the echo e2e; spec 9.2's "measured VRAM curves" remain unverified.
- After `idle_exit` kills an actor mid-run, whether that node's pool gets a replacement is decided by E3's autoscaler (E1 already has an on-demand pool-creation path, so it can naturally refill).
