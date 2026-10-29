"""Model subprocess environment: each real model runs in its own conda env.

The main env (roboray) does not install torch; adapters call
scripts/modelenvs/run_*.py through run_in_modelenv (executed by the model env's
python), exchanging NPZ files, with the last stdout line returning a
__RESULT__{json} summary.
"""

import json
import os
import selectors
import signal
import subprocess
import tempfile
import time
from pathlib import Path

from roboray.api.config import ModelEnvsConfig

__all__ = ["ModelEnvError", "run_in_modelenv", "resolve_env_request",
           "ModelEnvSession"]

_RESULT_MARKER = "__RESULT__"
_STDERR_TAIL = 2048


class ModelEnvError(RuntimeError):
    """Model subprocess failure (non-zero exit / timeout / missing protocol)."""


def resolve_env_request(models: ModelEnvsConfig) -> dict:
    """Flatten the model env config into runtime_env env_vars. Empty string = unset."""
    return {
        "ROBORAY_VIPE_PYTHON": models.vipe_python or "python",
        "ROBORAY_FFS_PYTHON": models.ffs_python or "python",
        "ROBORAY_HAMER_PYTHON": models.hammer_python or "python",
        "ROBORAY_VIPE_WEIGHTS": models.vipe_weights or "",
        "ROBORAY_FFS_WEIGHTS": models.ffs_weights or "",
        "ROBORAY_HAMER_WEIGHTS": models.hammer_weights or "",
        # LeRobot export env: read on the worker side by LeRobotWriter. It must ride
        # along with runtime_env — a multi-node cluster does not inherit the driver's
        # process environment (ROBORAY_LEROBOT_SCRIPT empty string = the real script).
        "ROBORAY_LEROBOT_PYTHON": models.lerobot_python or "python",
        "ROBORAY_LEROBOT_SCRIPT": models.lerobot_script or "",
    }


def run_in_modelenv(python: str, script: str, argv: list, timeout_s: float = 3600.0) -> dict:
    """Run an inference script in the model env; stdout must contain a __RESULT__{json} summary line."""
    cmd = [python, script, *argv]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise ModelEnvError(f"model subprocess timed out (> {timeout_s}s): {script}") from exc
    if proc.returncode != 0:
        tail = (proc.stderr or "")[-_STDERR_TAIL:]
        raise ModelEnvError(
            f"model subprocess failed (exit code {proc.returncode}): {script}\nstderr tail:\n{tail}"
        )
    for line in reversed((proc.stdout or "").splitlines()):
        if line.startswith(_RESULT_MARKER):
            try:
                return json.loads(line[len(_RESULT_MARKER):])
            except json.JSONDecodeError as exc:
                raise ModelEnvError(f"__RESULT__ line is not valid JSON: {line[:200]}") from exc
    raise ModelEnvError(
        f"model subprocess output is missing the {_RESULT_MARKER} summary line: {script}\nstdout tail:\n{(proc.stdout or '')[-_STDERR_TAIL:]}"
    )


class ModelEnvSession:
    """A long-lived worker subprocess session in the model env (spec §5.4).

    Why stderr goes to a temp file instead of a PIPE: nobody reads stderr while the
    session is resident, so once the child fills the pipe buffer it blocks entirely;
    a temp file has no such risk and we read its tail on error.
    Why stdout is binary + select: mixing text=True's internal buffering with select
    loses lines / deadlocks; manual line buffering keeps the "select until deadline"
    semantics clean.
    """

    def __init__(self, python: str, script: str, timeout_s: float = 3600.0):
        self.python = python
        self.script = script
        self.timeout_s = timeout_s
        self._proc: subprocess.Popen | None = None
        self._err_file = None
        self._buf = b""
        self._next_id = 1

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        self._err_file = tempfile.TemporaryFile()
        self._proc = subprocess.Popen(
            [self.python, "-u", self.script, "--serve"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._err_file,
            start_new_session=True,  # own process group: cleanup kills the whole group, no orphans
        )
        self._buf = b""
        # Handshake: the serve loop emits {"id":0,"ready":true} on startup (hello
        # inside the loop is a no-op that returns no line), consume this id=0 ready
        # line directly; later request ids start at 1 so they never collide.
        # On failure (timeout/EOF) always close() to clean up _proc/_err_file before
        # re-raising — otherwise a start() exception leaks the temp file and the bare
        # process handle.
        try:
            ready = self._read_response(req_id=0, deadline=time.monotonic() + 60.0,
                                        timeout_s=60.0)
        except Exception:
            self.close()
            raise
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
            pass  # process group is already gone
        finally:
            if self._proc.stdout:
                self._proc.stdout.close()
            if self._err_file:
                self._err_file.close()
                self._err_file = None
            self._proc = None
            self._buf = b""

    # -- requests -----------------------------------------------------------
    def ping(self) -> bool:
        if not self.alive:
            return False
        try:
            return bool(self._request({"cmd": "ping"}, timeout_s=10.0).get("ok"))
        except (ModelEnvError, OSError):
            return False

    def infer(self, input_path: str, output_path: str, weights: str = "",
              extra: str = "{}", timeout_s: float | None = None) -> dict:
        """Run one inference. If the child is dead, **restart it first** before running
        (the request that hit the crash already errored in the previous call; this one
        is not silently retried)."""
        if not self.alive:
            self.close()  # clean up remains (err file, etc.)
            self.start()
        req = {"cmd": "infer", "input": str(input_path), "output": str(output_path),
               "weights": weights, "extra": extra}
        return self._request(req, timeout_s=timeout_s if timeout_s is not None else self.timeout_s)

    # -- internals ----------------------------------------------------------
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
        resp = self._read_response(req_id, deadline=time.monotonic() + timeout_s,
                                   timeout_s=timeout_s)
        if not resp.get("ok", True):
            # Protocol-level failure (exception inside the script): the process is
            # still alive, the error is returned as data
            return resp
        return resp

    def _read_response(self, req_id: int, deadline: float, timeout_s: float) -> dict:
        sel = selectors.DefaultSelector()
        try:
            sel.register(self._proc.stdout, selectors.EVENT_READ)
            while True:
                while b"\n" in self._buf:
                    line, self._buf = self._buf.split(b"\n", 1)
                    text = line.decode("utf-8", errors="replace").strip()
                    if not text.startswith(_RESULT_MARKER):
                        continue  # other child prints (progress bars, etc.): skip and keep waiting
                    try:
                        obj = json.loads(text[len(_RESULT_MARKER):])
                    except json.JSONDecodeError as exc:
                        raise ModelEnvError(f"__RESULT__ line is invalid JSON: {text[:200]}") from exc
                    if obj.get("id") == req_id:
                        return obj
                    # Late response from an old id (a request abandoned on timeout): drop and keep waiting
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._kill_group()
                    raise ModelEnvError(
                        f"serve worker response timed out (> {timeout_s}s, deadline reached): {self.script}\n"
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
