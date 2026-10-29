"""Shared contract for the model-env inference scripts (run inside a model env; may only depend on that env's packages)."""

import argparse
import json
import sys

import numpy as np

RESULT_MARKER = "__RESULT__"


def compute_stride(fps, inference_fps) -> int:
    """Hand-inference sampling stride: stride = round(fps / inference_fps), min 1.

    Missing/invalid fps or inference_fps -> 1 (all frames; byte-identical to the
    pre-change behavior). Skipped frames get hand_present=False and are filled by the
    short-gap completion (gaps up to 6 frames are fully filled; stride>7 gaps are kept
    as-is - better missing than wrong).
    """
    try:
        if not fps or not inference_fps:
            return 1
        return max(1, int(round(float(fps) / float(inference_fps))))
    except (TypeError, ValueError, ZeroDivisionError):
        return 1


def parse_args(description: str) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--input", required=True, help="input NPZ (frames + metadata)")
    ap.add_argument("--output", required=True, help="output artifact NPZ")
    ap.add_argument("--weights", default="", help="model weights directory")
    ap.add_argument("--extra", default="{}", help="extra parameters as JSON")
    ap.add_argument("--serve", action="store_true", help="resident server mode")
    return ap.parse_args()


def load_input(path: str) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def save_with_result(path: str, arrays: dict, summary: dict) -> None:
    np.savez(path, __schema_version__=1, **arrays)
    print(RESULT_MARKER + json.dumps(summary))
    sys.stdout.flush()


def save_arrays(path: str, arrays: dict) -> None:
    """Write the npz (with schema stamp) but do **not** print __RESULT__ - responses are emitted by the serve loop."""
    np.savez(path, __schema_version__=1, **arrays)


def run_serve_loop(build_model, infer_once) -> None:
    """Resident serve loop (--serve mode). Protocol: see the module docstring and spec §5.2.

    - The model is lazily built once, keyed by ``(weights, extra)``; only rebuilt when the key changes.
    - Any Exception from infer (including ValueError) just returns an error line - one bad input
      must not drop a loaded model. Note SystemExit is not caught: code moved into the infer path
      must not use SystemExit (it does not inherit from Exception and would kill the resident process).
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
            _emit({"id": req_id, "ok": True})
            return
        if cmd == "ping":
            _emit({"id": req_id, "ok": True, "uptime_s": round(time.time() - start, 3)})
            continue
        if cmd == "hello":
            # The ready response was already emitted at startup as {"id":0,"ready":true};
            # hello is idempotent, so do not emit another line (it would pollute the response stream).
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
        except Exception as exc:  # noqa: BLE001 - see docstring: a failure must not kill the process
            _emit({"id": req_id, "ok": False,
                   "error": f"{type(exc).__name__}: {exc}"})
