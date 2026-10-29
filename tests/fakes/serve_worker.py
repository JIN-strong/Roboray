"""Echo worker for serve-protocol testing: implements the minimal build_model/infer_once contract.

- build_model increments a counter each call; the infer response carries the cumulative
  build count and whether this call rebuilt (for "builds only once" assertions).
- infer: reads x from the input npz, outputs y = x * factor (extra may pass factor, default 2).
- When ``extra["contract"] == "scene"`` it writes a **valid scene-contract npz**
  (camera_poses/intrinsics/depth/depth_source/n_frames/fps, geometry taken from the input
  frames) and stores the build evidence as ``builds``/``built_flag`` arrays - so e2e can
  assert "two inferences build only once" directly in the artifact, **outside the response
  side**. The default contract is unchanged, so the protocol tests are unaffected.
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "modelenvs"))

from _common import load_input, run_serve_loop, save_arrays  # noqa: E402

_LAST_SEEN = {"builds": 0}


def build_model(weights: str, extra: dict) -> dict:
    return {"builds": _LAST_SEEN["builds"] + 1, "weights": weights,
            "factor": float(extra.get("factor", 2.0))}


def _scene_arrays(data: dict) -> dict:
    """Wrap the echo worker's output into a valid scene contract (camera_poses/intrinsics/...)."""
    frames = np.asarray(data["frames"])
    t = int(data["n_frames"]) if "n_frames" in data else frames.shape[0]
    h, w = frames.shape[1], frames.shape[2]
    return {
        "camera_poses": np.tile(np.eye(4, dtype=np.float32), (t, 1, 1)),
        "intrinsics": np.tile(np.eye(3, dtype=np.float32), (t, 1, 1)),
        "depth": np.full((t, h, w), 0.5, np.float32),
        "depth_source": np.array("monocular"),
        "n_frames": np.int32(t),
        "fps": data["fps"],
    }


def infer_once(model: dict, input_path: str, output_path: str, extra: dict) -> dict:
    sleep_s = float(extra.get("sleep_s", 0.0))
    if sleep_s > 0:
        time.sleep(sleep_s)  # test-only: simulate a slow inference for ModelEnvSession timeout cases
    data = load_input(input_path)
    # built_this_call: whether a model build happened this call (build count advanced = new build)
    built_this_call = model["builds"] != _LAST_SEEN["builds"]
    if extra.get("contract") == "scene":
        arrays = _scene_arrays(data)
        # Write evidence into the artifact: e2e asserts directly from scene.npz (not via actor/response side).
        arrays["builds"] = np.array(model["builds"], np.int32)
        arrays["built_flag"] = np.array(built_this_call, np.bool_)
        save_arrays(output_path, arrays)
        summary = {"ok": True, "contract": "scene", "n_frames": int(data["n_frames"]),
                   "built_this_call": built_this_call, "builds": model["builds"]}
    else:
        y = data["x"] * model["factor"]
        save_arrays(output_path, {"y": y})
        summary = {"n": int(y.shape[0]), "built_this_call": built_this_call,
                   "builds": model["builds"]}
    _LAST_SEEN["builds"] = model["builds"]
    return summary


if __name__ == "__main__":
    assert "--serve" in sys.argv, "echo worker only supports --serve mode"
    run_serve_loop(build_model, infer_once)
