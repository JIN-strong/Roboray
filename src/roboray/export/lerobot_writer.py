"""LeRobot v3 export: calls the lerobot public API in a subprocess using a separate env.

lerobot hard-depends on torch>=2.7 and numpy>=2.0, which conflicts with the main
env's numpy<2.0, so it runs in a subprocess (the same pattern as the model adapters).
"""

import os
import tempfile
from pathlib import Path

import numpy as np

from roboray.models.modelenv import ModelEnvError, run_in_modelenv

__all__ = ["LeRobotWriter"]

_DEFAULT_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "modelenvs" / "run_lerobot_export.py"


class LeRobotWriter:
    def __init__(self, script: str | None = None, env_var: str = "ROBORAY_LEROBOT_PYTHON"):
        # Script resolution priority: explicit script > ROBORAY_LEROBOT_SCRIPT >
        # the real script in the repo. The env var lets publish swap the export
        # implementation inside Ray workers (e2e uses a fake script), same source as
        # the model adapters' `script=` injection (publish cannot swap via registry).
        self.script = script or os.environ.get("ROBORAY_LEROBOT_SCRIPT") or str(_DEFAULT_SCRIPT)
        self.env_var = env_var

    def write(self, episode_id, state, action, tasks, fps, out_dir) -> dict:
        state = np.asarray(state, np.float32)
        action = np.asarray(action, np.float32)
        if state.shape[0] != action.shape[0]:
            raise ValueError(
                f"state/action length mismatch: state shape {state.shape} vs action shape {action.shape}"
            )
        if state.ndim != 2 or action.ndim != 2 or state.shape[1] != action.shape[1]:
            raise ValueError(f"state/action must be same-width 2D (T,D): {state.shape} / {action.shape}")
        python = os.environ.get(self.env_var, "python")
        if python != "python" and not Path(python).exists():
            raise ModelEnvError(
                f"{self.env_var} points to a non-existent interpreter: {python}. "
                "Build the export env with deploy/modelenvs/build_lerobot.sh."
            )
        out_dir = Path(out_dir)
        out_dir.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as td:
            inp = Path(td) / "export_in.npz"
            np.savez(inp, state=state, action=action,
                     task_str=np.array(tasks[0] if tasks else "unknown"))
            argv = ["--input", str(inp), "--output", str(out_dir),
                    "--extra", _json({"repo_id": episode_id, "fps": int(fps),
                                      "task": (tasks[0] if tasks else "unknown")})]
            summary = run_in_modelenv(python, self.script, argv, timeout_s=3600.0)
        if not summary.get("ok"):
            raise ModelEnvError(f"LeRobot export failed: {summary}")
        if not (out_dir / "meta" / "info.json").exists():
            raise ModelEnvError(f"LeRobot export did not produce meta/info.json: {out_dir}")
        return {"dataset_dir": str(out_dir), "n_frames": int(summary["n_frames"]),
                "n_episodes": int(summary.get("n_episodes", 1))}


def _json(obj) -> str:
    import json

    return json.dumps(obj, ensure_ascii=False)
