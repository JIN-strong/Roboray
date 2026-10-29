"""Scene adapter for the echo worker (**subprocess bridge**, isomorphic to VipeSceneModel).

Test-only: the worker process registers itself as the ``echo_scene`` scene model when
``ROBORAY_TEST_ECHO_SCENE=1``. It inherits ``_SceneBase`` and thus naturally gets
open_session/close_session and the session-branch infer() - e2e uses it under a real
Ray actor + real subprocess bridge to prove "resident mode builds the model only once
across two inferences".
``python`` uses the current interpreter (sys.executable, no torch dependency); ``script``
points to ``tests/fakes/serve_worker.py``; the worker's ``infer_once`` writes a valid
scene-contract npz when ``extra={"contract": "scene"}`` (see serve_worker).
"""

import sys
from pathlib import Path

from roboray.models.scene_models import _SceneBase, _write_input_npz

__all__ = ["EchoSceneModel"]


class EchoSceneModel(_SceneBase):
    name = "echo_scene"
    _script_name = "serve_worker.py"  # placeholder: overwritten in __init__

    def __init__(self, script: str | None = None):
        super().__init__(script or str(Path(__file__).resolve().parent / "serve_worker.py"))
        self._session = None

    def _sec_per_frame(self, width: int) -> float:
        return 0.0

    # -- session contract (_SceneBase provides open/close/infer branches; only details here) --
    def _env_python(self) -> str:
        """The echo bridge needs no separate conda env: run the current interpreter."""
        return sys.executable

    def _weights(self) -> str:
        return ""

    def _extra(self, calib: dict) -> dict:
        return {"contract": "scene"}

    def _write_input(self, path: Path, frames: dict, calib: dict) -> None:
        _write_input_npz(path, frames, calib)
