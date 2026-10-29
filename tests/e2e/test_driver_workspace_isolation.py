"""Hard invariant: **the driver process never accesses any file under the workspace**.

This invariant is the core promise of the node-local workspace design - under
multi-node the driver runs on the head/CPU worker and cannot read node-local
workspaces. All the routing and record changes serve this, and this test is the
end-to-end acceptance check.
Approach: wrap ``Path.read_bytes`` / ``Path.read_text`` / ``np.load`` inside the
driver process, recording every access under ``workspace_root``; run a full fake e2e;
assert the access count is 0.
Note: Ray actors run in **separate processes**, so their disk reads do not pass
through this patch - which is exactly what we want to verify: all disk reads
should happen inside the actor processes.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from roboray.api.config import ReconstructionRequest
from roboray.api.reconstruction import EgoReconstruction

pytestmark = pytest.mark.slow


def _make_input(root: Path, n_episodes: int = 2, n_frames: int = 4) -> Path:
    """Build a minimal input: only the mp4 + calibration + meta that prepare needs."""
    import cv2

    root.mkdir(parents=True, exist_ok=True)
    for e in range(n_episodes):
        ep = root / f"ep_{e + 1:04d}"
        ep.mkdir(parents=True, exist_ok=True)
        w, h = 64, 48
        writer = cv2.VideoWriter(
            str(ep / "rgb_left.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h)
        )
        for i in range(n_frames):
            writer.write(np.full((h, w, 3), 40 + i, np.uint8))
        writer.release()
        (ep / "calibration.json").write_text(json.dumps({
            "K": [[51.2, 0.0, w / 2], [0.0, 51.2, h / 2], [0.0, 0.0, 1.0]],
            "dist": [0.0] * 5, "T_left_to_right": np.eye(4).tolist(),
            "fps": 30.0, "width": w, "height": h,
        }))
        (ep / "meta.json").write_text(json.dumps(
            {"episode_id": f"ep_{e + 1:04d}", "device": "test", "n_frames": n_frames}
        ))
    return root


def test_driver_never_reads_workspace_files(tmp_path, monkeypatch):
    """The driver process's actual access count for files under workspace_root must be 0."""
    storage_root = _make_input(tmp_path / "input")
    workspace_root = tmp_path / "run"
    output_root = tmp_path / "out"

    hits: list[str] = []
    ws_prefix = str(workspace_root.resolve())

    real_read_bytes = Path.read_bytes
    real_read_text = Path.read_text
    real_np_load = np.load

    def _note(p) -> None:
        try:
            resolved = str(Path(p).resolve())
        except OSError:
            return
        if resolved.startswith(ws_prefix):
            hits.append(resolved)

    def _read_bytes(self):
        _note(self)
        return real_read_bytes(self)

    def _read_text(self, *a, **k):
        _note(self)
        return real_read_text(self, *a, **k)

    def _np_load(file, *a, **k):
        _note(file)
        return real_np_load(file, *a, **k)

    monkeypatch.setattr(Path, "read_bytes", _read_bytes)
    monkeypatch.setattr(Path, "read_text", _read_text)
    monkeypatch.setattr(np, "load", _np_load)

    request = ReconstructionRequest.from_dict({
        "input": {"uri": str(storage_root)},
        "output": {"uri": str(output_root), "format": "lerobot_v3"},
        "execution": {"max_gpus": 1, "num_cpu_actors": 2,
                      "max_episodes_in_flight": 2, "backend": "test"},
        "stages": {
            "scene_reconstruction": {"model": "fake_scene"},
            "hand_reconstruction": {"model": "fake_hand"},
            "action_understanding": {"model": "fake_vlm"},
            "visualization": {"camera_view": True, "world_view": True},
        },
        "models": {
            "lerobot_python": "python",
            "lerobot_script": "tests/fakes/modelenv_scripts/fake_lerobot_export.py",
        },
    })
    result = EgoReconstruction(request, run_root=str(workspace_root)).run()
    assert not result.failed, f"e2e failed: {result.failed}"

    assert hits == [], (
        "the driver process accessed workspace files (hard invariant broken):\n  "
        + "\n  ".join(hits[:10])
    )
