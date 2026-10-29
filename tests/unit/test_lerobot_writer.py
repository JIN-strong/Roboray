import os
import sys
from pathlib import Path

import numpy as np
import pytest

from roboray.export.lerobot_writer import LeRobotWriter

FAKES = Path(__file__).resolve().parents[1] / "fakes" / "modelenv_scripts"


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("ROBORAY_LEROBOT_PYTHON", sys.executable)
    yield


def test_writer_creates_dataset_dir_via_fake(tmp_path):
    writer = LeRobotWriter(script=str(FAKES / "fake_lerobot_export.py"))
    t = 6
    state = np.zeros((t, 126), np.float32)
    action = np.ones((t, 126), np.float32)
    info = writer.write("ep_0001", state, action, tasks=["reach"], fps=30, out_dir=tmp_path / "ds")
    assert info["n_frames"] == t
    ds = tmp_path / "ds"
    assert (ds / "meta" / "info.json").exists()
    assert (ds / "data").is_dir()
    assert (ds / "meta" / "episodes").is_dir()


def test_writer_rejects_mismatched_lengths(tmp_path):
    writer = LeRobotWriter(script=str(FAKES / "fake_lerobot_export.py"))
    with pytest.raises(ValueError) as exc:
        writer.write("ep", np.zeros((4, 126), np.float32), np.zeros((3, 126), np.float32),
                     tasks=["t"], fps=30, out_dir=tmp_path / "ds")
    assert "length" in str(exc.value) or "shape" in str(exc.value)


def test_writer_missing_env_gives_build_hint(tmp_path, monkeypatch):
    from roboray.models.modelenv import ModelEnvError

    monkeypatch.setenv("ROBORAY_LEROBOT_PYTHON", "/nonexistent/python")
    writer = LeRobotWriter(script=str(FAKES / "fake_lerobot_export.py"))
    with pytest.raises(ModelEnvError) as exc:
        writer.write("ep", np.zeros((2, 126), np.float32), np.zeros((2, 126), np.float32),
                     tasks=["t"], fps=30, out_dir=tmp_path / "ds")
    assert "build_lerobot" in str(exc.value)
