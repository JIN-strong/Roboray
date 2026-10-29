import json
import sys
import textwrap

import pytest

from roboray.models.modelenv import ModelEnvError, resolve_env_request, run_in_modelenv
from roboray.api.config import ModelEnvsConfig


def test_resolve_env_request_fills_defaults():
    env = resolve_env_request(ModelEnvsConfig())
    assert env["ROBORAY_VIPE_PYTHON"] == "python"
    assert env["ROBORAY_FFS_PYTHON"] == "python"
    assert env["ROBORAY_HAMER_PYTHON"] == "python"
    assert env["ROBORAY_VIPE_WEIGHTS"] == ""
    # The export env must ride along with runtime_env (otherwise multi-node workers
    # never get it -> export silently degrades)
    assert env["ROBORAY_LEROBOT_PYTHON"] == "python"
    assert env["ROBORAY_LEROBOT_SCRIPT"] == ""


def test_resolve_env_request_passes_values():
    env = resolve_env_request(ModelEnvsConfig(vipe_python="/x/py", ffs_weights="/w"))
    assert env["ROBORAY_VIPE_PYTHON"] == "/x/py"
    assert env["ROBORAY_FFS_WEIGHTS"] == "/w"
    assert env["ROBORAY_VIPE_WEIGHTS"] == ""


def test_resolve_env_request_passes_lerobot_values():
    env = resolve_env_request(
        ModelEnvsConfig(lerobot_python="/opt/ego-lerobot/bin/python", lerobot_script="/tmp/fake.py")
    )
    assert env["ROBORAY_LEROBOT_PYTHON"] == "/opt/ego-lerobot/bin/python"
    assert env["ROBORAY_LEROBOT_SCRIPT"] == "/tmp/fake.py"


def test_run_in_modelenv_parses_result_line(tmp_path):
    script = tmp_path / "ok.py"
    script.write_text(
        textwrap.dedent(
            """
            import json, sys
            args = sys.argv[1:]
            print("noise line")
            print("__RESULT__" + json.dumps({"ok": True, "argv": args}))
            """
        )
    )
    res = run_in_modelenv(sys.executable, str(script), ["a", "b"], timeout_s=60.0)
    assert res["ok"] is True
    assert res["argv"] == ["a", "b"]


def test_run_in_modelenv_nonzero_exit_raises_with_stderr(tmp_path):
    script = tmp_path / "bad.py"
    script.write_text("import sys\nprint('boom-reason', file=sys.stderr)\nsys.exit(3)\n")
    with pytest.raises(ModelEnvError) as exc:
        run_in_modelenv(sys.executable, str(script), [], timeout_s=60.0)
    assert "boom-reason" in str(exc.value)
    assert "exit code 3" in str(exc.value)


def test_run_in_modelenv_timeout_raises(tmp_path):
    script = tmp_path / "slow.py"
    script.write_text("import time\ntime.sleep(30)\n")
    with pytest.raises(ModelEnvError) as exc:
        run_in_modelenv(sys.executable, str(script), [], timeout_s=1.0)
    assert "timed out" in str(exc.value)


def test_run_in_modelenv_missing_result_line_raises(tmp_path):
    script = tmp_path / "silent.py"
    script.write_text("print('no marker here')\n")
    with pytest.raises(ModelEnvError) as exc:
        run_in_modelenv(sys.executable, str(script), [], timeout_s=60.0)
    assert "__RESULT__" in str(exc.value)
