"""CLI contract: config read/parse/validation failure -> rc 2 (no Ray needed, pure
main() invocation)."""

from roboray.cli import main


def test_cli_missing_config_file_returns_2(tmp_path):
    rc = main(["run", "--config", str(tmp_path / "nope.yaml")])
    assert rc == 2


def test_cli_malformed_yaml_returns_2(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(":: not: valid: yaml: [[[")
    rc = main(["run", "--config", str(bad)])
    assert rc == 2


def test_mode_flags_reach_reconstruction(monkeypatch, tmp_path):
    """Mode-flag drift guard (spec §7): --serial-baseline/--no-adaptive must
    propagate to the API."""
    import types

    import yaml

    from roboray.cli import main

    captured = {}

    class _FakeRun:
        def __init__(self, request, run_root=None, serial_baseline=False,
                     adaptive=True, ray_address=None):
            captured["serial_baseline"] = serial_baseline
            captured["adaptive"] = adaptive

        def run(self):
            return types.SimpleNamespace(total_seconds=0.0, failed=[],
                                         manifest_path=None, timeline=[])

    monkeypatch.setattr("roboray.cli.EgoReconstruction", _FakeRun)
    cfg = tmp_path / "c.yaml"
    cfg.write_text(yaml.safe_dump({
        "input": {"uri": "/tmp/in"},
        "output": {"uri": "/tmp/out", "format": "lerobot_v3"},
    }))
    rc = main(["run", "--config", str(cfg), "--serial-baseline"])
    assert rc == 0
    assert captured == {"serial_baseline": True, "adaptive": True}
