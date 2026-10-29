import json
import warnings
from pathlib import Path

import numpy as np

from roboray.runtime.episode import EpisodeWorkspace
from roboray.stages.publish import run_publish
from roboray.viz.qc_plots import plot_reprojection_curve

FLAG = 1e6
FAKES = Path(__file__).resolve().parents[1] / "fakes" / "modelenv_scripts"


def _ws_with_artifacts(tmp_path, t=8):
    ws = EpisodeWorkspace(tmp_path, "ep_0001")
    ws.write_artifact("fused.npz", {
        "joints_world": np.zeros((t, 2, 21, 3), np.float32),
        "reprojection_error": np.array([[1.0, 2.0]] * t, np.float32),
        "depth_source": np.array(["stereo"] * t, dtype=object),
        "identity_switches": np.zeros((0, 2), np.int32),
        "gapfill_filled": np.int32(0), "gapfill_rejected": np.int32(1),
        "gapfill_reasons": np.array(json.dumps({"insufficient_keypoint_depth": 1})),
        "gapfill_skipped": np.array(json.dumps({})),
        "joint_source": np.array("mano"),
        "n_frames": np.int32(t), "fps": np.int32(30),
    })
    ws.write_artifact("action.npz", {
        "labels": np.array(["reach"], dtype=object),
        "segment_start": np.array([0], np.int32), "segment_end": np.array([t], np.int32),
        "state": np.zeros((t, 2, 21, 3), np.float32), "action": np.zeros((t, 2, 21, 3), np.float32),
        "n_frames": np.int32(t), "fps": np.int32(30),
    })
    np.save(Path(ws.prepared_dir) / "frames_left.npy", np.zeros((t, 16, 16, 3), np.uint8))
    (Path(ws.dir) / "state.json").write_text(json.dumps({"timings": {"fusion": 0.1}, "error": None}))
    return ws


def test_curve_plot_marks_flagged_frames(tmp_path):
    err = np.array([[1.0, 1.0], [FLAG, 1.0], [3.0, 1.0]], np.float32)
    out = tmp_path / "c.png"
    res = plot_reprojection_curve(err, out, flag_value=FLAG)
    assert res["plotted"] is True
    # The real count comes from the input array: of the 6 hand frames exactly 1 is a
    # sentinel, the other 5 are trusted
    assert (res["trusted_frames"], res["flagged_frames"]) == (5, 1)
    assert out.exists() and out.stat().st_size > 0
    # Flagged frames really change the drawn content: turning that frame trusted must change
    # the bytes
    trusted_variant = err.copy()
    trusted_variant[1, 0] = 1.0
    out2 = tmp_path / "c2.png"
    res2 = plot_reprojection_curve(trusted_variant, out2, flag_value=FLAG)
    assert res2["flagged_frames"] == 0
    assert out.read_bytes() != out2.read_bytes()
    # Sentinel values themselves do not participate in the drawing: swapping in another
    # (still >= threshold) sentinel must give a byte-identical image. If the implementation
    # plotted all frames (including flagged ones), this equality assertion would fail.
    other_sentinel = err.copy()
    other_sentinel[1, 0] = FLAG * 10
    out3 = tmp_path / "c3.png"
    plot_reprojection_curve(other_sentinel, out3, flag_value=FLAG)
    assert out.read_bytes() == out3.read_bytes()


def test_curve_plot_all_flagged_reports_instead_of_faking(tmp_path):
    err = np.full((4, 2), FLAG, np.float32)
    res = plot_reprojection_curve(err, tmp_path / "c.png", flag_value=FLAG)
    assert res["plotted"] is False
    assert res.get("reason")
    assert not (tmp_path / "c.png").exists()


def test_curve_rendering_has_no_missing_glyph_warning(tmp_path):
    """Text on the plot (including CJK) must not produce a missing-glyph warning: otherwise
    it would render as a row of tofu boxes (□□□).

    With a CJK font installed this machine uses CJK labels; without one it uses ASCII
    labels — neither path may emit a ``Glyph ... missing from font`` warning.
    """
    err = np.array([[1.0, 2.0], [FLAG, 1.0], [3.0, 0.5]], np.float32)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        plot_reprojection_curve(err, tmp_path / "c.png", flag_value=FLAG)
    missing = [str(w.message) for w in caught if "missing from font" in str(w.message)]
    assert not missing, f"plot shows a missing-glyph (tofu box) warning: {missing}"


def test_qc_report_contains_curve_and_depth_source(tmp_path):
    _ws_with_artifacts(tmp_path)
    info = run_publish(workspace_root=str(tmp_path), output_root=str(tmp_path / "out"),
                       episode_id="ep_0001", make_visualization=False,
                       camera_view=False, world_view=False, fps=30,
                       export_script=str(FAKES / "fake_lerobot_export.py"))
    html = (tmp_path / "out" / "qc" / "ep_0001" / "report.html").read_text()
    assert "Reprojection error" in html
    # Depth-source counts come from fused.npz's depth_source (8 frames, all stereo)
    assert "<td>stereo</td><td>8</td>" in html
    # mean/max derived directly from reprojection_error=[[1.0, 2.0]]*8: mean 1.5 / max 2.0
    assert "mean 1.5000, max 2.0000" in html
    assert "no curve" not in html
    assert "data:image/png;base64," in html
    assert info["outputs"]["qc_report"]
