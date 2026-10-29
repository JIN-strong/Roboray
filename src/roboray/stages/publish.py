"""Stage 4: LeRobot v3 dataset, visualization, QC report, publish in completion order."""

import base64
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .fusion import UNTRUSTED_REPROJ_ERROR

__all__ = ["run_publish"]

_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>QC {ep}</title>
<style>body{{font-family:sans-serif;margin:2rem}}table{{border-collapse:collapse}}
td,th{{border:1px solid #ccc;padding:4px 8px}}</style></head><body>
<h1>QC Report · {ep}</h1>
<h2>Stage timings (s)</h2><table><tr><th>Stage</th><th>Duration</th></tr>{rows}</table>
{stats_html}
</body></html>"""

# Data sections: **rendered only when fused.npz is available**. When it is missing they
# must be omitted entirely, otherwise zeros would masquerade as measured values like
# "mean 0.0000 px / untrusted hand frames 0 / 0 total".
_STATS_HTML = """<h2>Gap-fill audit</h2><p>filled {filled} frames, dropped {rejected} frames</p>
<p>Rejection reasons: {reasons}</p>
<p>Trajectory-check skipped (passed) counts: {skipped}</p>
<h2>Reprojection error</h2><p>Metric source: {metric} ({metric_note})</p>
<p>mean {mean_err:.4f}, max {max_err:.4f} px (trusted hand frames only);
untrusted hand frames {flagged_hands} (excluded by the {flag_value:.0f} sentinel)</p>
{curve_html}
<h2>Depth source</h2><table><tr><th>Source</th><th>Frames</th></tr>{depth_rows}</table>
<h2>Identity switches</h2><p>{switches} total</p>
<h2>Per-frame diagnostics (every {stride} frames / {n_frames} total, {listed} rows listed)</h2>
<table>{diag_header}{diag_rows}</table>"""

# The only body text when there is no measured data: clearly state the report has timings
# only, with no reconstruction statistics.
_NO_DATA_HTML = ("<p>no measured data: {reason}; "
                 "this report contains stage timings only, no reconstruction statistics</p>")

# Human-readable notes per metric source: mano = real metric; joints_2d = always-zero
# round-trip, do not read it as quality
_METRIC_NOTES = {
    "mano": "consistency of MANO structure vs image evidence (+root depth), informative",
    "joints_2d": "fallback when 3D joints are missing: backprojection round-trip, "
                 "structurally always 0/sentinel, consistency check only",
}
_METRIC_NOTE_DEFAULT = "unknown source (old artifact without joint_source)"
# Gap-fill audit counts (gapfill_reasons / gapfill_skipped) persist in fused.npz as JSON string arrays.
_OLD_ARTIFACT = "(not recorded in old artifact)"


def _fmt_counts(counts: dict | None) -> str:
    """Render {reason: count} as a readable "reason=count" string; empty counts = none,
    missing key = old artifact."""
    if counts is None:
        return _OLD_ARTIFACT
    if not counts:
        return "none"
    return ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))


def _fmt_err(value, flag_value: float) -> str:
    """Single-cell error in the per-frame diagnostics: sentinel/missing spelled out, never
    present a 1e6 as a measured value.

    Two non-measurement values are distinguished: NaN is "no observation", >= flag_value is
    an "untrusted/fill sentinel" — the semantics differ and the report must not conflate them.
    """
    v = float(value)
    if not np.isfinite(v):
        return "NaN (no observation)"
    if v >= flag_value:
        return f"flagged (≥{flag_value:.0f})"
    return f"{v:.4f}"


def _depth_rows(depth_source) -> str:
    """Render depth-source statistics (stereo/monocular frame counts) as table rows;
    missing key = old artifact (do not fabricate 0 frames)."""
    if depth_source is None:
        return f"<tr><td colspan=2>{_OLD_ARTIFACT}</td></tr>"
    counts = Counter(str(x) for x in np.asarray(depth_source, dtype=object).ravel())
    if not counts:
        return "<tr><td colspan=2>none</td></tr>"
    return "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in sorted(counts.items()))


def _diag_rows(reproj, depth_source, flag_value: float, stride: int):
    """Per-frame diagnostics table (thinned): one row every ``stride`` frames with frame
    number / per-hand errors / depth source.

    Thinning keeps long episodes (thousands of frames) from bloating the HTML; returns
    (header HTML, rows HTML, row count). Hand count is dynamic from ``reproj.shape[1]``
    (at least two columns; missing hands get "—"), so works with >2 hands without silently
    dropping any. When ``depth_source`` is not in the artifact (old artifact) it is honestly
    written as "not recorded", never fabricated.
    """
    t, hands = reproj.shape
    n_cols = max(2, hands)
    col_names = ["left-hand error", "right-hand error"] + [f"hand {h} error" for h in range(2, n_cols)]
    header = ("<tr><th>frame</th>"
              + "".join(f"<th>{n}</th>" for n in col_names)
              + "<th>depth source</th></tr>")
    ds = None if depth_source is None else np.asarray(depth_source, dtype=object).ravel()
    rows, listed = [], 0
    for i in range(0, t, stride):
        cells = [
            _fmt_err(reproj[i, h], flag_value) if h < hands else "—"
            for h in range(n_cols)
        ]
        if ds is None:
            src = _OLD_ARTIFACT
        elif i < ds.size:
            src = str(ds[i])
        else:
            src = "—"
        cells_html = "".join(f"<td>{c}</td>" for c in cells)
        rows.append(f"<tr><td>{i}</td>{cells_html}<td>{src}</td></tr>")
        listed += 1
    return header, "".join(rows), listed


def _curve_html(reproj, png_path: Path, flag_value: float) -> str:
    """Render the reprojection error curve and embed it as base64; when there is no curve,
    state the reason explicitly and remove any stale image.

    Curve numbers come directly from ``fused.npz``'s ``reprojection_error`` (trusted frames
    plotted, flagged frames crossed separately), same source as the rest of this report —
    the report must attest to its own data authenticity.
    """
    from roboray.viz.qc_plots import plot_reprojection_curve

    res = plot_reprojection_curve(reproj, png_path, flag_value=flag_value)
    if not res.get("plotted"):
        if png_path.exists():
            png_path.unlink()  # don't let the previous round's plot pass as this round's product
        return f"<p>no curve: {res.get('reason', 'unknown')}</p>"
    b64 = base64.b64encode(png_path.read_bytes()).decode("ascii")
    return f'<img alt="reprojection error curve" src="data:image/png;base64,{b64}">'



def _note_skip(outputs: dict, key: str, reason: str) -> None:
    """Merge a "view skipped" reason into outputs (multiple reasons for the same key joined
    by "; ")."""
    prev = outputs.get(key)
    outputs[key] = f"{prev}; {reason}" if prev else reason


def _read_npz_keys(path: Path, keys: tuple, label: str) -> dict:
    """Read the requested keys that **exist** in the NPZ; missing/corrupt file raises a
    readable ValueError.

    A missing key is not an error (the returned dict just lacks that key); the caller
    decides required vs optional: an old artifact may lack optional fields, and a missing
    ``intrinsics`` should only drop the point cloud, not the whole view. Corruption takes
    many forms (``BadZipFile`` / ``UnpicklingError`` / ``OSError`` …), so a broad catch is
    converted into one readable ValueError: the caller always treats it as "skip that view",
    never crashing the episode.
    """
    if not path.exists():
        raise ValueError(f"{label} missing")
    try:
        z = np.load(path, allow_pickle=True)
    except Exception as exc:
        raise ValueError(f"{label} corrupt ({type(exc).__name__}: {exc})") from exc
    with z:
        present = [k for k in keys if k in z.files]
        try:
            # Force the read (NPZ members are lazily decompressed; truncated files only error
            # on access) and detach from the already-closed file handle
            return {k: np.asarray(z[k]) for k in present}
        except Exception as exc:
            raise ValueError(f"{label} corrupt ({type(exc).__name__}: {exc})") from exc


def _load_viz_inputs(ws: Path, outputs: dict) -> dict:
    """Read the visualization-input artifacts per view (cross-file seams; see spec Step 4).

    Camera view: ``prepared/frames_left.npy`` -> frames,
    ``artifacts/hand.npz`` -> joints_2d/hand_present;
    World view: ``artifacts/scene.npz`` -> camera_poses/depth (required) + intrinsics (optional),
    ``artifacts/fused.npz`` -> joints_world.

    ``intrinsics`` is only used by the point cloud: when absent, **do not** drop the whole
    world view — skeleton/camera trajectory/future trajectory render as usual
    (``render_world_view`` records ``pointcloud_skipped`` in its return value). Other
    required keys missing or a corrupt artifact only drop the corresponding view and record
    the reason in ``outputs`` (``camera_view_skipped`` / ``world_view_skipped``), never
    crashing the episode: partial visualization beats total failure.
    """
    data: dict = {"camera": {}, "world": {}}
    try:
        data["camera"]["frames"] = np.load(ws / "prepared" / "frames_left.npy")
    except Exception as exc:
        _note_skip(
            outputs, "camera_view_skipped",
            f"frames_left.npy missing or corrupt ({type(exc).__name__}: {exc})",
        )
    for view, path, keys, required, label in (
        ("camera", ws / "artifacts" / "hand.npz",
         ("joints_2d", "hand_present"), ("joints_2d", "hand_present"), "hand.npz"),
        ("world", ws / "artifacts" / "scene.npz",
         ("camera_poses", "depth", "intrinsics"), ("camera_poses", "depth"), "scene.npz"),
        ("world", ws / "artifacts" / "fused.npz", ("joints_world",), ("joints_world",),
         "fused.npz"),
    ):
        try:
            got = _read_npz_keys(path, keys, label)
        except ValueError as exc:
            _note_skip(outputs, f"{view}_view_skipped", str(exc))
            continue
        missing = [k for k in required if k not in got]
        if missing:
            _note_skip(outputs, f"{view}_view_skipped", f"{label} missing keys {missing}")
            continue
        data[view].update(got)
    return data


def run_publish(
    workspace_root: str,
    output_root: str,
    episode_id: str,
    make_visualization: bool = True,
    camera_view: bool = True,
    world_view: bool = True,
    fps: int = 30,
    export_script: str | None = None,
) -> dict:
    ws = Path(workspace_root) / episode_id
    out = Path(output_root)
    state = json.loads((ws / "state.json").read_text())

    try:
        with np.load(ws / "artifacts" / "action.npz", allow_pickle=True) as z:
            state_arr = z["state"]
            action = z["action"]
            labels = [str(x) for x in z["labels"]]
            seg_start = z["segment_start"]
            seg_end = z["segment_end"]
    except (OSError, ValueError, EOFError) as exc:
        raise ValueError(f"publish artifact missing or corrupt: action.npz ({exc})") from exc

    # --- LeRobot v3 (spec §13: must go through the lerobot public API) ---
    from roboray.export.lerobot_writer import LeRobotWriter

    ds_dir = out / "lerobot_v3" / episode_id
    t_state = np.asarray(state_arr, np.float32)
    t_action = np.asarray(action, np.float32)
    if t_state.ndim == 4:  # (T, hands, joints, 3) -> (T, 126)
        t_state = t_state.reshape(t_state.shape[0], -1)
    if t_action.ndim == 4:
        t_action = t_action.reshape(t_action.shape[0], -1)
    # export_script: explicitly inject the export script (test stand-in); default None ->
    # use the real script from the env/repo.
    info = LeRobotWriter(script=export_script).write(
        episode_id=episode_id,
        state=t_state,
        action=t_action,
        tasks=[labels[0] if labels else "unknown"],
        fps=fps,
        out_dir=ds_dir,
    )

    # Per-segment labels/segments info: a real LeRobot dataset layout has no slot for it
    # (writing it would be treated by lerobot as an unknown file/field), so it is mirrored to
    # qc/<ep>/segments.json to stay traceable.
    qc_dir = out / "qc" / episode_id
    qc_dir.mkdir(parents=True, exist_ok=True)
    (qc_dir / "segments.json").write_text(
        json.dumps(
            {
                "episode_id": episode_id,
                "fps": fps,
                "n_frames": int(state_arr.shape[0]),
                "labels": sorted(set(labels)),
                "segments": [
                    {"label": label, "start": int(s), "end": int(e)}
                    for label, s, e in zip(labels, seg_start, seg_end)
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    # --- Visualization (camera view = original frames + 2D skeleton overlay; world view =
    # --- matplotlib Agg 3D render)
    outputs = {"dataset": info["dataset_dir"]}
    if make_visualization:
        viz_dir = out / "viz" / episode_id
        viz_dir.mkdir(parents=True, exist_ok=True)
        inputs = _load_viz_inputs(ws, outputs)
        if camera_view:
            cam_needs = ("frames", "joints_2d", "hand_present")
            cam = inputs["camera"]
            cam_missing = sorted(k for k in cam_needs if k not in cam)
            if cam_missing:
                # The specific reason (which artifact missing/corrupt) was already recorded;
                # here add which keys are missing
                _note_skip(outputs, "camera_view_skipped",
                           f"visualization inputs incomplete (missing {cam_missing})")
            else:
                from roboray.viz.camera_view import render_camera_view

                try:
                    info_cam = render_camera_view(
                        cam["frames"], cam["joints_2d"], cam["hand_present"],
                        fps, viz_dir / "camera.mp4",
                    )
                    if info_cam["frames"] == 0:
                        # No frames: do not record a view product, only the reason (so an
                        # empty video is not treated as a success)
                        _note_skip(outputs, "camera_view_skipped",
                                   info_cam.get("skipped_reason", "unknown"))
                    else:
                        outputs["camera_view"] = info_cam["out_path"]
                except Exception as exc:  # render failure must not sink the episode (partial viz > total failure)
                    _note_skip(outputs, "camera_view_skipped",
                               f"render failed ({type(exc).__name__}: {exc})")
        if world_view:
            # intrinsics is only used by the point cloud -> not in the required keys (when
            # absent only the point cloud is dropped, not the whole view)
            wld_needs = ("joints_world", "camera_poses", "depth")
            wld = inputs["world"]
            wld_missing = sorted(k for k in wld_needs if k not in wld)
            if wld_missing:
                _note_skip(outputs, "world_view_skipped",
                           f"visualization inputs incomplete (missing {wld_missing})")
            else:
                from roboray.viz.world_view import render_world_view

                try:
                    info_world = render_world_view(
                        wld["joints_world"], wld["camera_poses"], wld["depth"],
                        fps, viz_dir / "world.mp4", intrinsics=wld.get("intrinsics"),
                    )
                    if info_world.get("pointcloud_skipped"):
                        # Degraded view (point cloud missing): still record the product while
                        # honestly noting what is missing
                        outputs["pointcloud_skipped"] = info_world["pointcloud_skipped"]
                    if info_world["frames"] == 0:
                        # All-untrusted hand joints: do not produce a misleading video, only
                        # report the reason
                        _note_skip(
                            outputs, "world_view_skipped",
                            info_world.get("skipped_reason", "unknown"),
                        )
                    else:
                        outputs["world_view"] = info_world["out_path"]
                except Exception as exc:  # render failure must not sink the episode
                    _note_skip(outputs, "world_view_skipped",
                               f"render failed ({type(exc).__name__}: {exc})")

    # --- QC report ---
    # fused.npz goes through the same fault-tolerant read path as the visualization: when
    # missing/corrupt the episode no longer crashes, it degrades to a report with "stage
    # timings only, no measured data", reason recorded in outputs["qc_skipped"].
    _QC_REQUIRED = ("reprojection_error", "gapfill_filled", "gapfill_rejected",
                    "identity_switches")
    try:
        fused = _read_npz_keys(
            ws / "artifacts" / "fused.npz",
            _QC_REQUIRED + ("joint_source", "gapfill_reasons", "gapfill_skipped",
                            "depth_source"),
            "fused.npz",
        )
        missing_required = [k for k in _QC_REQUIRED if k not in fused]
        if missing_required:
            raise ValueError(f"fused.npz missing keys {missing_required}")
    except ValueError as exc:
        _note_skip(outputs, "qc_skipped", f"QC statistics lack measured data ({exc})")
        fused = {}

    if fused:
        # Error matrix normalized to 2D (T, hands): both the curve and the per-frame
        # diagnostics index by this.
        reproj = np.asarray(fused["reprojection_error"])
        if reproj.ndim == 0:
            reproj = reproj.reshape(1, 1)
        elif reproj.ndim == 1:
            reproj = reproj.reshape(-1, 1)
        elif reproj.ndim > 2:
            reproj = reproj.reshape(reproj.shape[0], -1)
        filled = int(fused["gapfill_filled"])
        rejected = int(fused["gapfill_rejected"])
        switches = fused["identity_switches"]
        metric = str(fused["joint_source"]) if "joint_source" in fused else "unknown"
        # Gap-fill audit (rejection-reason distribution / trajectory-check skipped counts):
        # JSON string arrays; old artifacts may lack the keys
        reasons = json.loads(str(fused["gapfill_reasons"])) if "gapfill_reasons" in fused else None
        skipped = json.loads(str(fused["gapfill_skipped"])) if "gapfill_skipped" in fused else None
        depth_source = fused.get("depth_source")  # old artifacts may not record it
        flag_value = float(UNTRUSTED_REPROJ_ERROR)
        # Untrusted hand frames are flagged with UNTRUSTED_REPROJ_ERROR: statistics must
        # exclude the sentinel (otherwise mean/max are polluted by 1e6), and the untrusted
        # count is reported separately. The curve shares the same source (same reproj array).
        finite = np.isfinite(reproj)
        flagged_mask = ~finite | (reproj >= UNTRUSTED_REPROJ_ERROR)
        trusted = reproj[~flagged_mask]
        mean_err = float(trusted.mean()) if trusted.size else 0.0
        max_err = float(trusted.max()) if trusted.size else 0.0
        flagged_hands = int(flagged_mask.sum())
        t = int(reproj.shape[0])
        stride = max(1, t // 40)  # thinning: long episodes must not produce a giant HTML
        diag_header, diag_rows, listed = _diag_rows(reproj, depth_source, flag_value, stride)
        stats_html = _STATS_HTML.format(
            filled=filled,
            rejected=rejected,
            reasons=_fmt_counts(reasons),
            skipped=_fmt_counts(skipped),
            metric=metric,
            metric_note=_METRIC_NOTES.get(metric, _METRIC_NOTE_DEFAULT),
            mean_err=mean_err,
            max_err=max_err,
            flagged_hands=flagged_hands,
            flag_value=flag_value,
            curve_html=_curve_html(reproj, qc_dir / "reprojection.png", flag_value),
            depth_rows=_depth_rows(depth_source),
            switches=len(switches),
            stride=stride,
            n_frames=t,
            listed=listed,
            diag_header=diag_header,
            diag_rows=diag_rows,
        )
    else:
        # Degraded: write only the "no measured data" note and the stage-timing table;
        # **never** render measurement values fabricated from missing data like "mean 0.0000 /
        # untrusted hand frames 0 / 0 total".
        mean_err = max_err = 0.0
        flagged_hands = 0
        metric = "unknown"
        stale_png = qc_dir / "reprojection.png"
        if stale_png.exists():
            stale_png.unlink()  # when the report shows no curve, don't leave the previous plot lying beside it
        stats_html = _NO_DATA_HTML.format(
            reason=outputs.get("qc_skipped", "fused.npz unavailable")
        )
    rows = "".join(f"<tr><td>{k}</td><td>{v:.3f}</td></tr>" for k, v in state["timings"].items())
    (qc_dir / "report.html").write_text(
        _HTML.format(
            ep=episode_id,
            rows=rows or "<tr><td colspan=2>none</td></tr>",
            stats_html=stats_html,
        )
    )
    outputs["qc_report"] = str(qc_dir / "report.html")
    return {
        "episode_id": episode_id,
        "outputs": outputs,
        "joint_source": metric,
        "reprojection_error_mean_px": mean_err,
        "reprojection_error_max_px": max_err,
        "flagged_hands": flagged_hands,
    }
