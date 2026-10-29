"""ViPE inference script (run by the vipe venv's python).

**Real API (verified item-by-item on an A800 GPU, replacing all `[verify on GPU]` assumptions)**:

1. Config goes through hydra, but the **pipeline group is nested under `cfg.pipeline`** (not
   flattened to the root): `cfg = compose(config_name="default", overrides=["pipeline=pose_only", ...])`
   -> overrides must be written as `pipeline.output.path=...`; `pipeline.init.instance` is not yet
   in the struct at compose time and must be set after compose via `OmegaConf.update(cfg, ...)`.
2. `validate_typed_config(cfg)` -> typed config; `StreamList.make(cfg.streams)` -> the video stream
   list; `make_pipeline(cfg.pipeline)` -> the pipeline object.
3. With `pipeline.return_payload = True`, `pipeline.run(stream)` returns
   `AnnotationPipelineOutput(payload=SLAMOutput)`.
4. `SLAMOutput.get_view_trajectory(0).matrix()` is a **(N,4,4) c2w**
   (per `scripts/vipe_to_colmap.py`: it inverts that matrix to get w2c).
5. `SLAMOutput.intrinsics[0]` is `[fx, fy, cx, cy]`.
6. `SLAMOutput.slam_map.project_map(...)` can back-project the sparse SLAM point cloud into a
   dense depth map for a given frame (the `pose_only` pipeline has no per-frame dense-depth post-processing).

Weights (placed offline; `TORCH_HOME` points at their parent):
  `$TORCH_HOME/hub/geocalib/pinhole.tar` (GeoCalib intrinsics)
  `$TORCH_HOME/hub/moge2/moge-2-vitl.pt` (keyframe metric depth)
  `$TORCH_HOME/hub/droid_slam/droid.pth` (DROID-SLAM front/back end)
  `$TORCH_HOME/hub/checkpoints/superpoint_v6_from_tf.pth` (sparse tracking)
And set `HF_HUB_OFFLINE=1` to avoid network access.
"""

import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

from _common import RESULT_MARKER, load_input, parse_args, save_arrays


def _write_video(frames: np.ndarray, fps: float, path: Path) -> None:
    h, w = frames.shape[1], frames.shape[2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError("could not open VideoWriter")
    try:
        for f in frames:
            writer.write(f)
    finally:
        writer.release()


def build_model(weights: str, extra: dict):
    """Build the ViPE pipeline (hydra compose + make_pipeline), returning the pipeline.

    weights/extra are unused in this version (weights are located via TORCH_HOME and the pipeline
    config is inline), but the signature stays isomorphic with the other model scripts - the serve
    loop caches models keyed by ``(weights, extra)``.
    make_pipeline creates the directory at ``pipeline.output.path``; in return_payload mode that
    path **is not written to** (run's early-return branch returns the payload before writing any
    artifact); only __init__ does the mkdir. So the placeholder uses a system temp dir instead of
    CWD - a serve worker's CWD may be a read-only Ray workdir, and creating a directory in CWD
    would fail build_model outright; the temp dir is mkdtemp'd once at model build time.
    """
    import atexit
    import shutil

    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    from vipe._paths import get_config_path
    from vipe.config import validate_typed_config
    from vipe.pipeline import make_pipeline

    # The placeholder dir is referenced by no one; clean it up at process exit (serve workers are
    # resident, so hooking atexit is enough).
    placeholder_out = tempfile.mkdtemp(prefix="vipe_out_")
    atexit.register(shutil.rmtree, placeholder_out, ignore_errors=True)

    config_dir = str(get_config_path())
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(
            config_name="default",
            overrides=[
                "pipeline=pose_only",
                "streams=raw_mp4_stream",
                "streams.base_path=input.mp4",  # placeholder: the input stream is rebuilt from the real video in infer_once
                "streams.frame_end=-1",         # placeholder: the per-frame upper bound is overwritten with n_frames in infer_once
                "pipeline.output.path=" + placeholder_out,  # placeholder: return_payload writes nothing
                "pipeline.output.save_artifacts=false",
                "pipeline.output.save_viz=false",
            ],
        )
    # `init` is provided by the pipeline group and is not yet in the struct at compose time, so set it here.
    # TrackAnything (GroundingDINO+SAM) is not needed for pose/depth only; null it to save VRAM and weights.
    OmegaConf.update(cfg, "pipeline.init.instance", None)

    typed = validate_typed_config(cfg)
    pipeline = make_pipeline(typed.pipeline)
    pipeline.return_payload = True
    return pipeline


def infer_once(model, input_path: str, output_path: str, extra: dict) -> dict:
    """Infer once on one input: write a temp video -> pipeline.run -> payload extraction ->
    project_map depth loop -> assemble and write the npz (save_arrays) -> return a summary dict."""
    data = load_input(input_path)
    frames = data["frames"]
    t, h, w = int(data["n_frames"]), frames.shape[1], frames.shape[2]
    fps = float(data["fps"])

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("ViPE requires CUDA (the SLAM backend hardcodes device='cuda')")

    pipeline = model
    with tempfile.TemporaryDirectory() as td:
        video_path = Path(td) / "input.mp4"
        _write_video(frames, fps, video_path)
        # The input stream is item-for-item equivalent to build_model's raw_mp4_stream.yaml defaults
        # (frame_start=0/frame_skip=1/cached=false) plus the original compose override (frame_end=t);
        # base_path is the real video. Stream construction is cheap (reads video metadata), so it
        # stays in infer_once so each input gets its own temp video.
        from vipe.streams.raw_mp4_stream import RawMP4StreamList

        stream = RawMP4StreamList(str(video_path), 0, t, 1, cached=False)[0]
        slam = pipeline.run(stream).payload
        if slam is None:
            raise RuntimeError("ViPE returned no SLAMOutput payload")

        traj = slam.get_view_trajectory(0)  # SE3, c2w
        c2w = np.asarray(traj.matrix().cpu().numpy(), dtype=np.float64)  # (N,4,4)
        n = c2w.shape[0]

        intr4 = np.asarray(slam.intrinsics[0].cpu().numpy(), dtype=np.float64)  # [fx,fy,cx,cy]
        k = np.repeat(np.eye(3, dtype=np.float64)[None], n, axis=0)
        k[:, 0, 0], k[:, 1, 1], k[:, 0, 2], k[:, 1, 2] = intr4

        h, w = stream.frame_size()
        depth = np.zeros((n, h, w), dtype=np.float32)
        # The artifact's `depth_source` is an **enum contract** (only "monocular"/"stereo" allowed;
        # downstream branches on it). Finer provenance goes into the `__RESULT__` summary's
        # depth_detail, not into the enum.
        depth_src = "monocular"
        depth_detail = "slam_map_keyframe_projection"
        try:
            slam_map = slam.slam_map
            if slam_map is None:
                raise RuntimeError("slam_map unavailable")
            for i in range(n):
                d = slam_map.project_map(
                    frame_tstamp=i,
                    view_idx=0,
                    target_size=(h, w),
                    target_intrinsics=slam.intrinsics[0],
                    target_pose=traj[i],
                    target_camera_type=pipeline.camera_type,
                    infill=True,
                    tstamp_nn=3,
                )
                depth[i] = np.asarray(d.cpu().numpy(), dtype=np.float32)
        except Exception as exc:  # a depth failure must not mask the pose result, but be stated honestly
            import traceback

            traceback.print_exc()
            depth_detail = f"unavailable: {type(exc).__name__}: {exc}"
            depth = np.zeros((0, h, w), dtype=np.float32)
        if depth.shape[0] < n:
            # On failure, be **loud and explicit about the cause**: otherwise the caller only sees
            # "Scene artifact missing key: ['depth']" with no clue whether the SLAM point cloud was
            # too sparse, the projection had no inliers, or a CUDA op got degenerate input.
            # Observed trigger: synthetic random-noise video -> DROID-SLAM has nothing trackable ->
            # keyframe clouds have only ~100 points per frame -> project_map's kd-tree raises
            # cudaErrorInvalidValue.
            raise RuntimeError(
                f"ViPE could not produce per-frame dense depth ({depth_detail}): "
                f"the SLAM point cloud is too sparse or the projection has no inliers. "
                f"Common with textureless synthetic input; use a real video instead, "
                f"or switch to pipeline=default with its dense-depth post-processing."
            )

        diag = {
            "intrinsics": [float(x) for x in intr4],
            "keyframe_ids": [int(i) for i in slam.slam_map.dense_disp_frame_inds] if slam.slam_map else [],
            "depth_source": depth_src,
            "depth_detail": depth_detail,
        }

    n = min(t, c2w.shape[0])
    poses_w2c = np.linalg.inv(c2w[:n])  # output contract: world-to-camera
    npz = {
        "camera_poses": poses_w2c.astype(np.float64),
        "intrinsics": k[:n].astype(np.float64),
        "depth_source": np.array(diag["depth_source"]),
        "n_frames": np.int32(n),
        "fps": np.int32(round(fps)),
    }
    if depth.shape[0] >= n and n > 0:
        npz["depth"] = depth[:n].astype(np.float32)

    save_arrays(output_path, npz)
    return {
        "ok": True,
        "model": "vipe",
        "n_frames": int(n),
        "intrinsics": diag["intrinsics"],
        "keyframe_ids": diag["keyframe_ids"],
        "depth_source": diag["depth_source"],
        "depth_detail": diag["depth_detail"],
        "depth_finite": bool(np.isfinite(npz["depth"]).all()) if "depth" in npz else False,
    }


def main() -> None:
    args = parse_args("ViPE: camera poses + near-metric depth (monocular)")
    extra = json.loads(args.extra) if args.extra else {}
    try:
        model = build_model(args.weights, extra)
        summary = infer_once(model, args.input, args.output, extra)
    except ValueError as exc:
        # In serve mode ValueError is caught by run_serve_loop and returned as an error line;
        # oneshot keeps the CLI exit semantics.
        raise SystemExit(str(exc)) from exc
    print(RESULT_MARKER + json.dumps(summary))
    sys.stdout.flush()


if __name__ == "__main__":
    if "--serve" in sys.argv:
        from _common import run_serve_loop

        run_serve_loop(build_model, infer_once)
    else:
        main()
