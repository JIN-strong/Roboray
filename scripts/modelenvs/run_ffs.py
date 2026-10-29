"""Fast-FoundationStereo inference script (runs in the ffs env).

Assumed API contract (per nvlabs/fast-foundationstereo docs, [verify on GPU]):
1. The weights directory contains model_best_bp2_serialize.pth [verify on GPU: HF nvidia/c-fast-foundationstereo download artifact name]
2. Python API for model loading and stereo forward pass [verify on GPU: see scripts/run_demo.py internals]
3. Input must be rectified horizontal-epipolar stereo; this pipeline's layout is already rectified

Depth conversion (deterministic, not GPU-verification): depth = fx * baseline / disparity,
baseline taken from --extra T_left_to_right[0][3] (right-eye +x convention).

Input-resolution control (deterministic, not GPU-verification): --extra
stereo_max_image_side_px scales the stereo pair so max(H, W) <= that limit and scales fx
proportionally (0/missing = no scaling). Because disparity scales with resolution, depth is
invariant to scaling - this knob trades accuracy for speed. Inference runs on the scaled
grid, but the output depth / intrinsics always return to the **original frame geometry**
(fusion samples depth on the original pixels; a scaled grid would go out of bounds or
misalign - Plan B review residual #1); __RESULT__ also reports model_input_size and depth_grid.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

from _common import RESULT_MARKER, load_input, parse_args, save_arrays


def _write_png(path: Path, frame: np.ndarray) -> None:
    if not cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"PNG write failed: {path}")


def build_model(weights: str, extra: dict):
    """Load the FFS model: validate the weights dir -> torch.load -> merge cfg.yaml -> cuda().eval().

    Input-side semantic validation (frames_right/baseline) lives in infer_once - in serve
    mode build_model is cached by ``(weights, extra)``, so input-independent checks should not
    repeat per frame; input-dependent checks must run with every inference. A missing/corrupt
    weights dir raises ValueError here (SystemExit does not inherit from Exception, the serve
    loop cannot catch it, and it would kill the resident process).
    """
    # The repo root must be on sys.path (FFS has no setup.py; it is a "import core from repo root" project)
    repo = os.environ.get("ROBORAY_FFS_REPO", "")
    if repo and repo not in sys.path:
        sys.path.insert(0, repo)
    try:
        import torch

        from core.utils.utils import InputPadder  # verified real import (run_demo.py)
    except ImportError as exc:
        raise RuntimeError(
            f"ffs env unavailable (missing package {exc}); build it with deploy/modelenvs/build_ffs.sh "
            "and set ROBORAY_FFS_REPO to the FFS repo root"
        ) from exc

    weights_dir = Path(weights) if weights else None
    if weights_dir is None or not weights_dir.exists():
        raise ValueError("FFS weights directory is not set or does not exist (--weights / ROBORAY_FFS_WEIGHTS)")
    ckpt = weights_dir / "model_best_bp2_serialize.pth"
    if not ckpt.exists():
        raise ValueError(f"missing weights file: {ckpt}")

    # Verified real loading path: the whole model is saved via pickle, so torch.load directly;
    # the cfg.yaml next to the weights must be merged into model.args (as run_demo.py does),
    # otherwise the forward pass fails on missing keys with omegaconf ConfigAttributeError (e.g. normalize).
    model = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    cfg_path = weights_dir / "cfg.yaml"
    if cfg_path.exists():
        import yaml as _yaml
        from omegaconf import OmegaConf

        cfg = _yaml.safe_load(cfg_path.read_text()) or {}
        merged = dict(model.args) if isinstance(model.args, dict) else dict(cfg)
        if isinstance(model.args, dict):
            merged = {**model.args, **cfg}
        # foundation_stereo.forward reads self.args.normalize on the optimize_build_volume='pytorch1'
        # path, but neither the pickled args nor cfg.yaml provide that key
        # (build_gwc_volume_optimized_pytorch1 defaults it to True).
        merged.setdefault("normalize", True)
        model.args = OmegaConf.create(merged)
    model.args.valid_iters = 8
    model.args.max_disp = 192
    model.cuda().eval()
    return model


def infer_once(model, input_path: str, output_path: str, extra: dict) -> dict:
    """Infer once on one stereo input: input validation/scaling -> per-frame forward -> depth back to original geometry -> write out."""
    data = load_input(input_path)
    if "frames_right" not in data:
        raise ValueError("missing frames_right: FFS needs stereo input")
    baseline = float(extra.get("baseline_m", 0.0))
    if baseline <= 0:
        raise ValueError("--extra must include a positive baseline_m (from T_left_to_right[0][3])")

    left, right = data["frames"], data["frames_right"]
    t = int(data["n_frames"])
    # Original frame geometry (caller grid): the final depth and intrinsics geometry is always this.
    orig_h, orig_w = int(left.shape[1]), int(left.shape[2])
    h, w = orig_h, orig_w
    # Intrinsics come from calib (fx passed via --extra); fall back to a width-based approximation
    fx = float(extra.get("fx", float(w)))
    fx_orig = fx  # unscaled intrinsics: output intrinsics use original geometry (model-input scaling does not change physical intrinsics)

    # spec §6 input-resolution control: scale the stereo pair so max(H, W) <= limit, scaling fx
    # proportionally. Physics: real disparity scales with resolution (d' ~ d*s), so fx'*B/d' == fx*B/d -
    # **depth is unchanged**; this knob trades accuracy (coarser small-disparity quantization) for speed.
    # limit <= 0 / missing = no scaling.
    limit = int(extra.get("stereo_max_image_side_px", 0) or 0)
    if limit > 0 and max(h, w) > limit:
        scale = limit / float(max(h, w))
        new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        left = np.stack([cv2.resize(f, (new_w, new_h), interpolation=cv2.INTER_AREA) for f in left])
        right = np.stack([cv2.resize(f, (new_w, new_h), interpolation=cv2.INTER_AREA) for f in right])
        fx = fx * new_w / float(w)
        w, h = new_w, new_h

    # Model-input grid used for inference after scaling (for __RESULT__/QC verification)
    model_h, model_w = int(h), int(w)

    import torch

    # InputPadder must be bound inside this function (harmless alongside the import in
    # build_model) - it is used here, so importing only in build_model would NameError on the
    # very first inference in a real env. build_model already put ROBORAY_FFS_REPO on sys.path.
    from core.utils.utils import InputPadder

    depths = []
    # Verified real forward pass (run_demo.py): HWC uint8 -> CHW float [1,3,H,W] -> InputPadder(/32)
    # -> bf16 autocast -> model.forward(iters, test_mode, optimize_build_volume) -> unpad
    with torch.no_grad():
        for i in range(t):
            img0 = torch.as_tensor(np.ascontiguousarray(left[i])).cuda().float()[None].permute(0, 3, 1, 2)
            img1 = torch.as_tensor(np.ascontiguousarray(right[i])).cuda().float()[None].permute(0, 3, 1, 2)
            padder = InputPadder(img0.shape, divis_by=32, force_square=False)
            p0, p1 = padder.pad(img0, img1)
            with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                disp = model.forward(p0, p1, iters=8, test_mode=True,
                                     optimize_build_volume="pytorch1")
            disp = padder.unpad(disp)
            disparity = np.asarray(disp.squeeze().float().cpu(), dtype=np.float32)
            disparity = np.clip(disparity, 1e-3, None)
            depths.append((fx * baseline / disparity).astype(np.float32))
    depth = np.stack(depths)[:t]

    # Depth must return to the caller's (original-frame) geometry: fusion samples depth on the
    # original pixels, and a scaled grid would make _sample_depth go out of bounds or misalign
    # (Plan B review residual #1).
    if (model_h, model_w) != (orig_h, orig_w):
        depth = np.stack([
            cv2.resize(d, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            for d in depth
        ]).astype(np.float32)

    poses = np.tile(np.eye(4, dtype=np.float64), (t, 1, 1))  # poses come from the ViPE path; placeholder identity here
    # intrinsics are always the original-geometry intrinsics (consistent with the depth grid; model-input scaling does not change physical intrinsics)
    intr = np.tile(np.array([[fx_orig, 0, orig_w / 2], [0, fx_orig, orig_h / 2], [0, 0, 1]], np.float64), (t, 1, 1))
    save_arrays(
        output_path,
        {
            "camera_poses": poses,
            "intrinsics": intr,
            "depth": depth,
            "depth_source": np.array("stereo"),
            "n_frames": np.int32(t),
            "fps": data["fps"],
        },
    )
    return {"ok": True, "model": "ffs", "n_frames": t,
            "model_input_size": [int(model_h), int(model_w)],
            "depth_grid": [int(depth.shape[1]), int(depth.shape[2])]}


def main() -> None:
    args = parse_args("Fast-FoundationStereo: rectified stereo disparity -> metric depth")
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
