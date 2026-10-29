"""HaMeR inference script (runs in the hamer env).

**Real API (verified item-by-item on an A800 GPU, replacing all `[verify on GPU]` assumptions)**:

1. `from hamer.models import load_hamer` -> `model, model_cfg = load_hamer(<weights>/hamer.ckpt)`
   (the weights dir is the `_DATA` extracted by the official `fetch_demo_data.sh`, with
   `MANO_RIGHT.pkl` under the `mano/` subdirectory).
2. Detection boxes -> `hamer.datasets.vitdet_dataset.ViTDetDataset(model_cfg, img_bgr, boxes, right,
   rescale_factor=2.0)` + `DataLoader` -> `out = model(batch)`.
   `out` keys: `pred_cam (B,3)`, `pred_cam_t (B,3)`, `pred_mano_params`,
   `pred_keypoints_3d (B,21,3)`, `pred_keypoints_2d (B,21,2)`, `pred_vertices`.
3. `pred_mano_params` = `{'global_orient': (B,1,3,3), 'hand_pose': (B,15,3,3),
   'betas': (B,10)}` - **all rotation matrices and contain no transl** (`hamer/models/heads/mano_head.py:108`).
   `hamer/models/hamer.py:120` forwards them to MANO with `pose2rot=False`, so
   `pred_keypoints_3d` are **root-relative** joints.
4. Full-image camera translation: `hamer.utils.renderer.cam_crop_to_full(pred_cam, box_center, box_size,
   img_size, focal_length)`, and `pred_cam[:,1] *= (2*right-1)` must run before the call.
   **Camera-frame joints = pred_keypoints_3d + cam_t_full[:,None,:]**.
5. Scale: `cam_crop_to_full` computes `tz = 2*f/bs`, proportional to f. The official demo uses
   `FOCAL_LENGTH/IMAGE_SIZE*img_size.max()` (a prior focal length, ~25000 px), yielding
   **relative scale**. This script uses the **measured focal length fx** when `calib` provides K,
   giving **metric** translation - exactly the unit this repo's contract requires.

**Detection-box source (important, stated as-is)**: the HaMeR official demo uses detectron2(ViTDet)
+ mmpose(ViTPose) to extract hand keypoints and derive boxes. Neither installs in this environment
(`mmcv==1.3.9` is incompatible with torch 2.7; ViTDet weights alone are 2.4 GB), so this script
takes boxes in this order:
   a) `--extra {"hand_boxes": [[x1,y1,x2,y2,is_right], ...]}` - explicitly given (reproducible)
   b) `mediapipe` hand detection (**a substitute, not the official chain**; HaMeR's MANO fitting is unchanged)
   c) otherwise it errors out explaining what is missing

**candidates protocol (detection-side data premise)**: npz optional key `candidates` - an object
array (T,) whose elements are list[dict] (`bbox` (4,), `side` int (0=left 1=right), `score` float,
`joints_2d` (21,3) (u,v,score, same convention as the 3rd dim of the main array), `joints_3d_cam` (21,3)).
**bbox and joints_2d u,v are full-image pixel coordinates** (same projection convention as the main
array's joints_2d). The default detection cap is `max_hands=4` (configurable via payload/extra
`max_hands`, constraining only the mediapipe detector); **same-side multiple detections are no
longer dropped** - all go into candidates. The main array is still filled with "first left / first
right" - the first-detection semantics are unchanged, so the main-array output stays byte-compatible
(extra keys are consumed downstream).
"""

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

from _common import RESULT_MARKER, compute_stride, load_input, parse_args, save_arrays

N_JOINTS = 21
HANDS = 2  # main-array slot count (0=left 1=right) - shape semantics unchanged; all detections go to the candidates key
DEFAULT_MAX_HANDS = 4  # default detection cap (configurable via payload/extra max_hands)
_HAMER_HINT = (
    "The weights directory must contain hamer.ckpt and mano/MANO_RIGHT.pkl: "
    "hamer.ckpt comes from the _DATA extracted by third_party/hamer/fetch_demo_data.sh; "
    "MANO_RIGHT.pkl must be downloaded after registering at https://mano.is.tue.mpg.de."
)


def _install_numpy_shim() -> None:
    """chumpy 0.70 runs `from numpy import bool, int, float, ...` at import time, and
    numpy>=1.24 removed those aliases. MANO_RIGHT.pkl is a chumpy pickle, so deserialization
    must be able to import chumpy - restore the aliases first."""
    for name, typ in (("bool", bool), ("int", int), ("float", float), ("complex", complex),
                      ("object", object), ("str", str), ("unicode", str), ("long", int)):
        if not hasattr(np, name):
            setattr(np, name, typ)


def _rotmat_to_axis_angle(rotmat: np.ndarray) -> np.ndarray:
    """(...,3,3) rotation matrices -> (...,3) axis-angle. Uses scipy (available in the hamer env)."""
    from scipy.spatial.transform import Rotation

    flat = np.asarray(rotmat, np.float64).reshape(-1, 3, 3)
    return Rotation.from_matrix(flat).as_rotvec().reshape(*rotmat.shape[:-2], 3).astype(np.float32)


def _boxes_from_mediapipe(img_bgr: np.ndarray, mirrored: bool,
                          max_num_hands: int = DEFAULT_MAX_HANDS) -> list:
    """mediapipe hand detection -> [[x1,y1,x2,y2,is_right,score], ...] (not the official chain; see file header).

    max_num_hands is the detection cap (default 4, configurable); it is no longer constrained by the
    main array's HANDS=2 - extra detections only affect candidates, and the main array is still filled
    with "first left / first right".
    """
    import mediapipe as mp

    with mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=max_num_hands,
                                  min_detection_confidence=0.3) as hands:
        res = hands.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    out = []
    if not res.multi_hand_landmarks:
        return out
    h, w = img_bgr.shape[:2]
    for lm, handed in zip(res.multi_hand_landmarks, res.multi_handedness or []):
        xs = [p.x * w for p in lm.landmark]
        ys = [p.y * h for p in lm.landmark]
        pad = 0.15 * max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
        # mediapipe's left/right labels assume a mirrored image; flip the labels for non-mirrored input.
        label, score = "Right", 0.0
        if handed.classification:
            label = handed.classification[0].label
            score = float(handed.classification[0].score)
        is_right = 1 if label == "Right" else 0
        if not mirrored:
            is_right = 1 - is_right
        out.append([min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad,
                    float(is_right), score])
    return out


def build_model(weights: str, extra: dict):
    """Load the HaMeR model, returning a ``(model, model_cfg, device)`` tuple.

    The serve loop treats it as an opaque ``model`` and rebuilds it keyed by ``(weights, extra)``;
    infer_once unpacks it. **The numpy shim and os.chdir(weights_dir) must live here** -
    `hamer.configs.CACHE_DIR_HAMER = "./_DATA"` is a CWD-relative constant, and
    get_config(update_cachedir=True) joins MANO.MODEL_PATH/MEAN_PARAMS onto `./_DATA/...`;
    in serve mode main() does not run first, so only build_model can change CWD.
    """
    if not weights or not Path(weights).exists():
        raise ValueError(f"HaMeR weights directory is not set or does not exist. {_HAMER_HINT}")

    weights_dir = Path(weights).resolve()
    ckpt = Path(extra.get("checkpoint") or (weights_dir / "_DATA" / "hamer_ckpts"
                                            / "checkpoints" / "hamer.ckpt"))
    if not ckpt.exists():
        raise ValueError(f"HaMeR checkpoint not found: {ckpt}. {_HAMER_HINT}")

    # `hamer.configs.CACHE_DIR_HAMER = "./_DATA"` is a **CWD-relative** constant, and
    # `get_config(..., update_cachedir=True)` joins MANO.MODEL_PATH / MEAN_PARAMS onto
    # `./_DATA/data/mano` and `./_DATA/data/mano_mean_params.npz`.
    # So CWD must land on the weights directory - otherwise MANO looks in the wrong place.
    os.chdir(weights_dir)

    _install_numpy_shim()
    import torch

    from hamer.models import load_hamer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_cfg = load_hamer(str(ckpt))
    model = model.to(device).eval()
    mano = model.mano
    return model, model_cfg, device


def infer_once(model, input_path: str, output_path: str, extra: dict) -> dict:
    """Infer once on one input: per-frame boxes + MANO fitting -> assemble and write the npz -> return summary."""
    data = load_input(input_path)
    frames = data["frames"]
    t, h, w = int(data["n_frames"]), frames.shape[1], frames.shape[2]
    fps = float(data["fps"])
    stride = compute_stride(fps, extra.get("inference_fps") or data.get("inference_fps"))

    model, model_cfg, device = model
    import torch

    from hamer.datasets.vitdet_dataset import ViTDetDataset
    from hamer.utils import recursive_to
    from hamer.utils.renderer import cam_crop_to_full

    # mano must be bound inside this function (build_model also binds it once, but the serve loop
    # passes the build_model return value to this function as an opaque model, so that local binding
    # never reaches here) - if `mano = model.mano` only happened in build_model, the first `mano(...)`
    # in the per-frame loop would NameError in a real env.
    mano = model.mano

    explicit_boxes = extra.get("hand_boxes")  # [[x1,y1,x2,y2,is_right], ...] per frame
    mirrored = bool(extra.get("mirrored", False))
    # Detection cap: configurable via payload/extra, default 4. It only constrains the mediapipe
    # detector; explicit hand_boxes are caller-supplied and are not truncated - all go into candidates.
    max_hands = int(extra.get("max_hands") or data.get("max_hands") or DEFAULT_MAX_HANDS)

    # Focal length: prefer calib's measured fx (-> metric translation); fall back to the demo's prior
    # focal length (relative scale) when missing.
    k_mat = extra.get("K") or data.get("K")
    fx_measured = float(np.asarray(k_mat)[0, 0]) if k_mat is not None else None

    hand_present = np.zeros((t, HANDS), bool)
    joints_2d = np.zeros((t, HANDS, N_JOINTS, 3), np.float32)
    joints_3d_cam = np.zeros((t, HANDS, N_JOINTS, 3), np.float32)
    mano_pose = np.zeros((t, HANDS, 48), np.float32)
    mano_shape = np.zeros((t, HANDS, 10), np.float32)
    mano_trans = np.zeros((t, HANDS, 3), np.float32)
    confidence = np.zeros((t, HANDS), np.float32)
    candidates_rows: list = [[] for _ in range(t)]  # list[dict] per frame, holding all detections

    n_boxes_total = 0
    for i in range(t):
        if stride > 1 and i % stride != 0:
            continue  # skip HaMeR for non-sampled frames: present=False, filled by short-gap completion (same as the fake contract)
        img_bgr = cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR)
        if explicit_boxes is not None:
            rows = explicit_boxes[i] if i < len(explicit_boxes) else []
        else:
            rows = _boxes_from_mediapipe(img_bgr, mirrored, max_num_hands=max_hands)
        if not rows:
            continue
        boxes = np.asarray([[r[0], r[1], r[2], r[3]] for r in rows], np.float32)
        right = np.asarray([int(r[4]) for r in rows], np.int32)
        box_scores = np.asarray([float(r[5]) if len(r) > 5 else 0.0 for r in rows], np.float32)
        n_boxes_total += len(rows)

        dataset = ViTDetDataset(model_cfg, img_bgr, boxes, right, rescale_factor=2.0)
        loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
        for batch in loader:
            batch = recursive_to(batch, device)
            with torch.no_grad():
                out = model(batch)

            multiplier = (2 * batch["right"] - 1)
            pred_cam = out["pred_cam"].clone()
            pred_cam[:, 1] = multiplier * pred_cam[:, 1]
            img_size = batch["img_size"].float()
            # Build the focal length as a (B,) 1-D tensor: the official
            # `FOCAL_LENGTH/IMAGE_SIZE*img_size.max()` is a 0-dim scalar, and indexing it as
            # focal[n] would raise "invalid index of a 0-dim tensor".
            f_value = (float(fx_measured) if fx_measured is not None
                       else float(model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE
                                  * img_size.max()))
            focal = torch.full((img_size.shape[0],), f_value, device=device)
            cam_t_full = cam_crop_to_full(pred_cam, batch["box_center"].float(),
                                          batch["box_size"].float(), img_size, focal).cpu().numpy()

            params = out["pred_mano_params"]
            with torch.no_grad():
                mano_out = mano(global_orient=params["global_orient"].float(),
                                hand_pose=params["hand_pose"].float(),
                                betas=params["betas"].float(),
                                pose2rot=False)
            joints_root_rel = mano_out.joints.detach().cpu().numpy()  # (B,21,3) root-relative

            g_orient = params["global_orient"].detach().cpu().numpy()  # (B,1,3,3)
            h_pose = params["hand_pose"].detach().cpu().numpy()        # (B,15,3,3)
            betas = params["betas"].detach().cpu().numpy()             # (B,10)
            # Confidence uses the **detector's real score** (mediapipe handedness score or the 6th
            # field of explicit boxes); not a placeholder 1.0 - that would disguise "no score" as "full score".
            scores = np.asarray(box_scores, np.float32)

            for n in range(joints_root_rel.shape[0]):
                side = int(batch["right"][n].cpu().item())
                score_n = float(scores[n]) if n < len(scores) else 0.0
                trans = cam_t_full[n]  # full-image camera translation (metric when K is available)
                j3d_cam = (joints_root_rel[n] + trans[None, :]).astype(np.float32)
                # 2D: project the camera-frame 3D back to the full image with the same focal length
                # and principal point (same convention as cam_crop_to_full)
                z = np.clip(j3d_cam[:, 2], 1e-6, None)
                j2d = np.stack([
                    focal[n].item() * j3d_cam[:, 0] / z + w / 2.0,
                    focal[n].item() * j3d_cam[:, 1] / z + h / 2.0,
                    np.full(N_JOINTS, score_n, np.float32),
                ], axis=-1).astype(np.float32)
                # Same-side multiple detections are no longer dropped - every detection goes to candidates
                # (bbox/joints are filled from the same source as the main array: same batch output, same projection convention).
                candidates_rows[i].append({
                    "bbox": boxes[n].astype(np.float32),
                    "side": side,
                    "score": score_n,
                    "joints_2d": j2d,
                    "joints_3d_cam": j3d_cam,
                })
                # The main array is still filled with "first left / first right" (pre-change "first"
                # meant the first unoccupied detection in order - same semantics), keeping the
                # main-array output byte-compatible with before.
                if hand_present[i, side]:
                    continue
                hand_present[i, side] = True
                confidence[i, side] = score_n
                mano_trans[i, side] = trans
                joints_3d_cam[i, side] = j3d_cam
                mano_shape[i, side] = betas[n].astype(np.float32)
                mano_pose[i, side] = np.concatenate([
                    _rotmat_to_axis_angle(g_orient[n]).reshape(-1),
                    _rotmat_to_axis_angle(h_pose[n]).reshape(-1),
                ]).astype(np.float32)
                joints_2d[i, side] = j2d

    # Write candidates: an object array (T,) whose elements are list[dict] (protocol in the file
    # header). np.array(list-of-lists, dtype=object) collapses into 2-D when the sublists are
    # equal-length, so assign per frame.
    candidates = np.empty(t, dtype=object)
    for i, row in enumerate(candidates_rows):
        candidates[i] = row
    save_arrays(
        output_path,
        {
            "hand_present": hand_present,
            "joints_2d": joints_2d,
            "joints_3d_cam": joints_3d_cam,
            "mano_pose": mano_pose,
            "mano_shape": mano_shape,
            "mano_trans": mano_trans,
            "confidence": confidence,
            "candidates": candidates,
            "n_frames": np.int32(t),
            "fps": np.int32(round(fps)),
        },
    )
    return {"ok": True, "model": "hamer", "n_frames": t,
            "detection_rate": float(hand_present.any(axis=1).mean()),
            "n_boxes": n_boxes_total,
            "n_candidates": int(sum(len(r) for r in candidates_rows)),
            "focal_px": fx_measured,
            "box_source": "explicit" if explicit_boxes is not None else "mediapipe",
            "joints_3d_finite": bool(np.isfinite(joints_3d_cam).all()),
            "mano_trans_median_norm": float(np.median(np.linalg.norm(mano_trans[hand_present], axis=-1)))
            if hand_present.any() else None}


def main() -> None:
    args = parse_args("HaMeR: egocentric hand detection + MANO fitting")
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
