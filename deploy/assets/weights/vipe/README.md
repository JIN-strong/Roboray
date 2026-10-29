# ViPE weights (offline placement manifest)

On first run ViPE pulls weights from HuggingFace / GitHub / Google Drive itself, and the
cluster nodes have **no internet access**, so the weights must be fetched **on a machine with
internet access** and placed on the node in the layout below, then `TORCH_HOME` set to its
parent directory.

Validated layout (2026-10-20, `PoseOnlyAnnotationPipeline` run on an A800):

```
$TORCH_HOME/
└── hub/
    ├── geocalib/pinhole.tar                     # GeoCalib intrinsics estimation
    ├── moge2/moge-2-vitl.pt                     # MoGe-2 ViT-L, keyframe metric depth
    ├── droid_slam/droid.pth                     # DROID-SLAM frontend/backend
    └── checkpoints/superpoint_v6_from_tf.pth    # SuperPoint sparse tracking
```

## Sources and destinations

| Weight | Source | Size | Destination |
| --- | --- | --- | --- |
| GeoCalib pinhole | `https://github.com/cvg/GeoCalib/releases/download/v1.0/geocalib-pinhole.tar` | 116 MB | `hub/geocalib/pinhole.tar` |
| MoGe-2 ViT-L | `model.pt` from HF `Ruicheng/moge-2-vitl` | 1.30 GB | `hub/moge2/moge-2-vitl.pt` |
| DROID-SLAM | gdown id `1PpqVt1H4maBa_GbPJp4NwxRsd9jk-elh` (Google Drive) | 16 MB | `hub/droid_slam/droid.pth` |
| SuperPoint | `https://github.com/rpautrat/SuperPoint/raw/refs/heads/master/weights/superpoint_v6_from_tf.pth` | 2.6 MB | `hub/checkpoints/superpoint_v6_from_tf.pth` |

## Fetch commands (on a machine with internet access)

```bash
W=$TORCH_HOME/hub && mkdir -p $W/{geocalib,moge2,droid_slam,checkpoints}

curl -L -o $W/geocalib/pinhole.tar \
  https://github.com/cvg/GeoCalib/releases/download/v1.0/geocalib-pinhole.tar

curl -L -o $W/checkpoints/superpoint_v6_from_tf.pth \
  https://github.com/rpautrat/SuperPoint/raw/refs/heads/master/weights/superpoint_v6_from_tf.pth

# DROID-SLAM: ViPE internally uses gdown by id, equivalent to
gdown -O $W/droid_slam/droid.pth 1PpqVt1H4maBa_GbPJp4NwxRsd9jk-elh

# MoGe-2: when HF goes through an http proxy, be sure to unset all_proxy (a socks://
# proxy makes httpx fail outright)
env -u all_proxy -u ALL_PROXY python -c \
  "from huggingface_hub import hf_hub_download; \
   print(hf_hub_download('Ruicheng/moge-2-vitl','model.pt',local_dir='$W/moge2'))"
# afterwards rename $W/moge2/model.pt to moge-2-vitl.pt (ViPE's offline fast-path name)
mv $W/moge2/model.pt $W/moge2/moge-2-vitl.pt
```

## Correspondence (code provenance)

- `hub/geocalib/pinhole.tar` <- `vipe/priors/geocalib/extractor.py`:
  `torch.hub.load_state_dict_from_url(url, f"{torch.hub.get_dir()}/geocalib", file_name="pinhole.tar")`
- `hub/moge2/moge-2-vitl.pt` <- the **offline fast path** in `vipe/priors/depth/moge.py`:
  `Path(torch.hub.get_dir())/"moge2"/MOGE2_CHECKPOINT_NAMES[variant]`; on a hit no network
  is used; otherwise it falls back to `hf_hub_download("Ruicheng/moge-2-vitl", "model.pt")`
- `hub/droid_slam/droid.pth` <- `vipe/slam/networks/droid_net.py::load_weights`:
  `Path(torch.hub.get_dir())/"droid_slam"/"droid.pth"`; only gdowns when absent
- `hub/checkpoints/superpoint_v6_from_tf.pth` <- `vipe/slam/networks/superpoint.py`:
  `torch.hub.load_state_dict_from_url(...)`; torch's default cache dir is `hub/checkpoints/`

## Weights that are not needed

With `pipeline=pose_only` + `pipeline.init.instance=null`, TrackAnything
(GroundingDINO + SAM + AOT, several GB) is **not loaded**; pure pose/depth does not use it.
To run `pipeline=default` (includes per-frame dense depth and instance segmentation) you also
need `depth-anything/DA3-GIANT`, `lpiccinelli/unidepth-v2-vitl14`, etc., and `post.depth_align_model`
defaults to `adaptive_unidepth-l_svda`.

## Runtime environment variables

```bash
TORCH_HOME=<parent directory of the $TORCH_HOME above>
HF_HUB_OFFLINE=1          # weights are placed offline; forbid re-downloading (also avoids long retries offline)
LD_LIBRARY_PATH=$SP/torch/lib:$CUDA_HOME/lib64
```
