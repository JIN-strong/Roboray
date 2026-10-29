#!/usr/bin/env bash
set -euo pipefail
# Build a dedicated HaMeR env -- **run inside a container** (recipe validated end-to-end
# on an A800 on 2026-10-20).
#
# Usage (on the node):
#   sudo docker run --rm --network host --entrypoint bash \
#     --security-opt seccomp=unconfined -e OPENBLAS_NUM_THREADS=1 \
#     -v /home/ubuntu/ego:/ego ego-hamer:l2 /ego/build_hamer.sh
#
# Prerequisites:
#  1. `third_party/hamer` cloned (incl. the third-party/ViTPose submodule) to $EGO_HOME/hamer;
#  2. weights placed per deploy/assets/weights/hamer/README.md;
#  3. the base image has libegl1 installed (see deploy/Dockerfile.hamer) -- **HaMeR's
#     `hamer.utils` pulls pyrender/OpenGL at import time**, and a missing libEGL is an
#     outright ImportError.
#
# Why not a single `pip install -e third_party/hamer[all]` -- pitfalls hit in practice:
#  1. `hamer`'s install_requires includes `detectron2 @ git+...` and `mmcv==1.3.9`:
#     mmcv 1.3.9 is a 2021 release incompatible with torch 2.x and **will not install**;
#     detectron2 additionally needs another 2.4GB of ViTDet weights. So this script installs
#     hamer itself with `--no-deps` and handles the detection chain separately.
#  2. `chumpy`'s setup.py imports numpy at **build time**, so `--no-build-isolation` is required.
#  3. `chumpy` 0.70 executes `from numpy import bool, int, ...` at import time, but numpy>=1.24
#     removed those aliases -> a runtime numpy-alias shim is needed (see scripts/modelenvs/run_hamer.py).
#  4. An `-e` install without disabling build isolation makes pip fetch setuptools from
#     **pypi.org** and hang (the node has no internet) -> same `--no-build-isolation`.
#  5. `hamer.datasets` imports `webdataset` at the top level; it must be installed additionally.
#  6. `hamer.configs.CACHE_DIR_HAMER = "./_DATA"` is a constant **relative to CWD** -> at
#     runtime the CWD must land in the weights directory (run_hamer.py already os.chdir's).
#
# Detection chain (as honestly noted): upstream uses detectron2(ViTDet) + mmpose(ViTPose) to
# get hand keypoints then draw boxes. Neither installs in this environment, so run_hamer.py
# uses **mediapipe hand detection** as a box-drawing substitute (the MANO fitting part is
# unchanged). On a machine where detectron2+mmpose install, you can switch back to the
# upstream chain yourself.

EGO_HOME="${EGO_HOME:-/ego}"
VENV="${VENV:-$EGO_HOME/hamer-venv}"
HAMER_SRC="${HAMER_SRC:-$EGO_HOME/hamer}"
BASE_PY="${BASE_PY:-$EGO_HOME/ego-models/bin/python}"
PIP_INDEX="${PIP_INDEX:-https://pypi.org/simple}"
PIP_HOST="${PIP_HOST:-pypi.org}"
PIPARGS=(-i "$PIP_INDEX" --trusted-host "$PIP_HOST" --timeout 180 --retries 6)

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=4

echo "### 0. Create venv (reuse ego-models' torch, saving ~2.5GB of download)"
"$BASE_PY" -V
rm -rf "$VENV"
"$BASE_PY" -m venv --system-site-packages "$VENV"
V="$VENV/bin/python"
"$V" -c "import torch;print('torch', torch.__version__, torch.version.cuda)"
"$V" -m pip install "${PIPARGS[@]}" -U pip setuptools wheel

echo "### 1. HaMeR runtime deps (no detectron2 / mmcv / mmpose; see header)"
"$V" -m pip install "${PIPARGS[@]}" \
  "numpy<2" opencv-python-headless pyrender pytorch-lightning scikit-image \
  "smplx==0.1.28" yacs timm einops xtcocotools pandas gdown \
  hydra-core rich pyrootutils webdataset "mediapipe==0.10.14"

echo "### 2. chumpy (required to deserialize MANO_RIGHT.pkl; --no-build-isolation per header item 2)"
"$V" -m pip install "${PIPARGS[@]}" --no-build-isolation chumpy

echo "### 3. hamer itself (--no-deps --no-build-isolation, per header items 1 and 4)"
"$V" -m pip install "${PIPARGS[@]}" --no-deps --no-build-isolation -e "$HAMER_SRC"

echo "### 4. Self-check"
"$V" -c "import chumpy; print('chumpy OK')"
"$V" -c "import webdataset; print('webdataset OK')"
"$V" -c "import hamer; print('hamer OK', hamer.__file__)"
"$V" -c "from hamer.models import load_hamer, DEFAULT_CHECKPOINT; print('load_hamer OK')"

cat <<EOF

Done. Set:
  ROBORAY_HAMER_PYTHON=$V
  ROBORAY_HAMER_WEIGHTS=<directory containing _DATA/ and mano/, see deploy/assets/weights/hamer/README.md>
  PYOPENGL_PLATFORM=egl        # hamer.utils pulls pyrender, must go through EGL

  The script supports --serve resident mode: weights load once and are reused across Episodes (controlled by ROBORAY_MODEL_WORKER, default resident)
EOF
