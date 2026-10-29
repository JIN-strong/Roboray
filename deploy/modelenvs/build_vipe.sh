#!/usr/bin/env bash
set -euo pipefail
# Build a dedicated ViPE env -- **run inside a container** (recipe validated end-to-end on
# an A800 on 2026-10-20).
#
# Usage (on the node):
#   sudo docker run --rm --network host --entrypoint bash \
#     --security-opt seccomp=unconfined \
#     -e OPENBLAS_NUM_THREADS=1 -v /home/ubuntu/ego:/ego \
#     registry.example.com/roboray/base:v0.26.0 /ego/build_vipe.sh
#
# Prerequisites: third_party/vipe is cloned and pinned (scripts/vendor_models.sh); the two
# conda packages needed for the "CUDA toolchain" step below are already placed in $EGO_HOME/.
#
# Why a single `pip install -e third_party/vipe` does not work -- four pitfalls hit in
# practice (details in docs/superpowers/plans/D-l2-results.md):
#  1. ViPE requires Python >= 3.11 (its pyproject is `>=3.11,<3.15`); the base image's
#     system python 3.12 works, so **no conda env is used**; a plain venv is created instead.
#  2. Its csrc/ holds DROID-SLAM's CUDA operators (corr_ext / droid_net_ext / lietorch_ext /
#     slam_ext / scatter_ext / utils_ext), which **must be compiled** and need nvcc.
#  3. The PyPI `nvidia-cuda-nvcc-cu12` wheel's `bin/` **has only ptxas, no nvcc**; the
#     `nvidia-cuda-cccl-cu12` wheel has **no `nv/`** (`cuda_fp16.h` errors with
#     `fatal error: nv/target`). Both must be taken as full packages from the **conda nvidia
#     channel**.
#  4. Inside the container pip reports `RuntimeError: can't start new thread`: the **real
#     cause is seccomp** (`/proc/self/status` shows `Seccomp: 2`; even the first thread
#     cannot be created), unrelated to ulimit/cgroup -> `--security-opt seccomp=unconfined`
#     is required.
#
# Also note: on the internal mirror `hydra-core` / `omegaconf` resolve to 2019-era
# 0.11.3 / 1.4.1, which crashes `import vipe` at `OmegaConf.has_resolver` -> the versions
# must be pinned explicitly. `gtsam-develop==4.3a2.dev202609030306` (the dev version ViPE
# specifies) has no source on the mirror, so the official `gtsam==4.3.0` is used instead.

EGO_HOME="${EGO_HOME:-/ego}"
VENV="${VENV:-$EGO_HOME/vipe-venv}"
CUDA_HOME_TARGET="${CUDA_HOME_TARGET:-$EGO_HOME/cuda-12.4}"
VIPE_SRC="${VIPE_SRC:-$EGO_HOME/vipe}"
PY="${PY:-/usr/bin/python3}"
PIP_INDEX="${PIP_INDEX:-https://pypi.org/simple}"
PIP_HOST="${PIP_HOST:-pypi.org}"
PIPARGS=(-i "$PIP_INDEX" --trusted-host "$PIP_HOST" --timeout 180 --retries 6)

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=4

echo "### 0. Create venv (system python 3.12 satisfies ViPE's >=3.11)"
"$PY" -V
[ -d "$VENV" ] || "$PY" -m venv "$VENV"
V="$VENV/bin/python"

echo "### 1. Runtime deps (versions must be pinned explicitly; see header)"
"$V" -m pip install "${PIPARGS[@]}" -U pip
"$V" -m pip install "${PIPARGS[@]}" \
  torch torchvision \
  click einops gdown hydra-core==1.3.2 omegaconf==2.3.1 \
  huggingface_hub imageio imageio-ffmpeg kornia matplotlib ninja numpy \
  opencv-python "OpenEXR<3.3.0" pydantic pillow python-pycg ray rerun-sdk \
  safetensors scipy timm tqdm transformers viser gtsam setuptools wheel

echo "### 2. CUDA toolchain: nvcc (conda pkg) + CCCL (conda pkg)"
# Both conda packages must be placed in $EGO_HOME/ beforehand (the build machine has
# internet access; fetch from https://conda.anaconda.org/nvidia/linux-64/ and upload):
#   cuda-nvcc-12.4.131-0.tar.bz2   (contains bin/nvcc, nvvm/bin/cicc, nvvm/libdevice)
#   cuda-cccl-12.4.127-0.tar.bz2   (contains include/nv/target)
NVCC_TAR="${NVCC_TAR:-$EGO_HOME/cuda-nvcc-12.4.131-0.tar.bz2}"
CCCL_TAR="${CCCL_TAR:-$EGO_HOME/cuda-cccl-12.4.127-0.tar.bz2}"
for f in "$NVCC_TAR" "$CCCL_TAR"; do
  [ -f "$f" ] || { echo "missing $f (see header item 3)" >&2; exit 1; }
done

mkdir -p "$CUDA_HOME_TARGET"
tar -xjf "$NVCC_TAR" -C "$CUDA_HOME_TARGET" --exclude='info/*' --no-same-owner
tar -xjf "$CCCL_TAR" -C "$CUDA_HOME_TARGET" --exclude='info/*' --exclude='lib/cmake/*' --no-same-owner

SP="$VENV/lib/python3.12/site-packages"
echo "### 3. Merge each nvidia wheel's include/lib in the venv into CUDA_HOME"
# torch's ATen headers include cusparse.h / cublas.h etc.; those headers only exist in the
# nvidia-*-cu12 wheels.
for d in "$SP"/nvidia/*/; do
  [ -d "$d/include" ] && cp -rn "$d/include/." "$CUDA_HOME_TARGET/include/" 2>/dev/null || true
  [ -d "$d/lib" ] && cp -rn "$d/lib/." "$CUDA_HOME_TARGET/lib64/" 2>/dev/null || true
done
# The wheels only ship lib*.so.<ver>; the linker wants the unversioned names.
(cd "$CUDA_HOME_TARGET/lib64"
 for f in lib*.so.*; do
   base="$(echo "$f" | sed -E 's/\.so\..*$/.so/')"
   [ -e "$base" ] || ln -sf "$f" "$base"
 done)

export CUDA_HOME="$CUDA_HOME_TARGET"
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"   # A800 = sm80
export MAX_JOBS="${MAX_JOBS:-16}"
export LD_LIBRARY_PATH="$SP/torch/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

echo "### 4. Verify the toolchain"
"$CUDA_HOME/bin/nvcc" --version | grep release
"$V" -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

echo "### 5. Compile and install ViPE (editable; --no-deps because deps are already installed in step 1)"
"$V" -m pip install --no-deps --no-build-isolation -e "$VIPE_SRC"

echo "### 6. Self-check"
"$V" -c "import vipe, vipe.pipeline, vipe.slam, vipe.ext, vipe_ext; print('vipe OK')"

cat <<EOF

Done. Set:
  ROBORAY_VIPE_PYTHON=$V
  CUDA_HOME=$CUDA_HOME_TARGET
  LD_LIBRARY_PATH=$SP/torch/lib:$CUDA_HOME_TARGET/lib64
  TORCH_HOME=$EGO_HOME/torch_home     # weights cache, see deploy/assets/weights/vipe/README.md
  HF_HUB_OFFLINE=1                    # weights are placed offline; forbid re-downloading

  The script supports --serve resident mode: weights load once and are reused across Episodes (controlled by ROBORAY_MODEL_WORKER, default resident)
EOF
