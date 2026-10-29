#!/usr/bin/env bash
set -euo pipefail
# Build a dedicated Fast-FoundationStereo env (py3.12 + torch2.6 cu124 + xformers).
#
# Prerequisites: run scripts/vendor_models.sh first to clone and pin third_party/fast-foundationstereo.
# Output: conda env `ego-ffs`; then point ROBORAY_FFS_PYTHON at its python (see the echo at the end).
# Weights: download from HuggingFace nvidia/c-fast-foundationstereo into ROBORAY_FFS_WEIGHTS.
conda create -y -n ego-ffs python=3.12
conda run -n ego-ffs python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
conda run -n ego-ffs python -m pip install xformers --index-url https://download.pytorch.org/whl/cu124
VENDOR_DIR="$(cd "$(dirname "$0")/../.." && pwd)/third_party/fast-foundationstereo"
conda run -n ego-ffs python -m pip install -e "$VENDOR_DIR"
echo "Done. Set ROBORAY_FFS_PYTHON=$(conda info --base)/envs/ego-ffs/bin/python"
echo "Weights: download from HF nvidia/c-fast-foundationstereo into \$ROBORAY_FFS_WEIGHTS"
echo "The script supports --serve resident mode: weights load once and are reused across Episodes (controlled by ROBORAY_MODEL_WORKER, default resident)"
