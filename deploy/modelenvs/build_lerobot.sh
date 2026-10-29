#!/usr/bin/env bash
set -euo pipefail
# Build the LeRobot export env (CPU torch; lerobot 0.6.1 hard-requires torch>=2.7 and
# numpy>=2.0, which conflicts with the main env's numpy<2.0, so it must be separate).
# python=3.12 (not 3.10): since lerobot>=0.5 declares Requires-Python>=3.12, under py3.10
# pip reports "No matching distribution found for lerobot==0.6.1" (0.6.1 is filtered out by
# the python version).
conda create -y -n ego-lerobot python=3.12
conda run -n ego-lerobot python -m pip install "torch>=2.7" torchvision --index-url https://download.pytorch.org/whl/cpu
conda run -n ego-lerobot python -m pip install "lerobot[dataset]==0.6.1"
conda run -n ego-lerobot python -c "import lerobot, torch; print('lerobot', lerobot.__version__, 'torch', torch.__version__)"
echo "Done. Set ROBORAY_LEROBOT_PYTHON=$(conda info --base)/envs/ego-lerobot/bin/python"
