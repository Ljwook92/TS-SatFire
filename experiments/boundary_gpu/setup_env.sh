#!/usr/bin/env bash
set -euo pipefail
prefix=${SATFIRE_GPU_ENV:-/mnt/pixstor/data/$USER/SatFire/envs/boundary-gpu-v1}
if [[ ! -x "$prefix/bin/python" ]]; then
  mkdir -p "$(dirname "$prefix")"
  conda create -y --prefix "$prefix" --override-channels -c conda-forge python=3.11 pip
fi
export PYTHONNOUSERSITE=1
python_bin="$prefix/bin/python"
"$python_bin" -m pip install 'numpy==1.26.4' 'pandas==2.2.3' 'scipy==1.14.1' 'matplotlib==3.9.4' 'rasterio==1.4.3' 'pyproj==3.7.0'
"$python_bin" -m pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
"$python_bin" -c 'import torch, rasterio, pandas, scipy, matplotlib, pyproj; assert torch.version.cuda; print("Environment ready:", torch.__version__, "CUDA runtime", torch.version.cuda)'
