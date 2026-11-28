#!/usr/bin/env bash
set -euo pipefail
stage=$1
run=$2
cd "$run/code"
exec 9>"$run/$stage.lock"
flock 9
py="${SATFIRE_PYTHON:-/home/$USER/data/SatFire/envs/boundary-gpu-v1/bin/python}"
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
if [[ "$stage" == prepare ]]; then
  if [[ ! -f "$run/cache/manifest.csv" ]]; then
    "$py" prepare.py --raw "$TS_SATFIRE_RAW_ROOT" --cache "$run/cache" --firms assets/firms.csv.gz
  fi
else
  if [[ ! -f "$run/fit_done" ]]; then
    "$py" run.py fit --cache "$run/cache" --run "$run" --device cuda --epochs "$VIT_EPOCHS" --batch "$VIT_BATCH"
  fi
  if [[ ! -f "$run/summary.csv" ]]; then
    "$py" run.py evaluate --cache "$run/cache" --run "$run" --device cuda --batch "$VIT_BATCH"
  fi
  "$py" export_maps.py "$run/cache" "$run"
  touch "$run/complete"
fi
