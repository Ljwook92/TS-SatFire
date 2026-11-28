#!/usr/bin/env bash
set -euo pipefail
run=$1
cd "$run/code"
exec 9>"$run/run.lock"
flock -n 9 || { echo "This run already has a worker: $run" >&2; exit 1; }
[[ ! -f "$run/complete" ]] || { cat "$run/summary.csv"; exit 0; }
export PYTHONPATH="$PWD/shared" PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
py=$SATFIRE_PYTHON
"$py" -c 'import rasterio, torch; print(torch.cuda.get_device_name(0), torch.ones(1, device="cuda").item(), flush=True)'
read -r epochs batch < "$run/settings"
if [[ ! -f "$run/cache/manifest.csv" ]]; then
  "$py" build_cache.py --cache "$run/cache" --raw "$TS_SATFIRE_RAW_ROOT" --assets assets --firms assets/firms.csv.gz
fi
if [[ ! -f "$run/fit_done" ]]; then
  "$py" window_run.py fit --cache "$run/cache" --run "$run" --epochs "$epochs" --batch "$batch"
fi
if [[ ! -f "$run/summary.csv" ]]; then
  "$py" window_run.py evaluate --cache "$run/cache" --run "$run" --batch "$batch"
fi
"$py" export.py "$run"
touch "$run/complete"
