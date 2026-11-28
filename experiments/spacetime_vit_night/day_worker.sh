#!/usr/bin/env bash
set -euo pipefail
night=$1
day="$night/day_control"
exec 9>"$day/run.lock"
flock -n 9 || { echo 'Day control already running.' >&2; exit 1; }
export PYTHONPATH="$night/code:$night/code/shared" PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
py=$SATFIRE_PYTHON
while [[ ! -f "$night/cache/manifest.csv" ]]; do
  if flock -n "$night/run.lock" true && [[ -z "$(squeue -h -u "$USER" -n vit-night -o '%o' | grep -F "$night/code/worker.sh" || true)" && ! -f "$night/cache/manifest.csv" ]]; then
    echo "Night preprocessing stopped without a manifest. Resume the night run: $night" >&2
    exit 1
  fi
  sleep 30
done
read -r epochs batch < "$night/settings"
common=(--day-only --cache "$night/cache" --run "$day" --batch "$batch")
if [[ ! -f "$day/fit_done" ]]; then
  "$py" -c 'import torch; print(torch.cuda.get_device_name(0))'
  "$py" "$day/window_run.py" fit "${common[@]}" --epochs "$epochs"
fi
if [[ ! -f "$day/summary.csv" ]]; then
  "$py" "$day/window_run.py" evaluate "${common[@]}"
fi
if [[ -f "$night/summary.csv" ]]; then
  "$py" "$day/compare_af.py" "$night"
else
  echo "Day control finished. After the night evaluation: bash day_control.sh report '$night'"
fi
