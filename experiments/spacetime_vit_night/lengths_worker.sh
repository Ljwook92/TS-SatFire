#!/usr/bin/env bash
set -euo pipefail
parent=$1
index=${2:-${SLURM_ARRAY_TASK_ID:?Missing array index}}
case "$index" in
  0) ts=2; arm=day ;;
  1) ts=2; arm=night ;;
  2) ts=6; arm=day ;;
  3) ts=6; arm=night ;;
  *) exit 2 ;;
esac
out="$parent/lengths_2_4_6"
cd "$out/code"
run="$out/ts$ts-$arm"
mkdir -p "$run"
exec 9>"$run/run.lock"
flock -n 9 || { echo "Already running: $run" >&2; exit 1; }
export PYTHONPATH="$PWD/shared" PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
read -r epochs batch < settings
args=(--ts "$ts" --run "$run" --cache "$parent/cache" --batch "$batch")
if [[ "$arm" == day ]]; then args+=(--day-only); fi
echo "TS=$ts inputs=$arm epochs=$epochs batch=$batch run=$run"
if [[ ! -f "$run/fit_done" ]]; then
  "$SATFIRE_PYTHON" window_run.py fit "${args[@]}" --epochs "$epochs"
fi
if [[ ! -s "$run/summary.csv" ]]; then
  "$SATFIRE_PYTHON" window_run.py evaluate "${args[@]}"
fi
