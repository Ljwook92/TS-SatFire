#!/usr/bin/env bash
set -euo pipefail
stage=$1
run=$2
cd "$run/code"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR="$run/mpl-${SLURM_JOB_ID:-local}"
case "$stage" in
  prepare) "$BOUNDARY_PYTHON" prepare.py --raw "$TS_SATFIRE_RAW_ROOT" --cache "$run/cache" ;;
  train) "$BOUNDARY_PYTHON" train.py --ts "$SLURM_ARRAY_TASK_ID" --cache "$run/cache" --run "$run" ;;
  report) "$BOUNDARY_PYTHON" report.py --run "$run" ;;
  *) exit 2 ;;
esac
