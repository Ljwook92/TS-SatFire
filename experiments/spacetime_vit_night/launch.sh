#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mode=${1:-interactive}
case "$mode" in
  interactive|batch|snapshot) ;;
  *) echo 'Usage: bash launch.sh interactive|batch [existing run]' >&2; exit 2 ;;
esac
export TS_SATFIRE_RAW_ROOT=${TS_SATFIRE_RAW_ROOT:-/home/$USER/data/SatFire/ts-satfire}
export SATFIRE_PYTHON=${SATFIRE_PYTHON:-/home/$USER/data/SatFire/envs/boundary-gpu-v1/bin/python}
[[ -x "$SATFIRE_PYTHON" && -d "$TS_SATFIRE_RAW_ROOT/camp_fire" ]]
if [[ $# -gt 1 ]]; then
  run=$2
  [[ -f "$run/code/worker.sh" && -f "$run/settings" ]]
else
  root=${VIT_RUNS:-/home/$USER/data/SatFire/spacetime_vit_runs}
  mkdir -p "$root"
  run=$(mktemp -d "$root/ts4-night-window-XXXXXX")
  shared=../spacetime_vit
  mkdir -p "$run/code/shared" "$run/code/assets"
  cp *.py worker.sh official_af_hashes.csv "$run/code/"
  cp "$shared"/{prepare,model,run}.py "$run/code/shared/"
  cp "$shared"/assets/{split.csv,af_firms.csv.gz,af_thresholds.json,overlap.csv} "$run/code/assets/"
  cp ../boundary_gpu/assets/firms.csv.gz "$run/code/assets/"
  git rev-parse HEAD > "$run/commit.txt"
  echo "${VIT_EPOCHS:-100} ${VIT_BATCH:-1}" > "$run/settings"
fi
echo "Run: $run"
if [[ "$mode" == snapshot ]]; then
  exit 0
fi
if [[ "$mode" == batch ]]; then
  sbatch --job-name=vit-night --partition="${SATFIRE_GPU_PARTITION:-gpu}" \
    --gres="${VIT_GRES:-gpu:1}" --cpus-per-task=4 --mem=24G \
    --time="${VIT_WALLTIME:-24:00:00}" --output="$run/slurm-%j.log" \
    --export=ALL "$run/code/worker.sh" "$run"
else
  bash "$run/code/worker.sh" "$run" 2>&1 | tee -a "$run/interactive.log"
fi
