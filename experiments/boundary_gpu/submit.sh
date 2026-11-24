#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
root=${SATFIRE_RUNS:-/mnt/pixstor/data/$USER/SatFire/boundary_runs}
export TS_SATFIRE_RAW_ROOT=${TS_SATFIRE_RAW_ROOT:-/home/$USER/data/SatFire/ts-satfire}
export BOUNDARY_PYTHON=${BOUNDARY_PYTHON:-${SATFIRE_GPU_ENV:-/mnt/pixstor/data/$USER/SatFire/envs/boundary-gpu-v1}/bin/python}
[[ -x "$BOUNDARY_PYTHON" ]] || { echo 'Run bash setup_env.sh first.' >&2; exit 1; }
[[ -d "$TS_SATFIRE_RAW_ROOT/US_2021_NM3323810847220210520/VIIRS_Day" ]] || { echo "Check TS_SATFIRE_RAW_ROOT: $TS_SATFIRE_RAW_ROOT" >&2; exit 1; }
mkdir -p "$root"
run=$(mktemp -d "$root/run-$(date +%Y%m%d-%H%M%S)-XXXXXX")
echo "Run directory: $run" >&2
mkdir -p "$run/code" "$run/logs"
cp -R assets *.py job.sh "$run/code/"
git rev-parse HEAD > "$run/commit.txt"
cpu=${SATFIRE_CPU_PARTITION:-general}
gpu=${SATFIRE_GPU_PARTITION:-gpu}
args=(--parsable --chdir="$run/code" --export=ALL --output="$run/logs/%x-%A_%a.log")
submit_job() {
  local reply
  reply=$(sbatch "${args[@]}" "$@")
  echo "${reply%%;*}"
}
prep=$(submit_job -J fire-prep -p "$cpu" --cpus-per-task=2 --mem=12G -t 12:00:00 job.sh prepare "$run")
echo "prepare $prep" | tee -a "$run/jobs.txt" >&2
fit=$(submit_job -J fire-fit -p "$gpu" --gres=gpu:1 --cpus-per-task=4 --mem=32G -t 24:00:00 --array=2,4,6 --dependency="afterok:$prep" --kill-on-invalid-dep=yes job.sh train "$run")
echo "train $fit (TS=2,4,6)" | tee -a "$run/jobs.txt" >&2
report=$(submit_job -J fire-report -p "$cpu" --cpus-per-task=1 --mem=4G -t 00:30:00 --dependency="afterok:$fit" --kill-on-invalid-dep=yes job.sh report "$run")
echo "report $report" | tee -a "$run/jobs.txt" >&2
echo "$run"
