#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
root="${VIT_RUNS:-/home/$USER/data/SatFire/spacetime_vit_runs}"
export TS_SATFIRE_RAW_ROOT="${TS_SATFIRE_RAW_ROOT:-/home/$USER/data/SatFire/ts-satfire}"
export SATFIRE_PYTHON="${SATFIRE_PYTHON:-/home/$USER/data/SatFire/envs/boundary-gpu-v1/bin/python}"
[[ -x "$SATFIRE_PYTHON" && -d "$TS_SATFIRE_RAW_ROOT/camp_fire" ]] || { echo 'Check SATFIRE_PYTHON and TS_SATFIRE_RAW_ROOT.' >&2; exit 1; }
if [[ $# -gt 0 ]]; then
  run=$1
  [[ -f "$run/settings" && -f "$run/code/job.sh" && ! -f "$run/complete" ]] || { echo 'Pass an existing incomplete run to resume.' >&2; exit 1; }
else
  mkdir -p "$root"
  run=$(mktemp -d "$root/ts4-vit-XXXXXX")
  mkdir -p "$run/code" "$run/logs"
  cp -R assets *.py job.sh "$run/code/"
  cp ../boundary_gpu/assets/firms.csv.gz "$run/code/assets/"
  git rev-parse HEAD > "$run/commit.txt"
  echo "${VIT_EPOCHS:-100} ${VIT_BATCH:-1}" > "$run/settings"
fi
read -r VIT_EPOCHS VIT_BATCH < "$run/settings"
export VIT_EPOCHS VIT_BATCH
opts=(--parsable --export=ALL --output="$run/logs/%x-%j.log")
dep=()
if [[ ! -f "$run/cache/manifest.csv" ]]; then
  prep=$(sbatch "${opts[@]}" -J vit-prep -p "${SATFIRE_CPU_PARTITION:-general}" --cpus-per-task=2 --mem=12G -t 12:00:00 "$run/code/job.sh" prepare "$run")
  dep=(--dependency="afterok:${prep%%;*}" --kill-on-invalid-dep=yes)
  echo "prepare $prep" | tee -a "$run/jobs.txt" >&2
fi
fit=$(sbatch "${opts[@]}" "${dep[@]}" -J vit-afba -p "${SATFIRE_GPU_PARTITION:-gpu}" --gres="${VIT_GRES:-gpu:1}" --cpus-per-task=4 --mem=24G -t "${VIT_WALLTIME:-24:00:00}" "$run/code/job.sh" gpu "$run")
echo "train and evaluate $fit" | tee -a "$run/jobs.txt" >&2
echo "$run"
