#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mode=${1:-}
parent=${2:?Usage: bash lengths.sh batch|interactive|report /absolute/path/to/completed-ts4-night-run}
parent=${parent%/}
[[ "$parent" = /* && -s "$parent/settings" && -s "$parent/cache/manifest.csv" &&
   -s "$parent/window_slots.csv" && -s "$parent/day_control/window_slots.csv" ]] || {
  echo 'Supply the completed four-slot night run with its cache and day control.' >&2; exit 2; }
export SATFIRE_PYTHON=${SATFIRE_PYTHON:-/home/$USER/data/SatFire/envs/boundary-gpu-v1/bin/python}
out="$parent/lengths_2_4_6"
mkdir -p "$out"
exec 8>"$out/launch.lock"
flock 8
case "$mode" in
  batch|interactive|report) ;;
  *) echo 'Select batch, interactive, or report.' >&2; exit 2 ;;
esac
if [[ ! -d "$out/code" ]]; then
  stage=$(mktemp -d "$out/code-XXXXXX")
  cp window_run.py night_model.py lengths_worker.sh lengths_report.py "$stage/"
  cp "$parent/settings" "$stage/settings"
  cp -R "$parent/code/shared" "$stage/shared"
  git rev-parse HEAD > "$stage/commit.txt"
  mv "$stage" "$out/code"
fi
if [[ "$mode" == report ]]; then
  exec "$SATFIRE_PYTHON" "$out/code/lengths_report.py" "$parent"
fi
pending=()
for i in 0 1 2 3; do
  ts=$((2+4*(i/2)))
  arm=day
  if (( i%2 )); then arm=night; fi
  if [[ ! -s "$out/ts$ts-$arm/summary.csv" ]]; then pending+=("$i"); fi
done
if (( ${#pending[@]} == 0 )); then exec "$SATFIRE_PYTHON" "$out/code/lengths_report.py" "$parent"; fi
if [[ -s "$out/array.job" ]]; then
  job=$(cat "$out/array.job")
  if squeue -h -u "$USER" -o '%F' | grep -Fx "$job" >/dev/null; then
    echo "GPU array $job is already queued or running."; exit 0
  fi
fi
if [[ "$mode" == batch ]]; then
  indices=$(IFS=,; echo "${pending[*]}")
  job=$(sbatch --parsable --job-name=vit-lengths --array="$indices" \
    --partition="${SATFIRE_GPU_PARTITION:-gpu}" --gres="${VIT_GRES:-gpu:1}" \
    --cpus-per-task=4 --mem=24G --time="${VIT_WALLTIME:-24:00:00}" \
    --output="$out/slurm-%A_%a.log" "$out/code/lengths_worker.sh" "$parent")
  job=${job%%;*}
  echo "$job" > "$out/array.job"
  echo "GPU array $job; results: $out"
else
  for i in "${pending[@]}"; do
    bash "$out/code/lengths_worker.sh" "$parent" "$i" 2>&1 | tee -a "$out/interactive-$i.log"
  done
  "$SATFIRE_PYTHON" "$out/code/lengths_report.py" "$parent"
fi
