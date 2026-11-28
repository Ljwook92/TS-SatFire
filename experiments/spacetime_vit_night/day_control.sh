#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mode=${1:-batch}
case "$mode" in
  batch|interactive|report) ;;
  *) echo 'Usage: bash day_control.sh batch|interactive|report [night run]' >&2; exit 2 ;;
esac
export SATFIRE_PYTHON=${SATFIRE_PYTHON:-/home/$USER/data/SatFire/envs/boundary-gpu-v1/bin/python}
root=${VIT_RUNS:-/home/$USER/data/SatFire/spacetime_vit_runs}
if [[ $# -ge 2 ]]; then
  night=$2
else
  shopt -s nullglob
  runs=("$root"/ts4-night-window-*/)
  case ${#runs[@]} in
    0) [[ "$mode" != report ]] || { echo "No night run under $root"; exit 1; }
       message=$(bash launch.sh snapshot)
       night=${message#Run: } ;;
    1) night=${runs[0]%/} ;;
    *) printf 'Select one run and pass its full path as the second argument:\n%s\n' "${runs[0]}"
       printf '%s\n' "${runs[@]:1}"; exit 2 ;;
  esac
fi
night=${night%/}
[[ -f "$night/code/window_run.py" && -f "$night/settings" ]] || { echo "Invalid night run: $night"; exit 2; }
day="$night/day_control"
echo "Night: $night"
echo "Day:   $day"
if [[ "$mode" == report ]]; then
  exec "$SATFIRE_PYTHON" compare_af.py "$night"
fi
mkdir -p "$day"
# Model, loss and data come from the chosen night snapshot. Only the loader gains a mask.
if ! cmp -s "$night/code/window_run.py" window_run.py; then
  old=$(sha256sum "$night/code/window_run.py" | cut -d' ' -f1)
  [[ "$old" == 6f1e720c21f5bd78fdc0a3d80a963a1fbb8ec3509cddbefbad51a8e879f5838f ]] || {
    echo 'Unrecognized night trainer. Compare implementations before pairing.' >&2; exit 2; }
fi
for script in window_run.py compare_af.py day_worker.sh; do
  if [[ -e "$day/$script" ]]; then
    cmp "$script" "$day/$script"
  else
    cp "$script" "$day/$script"
  fi
done
job_opts=(--job-name=vit-day-control)
if [[ ! -f "$night/summary.csv" ]]; then
  # Match the batch job's full command, since several experiments can share its name.
  queued=$(squeue -h -u "$USER" -n vit-night -o '%i|%o' |
           awk -F'|' -v path="$night/code/worker.sh" 'index($2,path)>0 {print $1; exit}')
  if [[ -n "$queued" ]]; then
    if [[ "$mode" == batch ]]; then
      job_opts+=(--dependency="afterok:$queued" --kill-on-invalid-dep=yes)
    else
      echo "Night job $queued is queued or running. Waiting for its cache."
    fi
  elif flock -n "$night/run.lock" true && [[ ! -f "$night/summary.csv" ]]; then
    if [[ "$mode" == interactive ]]; then
      bash launch.sh interactive "$night"
    else
      submission=$(bash launch.sh batch "$night")
      echo "$submission"
      job=$(awk '/^Submitted batch job [0-9]+$/ {print $4}' <<< "$submission")
      [[ -n "$job" ]] || { echo 'Cannot identify the night job ID.' >&2; exit 1; }
      job_opts+=(--dependency="afterok:$job" --kill-on-invalid-dep=yes)
    fi
  elif [[ "$mode" == interactive ]]; then
    echo 'Waiting for the active night worker before using the interactive GPU.'
    flock "$night/run.lock" true
    if [[ ! -f "$night/summary.csv" ]]; then
      bash launch.sh interactive "$night"
    fi
  fi
fi
if [[ "$mode" == batch ]]; then
  sbatch "${job_opts[@]}" --partition="${SATFIRE_GPU_PARTITION:-gpu}" \
    --gres="${VIT_GRES:-gpu:1}" --cpus-per-task=4 --mem=24G --time="${VIT_WALLTIME:-24:00:00}" \
    --output="$day/slurm-%j.log" --export=ALL "$day/day_worker.sh" "$night"
else
  bash "$day/day_worker.sh" "$night" 2>&1 | tee -a "$day/interactive.log"
fi
