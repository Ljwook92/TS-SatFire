#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
cache=${1:?Pass the completed boundary cache directory}
reference=${2:?Pass the original interactive run containing TS2, TS4 and TS6}
raw=${TS_SATFIRE_RAW_ROOT:-/home/$USER/data/SatFire/ts-satfire}
py=${BOUNDARY_PYTHON:-/home/$USER/data/SatFire/envs/boundary-gpu-v1/bin/python}
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
"$py" -c 'import torch; assert torch.cuda.is_available(), "Use a GPU allocation"'
for ts in 2 4 6; do
    for file in best.pt choices.json; do
        test -s "$reference/TS$ts/$file" || { echo "Missing $reference/TS$ts/$file" >&2; exit 1; }
    done
done
root=${SATFIRE_NIGHT_RUNS:-"$PWD/runs"}
mkdir -p "$root"
out=$(mktemp -d "$root/night-i4-XXXXXX")
echo "Experiment directory: $out" >&2
mkdir "$out/code"
cp -R assets train.py prepare.py encoder.py directions.py vegetation.py night_i4.py night_boundary.py "$out/code/"
git rev-parse HEAD > "$out/commit.txt"
"$py" "$out/code/night_i4.py" --raw "$raw" --cache "$cache" --out "$out/nights" | tee "$out/prepare.log" >&2
for ts in 2 4 6; do
    "$py" "$out/code/night_boundary.py" --cache "$cache" --reference "$reference" --out "$out" --ts "$ts" | tee "$out/TS$ts.log" >&2
done
"$py" "$out/code/night_boundary.py" --out "$out" >&2
echo "$out"
