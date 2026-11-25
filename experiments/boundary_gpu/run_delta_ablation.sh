#!/usr/bin/env bash
# Run inside an allocated GPU session, using the completed baseline cache.
set -euo pipefail
cd "$(dirname "$0")"
cache=${1:?Supply the completed baseline run cache directory}
raw=${TS_SATFIRE_RAW_ROOT:-/home/$USER/data/SatFire/ts-satfire}
py=${BOUNDARY_PYTHON:-/home/$USER/data/SatFire/envs/boundary-gpu-v1/bin/python}
root=${SATFIRE_DELTA_RUNS:-"$PWD/runs"}
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
cache=$(cd "$cache" && pwd)
"$py" -c 'import torch; assert torch.cuda.is_available(), "Run inside an allocated CUDA job"'
mkdir -p "$root"
out=$(mktemp -d "$root/veg-delta-XXXXXX")
echo "Experiment directory: $out"
mkdir "$out/code"
cp -R assets encoder.py prepare.py train.py directions.py vegetation.py delta_transformer.py "$out/code/"
git rev-parse HEAD > "$out/commit.txt"
"$py" "$out/code/vegetation.py" --raw "$raw" --out "$out/indices" | tee "$out/prepare.log"
for ts in 2 4 6; do
    "$py" "$out/code/delta_transformer.py" --cache "$cache" --indices "$out/indices" --out "$out" --ts "$ts" --seed "${SATFIRE_SEED:-804}" | tee "$out/TS$ts.log"
    cat "$out/TS$ts-seed${SATFIRE_SEED:-804}/validation_choices.csv"
done
