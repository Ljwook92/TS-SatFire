# Rolling boundary-pixel fire progression on Hellbender

From an HPC shell with Conda available:

```bash
cd /mnt/pixstor/data/jlc3q/SatFire/TS-SatFire
git pull --ff-only origin main
cd experiments/boundary_gpu
bash setup_env.sh
export TS_SATFIRE_RAW_ROOT=/home/jlc3q/data/SatFire/ts-satfire
RUN_DIR=$(bash submit.sh)
squeue -u "$USER"
```

The launcher prints job IDs to stderr and the unique output directory to stdout. It snapshots the code and assets there, using a CPU job to prepare the existing rasters, a dependent one-GPU-per-task array for TS 2, 4 and 6, and a final CPU summary. Allow about 35–40 GB on PixStor per run plus the separate Conda environment. Existing experimental outputs are preserved. Overrides include SATFIRE_RUNS (absolute shared output parent), SATFIRE_GPU_ENV, BOUNDARY_PYTHON (an existing CUDA Python with the same dependencies), SATFIRE_CPU_PARTITION and SATFIRE_GPU_PARTITION. The CUDA 12.1 wheel requires a compatible NVIDIA driver. [Hellbender GPU allocations](https://tdx.umsystem.edu/TDClient/36/DoIT/KB/Article/2829/How-To-How-and-why-to-get-an-interactive-session-on-Hellbender)

## Inputs and evaluation

The assets directory ships frozen project Transformer weights and training-event normalization, the rolling reference split and date manifest, the eight-sector priors and input-defined candidate domains, the previous TS2 test prediction masks, and an event/day subset of NASA FIRMS VIIRS S-NPP daytime wildfire hotspots. FRP is radiative power in MW. A missing hotspot has an indicator. The only external data needed at run time are the existing TS-SatFire GeoTIFFs. [TS-SatFire reference implementation](https://github.com/zhaoyutim/TS-SatFire) · [NASA FIRMS archive](https://firms.modaps.eosdis.nasa.gov/download/)

A window advances one observed file at a time through each event. The next file supplies only the target, newly recorded AF or AF/BA under the event's reference label policy. Nonfinite label values contribute no detection. The full 256 × 256 crop is scored. The historical thermal maxima and cumulative recorded footprint use observations through the input day. The local thermal-change channels use raw I4 differences inside the input window, with a validity flag and a reset after calendar gaps. Wind projection interprets the observed direction as meteorological wind-from. Inputs include same-day composites and archived FirePred channels. This is a retrospective experiment with an already explored test set. Operational issue-time availability is unverified.

| TS | Train windows | Validation | Test |
|---:|---:|---:|---:|
| 2 | 2006 | 252 | 806 |
| 4 | 1759 | 226 | 758 |
| 6 | 1519 | 200 | 710 |

The head learns a per-pixel correction to frozen Transformer logits within 4 km of the known perimeter. It reads the entire input sequence of local features, including I4 change at a candidate and its nearest recorded edge, wind projected toward the candidate, uphill alignment and elevation change, nearby FRP, distance, and vegetation levels. The loss combines weighted pixel BCE, Dice and a count loss for each nearest edge cell and outward distance band (0–0.5, 0.5–1, 1–2 and 2–4 km). The last term estimates spread extent by matching expected and observed new pixel counts per group. Beyond this neighborhood the learned correction is zero. Previously recorded pixels cannot be predicted as new.

For validation and test, the direction distribution uses the frozen input-only combined model where available, then smoothed previous spread or a neutral fallback. Within its domain, the allocator keeps the total expected area and assigns the eight sector shares before thresholding, subject to available pixel capacity. Binary masks can have different shares after thresholding. Forecasts elsewhere still enter the full-crop IoU. Archived TS2 masks enter only the final paired test comparison. Each new model's logit cutoff and the learned head epoch are selected on validation pooled IoU.

| Output | Interpretation |
|---|---|
| summary.csv | Full rolling, paired common target dates, and common strictly consecutive windows. Pooled IoU, delta versus archived TS2 and 0.40 target. Also event-balanced and active-frame metrics. Empty–empty frames score 1 only in frame-based averages. |
| TS2, TS4, TS6/frame_metrics.csv | Counts for the exact same event, date and grid per arm. archived_ts2 is the actual prior experiment, direct is the frozen TS encoder, legacy_prior its earlier direction rule, allocation_only the new sector allocation, and local_extent adds the learned boundary correction. Compare local_extent with allocation_only to isolate that correction. |
| TS*/new_mexico_east.png and new_mexico_east.csv | June 1 observed and predicted pixels and E candidate outline, with TP/FP/FN for the strict E sector, NE/E/SE and full crop. E spans 67.5–112.5 degrees around the input-derived center. |
| TS*/masks and choices.json | Packed full-crop test masks (unpack bits then reshape to 256 × 256) and validation-selected cutoffs. |

Inspect after all jobs finish:

```bash
cat "$RUN_DIR/summary.csv"
cat "$RUN_DIR"/TS{2,4,6}/new_mexico_east.csv
ls "$RUN_DIR"/TS{2,4,6}/new_mexico_east.png
```
