# Four-day spatiotemporal ViT for active fire and burned area mapping

Run on Hellbender from experiments/spacetime_vit:

```bash
RUN=$(bash submit.sh)
echo "$RUN"
cat "$RUN/jobs.txt"
squeue -u "$USER"
```

After completion, inspect summary.csv, events.csv and maps/EVENT/DATE.tif under RUN. Each four-band GeoTIFF contains AF and BA sigmoid scores, then the two binary decisions. A score of -1 denotes unavailable input. The official labels evaluate AF and BA on different event sets. Both heads produce maps on both sets.

The submitter snapshots the code and fixed input assets in a fresh run and uses the existing boundary-gpu-v1 environment. By default data are read from /home/$USER/data/SatFire/ts-satfire, results are stored under /home/$USER/data/SatFire/spacetime_vit_runs, and Slurm schedules a CPU preparation job followed by one GPU job. Allow approximately 15 GiB free per run for the cache, prediction maps and checkpoints. A GPU job defaults to 24 hours. After timeout, resubmit with `bash submit.sh "$RUN"` to resume from the last completed epoch. Jobs in a run are serialized by stage. The original settings and code snapshot are reused. Optional overrides before submission include TS_SATFIRE_RAW_ROOT, SATFIRE_PYTHON, VIT_RUNS, VIT_GRES, VIT_WALLTIME, VIT_EPOCHS and VIT_BATCH.

## Inputs and splits

Daytime VIIRS inputs comprise I4, I5, I4-I5, I4 minus its local 7x7 mean, reference-minus-day NDVI/NBR1/NIR/NBR2/MIRBI, negative distance to the previous day's nearest daytime FIRMS S-NPP hotspot (50 km cap) and prior-day hotspot density in a 2 km radius. Missing temporal slots and invalid pixels carry a validity channel. The first available image anchors spectral differences, including incidents already burning at the start of the archive. Each following date is a target; earlier dates within its four-calendar-day input are used when available. No future imagery or label masks are used as predictors. AF receives daily features, BA their per-pixel running maxima through each input date. Norms use a systematic pixel sample from training fires only. These are same-day detection and cumulative mapping tasks.

The official development table lists 125 training IDs and 13 validation IDs. In the inspected archive 24332700 has no day imagery. The other 124 training events supply 2,130 targets, 1,759 with four post-reference consecutive observations. Training includes partially observed sequences, including 24463187 with three raw scenes. All 13 validation events yield 265 targets. Preparation writes the actual per-event inventory. The first training event-day is a reference only. In development the AF target is the daily positive band 7 record and the BA target is the running union of positive AF and perimeter records (bands 7 and 8). A missing positive record on observed imagery counts as unrecorded; this training target can omit real burn. This BA supervision and full training cohort differ from the earlier TCN.

## Architecture and training

Pixel maps use 4x4 patch projection (64x64 tokens per date), a learned spatial and calendar-slot embedding, and two factorized attention blocks (32 channels, 4 heads). Each block first attends within 8x8 spatial token windows and then across the four dates at a location. The second block offsets spatial windows by four tokens. Padded keys are masked. The date-t tokens are upsampled and combined with a full-resolution skip connection to preserve small hotspots. AF and BA share encoder weights, but have separate decoders and daily/cumulative input streams. The train loop backpropagates both task losses into the shared encoder. Default settings are AdamW, learning rate 0.0003, batch 1 whole-fire crop, masked BCE plus soft Dice, one seed and up to 100 epochs. Per-task thresholds and checkpoint selection use validation fires only; selection maximizes the mean of the two validation pooled IoUs, with patience 12. Each logged epoch is a full pass through every development training target.

## Evaluation

Split lists and AF day/night threshold tables are derived from the official repository commit da7573219967edfe17e317310b66fd8a913f4a2e. BA uses the 24 US 2021 cases with label_sel from the evaluation script. Perimeter nodata are ignored, and the AF-accumulation label is used where requested. AF uses the official centered crop and union of daytime and nighttime threshold masks on 17 cases. Development and BA use the reference pixel window at offset 128. Nighttime thresholds define AF truth only. The project FIRMS sample lives in ../boundary_gpu/assets; supplementary AF-site FIRMS data and their sources are included here.

The full run asserts date, pixel-support and truth hashes against the earlier 11-channel TCN evaluation (153 AF and 830 BA targets). Each date is scored once. summary.csv contains pooled foreground IoU, event-averaged daily IoU and pixel-date counts. Empty/empty scores 1 on the all-dates summary. The known-dates and four-observed-dates summaries omit completely unknown dates. The BA known-background subset uses the 12 accumulated-AF sites. AF also excludes Tubbs, Camp and Creek in an additional summary because their crop footprints and dates overlap official development scenes. The archived TCN was fit on a smaller subset with different BA supervision; the comparison measures end-to-end performance and cannot isolate the effect of attention. The published reference used overlapping windows with a different aggregation.

Sources: https://github.com/zhaoyutim/TS-SatFire/tree/da7573219967edfe17e317310b66fd8a913f4a2e and https://firms.modaps.eosdis.nasa.gov/country/
