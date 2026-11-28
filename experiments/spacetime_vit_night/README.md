# Four-slot day/night ViT for TS-SatFire

This experiment retains the project's factorized spatial-window and temporal attention architecture. Every slot passes through the full-resolution decoder. AF and BA have separate heads and share encoder weights. Each daily 256 × 256 input has the previous 11 variables plus nighttime I4 and I5. The AF stream uses daily features and BA uses running maxima through each observation. Two indicator channels identify available day I4 and night thermal observations. Individual missing features are imputed to their training mean in standardized space. The first scene participates in the model and in scoring, with zero reference-minus-day differences wherever defined. No AF or perimeter mask is a predictor. Four-slot attention is bidirectional within the observed window. Its early-slot predictions are retrospective.

## Cohort and optimization

Use the official 125 training and 13 validation IDs in the shared split, with full images and all labeled dates, including days with no fire records. The inspected dataset has 124 training IDs with imagery. Event 24332700 has no VIIRS_Day and event 24463187 has three scenes. For this last event, a single masked prefix slot permits learning from the three observed targets. The other 123 training events have at least four frames. Every run writes the actual inventory of files. Training uses interval one and the public generator's right-aligned tail behavior for full windows. Local inventory yields 2,252 training windows per epoch. Fixed seed, batch 1, AdamW at 0.0003, and masked BCE plus soft Dice are retained. Best checkpoint selection uses the mean of AF and BA event macro IoUs on validation windows at the fixed 0.5 cutoff. The default limit is 100 epochs with patience 12. Only training scenes contribute to normalization. Official train/test event IDs differ, but Tubbs, Camp and Creek AF test crops have geographic and date overlap with some development crops.

## Test scoring

The Scientific Data Methods specifies interval 1 during training and interval T for AF/BA testing. The public slicer right-aligns the final window. At T=4 on each of the 17 ten-scene AF fires this yields one-based slots [1,2,3,4], [5,6,7,8], [7,8,9,10]. All 204 slot maps enter the primary score. The secondary dense analysis uses all seven distinct four-frame windows at stride one, giving 476 slot maps. Both use the same checkpoint. Known AF truth is the published union of daily daytime and nighttime I4/I5 threshold masks over the full centered crop. All 170 mask hashes were produced by running the original generator at commit da7573219967edfe17e317310b66fd8a913f4a2e on the archived imagery. Preparation checks every reproduced test AF mask. Hashes are used only for integrity checks.

For every slot threshold the foreground sigmoid at 0.5, compute TP/(TP+FP+FN) over known pixels and assign 1 to an empty union. Average slot IoUs within each event, then give each event equal weight. Pooled IoU and error counts sum over scored slot occurrences and therefore include repeated dates. The BA output uses the project's existing official-case label selection for its 24 test fires. Source architecture, engineered features, BA supervision, joint optimization and input normalization differ from the source benchmark.

Sources: [TS-SatFire Methods](https://www.nature.com/articles/s41597-025-06271-3), [original slicer](https://github.com/zhaoyutim/TS-SatFire/blob/da7573219967edfe17e317310b66fd8a913f4a2e/satimg_dataset_processor/satimg_dataset_processor.py), [original scorer](https://github.com/zhaoyutim/TS-SatFire/blob/da7573219967edfe17e317310b66fd8a913f4a2e/run_spatial_temp_model.py).

## Hellbender

From experiments/spacetime_vit_night run the following in an allocated GPU shell:

```bash
bash launch.sh interactive
```

For a queued GPU job use `bash launch.sh batch`. Each new run snapshots code and assets under /home/$USER/data/SatFire/spacetime_vit_runs/ts4-night-window-*. The printed run path identifies the snapshot. To resume the same run after interruption, pass its path as the second argument in either mode. Training resumes from the last completed epoch. CPU preprocessing of the full dataset precedes GPU training. Allow about 15 GiB or more of free space. Device, training progress and validation results are written to the run log. TS_SATFIRE_RAW_ROOT, SATFIRE_PYTHON, VIT_RUNS, VIT_EPOCHS and VIT_BATCH are configurable at launch.

summary.csv contains the paired protocols; read the paper/AF/event_macro_iou cell for the 17-event benchmark. events.csv and window_slots.csv show the means and constituent observations. Each four-band GeoTIFF in maps/TASK/EVENT is named by its window endpoints and identifies its dates in the band descriptions. Companion NPZ files store packed decisions and evaluation-support masks.
