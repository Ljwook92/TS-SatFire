# Existing-code and data-contract audit — 2026-11-19

## Actually inspected locally

Repository: `/Users/lee/Documents/Playground/TS-Agentic-AI` (read-only).

- `legacy/support/path_config.py`: default raw root `/home/jlc3q/data/SatFire/ts-satfire`.
- `legacy/roi/fire_all_test.csv`, line 23: `US_2021_NM3323810847220210520`, center 33.191/-108.474, ROI dates 2021-05-17 to 2021-06-24, label_sel=0. These are ROI metadata, not a verified actual file inventory or confirmed ignition dates.
- `legacy/scripts/analyze_pred_event_windows.py`, load_daily_masks: VIIRS_Day raw 7=AF, 8=BA, cumulative union for label_sel=0. Original uses 256x256 window offset 128 and NaN->0.
- `legacy/satimg_dataset_processor/satimg_dataset_processor.py`, PredDatasetProcessor: concatenates six day channels, two night thermal channels, 19 FirePred channels; accumulates thermal maxima and derives positive cumulative-label differences. Original code indexes file positions; our new code requires calendar-consecutive days.
- `legacy/scripts/join_pred_event_candidates_firepred.py`: verified one-based FirePred mapping (below); its zero filling is NOT inherited.
- `knowledge/data_sources_ts_satfire.md`: records an earlier gdalinfo inspection of a DIFFERENT event, 24462335/2020-09-26_FirePred.tif. This is supporting documentation, not live verification of the requested event. EVI2_last retained as evi2; not silently renamed EVI.

## Not verified

No requested-event GeoTIFFs were found in the inspected local project. Noninteractive SSH to the configured Hellbender host failed authentication. No HPC job submitted, raw download performed, or real event raster opened. The exact earlier notebook described in chat was not located among the inspected project notebooks; its prose was used as requirements, not its unseen implementation.

The user's requirement to inspect actual data is therefore only partially satisfiable here: the provided notebook performs the actual inventory on HPC FIRST and stops at explicit schema/label/timing gates before fitting. No claim is made of actual data coverage, nodata semantics, clouds, variograms, fire-growth counts or performance.

## One-based FirePred mapping

1 NDVI_last; 2 EVI2_last; 3 precipitation; 4 wind speed; 5 wind direction; 6 minimum temperature; 7 maximum temperature; 8 ERC; 9 specific humidity; 10 slope; 11 aspect; 12 elevation; 13 PDSI; 14 LC_Type1; 15 forecast precipitation; 16 forecast wind speed; 17 forecast wind direction; 18 forecast temperature; 19 forecast specific humidity.

Descriptions in actual files override any assumption: review inventory and modify SPECS if necessary. GDAL scale/offset are applied; only add a value_rules multiplier when known not already applied. Source raster QA not present in exported files cannot be reconstructed from zeros. Raw label NaNs are UNKNOWN by default; legacy-compatible NaN->background is an explicit, documented option only.

## Deliberate differences from paper/legacy

Full raster by default; two input dates; raw thermal values (no cumulative thermal maximum); cumulative observed AF|BA target with explicit validity; chronological within-event training/validation/test and 2-window purge; scalar GAM vs deep spatial architecture. Pixel distance valid only within this fixed grid. Label-observation delay remains. No official 256x256 macro-metric reproduction. No claim of ground-truth physical propagation. No model training on this event may be reported as independent FP test evidence.

## Sources

- https://arxiv.org/html/2412.11555v1
- https://doi.org/10.1038/s41597-025-06271-3
- https://geostat-framework.readthedocs.io/projects/pykrige/en/stable/generated/pykrige.ok.OrdinaryKriging.html
- https://pygam.readthedocs.io/en/stable/reference/_autosummary/pygam.pygam.LogisticGAM.html
