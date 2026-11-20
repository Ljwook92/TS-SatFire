# Validation record — 2026-11-20

## Completed locally

- Five synthetic regression/integration tests passed (`python test_window_gam.py`).
- Six combinations (GAM/SpatialGAM × TS2/4/6) trained and evaluated.
- Full notebook executed top-to-bottom in a fresh geo_env Jupyter kernel, with only `DEMO_MODE` switched to True in the execution copy. Final plotting revision was re-executed too.
- `synthetic_execution.ipynb` contains that run. It is a synthetic fixture, **not** the real New Mexico event result.
- All six synthetic models evaluated identical 1,616 pixel-date rows / 38 positive records. Per-date pixel IDs and labels matched exactly; finite probabilities stayed within [0,1].
- Guarded feature access rejects reads on or after the target day; feature construction passed for all windows.
- Guarded model selection rejects test-period raster reads; all six selections passed.
- Evaluation was run with the fitting method disabled; no test refit occurred.
- Checked maximum-window blocks have no shared input/target raster dates, missing-feature rows remain present, negative inclusion weights are correct, all positive training rows are retained, and date gaps raise errors.
- Verified 38-date split gives 13 train / 3 val / 4 test target dates; first test date 2021-06-20.
- Rendered and inspected probability-map output with a shared color scale and excluded pixels shown separately.
- Existing tracked workspace files were not changed; all work is in the new output/satfire_window_gam directory.

## Not completed / limitations

- Real HPC GeoTIFFs are not available in this local environment. Actual event fit, performance, timing and memory use remain unverified.
- Product release times, upstream compositing and AF/BA observation completeness are not established by these code tests. The result remains retrospective recorded-fire expansion, not leakage-certified operational forecasting.
- Temporal/label access tests establish code boundaries; they cannot detect leakage already embedded in upstream source products.
- Only a few test dates within one previously explored event are available. This is not an independent-event or official TS-SatFire benchmark evaluation and no significance claim is made.
- This spatial GAM is a finite Gaussian basis with a ridge prior, not exact covariance-fitted kriging. No kriging posterior or prediction intervals are produced.
