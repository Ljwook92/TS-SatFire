# Validation — 2026-11-19

- Actual HPC read-only SSH: authentication failed; no real-event raster or model run verified.
- Existing source code/band mapping/ROI membership inspected; see SOURCE_AUDIT.md.
- Notebook validated with nbformat; every code cell parsed with Python ast.
- Entire notebook executed in a fresh Jupyter kernel using DEMO_MODE=True and 40 synthetic daily GeoTIFF pairs (28x28, projected CRS, explicit nodata, separate AF/BA and 19-band FirePred).
- Final synthetic run: `synthetic_20261119_123253`. All code cells completed. Real-data default notebook remains DEMO_MODE=False with review gates False.
- Produced restoration rasters, block masking scores, three fitted models plus prevalence baseline, pooled/day/group scores, calibration plot and georeferenced prediction/error rasters.
- Verified both observed-input and missing-input evaluation groups are exercised.
- Six contract tests passed: calendar-consecutive windows; temporal split purge; observed-value immutability and unsupported gaps; strict vs explicit legacy NaN label behavior; categorical specification; known metric values.
- Visually inspected synthetic prediction panels. Synthetic accuracy is not scientific evidence and is intentionally not reported as a result for NM332.
- Conditional kriging variance is not propagated to final probability uncertainty; no independent-event benchmark performed.

Actual runtime package versions:

```json
{
  "numpy": "1.26.4",
  "scipy": "1.15.2",
  "pandas": "2.3.3",
  "rasterio": "1.4.3",
  "scikit-learn": "1.7.2",
  "pykrige": "1.7.3",
  "pygam": "0.12.0",
  "matplotlib": "3.10.8"
}
```

The executed demonstration notebook is in the workspace at `tmp/jupyter-notebook/satfire_kriging_gam_demo_executed.ipynb`. It is separate from the clean HPC notebook. Dependencies for local validation were installed under /private/tmp, leaving the existing geo_env package installation unchanged.
