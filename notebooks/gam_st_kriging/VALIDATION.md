# Validation — 2026-11-20

## Completed

- 10 tests in `test_st_kriging.py` passed in local Python geo_env.
- Kernel covariance symmetry/positive definiteness and temporal decay checked.
- One-node Bernoulli posterior mode compared against independent scalar optimization.
- Laplace kriging mean/variance compared against the explicit `(K + W^-1)` system; objective descent and future-time restriction checked.
- Gaussian quadrature probability bounds/symmetry and exact alpha=0 identity checked.
- 38-date split verified: 13 train / 3 validation / 4 test target dates, with no cross-split TS-input/target raster overlap.
- Model selection executed with test-period raster and target-label access forbidden.
- Prediction executed with current/future raster access forbidden.
- Current-day predictors and labels deliberately changed: recomputed predictions remained identical.
- Past observations deliberately changed: conditional kriging predictions changed.
- In-sample GAM residual observations and current-day conditioning observations are rejected.
- All six pipelines evaluated common pixel IDs and labels with GAM fitting disabled during test.
- Selection seal rejects changed parameters. All positive GAM training samples retained; negative inclusion weights checked.
- New notebook executed all cells in a fresh Jupyter kernel with DEMO_MODE=True. `synthetic_execution.ipynb` preserves this run; the main HPC notebook remains unexecuted with DEMO_MODE=False.
- Synthetic full run had 1,616 common test pixel-date rows and 38 positives. Nonzero kriging strength was selected for every TS, exercising actual test-stage kriging as well as the GAM baseline. These are synthetic checks, not real event results.
- Rendered probability maps inspected; shared scale and excluded-pixel colors verified.
- Headless runner argument parsing checked.

## Scope and limitations

Real event GeoTIFFs are only on HPC and were not accessible during authoring. Actual performance, memory consumption and runtime remain unverified. The block-centroid latent model, separable exponential covariance and fixed prior variance are declared modeling approximations/assumptions, not learned fire-spread physics. Laplace conditional uncertainty does not include GAM coefficient, model-selection, label, or input-data uncertainty.

Code-level causal access tests cannot establish real-time availability of upstream satellite composites or weather products. Daily rolling-origin explicitly allows yesterday's test observations in today's conditioning; it does not claim frozen-origin multi-day prediction or independent-event generalization. This already explored single event remains an exploratory comparison.

## Method references

- Rasmussen & Williams (2006), Gaussian Processes for Machine Learning, Chapter 3: https://gaussianprocess.org/gpml/chapters/RW.pdf
- Kuss & Rasmussen (2005), Assessing Approximate Inference for Binary Gaussian Process Classification: https://www.jmlr.org/papers/v6/kuss05a.html

The references justify the latent Bernoulli GP/Laplace inference framework. They do not validate the performance of this application or claim it reproduces a particular continuous-response GAM residual-cokriging paper.
