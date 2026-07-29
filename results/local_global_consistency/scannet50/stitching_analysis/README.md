# ScanNet-50 Local-Global Stitching Analysis

## Experiment Identity

- Calibration: `3d5de75_c5c5ae0e55fe`, 10 scenes and 90 windows.
- Holdout: `3d5de75_35564d765bb5`, 40 scenes and 359 windows.
- Source context run: `d33d98b_309a9a586242`.
- Window length/stride: 100/50; Camera iterations: 4.
- Split digest: `69c283245c4f220965e6fde3b96192de298e292eb8ca625c94851fe8932cdb8a`.
- Frozen-threshold digest: `ee010678b5a4dc2c0639a3e3e4df1ca815748dad76393286f9ae759ccd9b4f09`.

All aggregate estimates give each holdout scene equal weight. Confidence intervals use
10,000 scene bootstrap samples with seed 33.

## Metric Policy

Any metric containing predictions uses aligned predictions and raw GT. GT is never
aligned. Relative rotation error (RRE) is invariant to a constant global alignment
rotation, so its value is identical before and after prediction alignment; GT relative
rotations remain raw.

The old absolute rotation fields in `summary.json` apply the rotation from a
camera-center Sim(3) alignment to orientations. They are retained only as legacy
diagnostics and must not be used to compare intrinsic rotation quality.

## Representation and Detection Results

Independently aligned 100-frame windows reduce mean translation error from `0.08806`
to `0.04359`; the mean global-minus-local difference is `+0.04447`
(`95% CI [0.03382, 0.05697]`). This is an oracle-style window result, not a complete
trajectory.

Prediction-only global-local disagreement weakly but reproducibly ranks translation
degradation:

| Score | Pearson | Spearman | Translation Q4-Q1 |
|---|---:|---:|---:|
| Token cosine distance | `0.124` | `0.125` | `+0.01712` |
| Pose translation | `0.144` | `0.095` | `+0.01444` |

The frozen local-local reliability gate did not become a general quality filter. It
removed relatively few samples and its disagreement scores were not monotonically
related to actual local-vs-global quality.

## Stitching Results

Three prediction-only assembly baselines were evaluated before consulting GT:

1. Sequential overlap Sim(3): align each new window to the assembled overlap.
2. Global-anchored local: align every local window to the corresponding global segment.
3. Translation hybrid: use global-anchored local camera centers with global rotations.

| Holdout translation error | Mean | Difference from global | Improved scenes |
|---|---:|---:|---:|
| Global | `0.08806` | - | - |
| Independent local oracle | `0.04359` | `-0.04447` | - |
| Sequential stitching | `0.15749` | `+0.06943` | `7/40` |
| Global-anchored local | `0.08278` | `-0.00528` | `33/40` |
| Translation hybrid | `0.08278` | `-0.00528` | `33/40` |

Naive sequential stitching accumulates gauge errors and is substantially worse than
the global trajectory. Global anchoring yields a small but robust `0.00528` translation
improvement (`95% CI [-0.00848, -0.00257]`), recovering only about 12% of the
independently aligned local-window advantage. The replacement quantity is the camera
center in camera-to-world coordinates, not Camera Head's raw world-to-camera
translation vector.

## Corrected Rotation Result

A control exposed an evaluation artifact: the same global prediction scores `3.98 deg`
when aligned over all 500 frames but `9.53 deg` when split into independently
camera-center-aligned 100-frame segments. Camera-center Sim(3) therefore makes the old
`global 3.98 deg` versus `local 7.34 deg` comparison invalid.

Gauge-invariant RRE instead shows a small local-window advantage:

| Lag | Global | Local | Local - global | 95% CI |
|---:|---:|---:|---:|---:|
| 1 | `0.4892 deg` | `0.3653 deg` | `-0.1239 deg` | `[-0.1778, -0.0837]` |
| 5 | `0.9940 deg` | `0.8326 deg` | `-0.1615 deg` | `[-0.2431, -0.1000]` |
| 10 | `1.4366 deg` | `1.2502 deg` | `-0.1863 deg` | `[-0.3117, -0.0946]` |
| 25 | `2.2773 deg` | `2.0861 deg` | `-0.1912 deg` | `[-0.3873, -0.0470]` |
| 50 | `3.0941 deg` | `2.9710 deg` | `-0.1230 deg` | `[-0.3738, 0.0814]` |

With orientation-only SO(3) alignment at the same 100-frame scale, global segments
score `1.7545 deg` and local windows score `1.6843 deg`; their difference
`-0.0701 deg` has CI `[-0.2080, 0.0415]`. There is no evidence that global context
improves intrinsic rotation. Local context is slightly better for short-range relative
rotation at lags 1-25, while equal-scale absolute orientation precision is essentially
the same.

## Decision

The useful training-free baseline is component-aware: retain global rotations and the
global long-range gauge, then replace camera centers with globally anchored local
estimates. Formal testing should reproduce the translation gain on independent data
and use RRE or orientation-only SO(3) alignment for rotation. Detailed values are in
`summary.json`, `rotation_analysis.json`, and `per_scene.csv`.
