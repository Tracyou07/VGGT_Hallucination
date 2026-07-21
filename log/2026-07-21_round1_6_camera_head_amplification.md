# 2026-07-21 Round 1.6 Camera Head Amplification

## Why This Round Was Added

Round 1.5 showed that long-context failures coincide with drift in normalized
Aggregator Camera Tokens, but it did not exclude amplification by the Camera
Head's four transformer blocks. Round 1.6 isolates that downstream path before
the project enters geometry-residual validation.

## Implementation Completed

- Created branch/worktree `camera-head-amplification-preexperiment` from the
  completed Round 1.5 result commit.
- Added selective Camera Head checkpoint loading; the Aggregator and image
  encoder are not constructed or executed.
- Added hook-based capture for AdaLN input, four trunk blocks, trunk norm, 9D
  delta, and accumulated raw pose across refinement iterations.
- Added fixed-length token-perturbation and extra-head-context comparisons.
- Added strict replay validation against saved 200- and 500-frame activated
  pose encodings.
- Added independently aligned pose metrics against raw GT, a fixed AutoDL
  runner, CPU unit tests, and scalar-only result publishing.

## Local Verification Constraint

The Windows host currently exposes no `python`, `python3`, or `py` executable,
so the unit suite cannot run locally. The AutoDL `vggt` environment must run
`python -m unittest discover -s tests` before the four-scene replay. Bash syntax
and repository-level static checks are performed locally.

## Result Status

The one-scene smoke run is published at
`results/camera_head_amplification/9037f01_80bde20e63f2`. It covers only
`scene0000_00`; the following statements are preliminary Round 1.6 conclusions,
not a four-scene generalization.

Both replay gates passed with exactly zero maximum and mean absolute error for
`H200(Z200)` and `H500(Z500)`. The offline Camera Head replay therefore exactly
reproduces the saved activated 9D pose encodings for this scene.

## Preliminary Conclusion

Replacing the 200-context tokens with the same 200 frames' tokens produced in
the 500-frame Aggregator context changes aligned ATE from `0.193299` to
`1.377913`. Aligned ARE changes from `3.7112` to `35.6646` degrees. Translation
error increases on 99.5% of shared frames, with mean change `+1.06197` and
median change `+1.02363`. Input-token L2 drift has Pearson correlation `0.4359`
with per-frame aligned translation-error growth.

The Camera Head residual stream expands token drift internally: Block 4 RMS
drift is `3.85x` to `5.35x` the input-token RMS drift across the four refinement
iterations. However, trunk normalization reduces this to `0.57x` to `0.92x`,
and raw 9D output drift is dominated by iteration 1. Later pose deltas are too
small to correct the first decoded trajectory difference. Camera Head is thus
sensitive to the changed representation, but it does not monotonically amplify
the perturbation through every downstream stage.

Holding the shared 500-context tokens fixed while changing Camera Head length
from 200 to 500 changes shared-frame ATE only from `1.377913` to `1.376771` and
mean per-frame translation error by `-0.00125`. For `scene0000_00`, the dominant
long-sequence failure is therefore already present in the Aggregator Camera
Tokens; extra Camera Head self-attention context is not the main cause.

## Pending Formal Test

- [ ] Run the fixed four-scene panel with `SCENE_LIMIT=0`: failure scenes
  `scene0000_00` and `scene0691_00`, plus stable controls `scene0013_02` and
  `scene0029_01`.
- [ ] Require all eight 200/500 replay baseline gates to pass before analysis.
- [ ] Confirm whether fixed-length token replacement reproduces degradation in
  both failure scenes but remains small in both controls.
- [ ] Compare per-layer drift ratios and the pure extra-Head-context effect
  across failure and control scenes.
- [ ] Publish only the exporter-approved CSV/JSON result and replace this
  preliminary scope with the formal Round 1.6 conclusion.
