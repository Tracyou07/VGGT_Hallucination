# 2026-07-30 Camera Hidden Causal Preference Pipeline

## Motivation

The existing hidden-state attribution identified stable and causally important
units under zero ablation, but its translation/rotation/FoV label came from the
final linear-layer weights. It did not measure each unit's effect after the
remaining Camera Head refinement iterations.

## Implemented

- Added opt-in, batch-specific perturbations at the post-GELU hidden and 9D
  pose-delta locations without changing the default VGGT path.
- Built a centered finite-difference Jacobian from every refinement iteration's
  9D pose-delta input to final camera centers, rotations, and FoV.
- Projected the shared `fc2` columns through that Jacobian to recover a
  `4 x 1024` standardized hidden-unit atlas.
- Reduced one scene from 8192 direct perturbation samples to 72 basis samples.
- Added deterministic direct hidden perturbations to check the first-order
  projection on influential positions.
- Added strict per-scene NPZ contracts, scene-equal aggregation, calibration-only
  90th-percentile group scales, frozen provenance, and untouched holdout
  comparisons.
- Added a resumable AutoDL smoke/calibration/holdout/export entry point and a
  numeric-only authenticated exporter.

## Protocol Boundary

The atlas is prediction-only. Calibration uses ten frozen scenes; holdout uses
forty scenes and cannot refit normalization. A formal run requires all nine
pose-delta basis dimensions. Raw per-scene causal-effect artifacts stay under
`/root/autodl-tmp` and are not committed.

## Current Status

The formal ScanNet-50 run is complete:

- Calibration: `9368808_99c8a9ed393c`, 10 scenes.
- Holdout: `9368808_1d91735181c4`, 40 scenes.
- Both runs report `protocol_complete=true` and `analysis_complete=true`.
- Calibration-holdout causal top-64 overlap is 61/64 for translation, 62/64
  for rotation, and 59/64 for FoV.
- Every causal top-64 position is in the first Camera Head refinement.

The causal atlas was joined with the completed context-drift attribution by
`(iteration, unit)`. Calibration overlap is 41/64 for translation, 23/64 for
rotation, and 12/64 for FoV. After freezing those calibration intersections,
39/41, 23/23, and 11/12 positions respectively remain in both corresponding
holdout top-64 lists.

Translation and FoV direct projection checks pass at low relative error.
Rotation direct checks expose a small-angle numerical problem and must be
repeated with a robust rotation metric before rotation-specific intervention
claims are accepted.

The full interpretation, protocol boundary, and reproducible visualization
command are recorded in
`doc/2026-07-30_Camera_Hidden_Causal_Tracing_Findings.md`.
