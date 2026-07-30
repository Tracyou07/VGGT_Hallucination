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

CPU tests validate tensor isolation, finite-difference projection, direct
checks, frozen-scale reuse, provenance, and export rejection. No real VGGT
checkpoint/ScanNet causal atlas has been produced yet, so this entry records an
implemented experiment rather than a research conclusion.
