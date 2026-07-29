# Camera Hidden-State Attribution Experiment Design

## Objective

Determine whether Camera Head contains stable hidden feature units that are
preferentially associated with camera-center translation, rotation, or field of view,
and measure how those units change when the same target frame is processed in
100-frame and 500-frame contexts. The experiment must decide whether a future refiner
should operate in Camera Head hidden space or remain an output-space camera-center
residual model.

## Fixed Data Protocol

Reuse the frozen ScanNet-50 split and existing numeric artifacts:

- 10 calibration scenes select and freeze feature-unit rankings.
- 40 holdout scenes validate the frozen rankings without refitting.
- Global contexts contain up to 500 frames; local windows use length 100 and stride 50.
- Frames are paired only by exact scene and frame ID.
- Camera Head uses four refinement iterations and the published VGGT checkpoint.

Existing `normalized_camera_tokens` are the experiment input. The pipeline must not
rerun image preprocessing or Aggregator inference. Replayed final poses must match the
stored poses within a fixed numerical tolerance before a scene is accepted.

## Camera Head Trace

For every refinement iteration, record:

- trunk output, shape `[B, S, 2048]`;
- post-GELU `pose_branch` hidden activation, shape `[B, S, 1024]`;
- raw pose delta, shape `[B, S, 9]`;
- accumulated raw pose encoding, shape `[B, S, 9]`.

The final linear layer maps each 1024-dimensional hidden activation to translation
dimensions `0:3`, quaternion dimensions `3:7`, and FoV dimensions `7:9`. Per-unit
contribution is the hidden activation multiplied by the corresponding output weight.
This decomposition is exact for the raw pose delta, but not for final camera-center or
rotation error; those require output-space intervention and decoding.

## Calibration and Frozen Unit Sets

For each unit and iteration, aggregate:

- absolute translation, quaternion, and FoV contribution;
- local-global activation and contribution drift for matched frames;
- contribution-group specificity;
- stability across scenes, windows, and distance from a local-window boundary.

Freeze the top 64 translation, rotation, and FoV units on calibration. Rankings use
prediction-only values. Also freeze deterministic random control sets of 64 units,
matched by iteration and generated with seed 33. Ties are resolved by ascending unit
index.

## Holdout Validation and Intervention

Holdout analysis reports activation drift, contribution drift, top-k ranking overlap,
window-boundary strata, and scene-bootstrap confidence intervals. Frozen feature sets
must not be changed after inspecting holdout.

For causal validation, zero each frozen set at the post-GELU hidden activation and
rerun the remaining Camera Head computation. Compare against its matched random
control using:

- camera-center displacement after decoding `C = -R^T T`;
- relative rotation change in degrees;
- FoV change;
- final aligned prediction error against raw GT.

Any metric containing predictions uses aligned predictions. GT is always raw and is
never aligned. Prediction-only ranking and selection outputs remain separate from
GT-based validation outputs.

## Pipeline and Artifacts

The AutoDL entry point supports `smoke`, `calibration`, `holdout`, and `export` stages,
with deterministic run identity and resumable per-scene completion markers. Raw
high-dimensional traces stay under `/root/autodl-tmp` and are not committed.

Published numeric artifacts are limited to:

- `per_unit.csv`;
- `per_scene.csv`;
- `frozen_units.json`;
- `summary.json`;
- run metadata and completion JSON.

No images, datasets, checkpoints, PLY files, or raw trace NPZ files are published.

## Focused Cleanup

Remove the invalid trajectory-overlay implementation and its design document because
independently GT-aligned windows cannot form a realizable stitched trajectory. Replace
the misleading `pose_tokens_modulated` trace name with `trunk_output`, while preserving
the default Camera Head prediction API. Do not duplicate environment, checkpoint,
dataset, or local-global inference scripts.

## Decision Rule for the Refiner

Hidden-space refinement remains viable only if calibration-selected translation units:

1. retain their specificity and ranking on holdout;
2. respond consistently to context length across scenes;
3. alter camera centers more than rotation and FoV under intervention;
4. outperform matched random controls.

If these conditions fail, the refiner should operate on decoded camera centers while
preserving global rotation and FoV. A successful attribution experiment identifies an
intervention layer and iteration, but does not by itself prove that a learned refiner
will improve trajectory accuracy.
