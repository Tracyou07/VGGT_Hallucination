# 2026-07-29 Camera Hidden-State Attribution Pipeline

## Implemented

- Created branch/worktree `camera-hidden-state-attribution-preexperiment`.
- Added opt-in Camera Head tracing for each refinement iteration:
  trunk output, 1024-dimensional post-GELU activation, pose delta, and accumulated pose.
- Added `(iteration, hidden_index)` ablation without changing default VGGT predictions.
- Added token replay from existing 100-frame and 500-frame normalized Camera tokens.
- Added exact per-unit translation, quaternion, and FoV contribution decomposition.
- Fixed the protocol to 10 calibration scenes and 40 untouched holdout scenes.
- Added deterministic top-64 selection, iteration-matched random controls, window
  edge/interior analysis, and holdout ranking overlap.
- Added component-wise intervention metrics and aligned-translation validation against
  raw GT.
- Added strict numeric export and one AutoDL entry point with smoke, calibration,
  holdout, export, and resume stages.

## Cleanup

- Removed the invalid trajectory-overlay implementation and its design document.
- Replaced the misleading `pose_tokens_modulated_list` trace field with
  `trunk_output_list`.
- Reused existing ScanNet, checkpoint, split, and local-global artifacts instead of
  adding duplicate setup or inference code.

## Refiner Decision

This experiment is a gate for hidden-space refinement. A latent refiner is supported
only when frozen translation units retain holdout rank, show consistent context-length
drift, change camera centers more than rotation/FoV, and outperform matched random
controls. If those conditions fail, the next refiner should predict a decoded
camera-center residual while preserving global rotation and FoV.

The pipeline has CPU contract coverage but has not yet been run against the real VGGT
checkpoint and ScanNet-50 artifacts in this worktree.
