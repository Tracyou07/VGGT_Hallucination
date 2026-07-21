# 2026-07-21 Round 2A Local-Global Consistency

## Scope

Implemented the fixed Round 2A diagnostic pipeline only. Dataset screening,
pair construction, token intervention, optimization, fine-tuning, and DiT work
remain deferred.

## Implementation

- Added deterministic 100-frame windows with stride 50 over the exact Round
  1.5 500-frame IDs.
- Added prediction-to-prediction Sim(3) alignment for global-local and
  adjacent local-local trajectory disagreement.
- Added a resumable camera-only GPU runner that saves final normalized Camera
  Tokens, raw predicted c2w, and raw GT c2w in external per-window NPZ files.
- Added CPU analysis for token cosine drift, aligned pose residuals,
  stable-control p95 reliability thresholds, per-frame aggregation, and
  correlation with separately computed GT validation labels.
- Added a strict exporter that publishes only fixed CSV/JSON numeric outputs;
  high-dimensional NPZ data remains outside Git.
- Invalidated stale analysis completion markers before rewriting tables,
  validated resumed NPZ schemas, and passed the exact completed run directory
  from the GPU runner to CPU analysis.
- Added unit contracts for windows, alignment, artifacts, resume identity,
  score/GT separation, threshold fitting, complete-window enforcement,
  summaries, shell protocol, Git ignore rules, and scalar export.

## Metric Boundary

All detection scores use predictions only. Validation aligns each prediction
trajectory to raw GT independently. GT is stored and read only as
`gt_c2w_raw`; no aligned-GT artifact or column is permitted.

## Verification Status

Local Windows currently has no usable Python interpreter, so the Python test
suite must be run in the AutoDL `vggt` conda environment before formal GPU
execution. Static checks and shell syntax checks are recorded with the final
implementation commit. No Round 2 data has been generated in this worktree.
