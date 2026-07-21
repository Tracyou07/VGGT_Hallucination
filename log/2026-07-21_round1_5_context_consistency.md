# Round 1.5 Camera Context Consistency Setup

## Branch Boundary

- Created `camera-context-consistency-preexperiment` from the completed Round 1
  branch at commit `6d07be5`.
- Kept Round 1 code, numeric results, and its conclusion log on
  `camera-iteration-preexperiment`.
- Kept repository-wide research guidance and the gated implementation plan on
  `main`.

## Diagnostic Contract

- Fix Camera Head iterations at 4.
- Run `scene0000_00` and `scene0691_00` as observed long-context failures, with
  `scene0013_02` and `scene0029_01` as stable controls.
- Reuse nested selections at 25, 50, 100, 200, and 500 frames.
- Save final normalized Camera Tokens, raw predictions, independently aligned
  predictions, raw GT, per-frame translation/rotation errors, and update norms.
- Match only shared frame IDs across contexts. Report token cosine drift,
  pairwise-affinity drift, aligned pose/error changes, update changes, and
  correlations. Do not interpret raw token MSE as physical geometry error.

## Execution Boundary

The VGGT forward pass requires CUDA and reuses the existing `vggt` conda
environment, checkpoint, and processed ScanNet data. Dense Depth, Point, and
Track Heads remain disabled. Matched-frame analysis and result publishing run
on CPU. The numeric exporter validates exact NPZ members and excludes images,
point clouds, datasets, checkpoints, and per-iteration modulated token dumps.
