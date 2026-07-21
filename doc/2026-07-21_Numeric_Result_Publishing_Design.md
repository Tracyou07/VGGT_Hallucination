# Camera Iteration Numeric Result Publishing Design

## Goal

Publish reproducible Camera Iteration measurements from AutoDL to the feature
branch without committing ScanNet, images, point clouds, model weights, or
high-dimensional Camera Token tensors.

## Chosen Approach

A standard-library Python exporter copies an immutable completed run into
`results/camera_iteration/<run_id>/`. It uses an explicit file whitelist:

- Run level: `run_metadata.json`, `summary.json`, and `summary.csv`.
- Selection level: `iteration_metrics.json`, `iteration_metrics.csv`,
  `selected_frame_ids.json`, `complete.json`, and `camera_trace.npz`.

The exporter ignores unrelated source files but rejects an allowed file above
50 MiB. It inspects NPZ member names and permits only frame IDs, activated/raw
pose encodings, pose deltas, and delta norms. Camera Token arrays are rejected.
Every exported file is recorded in `publish_manifest.json` with its relative
path, byte size, and SHA-256 digest.

Direct `git add -f results` is rejected because it can include large generated
artifacts. Git LFS is unnecessary for the expected compact numeric traces and
would add repository and authentication setup.

## Safety and Reproduction

The source must contain valid run metadata whose `run_id` matches every
selection's `complete.json`. A destination run directory must not already
exist, preventing stale files from surviving a repeated export. Git ignore
rules continue to ignore all general results while allowing only JSON, CSV,
and NPZ beneath `results/camera_iteration/`.

The exporter never commits or pushes automatically. It prints the exact
destination; the AutoDL operator reviews its total size, then runs explicit
`git add`, `git commit`, and `git push` commands. Tests use temporary synthetic
trees and require no Torch, CUDA, network, checkpoint, or ScanNet data.
