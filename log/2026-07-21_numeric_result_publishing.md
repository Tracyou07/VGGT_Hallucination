# Numeric Result Publishing

## Scope

Added a white-listed export path for Camera Iteration results produced on
AutoDL. Published runs contain metrics, metadata, selected frame IDs, completion
markers, and compact pose traces. ScanNet, RGB/depth images, point clouds,
weights, and high-dimensional Camera Token arrays remain external.

## Safety Contract

- The source run is read-only and must contain matching run IDs.
- Only fixed JSON, CSV, and pose-trace NPZ filenames are copied.
- Allowed files default to a 50 MiB maximum.
- Each copy is recorded by path, byte size, and SHA-256 digest.
- Existing destination runs are rejected instead of overwritten.
- Git commit and push remain explicit operator actions.

## Verification

Pure temporary-tree tests cover successful export, ignored artifacts, manifest
hashes, destination collisions, run-ID mismatch, oversized files, Camera Token
rejection, and Git ignore boundaries. No model inference or external download
is required for this publishing-only change.
