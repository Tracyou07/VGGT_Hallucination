# AutoDL Three-Stage Setup

## Completed

- Split new-machine preparation into independent environment, VGGT weight, and
  authorized ScanNet scripts.
- Standardized the experiment environment as `vggt`, cloned from `base`, while
  preserving the AutoDL image's existing Torch/CUDA installation.
- Added resumable VGGT-1B acquisition through `huggingface_hub` and a
  configurable Hugging Face endpoint.
- Added official ScanNet `.sens` acquisition for the configured ten scenes,
  gated by explicit ToS acceptance, followed by color/raw-pose extraction.
- Made the experiment runner execution-only.
- Tightened preflight checks: checkpoints and `.sens` files must be non-empty;
  every requested processed scene needs a matching image and finite 4x4 pose.

## Verification

Local CPU tests and Bash syntax checks were run without network access, CUDA,
weights, or ScanNet. Real downloads and checkpoint-backed execution remain for
the AutoDL machine and must be recorded separately with commit and output path.
