# Repository Guidelines

## Project Structure & Module Organization

`vggt/` contains the frozen VGGT baseline plus opt-in Camera Head tracing.
All experiment code belongs under `pre_experiments/camera_refiner_training/`,
including windowing, geometry, artifact I/O, training, inference, and metrics.
Matching tests belong under
`tests/camera_refiner_training/`. AutoDL entry points belong under
`scripts/autodl/camera_refiner_training/`.

## Build, Test, and Development Commands

- `pip install -e .` installs the checkout into the active environment.
- `python -m unittest discover -s tests -v` runs retained CPU regression tests.
- `python -m compileall -q pre_experiments` checks Python syntax.
- `bash -n scripts/autodl/camera_refiner_training/{train,infer}.sh` validates the
  AutoDL shell entry points.

Assume AutoDL already has the `vggt` Conda environment, compatible CUDA/PyTorch,
processed ScanNet data, and the VGGT checkpoint. Do not recreate environments or
download weights.

## Coding Style & Naming Conventions

Use Python 3.10+, four-space indentation, `snake_case` functions and variables, and
`CamelCase` classes. Preserve upstream VGGT APIs. Document tensor shapes, pose
direction, coordinate gauge, and alignment rules at module boundaries. Shell scripts
use `set -euo pipefail` and quote paths.

## Training and Metric Rules

The refiner predicts camera-center translation residuals only. Final rotations must
remain numerically identical to global VGGT rotations. Translation-preferred units
are conditions, not direct hidden-state edits. Keep training, validation, development,
and final-test scenes disjoint. GT-derived training labels must be clearly named; raw
GT remains unchanged for evaluation. Prediction metrics use aligned predictions.

Tests use `unittest`, follow `test_<behavior>`, and must not require CUDA, network
access, checkpoints, or ScanNet credentials. Add regression tests for coordinate
conversion, overlap fusion, resumability, and exact rotation preservation.

## Artifacts and Commits

Store remote artifacts under
`/root/autodl-tmp/results/camera_refiner_training/<run_id>`. Never commit datasets,
checkpoints, images, PLY files, full hidden traces, or raw window NPZ files. Export
only frozen manifests, scalar CSV/JSON summaries, and concise analysis documents.
Use short imperative commit messages and keep each commit independently testable.
