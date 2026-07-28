# Repository Guidelines

## Project Structure & Module Organization

`vggt/` contains the frozen baseline model plus camera-trace observability in
the Camera Head and model forward path. `pre_experiments/common/` contains
shared checkpoint, pose, ScanNet, and metadata helpers.
`pre_experiments/local_global_consistency/` owns the Round 2A window runner,
prediction-only scoring, aligned validation, and analysis.
`scripts/autodl/scannet/` contains `.sens` extraction support; the active shell
entries are `prepare_scannet50.sh` and `run_local_global_consistency.sh`.
Focused CPU tests remain under `tests/local_global_consistency/`.

The four committed `frames_500/context_diagnostics.npz` files under
`results/camera_context/911b598_f4577f584448/` are frozen Round 2 inputs, not
an active Round 1.5 result tree. Do not add other files there.

## Build, Test, and Development Commands

- `pip install -e .` installs the checkout into the existing environment.
- `SCANNET_TOS_ACCEPTED=1 DOWNLOAD_GT_PLY=1 bash
  scripts/autodl/prepare_scannet50.sh` prepares the official ScanNet-50 data.
- `SCENE_LIMIT=1 bash scripts/autodl/run_local_global_consistency.sh` runs a
  one-scene GPU smoke test.
- `bash scripts/autodl/run_local_global_consistency.sh` runs the fixed
  four-scene, 500-frame, 100/50-window protocol.
- `python -m unittest discover -s tests/local_global_consistency -v` runs CPU
  tests.
- `python scripts/autodl/local_global_consistency/export_numeric_results.py
  --source /absolute/run` publishes completed scalar CSV/JSON outputs.

## Coding Style & Naming Conventions

Use Python 3.10+, four-space indentation, `snake_case` functions and variables,
and `CamelCase` classes. Preserve upstream VGGT APIs. Document tensor shapes
and coordinate conventions at module boundaries. Shell scripts use
`set -euo pipefail` and quoted paths.

## Testing and Metric Rules

Use `unittest` and name tests `test_<behavior>`. Unit tests must not require
CUDA, checkpoints, network access, or ScanNet credentials. Any metric
containing predictions uses aligned prediction data for conclusions. GT is
always raw. Round 2 detection scores must remain prediction-only; GT appears
only in separately named validation outputs.

## Reproduction and Commit Rules

Assume the remote machine already has the `vggt` Conda environment and VGGT
checkpoint. Do not restore environment-creation or weight-download scripts.
Never commit datasets, checkpoints, images, PLY files, or raw window NPZ
outputs. Use the strict local-global exporter and keep commits independently
testable.
