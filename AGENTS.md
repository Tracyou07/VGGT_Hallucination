# Repository Guidelines

## Project Structure & Module Organization

`vggt/` contains the frozen baseline plus opt-in Camera Head hidden tracing and
ablation. `pre_experiments/camera_hidden_state_attribution/` owns token replay,
unit ranking, intervention metrics, and numeric aggregation. It consumes frozen
artifacts from `pre_experiments/local_global_consistency/`; do not duplicate
that inference pipeline. The AutoDL entry point is
`scripts/autodl/run_camera_hidden_state_attribution.sh`. Focused CPU tests live
under `tests/camera_hidden_state_attribution/`.

The four committed `frames_500/context_diagnostics.npz` files under
`results/camera_context/911b598_f4577f584448/` are frozen Round 2 inputs, not
an active Round 1.5 result tree. Do not add other files there.

## Build, Test, and Development Commands

- `pip install -e .` installs the checkout into the existing environment.
- `SCANNET_TOS_ACCEPTED=1 DOWNLOAD_GT_PLY=1 bash
  scripts/autodl/prepare_scannet50.sh` prepares the official ScanNet-50 data.
- `STAGE=smoke bash scripts/autodl/run_camera_hidden_state_attribution.sh`
  replays one calibration scene.
- `STAGE=all bash scripts/autodl/run_camera_hidden_state_attribution.sh` runs
  smoke, calibration, holdout, and numeric export in order.
- `python -m unittest discover -s tests/camera_hidden_state_attribution -v`
  runs attribution CPU tests.
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
CUDA, checkpoints, network access, or ScanNet credentials. Unit identity is
`(iteration, hidden_index)`. Calibration selects units; holdout must never
refit them. Any error metric containing predictions uses aligned prediction
data. GT is always raw. Ranking remains prediction-only; GT appears only in
separately named validation outputs.

## Reproduction and Commit Rules

Assume the remote machine already has the `vggt` Conda environment and VGGT
checkpoint. Do not restore environment-creation or weight-download scripts.
Never commit datasets, checkpoints, images, PLY files, or raw window NPZ
outputs. Use the strict local-global exporter and keep commits independently
testable.
