# Repository Guidelines

## Project Structure & Module Organization

`vggt/` contains the baseline model. Camera observability changes are limited
to `vggt/heads/camera_head.py` and `vggt/models/vggt.py`.
`pre_experiments/camera_iteration/` owns the method study: run contracts,
ScanNet input handling, local checkpoint loading, pose metrics, and the CLI.
`tests/camera_iteration/` contains CPU-only unit tests. Default scenes live in
`configs/camera_iteration_scannet.txt`; AutoDL tooling lives in
`scripts/autodl/`. This branch keeps only its worktree reproduction design in
`doc/`; repository-wide research and implementation guides live on `main`.

This branch must not contain or import `experiments/scannet_hallucination` or
write to its result namespace. Generated outputs belong under
`results/pre_experiments/camera_iteration/` locally or the explicit external
AutoDL result directory. Only artifacts filtered by
`export_numeric_results.py` may be committed beneath
`results/camera_iteration/<run_id>/`.

## Build, Test, and Development Commands

- `pip install -e .` installs VGGT in editable mode.
- `pip install -r requirements-camera-iteration.txt` installs study helpers;
  use the AutoDL image's existing PyTorch installation.
- `python -m unittest discover -s tests` runs the complete CPU test suite.
- `python -m pre_experiments.camera_iteration.run_study --help` checks the CLI.
- `bash -n scripts/autodl/run_camera_iteration.sh` validates runner syntax.
- `bash scripts/autodl/setup_vggt_env.sh` creates/reuses the shared `vggt` env.
- `bash scripts/autodl/download_vggt_weights.sh` prepares only VGGT weights.
- `SCANNET_TOS_ACCEPTED=1 bash scripts/autodl/prepare_scannet_camera_iteration.sh`
  officially downloads and extracts only the configured ScanNet `.sens` files.
- `python scripts/autodl/camera_iteration/export_numeric_results.py --source
  /absolute/run/path` exports compact numeric artifacts for review and commit.

## Coding Style & Naming Conventions

Use Python 3.10+, four-space indentation, `snake_case` functions and variables,
and `CamelCase` classes. Follow neighboring PyTorch code and preserve existing
default APIs. Document tensor shapes and coordinate conventions at module
boundaries. Avoid unrelated formatting and refactoring.

## Testing and Metric Rules

Use `unittest` and name tests `test_<behavior>`. New unit tests must not require
CUDA, checkpoints, network access, or ScanNet. Follow TDD for model and study
code: observe the focused test fail, implement the minimum change, then rerun
the focused and full suites.

Any metric containing a VGGT prediction uses aligned data for the primary
conclusion, including pose ATE/ARE/RPE and predicted depth or point metrics.
Raw values and recovered scale are diagnostics only. Pure GT baselines use raw
data only; mixed prediction/GT metrics still follow the prediction rule.

## Worktree, AutoDL, and Commit Guidelines

This worktree must remain attached to `camera-iteration-preexperiment`; a
worktree does not replace its branch. Do not continue research on detached
HEAD. AutoDL reproduces a pushed branch or recorded commit, never local
worktree metadata. Repository-wide guides remain only on `main`.

The three preparation scripts remain independent. The runner assumes the
`vggt` environment plus complete weights and processed data; it must not create
environments, install packages, download files, or extract `.sens`. Record
commands, resolved paths, commit, and result location in metadata and `log/`.
Keep commits independently testable. Pull requests must list protocol changes
and verification. Never commit datasets, checkpoints, images, point clouds,
high-dimensional Camera Tokens, or files bypassing the numeric exporter.
