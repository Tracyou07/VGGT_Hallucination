# Repository Guidelines

## Project Structure & Module Organization

`vggt/` contains the baseline model. Camera observability changes are limited
to `vggt/heads/camera_head.py` and `vggt/models/vggt.py`.
`pre_experiments/camera_iteration/` owns the method study: run contracts,
ScanNet input handling, local checkpoint loading, pose metrics, and the CLI.
`tests/camera_iteration/` contains CPU-only unit tests. Default scenes live in
`configs/camera_iteration_scannet.txt`; AutoDL tooling lives in
`scripts/autodl/`. Research rationale and the executable plan live in `doc/`.

This branch must not contain or import `experiments/scannet_hallucination` or
write to its result namespace. Generated outputs belong under
`results/pre_experiments/camera_iteration/` locally or the explicit external
AutoDL result directory and must not be committed.

## Build, Test, and Development Commands

- `pip install -e .` installs VGGT in editable mode.
- `pip install -r requirements-camera-iteration.txt` installs study helpers;
  use the AutoDL image's existing PyTorch installation.
- `python -m unittest discover -s tests` runs the complete CPU test suite.
- `python -m pre_experiments.camera_iteration.run_study --help` checks the CLI.
- `bash -n scripts/autodl/run_camera_iteration.sh` validates runner syntax.

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

## AutoDL and Commit Guidelines

The AutoDL runner assumes existing weights and data, validates inputs before
model construction, supports processed frames or raw `.sens`, and must never
add download commands. Record actual commands, resolved paths, Git commit, and
result location in run metadata and `log/`. Use short imperative commit titles
such as `Expose camera iteration controls`; keep each commit independently
testable. Never commit datasets, checkpoints, tokens, or generated traces.
