# Repository Guidelines

## Project Structure & Module Organization

`vggt/` contains the frozen model and opt-in Camera Head tracing hooks.
`pre_experiments/common/` provides shared artifact, model-loading, ScanNet, and
pose utilities. `pre_experiments/camera_hidden_state_attribution/` and
`pre_experiments/local_global_consistency/` are retained infrastructure for
hidden replay, context windows, alignment, and split validation. New work must
live under `pre_experiments/camera_refiner_data_construction/`, with matching
CPU tests in `tests/camera_refiner_data_construction/` and AutoDL entry points
under `scripts/autodl/camera_refiner_data_construction/`.

Keep design documents in `doc/`. Store immutable scene lists and split
manifests in `configs/`. Do not commit generated `results/` contents.

## Build, Test, and Development Commands

- `pip install -e .` installs the checkout in the existing environment.
- `python -m pytest -q` runs the complete CPU test suite.
- `python -m compileall -q pre_experiments vggt` validates Python imports and
  syntax.
- `SCANNET_TOS_ACCEPTED=1 bash scripts/autodl/prepare_scannet50.sh` prepares
  the authorized ScanNet scenes while reusing completed downloads.
- `SCANNET_TOS_ACCEPTED=1 bash scripts/autodl/camera_refiner_data_construction/prepare_scannet_adaptation200.sh`
  serially prepares the disjoint 160/20/20 ScanNet adaptation split.

## Coding Style & Naming Conventions

Use Python 3.10+, four-space indentation, `snake_case` functions and variables,
and `CamelCase` classes. Add type hints at public boundaries. Document tensor
shapes, frame identity, refinement iteration, and coordinate conventions.
Shell scripts use `set -euo pipefail` and quote all paths.

## Testing and Metric Rules

Name tests `test_<behavior>` and keep unit tests independent of CUDA,
checkpoints, network access, and ScanNet credentials. Test window tails,
overlap tie-breaking, split leakage, artifact provenance, and non-finite
inputs explicitly.

ScanNet adaptation scenes must come from `configs/scannetv2_train_official.txt`,
exclude `configs/fastvggt_scannet50.txt`, and contain at least 500 matching RGB
frames and finite poses. Preserve the frozen candidate order and role split.

Any metric containing a prediction must use the aligned prediction. Ground
truth always remains raw; never align or replace GT. Calibration may select
scales and mixtures. Holdout must consume a frozen policy without refitting.

## Commit & Pull Request Guidelines

Use short imperative commits such as `Add multiscale hidden manifest`.
Keep generated tensors and figures out of Git. Pull requests must state the
tested command, data split, source commit, checkpoint provenance, and whether
the change affects calibration or untouched holdout evaluation.
