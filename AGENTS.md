# Repository Guidelines

## Project Structure & Module Organization

`vggt/` contains the baseline model. The active characterization pipeline is
under `experiments/scannet_hallucination/`; fast CPU tests are in
`probe/tests/`. Scene lists live in `configs/`, and AutoDL validation, ScanNet
preparation, execution, and numeric publishing tools live in `scripts/autodl/`.
Committed observations belong under `results/scannet_hallucination/`.
Branch-specific designs and research records remain in `doc/` and `log/`;
repository-wide DiT guides stay on `main`.

## Build, Test, and Development Commands

- `pip install -e .` installs VGGT in editable mode.
- `python -m unittest discover -s probe/tests` runs CPU tests.
- `python -m experiments.scannet_hallucination.run_eval --help` checks the
  experiment CLI.
- `bash scripts/autodl/run_scannet_hallucination.sh` runs with the existing
  `vggt` environment, checkpoint, and processed data.
- `RUN_DATA_DOWNLOAD=1 bash scripts/autodl/run_scannet_hallucination.sh`
  invokes the authorized ScanNet-only preparation path.

## Coding Style & Testing

Use Python 3.10+, four-space indentation, `snake_case` functions and variables,
and `CamelCase` classes. Follow neighboring PyTorch style and document tensor
shapes and coordinate conventions. Tests use `unittest`, are named
`test_<behavior>`, and must not require CUDA, checkpoints, network, or ScanNet.
Avoid unrelated formatting and refactoring.

## Metric Interpretation Rules

Any metric containing a VGGT prediction uses aligned predictions for primary
conclusions, including pose, depth, and point metrics. Raw predictions and
recovered scale are diagnostics only. Pure GT baselines use raw GT. GT is
always raw and is never replaced by an aligned copy.

## Worktree and Commit Guidelines

Keep this worktree attached to `phenomenon-characterization`. Existing
observations are preliminary and should be expanded with more scenes rather
than treated as archived conclusions. Record commands, scene selection, commit,
and output paths in `log/`. Use concise imperative commit messages. Never
commit private datasets, checkpoints, credentials, or point clouds; publish
only the reviewed result contract.
