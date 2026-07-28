# Repository Guidelines

## Project Structure & Module Organization

`vggt/` contains the baseline model and Camera Head observability hooks.
`pre_experiments/camera_head_amplification/` owns Round 1.6 checkpoint
selection, replay, metrics, and its CLI. The retained
`pre_experiments/camera_iteration/{contracts,model_io,pose_metrics}.py` files
are shared dependencies, not a runnable Round 1 study. Tests remain under
`tests/camera_iteration/` to preserve history. AutoDL entrypoints are in
`scripts/autodl/`; only the Camera Head replay runner belongs on this branch.

`results/camera_context/911b598_f4577f584448/` is the frozen Round 1.5 input.
Round 1.6 scalar results belong under `results/camera_head_amplification/`.
Keep all raw activation arrays and checkpoints outside Git.

## Build, Test, and Development Commands

- `pip install -e .` installs VGGT in editable mode.
- `pip install -r requirements-camera-head-amplification.txt` installs the
  replay-specific Python dependencies without replacing Torch/CUDA.
- `python -m unittest discover -s tests` runs the CPU regression suite.
- `bash scripts/autodl/run_camera_head_amplification.sh` runs the frozen-head
  replay from committed Round 1.5 tokens.
- `python scripts/autodl/camera_head_amplification/export_numeric_results.py
  --source /absolute/run` publishes a validated scalar-only run.

## Coding, Testing, and Metric Rules

Use Python 3.10+, four-space indentation, `snake_case` functions and variables,
and `CamelCase` classes. Match neighboring PyTorch code and document tensor
shapes at module boundaries. Tests use `unittest` and names of the form
`test_<behavior>`; unit tests must not require CUDA, checkpoints, or network.

Any metric containing a prediction uses independently aligned predictions for
the primary conclusion. Raw predictions and recovered scale are diagnostics.
GT is always raw and is never replaced by an aligned copy.

## Worktree and Commit Guidelines

Keep this worktree attached to `camera-head-amplification-preexperiment`.
Round 1 and Round 1.5 runnable code remain on their own branches; only the
frozen Round 1.5 numeric input is retained here. Keep `doc/` and `log/` as the
research record. Commits should be independently testable, and pull requests
must state protocol changes and verification. Never commit datasets, images,
point clouds, checkpoints, high-dimensional activations, or files bypassing
the strict scalar exporter.
