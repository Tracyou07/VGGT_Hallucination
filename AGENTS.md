# Repository Guidelines

## Project Structure & Module Organization

`vggt/` contains the upstream model package. CO3Dv2 subset construction lives
in `pre_experiments/camera_refiner_data_construction/co3d_download.py`.
`configs/co3d_train41.txt` is the immutable training-category split, and
`scripts/autodl/camera_refiner_data_construction/download_co3d_2050.sh` is the
AutoDL entry point. Focused CPU tests live in
`tests/camera_refiner_data_construction/`.

This branch is data-only. Multiscale Camera Head experiments, ScanNet tools,
and their results belong to the `01-camera-refiner-multiscale` branch and must not
be added here.

## Build, Test, and Development Commands

- `python -m unittest discover -s tests/camera_refiner_data_construction -v`
  runs the offline downloader and contract tests.
- `python -m compileall -q pre_experiments` validates Python syntax.
- `bash -n scripts/autodl/camera_refiner_data_construction/download_co3d_2050.sh`
  validates the shell entry point.
- `bash scripts/autodl/camera_refiner_data_construction/download_co3d_2050.sh`
  starts or resumes the AutoDL dataset build.

## Coding Style & Naming Conventions

Use Python 3.10+, four-space indentation, `snake_case` functions and variables,
and `CamelCase` classes. Add type hints at public boundaries. Shell scripts use
`set -euo pipefail`, quote paths, and reuse the existing `vggt` Conda
environment. Do not add setup, package-install, or checkpoint-download steps.

## Testing and Data Rules

Name tests `test_<behavior>` and keep them independent of CUDA, network access,
checkpoints, and real CO3D archives. Test deterministic selection, valid GT
camera filtering, RGB-only extraction, path traversal rejection, exact quotas,
and restart behavior. Keep the 41-category, 50-sequence-per-category protocol
stable unless the experiment definition changes explicitly.

## Commit & Pull Request Guidelines

Use short imperative commits such as `Add CO3D resume validation`. Never commit
datasets, ZIP files, checkpoints, generated manifests, or results. Pull
requests should state the tested commands and any changes to category, frame,
quality, or sequence quotas.
