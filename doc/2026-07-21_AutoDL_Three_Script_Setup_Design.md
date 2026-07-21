# AutoDL Three-Script Setup Design

## Purpose

Prepare a new AutoDL machine for the camera-iteration experiment without
mixing acquisition work into the experiment runner. The machine starts with a
Miniconda `base` environment that already contains a compatible PyTorch build
and CUDA runtime. All VGGT experiments use one conda environment named `vggt`.

Environment setup, model acquisition, and ScanNet preparation are three
independent, repeatable operations. `run_camera_iteration.sh` remains an
execution-only entrypoint.

This file is temporary implementation guidance. After all three scripts,
tests, durable README instructions, and local verification are complete, the
implementation must delete this document. The existing camera worktree design
is not part of that cleanup.

## Entry Points

The camera branch will expose exactly these preparation scripts:

```text
scripts/autodl/setup_vggt_env.sh
scripts/autodl/download_vggt_weights.sh
scripts/autodl/prepare_scannet_camera_iteration.sh
```

They run in this order on a new machine but remain independently callable. No
wrapper implicitly invokes all three.

## Environment Setup

`setup_vggt_env.sh` sources
`$CONDA_ROOT/etc/profile.d/conda.sh`, with `CONDA_ROOT` defaulting to
`/root/miniconda3`. It creates `CONDA_ENV_NAME=vggt` by cloning
`CONDA_CLONE_FROM=base` only when the target environment does not exist. This
reuses the image's existing PyTorch/CUDA packages without downloading or
upgrading Torch.

After activation, the script verifies `import torch`, CUDA availability, Torch
version, CUDA version, and GPU name. It installs the repository with
`pip install --no-deps --no-build-isolation -e .`, then installs only missing
lightweight packages represented by `requirements-camera-iteration.txt`.
Neither the requirements file nor the script may contain a Torch package spec.

## VGGT Weight Acquisition

`download_vggt_weights.sh` requires the `vggt` environment and uses its Python
and `huggingface_hub`. Defaults are:

```text
HF_REPO=facebook/VGGT-1B
HF_ENDPOINT=https://hf-mirror.com
HF_HOME=/root/autodl-tmp/hf_home
CKPT_DIR=/root/autodl-tmp/ckpt/VGGT-1B
```

The script downloads with one worker and bounded retries to tolerate restricted
network connections. It accepts `model.safetensors` or `model.pt` as complete
and skips acquisition when either already exists. A successful run must leave
one supported, non-empty checkpoint at `CKPT_DIR`; partial files do not count
as completion. Endpoint, repository, cache, destination, and retry count remain
overridable through environment variables.

## ScanNet Acquisition and Extraction

`prepare_scannet_camera_iteration.sh` uses the official ScanNet
`download-scannet.py` workflow. It does not bypass the data license. The caller
must set `SCANNET_TOS_ACCEPTED=1` to record that the official terms were
accepted; without it, the script exits before downloading. Once this gate is
present, expected interactive confirmations are supplied automatically.

Defaults are:

```text
SCANNET_ROOT=/root/autodl-tmp/datasets/scannetv2
RAW_DIR=$SCANNET_ROOT/raw_sens/scans
PROCESS_DIR=$SCANNET_ROOT/process_scannet
SCENE_LIST=configs/camera_iteration_scannet.txt
SCENE_LIMIT=10
```

The official downloader path can be supplied through
`SCANNET_DOWNLOAD_SCRIPT`. When absent, the script attempts to retrieve it from
the official ScanNet URL and fails with an actionable message if unavailable.
Only each configured scene's `.sens` file is requested; depth images, mesh
files, labels, and the full ScanNet release are outside this experiment.

For each scene, an existing non-empty `.sens` file is skipped. The branch-local
extractor then creates `process_scannet/<scene>/color` and `pose`. A scene with
at least one matching image/finite-pose frame is complete and skipped on later
runs. Incomplete extraction is retried. The script finishes by validating all
scenes selected by `SCENE_LIST` and `SCENE_LIMIT`, so a partial requested
subset cannot be reported as ready.

## Experiment Runner Boundary

`run_camera_iteration.sh` no longer creates a conda environment or installs
packages. It activates `vggt`, verifies CUDA, checkpoint, configured scenes,
and processed color/pose data, then invokes
`pre_experiments.camera_iteration.run_study`. It contains no model or dataset
download command. Missing prerequisites name the preparation script that must
be run.

## Failure and Resume Behavior

- Every script uses `set -euo pipefail` and prints resolved paths before work.
- Existing complete outputs are skipped; missing or empty outputs are retried.
- Downloads use external cache/dataset directories under `/root/autodl-tmp`.
- No checkpoint, dataset, environment, or generated result is committed.
- Preparation failures occur before model construction or experiment output.

## Verification

CPU tests inspect all three scripts and verify fixed defaults, absence of Torch
installation, checkpoint skip behavior, ScanNet license gating, configured
scene selection, and runner separation. Python helper tests use temporary
directories and never contact GitHub, Hugging Face, or ScanNet. Shell scripts
must pass `bash -n`. The final remote gate runs:

```bash
bash scripts/autodl/setup_vggt_env.sh
bash scripts/autodl/download_vggt_weights.sh
SCANNET_TOS_ACCEPTED=1 \
  bash scripts/autodl/prepare_scannet_camera_iteration.sh
SCENE_LIMIT=1 FRAME_COUNTS="25" ITERATIONS="1 2 4 8 16" \
  bash scripts/autodl/run_camera_iteration.sh
```

The smoke run must create complete metrics and trace artifacts, and an
identical rerun must resume without another model forward pass.
