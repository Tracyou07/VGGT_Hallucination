# Camera Iteration Worktree Design

## Purpose

The `camera-iteration-preexperiment` branch must be a self-contained method
pre-experiment built on a clean VGGT baseline. It must not contain or import the
ScanNet hallucination characterization program, its committed results, or its
download workflow. A fresh AutoDL checkout of this branch must be able to run
the camera-iteration study with one command when weights and ScanNet data are
already present.

On AutoDL, users clone the repository and check out the branch; Git worktrees
are only a local development mechanism:

```bash
git clone https://github.com/Tracyou07/VGGT_Hallucination.git
cd VGGT_Hallucination
git switch camera-iteration-preexperiment
bash scripts/autodl/run_camera_iteration.sh
```

## Chosen Boundary

The branch will be fully self-contained. Reusable loading and pose-evaluation
logic needed by this study will live under
`pre_experiments/camera_iteration/`; it will not import
`experiments.scannet_hallucination`.

Two alternatives were rejected:

- A shared research package or branch would reduce duplication but make a
  checkout depend on synchronized cross-branch revisions.
- Retaining the phenomenon-characterization evaluator would be quicker, but it
  would preserve the current ownership and documentation mismatch.

Small, study-specific duplication is acceptable because reproducibility and a
clear branch boundary are more important than premature reuse.

## Target Repository Layout

```text
vggt/                                  Core model, including trace hooks
pre_experiments/camera_iteration/      Study contracts, metrics, I/O, and CLI
tests/camera_iteration/                CPU-only unit and contract tests
configs/camera_iteration_scannet.txt   Default scene list
scripts/autodl/run_camera_iteration.sh One-command AutoDL entrypoint
scripts/autodl/camera_iteration/       Preflight and optional extraction tools
doc/2026-07-16_Camera_Iteration_Worktree_Design.md  Branch reproduction contract
results/pre_experiments/camera_iteration/  Runtime output, ignored by Git
requirements-camera-iteration.txt      Study-only dependencies
```

`VGGT_DiT_Research_Guide.md` and `VGGT_DiT_Implementation_Plan.md` are
repository-wide guidance maintained only on `main`. They are intentionally not
duplicated in this method worktree.

The inherited `experiments/scannet_hallucination/`,
`results/scannet_hallucination/`, old AutoDL scripts, and hallucination scene
configuration will be absent. `README.md` and `AGENTS.md` will describe only
the baseline plus this method pre-experiment.

## Camera Iteration Interface

`CameraHead` will optionally return a trace for each refinement iteration while
preserving its current default return value. The top-level `VGGT.forward()`
will expose the requested camera iteration count and trace flag. Existing
callers must receive byte-for-byte equivalent output structure and numerically
equivalent final poses when tracing is disabled.

The study CLI will perform one maximum-iteration forward pass per selected
sequence, evaluate every returned iteration, and write tidy per-iteration rows.
Any metric containing a prediction uses aligned data for primary conclusions;
raw values and recovered scale are diagnostic only. Pure GT baselines use raw
data only.

## AutoDL Reproduction Contract

Defaults are:

- `SCANNET_ROOT=/root/autodl-tmp/datasets/scannetv2`
- `CKPT_DIR=/root/autodl-tmp/ckpt/VGGT-1B`
- `RESULT_DIR=/root/autodl-tmp/camera_iteration/results`
- `CONDA_ROOT=/root/miniconda3`
- `CONDA_ENV_NAME=vggt_camera_iteration`

The runner will never download weights or ScanNet. It will fail before model
startup when required paths are missing and print the exact override variable.
If `SCANNET_ROOT/process_scannet` already exists, it runs directly. Otherwise,
when raw `.sens` files are present, it extracts only requested scenes with the
branch-local extraction utility. If neither representation exists, it exits
with an actionable error.

The runner will reuse an existing conda environment when available; otherwise
it clones the AutoDL PyTorch environment and installs only missing study
dependencies. It validates Python, PyTorch, CUDA, the local checkpoint, scene
list, processed frames, and GT poses before inference. Every option remains
overridable through environment variables.

## Outputs and Resume Behavior

All runs write beneath `RESULT_DIR`, using a deterministic run ID derived from
scene, frame selection, iteration count, and relevant evaluation options. Each
run records the Git commit, actual invocation, resolved input paths, frame IDs,
iteration configuration, environment versions, and metric schema. A run is
considered complete only after its metrics and completion metadata are written
atomically. Re-running skips complete runs and retries incomplete ones.

Generated predictions, checkpoints, datasets, images, and point clouds are not
committed. Completed runs may publish only the compact JSON, CSV, and pose-trace
NPZ artifacts validated by `export_numeric_results.py`; Camera Token arrays and
oversized files remain external. Small schema fixtures used by CPU-only tests
may also be tracked.

## Verification

Local verification covers contracts, tensor shapes, default compatibility,
path isolation, metric row generation, preflight behavior, and shell syntax.
Tests must not require CUDA, model downloads, or ScanNet. The final AutoDL gate
runs a small scene/frame configuration, reruns it to verify resume behavior,
and records the command and output location in the branch log.

## Migration Sequence

1. Bring the clean `main` baseline into this branch.
2. Keep repository-wide research and implementation guidance only on `main`.
3. Replace branch-level README and contributor guidance.
4. Create the dedicated package, tests, configuration, and AutoDL runner.
5. Implement camera tracing and the study pipeline test-first.
6. Run local static checks, then execute the small AutoDL reproduction gate.
