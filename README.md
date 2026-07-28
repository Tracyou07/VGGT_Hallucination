# VGGT Camera Iteration Pre-experiment

The `camera-iteration-preexperiment` branch exposes and evaluates intermediate
VGGT Camera Head refinements without training or parameter updates. Its model
hooks, ScanNet reader, metrics, tests, and AutoDL entrypoint are self-contained;
no phenomenon-characterization code or result tree is imported.

## AutoDL Quick Start

This branch assumes the AutoDL machine already has the `vggt` conda
environment, VGGT-1B checkpoint, and ScanNet data. Run the experiment with:

```bash
git clone https://github.com/Tracyou07/VGGT_Hallucination.git
cd VGGT_Hallucination
git switch camera-iteration-preexperiment
bash scripts/autodl/run_camera_iteration.sh
```

Default external paths are:

- ScanNet: `/root/autodl-tmp/datasets/scannetv2`
- VGGT-1B: `/root/autodl-tmp/ckpt/VGGT-1B`
- Results: `/root/autodl-tmp/camera_iteration/results`
- Conda environment: `vggt`

The runner validates these existing inputs and executes the study. If ScanNet
must be repaired or extended, the retained data-only script invokes the
official downloader:

```bash
SCANNET_TOS_ACCEPTED=1 bash scripts/autodl/prepare_scannet_camera_iteration.sh
```

Every path and experiment size can be overridden with environment variables
documented in `pre_experiments/camera_iteration/README.md`.

A smaller smoke configuration is:

```bash
SCENE_LIMIT=1 FRAME_COUNTS="25" ITERATIONS="1 2 4 8 16" \
  bash scripts/autodl/run_camera_iteration.sh
```

## Publish Numeric Results

After a completed AutoDL run, export its compact measurements into the
repository. Pass the exact run directory printed by the study:

```bash
python scripts/autodl/camera_iteration/export_numeric_results.py \
  --source /root/autodl-tmp/camera_iteration/results/<run_id>

du -sh results/camera_iteration/<run_id>
git status --short results/camera_iteration/<run_id>
git add results/camera_iteration/<run_id>
git commit -m "Add camera iteration numeric results <run_id>"
git push origin camera-iteration-preexperiment
```

The exporter includes JSON/CSV measurements and compact pose-only
`camera_trace.npz` files. It rejects high-dimensional Camera Token arrays and
files over 50 MiB, and never copies images, point clouds, datasets, or weights.
`publish_manifest.json` records every copied file's size and SHA-256 digest.

## Development

```bash
pip install -r requirements.txt
pip install -r requirements-camera-iteration.txt
pip install -e .
python -m unittest discover -s tests
```

Core model changes live in `vggt/`. The method package lives in
`pre_experiments/camera_iteration/`, CPU tests in `tests/camera_iteration/`,
and the branch-specific reproduction design in
`doc/2026-07-16_Camera_Iteration_Worktree_Design.md`. The repository-wide
research guide and implementation plan are maintained only on `main`.

Checkpoint-backed numeric results are stored under
`results/camera_iteration/faf32cb_7ed95de4ff81/`.
