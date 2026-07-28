# VGGT Phenomenon Characterization

This branch contains the original ScanNet experiments used to characterize
VGGT camera, depth, and point-cloud failure patterns. These observations remain
active evidence to be expanded, not archived conclusions.

## AutoDL Run

The runner expects the existing `vggt` conda environment, VGGT-1B checkpoint,
and processed ScanNet data:

```bash
git switch phenomenon-characterization
bash scripts/autodl/run_scannet_hallucination.sh
```

Default locations are:

- Environment: `/root/miniconda3/envs/vggt`
- Data: `/root/autodl-tmp/datasets/scannetv2`
- Weights: `/root/autodl-tmp/ckpt/VGGT-1B`
- Results: `/root/autodl-tmp/vggt_hallucination/results`

The runner validates dependencies, CUDA, weights, and then executes inference.
It never creates environments, installs packages, or downloads checkpoints.
Use `RUN_DATA_DOWNLOAD=1` only to invoke the authorized ScanNet data script, or
`RUN_EXTRACT=1` to re-extract existing `.sens` files.

```bash
SCENE_LIMIT=2 FRAME_COUNTS="100 300 500" \
  bash scripts/autodl/run_scannet_hallucination.sh
```

Evaluation resumes by default: complete `metrics.json` selections are reused.
Published observations and numeric artifacts live under
`results/scannet_hallucination/`. See
`scripts/autodl/README_scannet_hallucination.md` for protocol details.
