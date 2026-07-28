# AutoDL ScanNet Hallucination Probe

The pipeline evaluates camera pose, depth, derived point clouds, and optional
native `world_points` using an existing AutoDL setup.

## Run

```bash
git switch phenomenon-characterization
bash scripts/autodl/run_scannet_hallucination.sh
```

Required defaults are the `vggt` conda environment,
`/root/autodl-tmp/ckpt/VGGT-1B`, and processed data under
`/root/autodl-tmp/datasets/scannetv2/process_scannet`.

The runner does not create environments, install packages, or download model
weights. For authorized ScanNet maintenance:

```bash
RUN_DATA_DOWNLOAD=1 bash scripts/autodl/run_scannet_hallucination.sh
RUN_EXTRACT=1 bash scripts/autodl/run_scannet_hallucination.sh
```

ScanNet downloading requires prior terms-of-use acceptance and the official
downloader. Override its location with `SCANNET_DOWNLOAD_SCRIPT`.

## Protocol Overrides

```bash
SCENE_LIMIT=2 FRAME_COUNTS="100 300" bash scripts/autodl/run_scannet_hallucination.sh
SAMPLING=regime_step bash scripts/autodl/run_scannet_hallucination.sh
EVAL_NATIVE_POINTS=0 EVAL_COUNTERFACTUALS=0 \
  bash scripts/autodl/run_scannet_hallucination.sh
```

Sampling modes are `prefix`, `uniform`, `nested_uniform`, and `regime_step`.
Each selection writes metrics, selected frame IDs, predicted cameras, and a
trajectory preview. The run root contains `summary.csv` and `summary.json`.
