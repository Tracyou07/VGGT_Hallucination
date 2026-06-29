# AutoDL ScanNet Hallucination Probe

This folder contains one-click AutoDL scripts for the VGGT hallucination study.
The pipeline downloads VGGT weights, downloads/extracts a licensed ScanNet subset,
then evaluates depth, camera pose, derived point clouds, and optional native
`world_points`.

## Quick Start

```bash
git clone https://github.com/Tracyou07/VGGT_Hallucination.git
cd VGGT_Hallucination
bash scripts/autodl/run_scannet_hallucination.sh
```

Default output paths:

- Data: `/root/autodl-tmp/datasets/scannetv2`
- Weights: `/root/autodl-tmp/ckpt/VGGT-1B`
- Results: `/root/autodl-tmp/vggt_hallucination/results`

## ScanNet Access

ScanNet requires terms-of-use acceptance. If the automatic script download fails,
place the official `download-scannet.py` on AutoDL and run:

```bash
SCANNET_DOWNLOAD_SCRIPT=/root/autodl-tmp/download-scannet.py \
bash scripts/autodl/run_scannet_hallucination.sh
```

## Common Overrides

```bash
SCENE_LIMIT=2 FRAME_COUNTS="100 300" bash scripts/autodl/run_scannet_hallucination.sh
SAMPLING=regime_step bash scripts/autodl/run_scannet_hallucination.sh
RUN_DOWNLOADS=0 INSTALL_ENV=0 bash scripts/autodl/run_scannet_hallucination.sh
EVAL_NATIVE_POINTS=0 EVAL_COUNTERFACTUALS=0 bash scripts/autodl/run_scannet_hallucination.sh
```

Sampling modes:

- `prefix`: strict cumulative sequence; best for observing error accumulation.
- `uniform`: independent uniform sampling for each frame count.
- `nested_uniform`: samples from a shared long-frame base.
- `regime_step`: matches the Regime/FastVGGT ScanNet eval style.

## Result Files

Each scene/count writes:

- `metrics.json`: pose, depth, point-cloud, and counterfactual metrics.
- `selected_frame_ids.json`: exact frames used.
- `predicted_cameras.npz`: predicted camera matrices and GT poses.
- `trajectory.png`: aligned predicted/GT trajectory preview.

The full run also writes `summary.csv` and `summary.json`.
