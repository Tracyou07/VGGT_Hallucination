# VGGT Hallucination

This repository contains a minimal VGGT codebase plus AutoDL scripts for
observing hallucination across the camera pose, depth, and point-cloud outputs.

## AutoDL One-Click Run

```bash
git clone https://github.com/Tracyou07/VGGT_Hallucination.git
cd VGGT_Hallucination
bash scripts/autodl/run_scannet_hallucination.sh
```

The script installs dependencies, downloads VGGT-1B weights, downloads and
extracts a licensed ScanNet subset, and runs the ScanNet hallucination eval.

Default locations:

- Data: `/root/autodl-tmp/datasets/scannetv2`
- Weights: `/root/autodl-tmp/ckpt/VGGT-1B`
- Results: `/root/autodl-tmp/vggt_hallucination/results`

ScanNet requires official data access. If automatic download fails, place the
official `download-scannet.py` on AutoDL and run:

```bash
SCANNET_DOWNLOAD_SCRIPT=/root/autodl-tmp/download-scannet.py \
bash scripts/autodl/run_scannet_hallucination.sh
```

See `scripts/autodl/README_scannet_hallucination.md` for sampling modes and
common overrides.
