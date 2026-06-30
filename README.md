# VGGT Hallucination

This repository contains a minimal VGGT codebase plus AutoDL scripts for
observing hallucination across the camera pose, depth, and point-cloud outputs.

## AutoDL One-Click Run

```bash
cd /root/autodl-tmp
git clone https://github.com/Tracyou07/VGGT_Hallucination.git
cd VGGT_Hallucination
bash scripts/autodl/run_scannet_hallucination.sh
```

The script clones the AutoDL image's existing CUDA/PyTorch conda environment,
installs only the missing VGGT helper dependencies, downloads VGGT-1B weights,
downloads/extracts a licensed ScanNet subset, and runs the eval.

Default locations:

- Code: `/root/autodl-tmp/VGGT_Hallucination`
- Conda env: `/root/miniconda3/envs/vggt_hallucination`
- Data: `/root/autodl-tmp/datasets/scannetv2`
- Weights: `/root/autodl-tmp/ckpt/VGGT-1B`
- Results: `/root/autodl-tmp/vggt_hallucination/results`

ScanNet requires official data access. If automatic download fails, place the
official `download-scannet.py` on AutoDL and run:

```bash
SCANNET_DOWNLOAD_SCRIPT=/root/autodl-tmp/download-scannet.py \
bash scripts/autodl/run_scannet_hallucination.sh
```

If VGGT weight download from Hugging Face is reset, rerun with the mirror
endpoint:

```bash
HF_ENDPOINT=https://hf-mirror.com bash scripts/autodl/run_scannet_hallucination.sh
```

If the conda environment, weights, and uploaded ScanNet files already exist,
reuse them without reinstalling dependencies:

```bash
INSTALL_ENV=0 RUN_DOWNLOADS=0 SCENE_LIMIT=5 FRAME_COUNTS="100 300" \
bash scripts/autodl/run_scannet_hallucination.sh
```

This still activates `/root/miniconda3/envs/vggt_hallucination` and extracts
uploaded `.sens` files into `process_scannet` before evaluation.

See `scripts/autodl/README_scannet_hallucination.md` for sampling modes and
common overrides.
