# VGGT Local-Global Consistency Pre-experiment

This branch contains the Round 2A training-free study of camera consistency
between overlapping local windows and frozen 500-frame global inference.
Historical experiment documents and logs remain for traceability, but retired
Round 1, Round 1.5, and Round 1.6 runtime code is intentionally absent.

## Prerequisites

The machine must already provide:

- a Conda environment named `vggt` with compatible PyTorch and CUDA;
- a nonempty `model.safetensors` or `model.pt` under
  `/root/autodl-tmp/ckpt/VGGT-1B`;
- accepted ScanNet terms of use.

Install the checked-out package and any missing project dependencies in that
environment. This branch does not create environments or download weights.

## Prepare ScanNet-50

The default command downloads all scenes in
`configs/fastvggt_scannet50.txt`, extracts RGB frames and raw GT poses, and
optionally downloads the GT PLY files:

```bash
conda activate vggt
SCANNET_TOS_ACCEPTED=1 DOWNLOAD_GT_PLY=1 \
  bash scripts/autodl/prepare_scannet50.sh
```

Existing nonempty assets and processed scenes are reused. Failed official
downloads are retried in isolated staging directories.

## Run Round 2A

The repository retains only the four 500-frame context artifacts required as
frozen input under `results/camera_context/911b598_f4577f584448/`.

```bash
# One-scene smoke
SCENE_LIMIT=1 bash scripts/autodl/run_local_global_consistency.sh

# Fixed four-scene protocol
bash scripts/autodl/run_local_global_consistency.sh
```

Raw window artifacts remain outside Git under
`/root/autodl-tmp/local_global_consistency/results/`. Publish completed scalar
tables only:

```bash
python scripts/autodl/local_global_consistency/export_numeric_results.py \
  --source /root/autodl-tmp/local_global_consistency/results/<run_id>
```

## Development

```bash
pip install -e .
python -m unittest discover -s tests/local_global_consistency -v
bash -n scripts/autodl/prepare_scannet50.sh
bash -n scripts/autodl/run_local_global_consistency.sh
```

Prediction metrics use aligned predictions against raw GT. GT data is never
aligned or replaced.
