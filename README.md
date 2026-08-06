# VGGT ScanNet-50 Local-Global Validation

This branch tests whether prediction-only disagreement between 100-frame local
inference and the requested 500-frame global inference identifies long-context
Camera Pose degradation. It does not train or modify VGGT.

## Prerequisites

AutoDL must already contain:

- the `vggt` Conda environment with compatible CUDA and PyTorch;
- the checkout under `/root/autodl-tmp/VGGT_Hallucination`;
- `VGGT-1B` under `/root/autodl-tmp/ckpt/VGGT-1B`;
- processed ScanNet scenes under
  `/root/autodl-tmp/datasets/scannetv2/process_scannet`;
- one complete 50-scene Camera Context source run: 500 frames per scene, except
  `scene0150_00`, which must contain exactly its 430 available frames.

The scripts never create an environment, download weights, or select the
newest result directory. Matplotlib is required only for PNG diagnostics and
is available through the existing `viz` optional dependency.

## Freeze The 10/40 Split

Set the exact global source directory, then generate the raw-GT-only split
before running any new local-window inference:

```bash
conda activate vggt
cd /root/autodl-tmp/VGGT_Hallucination
export SOURCE_RUN_DIR=/root/autodl-tmp/results/camera_context/results/RUN_ID

python -m pre_experiments.local_global_consistency.split \
  --data-dir /root/autodl-tmp/datasets/scannetv2/process_scannet \
  --scene-list configs/fastvggt_scannet50.txt \
  --source-run-dir "$SOURCE_RUN_DIR" \
  --output configs/scannet50_local_global_split.json \
  --seed 33

python -m pre_experiments.local_global_consistency.split \
  --validate configs/scannet50_local_global_split.json \
  --scene-list configs/fastvggt_scannet50.txt
```

Commit the split JSON before formal inference. Construction reads source frame
IDs and raw GT poses, never VGGT prediction arrays.

## Run On AutoDL

A one-scene smoke runs one calibration scene and one holdout scene without
formal analysis:

```bash
SOURCE_RUN_DIR="$SOURCE_RUN_DIR" SCENE_LIMIT=1 STAGE=all \
  bash scripts/autodl/run_scannet50_local_global.sh
```

Run complete calibration followed by the frozen-threshold holdout:

```bash
SOURCE_RUN_DIR="$SOURCE_RUN_DIR" STAGE=all \
  bash scripts/autodl/run_scannet50_local_global.sh
```

Rerun the same command to resume completed windows. Stable pointers and logs
are written under
`/root/autodl-tmp/results/local_global_consistency/scannet50/{pointers,logs}/`.
For separate jobs, run `STAGE=calibration`, then:

```bash
SOURCE_RUN_DIR="$SOURCE_RUN_DIR" \
CALIBRATION_RUN_DIR=/absolute/calibration/run \
STAGE=holdout bash scripts/autodl/run_scannet50_local_global.sh
```

Each completed run writes PNG diagnostics under `visualizations/`. Holdout
figures cover split difficulty, per-scene error growth, score-versus-GT
association, frozen reliability coverage, and scene-bootstrap confidence
intervals.

Normal scenes produce nine length-100, stride-50 windows.
`scene0150_00` produces eight, including the tail window `[330, 430)`. The
formal workload is therefore 449 windows: calibration/holdout counts are
89/360 or 90/359 according to the frozen split.

## Export Numeric Evidence

Figures and raw window NPZ files stay outside Git. Publish only authenticated
split, threshold, CSV, JSON, and manifest files:

```bash
python scripts/autodl/local_global_consistency/export_numeric_results.py \
  --calibration-run /absolute/calibration/run \
  --holdout-run /absolute/holdout/run \
  --split-manifest configs/scannet50_local_global_split.json
```

Prediction-only scores never contain GT-derived values. Prediction-versus-GT
metrics align predictions to raw GT; GT arrays are never aligned or replaced.

## Development

```bash
python -m unittest discover -s tests/local_global_consistency -v
bash -n scripts/autodl/run_scannet50_local_global.sh
python -m compileall -q pre_experiments/local_global_consistency
```
