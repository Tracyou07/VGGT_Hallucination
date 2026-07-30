# Camera Hidden-State Attribution

This pre-experiment replays saved normalized Camera tokens through the frozen VGGT
Camera Head. It does not rerun image preprocessing or the Aggregator.

## Inputs

- Global camera-context result containing `frames_500/context_diagnostics.npz`.
- Separate calibration and holdout local-global run directories.
- Frozen ScanNet-50 split manifest.
- Local `VGGT-1B` checkpoint.

## Run

```bash
conda activate vggt
export SOURCE_RUN_DIR=/root/autodl-tmp/camera_context/results/<run>
export CALIBRATION_LOCAL_RUN_DIR=/root/autodl-tmp/local_global_consistency/results/<calibration>
export HOLDOUT_LOCAL_RUN_DIR=/root/autodl-tmp/local_global_consistency/results/<holdout>
export SPLIT_MANIFEST=/root/autodl-tmp/local_global_consistency/scannet50_split.json
export CKPT_DIR=/root/autodl-tmp/ckpt/VGGT-1B
STAGE=all bash scripts/autodl/run_camera_hidden_state_attribution.sh
```

Use `STAGE=smoke`, `calibration`, `holdout`, or `export` to resume one stage.
Calibration freezes 64 translation, rotation, and FoV `(iteration, unit)` pairs.
Holdout refuses a mismatched frozen digest.

## Causal Preference Atlas

The causal stage measures whether each `(iteration, hidden_index)` position
primarily changes final translation, rotation, or FoV. It uses centered
finite differences in the 9D pose-delta space and projects the result through
the Camera Head `fc2` columns, reducing 4096 unit-specific perturbations to 72
basis samples per scene. Deterministic direct hidden perturbations verify the
projection.

```bash
conda activate vggt
export SOURCE_RUN_DIR=/root/autodl-tmp/camera_context/results/<run>
export SPLIT_MANIFEST=/root/autodl-tmp/local_global_consistency/scannet50_split.json
export CKPT_DIR=/root/autodl-tmp/ckpt/VGGT-1B
STAGE=smoke bash scripts/autodl/run_camera_hidden_causal_preference.sh
STAGE=all bash scripts/autodl/run_camera_hidden_causal_preference.sh
```

Set `BASIS_BATCH_SIZE=1` if the 500-frame replay exceeds available GPU memory.
Use `STAGE=calibration`, `holdout`, or `export` to resume. For a holdout-only
run, set `CALIBRATION_RUN_DIR` to the completed calibration directory.
Calibration freezes the 90th-percentile cross-group scales; holdout never
refits them. This atlas is prediction-only and never reads GT values.

## Outputs

Per-scene NPZ statistics remain under `/root/autodl-tmp`. The exporter publishes only
`per_unit.csv`, `per_scene.csv`, `frozen_units.json`, `summary.json`, and authenticated
run metadata. Unit selection is prediction-only. GT validation always aligns the
prediction to raw GT; GT is never aligned.

Causal scene NPZ files also remain under `/root/autodl-tmp`. Its exporter
publishes only `per_position.csv`, `direct_checks.csv`,
`frozen_causal_normalization.json`, `summary.json`, and authenticated run
metadata.
