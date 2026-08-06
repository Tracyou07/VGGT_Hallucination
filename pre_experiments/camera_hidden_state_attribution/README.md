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
export SOURCE_RUN_DIR=/root/autodl-tmp/results/camera_context/results/RUN_ID
export CALIBRATION_LOCAL_RUN_DIR=/root/autodl-tmp/results/local_global_consistency/results/CALIBRATION_RUN_ID
export HOLDOUT_LOCAL_RUN_DIR=/root/autodl-tmp/results/local_global_consistency/results/HOLDOUT_RUN_ID
export SPLIT_MANIFEST=/root/autodl-tmp/results/local_global_consistency/scannet50_split.json
export CKPT_DIR=/root/autodl-tmp/ckpt/VGGT-1B
STAGE=all bash scripts/autodl/run_camera_hidden_state_attribution.sh
```

Use `STAGE=smoke`, `calibration`, `holdout`, or `export` to resume one stage.
Calibration freezes 64 translation, rotation, and FoV `(iteration, unit)` pairs.
Holdout refuses a mismatched frozen digest.

## Outputs

Per-scene NPZ statistics remain under `/root/autodl-tmp`. The exporter publishes only
`per_unit.csv`, `per_scene.csv`, `frozen_units.json`, `summary.json`, and authenticated
run metadata. Unit selection is prediction-only. GT validation always aligns the
prediction to raw GT; GT is never aligned.
