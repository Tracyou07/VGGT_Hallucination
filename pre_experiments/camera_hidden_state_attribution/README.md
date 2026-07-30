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
export SPLIT_MANIFEST="$PWD/configs/scannet50_local_global_split.json"
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
export SPLIT_MANIFEST="$PWD/configs/scannet50_local_global_split.json"
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

## Short-to-Long Hidden Replacement

This stage freezes the translation intersection between the calibration
context-drift Top-64 and causal-effect Top-64. It interpolates those
long-context post-GELU pose-branch values toward matched short-window values:
`h_new = h_long + alpha * (h_short - h_long)`.
Overlapping short windows use the most interior observation, with the earlier
window breaking ties. Five frozen random sets outside both source Top-64 sets
are matched by refinement iteration and replacement count.

```bash
conda activate vggt
export SOURCE_RUN_DIR=/root/autodl-tmp/camera_context/results/d33d98b_309a9a586242
export CALIBRATION_LOCAL_RUN_DIR=/root/autodl-tmp/local_global_consistency/scannet50/runs/calibration/3d5de75_c5c5ae0e55fe
export HOLDOUT_LOCAL_RUN_DIR=/root/autodl-tmp/local_global_consistency/scannet50/runs/holdout/3d5de75_35564d765bb5
export ATTRIBUTION_CALIBRATION_DIR=/root/autodl-tmp/camera_hidden_state_attribution/results/bba0cdf_28eadd33cbf8
export CAUSAL_CALIBRATION_DIR=/root/autodl-tmp/camera_hidden_causal_preference/results/9368808_99c8a9ed393c
export SPLIT_MANIFEST="$PWD/configs/scannet50_local_global_split.json"
export CKPT_DIR=/root/autodl-tmp/ckpt/VGGT-1B
export ALPHAS=0.01,0.02,0.05,0.1,0.25,0.5,1.0
STAGE=smoke bash scripts/autodl/run_camera_hidden_replacement.sh
STAGE=all bash scripts/autodl/run_camera_hidden_replacement.sh
```

Smoke and calibration evaluate the full alpha grid using `control_00`, then
freeze the alpha with the lowest scene-mean aligned translation error delta.
Holdout evaluates only that alpha using every control set recorded in the
frozen manifest (`control_00` through `control_04`). Run identity includes the
evaluated control names, so an older single-control holdout cannot be reused.
The summary reports `configured_control_repeats` separately from
`evaluated_control_repeats`; multi-control comparisons first average controls
within each scene. Each predicted trajectory is aligned independently before
error is computed against raw GT. Per-scene NPZ files remain under
`/root/autodl-tmp`; the exporter publishes only strict numeric CSV/JSON
artifacts.

## Scene-Adaptive Alpha

This pre-experiment tests whether prediction-only consistency can select
`alpha` per scene before an AdaLN-style refiner is trained. It uses scene
medians of global-local token cosine, global-local pose translation, and
local-local pose translation. Calibration derives oracle labels only from the
candidate curves `0.01`, `0.02`, and `0.05`, reports leave-one-scene-out
performance, and freezes a ridge selector. The selector contains no GT labels.
Holdout predicts one alpha per scene, then evaluates the 41 selected units and
all five controls at that same alpha.

```bash
conda activate vggt
export CALIBRATION_SCORE_RUN_DIR=/root/autodl-tmp/local_global_consistency/scannet50/runs/calibration/3d5de75_c5c5ae0e55fe
export HOLDOUT_SCORE_RUN_DIR=/root/autodl-tmp/local_global_consistency/scannet50/runs/holdout/3d5de75_35564d765bb5
export REPLACEMENT_CALIBRATION_DIR="$(cat /root/autodl-tmp/camera_hidden_replacement/state/calibration_run.txt)"
export FIXED_REPLACEMENT_HOLDOUT_DIR=/root/autodl-tmp/camera_hidden_replacement/results/ae2bfc8_64df3fe10532
export SOURCE_RUN_DIR=/root/autodl-tmp/camera_context/results/d33d98b_309a9a586242
export HOLDOUT_LOCAL_RUN_DIR=/root/autodl-tmp/local_global_consistency/scannet50/runs/holdout/3d5de75_35564d765bb5
export SPLIT_MANIFEST="$PWD/configs/scannet50_local_global_split.json"
export CKPT_DIR=/root/autodl-tmp/ckpt/VGGT-1B

STAGE=calibration bash scripts/autodl/run_camera_hidden_adaptive_alpha.sh
STAGE=holdout bash scripts/autodl/run_camera_hidden_adaptive_alpha.sh
STAGE=export bash scripts/autodl/run_camera_hidden_adaptive_alpha.sh
```

Calibration is CPU-only. Holdout requires the GPU because it replays Camera
Head once per predicted scene alpha. GT is used only for aligned evaluation
after alpha assignment has been frozen.
