#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
AUTODL_TMP="${AUTODL_TMP:-/root/autodl-tmp}"
CALIBRATION_SCORE_RUN_DIR="${CALIBRATION_SCORE_RUN_DIR:?set CALIBRATION_SCORE_RUN_DIR}"
HOLDOUT_SCORE_RUN_DIR="${HOLDOUT_SCORE_RUN_DIR:?set HOLDOUT_SCORE_RUN_DIR}"
REPLACEMENT_CALIBRATION_DIR="${REPLACEMENT_CALIBRATION_DIR:?set REPLACEMENT_CALIBRATION_DIR}"
FIXED_REPLACEMENT_HOLDOUT_DIR="${FIXED_REPLACEMENT_HOLDOUT_DIR:?set FIXED_REPLACEMENT_HOLDOUT_DIR}"
SOURCE_RUN_DIR="${SOURCE_RUN_DIR:?set SOURCE_RUN_DIR}"
HOLDOUT_LOCAL_RUN_DIR="${HOLDOUT_LOCAL_RUN_DIR:?set HOLDOUT_LOCAL_RUN_DIR}"
SPLIT_MANIFEST="${SPLIT_MANIFEST:?set SPLIT_MANIFEST}"
CKPT_DIR="${CKPT_DIR:-$AUTODL_TMP/ckpt/VGGT-1B}"
OUT_DIR="${OUT_DIR:-$AUTODL_TMP/camera_hidden_adaptive_alpha/results}"
PUBLISH_ROOT="${PUBLISH_ROOT:-$ROOT/results/camera_hidden_adaptive_alpha}"
STATE_DIR="${STATE_DIR:-$AUTODL_TMP/camera_hidden_adaptive_alpha/state}"
STAGE="${STAGE:-all}"
DEVICE="${DEVICE:-cuda}"

for directory in \
  "$CALIBRATION_SCORE_RUN_DIR" \
  "$HOLDOUT_SCORE_RUN_DIR" \
  "$REPLACEMENT_CALIBRATION_DIR" \
  "$FIXED_REPLACEMENT_HOLDOUT_DIR" \
  "$SOURCE_RUN_DIR" \
  "$HOLDOUT_LOCAL_RUN_DIR"; do
  [[ -d "$directory" ]] || { echo "missing directory: $directory" >&2; exit 2; }
done
[[ -f "$SPLIT_MANIFEST" ]] || { echo "missing SPLIT_MANIFEST=$SPLIT_MANIFEST" >&2; exit 2; }
[[ -f "$CKPT_DIR/model.safetensors" || -f "$CKPT_DIR/model.pt" ]] || {
  echo "missing VGGT checkpoint in CKPT_DIR=$CKPT_DIR" >&2
  exit 2
}

mkdir -p "$STATE_DIR"
cd "$ROOT"

run_calibration() {
  python -m pre_experiments.camera_hidden_state_attribution.run_adaptive_replacement \
    --stage calibration \
    --score-run-dir "$CALIBRATION_SCORE_RUN_DIR" \
    --replacement-calibration-dir "$REPLACEMENT_CALIBRATION_DIR" \
    --split-manifest "$SPLIT_MANIFEST" \
    --out-dir "$OUT_DIR" \
    --run-dir-file "$STATE_DIR/calibration_run.txt" \
    --device cpu
}

run_holdout() {
  local calibration_run
  calibration_run="${ADAPTIVE_CALIBRATION_RUN_DIR:-$(<"$STATE_DIR/calibration_run.txt")}"
  python -m pre_experiments.camera_hidden_state_attribution.run_adaptive_replacement \
    --stage holdout \
    --score-run-dir "$HOLDOUT_SCORE_RUN_DIR" \
    --selector "$calibration_run/frozen_selector.json" \
    --frozen-replacement "$REPLACEMENT_CALIBRATION_DIR/frozen_replacement.json" \
    --fixed-holdout-dir "$FIXED_REPLACEMENT_HOLDOUT_DIR" \
    --source-run-dir "$SOURCE_RUN_DIR" \
    --local-run-dir "$HOLDOUT_LOCAL_RUN_DIR" \
    --split-manifest "$SPLIT_MANIFEST" \
    --ckpt-dir "$CKPT_DIR" \
    --out-dir "$OUT_DIR" \
    --run-dir-file "$STATE_DIR/holdout_run.txt" \
    --device "$DEVICE"
}

run_export() {
  local calibration_run holdout_run
  calibration_run="${ADAPTIVE_CALIBRATION_RUN_DIR:-$(<"$STATE_DIR/calibration_run.txt")}"
  holdout_run="${ADAPTIVE_HOLDOUT_RUN_DIR:-$(<"$STATE_DIR/holdout_run.txt")}"
  python -m scripts.autodl.camera_hidden_state_attribution.export_adaptive_alpha \
    --source "$calibration_run" \
    --destination-root "$PUBLISH_ROOT/calibration"
  python -m scripts.autodl.camera_hidden_state_attribution.export_adaptive_alpha \
    --source "$holdout_run" \
    --destination-root "$PUBLISH_ROOT/holdout"
}

case "$STAGE" in
  calibration) run_calibration ;;
  holdout) run_holdout ;;
  export) run_export ;;
  all)
    run_calibration
    run_holdout
    run_export
    ;;
  *) echo "STAGE must be calibration, holdout, export, or all" >&2; exit 2 ;;
esac
