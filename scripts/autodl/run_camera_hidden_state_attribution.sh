#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
AUTODL_TMP="${AUTODL_TMP:-/root/autodl-tmp}"
RESULTS_ROOT="${RESULTS_ROOT:-$AUTODL_TMP/results}"
SOURCE_RUN_DIR="${SOURCE_RUN_DIR:?set SOURCE_RUN_DIR to the global camera-context run}"
CALIBRATION_LOCAL_RUN_DIR="${CALIBRATION_LOCAL_RUN_DIR:?set CALIBRATION_LOCAL_RUN_DIR}"
HOLDOUT_LOCAL_RUN_DIR="${HOLDOUT_LOCAL_RUN_DIR:?set HOLDOUT_LOCAL_RUN_DIR}"
SPLIT_MANIFEST="${SPLIT_MANIFEST:?set SPLIT_MANIFEST}"
CKPT_DIR="${CKPT_DIR:-$AUTODL_TMP/ckpt/VGGT-1B}"
OUT_DIR="${OUT_DIR:-$RESULTS_ROOT/camera_hidden_state_attribution/results}"
PUBLISH_ROOT="${PUBLISH_ROOT:-$ROOT/results/camera_hidden_state_attribution}"
STATE_DIR="${STATE_DIR:-$RESULTS_ROOT/camera_hidden_state_attribution/state}"
STAGE="${STAGE:-all}"
DEVICE="${DEVICE:-cuda}"

mkdir -p "$STATE_DIR"
[[ -d "$SOURCE_RUN_DIR" ]] || { echo "missing SOURCE_RUN_DIR=$SOURCE_RUN_DIR" >&2; exit 2; }
[[ -d "$CALIBRATION_LOCAL_RUN_DIR" ]] || { echo "missing CALIBRATION_LOCAL_RUN_DIR=$CALIBRATION_LOCAL_RUN_DIR" >&2; exit 2; }
[[ -d "$HOLDOUT_LOCAL_RUN_DIR" ]] || { echo "missing HOLDOUT_LOCAL_RUN_DIR=$HOLDOUT_LOCAL_RUN_DIR" >&2; exit 2; }
[[ -f "$SPLIT_MANIFEST" ]] || { echo "missing SPLIT_MANIFEST=$SPLIT_MANIFEST" >&2; exit 2; }
[[ -f "$CKPT_DIR/model.safetensors" || -f "$CKPT_DIR/model.pt" ]] || {
  echo "missing VGGT checkpoint in CKPT_DIR=$CKPT_DIR" >&2
  exit 2
}

common_args=(
  --source-run-dir "$SOURCE_RUN_DIR"
  --split-manifest "$SPLIT_MANIFEST"
  --ckpt-dir "$CKPT_DIR"
  --out-dir "$OUT_DIR"
  --device "$DEVICE"
)

run_smoke() {
  python -m pre_experiments.camera_hidden_state_attribution.run_study \
    --stage smoke \
    --local-run-dir "$CALIBRATION_LOCAL_RUN_DIR" \
    --run-dir-file "$STATE_DIR/smoke_run.txt" \
    "${common_args[@]}"
}

run_calibration() {
  python -m pre_experiments.camera_hidden_state_attribution.run_study \
    --stage calibration \
    --local-run-dir "$CALIBRATION_LOCAL_RUN_DIR" \
    --run-dir-file "$STATE_DIR/calibration_run.txt" \
    "${common_args[@]}"
}

run_holdout() {
  local calibration_run
  calibration_run="${CALIBRATION_RUN_DIR:-$(<"$STATE_DIR/calibration_run.txt")}"
  python -m pre_experiments.camera_hidden_state_attribution.run_study \
    --stage holdout \
    --local-run-dir "$HOLDOUT_LOCAL_RUN_DIR" \
    --frozen-units "$calibration_run/frozen_units.json" \
    --run-dir-file "$STATE_DIR/holdout_run.txt" \
    "${common_args[@]}"
}

run_export() {
  local source
  source="${HOLDOUT_RUN_DIR:-$(<"$STATE_DIR/holdout_run.txt")}"
  python "$ROOT/scripts/autodl/camera_hidden_state_attribution/export_numeric_results.py" \
    --source "$source" \
    --destination-root "$PUBLISH_ROOT"
}

case "$STAGE" in
  smoke) run_smoke ;;
  calibration) run_calibration ;;
  holdout) run_holdout ;;
  export) run_export ;;
  all)
    run_smoke
    run_calibration
    run_holdout
    run_export
    ;;
  *) echo "STAGE must be smoke, calibration, holdout, export, or all" >&2; exit 2 ;;
esac
