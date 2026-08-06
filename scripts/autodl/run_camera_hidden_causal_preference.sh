#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
AUTODL_TMP="${AUTODL_TMP:-/root/autodl-tmp}"
RESULTS_ROOT="${RESULTS_ROOT:-$AUTODL_TMP/results}"
SOURCE_RUN_DIR="${SOURCE_RUN_DIR:?set SOURCE_RUN_DIR to the global camera-context run}"
SPLIT_MANIFEST="${SPLIT_MANIFEST:?set SPLIT_MANIFEST to the frozen ScanNet-50 split}"
CKPT_DIR="${CKPT_DIR:-$AUTODL_TMP/ckpt/VGGT-1B}"
OUT_DIR="${OUT_DIR:-$RESULTS_ROOT/camera_hidden_causal_preference/results}"
PUBLISH_ROOT="${PUBLISH_ROOT:-$ROOT/results/camera_hidden_causal_preference}"
STATE_DIR="${STATE_DIR:-$RESULTS_ROOT/camera_hidden_causal_preference/state}"
STAGE="${STAGE:-all}"
DEVICE="${DEVICE:-cuda}"
BASIS_STEP="${BASIS_STEP:-0.001}"
BASIS_BATCH_SIZE="${BASIS_BATCH_SIZE:-2}"
DIRECT_CHECKS_PER_ITERATION="${DIRECT_CHECKS_PER_ITERATION:-1}"
DIRECT_RELATIVE_STEP="${DIRECT_RELATIVE_STEP:-0.01}"
REPLAY_TOLERANCE="${REPLAY_TOLERANCE:-0.005}"
SMOKE_BASIS_DIMENSIONS="${SMOKE_BASIS_DIMENSIONS:-2}"

mkdir -p "$STATE_DIR"
[[ -d "$SOURCE_RUN_DIR" ]] || {
  echo "missing SOURCE_RUN_DIR=$SOURCE_RUN_DIR" >&2
  exit 2
}
[[ -f "$SPLIT_MANIFEST" ]] || {
  echo "missing SPLIT_MANIFEST=$SPLIT_MANIFEST" >&2
  exit 2
}
[[ -f "$CKPT_DIR/model.safetensors" || -f "$CKPT_DIR/model.pt" ]] || {
  echo "missing VGGT checkpoint in CKPT_DIR=$CKPT_DIR" >&2
  exit 2
}

cd "$ROOT"

common_args=(
  --source-run-dir "$SOURCE_RUN_DIR"
  --split-manifest "$SPLIT_MANIFEST"
  --ckpt-dir "$CKPT_DIR"
  --out-dir "$OUT_DIR"
  --device "$DEVICE"
  --basis-step "$BASIS_STEP"
  --basis-batch-size "$BASIS_BATCH_SIZE"
  --direct-checks-per-iteration "$DIRECT_CHECKS_PER_ITERATION"
  --direct-relative-step "$DIRECT_RELATIVE_STEP"
  --replay-tolerance "$REPLAY_TOLERANCE"
)

run_smoke() {
  python -m pre_experiments.camera_hidden_state_attribution.run_causal_preference \
    --stage smoke \
    --basis-dimension-limit "$SMOKE_BASIS_DIMENSIONS" \
    --run-dir-file "$STATE_DIR/smoke_run.txt" \
    "${common_args[@]}"
}

run_calibration() {
  python -m pre_experiments.camera_hidden_state_attribution.run_causal_preference \
    --stage calibration \
    --run-dir-file "$STATE_DIR/calibration_run.txt" \
    "${common_args[@]}"
}

run_holdout() {
  local calibration_run
  if [[ -n "${CALIBRATION_RUN_DIR:-}" ]]; then
    calibration_run="$CALIBRATION_RUN_DIR"
  elif [[ -f "$STATE_DIR/calibration_run.txt" ]]; then
    calibration_run="$(<"$STATE_DIR/calibration_run.txt")"
  else
    echo "set CALIBRATION_RUN_DIR or run calibration first" >&2
    exit 2
  fi
  python -m pre_experiments.camera_hidden_state_attribution.run_causal_preference \
    --stage holdout \
    --frozen-normalization "$calibration_run/frozen_causal_normalization.json" \
    --run-dir-file "$STATE_DIR/holdout_run.txt" \
    "${common_args[@]}"
}

run_export() {
  local source
  if [[ -n "${HOLDOUT_RUN_DIR:-}" ]]; then
    source="$HOLDOUT_RUN_DIR"
  elif [[ -f "$STATE_DIR/holdout_run.txt" ]]; then
    source="$(<"$STATE_DIR/holdout_run.txt")"
  else
    echo "set HOLDOUT_RUN_DIR or run holdout first" >&2
    exit 2
  fi
  python -m scripts.autodl.camera_hidden_state_attribution.export_causal_preference \
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
  *)
    echo "STAGE must be smoke, calibration, holdout, export, or all" >&2
    exit 2
    ;;
esac
