#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
AUTODL_TMP="${AUTODL_TMP:-/root/autodl-tmp}"
CONDA_ROOT="${CONDA_ROOT:-/root/miniconda3}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-vggt}"
DATA_DIR="${DATA_DIR:-$AUTODL_TMP/datasets/scannetv2/process_scannet}"
CKPT_DIR="${CKPT_DIR:-$AUTODL_TMP/ckpt/VGGT-1B}"
SOURCE_RUN_DIR="${SOURCE_RUN_DIR:-}"
SPLIT_MANIFEST="${SPLIT_MANIFEST:-$REPO_ROOT/configs/scannet50_local_global_split.json}"
RESULT_ROOT="${RESULT_ROOT:-$AUTODL_TMP/local_global_consistency/scannet50}"
STAGE="${STAGE:-all}"
SCENE_LIMIT="${SCENE_LIMIT:-0}"
CALIBRATION_RUN_DIR="${CALIBRATION_RUN_DIR:-}"
WINDOW_LENGTH="100"
WINDOW_STRIDE="50"
CAMERA_ITERATIONS="4"
PREPROCESS_MODE="pad"
DEVICE="cuda"

[[ -n "$SOURCE_RUN_DIR" ]] || {
  printf 'SOURCE_RUN_DIR must be set explicitly.\n' >&2
  exit 1
}
[[ -f "$SOURCE_RUN_DIR/run_metadata.json" ]] || {
  printf 'Missing Camera Context source run: %s\n' "$SOURCE_RUN_DIR" >&2
  exit 1
}
[[ -f "$SPLIT_MANIFEST" ]] || {
  printf 'Missing frozen split manifest: %s\n' "$SPLIT_MANIFEST" >&2
  exit 1
}
[[ -d "$DATA_DIR" ]] || {
  printf 'Missing processed ScanNet directory: %s\n' "$DATA_DIR" >&2
  exit 1
}
[[ -f "$CKPT_DIR/model.safetensors" || -f "$CKPT_DIR/model.pt" ]] || {
  printf 'Missing model.safetensors or model.pt under %s\n' "$CKPT_DIR" >&2
  exit 1
}
[[ -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]] || {
  printf 'Missing Conda activation script: %s\n' \
    "$CONDA_ROOT/etc/profile.d/conda.sh" >&2
  exit 1
}
[[ "$STAGE" == "calibration" || "$STAGE" == "holdout" || "$STAGE" == "all" ]] || {
  printf 'STAGE must be calibration, holdout, or all.\n' >&2
  exit 1
}
[[ "$SCENE_LIMIT" =~ ^[0-9]+$ ]] || {
  printf 'SCENE_LIMIT must be a non-negative integer.\n' >&2
  exit 1
}

mkdir -p \
  "$RESULT_ROOT/runs/calibration" \
  "$RESULT_ROOT/runs/holdout" \
  "$RESULT_ROOT/pointers" \
  "$RESULT_ROOT/logs"
LOG_FILE="$RESULT_ROOT/logs/scannet50_local_global.log"
exec > >(tee -a "$LOG_FILE") 2>&1

# shellcheck source=/dev/null
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME"
cd "$REPO_ROOT"

if [[ ! "${OMP_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then
  export OMP_NUM_THREADS=8
fi

TEMP_POINTERS=()
cleanup() {
  local path
  for path in "${TEMP_POINTERS[@]}"; do
    rm -f "$path"
  done
}
trap cleanup EXIT

write_pointer() {
  local destination="$1"
  local value="$2"
  local temporary="${destination}.tmp.$$"
  printf '%s\n' "$value" > "$temporary"
  mv "$temporary" "$destination"
}

LAST_RUN_DIR=""
run_partition() {
  local partition="$1"
  local output_root="$2"
  local pointer_file="$3"
  local temporary_pointer
  temporary_pointer="$(mktemp "$RESULT_ROOT/pointers/.${partition}.XXXXXX")"
  TEMP_POINTERS+=("$temporary_pointer")

  python -m pre_experiments.local_global_consistency.run_study \
    --data-dir "$DATA_DIR" \
    --source-run-dir "$SOURCE_RUN_DIR" \
    --split-manifest "$SPLIT_MANIFEST" \
    --partition "$partition" \
    --ckpt-dir "$CKPT_DIR" \
    --out-dir "$output_root" \
    --run-dir-file "$temporary_pointer" \
    --device "$DEVICE" \
    --scene-limit "$SCENE_LIMIT" \
    --window-length "$WINDOW_LENGTH" \
    --window-stride "$WINDOW_STRIDE" \
    --camera-iterations "$CAMERA_ITERATIONS" \
    --preprocess-mode "$PREPROCESS_MODE"

  LAST_RUN_DIR="$(tr -d '\r\n' < "$temporary_pointer")"
  [[ -n "$LAST_RUN_DIR" && -f "$LAST_RUN_DIR/run_metadata.json" ]] || {
    printf '%s runner did not report a valid run directory.\n' "$partition" >&2
    exit 1
  }
  write_pointer "$pointer_file" "$LAST_RUN_DIR"
}

run_calibration() {
  run_partition \
    calibration \
    "$RESULT_ROOT/runs/calibration" \
    "$RESULT_ROOT/pointers/calibration_run_dir.txt"
  CALIBRATION_RUN_DIR="$LAST_RUN_DIR"
  if [[ "$SCENE_LIMIT" != "0" ]]; then
    printf '[smoke] calibration inference only: %s\n' "$CALIBRATION_RUN_DIR"
    return
  fi

  python -m pre_experiments.local_global_consistency.analyze \
    --run-dir "$CALIBRATION_RUN_DIR" \
    --mode calibration
  python -m pre_experiments.local_global_consistency.visualize \
    --run-dir "$CALIBRATION_RUN_DIR" \
    --mode calibration \
    --split-manifest "$SPLIT_MANIFEST"
  local threshold_path
  threshold_path="$CALIBRATION_RUN_DIR/frozen_reliability_thresholds.json"
  [[ -f "$threshold_path" ]] || {
    printf 'Calibration did not produce frozen thresholds: %s\n' \
      "$threshold_path" >&2
    exit 1
  }
  write_pointer \
    "$RESULT_ROOT/pointers/frozen_threshold_path.txt" \
    "$threshold_path"
}

run_holdout() {
  local threshold_path="${1:-}"
  if [[ "$SCENE_LIMIT" == "0" ]]; then
    [[ -f "$threshold_path" ]] || {
      printf 'Missing frozen threshold artifact: %s\n' "$threshold_path" >&2
      exit 1
    }
  fi
  run_partition \
    holdout \
    "$RESULT_ROOT/runs/holdout" \
    "$RESULT_ROOT/pointers/holdout_run_dir.txt"
  local holdout_run_dir="$LAST_RUN_DIR"
  if [[ "$SCENE_LIMIT" != "0" ]]; then
    printf '[smoke] holdout inference only: %s\n' "$holdout_run_dir"
    return
  fi

  python -m pre_experiments.local_global_consistency.analyze \
    --run-dir "$holdout_run_dir" \
    --mode holdout \
    --thresholds "$threshold_path"
  python -m pre_experiments.local_global_consistency.visualize \
    --run-dir "$holdout_run_dir" \
    --mode holdout \
    --split-manifest "$SPLIT_MANIFEST"
}

case "$STAGE" in
  calibration)
    run_calibration
    ;;
  holdout)
    [[ -n "$CALIBRATION_RUN_DIR" ]] || {
      printf 'CALIBRATION_RUN_DIR is required when STAGE=holdout.\n' >&2
      exit 1
    }
    run_holdout \
      "$CALIBRATION_RUN_DIR/frozen_reliability_thresholds.json"
    ;;
  all)
    run_calibration
    if [[ "$SCENE_LIMIT" == "0" ]]; then
      run_holdout \
        "$CALIBRATION_RUN_DIR/frozen_reliability_thresholds.json"
    else
      run_holdout
    fi
    ;;
esac

printf '[done] stage=%s result_root=%s\n' "$STAGE" "$RESULT_ROOT"
