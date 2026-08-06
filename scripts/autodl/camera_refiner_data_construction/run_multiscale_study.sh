#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
AUTODL_TMP="${AUTODL_TMP:-/root/autodl-tmp}"
RESULTS_ROOT="${RESULTS_ROOT:-$AUTODL_TMP/results}"
CONDA_ROOT="${CONDA_ROOT:-/root/miniconda3}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-vggt}"
CONDA_SH="$CONDA_ROOT/etc/profile.d/conda.sh"

STAGE="${1:-${STAGE:-smoke}}"
SOURCE_RUN_DIR="${SOURCE_RUN_DIR:?Set SOURCE_RUN_DIR to the frozen Camera Context run}"
FROZEN_UNITS="${FROZEN_UNITS:?Set FROZEN_UNITS to the calibrated hidden-unit manifest}"
FROZEN_POLICY="${FROZEN_POLICY:-}"
DATA_DIR="${DATA_DIR:-$AUTODL_TMP/datasets/scannetv2/process_scannet}"
SPLIT_MANIFEST="${SPLIT_MANIFEST:-$REPO_ROOT/configs/scannet50_local_global_split.json}"
CKPT_DIR="${CKPT_DIR:-$AUTODL_TMP/ckpt/VGGT-1B}"
WORK_ROOT="${WORK_ROOT:-$RESULTS_ROOT/camera_refiner_data_construction}"
DEVICE="${DEVICE:-cuda}"
CANDIDATE_FAMILY="${CANDIDATE_FAMILY:-pure}"

case "$STAGE" in
  smoke)
    partition="calibration"
    default_scene_limit=1
    ;;
  calibration)
    partition="calibration"
    default_scene_limit=0
    ;;
  holdout)
    partition="holdout"
    default_scene_limit=0
    [[ -n "$FROZEN_POLICY" ]] || {
      printf 'Set FROZEN_POLICY to the completed calibration policy for holdout.\n' >&2
      exit 1
    }
    ;;
  *)
    printf 'STAGE must be smoke, calibration, or holdout; got %s\n' "$STAGE" >&2
    exit 1
    ;;
esac

SCENE_LIMIT="${SCENE_LIMIT:-$default_scene_limit}"
[[ "$SCENE_LIMIT" =~ ^[0-9]+$ ]] || {
  printf 'SCENE_LIMIT must be a non-negative integer.\n' >&2
  exit 1
}
[[ "$CANDIDATE_FAMILY" == "pure" || "$CANDIDATE_FAMILY" == "all" ]] || {
  printf 'CANDIDATE_FAMILY must be pure or all.\n' >&2
  exit 1
}
[[ -f "$CONDA_SH" ]] || { printf 'Missing Conda activation: %s\n' "$CONDA_SH" >&2; exit 1; }
[[ -d "$SOURCE_RUN_DIR" ]] || { printf 'Missing source run: %s\n' "$SOURCE_RUN_DIR" >&2; exit 1; }
[[ -d "$DATA_DIR" ]] || { printf 'Missing processed ScanNet data: %s\n' "$DATA_DIR" >&2; exit 1; }
[[ -d "$CKPT_DIR" ]] || { printf 'Missing VGGT checkpoint: %s\n' "$CKPT_DIR" >&2; exit 1; }
[[ -f "$SPLIT_MANIFEST" ]] || { printf 'Missing split manifest: %s\n' "$SPLIT_MANIFEST" >&2; exit 1; }
[[ -f "$FROZEN_UNITS" ]] || { printf 'Missing frozen units: %s\n' "$FROZEN_UNITS" >&2; exit 1; }
if [[ "$STAGE" == "holdout" && ! -f "$FROZEN_POLICY" ]]; then
  printf 'Missing frozen candidate policy: %s\n' "$FROZEN_POLICY" >&2
  exit 1
fi

if [[ ! "${OMP_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then
  export OMP_NUM_THREADS=8
fi
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Reuse the provisioned AutoDL environment; this script never installs packages.
# shellcheck source=/dev/null
source "$CONDA_SH"
conda activate "$CONDA_ENV_NAME"
cd "$REPO_ROOT"
python -c "import numpy, torch, vggt"

POINTER_ROOT="$WORK_ROOT/pointers/$STAGE"
LOCAL_ROOT="$WORK_ROOT/local_global/$partition"
RESULT_ROOT="$WORK_ROOT/results/$STAGE"
mkdir -p "$POINTER_ROOT" "$LOCAL_ROOT" "$RESULT_ROOT"

SCALE_PAIRS=("100:50" "200:100" "300:150")
scale_args=()
for pair in "${SCALE_PAIRS[@]}"; do
  IFS=: read -r scale stride <<<"$pair"
  pointer="$POINTER_ROOT/local_${scale}_${stride}.txt"
  python -m pre_experiments.local_global_consistency.run_study \
    --data-dir "$DATA_DIR" \
    --source-run-dir "$SOURCE_RUN_DIR" \
    --split-manifest "$SPLIT_MANIFEST" \
    --partition "$partition" \
    --ckpt-dir "$CKPT_DIR" \
    --out-dir "$LOCAL_ROOT/scale_${scale}_${stride}" \
    --run-dir-file "$pointer" \
    --device "$DEVICE" \
    --scene-limit "$SCENE_LIMIT" \
    --window-length "$scale" \
    --window-stride "$stride" \
    --camera-iterations 4 \
    --preprocess-mode pad
  IFS= read -r scale_run_dir < "$pointer"
  [[ -d "$scale_run_dir" ]] || {
    printf 'Invalid scale run pointer %s: %s\n' "$pointer" "$scale_run_dir" >&2
    exit 1
  }
  scale_args+=(--scale-run "$scale=$scale_run_dir")
done

pointer="$POINTER_ROOT/multiscale.txt"
study_args=(
  --stage "$STAGE"
  --source-run-dir "$SOURCE_RUN_DIR"
  "${scale_args[@]}"
  --split-manifest "$SPLIT_MANIFEST"
  --ckpt-dir "$CKPT_DIR"
  --frozen-units "$FROZEN_UNITS"
  --out-dir "$RESULT_ROOT"
  --run-dir-file "$pointer"
  --device "$DEVICE"
  --scene-limit "$SCENE_LIMIT"
)
if [[ "$STAGE" == "holdout" ]]; then
  study_args+=(--frozen-policy "$FROZEN_POLICY")
else
  study_args+=(--candidate-family "$CANDIDATE_FAMILY")
fi
python -m pre_experiments.camera_refiner_data_construction.run_study "${study_args[@]}"

IFS= read -r completed_run < "$pointer"
printf '[done] stage=%s run=%s\n' "$STAGE" "$completed_run"
