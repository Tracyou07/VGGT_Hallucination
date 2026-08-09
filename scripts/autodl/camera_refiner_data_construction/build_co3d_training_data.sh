#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-full}"
[[ "$MODE" == "smoke" || "$MODE" == "full" ]] || {
  printf 'Usage: %s [smoke|full]\n' "$0" >&2
  exit 1
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
AUTODL_TMP="${AUTODL_TMP:-/root/autodl-tmp}"
CONDA_ROOT="${CONDA_ROOT:-/root/miniconda3}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-vggt}"
CONDA_SH="$CONDA_ROOT/etc/profile.d/conda.sh"

DATA_ROOT="${DATA_ROOT:-$AUTODL_TMP/datasets/co3dv2_2050}"
DOWNLOAD_MANIFEST="${DOWNLOAD_MANIFEST:-$DATA_ROOT/download_manifest.json}"
CKPT_DIR="${CKPT_DIR:-$AUTODL_TMP/ckpt/VGGT-1B}"
if [[ ! -d "$CKPT_DIR" && -d "$AUTODL_TMP/hf_home/hub/models--facebook--VGGT-1B" ]]; then
  CKPT_DIR="$AUTODL_TMP/hf_home/hub/models--facebook--VGGT-1B"
fi
RESULTS_ROOT="${RESULTS_ROOT:-$AUTODL_TMP/results}"

CLIP_LENGTH="${CLIP_LENGTH:-100}"
SHORT_WINDOW="${SHORT_WINDOW:-50}"
SHORT_STRIDE="${SHORT_STRIDE:-25}"
MAX_CLIPS_PER_SEQUENCE="${MAX_CLIPS_PER_SEQUENCE:-1}"
TEMPORAL_STRIDES="${TEMPORAL_STRIDES:-1 2}"
VALIDATION_FRACTION="${VALIDATION_FRACTION:-0.1}"
SELECTION_SEED="${SELECTION_SEED:-33}"
CAMERA_ITERATIONS="${CAMERA_ITERATIONS:-4}"
FEATURE_DTYPE="${FEATURE_DTYPE:-float16}"
PREPROCESS_MODE="${PREPROCESS_MODE:-pad}"
DEVICE="${DEVICE:-cuda}"
REBUILD_CLIP_MANIFEST="${REBUILD_CLIP_MANIFEST:-0}"

if [[ "$MODE" == "smoke" ]]; then
  SEQUENCE_LIMIT="${SEQUENCE_LIMIT:-1}"
  RUN_ID="${RUN_ID:-smoke_l${CLIP_LENGTH}_s${SHORT_WINDOW}_seed${SELECTION_SEED}}"
else
  SEQUENCE_LIMIT="${SEQUENCE_LIMIT:-0}"
  RUN_ID="${RUN_ID:-co3d_l${CLIP_LENGTH}_s${SHORT_WINDOW}_seed${SELECTION_SEED}}"
fi
OUT_DIR="${OUT_DIR:-$RESULTS_ROOT/camera_refiner_data_construction/co3d/$RUN_ID}"
CLIP_MANIFEST="${CLIP_MANIFEST:-$OUT_DIR/clip_manifest.json}"

[[ -f "$CONDA_SH" ]] || { printf 'Missing Conda activation: %s\n' "$CONDA_SH" >&2; exit 1; }
[[ -f "$DOWNLOAD_MANIFEST" ]] || {
  printf 'Missing completed CO3D download manifest: %s\n' "$DOWNLOAD_MANIFEST" >&2
  exit 1
}
[[ -d "$CKPT_DIR" ]] || { printf 'Missing local VGGT checkpoint: %s\n' "$CKPT_DIR" >&2; exit 1; }
command -v flock >/dev/null || { printf 'Missing required command: flock\n' >&2; exit 1; }
for value in "$CLIP_LENGTH" "$SHORT_WINDOW" "$SHORT_STRIDE" "$MAX_CLIPS_PER_SEQUENCE" "$CAMERA_ITERATIONS"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || { printf 'Window and iteration values must be positive integers.\n' >&2; exit 1; }
done
[[ "$SEQUENCE_LIMIT" =~ ^[0-9]+$ ]] || { printf 'SEQUENCE_LIMIT must be non-negative.\n' >&2; exit 1; }
[[ "$REBUILD_CLIP_MANIFEST" == "0" || "$REBUILD_CLIP_MANIFEST" == "1" ]] || {
  printf 'REBUILD_CLIP_MANIFEST must be 0 or 1.\n' >&2
  exit 1
}

if [[ ! "${OMP_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then
  export OMP_NUM_THREADS=8
fi

# Reuse the provisioned AutoDL environment and local checkpoint. This script
# performs no package installation, environment creation, or network download.
# shellcheck source=/dev/null
source "$CONDA_SH"
conda activate "$CONDA_ENV_NAME"
cd "$REPO_ROOT"

mkdir -p "$OUT_DIR"
exec 9>"$OUT_DIR/.build.lock"
flock --nonblock 9 || {
  printf 'Another CO3D cache build is using %s\n' "$OUT_DIR" >&2
  exit 1
}

if [[ ! -f "$CLIP_MANIFEST" || "$REBUILD_CLIP_MANIFEST" == "1" ]]; then
  # TEMPORAL_STRIDES is intentionally shell-split into integer CLI arguments.
  # shellcheck disable=SC2086
  python -m pre_experiments.camera_refiner_data_construction.co3d_manifest \
    --data-root "$DATA_ROOT" \
    --download-manifest "$DOWNLOAD_MANIFEST" \
    --output "$CLIP_MANIFEST" \
    --clip-length "$CLIP_LENGTH" \
    --max-clips-per-sequence "$MAX_CLIPS_PER_SEQUENCE" \
    --temporal-strides $TEMPORAL_STRIDES \
    --validation-fraction "$VALIDATION_FRACTION" \
    --seed "$SELECTION_SEED"
fi

python -m pre_experiments.camera_refiner_data_construction.build_co3d_cache \
  --data-root "$DATA_ROOT" \
  --clip-manifest "$CLIP_MANIFEST" \
  --ckpt-dir "$CKPT_DIR" \
  --out-dir "$OUT_DIR" \
  --short-window "$SHORT_WINDOW" \
  --short-stride "$SHORT_STRIDE" \
  --camera-iterations "$CAMERA_ITERATIONS" \
  --feature-dtype "$FEATURE_DTYPE" \
  --preprocess-mode "$PREPROCESS_MODE" \
  --device "$DEVICE" \
  --sequence-limit "$SEQUENCE_LIMIT"

mkdir -p "$RESULTS_ROOT/camera_refiner_data_construction/co3d"
printf '%s\n' "$OUT_DIR" > "$RESULTS_ROOT/camera_refiner_data_construction/co3d/latest_run.txt"
printf '[done] training manifest: %s/manifest.json\n' "$OUT_DIR"
