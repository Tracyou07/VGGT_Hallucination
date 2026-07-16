#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

AUTODL_TMP="${AUTODL_TMP:-/root/autodl-tmp}"
CONDA_ROOT="${CONDA_ROOT:-/root/miniconda3}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-vggt_camera_iteration}"
CONDA_CLONE_FROM="${CONDA_CLONE_FROM:-base}"
SCANNET_ROOT="${SCANNET_ROOT:-$AUTODL_TMP/datasets/scannetv2}"
CKPT_DIR="${CKPT_DIR:-$AUTODL_TMP/ckpt/VGGT-1B}"
RESULT_DIR="${RESULT_DIR:-$AUTODL_TMP/camera_iteration/results}"
SCENE_LIST="${SCENE_LIST:-$REPO_ROOT/configs/camera_iteration_scannet.txt}"
SCENE_LIMIT="${SCENE_LIMIT:-10}"
FRAME_COUNTS="${FRAME_COUNTS:-25 50 100 200 500}"
ITERATIONS="${ITERATIONS:-1 2 4 8 16}"
SAMPLING="${SAMPLING:-nested_uniform}"
PREPROCESS_MODE="${PREPROCESS_MODE:-pad}"
SEED="${SEED:-33}"
RUN_EXTRACT="${RUN_EXTRACT:-1}"
SAVE_CAMERA_TOKENS="${SAVE_CAMERA_TOKENS:-0}"

CONDA_SH="$CONDA_ROOT/etc/profile.d/conda.sh"
if [[ ! -f "$CONDA_SH" ]]; then
  printf 'Conda initialization script not found: %s\n' "$CONDA_SH" >&2
  exit 1
fi
# shellcheck source=/dev/null
source "$CONDA_SH"

if ! conda run -n "$CONDA_ENV_NAME" python -c "import sys" >/dev/null 2>&1; then
  conda create --name "$CONDA_ENV_NAME" --clone "$CONDA_CLONE_FROM" -y
fi
conda activate "$CONDA_ENV_NAME"

python -m pip install --no-deps --no-build-isolation -e "$REPO_ROOT"
mapfile -t missing_specs < <(
  python "$REPO_ROOT/scripts/autodl/camera_iteration/preflight.py" --print-missing
)
if (( ${#missing_specs[@]} > 0 )); then
  python -m pip install "${missing_specs[@]}"
fi

python -c "import torch; assert torch.cuda.is_available(), 'CUDA is required'; print('torch=' + torch.__version__ + ' cuda=' + str(torch.version.cuda) + ' device=' + torch.cuda.get_device_name(0))"

preflight=(
  python "$REPO_ROOT/scripts/autodl/camera_iteration/preflight.py"
  --scannet-root "$SCANNET_ROOT"
  --ckpt-dir "$CKPT_DIR"
  --scene-list "$SCENE_LIST"
  --scene-limit "$SCENE_LIMIT"
)
layout="$("${preflight[@]}" --print-layout)"

if [[ "$layout" == "raw" ]]; then
  if [[ "$RUN_EXTRACT" != "1" ]]; then
    printf 'Raw .sens data found, but RUN_EXTRACT=%s. Set RUN_EXTRACT=1.\n' "$RUN_EXTRACT" >&2
    exit 1
  fi
  python "$REPO_ROOT/scripts/autodl/camera_iteration/extract_scannet_sens.py" \
    --raw-dir "$SCANNET_ROOT/raw_sens/scans" \
    --out-dir "$SCANNET_ROOT/process_scannet" \
    --scene-list "$SCENE_LIST" \
    --scene-limit "$SCENE_LIMIT"
  layout="$("${preflight[@]}" --print-layout)"
fi

if [[ "$layout" != "processed" ]]; then
  printf 'Expected processed ScanNet data after preflight, got: %s\n' "$layout" >&2
  exit 1
fi

read -r -a frame_args <<< "$FRAME_COUNTS"
read -r -a iteration_args <<< "$ITERATIONS"
if (( ${#frame_args[@]} == 0 || ${#iteration_args[@]} == 0 )); then
  printf 'FRAME_COUNTS and ITERATIONS must not be empty.\n' >&2
  exit 1
fi

study_args=(
  python -m pre_experiments.camera_iteration.run_study
  --data-dir "$SCANNET_ROOT/process_scannet"
  --scene-list "$SCENE_LIST"
  --scene-limit "$SCENE_LIMIT"
  --frame-counts "${frame_args[@]}"
  --iterations "${iteration_args[@]}"
  --sampling "$SAMPLING"
  --ckpt-dir "$CKPT_DIR"
  --device cuda
  --preprocess-mode "$PREPROCESS_MODE"
  --out-dir "$RESULT_DIR"
  --seed "$SEED"
)
if [[ "$SAVE_CAMERA_TOKENS" == "1" ]]; then
  study_args+=(--save-camera-tokens)
fi

cd "$REPO_ROOT"
"${study_args[@]}"
