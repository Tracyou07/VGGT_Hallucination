#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
AUTODL_TMP="${AUTODL_TMP:-/root/autodl-tmp}"
CONDA_ROOT="${CONDA_ROOT:-/root/miniconda3}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-vggt}"
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
SAVE_CAMERA_TOKENS="${SAVE_CAMERA_TOKENS:-0}"
CONDA_SH="$CONDA_ROOT/etc/profile.d/conda.sh"

[[ -f "$CONDA_SH" ]] || { printf 'Run scripts/autodl/setup_vggt_env.sh first.\n' >&2; exit 1; }
# shellcheck source=/dev/null
source "$CONDA_SH"
conda run -n "$CONDA_ENV_NAME" python -c "import torch, vggt" >/dev/null 2>&1 || {
  printf 'Run scripts/autodl/setup_vggt_env.sh first.\n' >&2
  exit 1
}
conda activate "$CONDA_ENV_NAME"

if ! python "$REPO_ROOT/scripts/autodl/camera_iteration/preflight.py" \
  --scannet-root "$SCANNET_ROOT" --ckpt-dir "$CKPT_DIR" \
  --scene-list "$SCENE_LIST" --scene-limit "$SCENE_LIMIT" --print-layout | grep -qx processed; then
  printf 'Inputs are not ready. Run download_vggt_weights.sh and prepare_scannet_camera_iteration.sh first.\n' >&2
  exit 1
fi

read -r -a frame_args <<< "$FRAME_COUNTS"
read -r -a iteration_args <<< "$ITERATIONS"
(( ${#frame_args[@]} > 0 && ${#iteration_args[@]} > 0 )) || { printf 'FRAME_COUNTS and ITERATIONS must not be empty.\n' >&2; exit 1; }

args=(python -m pre_experiments.camera_iteration.run_study
  --data-dir "$SCANNET_ROOT/process_scannet" --scene-list "$SCENE_LIST"
  --scene-limit "$SCENE_LIMIT" --frame-counts "${frame_args[@]}"
  --iterations "${iteration_args[@]}" --sampling "$SAMPLING"
  --ckpt-dir "$CKPT_DIR" --device cuda --preprocess-mode "$PREPROCESS_MODE"
  --out-dir "$RESULT_DIR" --seed "$SEED")
[[ "$SAVE_CAMERA_TOKENS" == "1" ]] && args+=(--save-camera-tokens)
cd "$REPO_ROOT"
"${args[@]}"
