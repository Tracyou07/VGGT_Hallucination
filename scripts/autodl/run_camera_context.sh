#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
AUTODL_TMP="${AUTODL_TMP:-/root/autodl-tmp}"
CONDA_ROOT="${CONDA_ROOT:-/root/miniconda3}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-vggt}"
RESULT_DIR="${RESULT_DIR:-$AUTODL_TMP/camera_context/results}"
SCENE_LIST="${SCENE_LIST:-$REPO_ROOT/configs/camera_context_scannet.txt}"
SCENE_LIMIT="${SCENE_LIMIT:-4}"
FRAME_COUNTS="${FRAME_COUNTS:-25 50 100 200 500}"
ITERATIONS="4"
SAVE_CONTEXT_DIAGNOSTICS="1"
SAVE_CAMERA_TOKENS="0"

export AUTODL_TMP CONDA_ROOT CONDA_ENV_NAME RESULT_DIR SCENE_LIST SCENE_LIMIT
export FRAME_COUNTS ITERATIONS SAVE_CONTEXT_DIAGNOSTICS
export SAVE_CAMERA_TOKENS

bash "$SCRIPT_DIR/run_camera_iteration.sh"

run_dir="$({
  find "$RESULT_DIR" -mindepth 2 -maxdepth 2 -name run_metadata.json \
    -printf '%T@ %h\n' 2>/dev/null || true
} | sort -nr | head -n 1 | cut -d' ' -f2-)"
[[ -n "$run_dir" ]] || { printf 'No completed context run found in %s\n' "$RESULT_DIR" >&2; exit 1; }

# shellcheck source=/dev/null
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME"
cd "$REPO_ROOT"
python -m pre_experiments.camera_context.analyze --run-dir "$run_dir"
