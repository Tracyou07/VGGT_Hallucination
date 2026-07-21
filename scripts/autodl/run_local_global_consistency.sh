#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
AUTODL_TMP="${AUTODL_TMP:-/root/autodl-tmp}"
CONDA_ROOT="${CONDA_ROOT:-/root/miniconda3}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-vggt}"
DATA_DIR="${DATA_DIR:-$AUTODL_TMP/datasets/scannetv2/process_scannet}"
SOURCE_RUN_DIR="${SOURCE_RUN_DIR:-$REPO_ROOT/results/camera_context/911b598_f4577f584448}"
CKPT_DIR="${CKPT_DIR:-$AUTODL_TMP/ckpt/VGGT-1B}"
RESULT_DIR="${RESULT_DIR:-$AUTODL_TMP/local_global_consistency/results}"
DEVICE="${DEVICE:-cuda}"
SCENE_LIMIT="${SCENE_LIMIT:-4}"
WINDOW_LENGTH="${WINDOW_LENGTH:-100}"
WINDOW_STRIDE="${WINDOW_STRIDE:-50}"
CAMERA_ITERATIONS="4"
PREPROCESS_MODE="${PREPROCESS_MODE:-pad}"

[[ -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]] || {
  printf 'Missing conda activation script: %s\n' "$CONDA_ROOT/etc/profile.d/conda.sh" >&2
  exit 1
}
[[ -d "$DATA_DIR" ]] || { printf 'Missing processed ScanNet: %s\n' "$DATA_DIR" >&2; exit 1; }
[[ -f "$SOURCE_RUN_DIR/run_metadata.json" ]] || {
  printf 'Missing published Round 1.5 global run: %s\n' "$SOURCE_RUN_DIR" >&2
  exit 1
}
[[ -f "$CKPT_DIR/model.safetensors" || -f "$CKPT_DIR/model.pt" ]] || {
  printf 'Missing model.safetensors or model.pt under %s\n' "$CKPT_DIR" >&2
  exit 1
}

# shellcheck source=/dev/null
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME"
cd "$REPO_ROOT"
RUN_DIR_FILE="$(mktemp)"
trap 'rm -f "$RUN_DIR_FILE"' EXIT

python -m pre_experiments.local_global_consistency.run_study \
  --data-dir "$DATA_DIR" \
  --source-run-dir "$SOURCE_RUN_DIR" \
  --ckpt-dir "$CKPT_DIR" \
  --out-dir "$RESULT_DIR" \
  --run-dir-file "$RUN_DIR_FILE" \
  --device "$DEVICE" \
  --scene-limit "$SCENE_LIMIT" \
  --window-length "$WINDOW_LENGTH" \
  --window-stride "$WINDOW_STRIDE" \
  --camera-iterations "$CAMERA_ITERATIONS" \
  --preprocess-mode "$PREPROCESS_MODE"

run_dir="$(<"$RUN_DIR_FILE")"
[[ -n "$run_dir" && -f "$run_dir/run_metadata.json" ]] || {
  printf 'Runner did not report a valid run directory: %s\n' "$run_dir" >&2
  exit 1
}
python -m pre_experiments.local_global_consistency.analyze --run-dir "$run_dir"
