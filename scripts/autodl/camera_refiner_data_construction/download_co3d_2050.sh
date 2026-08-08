#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
AUTODL_TMP="${AUTODL_TMP:-/root/autodl-tmp}"
CONDA_ROOT="${CONDA_ROOT:-/root/miniconda3}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-vggt}"
CONDA_SH="$CONDA_ROOT/etc/profile.d/conda.sh"

OUTPUT_ROOT="${OUTPUT_ROOT:-$AUTODL_TMP/datasets/co3dv2_2050}"
CATEGORY_FILE="${CATEGORY_FILE:-$REPO_ROOT/configs/co3d_train41.txt}"
SEQUENCES_PER_CATEGORY="${SEQUENCES_PER_CATEGORY:-50}"
MIN_FRAMES="${MIN_FRAMES:-50}"
MIN_QUALITY="${MIN_QUALITY:-0.5}"
SELECTION_SEED="${SELECTION_SEED:-33}"
CATEGORY_LIMIT="${CATEGORY_LIMIT:-0}"
MAX_DATA_ARCHIVES="${MAX_DATA_ARCHIVES:-0}"
KEEP_ARCHIVES="${KEEP_ARCHIVES:-0}"
CO3D_BASE_URL="${CO3D_BASE_URL:-https://dl.fbaipublicfiles.com/co3dv2_231130}"
CURL_BIN="${CURL_BIN:-curl}"

[[ -f "$CONDA_SH" ]] || { printf 'Missing Conda activation: %s\n' "$CONDA_SH" >&2; exit 1; }
[[ -f "$CATEGORY_FILE" ]] || { printf 'Missing category list: %s\n' "$CATEGORY_FILE" >&2; exit 1; }
command -v "$CURL_BIN" >/dev/null || { printf 'Missing curl executable: %s\n' "$CURL_BIN" >&2; exit 1; }
command -v flock >/dev/null || { printf 'Missing required command: flock\n' >&2; exit 1; }
for value in "$SEQUENCES_PER_CATEGORY" "$MIN_FRAMES"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || { printf 'Sequence and frame quotas must be positive integers.\n' >&2; exit 1; }
done
for value in "$CATEGORY_LIMIT" "$MAX_DATA_ARCHIVES"; do
  [[ "$value" =~ ^[0-9]+$ ]] || { printf 'Limits must be non-negative integers.\n' >&2; exit 1; }
done
[[ "$KEEP_ARCHIVES" == "0" || "$KEEP_ARCHIVES" == "1" ]] || {
  printf 'KEEP_ARCHIVES must be 0 or 1.\n' >&2
  exit 1
}

if [[ ! "${OMP_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then
  export OMP_NUM_THREADS=8
fi

# Reuse the provisioned AutoDL environment; no environment or checkpoint setup occurs here.
# shellcheck source=/dev/null
source "$CONDA_SH"
conda activate "$CONDA_ENV_NAME"
cd "$REPO_ROOT"

mkdir -p "$OUTPUT_ROOT"
LOCK_FILE="$OUTPUT_ROOT/.download.lock"
exec 9>"$LOCK_FILE"
flock --nonblock 9 || {
  printf 'Another CO3D download is already using %s\n' "$OUTPUT_ROOT" >&2
  exit 1
}
df -h "$AUTODL_TMP"

args=(
  --output-root "$OUTPUT_ROOT"
  --category-file "$CATEGORY_FILE"
  --sequences-per-category "$SEQUENCES_PER_CATEGORY"
  --min-frames "$MIN_FRAMES"
  --min-quality "$MIN_QUALITY"
  --seed "$SELECTION_SEED"
  --category-limit "$CATEGORY_LIMIT"
  --max-data-archives "$MAX_DATA_ARCHIVES"
  --base-url "$CO3D_BASE_URL"
  --curl-bin "$CURL_BIN"
)
if [[ "$KEEP_ARCHIVES" == "1" ]]; then
  args+=(--keep-archives)
fi

python -m pre_experiments.camera_refiner_data_construction.co3d_download "${args[@]}"
