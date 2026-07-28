#!/usr/bin/env bash
set -euo pipefail

if [[ "${SCANNET_TOS_ACCEPTED:-0}" != "1" ]]; then
  printf 'Set SCANNET_TOS_ACCEPTED=1 only after accepting the official ScanNet terms.\n' >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
AUTODL_TMP="${AUTODL_TMP:-/root/autodl-tmp}"
CONDA_ROOT="${CONDA_ROOT:-/root/miniconda3}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-vggt}"
SCANNET_ROOT="${SCANNET_ROOT:-$AUTODL_TMP/datasets/scannetv2}"
RAW_DOWNLOAD_ROOT="${RAW_DOWNLOAD_ROOT:-$SCANNET_ROOT/raw_sens}"
GT_DOWNLOAD_ROOT="${GT_DOWNLOAD_ROOT:-$SCANNET_ROOT/raw}"
RAW_DIR="${RAW_DIR:-$RAW_DOWNLOAD_ROOT/scans}"
if [[ "$RAW_DOWNLOAD_ROOT" != "/" ]]; then
  RAW_DOWNLOAD_ROOT="${RAW_DOWNLOAD_ROOT%/}"
fi
if [[ "$RAW_DIR" != "/" ]]; then
  RAW_DIR="${RAW_DIR%/}"
fi
PROCESS_DIR="${PROCESS_DIR:-$SCANNET_ROOT/process_scannet}"
SCENE_LIST="${SCENE_LIST:-$REPO_ROOT/configs/fastvggt_scannet50.txt}"
SCENE_LIMIT="${SCENE_LIMIT:-0}"
DOWNLOAD_RETRIES="${DOWNLOAD_RETRIES:-5}"
DOWNLOAD_GT_PLY="${DOWNLOAD_GT_PLY:-0}"
SCANNET_DOWNLOAD_SCRIPT="${SCANNET_DOWNLOAD_SCRIPT:-$SCANNET_ROOT/tools/download-scannet.py}"
SCANNET_DOWNLOAD_URL="${SCANNET_DOWNLOAD_URL:-http://kaldir.vc.in.tum.de/scannet/download-scannet.py}"
CONDA_SH="$CONDA_ROOT/etc/profile.d/conda.sh"
# shellcheck source=scannet_download.sh
source "$SCRIPT_DIR/scannet_download.sh"

[[ "$DOWNLOAD_RETRIES" =~ ^[1-9][0-9]*$ ]] || {
  printf 'DOWNLOAD_RETRIES must be a positive integer.\n' >&2
  exit 1
}
[[ "$DOWNLOAD_GT_PLY" == "0" || "$DOWNLOAD_GT_PLY" == "1" ]] || {
  printf 'DOWNLOAD_GT_PLY must be 0 or 1.\n' >&2
  exit 1
}

[[ -f "$CONDA_SH" ]] || { printf 'Missing Conda activation script: %s\n' "$CONDA_SH" >&2; exit 1; }
# shellcheck source=/dev/null
source "$CONDA_SH"
conda run -n "$CONDA_ENV_NAME" python -c "import imageio" >/dev/null 2>&1 || {
  printf 'Conda environment %s is unavailable or missing imageio.\n' "$CONDA_ENV_NAME" >&2
  exit 1
}
conda activate "$CONDA_ENV_NAME"
cd "$REPO_ROOT"

mapfile -t scenes < <(python - "$SCENE_LIST" "$SCENE_LIMIT" <<'PY'
from pathlib import Path
import sys
from scripts.autodl.scannet.preflight import read_scene_list
for scene in read_scene_list(Path(sys.argv[1]), int(sys.argv[2])):
    print(scene)
PY
)

python - "${scenes[@]}" <<'PY'
import re
import sys

scenes = sys.argv[1:]
if not scenes:
    raise SystemExit("Selected scene list is empty")
if len(scenes) != len(set(scenes)):
    raise SystemExit("Selected scene list contains duplicate IDs")
for scene in scenes:
    if not re.fullmatch(r"scene[0-9]{4}_[0-9]{2}", scene):
        raise SystemExit(f"Invalid ScanNet scene ID: {scene}")
PY

mkdir -p "$(dirname "$SCANNET_DOWNLOAD_SCRIPT")" "$RAW_DIR" "$PROCESS_DIR"
if [[ ! -s "$SCANNET_DOWNLOAD_SCRIPT" ]]; then
  if command -v curl >/dev/null 2>&1; then
    curl -fL --retry 5 "$SCANNET_DOWNLOAD_URL" -o "$SCANNET_DOWNLOAD_SCRIPT"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$SCANNET_DOWNLOAD_SCRIPT" "$SCANNET_DOWNLOAD_URL"
  else
    printf 'curl or wget is required to retrieve the official ScanNet downloader.\n' >&2
    exit 1
  fi
fi

for scene in "${scenes[@]}"; do
  sens="$RAW_DIR/$scene/$scene.sens"
  download_asset "$scene" .sens "$sens"
  if [[ "$DOWNLOAD_GT_PLY" == "1" ]]; then
    gt_ply="$GT_DOWNLOAD_ROOT/scans/$scene/${scene}_vh_clean_2.ply"
    download_asset "$scene" _vh_clean_2.ply "$gt_ply"
  fi
done

python "$REPO_ROOT/scripts/autodl/scannet/extract_scannet_sens.py" \
  --raw-dir "$RAW_DIR" --out-dir "$PROCESS_DIR" \
  --scene-list "$SCENE_LIST" --scene-limit "$SCENE_LIMIT"

python - "$SCANNET_ROOT" "$SCENE_LIST" "$SCENE_LIMIT" <<'PY'
from pathlib import Path
import sys
from scripts.autodl.scannet.preflight import missing_processed_scenes, read_scene_list
root, scene_list, limit = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
missing = missing_processed_scenes(root, read_scene_list(scene_list, limit))
if missing:
    raise SystemExit(f"Incomplete processed scenes: {missing}")
print("ScanNet camera-iteration data ready")
PY
