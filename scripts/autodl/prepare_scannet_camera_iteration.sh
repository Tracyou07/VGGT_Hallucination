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
RAW_DOWNLOAD_ROOT_WAS_SET="${RAW_DOWNLOAD_ROOT+x}"
RAW_DIR_WAS_SET="${RAW_DIR+x}"
RAW_DOWNLOAD_ROOT="${RAW_DOWNLOAD_ROOT:-$SCANNET_ROOT/raw_sens}"
GT_DOWNLOAD_ROOT="${GT_DOWNLOAD_ROOT:-$SCANNET_ROOT/raw}"
RAW_DIR="${RAW_DIR:-$RAW_DOWNLOAD_ROOT/scans}"
if [[ "$RAW_DOWNLOAD_ROOT" != "/" ]]; then
  RAW_DOWNLOAD_ROOT="${RAW_DOWNLOAD_ROOT%/}"
fi
RAW_DIR="${RAW_DIR%/}"
[[ "$(basename "$RAW_DIR")" == "scans" ]] || {
  printf 'RAW_DIR must name a scans directory: %s\n' "$RAW_DIR" >&2
  exit 1
}
RAW_DIR_DOWNLOAD_ROOT="$(dirname "$RAW_DIR")"
if [[ "$RAW_DOWNLOAD_ROOT_WAS_SET" == "x" && "$RAW_DIR_WAS_SET" == "x" \
  && "$RAW_DOWNLOAD_ROOT" != "$RAW_DIR_DOWNLOAD_ROOT" ]]; then
  printf 'RAW_DOWNLOAD_ROOT and RAW_DIR must identify the same scans root.\n' >&2
  exit 1
fi
RAW_DOWNLOAD_ROOT="$RAW_DIR_DOWNLOAD_ROOT"
PROCESS_DIR="${PROCESS_DIR:-$SCANNET_ROOT/process_scannet}"
SCENE_LIST="${SCENE_LIST:-$REPO_ROOT/configs/camera_iteration_scannet.txt}"
SCENE_LIMIT="${SCENE_LIMIT:-10}"
DOWNLOAD_RETRIES="${DOWNLOAD_RETRIES:-5}"
DOWNLOAD_GT_PLY="${DOWNLOAD_GT_PLY:-0}"
SCANNET_DOWNLOAD_SCRIPT="${SCANNET_DOWNLOAD_SCRIPT:-$SCANNET_ROOT/tools/download-scannet.py}"
SCANNET_DOWNLOAD_URL="${SCANNET_DOWNLOAD_URL:-http://kaldir.vc.in.tum.de/scannet/download-scannet.py}"
CONDA_SH="$CONDA_ROOT/etc/profile.d/conda.sh"

[[ "$DOWNLOAD_RETRIES" =~ ^[1-9][0-9]*$ ]] || {
  printf 'DOWNLOAD_RETRIES must be a positive integer.\n' >&2
  exit 1
}
[[ "$DOWNLOAD_GT_PLY" == "0" || "$DOWNLOAD_GT_PLY" == "1" ]] || {
  printf 'DOWNLOAD_GT_PLY must be 0 or 1.\n' >&2
  exit 1
}

[[ -f "$CONDA_SH" ]] || { printf 'Run setup_vggt_env.sh first.\n' >&2; exit 1; }
# shellcheck source=/dev/null
source "$CONDA_SH"
conda run -n "$CONDA_ENV_NAME" python -c "import imageio" >/dev/null 2>&1 || {
  printf 'Run setup_vggt_env.sh first; environment %s is unavailable.\n' "$CONDA_ENV_NAME" >&2
  exit 1
}
conda activate "$CONDA_ENV_NAME"
cd "$REPO_ROOT"

mapfile -t scenes < <(python - "$SCENE_LIST" "$SCENE_LIMIT" <<'PY'
from pathlib import Path
import sys
from scripts.autodl.camera_iteration.preflight import read_scene_list
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

download_asset() {
  local scene="$1"
  local file_type="$2"
  local download_root="$3"
  local expected="$4"
  local attempt

  if [[ -s "$expected" ]]; then
    printf '[scannet] reuse %s\n' "$expected"
    return 0
  fi

  mkdir -p "$(dirname "$expected")"
  for ((attempt = 1; attempt <= DOWNLOAD_RETRIES; attempt++)); do
    rm -f "${expected}.tmp"
    if printf '\n\n\n\n' | python "$SCANNET_DOWNLOAD_SCRIPT" \
      -o "$download_root" --id "$scene" --type "$file_type"; then
      if [[ -s "$expected" ]]; then
        return 0
      fi
    fi

    if [[ -e "$expected" && ! -s "$expected" ]]; then
      rm -f "$expected"
    fi
    rm -f "${expected}.tmp"
    printf '[scannet] attempt %s/%s failed for %s\n' \
      "$attempt" "$DOWNLOAD_RETRIES" "$expected" >&2
  done

  printf 'Official downloader did not produce %s after %s attempts.\n' \
    "$expected" "$DOWNLOAD_RETRIES" >&2
  return 1
}

for scene in "${scenes[@]}"; do
  sens="$RAW_DOWNLOAD_ROOT/scans/$scene/$scene.sens"
  download_asset "$scene" .sens "$RAW_DOWNLOAD_ROOT" "$sens"
  if [[ "$DOWNLOAD_GT_PLY" == "1" ]]; then
    gt_ply="$GT_DOWNLOAD_ROOT/scans/$scene/${scene}_vh_clean_2.ply"
    download_asset "$scene" _vh_clean_2.ply "$GT_DOWNLOAD_ROOT" "$gt_ply"
  fi
done

python "$REPO_ROOT/scripts/autodl/camera_iteration/extract_scannet_sens.py" \
  --raw-dir "$RAW_DIR" --out-dir "$PROCESS_DIR" \
  --scene-list "$SCENE_LIST" --scene-limit "$SCENE_LIMIT"

python - "$SCANNET_ROOT" "$SCENE_LIST" "$SCENE_LIMIT" <<'PY'
from pathlib import Path
import sys
from scripts.autodl.camera_iteration.preflight import missing_processed_scenes, read_scene_list
root, scene_list, limit = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
missing = missing_processed_scenes(root, read_scene_list(scene_list, limit))
if missing:
    raise SystemExit(f"Incomplete processed scenes: {missing}")
print("ScanNet camera-iteration data ready")
PY
