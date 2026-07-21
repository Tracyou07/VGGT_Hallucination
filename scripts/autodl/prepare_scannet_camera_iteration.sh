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
RAW_DIR="${RAW_DIR:-$RAW_DOWNLOAD_ROOT/scans}"
PROCESS_DIR="${PROCESS_DIR:-$SCANNET_ROOT/process_scannet}"
SCENE_LIST="${SCENE_LIST:-$REPO_ROOT/configs/camera_iteration_scannet.txt}"
SCENE_LIMIT="${SCENE_LIMIT:-10}"
SCANNET_DOWNLOAD_SCRIPT="${SCANNET_DOWNLOAD_SCRIPT:-$SCANNET_ROOT/tools/download-scannet.py}"
SCANNET_DOWNLOAD_URL="${SCANNET_DOWNLOAD_URL:-http://kaldir.vc.in.tum.de/scannet/download-scannet.py}"
CONDA_SH="$CONDA_ROOT/etc/profile.d/conda.sh"

[[ -f "$CONDA_SH" ]] || { printf 'Run setup_vggt_env.sh first.\n' >&2; exit 1; }
# shellcheck source=/dev/null
source "$CONDA_SH"
conda run -n "$CONDA_ENV_NAME" python -c "import imageio" >/dev/null 2>&1 || {
  printf 'Run setup_vggt_env.sh first; environment %s is unavailable.\n' "$CONDA_ENV_NAME" >&2
  exit 1
}
conda activate "$CONDA_ENV_NAME"
cd "$REPO_ROOT"

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

mapfile -t scenes < <(python - "$SCENE_LIST" "$SCENE_LIMIT" <<'PY'
from pathlib import Path
import sys
from scripts.autodl.camera_iteration.preflight import read_scene_list
for scene in read_scene_list(Path(sys.argv[1]), int(sys.argv[2])):
    print(scene)
PY
)

for scene in "${scenes[@]}"; do
  sens="$RAW_DIR/$scene/$scene.sens"
  if [[ -s "$sens" ]]; then
    printf '[scannet] reuse %s\n' "$sens"
    continue
  fi
  mkdir -p "$RAW_DIR/$scene"
  printf '\n\n\n\n' | python "$SCANNET_DOWNLOAD_SCRIPT" \
    -o "$RAW_DOWNLOAD_ROOT" --id "$scene" --type .sens
  found="$(find "$RAW_DOWNLOAD_ROOT" -type f -name "$scene.sens" -size +0c -print -quit)"
  [[ -n "$found" ]] || { printf 'Official downloader did not produce %s.sens\n' "$scene" >&2; exit 1; }
  if [[ "$found" != "$sens" ]]; then
    cp "$found" "$sens"
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
