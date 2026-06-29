#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

SCANNET_ROOT="${SCANNET_ROOT:-/root/autodl-tmp/datasets/scannetv2}"
RAW_DIR="${RAW_DIR:-$SCANNET_ROOT/raw_sens/scans}"
PROCESS_DIR="${PROCESS_DIR:-$SCANNET_ROOT/process_scannet}"
GT_PLY_DIR="${GT_PLY_DIR:-$SCANNET_ROOT/scannet/scans}"
SCENE_LIST="${SCENE_LIST:-$REPO_ROOT/configs/scannet_hallucination_10.txt}"
SCENE_LIMIT="${SCENE_LIMIT:-10}"
DOWNLOAD_SCRIPT="${SCANNET_DOWNLOAD_SCRIPT:-$SCANNET_ROOT/tools/download-scannet.py}"
SCANNET_DOWNLOAD_URL="${SCANNET_DOWNLOAD_URL:-http://kaldir.vc.in.tum.de/scannet/download-scannet.py}"

mkdir -p "$RAW_DIR" "$PROCESS_DIR" "$GT_PLY_DIR" "$(dirname "$DOWNLOAD_SCRIPT")"

if [[ ! -f "$DOWNLOAD_SCRIPT" ]]; then
    echo "[scannet] download-scannet.py not found at $DOWNLOAD_SCRIPT"
    echo "[scannet] trying official script URL: $SCANNET_DOWNLOAD_URL"
    if command -v curl >/dev/null 2>&1; then
        curl -L "$SCANNET_DOWNLOAD_URL" -o "$DOWNLOAD_SCRIPT" || true
    elif command -v wget >/dev/null 2>&1; then
        wget -O "$DOWNLOAD_SCRIPT" "$SCANNET_DOWNLOAD_URL" || true
    fi
fi

if [[ ! -s "$DOWNLOAD_SCRIPT" ]]; then
    cat >&2 <<EOF
[scannet] ERROR: ScanNet download script is unavailable.
ScanNet requires data terms acceptance. Apply at http://www.scan-net.org/,
then set SCANNET_DOWNLOAD_SCRIPT=/path/to/download-scannet.py and rerun.
EOF
    exit 1
fi

mapfile -t SCENES < <(grep -E '^scene[0-9]{4}_[0-9]{2}$' "$SCENE_LIST")
if [[ "$SCENE_LIMIT" != "0" ]]; then
    SCENES=("${SCENES[@]:0:$SCENE_LIMIT}")
fi

echo "[scannet] root=$SCANNET_ROOT"
echo "[scannet] scenes=${#SCENES[@]}"

for scene in "${SCENES[@]}"; do
    echo "[scannet] downloading $scene"
    if [[ ! -f "$RAW_DIR/$scene/$scene.sens" ]]; then
        python "$DOWNLOAD_SCRIPT" -o "$RAW_DIR" --id "$scene" --type .sens
    else
        echo "[scannet] skip existing $scene.sens"
    fi

    if [[ ! -f "$GT_PLY_DIR/$scene/${scene}_vh_clean_2.ply" ]]; then
        python "$DOWNLOAD_SCRIPT" -o "$GT_PLY_DIR" --id "$scene" --type _vh_clean_2.ply
    else
        echo "[scannet] skip existing ${scene}_vh_clean_2.ply"
    fi
done

python "$SCRIPT_DIR/extract_scannet_sens.py" \
    --raw-dir "$RAW_DIR" \
    --out-dir "$PROCESS_DIR" \
    --scene-list "$SCENE_LIST" \
    --scene-limit "$SCENE_LIMIT" \
    --export-depth

echo "[scannet] processed data: $PROCESS_DIR"
echo "[scannet] gt ply: $GT_PLY_DIR"
