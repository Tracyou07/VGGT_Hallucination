#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

SCANNET_ROOT="${SCANNET_ROOT:-/root/autodl-tmp/datasets/scannetv2}"
RAW_DOWNLOAD_ROOT="${RAW_DOWNLOAD_ROOT:-$SCANNET_ROOT/raw_sens}"
RAW_DIR="${RAW_DIR:-$RAW_DOWNLOAD_ROOT/scans}"
PROCESS_DIR="${PROCESS_DIR:-$SCANNET_ROOT/process_scannet}"
GT_DOWNLOAD_ROOT="${GT_DOWNLOAD_ROOT:-$SCANNET_ROOT/scannet}"
GT_PLY_DIR="${GT_PLY_DIR:-$GT_DOWNLOAD_ROOT/scans}"
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

repair_nested_file() {
    local base_dir="$1"
    local scene="$2"
    local filename="$3"
    local expected="$base_dir/$scene/$filename"
    local nested="$base_dir/scans/$scene/$filename"

    if [[ ! -f "$expected" && -f "$nested" ]]; then
        echo "[scannet] repairing nested download: $nested -> $expected"
        mkdir -p "$(dirname "$expected")"
        mv "$nested" "$expected"
    fi
}

run_scannet_download() {
    local output_root="$1"
    shift
    # The official ScanNet script asks for ToS confirmation and, for .sens,
    # asks whether to skip sensors. Blank answers continue and keep .sens.
    printf '\n\n\n\n' | python "$DOWNLOAD_SCRIPT" -o "$output_root" "$@"
}

for scene in "${SCENES[@]}"; do
    echo "[scannet] downloading $scene"
    repair_nested_file "$RAW_DIR" "$scene" "$scene.sens"
    repair_nested_file "$GT_PLY_DIR" "$scene" "${scene}_vh_clean_2.ply"

    if [[ ! -f "$RAW_DIR/$scene/$scene.sens" ]]; then
        run_scannet_download "$RAW_DOWNLOAD_ROOT" --id "$scene" --type .sens
    else
        echo "[scannet] skip existing $scene.sens"
    fi

    if [[ ! -f "$GT_PLY_DIR/$scene/${scene}_vh_clean_2.ply" ]]; then
        run_scannet_download "$GT_DOWNLOAD_ROOT" --id "$scene" --type _vh_clean_2.ply
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
