#!/usr/bin/env bash
set -euo pipefail

CKPT_DIR="${CKPT_DIR:-/root/autodl-tmp/ckpt/VGGT-1B}"
HF_REPO="${HF_REPO:-facebook/VGGT-1B}"

mkdir -p "$CKPT_DIR"

if [[ -f "$CKPT_DIR/model.safetensors" || -f "$CKPT_DIR/model.pt" ]]; then
    echo "[weights] found existing checkpoint in $CKPT_DIR"
    exit 0
fi

echo "[weights] downloading $HF_REPO -> $CKPT_DIR"
python - "$HF_REPO" "$CKPT_DIR" <<'PY'
import sys
from huggingface_hub import snapshot_download

repo_id = sys.argv[1]
local_dir = sys.argv[2]
snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    allow_patterns=["*.safetensors", "*.bin", "*.pt", "*.json", "*.md"],
)
print(f"downloaded={local_dir}")
PY
