#!/usr/bin/env bash
set -euo pipefail

CKPT_DIR="${CKPT_DIR:-/root/autodl-tmp/ckpt/VGGT-1B}"
HF_REPO="${HF_REPO:-facebook/VGGT-1B}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
HF_HOME="${HF_HOME:-/root/autodl-tmp/hf_home}"
HF_MAX_RETRIES="${HF_MAX_RETRIES:-5}"

mkdir -p "$CKPT_DIR"
mkdir -p "$HF_HOME"
export HF_ENDPOINT HF_HOME HF_HUB_DISABLE_TELEMETRY=1

if [[ -f "$CKPT_DIR/model.safetensors" || -f "$CKPT_DIR/model.pt" ]]; then
    echo "[weights] found existing checkpoint in $CKPT_DIR"
    exit 0
fi

echo "[weights] downloading $HF_REPO -> $CKPT_DIR"
echo "[weights] endpoint=$HF_ENDPOINT"
echo "[weights] cache=$HF_HOME"
python - "$HF_REPO" "$CKPT_DIR" "$HF_MAX_RETRIES" <<'PY'
import os
import sys
import time
from huggingface_hub import snapshot_download

repo_id = sys.argv[1]
local_dir = sys.argv[2]
max_retries = int(sys.argv[3])

last_error = None
for attempt in range(1, max_retries + 1):
    try:
        print(f"[weights] attempt={attempt}/{max_retries}", flush=True)
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            allow_patterns=["*.safetensors", "*.bin", "*.pt", "*.json", "*.md"],
            max_workers=1,
            etag_timeout=60,
        )
        last_error = None
        break
    except Exception as exc:
        last_error = exc
        print(f"[weights] attempt failed: {type(exc).__name__}: {exc}", flush=True)
        time.sleep(min(30, 5 * attempt))

if last_error is not None:
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
    raise SystemExit(
        "[weights] ERROR: failed to download VGGT weights.\n"
        f"[weights] endpoint={endpoint}\n"
        "[weights] Try rerunning with another endpoint, for example:\n"
        "  HF_ENDPOINT=https://hf-mirror.com bash scripts/autodl/run_scannet_hallucination.sh\n"
        "  HF_ENDPOINT=https://huggingface.co bash scripts/autodl/run_scannet_hallucination.sh"
    )

print(f"downloaded={local_dir}")
PY
