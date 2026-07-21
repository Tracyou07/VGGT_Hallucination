#!/usr/bin/env bash
set -euo pipefail

AUTODL_TMP="${AUTODL_TMP:-/root/autodl-tmp}"
CONDA_ROOT="${CONDA_ROOT:-/root/miniconda3}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-vggt}"
CKPT_DIR="${CKPT_DIR:-$AUTODL_TMP/ckpt/VGGT-1B}"
HF_REPO="${HF_REPO:-facebook/VGGT-1B}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
HF_HOME="${HF_HOME:-$AUTODL_TMP/hf_home}"
HF_MAX_RETRIES="${HF_MAX_RETRIES:-5}"
CONDA_SH="$CONDA_ROOT/etc/profile.d/conda.sh"

[[ -f "$CONDA_SH" ]] || { printf 'Run setup_vggt_env.sh first; conda not found.\n' >&2; exit 1; }
# shellcheck source=/dev/null
source "$CONDA_SH"
conda run -n "$CONDA_ENV_NAME" python -c "import huggingface_hub" >/dev/null 2>&1 || {
  printf 'Run setup_vggt_env.sh first; environment %s is unavailable.\n' "$CONDA_ENV_NAME" >&2
  exit 1
}
conda activate "$CONDA_ENV_NAME"

for checkpoint in "$CKPT_DIR/model.safetensors" "$CKPT_DIR/model.pt"; do
  if [[ -s "$checkpoint" ]]; then
    printf '[weights] reuse %s\n' "$checkpoint"
    exit 0
  fi
done
mkdir -p "$CKPT_DIR" "$HF_HOME"
export CKPT_DIR HF_REPO HF_ENDPOINT HF_HOME

for ((attempt=1; attempt<=HF_MAX_RETRIES; attempt++)); do
  if python - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id=os.environ["HF_REPO"],
    local_dir=os.environ["CKPT_DIR"],
    allow_patterns=["model.safetensors", "config.json"],
    max_workers=1,
)
PY
  then
    [[ -s "$CKPT_DIR/model.safetensors" ]] && exit 0
  fi
  printf '[weights] attempt %s/%s failed\n' "$attempt" "$HF_MAX_RETRIES" >&2
  sleep $((attempt * 5))
done
printf 'Failed to download a non-empty VGGT checkpoint into %s\n' "$CKPT_DIR" >&2
exit 1
