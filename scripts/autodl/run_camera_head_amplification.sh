#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
AUTODL_TMP="${AUTODL_TMP:-/root/autodl-tmp}"
CONDA_ROOT="${CONDA_ROOT:-/root/miniconda3}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-vggt}"
SOURCE_RUN_DIR="${SOURCE_RUN_DIR:-$REPO_ROOT/results/camera_context/911b598_f4577f584448}"
CKPT_DIR="${CKPT_DIR:-$AUTODL_TMP/ckpt/VGGT-1B}"
RESULT_DIR="${RESULT_DIR:-$AUTODL_TMP/camera_head_amplification/results}"
DEVICE="${DEVICE:-cuda}"
SHORT_FRAMES="${SHORT_FRAMES:-200}"
LONG_FRAMES="${LONG_FRAMES:-500}"
ITERATIONS="${ITERATIONS:-4}"
SCENE_LIMIT="${SCENE_LIMIT:-0}"
BASELINE_ATOL="${BASELINE_ATOL:-1e-5}"
BASELINE_RTOL="${BASELINE_RTOL:-1e-5}"

[[ -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]] || {
  printf 'Missing conda activation script: %s\n' "$CONDA_ROOT/etc/profile.d/conda.sh" >&2
  exit 1
}
[[ -f "$SOURCE_RUN_DIR/run_metadata.json" ]] || {
  printf 'Missing Round 1.5 run metadata: %s\n' "$SOURCE_RUN_DIR/run_metadata.json" >&2
  exit 1
}
find "$SOURCE_RUN_DIR" -name context_diagnostics.npz -print -quit | grep -q . || {
  printf 'No context_diagnostics.npz files under %s\n' "$SOURCE_RUN_DIR" >&2
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

python -m pre_experiments.camera_head_amplification.run_replay \
  --source-run-dir "$SOURCE_RUN_DIR" \
  --ckpt-dir "$CKPT_DIR" \
  --out-dir "$RESULT_DIR" \
  --device "$DEVICE" \
  --short-frames "$SHORT_FRAMES" \
  --long-frames "$LONG_FRAMES" \
  --iterations "$ITERATIONS" \
  --scene-limit "$SCENE_LIMIT" \
  --baseline-atol "$BASELINE_ATOL" \
  --baseline-rtol "$BASELINE_RTOL"
