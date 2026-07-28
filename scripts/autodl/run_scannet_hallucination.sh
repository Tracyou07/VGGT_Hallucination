#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

AUTODL_TMP="${AUTODL_TMP:-/root/autodl-tmp}"
CONDA_ROOT="${CONDA_ROOT:-/root/miniconda3}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-vggt}"
SCANNET_ROOT="${SCANNET_ROOT:-$AUTODL_TMP/datasets/scannetv2}"
CKPT_DIR="${CKPT_DIR:-$AUTODL_TMP/ckpt/VGGT-1B}"
RESULT_DIR="${RESULT_DIR:-$AUTODL_TMP/vggt_hallucination/results}"
SCENE_LIST="${SCENE_LIST:-$REPO_ROOT/configs/scannet_hallucination_10.txt}"
SCENE_LIMIT="${SCENE_LIMIT:-10}"
FRAME_COUNTS="${FRAME_COUNTS:-100 200 300 400 500}"
SAMPLING="${SAMPLING:-prefix}"
PREPROCESS_MODE="${PREPROCESS_MODE:-pad}"
RUN_DATA_DOWNLOAD="${RUN_DATA_DOWNLOAD:-0}"
RUN_EXTRACT="${RUN_EXTRACT:-0}"
EVAL_NATIVE_POINTS="${EVAL_NATIVE_POINTS:-1}"
EVAL_COUNTERFACTUALS="${EVAL_COUNTERFACTUALS:-1}"
CONDA_SH="$CONDA_ROOT/etc/profile.d/conda.sh"

[[ -f "$CONDA_SH" ]] || {
    printf '[env] conda initialization not found at %s\n' "$CONDA_SH" >&2
    exit 1
}
# shellcheck source=/dev/null
source "$CONDA_SH"
conda activate "$CONDA_ENV_NAME"

cd "$REPO_ROOT"
python "$SCRIPT_DIR/check_runtime_deps.py"
python - <<'PY'
import torch
import torchvision

print(f"[env] torch={torch.__version__} cuda={torch.version.cuda}")
print(f"[env] torchvision={torchvision.__version__}")
if not torch.cuda.is_available():
    raise SystemExit("[env] CUDA is not available in the active vggt environment")
PY

[[ -f "$CKPT_DIR/model.safetensors" || -f "$CKPT_DIR/model.pt" ]] || {
    printf '[ckpt] missing model.safetensors or model.pt under %s\n' "$CKPT_DIR" >&2
    exit 1
}

if [[ "$RUN_DATA_DOWNLOAD" == "1" ]]; then
    SCANNET_ROOT="$SCANNET_ROOT" \
    SCENE_LIST="$SCENE_LIST" \
    SCENE_LIMIT="$SCENE_LIMIT" \
    bash "$SCRIPT_DIR/download_scannet_subset.sh"
elif [[ "$RUN_EXTRACT" == "1" ]]; then
    python "$SCRIPT_DIR/extract_scannet_sens.py" \
        --raw-dir "$SCANNET_ROOT/raw_sens/scans" \
        --out-dir "$SCANNET_ROOT/process_scannet" \
        --scene-list "$SCENE_LIST" \
        --scene-limit "$SCENE_LIMIT" \
        --export-depth
fi

read -r -a FRAME_ARGS <<< "$FRAME_COUNTS"
(( ${#FRAME_ARGS[@]} > 0 )) || {
    printf 'FRAME_COUNTS must not be empty\n' >&2
    exit 1
}
mkdir -p "$RESULT_DIR"

native_flag=()
[[ "$EVAL_NATIVE_POINTS" == "1" ]] && native_flag=(--eval-native-points)
counterfactual_flag=()
[[ "$EVAL_COUNTERFACTUALS" == "1" ]] && counterfactual_flag=(--eval-counterfactuals)

python -m experiments.scannet_hallucination.run_eval \
    --data-dir "$SCANNET_ROOT/process_scannet" \
    --gt-ply-dir "$SCANNET_ROOT/scannet/scans" \
    --scene-list "$SCENE_LIST" \
    --scene-limit "$SCENE_LIMIT" \
    --frame-counts "${FRAME_ARGS[@]}" \
    --sampling "$SAMPLING" \
    --weights local \
    --ckpt-dir "$CKPT_DIR" \
    --device cuda \
    --preprocess-mode "$PREPROCESS_MODE" \
    --out-dir "$RESULT_DIR" \
    "${native_flag[@]}" \
    "${counterfactual_flag[@]}"

echo "[done] results: $RESULT_DIR"
