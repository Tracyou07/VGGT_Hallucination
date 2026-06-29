#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

AUTODL_WORKDIR="${AUTODL_WORKDIR:-/root/autodl-tmp/vggt_hallucination}"
VENV_DIR="${VENV_DIR:-$AUTODL_WORKDIR/venv}"
SCANNET_ROOT="${SCANNET_ROOT:-/root/autodl-tmp/datasets/scannetv2}"
CKPT_DIR="${CKPT_DIR:-/root/autodl-tmp/ckpt/VGGT-1B}"
RESULT_DIR="${RESULT_DIR:-$AUTODL_WORKDIR/results}"
SCENE_LIST="${SCENE_LIST:-$REPO_ROOT/configs/scannet_hallucination_10.txt}"
SCENE_LIMIT="${SCENE_LIMIT:-10}"
FRAME_COUNTS="${FRAME_COUNTS:-100 300 500 1000}"
SAMPLING="${SAMPLING:-prefix}"
PREPROCESS_MODE="${PREPROCESS_MODE:-pad}"
RUN_DOWNLOADS="${RUN_DOWNLOADS:-1}"
INSTALL_ENV="${INSTALL_ENV:-1}"
EVAL_NATIVE_POINTS="${EVAL_NATIVE_POINTS:-1}"
EVAL_COUNTERFACTUALS="${EVAL_COUNTERFACTUALS:-1}"

mkdir -p "$AUTODL_WORKDIR" "$RESULT_DIR"

cd "$REPO_ROOT"

if [[ "$INSTALL_ENV" == "1" ]]; then
    if [[ ! -d "$VENV_DIR" ]]; then
        python3 -m venv "$VENV_DIR"
    fi
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    python -m pip install --upgrade pip wheel
    python -m pip install -r requirements-autodl.txt
    python -m pip install -e .
else
    if [[ -f "$VENV_DIR/bin/activate" ]]; then
        # shellcheck disable=SC1091
        source "$VENV_DIR/bin/activate"
    fi
fi

if [[ "$RUN_DOWNLOADS" == "1" ]]; then
    CKPT_DIR="$CKPT_DIR" bash "$SCRIPT_DIR/download_vggt_weights.sh"
    SCANNET_ROOT="$SCANNET_ROOT" \
    SCENE_LIST="$SCENE_LIST" \
    SCENE_LIMIT="$SCENE_LIMIT" \
    bash "$SCRIPT_DIR/download_scannet_subset.sh"
fi

read -r -a FRAME_ARGS <<< "$FRAME_COUNTS"

native_flag=()
if [[ "$EVAL_NATIVE_POINTS" == "1" ]]; then
    native_flag=(--eval-native-points)
fi

counterfactual_flag=()
if [[ "$EVAL_COUNTERFACTUALS" == "1" ]]; then
    counterfactual_flag=(--eval-counterfactuals)
fi

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
