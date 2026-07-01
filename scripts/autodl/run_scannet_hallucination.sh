#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

AUTODL_TMP="${AUTODL_TMP:-/root/autodl-tmp}"
AUTODL_WORKDIR="${AUTODL_WORKDIR:-$AUTODL_TMP/vggt_hallucination}"
CONDA_ROOT="${CONDA_ROOT:-/root/miniconda3}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-vggt_hallucination}"
CONDA_CLONE_FROM="${CONDA_CLONE_FROM:-base}"
CONDA_CREATE_MODE="${CONDA_CREATE_MODE:-clone}"
SCANNET_ROOT="${SCANNET_ROOT:-$AUTODL_TMP/datasets/scannetv2}"
CKPT_DIR="${CKPT_DIR:-$AUTODL_TMP/ckpt/VGGT-1B}"
RESULT_DIR="${RESULT_DIR:-$AUTODL_WORKDIR/results}"
SCENE_LIST="${SCENE_LIST:-$REPO_ROOT/configs/scannet_hallucination_10.txt}"
SCENE_LIMIT="${SCENE_LIMIT:-10}"
FRAME_COUNTS="${FRAME_COUNTS:-100 200 300 400 500}"
SAMPLING="${SAMPLING:-prefix}"
PREPROCESS_MODE="${PREPROCESS_MODE:-pad}"
RUN_DOWNLOADS="${RUN_DOWNLOADS:-1}"
RUN_EXTRACT="${RUN_EXTRACT:-1}"
INSTALL_ENV="${INSTALL_ENV:-${SETUP_ENV:-1}}"
ACTIVATE_ENV="${ACTIVATE_ENV:-1}"
CHECK_RUNTIME_DEPS="${CHECK_RUNTIME_DEPS:-1}"
REPAIR_MISSING_DEPS="${REPAIR_MISSING_DEPS:-1}"
EVAL_NATIVE_POINTS="${EVAL_NATIVE_POINTS:-1}"
EVAL_COUNTERFACTUALS="${EVAL_COUNTERFACTUALS:-1}"

mkdir -p "$AUTODL_WORKDIR" "$RESULT_DIR"

cd "$REPO_ROOT"

if [[ ! "${OMP_NUM_THREADS:-}" =~ ^[0-9]+$ ]]; then
    export OMP_NUM_THREADS=1
fi

ensure_runtime_deps() {
    local missing_packages
    missing_packages="$(python "$SCRIPT_DIR/check_runtime_deps.py" --print-missing-packages)"
    if [[ -z "$missing_packages" ]]; then
        python "$SCRIPT_DIR/check_runtime_deps.py"
        return
    fi

    echo "[deps] missing runtime packages: $missing_packages"
    if [[ "$REPAIR_MISSING_DEPS" != "1" ]]; then
        echo "[deps] ERROR: set REPAIR_MISSING_DEPS=1 or install the packages above"
        exit 1
    fi

    if [[ "$missing_packages" == *"opencv-python-headless==4.11.0.86"* ]]; then
        python -m pip install --force-reinstall --no-deps --no-cache-dir opencv-python-headless==4.11.0.86
        missing_packages="${missing_packages/opencv-python-headless==4.11.0.86/}"
    fi
    # shellcheck disable=SC2086
    if [[ -n "${missing_packages// }" ]]; then
        python -m pip install $missing_packages
    fi
    python "$SCRIPT_DIR/check_runtime_deps.py"
}

if [[ "$ACTIVATE_ENV" == "1" || "$INSTALL_ENV" == "1" ]]; then
    if [[ -n "${VIRTUAL_ENV:-}" ]] && type deactivate >/dev/null 2>&1; then
        echo "[env] deactivating current virtualenv: $VIRTUAL_ENV"
        deactivate
    fi

    if [[ ! -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]]; then
        echo "[env] ERROR: conda not found at $CONDA_ROOT"
        echo "[env] set CONDA_ROOT=/path/to/miniconda3 if AutoDL uses another path"
        exit 1
    fi
    # shellcheck source=/dev/null
    source "$CONDA_ROOT/etc/profile.d/conda.sh"

    if [[ "$INSTALL_ENV" == "1" ]] && ! conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV_NAME"; then
        if [[ "$CONDA_CREATE_MODE" == "clone" ]]; then
            echo "[env] creating conda env $CONDA_ENV_NAME by cloning $CONDA_CLONE_FROM"
            conda create -y -n "$CONDA_ENV_NAME" --clone "$CONDA_CLONE_FROM"
        else
            echo "[env] creating conda env $CONDA_ENV_NAME with python=3.10"
            conda create -y -n "$CONDA_ENV_NAME" python=3.10
        fi
    fi

    conda activate "$CONDA_ENV_NAME"
    python - <<'PY'
import sys
try:
    import torch
except Exception as exc:
    raise SystemExit(f"[env] ERROR: torch is not available in this conda env: {exc}")
try:
    import torchvision
except Exception as exc:
    raise SystemExit(f"[env] ERROR: torchvision is not available in this conda env: {exc}")
print(f"[env] python={sys.version.split()[0]}")
print(f"[env] torch={torch.__version__} cuda={torch.version.cuda} cuda_available={torch.cuda.is_available()}")
print(f"[env] torchvision={torchvision.__version__}")
if not torch.cuda.is_available():
    raise SystemExit("[env] ERROR: CUDA is not available in torch. Use the AutoDL CUDA/PyTorch image or set CONDA_ENV_NAME to that env.")
PY

    if [[ "$INSTALL_ENV" == "1" ]]; then
        python -m pip uninstall -y opencv-python opencv-contrib-python opencv-contrib-python-headless || true
        python -m pip install -r requirements-autodl.txt
        python -m pip install -e . --no-deps
    fi
fi

if [[ "$CHECK_RUNTIME_DEPS" == "1" ]]; then
    ensure_runtime_deps
fi

if [[ "$RUN_DOWNLOADS" == "1" ]]; then
    CKPT_DIR="$CKPT_DIR" bash "$SCRIPT_DIR/download_vggt_weights.sh"
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
