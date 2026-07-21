#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONDA_ROOT="${CONDA_ROOT:-/root/miniconda3}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-vggt}"
CONDA_CLONE_FROM="${CONDA_CLONE_FROM:-base}"
CONDA_SH="$CONDA_ROOT/etc/profile.d/conda.sh"

if [[ ! -f "$CONDA_SH" ]]; then
  printf 'Conda initialization script not found: %s\n' "$CONDA_SH" >&2
  exit 1
fi
# shellcheck source=/dev/null
source "$CONDA_SH"

if ! conda run -n "$CONDA_ENV_NAME" python -c "import sys" >/dev/null 2>&1; then
  conda create --name "$CONDA_ENV_NAME" --clone "$CONDA_CLONE_FROM" -y
fi
conda activate "$CONDA_ENV_NAME"

python -c "import torch; assert torch.cuda.is_available(), 'Base environment must provide CUDA-enabled Torch'; print('torch=' + torch.__version__ + ' cuda=' + str(torch.version.cuda) + ' device=' + torch.cuda.get_device_name(0))"
python -m pip install --no-deps --no-build-isolation -e "$REPO_ROOT"
mapfile -t missing_specs < <(
  python "$REPO_ROOT/scripts/autodl/camera_iteration/preflight.py" --print-missing
)
if (( ${#missing_specs[@]} > 0 )); then
  python -m pip install "${missing_specs[@]}"
fi
python -c "import torch, vggt; assert torch.cuda.is_available(); print('vggt environment ready')"
