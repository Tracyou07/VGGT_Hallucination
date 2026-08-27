#!/usr/bin/env bash
set -euo pipefail

[[ "$(hostname)" == "VM-0-11-ubuntu" ]] || { echo "H20 host identity mismatch" >&2; exit 10; }
[[ "$(id -un)" == "ubuntu" ]] || { echo "H20 user identity mismatch" >&2; exit 11; }

REPO_ROOT="${REPO_ROOT:-/home/ubuntu/yjh/vggt/.worktrees/vrfm_candidate_selector}"
INPUT_ROOT="${INPUT_ROOT:-/data/yjh/output/variational_camera_latent/vrfm_camera_20260827T044926Z}"
RESULT_ROOT="${RESULT_ROOT:-/data/yjh/output/variational_camera_selector}"
PYTHON="${PYTHON:-/home/ubuntu/anaconda3/envs/vggt-gx/bin/python}"
RUN_ID="${RUN_ID:-selector_$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="${1:-${RESULT_ROOT}/${RUN_ID}}"

[[ -d "${REPO_ROOT}" ]] || { echo "missing selector worktree" >&2; exit 12; }
[[ -z "$(git -C "${REPO_ROOT}" status --porcelain)" ]] || { echo "dirty selector worktree" >&2; exit 13; }
[[ -x "${PYTHON}" ]] || { echo "missing selector Python environment" >&2; exit 14; }
[[ -f "${INPUT_ROOT}/verified_completion.json" ]] || { echo "missing sealed Phase 1 completion" >&2; exit 15; }
[[ -f "${INPUT_ROOT}/vrfm_residual_alpha_scan_full_context_verified_completion.json" ]] || { echo "missing sealed residual completion" >&2; exit 16; }
[[ "$(df --output=avail -BG /data | tail -n 1 | tr -dc '0-9')" -ge 20 ]] || { echo "less than 20 GiB free on /data" >&2; exit 17; }

if [[ -z "${GPU_INDEX:-}" ]]; then
  GPU_INDEX="$(nvidia-smi --query-gpu=index,name,memory.free,utilization.gpu --format=csv,noheader,nounits | awk -F',' '
    { gsub(/ /, "", $1); gsub(/^ +| +$/, "", $2); gsub(/ /, "", $3); gsub(/ /, "", $4) }
    $2 == "NVIDIAH20" && ($3 + 0) >= 80000 && ($4 + 0) <= 10 { print $1; exit }
  ')"
fi
[[ -n "${GPU_INDEX}" ]] || { echo "no non-conflicting H20 GPU has at least 80000 MiB free" >&2; exit 18; }
nvidia-smi --query-gpu=index,name --format=csv,noheader,nounits | grep -E "^${GPU_INDEX}, NVIDIA H20" >/dev/null || { echo "selected GPU is not H20" >&2; exit 19; }

mkdir -p "${RUN_ROOT}"
exec 9>"${RUN_ROOT}/.lock"
flock -n 9 || { echo "selector run is already active" >&2; exit 20; }
cd "${REPO_ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU_INDEX}"
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN

exec "${PYTHON}" -m pre_experiments.variational_camera_selector.pipeline \
  --stage auto \
  --run-root "${RUN_ROOT}" \
  --input-root "${INPUT_ROOT}" \
  --device cuda
