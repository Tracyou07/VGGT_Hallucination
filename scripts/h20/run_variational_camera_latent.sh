#!/usr/bin/env bash
set -euo pipefail

[[ "$(hostname)" == "VM-0-11-ubuntu" ]] || { echo "H20 host identity mismatch" >&2; exit 10; }
[[ "$(whoami)" == "ubuntu" ]] || { echo "H20 user identity mismatch" >&2; exit 11; }

REPO_ROOT="${REPO_ROOT:-/home/ubuntu/yjh/vggt/.worktrees/camera_velocity_ambiguity_02_pre_experiment}"
RESULT_ROOT="${RESULT_ROOT:-/data/yjh/output/variational_camera_latent}"
SOURCE_RUN="${SOURCE_RUN:-/data/yjh/output/camera_velocity_ambiguity/cva02_20260826T2319CST_7e6fd06}"
PREPARED_ROOT="${PREPARED_ROOT:-/data/yjh/share/datasets/ScanNet/processed_cva02_v1}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/data/yjh/share/pretrained/VGGT-1B}"
VERIFIED_MARKER="${VERIFIED_MARKER:-/data/yjh/share/datasets/ScanNet/verified_completion.json}"
PYTHON="${PYTHON:-/home/ubuntu/anaconda3/envs/vggt-gx/bin/python}"
RUN_ID="${RUN_ID:-vrfm_camera_$(date -u +%Y%m%dT%H%M%SZ)}"
GPU_INDEX="${GPU_INDEX:-0}"
SMOKE_SCENE_LIMIT="1"
CALIBRATION_SCENE_LIMIT="10"
RUN_ROOT="${RESULT_ROOT}/${RUN_ID}"

[[ -d "${REPO_ROOT}" ]] || { echo "missing worktree" >&2; exit 12; }
[[ "$(git -C "${REPO_ROOT}" branch --show-current)" == "codex/camera_velocity_ambiguity_02_pre_experiment" ]] || { echo "wrong branch" >&2; exit 13; }
[[ -z "$(git -C "${REPO_ROOT}" status --short)" ]] || { echo "dirty worktree" >&2; exit 14; }
[[ -f "${VERIFIED_MARKER}" ]] || { echo "missing verified ScanNet marker" >&2; exit 15; }
[[ -f "${SOURCE_RUN}/manifests/calibration_complete.json" ]] || { echo "missing authenticated CVA02 run" >&2; exit 16; }
[[ -d "${CHECKPOINT_DIR}" ]] || { echo "missing local VGGT checkpoint" >&2; exit 17; }
[[ "$(df --output=avail -BG /data | tail -n 1 | tr -dc '0-9')" -ge 50 ]] || { echo "less than 50 GiB free on /data" >&2; exit 18; }
nvidia-smi --query-gpu=index,name --format=csv,noheader | grep -E "^${GPU_INDEX}, NVIDIA H20" >/dev/null || { echo "selected GPU is not H20" >&2; exit 19; }

mkdir -p "${RUN_ROOT}/logs"
exec 9>"${RUN_ROOT}/.lock"
flock -n 9 || { echo "run is already active" >&2; exit 20; }
cd "${REPO_ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU_INDEX}"

run_stage() {
  local name="$1"
  shift
  "${PYTHON}" -m pre_experiments.variational_camera_latent.pipeline "$@" 2>&1 | tee -a "${RUN_ROOT}/logs/${name}.log"
}

COMMON=(--run-root "${RUN_ROOT}" --source-run "${SOURCE_RUN}" --prepared-root "${PREPARED_ROOT}" --checkpoint-dir "${CHECKPOINT_DIR}" --device cuda)
run_stage source --stage source --scene-limit "${CALIBRATION_SCENE_LIMIT}" "${COMMON[@]}"
run_stage smoke --stage smoke --scene-limit "${SMOKE_SCENE_LIMIT}" "${COMMON[@]}"
run_stage calibration --stage calibration --scene-limit "${CALIBRATION_SCENE_LIMIT}" "${COMMON[@]}"
run_stage privileged --stage privileged --scene-limit "${CALIBRATION_SCENE_LIMIT}" "${COMMON[@]}"
run_stage report --stage report --scene-limit "${CALIBRATION_SCENE_LIMIT}" "${COMMON[@]}"
run_stage verify --stage verify --run-root "${RUN_ROOT}"

echo "[vrfm] complete ${RUN_ROOT}"
