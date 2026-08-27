#!/usr/bin/env bash
set -euo pipefail

[[ "$(hostname)" == "VM-0-11-ubuntu" ]] || { echo "H20 host identity mismatch" >&2; exit 10; }
[[ "$(whoami)" == "ubuntu" ]] || { echo "H20 user identity mismatch" >&2; exit 11; }

REPO_ROOT="${REPO_ROOT:-/home/ubuntu/yjh/vggt/.worktrees/vrfm_random_null20}"
RUN_ROOT="${RUN_ROOT:-/data/yjh/output/variational_camera_latent/vrfm_camera_20260827T044926Z}"
SOURCE_RUN="${SOURCE_RUN:-/data/yjh/output/camera_velocity_ambiguity/cva02_20260826T2319CST_7e6fd06}"
PREPARED_ROOT="${PREPARED_ROOT:-/data/yjh/share/datasets/ScanNet/processed_cva02_v1}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/data/yjh/share/pretrained/VGGT-1B}"
VERIFIED_MARKER="${VERIFIED_MARKER:-/data/yjh/share/datasets/ScanNet/verified_completion.json}"
PYTHON="${PYTHON:-/home/ubuntu/anaconda3/envs/vggt-gx/bin/python}"
GPU_INDEX="${GPU_INDEX:-2}"
EXPECTED_BRANCH="codex/vrfm-random-null20"

usage() {
  echo "usage: $0 plan | predict REPLICATE_INDEX | finalize" >&2
  exit 2
}

PHASE="${1:-}"
[[ "${PHASE}" == "plan" || "${PHASE}" == "predict" || "${PHASE}" == "finalize" ]] || usage

[[ -d "${REPO_ROOT}" ]] || { echo "missing worktree" >&2; exit 12; }
[[ "$(git -C "${REPO_ROOT}" branch --show-current)" == "${EXPECTED_BRANCH}" ]] || { echo "wrong branch" >&2; exit 13; }
[[ -z "$(git -C "${REPO_ROOT}" status --short)" ]] || { echo "dirty worktree" >&2; exit 14; }
[[ -f "${VERIFIED_MARKER}" ]] || { echo "missing verified ScanNet marker" >&2; exit 15; }
[[ -f "${SOURCE_RUN}/manifests/calibration_complete.json" ]] || { echo "missing authenticated CVA02 run" >&2; exit 16; }
[[ -d "${CHECKPOINT_DIR}" ]] || { echo "missing local VGGT checkpoint" >&2; exit 17; }
[[ "$(df --output=avail -BG /data | tail -n 1 | tr -dc '0-9')" -ge 50 ]] || { echo "less than 50 GiB free on /data" >&2; exit 18; }

unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_TOKEN
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONNOUSERSITE=1

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/locks"
cd "${REPO_ROOT}"

COMMON=(
  --run-root "${RUN_ROOT}"
  --source-run "${SOURCE_RUN}"
  --prepared-root "${PREPARED_ROOT}"
  --checkpoint-dir "${CHECKPOINT_DIR}"
  --scene-limit 10
  --device cuda
)

run_stage() {
  local log_name="$1"
  shift
  "${PYTHON}" -m pre_experiments.variational_camera_latent.pipeline "$@" 2>&1 \
    | tee -a "${RUN_ROOT}/logs/${log_name}.log"
}

case "${PHASE}" in
  plan)
    exec 9>"${RUN_ROOT}/locks/matched_random_null20_plan.lock"
    flock -n 9 || { echo "plan stage is already active" >&2; exit 20; }
    run_stage matched_random_null20_plan --stage matched-random-plan "${COMMON[@]}"
    ;;
  predict)
    [[ $# -eq 2 ]] || usage
    REPLICATE_INDEX="$2"
    [[ "${REPLICATE_INDEX}" =~ ^([0-9]|1[0-9])$ ]] || { echo "replicate index must be 0..19" >&2; exit 21; }

    IFS=',' read -r GPU_NAME GPU_MEMORY_USED <<<"$(
      nvidia-smi --id="${GPU_INDEX}" --query-gpu=name,memory.used --format=csv,noheader,nounits
    )"
    [[ "${GPU_NAME}" == *"NVIDIA H20"* ]] || { echo "selected GPU is not H20" >&2; exit 19; }
    GPU_MEMORY_USED="${GPU_MEMORY_USED//[[:space:]]/}"
    [[ "${GPU_MEMORY_USED}" -le 1024 ]] || { echo "selected GPU is not idle" >&2; exit 22; }

    printf -v REPLICATE_TAG '%03d' "${REPLICATE_INDEX}"
    exec 8>"${RUN_ROOT}/locks/matched_random_null20_gpu_${GPU_INDEX}.lock"
    flock -n 8 || { echo "GPU ${GPU_INDEX} is already assigned" >&2; exit 23; }
    exec 9>"${RUN_ROOT}/locks/matched_random_null20_predict_${REPLICATE_TAG}.lock"
    flock -n 9 || { echo "replicate ${REPLICATE_INDEX} is already active" >&2; exit 20; }
    export CUDA_VISIBLE_DEVICES="${GPU_INDEX}"
    run_stage "matched_random_null20_predict_${REPLICATE_TAG}" \
      --stage matched-random-predict \
      --matched-random-replicate-index "${REPLICATE_INDEX}" \
      "${COMMON[@]}"
    ;;
  finalize)
    exec 9>"${RUN_ROOT}/locks/matched_random_null20_finalize.lock"
    flock -n 9 || { echo "finalize stage is already active" >&2; exit 20; }
    run_stage matched_random_null20_finalize --stage matched-random-finalize "${COMMON[@]}"
    ;;
esac

echo "[vrfm-null20] ${PHASE} complete"
