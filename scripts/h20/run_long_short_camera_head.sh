#!/usr/bin/env bash
set -Eeuo pipefail

[[ "$(hostname)" == "VM-0-11-ubuntu" ]] || { echo "H20 host identity mismatch" >&2; exit 10; }
[[ "$(id -un)" == "ubuntu" ]] || { echo "H20 user identity mismatch" >&2; exit 11; }

REPO_ROOT="${REPO_ROOT:-/home/ubuntu/yjh/vggt/.worktrees/long_short_camera_head}"
RESULT_ROOT="${RESULT_ROOT:-/data/yjh/output/vggt/long_short_camera_head}"
SOURCE_RUN="${SOURCE_RUN:-/data/yjh/output/vggt/variational_camera_latent/vrfm_camera_20260827T044926Z}"
PREPARED_ROOT="${PREPARED_ROOT:-/data/yjh/share/datasets/ScanNet/processed_cva02_v1}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/data/yjh/share/pretrained/VGGT-1B}"
PYTHON="${PYTHON:-/home/ubuntu/anaconda3/envs/vggt-gx/bin/python}"
RUN_ID="${RUN_ID:-long_short_head_$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="${RESULT_ROOT}/${RUN_ID}"
GPU_GT_ONLY="${GPU_GT_ONLY:-2}"
GPU_LONG_SHORT="${GPU_LONG_SHORT:-3}"
SMOKE_STEPS="20"
CALIBRATION_STEPS="400"
LEARNING_RATE="2e-6"
CHECKPOINT_INTERVAL="25"
PATIENCE="100"

[[ -d "${REPO_ROOT}" ]] || { echo "missing dedicated worktree" >&2; exit 12; }
[[ "$(git -C "${REPO_ROOT}" branch --show-current)" == "codex/long-short-camera-head-finetune" ]] || { echo "wrong branch" >&2; exit 13; }
[[ -z "$(git -C "${REPO_ROOT}" status --short)" ]] || { echo "dirty worktree" >&2; exit 14; }
[[ -x "${PYTHON}" ]] || { echo "missing VGGT Python environment" >&2; exit 15; }
[[ -f "${SOURCE_RUN}/verified_completion.json" ]] || { echo "missing verified VRFM source run" >&2; exit 16; }
[[ -d "${PREPARED_ROOT}" ]] || { echo "missing prepared ScanNet data" >&2; exit 17; }
[[ -f "${CHECKPOINT_DIR}/model.safetensors" || -f "${CHECKPOINT_DIR}/model.pt" ]] || { echo "missing local VGGT checkpoint" >&2; exit 18; }
[[ "$(df --output=avail -BG /data | tail -n 1 | tr -dc '0-9')" -ge 100 ]] || { echo "less than 100 GiB free on /data" >&2; exit 19; }
[[ "${GPU_GT_ONLY}" != "${GPU_LONG_SHORT}" ]] || { echo "matched variants need distinct GPUs" >&2; exit 20; }

check_idle_h20() {
  local gpu="$1"
  nvidia-smi -i "${gpu}" --query-gpu=name --format=csv,noheader | grep -Fx "NVIDIA H20" >/dev/null || return 1
  if nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits | grep -Eq '[0-9]+'; then
    return 1
  fi
}
check_idle_h20 "${GPU_GT_ONLY}" || { echo "GPU ${GPU_GT_ONLY} is not an idle NVIDIA H20" >&2; exit 21; }
check_idle_h20 "${GPU_LONG_SHORT}" || { echo "GPU ${GPU_LONG_SHORT} is not an idle NVIDIA H20" >&2; exit 22; }

mkdir -p "${RUN_ROOT}/logs"
exec 9>"${RUN_ROOT}/.lock"
flock -n 9 || { echo "run is already active" >&2; exit 23; }
cd "${REPO_ROOT}"

run_stage() {
  local gpu="$1"
  local name="$2"
  shift 2
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -m pre_experiments.long_short_camera_head.pipeline "$@" \
    >"${RUN_ROOT}/logs/${name}.out.log" \
    2>"${RUN_ROOT}/logs/${name}.err.log"
  [[ ! -s "${RUN_ROOT}/logs/${name}.err.log" ]] || { echo "${name} wrote stderr" >&2; exit 24; }
}

COMMON=(--run-root "${RUN_ROOT}" --checkpoint-dir "${CHECKPOINT_DIR}" --device cuda)
run_stage "${GPU_GT_ONLY}" prepare --stage prepare "${COMMON[@]}" \
  --source-run "${SOURCE_RUN}" --prepared-root "${PREPARED_ROOT}"
run_stage "${GPU_GT_ONLY}" preflight --stage preflight "${COMMON[@]}"
run_stage "${GPU_GT_ONLY}" smoke --stage smoke "${COMMON[@]}" \
  --max-steps "${SMOKE_STEPS}" --learning-rate "${LEARNING_RATE}"

CUDA_VISIBLE_DEVICES="${GPU_GT_ONLY}" "${PYTHON}" -m pre_experiments.long_short_camera_head.pipeline \
  --stage calibration "${COMMON[@]}" --variant gt_only \
  --max-steps "${CALIBRATION_STEPS}" --learning-rate "${LEARNING_RATE}" \
  --checkpoint-interval "${CHECKPOINT_INTERVAL}" --patience "${PATIENCE}" \
  >"${RUN_ROOT}/logs/calibration_gt_only.out.log" \
  2>"${RUN_ROOT}/logs/calibration_gt_only.err.log" &
PID_GT_ONLY=$!
CUDA_VISIBLE_DEVICES="${GPU_LONG_SHORT}" "${PYTHON}" -m pre_experiments.long_short_camera_head.pipeline \
  --stage calibration "${COMMON[@]}" --variant long_short \
  --max-steps "${CALIBRATION_STEPS}" --learning-rate "${LEARNING_RATE}" \
  --checkpoint-interval "${CHECKPOINT_INTERVAL}" --patience "${PATIENCE}" \
  >"${RUN_ROOT}/logs/calibration_long_short.out.log" \
  2>"${RUN_ROOT}/logs/calibration_long_short.err.log" &
PID_LONG_SHORT=$!

set +e
wait "${PID_GT_ONLY}"
RC_GT_ONLY=$?
wait "${PID_LONG_SHORT}"
RC_LONG_SHORT=$?
set -e
[[ "${RC_GT_ONLY}" -eq 0 && "${RC_LONG_SHORT}" -eq 0 ]] || { echo "matched calibration failed" >&2; exit 25; }
[[ ! -s "${RUN_ROOT}/logs/calibration_gt_only.err.log" ]] || { echo "gt_only calibration wrote stderr" >&2; exit 26; }
[[ ! -s "${RUN_ROOT}/logs/calibration_long_short.err.log" ]] || { echo "long_short calibration wrote stderr" >&2; exit 27; }

run_stage "${GPU_GT_ONLY}" evaluate_gt_only --stage evaluate "${COMMON[@]}" --variant gt_only
run_stage "${GPU_LONG_SHORT}" evaluate_long_short --stage evaluate "${COMMON[@]}" --variant long_short

"${PYTHON}" -m pre_experiments.long_short_camera_head.pipeline \
  --stage report --run-root "${RUN_ROOT}" \
  >"${RUN_ROOT}/logs/report.out.log" \
  2>"${RUN_ROOT}/logs/report.err.log"
[[ ! -s "${RUN_ROOT}/logs/report.err.log" ]] || { echo "report wrote stderr" >&2; exit 28; }
"${PYTHON}" -m pre_experiments.long_short_camera_head.pipeline \
  --stage verify --run-root "${RUN_ROOT}" \
  >"${RUN_ROOT}/logs/verify.out.log" \
  2>"${RUN_ROOT}/logs/verify.err.log"
[[ ! -s "${RUN_ROOT}/logs/verify.err.log" ]] || { echo "verification wrote stderr" >&2; exit 29; }

echo "[long-short-camera-head] complete ${RUN_ROOT}"
