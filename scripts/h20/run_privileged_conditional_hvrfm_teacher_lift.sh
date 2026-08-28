#!/usr/bin/env bash
set -Eeuo pipefail

PRELIGHT_ONLY=0
if (( $# > 1 )); then
  printf 'The only optional argument is --preflight-only.\n' >&2
  exit 2
fi
if (( $# == 1 )); then
  [[ "$1" == "--preflight-only" ]] || {
    printf 'The only optional argument is --preflight-only.\n' >&2
    exit 2
  }
  PRELIGHT_ONLY=1
fi

REPO_ROOT="${REPO_ROOT:-/home/ubuntu/yjh/vggt/.worktrees/privileged_conditional_hvrfm}"
EXPECTED_BRANCH="codex/privileged-conditional-hvrfm"
PYTHON="${PYTHON:-/home/ubuntu/anaconda3/envs/vggt-gx/bin/python}"
RESULT_ROOT="${RESULT_ROOT:-/data/yjh/output/vggt/privileged_conditional_hvrfm}"
SOURCE_RUN="${SOURCE_RUN:-/data/yjh/output/vggt/variational_camera_latent/vrfm_camera_20260827T044926Z}"
FORMAL_LABEL_ROOT="${FORMAL_LABEL_ROOT:-/data/yjh/output/vggt/long_short_camera_head/long_short_head_formal_20260828T072407Z}"
PREPARED_ROOT="${PREPARED_ROOT:-/data/yjh/share/datasets/ScanNet/processed_cva02_v1}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/data/yjh/share/pretrained/VGGT-1B}"
SCANNET_MARKER="${SCANNET_MARKER:-/data/yjh/share/datasets/ScanNet/verified_completion.json}"
RUN_ID="${RUN_ID:-privileged_teacher_lift_$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="${RESULT_ROOT}/${RUN_ID}"
CONTROL_ROOT="${RESULT_ROOT}/.runner_control/${RUN_ID}"
LOG_ROOT="${CONTROL_ROOT}/logs"
MIN_FREE_GIB=100
MAX_RUN_KIB=$((20 * 1024 * 1024))
PLANNED_STAGES=(preflight prepare smoke calibration report verify)

[[ "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || {
  printf 'RUN_ID must be one path-safe identifier.\n' >&2
  exit 3
}
[[ "$(hostname)" == "VM-0-11-ubuntu" ]] || {
  printf 'H20 host identity mismatch; expected VM-0-11-ubuntu.\n' >&2
  exit 10
}
[[ "$(id -un)" == "ubuntu" ]] || {
  printf 'H20 user identity mismatch; expected ubuntu.\n' >&2
  exit 11
}
[[ -d "$REPO_ROOT" && ( -d "$REPO_ROOT/.git" || -f "$REPO_ROOT/.git" ) ]] || {
  printf 'Missing dedicated H20 worktree: %s\n' "$REPO_ROOT" >&2
  exit 12
}
[[ "$(git -C "$REPO_ROOT" branch --show-current)" == "$EXPECTED_BRANCH" ]] || {
  printf 'Wrong branch; expected %s.\n' "$EXPECTED_BRANCH" >&2
  exit 13
}
[[ -z "$(git -C "$REPO_ROOT" status --short)" ]] || {
  printf 'Dirty worktree; refusing formal execution.\n' >&2
  exit 14
}
GIT_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD)"
[[ "$GIT_COMMIT" =~ ^[0-9a-f]{40}$ ]] || {
  printf 'Worktree HEAD is not a full Git commit.\n' >&2
  exit 15
}
[[ -x "$PYTHON" ]] || {
  printf 'Missing frozen VGGT Python environment: %s\n' "$PYTHON" >&2
  exit 16
}
[[ -f "$SOURCE_RUN/verified_completion.json" ]] || {
  printf 'Missing verified variational-camera source completion.\n' >&2
  exit 17
}
[[ -f "$FORMAL_LABEL_ROOT/verified_completion.json" ]] || {
  printf 'Missing verified long-short formal completion.\n' >&2
  exit 18
}
[[ -d "$PREPARED_ROOT" ]] || {
  printf 'Missing prepared ScanNet root: %s\n' "$PREPARED_ROOT" >&2
  exit 19
}
[[ -f "$CHECKPOINT_DIR/model.safetensors" || -f "$CHECKPOINT_DIR/model.pt" ]] || {
  printf 'Missing local VGGT checkpoint in %s.\n' "$CHECKPOINT_DIR" >&2
  exit 20
}
[[ -f "$SCANNET_MARKER" ]] || {
  printf 'Missing verified ScanNet completion marker: %s\n' "$SCANNET_MARKER" >&2
  exit 21
}

require_free_space() {
  local free_gib
  free_gib="$(df --output=avail -BG /data | tail -n 1 | tr -dc '0-9')"
  [[ "$free_gib" =~ ^[0-9]+$ ]] || {
    printf 'Could not determine free GiB on /data.\n' >&2
    exit 22
  }
  (( free_gib >= MIN_FREE_GIB )) || {
    printf 'At least 100 GiB free on /data is required.\n' >&2
    exit 22
  }
}

require_run_size() {
  [[ ! -e "$RUN_ROOT" ]] || [[ ! -L "$RUN_ROOT" ]] || {
    printf 'Run root may not be a symlink.\n' >&2
    exit 23
  }
  [[ -e "$RUN_ROOT" ]] || return 0
  local run_kib
  run_kib="$(du -sk -- "$RUN_ROOT" | awk 'NR == 1 {print $1}')"
  [[ "$run_kib" =~ ^[0-9]+$ ]] || {
    printf 'Could not determine run-root size.\n' >&2
    exit 23
  }
  (( run_kib < MAX_RUN_KIB )) || {
    printf 'The run root reached 20 GiB; refusing to continue.\n' >&2
    exit 23
  }
}

select_idle_h20() {
  local rows raw_index raw_name index name pids
  local h20_count=0
  if ! rows="$(nvidia-smi --query-gpu=index,name --format=csv,noheader,nounits)"; then
    printf 'Could not inventory H20 GPUs.\n' >&2
    return 1
  fi
  while IFS=',' read -r raw_index raw_name; do
    index="${raw_index#"${raw_index%%[![:space:]]*}"}"
    index="${index%"${index##*[![:space:]]}"}"
    name="${raw_name#"${raw_name%%[![:space:]]*}"}"
    name="${name%"${name##*[![:space:]]}"}"
    [[ "$index" =~ ^[0-9]+$ && "$name" == "NVIDIA H20" ]] || continue
    h20_count=$((h20_count + 1))
    if ! pids="$(nvidia-smi -i "$index" --query-compute-apps=pid --format=csv,noheader,nounits)"; then
      printf 'Could not inspect compute processes on GPU %s.\n' "$index" >&2
      return 1
    fi
    if ! grep -Eq '[0-9]+' <<< "$pids"; then
      printf '%s\n' "$index"
      return 0
    fi
  done <<< "$rows"
  if (( h20_count > 0 )); then
    printf 'Every NVIDIA H20 has an active compute process.\n' >&2
  else
    printf 'No NVIDIA H20 GPU was found.\n' >&2
  fi
  return 1
}

require_free_space
GPU_INDEX="$(select_idle_h20)"
require_run_size

if (( PRELIGHT_ONLY == 1 )); then
  printf '{"result_root":"%s","planned_stages":["preflight","prepare","smoke","calibration","report","verify"],"gpu_index":"%s","git_commit":"%s"}\n' \
    "$RESULT_ROOT" "$GPU_INDEX" "$GIT_COMMIT"
  exit 0
fi

mkdir -p "$RUN_ROOT" "$LOG_ROOT"
exec 9>"$CONTROL_ROOT/run.lock"
flock -n 9 || {
  printf 'Run ID %s is already active.\n' "$RUN_ID" >&2
  exit 24
}

unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HF_HUB_TOKEN
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
cd "$REPO_ROOT"

run_stage() {
  local stage="$1"
  shift
  local stdout_log="$LOG_ROOT/${stage}.out.log"
  local stderr_log="$LOG_ROOT/${stage}.err.log"
  local return_code=0
  require_free_space
  CUDA_VISIBLE_DEVICES="$GPU_INDEX" "$PYTHON" \
    -m pre_experiments.conditional_hierarchical_vrfm.pipeline \
    "$stage" "$@" >"$stdout_log" 2>"$stderr_log" || return_code=$?
  if (( return_code != 0 )); then
    printf '%s stage failed with exit code %s; run artifacts were preserved.\n' \
      "$stage" "$return_code" >&2
    exit 25
  fi
  [[ ! -s "$stderr_log" ]] || {
    printf '%s wrote stderr; refusing to continue.\n' "$stage" >&2
    exit 26
  }
  require_free_space
  require_run_size
}

COMMON_ARGS=(
  --run-root "$RUN_ROOT"
  --git-commit "$GIT_COMMIT"
  --checkpoint-dir "$CHECKPOINT_DIR"
  --device cuda
)

run_stage preflight "${COMMON_ARGS[@]}"
run_stage prepare "${COMMON_ARGS[@]}" \
  --source-run "$SOURCE_RUN" \
  --formal-label-root "$FORMAL_LABEL_ROOT" \
  --prepared-root "$PREPARED_ROOT"
run_stage smoke "${COMMON_ARGS[@]}"
run_stage calibration "${COMMON_ARGS[@]}"
run_stage report "${COMMON_ARGS[@]}"
run_stage verify "${COMMON_ARGS[@]}"

[[ -f "$RUN_ROOT/verified_completion.json" ]] || {
  printf 'Verify stage did not publish verified_completion.json.\n' >&2
  exit 27
}
printf '[privileged-teacher-lift] verified %s\n' "$RUN_ROOT"
