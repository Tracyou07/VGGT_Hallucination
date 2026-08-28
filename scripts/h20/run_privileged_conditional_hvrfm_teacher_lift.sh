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

EXPECTED_BRANCH="codex/privileged-conditional-hvrfm"
TEST_MODE="${HVRFM_RUNNER_TEST_MODE:-0}"
[[ "$TEST_MODE" == "0" || "$TEST_MODE" == "1" ]] || {
  printf 'HVRFM_RUNNER_TEST_MODE must be exactly 0 or 1.\n' >&2
  exit 3
}
if [[ "$TEST_MODE" == "1" ]]; then
  REPO_ROOT="${REPO_ROOT:?}"
  PYTHON="${PYTHON:?}"
  RESULT_ROOT="${RESULT_ROOT:-/data/yjh/output/vggt/privileged_conditional_hvrfm}"
  SOURCE_RUN="${SOURCE_RUN:?}"
  FORMAL_LABEL_ROOT="${FORMAL_LABEL_ROOT:?}"
  PREPARED_ROOT="${PREPARED_ROOT:?}"
  CHECKPOINT_DIR="${CHECKPOINT_DIR:?}"
  SCANNET_MARKER="${SCANNET_MARKER:?}"
else
  REPO_ROOT="/home/ubuntu/yjh/vggt/.worktrees/privileged_conditional_hvrfm"
  PYTHON="/home/ubuntu/anaconda3/envs/vggt-gx/bin/python"
  RESULT_ROOT="/data/yjh/output/vggt/privileged_conditional_hvrfm"
  SOURCE_RUN="/data/yjh/output/vggt/variational_camera_latent/vrfm_camera_20260827T044926Z"
  FORMAL_LABEL_ROOT="/data/yjh/output/vggt/long_short_camera_head/long_short_head_formal_20260828T072407Z"
  PREPARED_ROOT="/data/yjh/share/datasets/ScanNet/processed_cva02_v1"
  CHECKPOINT_DIR="/data/yjh/share/pretrained/VGGT-1B"
  SCANNET_MARKER="/data/yjh/share/datasets/ScanNet/verified_completion.json"
fi
VRFM_MARKER="${SOURCE_RUN}/verified_completion.json"
SOURCE_MANIFEST="${SOURCE_RUN}/manifests/source_manifest.json"
FORMAL_MARKER="${FORMAL_LABEL_ROOT}/verified_completion.json"
FORMAL_MANIFEST="${FORMAL_LABEL_ROOT}/manifests/data_manifest.json"
RUN_ID="${RUN_ID:-privileged_teacher_lift_$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="${RESULT_ROOT}/${RUN_ID}"
CONTROL_ROOT="${RESULT_ROOT}/.runner_control/${RUN_ID}"
LOG_ROOT="${CONTROL_ROOT}/logs"
MIN_FREE_BYTES=$((100 * 1024 * 1024 * 1024))
MAX_RUN_KIB=$((20 * 1024 * 1024))
PLANNED_STAGES=(preflight prepare smoke calibration report verify)

if [[ "$TEST_MODE" == "1" ]]; then
  EXPECTED_SCANNET_MARKER_SHA256="${HVRFM_TEST_SCANNET_MARKER_SHA256:?}"
  EXPECTED_VRFM_MARKER_SHA256="${HVRFM_TEST_VRFM_MARKER_SHA256:?}"
  EXPECTED_SOURCE_MANIFEST_SHA256="${HVRFM_TEST_SOURCE_MANIFEST_SHA256:?}"
  EXPECTED_FORMAL_MARKER_SHA256="${HVRFM_TEST_FORMAL_MARKER_SHA256:?}"
  EXPECTED_FORMAL_DATA_MANIFEST_SHA256="${HVRFM_TEST_FORMAL_DATA_MANIFEST_SHA256:?}"
  EXPECTED_CHECKPOINT_SHA256="${HVRFM_TEST_CHECKPOINT_SHA256:?}"
else
  EXPECTED_SCANNET_MARKER_SHA256="41ed5a72bdf74bdb93d55a87a48d51000a823d4545cb7025c53dbc6a59cc311e"
  EXPECTED_VRFM_MARKER_SHA256="fd1b93caa16f45f0dbdc55fd7000aba9ab8bf166a7240f5ac2a716a0b3de9a32"
  EXPECTED_SOURCE_MANIFEST_SHA256="be5aaa1b61be5e25709e40b3912e48aab38b6bbfac4be3b7ed183140219d6054"
  EXPECTED_FORMAL_MARKER_SHA256="4d24b944792f348ccc8c180a99f3e0ee11397ce472900eb6abe38f6924732667"
  EXPECTED_FORMAL_DATA_MANIFEST_SHA256="944ee57a75a68af45fc0ea6037070267552ea3f042bd2346638cdc65f2dd4a6e"
  EXPECTED_CHECKPOINT_SHA256="f164acf60724910d8fe1578bb499d800850c7bb0948db7555c413f9fbe60467e"
fi
for expected_sha256 in \
  "$EXPECTED_SCANNET_MARKER_SHA256" "$EXPECTED_VRFM_MARKER_SHA256" \
  "$EXPECTED_SOURCE_MANIFEST_SHA256" "$EXPECTED_FORMAL_MARKER_SHA256" \
  "$EXPECTED_FORMAL_DATA_MANIFEST_SHA256" "$EXPECTED_CHECKPOINT_SHA256"; do
  [[ "$expected_sha256" =~ ^[0-9a-f]{64}$ ]] || {
    printf 'Every expected fixture/content digest must be lowercase SHA-256.\n' >&2
    exit 3
  }
done

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
INITIAL_GIT_STATUS="$(git -C "$REPO_ROOT" status --short)" || {
  printf 'Could not inspect worktree cleanliness.\n' >&2
  exit 14
}
[[ -z "$INITIAL_GIT_STATUS" ]] || {
  printf 'Dirty worktree; refusing formal execution.\n' >&2
  exit 14
}
GIT_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD)"
[[ "$GIT_COMMIT" =~ ^[0-9a-f]{40}$ ]] || {
  printf 'Worktree HEAD is not a full Git commit.\n' >&2
  exit 15
}
require_git_state() {
  local branch head status
  branch="$(git -C "$REPO_ROOT" branch --show-current)" || {
    printf 'Could not recheck the worktree branch.\n' >&2
    return 1
  }
  [[ "$branch" == "$EXPECTED_BRANCH" ]] || {
    printf 'Worktree branch changed after preflight.\n' >&2
    return 1
  }
  head="$(git -C "$REPO_ROOT" rev-parse HEAD)" || {
    printf 'Could not recheck the worktree HEAD.\n' >&2
    return 1
  }
  [[ "$head" == "$GIT_COMMIT" ]] || {
    printf 'Worktree HEAD changed after preflight.\n' >&2
    return 1
  }
  status="$(git -C "$REPO_ROOT" status --short)" || {
    printf 'Could not recheck worktree cleanliness.\n' >&2
    return 1
  }
  [[ -z "$status" ]] || {
    printf 'Worktree became dirty after preflight.\n' >&2
    return 1
  }
}
[[ -x "$PYTHON" ]] || {
  printf 'Missing frozen VGGT Python environment: %s\n' "$PYTHON" >&2
  exit 16
}
[[ -f "$VRFM_MARKER" && -f "$SOURCE_MANIFEST" ]] || {
  printf 'Missing verified variational-camera source completion.\n' >&2
  exit 17
}
[[ -f "$FORMAL_MARKER" && -f "$FORMAL_MANIFEST" ]] || {
  printf 'Missing verified long-short formal completion.\n' >&2
  exit 18
}
[[ -d "$PREPARED_ROOT" ]] || {
  printf 'Missing prepared ScanNet root: %s\n' "$PREPARED_ROOT" >&2
  exit 19
}
if [[ -f "$CHECKPOINT_DIR/model.safetensors" ]]; then
  CHECKPOINT_FILE="$CHECKPOINT_DIR/model.safetensors"
elif [[ -f "$CHECKPOINT_DIR/model.pt" ]]; then
  CHECKPOINT_FILE="$CHECKPOINT_DIR/model.pt"
else
  printf 'Missing local VGGT checkpoint in %s.\n' "$CHECKPOINT_DIR" >&2
  exit 20
fi
[[ -f "$SCANNET_MARKER" ]] || {
  printf 'Missing verified ScanNet completion marker: %s\n' "$SCANNET_MARKER" >&2
  exit 21
}

command -v jq >/dev/null 2>&1 && command -v sha256sum >/dev/null 2>&1 \
  && command -v realpath >/dev/null 2>&1 && command -v flock >/dev/null 2>&1 || {
  printf 'jq, sha256sum, realpath, and flock are required for authenticated preflight.\n' >&2
  exit 21
}

reject_symlink_components() {
  local path="$1"
  local label="$2"
  local lexical resolved
  lexical="$(realpath -m -s -- "$path")" || {
    printf 'Could not normalize %s.\n' "$label" >&2
    return 1
  }
  resolved="$(realpath -m -- "$path")" || {
    printf 'Could not resolve %s.\n' "$label" >&2
    return 1
  }
  [[ "$lexical" == "$resolved" ]] || {
    printf '%s contains a symlink or junction component.\n' "$label" >&2
    return 1
  }
}

require_child_of_result_root() {
  local path="$1"
  local label="$2"
  local result_resolved child_resolved
  result_resolved="$(realpath -m -- "$RESULT_ROOT")" || return 1
  child_resolved="$(realpath -m -- "$path")" || return 1
  case "$child_resolved" in
    "$result_resolved"/*) ;;
    *)
      printf '%s escapes the resolved result root.\n' "$label" >&2
      return 1
      ;;
  esac
}

validate_output_paths() {
  local gpu_lock_path="${RESULT_ROOT}/.runner_control/gpu_${GPU_INDEX:-pending}.lock"
  reject_symlink_components "$RESULT_ROOT" "Result root" || return 1
  reject_symlink_components "${RESULT_ROOT}/.runner_control" "Control parent" || return 1
  reject_symlink_components "$RUN_ROOT" "Run root" || return 1
  reject_symlink_components "$CONTROL_ROOT" "Control root" || return 1
  reject_symlink_components "$LOG_ROOT" "Log root" || return 1
  reject_symlink_components "$CONTROL_ROOT/run.lock" "Run lock" || return 1
  reject_symlink_components "$gpu_lock_path" "GPU lock" || return 1
  require_child_of_result_root "$RUN_ROOT" "Run root" || return 1
  require_child_of_result_root "$CONTROL_ROOT" "Control root" || return 1
  require_child_of_result_root "$LOG_ROOT" "Log root" || return 1
  require_child_of_result_root "$CONTROL_ROOT/run.lock" "Run lock" || return 1
  require_child_of_result_root "$gpu_lock_path" "GPU lock" || return 1
}

sha256_file() {
  sha256sum -- "$1" | awk 'NR == 1 {print $1}'
}

require_file_sha256() {
  local path="$1"
  local expected="$2"
  local label="$3"
  local observed
  observed="$(sha256_file "$path")"
  [[ "$observed" == "$expected" ]] || {
    printf '%s SHA-256 mismatch.\n' "$label" >&2
    return 1
  }
}

authenticate_inputs() {
  [[ -f "$SCANNET_MARKER" && -f "$VRFM_MARKER" && -f "$SOURCE_MANIFEST" ]] || {
    printf 'Authenticated ScanNet/VRFM inputs disappeared.\n' >&2
    return 1
  }
  [[ -f "$FORMAL_MARKER" && -f "$FORMAL_MANIFEST" && -f "$CHECKPOINT_FILE" ]] || {
    printf 'Authenticated formal/checkpoint inputs disappeared.\n' >&2
    return 1
  }
  require_file_sha256 "$SCANNET_MARKER" "$EXPECTED_SCANNET_MARKER_SHA256" "ScanNet marker" || return 1
  jq -e \
    --arg schema "camera_solution_space_01.scannet50_verified_completion.v1" \
    --argjson scene_count 50 --argjson asset_count 100 \
    --argjson total_bytes 37587327416 \
    'type == "object" and .schema == $schema and .scene_count == $scene_count and .asset_count == $asset_count and .total_bytes == $total_bytes' \
    "$SCANNET_MARKER" >/dev/null || {
      printf 'ScanNet marker schema or identity mismatch.\n' >&2
      return 1
    }

  require_file_sha256 "$VRFM_MARKER" "$EXPECTED_VRFM_MARKER_SHA256" "VRFM marker" || return 1
  jq -e \
    --arg schema "variational_camera_latent.verified_completion.v1" \
    --argjson scene_count 10 \
    --arg completion_digest "3fdc97395eef8261ad7eaa055aec0bd441cf8d43fee9847464f190e269ab474e" \
    'type == "object" and .schema == $schema and .scene_count == $scene_count and .completion_digest == $completion_digest' \
    "$VRFM_MARKER" >/dev/null || {
      printf 'VRFM marker schema or identity mismatch.\n' >&2
      return 1
    }
  require_file_sha256 "$SOURCE_MANIFEST" "$EXPECTED_SOURCE_MANIFEST_SHA256" "VRFM source manifest" || return 1
  jq -e \
    --arg schema "variational_camera_latent.source.v1" --argjson records_length 10 \
    'type == "object" and .schema == $schema and (.records | type == "array" and length == $records_length)' \
    "$SOURCE_MANIFEST" >/dev/null || {
      printf 'VRFM source manifest schema or cohort mismatch.\n' >&2
      return 1
    }

  require_file_sha256 "$FORMAL_MANIFEST" "$EXPECTED_FORMAL_DATA_MANIFEST_SHA256" "formal data manifest" || return 1
  jq -e \
    --arg schema "long_short_camera_head.data_manifest.v1" \
    --arg git_revision "2476a59f583ce4c39bbe66dc65d6a8e5cddfb52e" \
    --arg source_run "$SOURCE_RUN" \
    --arg source_manifest_sha256 "$EXPECTED_SOURCE_MANIFEST_SHA256" \
    --arg prepared_root "$PREPARED_ROOT" --arg checkpoint_dir "$CHECKPOINT_DIR" \
    --arg base_checkpoint_sha256 "$EXPECTED_CHECKPOINT_SHA256" \
    --argjson records_length 10 \
    'type == "object" and .schema == $schema and .git_revision == $git_revision and .source_run == $source_run and .source_manifest_sha256 == $source_manifest_sha256 and .prepared_root == $prepared_root and .checkpoint_dir == $checkpoint_dir and .base_checkpoint_sha256 == $base_checkpoint_sha256 and (.records | type == "array" and length == $records_length)' \
    "$FORMAL_MANIFEST" >/dev/null || {
      printf 'Formal data manifest schema or provenance mismatch.\n' >&2
      return 1
    }

  require_file_sha256 "$FORMAL_MARKER" "$EXPECTED_FORMAL_MARKER_SHA256" "formal marker" || return 1
  jq -e \
    --arg schema "long_short_camera_head.verified_completion.v1" \
    --arg git_revision "2476a59f583ce4c39bbe66dc65d6a8e5cddfb52e" \
    --arg classification "NO_SOURCE_HEAD_SIGNAL" \
    --arg source_manifest_sha256 "$EXPECTED_SOURCE_MANIFEST_SHA256" \
    --arg data_manifest_sha256 "$EXPECTED_FORMAL_DATA_MANIFEST_SHA256" \
    --arg base_checkpoint_sha256 "$EXPECTED_CHECKPOINT_SHA256" \
    --argjson scene_count 10 --argjson train_scene_count 8 \
    --argjson locked_replay_scene_count 2 --argjson inference_leakage_audit true \
    'type == "object" and .schema == $schema and .git_revision == $git_revision and .classification == $classification and .source_manifest_sha256 == $source_manifest_sha256 and .data_manifest_sha256 == $data_manifest_sha256 and .base_checkpoint_sha256 == $base_checkpoint_sha256 and .scene_count == $scene_count and .train_scene_count == $train_scene_count and .locked_replay_scene_count == $locked_replay_scene_count and .inference_leakage_audit == $inference_leakage_audit' \
    "$FORMAL_MARKER" >/dev/null || {
      printf 'Formal marker schema or provenance mismatch.\n' >&2
      return 1
    }
  require_file_sha256 "$CHECKPOINT_FILE" "$EXPECTED_CHECKPOINT_SHA256" "VGGT checkpoint" || return 1
}

require_free_space() {
  local free_bytes
  free_bytes="$(df --output=avail -B1 /data | awk 'NR == 2 {print $1}')"
  [[ "$free_bytes" =~ ^[0-9]+$ ]] || {
    printf 'Could not determine free bytes on /data.\n' >&2
    exit 22
  }
  (( free_bytes >= MIN_FREE_BYTES )) || {
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
    if [[ -z "$pids" ]]; then
      printf '%s\n' "$index"
      return 0
    fi
    if [[ ! "$pids" =~ ^[0-9]+([[:space:]]+[0-9]+)*$ ]]; then
      printf 'GPU %s process inventory is not a strict PID list or empty output.\n' "$index" >&2
      return 1
    fi
  done <<< "$rows"
  if (( h20_count > 0 )); then
    printf 'Every NVIDIA H20 has an active compute process.\n' >&2
  else
    printf 'No NVIDIA H20 GPU was found.\n' >&2
  fi
  return 1
}

require_selected_gpu_idle() {
  local pids
  if ! pids="$(nvidia-smi -i "$GPU_INDEX" --query-compute-apps=pid --format=csv,noheader,nounits)"; then
    printf 'Could not recheck compute processes on GPU %s.\n' "$GPU_INDEX" >&2
    return 1
  fi
  [[ -z "$pids" ]] || {
    printf 'GPU %s is no longer strictly idle; refusing the next stage.\n' "$GPU_INDEX" >&2
    return 1
  }
}

require_free_space
authenticate_inputs
GPU_INDEX="$(select_idle_h20)"
validate_output_paths
require_run_size

if (( PRELIGHT_ONLY == 1 )); then
  jq -cn \
    --arg result_root "$RESULT_ROOT" \
    --arg gpu_index "$GPU_INDEX" \
    --arg git_commit "$GIT_COMMIT" \
    '{result_root:$result_root,planned_stages:["preflight","prepare","smoke","calibration","report","verify"],gpu_index:$gpu_index,git_commit:$git_commit}'
  exit 0
fi

mkdir -p "$RUN_ROOT" "$LOG_ROOT"
exec 9>"$CONTROL_ROOT/run.lock"
flock -n 9 || {
  printf 'Run ID %s is already active.\n' "$RUN_ID" >&2
  exit 24
}
GPU_LOCK_PATH="${RESULT_ROOT}/.runner_control/gpu_${GPU_INDEX}.lock"
exec 8>"$GPU_LOCK_PATH"
flock -n 8 || {
  printf 'GPU %s runner lock is already held.\n' "$GPU_INDEX" >&2
  exit 24
}
release_locks() {
  flock -u 8 >/dev/null 2>&1 || true
  flock -u 9 >/dev/null 2>&1 || true
}
trap release_locks EXIT

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
  validate_output_paths
  require_git_state
  require_free_space
  authenticate_inputs
  require_selected_gpu_idle
  require_run_size
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
