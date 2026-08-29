#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  printf '%s\n' \
    'Usage: run_camera_translation_hvrfm_targets.sh --expected-git-commit <40hex> --expected-python-sha256 <64hex> [--preflight-only]' \
    'Locks are create-new files. A crash intentionally leaves a stale lock; audit its PID, path, and inode before manual removal.' >&2
}

PREFLIGHT_ONLY=0
EXPECTED_GIT_COMMIT=""
EXPECTED_PYTHON_SHA256=""
while (( $# )); do
  case "$1" in
    --expected-git-commit)
      [[ -z "$EXPECTED_GIT_COMMIT" && $# -ge 2 ]] || { usage; exit 2; }
      EXPECTED_GIT_COMMIT="$2"
      shift 2
      ;;
    --expected-python-sha256)
      [[ -z "$EXPECTED_PYTHON_SHA256" && $# -ge 2 ]] || { usage; exit 2; }
      EXPECTED_PYTHON_SHA256="$2"
      shift 2
      ;;
    --preflight-only)
      (( PREFLIGHT_ONLY == 0 )) || { usage; exit 2; }
      PREFLIGHT_ONLY=1
      shift
      ;;
    *)
      usage
      exit 2
      ;;
  esac
done
[[ "$EXPECTED_GIT_COMMIT" =~ ^[0-9a-f]{40}$ ]] || {
  printf 'A preverified --expected-git-commit full 40-hex commit is required.\n' >&2
  exit 2
}
[[ "$EXPECTED_PYTHON_SHA256" =~ ^[0-9a-f]{64}$ ]] || {
  printf 'A preverified --expected-python-sha256 lowercase SHA-256 is required.\n' >&2
  exit 2
}

SECRET_VARIABLES=(
  HF_TOKEN
  HUGGING_FACE_HUB_TOKEN
  HF_HUB_TOKEN
  HUGGINGFACE_TOKEN
)
for secret_variable in "${SECRET_VARIABLES[@]}"; do
  [[ -z "${!secret_variable:-}" ]] || {
    printf 'Credential environment variable %s must not be present.\n' \
      "$secret_variable" >&2
    exit 3
  }
done
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HF_HUB_TOKEN HUGGINGFACE_TOKEN

EXPECTED_BRANCH="codex/privileged-conditional-hvrfm"
TEST_MODE="${CAMERA_TRANSLATION_HVRFM_RUNNER_TEST_MODE:-0}"
[[ "$TEST_MODE" == "0" || "$TEST_MODE" == "1" ]] || {
  printf 'CAMERA_TRANSLATION_HVRFM_RUNNER_TEST_MODE must be exactly 0 or 1.\n' >&2
  exit 3
}

if [[ "$TEST_MODE" == "1" ]]; then
  REPO_ROOT="${REPO_ROOT:?}"
  PYTHON="${PYTHON:?}"
  EXPECTED_PYTHON_REALPATH="${CTHVRFM_TEST_PYTHON_REALPATH:?}"
  RESULT_ROOT="${RESULT_ROOT:?}"
  SOURCE_RUN="${SOURCE_RUN:?}"
  REFERENCE_RUN="${REFERENCE_RUN:?}"
  FORMAL_RUN="${FORMAL_RUN:?}"
  CHECKPOINT_DIR="${CHECKPOINT_DIR:?}"
  EXPECTED_SOURCE_COMPLETION_SHA256="${CTHVRFM_TEST_SOURCE_COMPLETION_SHA256:?}"
  EXPECTED_SOURCE_MANIFEST_SHA256="${CTHVRFM_TEST_SOURCE_MANIFEST_SHA256:?}"
  EXPECTED_REFERENCE_COMPLETION_SHA256="${CTHVRFM_TEST_REFERENCE_COMPLETION_SHA256:?}"
  EXPECTED_REFERENCE_INVENTORY_SHA256="${CTHVRFM_TEST_REFERENCE_INVENTORY_SHA256:?}"
  EXPECTED_REFERENCE_CONFIG_SHA256="${CTHVRFM_TEST_REFERENCE_CONFIG_SHA256:?}"
  EXPECTED_REFERENCE_REPORT_SHA256="${CTHVRFM_TEST_REFERENCE_REPORT_SHA256:?}"
  EXPECTED_REFERENCE_LONG_MANIFEST_SHA256="${CTHVRFM_TEST_REFERENCE_LONG_MANIFEST_SHA256:?}"
  EXPECTED_REFERENCE_TEACHER_MANIFEST_SHA256="${CTHVRFM_TEST_REFERENCE_TEACHER_MANIFEST_SHA256:?}"
  EXPECTED_FORMAL_COMPLETION_SHA256="${CTHVRFM_TEST_FORMAL_COMPLETION_SHA256:?}"
  EXPECTED_FORMAL_MANIFEST_SHA256="${CTHVRFM_TEST_FORMAL_MANIFEST_SHA256:?}"
  EXPECTED_CHECKPOINT_SHA256="${CTHVRFM_TEST_CHECKPOINT_SHA256:?}"
else
  REPO_ROOT="/home/ubuntu/yjh/vggt/.worktrees/privileged_conditional_hvrfm"
  PYTHON="/home/ubuntu/anaconda3/envs/vggt-gx/bin/python"
  EXPECTED_PYTHON_REALPATH="/home/ubuntu/anaconda3/envs/vggt-gx/bin/python3.10"
  RESULT_ROOT="/data/yjh/output/vggt/camera_translation_hvrfm"
  SOURCE_RUN="/data/yjh/output/vggt/variational_camera_latent/vrfm_camera_20260827T044926Z"
  REFERENCE_RUN="/data/yjh/output/vggt/privileged_conditional_hvrfm/privileged_teacher_lift_20260829T012716Z_tolfix"
  FORMAL_RUN="/data/yjh/output/vggt/long_short_camera_head/long_short_head_formal_20260828T072407Z"
  CHECKPOINT_DIR="/data/yjh/share/pretrained/VGGT-1B"
  EXPECTED_SOURCE_COMPLETION_SHA256="fd1b93caa16f45f0dbdc55fd7000aba9ab8bf166a7240f5ac2a716a0b3de9a32"
  EXPECTED_SOURCE_MANIFEST_SHA256="be5aaa1b61be5e25709e40b3912e48aab38b6bbfac4be3b7ed183140219d6054"
  EXPECTED_REFERENCE_COMPLETION_SHA256="7e63ca36e6fc4c08772e3356255f84c2853c9d46310ae546cc5e53dc1792048c"
  EXPECTED_REFERENCE_INVENTORY_SHA256="046cf50cc7c7610a24d9f02571f7f0c438c79e43e89becf972e5d8594c465309"
  EXPECTED_REFERENCE_CONFIG_SHA256="525333c71cc6e94300591def1191c9c02294380ecf77055e6cf44ea2028c6b5f"
  EXPECTED_REFERENCE_REPORT_SHA256="5e0aedb1411c94ab839a7287750fa947731dbd4f10bfd9b4c89f8571a2474efc"
  EXPECTED_REFERENCE_LONG_MANIFEST_SHA256="6b6ab434bb4cd8bd4afbeaf8a2d11354f321d8791501ada3ef2f9376eb064166"
  EXPECTED_REFERENCE_TEACHER_MANIFEST_SHA256="d4c113515a72a2d79cd5e2f5139e290a787dc4d438c7caa22d9725d8fd99691e"
  EXPECTED_FORMAL_COMPLETION_SHA256="4d24b944792f348ccc8c180a99f3e0ee11397ce472900eb6abe38f6924732667"
  EXPECTED_FORMAL_MANIFEST_SHA256="944ee57a75a68af45fc0ea6037070267552ea3f042bd2346638cdc65f2dd4a6e"
  EXPECTED_CHECKPOINT_SHA256="f164acf60724910d8fe1578bb499d800850c7bb0948db7555c413f9fbe60467e"
fi

if [[ "$TEST_MODE" == "1" ]]; then
  RUN_ID="${RUN_ID:-camera_translation_targets_fixture}"
else
  RUN_ID="camera_translation_targets_$(date -u +%Y%m%dT%H%M%SZ)_$$_${RANDOM}"
fi

SOURCE_COMPLETION="${SOURCE_RUN}/verified_completion.json"
SOURCE_MANIFEST="${SOURCE_RUN}/manifests/source_manifest.json"
REFERENCE_COMPLETION="${REFERENCE_RUN}/verified_completion.json"
REFERENCE_INVENTORY="${REFERENCE_RUN}/manifests/verification_inventory.json"
REFERENCE_CONFIG="${REFERENCE_RUN}/config.json"
REFERENCE_REPORT="${REFERENCE_RUN}/reports/stage_a.json"
REFERENCE_LONG_MANIFEST="${REFERENCE_RUN}/manifests/long_context.json"
REFERENCE_TEACHER_MANIFEST="${REFERENCE_RUN}/manifests/teacher.json"
FORMAL_COMPLETION="${FORMAL_RUN}/verified_completion.json"
FORMAL_MANIFEST="${FORMAL_RUN}/manifests/data_manifest.json"
CHECKPOINT_FILE="${CHECKPOINT_DIR}/model.safetensors"
RUN_ROOT="${RESULT_ROOT}/${RUN_ID}"
CONTROL_PARENT="${RESULT_ROOT}/.runner_control"
CONTROL_ROOT="${CONTROL_PARENT}/${RUN_ID}"
LOG_ROOT="${CONTROL_ROOT}/logs"
MIN_FREE_BYTES=$((100 * 1024 * 1024 * 1024))
MAX_RUN_KIB=$((20 * 1024 * 1024))
PLANNED_STAGES=(preflight prepare smoke calibration report verify)

for expected_sha256 in \
  "$EXPECTED_SOURCE_COMPLETION_SHA256" \
  "$EXPECTED_SOURCE_MANIFEST_SHA256" \
  "$EXPECTED_REFERENCE_COMPLETION_SHA256" \
  "$EXPECTED_REFERENCE_INVENTORY_SHA256" \
  "$EXPECTED_REFERENCE_CONFIG_SHA256" \
  "$EXPECTED_REFERENCE_REPORT_SHA256" \
  "$EXPECTED_REFERENCE_LONG_MANIFEST_SHA256" \
  "$EXPECTED_REFERENCE_TEACHER_MANIFEST_SHA256" \
  "$EXPECTED_FORMAL_COMPLETION_SHA256" \
  "$EXPECTED_FORMAL_MANIFEST_SHA256" \
  "$EXPECTED_CHECKPOINT_SHA256" \
  "$EXPECTED_PYTHON_SHA256"; do
  [[ "$expected_sha256" =~ ^[0-9a-f]{64}$ ]] || {
    printf 'Every frozen input digest must be lowercase SHA-256.\n' >&2
    exit 3
  }
done

[[ "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] || {
  printf 'RUN_ID must be one path-safe identifier.\n' >&2
  exit 3
}
[[ "$RUN_ID" != ".runner_control" ]] || {
  printf 'RUN_ID collides with the runner control directory.\n' >&2
  exit 3
}

for required_command in jq sha256sum realpath stat flock nvidia-smi git df du awk env; do
  command -v "$required_command" >/dev/null 2>&1 || {
    printf 'Required command %s is unavailable.\n' "$required_command" >&2
    exit 4
  }
done
[[ -x /usr/bin/stat && -x /usr/bin/rm ]] || {
  printf 'Pinned lock-cleanup stat/rm commands are unavailable.\n' >&2
  exit 4
}
if [[ "$TEST_MODE" == "0" && ! -x /usr/bin/flock ]]; then
  printf 'Pinned lock-cleanup flock command is unavailable.\n' >&2
  exit 4
fi

require_system_identity() {
  [[ "$(hostname)" == "VM-0-11-ubuntu" ]] || {
    printf 'H20 host identity mismatch; expected VM-0-11-ubuntu.\n' >&2
    return 1
  }
  [[ "$(id -un)" == "ubuntu" ]] || {
    printf 'H20 user identity mismatch; expected ubuntu.\n' >&2
    return 1
  }
}

require_initial_git_state() {
  [[ -d "$REPO_ROOT" && ( -d "$REPO_ROOT/.git" || -f "$REPO_ROOT/.git" ) ]] || {
    printf 'Missing dedicated H20 worktree: %s\n' "$REPO_ROOT" >&2
    return 1
  }
  [[ "$(git -C "$REPO_ROOT" branch --show-current)" == "$EXPECTED_BRANCH" ]] || {
    printf 'Wrong branch; expected %s.\n' "$EXPECTED_BRANCH" >&2
    return 1
  }
  local status
  status="$(git -C "$REPO_ROOT" status --short)" || {
    printf 'Could not inspect worktree cleanliness.\n' >&2
    return 1
  }
  [[ -z "$status" ]] || {
    printf 'Dirty worktree; refusing formal execution.\n' >&2
    return 1
  }
  GIT_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD)" || {
    printf 'Could not inspect worktree HEAD.\n' >&2
    return 1
  }
  [[ "$GIT_COMMIT" =~ ^[0-9a-f]{40}$ ]] || {
    printf 'Worktree HEAD is not a full Git commit.\n' >&2
    return 1
  }
  [[ "$GIT_COMMIT" == "$EXPECTED_GIT_COMMIT" ]] || {
    printf 'Worktree HEAD does not match the preverified expected Git commit.\n' >&2
    return 1
  }
}

require_git_state() {
  local branch head status
  branch="$(git -C "$REPO_ROOT" branch --show-current)" || {
    printf 'Could not recheck the worktree branch.\n' >&2
    return 1
  }
  [[ "$branch" == "$EXPECTED_BRANCH" ]] || {
    printf 'Worktree branch changed after shell preflight.\n' >&2
    return 1
  }
  head="$(git -C "$REPO_ROOT" rev-parse HEAD)" || {
    printf 'Could not recheck the worktree HEAD.\n' >&2
    return 1
  }
  [[ "$head" == "$GIT_COMMIT" && "$head" == "$EXPECTED_GIT_COMMIT" ]] || {
    printf 'Worktree HEAD changed or differs from the expected Git commit.\n' >&2
    return 1
  }
  status="$(git -C "$REPO_ROOT" status --short)" || {
    printf 'Could not recheck worktree cleanliness.\n' >&2
    return 1
  }
  [[ -z "$status" ]] || {
    printf 'Worktree became dirty after shell preflight.\n' >&2
    return 1
  }
}

reject_symlink_components() {
  local path="$1" label="$2" lexical resolved
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

require_child_of() {
  local parent="$1" child="$2" label="$3" parent_resolved child_resolved
  parent_resolved="$(realpath -m -- "$parent")" || return 1
  child_resolved="$(realpath -m -- "$child")" || return 1
  case "$child_resolved" in
    "$parent_resolved"/*) ;;
    *)
      printf '%s escapes its expected parent.\n' "$label" >&2
      return 1
      ;;
  esac
}

paths_overlap() {
  local first second
  first="$(realpath -m -- "$1")" || return 2
  second="$(realpath -m -- "$2")" || return 2
  [[ "$first" == "$second" || "$first" == "$second"/* || "$second" == "$first"/* ]]
}

require_tree_isolation() {
  local output input left right
  local raw=(
    "$RESULT_ROOT" "$RUN_ROOT" "$CONTROL_ROOT"
    "$REPO_ROOT" "$SOURCE_RUN" "$REFERENCE_RUN" "$FORMAL_RUN" "$CHECKPOINT_DIR"
  )
  local normalized=()
  mapfile -t normalized < <(realpath -m -- "${raw[@]}")
  (( ${#normalized[@]} == ${#raw[@]} )) || return 1
  local outputs=("${normalized[@]:0:3}")
  local inputs=("${normalized[@]:3:5}")
  for output in "${outputs[@]}"; do
    for input in "${inputs[@]}"; do
      if [[ "$output" == "$input" || "$output" == "$input"/* || "$input" == "$output"/* ]]; then
        printf 'Output and frozen input trees overlap.\n' >&2
        return 1
      fi
    done
  done
  for (( left = 0; left < ${#inputs[@]}; left++ )); do
    for (( right = left + 1; right < ${#inputs[@]}; right++ )); do
      if [[ "${inputs[left]}" == "${inputs[right]}" \
        || "${inputs[left]}" == "${inputs[right]}"/* \
        || "${inputs[right]}" == "${inputs[left]}"/* ]]; then
        printf 'Frozen input trees overlap.\n' >&2
        return 1
      fi
    done
  done
  if [[ "${outputs[1]}" == "${outputs[2]}" \
    || "${outputs[1]}" == "${outputs[2]}"/* \
    || "${outputs[2]}" == "${outputs[1]}"/* ]]; then
    printf 'Run and control trees overlap.\n' >&2
    return 1
  fi
}

validate_paths() {
  local lexical resolved
  local paths=(
    "$REPO_ROOT"
    "$SOURCE_RUN" "$SOURCE_RUN/manifests" "$SOURCE_COMPLETION" "$SOURCE_MANIFEST"
    "$REFERENCE_RUN" "$REFERENCE_RUN/manifests" "$REFERENCE_RUN/reports"
    "$REFERENCE_COMPLETION" "$REFERENCE_INVENTORY" "$REFERENCE_CONFIG"
    "$REFERENCE_REPORT" "$REFERENCE_LONG_MANIFEST" "$REFERENCE_TEACHER_MANIFEST"
    "$FORMAL_RUN" "$FORMAL_RUN/manifests" "$FORMAL_COMPLETION" "$FORMAL_MANIFEST"
    "$CHECKPOINT_DIR" "$CHECKPOINT_FILE"
    "$RESULT_ROOT" "$CONTROL_PARENT" "$RUN_ROOT" "$CONTROL_ROOT" "$LOG_ROOT"
  )
  local labels=(
    "Repository root"
    "Source run" "Source manifests" "Source completion" "Source manifest"
    "Reference run" "Reference manifests" "Reference reports"
    "Reference completion" "Reference inventory" "Reference config"
    "Reference report" "Reference long manifest" "Reference teacher manifest"
    "Formal run" "Formal manifests" "Formal completion" "Formal manifest"
    "Checkpoint directory" "Checkpoint file"
    "Result root" "Control parent" "Run root" "Control root" "Log root"
  )
  lexical="$(realpath -m -s -- "${paths[@]}")" || {
    printf 'Could not normalize all critical paths.\n' >&2
    return 1
  }
  resolved="$(realpath -m -- "${paths[@]}")" || {
    printf 'Could not resolve all critical paths.\n' >&2
    return 1
  }
  [[ "$lexical" == "$resolved" ]] || {
    printf 'A critical path contains a symlink or junction component.\n' >&2
    return 1
  }
  require_child_of "$RESULT_ROOT" "$RUN_ROOT" "Run root" || return 1
  require_child_of "$RESULT_ROOT" "$CONTROL_ROOT" "Control root" || return 1
  require_child_of "$CONTROL_ROOT" "$LOG_ROOT" "Log root" || return 1
}

require_static_layout() {
  local required_file
  for required_file in \
    "$SOURCE_COMPLETION" "$SOURCE_MANIFEST" \
    "$REFERENCE_COMPLETION" "$REFERENCE_INVENTORY" "$REFERENCE_CONFIG" \
    "$REFERENCE_REPORT" "$REFERENCE_LONG_MANIFEST" \
    "$REFERENCE_TEACHER_MANIFEST" "$FORMAL_COMPLETION" "$FORMAL_MANIFEST" \
    "$CHECKPOINT_FILE"; do
    [[ -f "$required_file" && ! -L "$required_file" ]] || {
      printf 'Missing or linked frozen input: %s\n' "$required_file" >&2
      return 1
    }
  done
}

sha256_file() {
  sha256sum -- "$1" | awk 'NR == 1 {print $1}'
}

require_file_sha256() {
  local path="$1" expected="$2" label="$3" observed
  observed="$(sha256_file "$path")" || {
    printf 'Could not hash %s.\n' "$label" >&2
    return 1
  }
  [[ "$observed" == "$expected" ]] || {
    printf '%s SHA-256 mismatch.\n' "$label" >&2
    return 1
  }
}

require_python_identity() {
  local configured_parent observed_realpath
  configured_parent="$(dirname -- "$PYTHON")" || return 1
  reject_symlink_components "$configured_parent" "Python configured-path ancestors" || return 1
  [[ -e "$PYTHON" || -L "$PYTHON" ]] || {
    printf 'Missing frozen VGGT Python environment: %s\n' "$PYTHON" >&2
    return 1
  }
  observed_realpath="$(realpath -e -- "$PYTHON")" || {
    printf 'Could not resolve the configured Python leaf.\n' >&2
    return 1
  }
  [[ "$observed_realpath" == "$EXPECTED_PYTHON_REALPATH" ]] || {
    printf 'Python symlink/realpath identity mismatch.\n' >&2
    return 1
  }
  reject_symlink_components "$observed_realpath" "Python real target" || return 1
  [[ -f "$observed_realpath" && -x "$observed_realpath" ]] || {
    printf 'Python real target must be a regular executable file.\n' >&2
    return 1
  }
  require_file_sha256 "$PYTHON" "$EXPECTED_PYTHON_SHA256" "Python" || return 1
  PYTHON_REALPATH="$observed_realpath"
}

authenticate_inputs() {
  require_python_identity || return 1
  require_static_layout || return 1
  local digest_output line observed index
  local paths=(
    "$SOURCE_COMPLETION" "$SOURCE_MANIFEST" "$REFERENCE_COMPLETION"
    "$REFERENCE_INVENTORY" "$REFERENCE_CONFIG" "$REFERENCE_REPORT"
    "$REFERENCE_LONG_MANIFEST" "$REFERENCE_TEACHER_MANIFEST"
    "$FORMAL_COMPLETION" "$FORMAL_MANIFEST" "$CHECKPOINT_FILE"
  )
  local expected=(
    "$EXPECTED_SOURCE_COMPLETION_SHA256" "$EXPECTED_SOURCE_MANIFEST_SHA256"
    "$EXPECTED_REFERENCE_COMPLETION_SHA256" "$EXPECTED_REFERENCE_INVENTORY_SHA256"
    "$EXPECTED_REFERENCE_CONFIG_SHA256" "$EXPECTED_REFERENCE_REPORT_SHA256"
    "$EXPECTED_REFERENCE_LONG_MANIFEST_SHA256"
    "$EXPECTED_REFERENCE_TEACHER_MANIFEST_SHA256"
    "$EXPECTED_FORMAL_COMPLETION_SHA256" "$EXPECTED_FORMAL_MANIFEST_SHA256"
    "$EXPECTED_CHECKPOINT_SHA256"
  )
  local labels=(
    "Source completion" "Source manifest" "Reference completion"
    "Reference inventory" "Reference config" "Reference report"
    "Reference long manifest" "Reference teacher manifest"
    "Formal completion" "Formal manifest" "VGGT checkpoint"
  )
  digest_output="$(sha256sum -- "${paths[@]}")" || {
    printf 'Could not hash every frozen input.\n' >&2
    return 1
  }
  mapfile -t digest_lines <<< "$digest_output"
  (( ${#digest_lines[@]} == ${#paths[@]} )) || {
    printf 'Frozen input hash inventory is incomplete.\n' >&2
    return 1
  }
  for (( index = 0; index < ${#paths[@]}; index++ )); do
    line="${digest_lines[index]}"
    observed="${line%%[[:space:]]*}"
    [[ "$observed" == "${expected[index]}" ]] || {
      printf '%s SHA-256 mismatch.\n' "${labels[index]}" >&2
      return 1
    }
  done

  jq -e --arg schema "variational_camera_latent.verified_completion.v1" \
    --arg signal "WEAK_SIGNAL" --argjson scene_count 10 \
    --argjson overlap_count 80 --argjson candidate_count 2560 \
    '.schema == $schema and .signal == $signal and .scene_count == $scene_count and .overlap_count == $overlap_count and .candidate_count == $candidate_count' \
    "$SOURCE_COMPLETION" >/dev/null || {
    printf 'Source completion provenance mismatch.\n' >&2
    return 1
  }
  jq -e --arg schema "variational_camera_latent.source.v1" \
    --argjson records_length 10 \
    '.schema == $schema and (.records | type == "array" and length == $records_length)' \
    "$SOURCE_MANIFEST" >/dev/null || {
    printf 'Source manifest provenance mismatch.\n' >&2
    return 1
  }
  jq -e --arg schema "conditional_hierarchical_vrfm.verified_completion.v1" \
    --arg git_commit "cee41a09ac4085c8d6b0b343ca07d8e8c53ace3c" \
    --arg classification "LATENT_LIFT_FAILED" --argjson file_count 87 \
    --arg inventory_sha256 "$EXPECTED_REFERENCE_INVENTORY_SHA256" \
    '.schema == $schema and .git_commit == $git_commit and .classification == $classification and .file_count == $file_count and .inventory_sha256 == $inventory_sha256' \
    "$REFERENCE_COMPLETION" >/dev/null || {
    printf 'Reference completion provenance mismatch.\n' >&2
    return 1
  }
  jq -e --arg schema "conditional_hierarchical_vrfm.verification_inventory.v1" \
    --arg git_commit "cee41a09ac4085c8d6b0b343ca07d8e8c53ace3c" \
    --arg classification "LATENT_LIFT_FAILED" --argjson files_length 87 \
    '.schema == $schema and .git_commit == $git_commit and .classification == $classification and (.files | type == "object" and length == $files_length)' \
    "$REFERENCE_INVENTORY" >/dev/null || {
    printf 'Reference inventory provenance mismatch.\n' >&2
    return 1
  }
  jq -e --arg schema "conditional_hierarchical_vrfm.run_config.v1" \
    --arg git_commit "cee41a09ac4085c8d6b0b343ca07d8e8c53ace3c" \
    --arg checkpoint_sha256 "$EXPECTED_CHECKPOINT_SHA256" \
    --arg source_manifest_sha256 "$EXPECTED_SOURCE_MANIFEST_SHA256" \
    --arg formal_completion_sha256 "$EXPECTED_FORMAL_COMPLETION_SHA256" \
    --arg formal_data_manifest_sha256 "$EXPECTED_FORMAL_MANIFEST_SHA256" \
    --arg long_manifest_sha256 "$EXPECTED_REFERENCE_LONG_MANIFEST_SHA256" \
    --arg teacher_manifest_sha256 "$EXPECTED_REFERENCE_TEACHER_MANIFEST_SHA256" \
    --arg source_run "$SOURCE_RUN" --arg formal_run_root "$FORMAL_RUN" \
    --argjson scene_count 10 --argjson variant_count 4 \
    '.schema == $schema and .git_commit == $git_commit and .checkpoint_sha256 == $checkpoint_sha256 and .source_manifest_sha256 == $source_manifest_sha256 and .formal_completion_sha256 == $formal_completion_sha256 and .formal_data_manifest_sha256 == $formal_data_manifest_sha256 and .long_manifest_sha256 == $long_manifest_sha256 and .teacher_manifest_sha256 == $teacher_manifest_sha256 and .source_run == $source_run and .formal_run_root == $formal_run_root and .scene_count == $scene_count and .variant_count == $variant_count' \
    "$REFERENCE_CONFIG" >/dev/null || {
    printf 'Reference config provenance mismatch.\n' >&2
    return 1
  }
  jq -e --arg schema "conditional_hierarchical_vrfm.long_context_manifest.v1" \
    --argjson records_length 10 \
    '.schema == $schema and (.records | type == "array" and length == $records_length)' \
    "$REFERENCE_LONG_MANIFEST" >/dev/null || {
    printf 'Reference long manifest provenance mismatch.\n' >&2
    return 1
  }
  jq -e --arg schema "conditional_hierarchical_vrfm.teacher_manifest.v1" \
    --arg git_commit "cee41a09ac4085c8d6b0b343ca07d8e8c53ace3c" \
    --arg checkpoint_sha256 "$EXPECTED_CHECKPOINT_SHA256" \
    --arg formal_completion_sha256 "$EXPECTED_FORMAL_COMPLETION_SHA256" \
    --arg formal_data_manifest_sha256 "$EXPECTED_FORMAL_MANIFEST_SHA256" \
    --argjson records_length 10 \
    '.schema == $schema and .git_commit == $git_commit and .checkpoint_sha256 == $checkpoint_sha256 and .formal_completion_sha256 == $formal_completion_sha256 and .formal_data_manifest_sha256 == $formal_data_manifest_sha256 and (.records | type == "array" and length == $records_length)' \
    "$REFERENCE_TEACHER_MANIFEST" >/dev/null || {
    printf 'Reference teacher manifest provenance mismatch.\n' >&2
    return 1
  }
  jq -e --arg schema "conditional_hierarchical_vrfm.stage_a_report.v1" \
    --arg git_commit "cee41a09ac4085c8d6b0b343ca07d8e8c53ace3c" \
    --arg classification "LATENT_LIFT_FAILED" --argjson scene_metrics_length 10 \
    '.schema == $schema and .git_commit == $git_commit and .classification == $classification and (.scene_metrics | type == "array" and length == $scene_metrics_length)' \
    "$REFERENCE_REPORT" >/dev/null || {
    printf 'Reference report provenance mismatch.\n' >&2
    return 1
  }
  jq -e --arg schema "long_short_camera_head.verified_completion.v1" \
    --arg git_revision "2476a59f583ce4c39bbe66dc65d6a8e5cddfb52e" \
    --arg source_manifest_sha256 "$EXPECTED_SOURCE_MANIFEST_SHA256" \
    --arg base_checkpoint_sha256 "$EXPECTED_CHECKPOINT_SHA256" \
    --arg data_manifest_sha256 "$EXPECTED_FORMAL_MANIFEST_SHA256" \
    --arg classification "NO_SOURCE_HEAD_SIGNAL" --argjson scene_count 10 \
    --argjson train_scene_count 8 --argjson locked_replay_scene_count 2 \
    --argjson inference_leakage_audit true \
    '.schema == $schema and .git_revision == $git_revision and .source_manifest_sha256 == $source_manifest_sha256 and .base_checkpoint_sha256 == $base_checkpoint_sha256 and .data_manifest_sha256 == $data_manifest_sha256 and .classification == $classification and .scene_count == $scene_count and .train_scene_count == $train_scene_count and .locked_replay_scene_count == $locked_replay_scene_count and .inference_leakage_audit == $inference_leakage_audit' \
    "$FORMAL_COMPLETION" >/dev/null || {
    printf 'Formal completion provenance mismatch.\n' >&2
    return 1
  }
  jq -e --arg schema "long_short_camera_head.data_manifest.v1" \
    --arg git_revision "2476a59f583ce4c39bbe66dc65d6a8e5cddfb52e" \
    --arg source_run "$SOURCE_RUN" \
    --arg source_manifest_sha256 "$EXPECTED_SOURCE_MANIFEST_SHA256" \
    --arg checkpoint_dir "$CHECKPOINT_DIR" \
    --arg base_checkpoint_sha256 "$EXPECTED_CHECKPOINT_SHA256" \
    --argjson records_length 10 \
    '.schema == $schema and .git_revision == $git_revision and .source_run == $source_run and .source_manifest_sha256 == $source_manifest_sha256 and .checkpoint_dir == $checkpoint_dir and .base_checkpoint_sha256 == $base_checkpoint_sha256 and (.records | type == "array" and length == $records_length)' \
    "$FORMAL_MANIFEST" >/dev/null || {
    printf 'Formal manifest provenance mismatch.\n' >&2
    return 1
  }
}

require_free_space() {
  local free_bytes df_output
  df_output="$(df --output=avail -B1 /data)" || {
    printf 'Could not determine free bytes on /data.\n' >&2
    return 1
  }
  free_bytes="${df_output##*$'\n'}"
  [[ "$free_bytes" =~ ^[0-9]+$ ]] || {
    printf 'Could not determine free bytes on /data.\n' >&2
    return 1
  }
  (( free_bytes >= MIN_FREE_BYTES )) || {
    printf 'At least 100 GiB free on /data is required.\n' >&2
    return 1
  }
}

require_h20_inventory() {
  local rows raw_index raw_name index name expected_index=0 count=0
  rows="$(nvidia-smi --query-gpu=index,name --format=csv,noheader,nounits)" || {
    printf 'Could not inventory H20 GPUs.\n' >&2
    return 1
  }
  while IFS=',' read -r raw_index raw_name; do
    [[ -n "${raw_index}${raw_name}" ]] || continue
    index="${raw_index#"${raw_index%%[![:space:]]*}"}"
    index="${index%"${index##*[![:space:]]}"}"
    name="${raw_name#"${raw_name%%[![:space:]]*}"}"
    name="${name%"${name##*[![:space:]]}"}"
    [[ "$index" == "$expected_index" && "$name" == "NVIDIA H20" ]] || {
      printf 'H20 inventory must be exactly indices 0-7 with NVIDIA H20 names.\n' >&2
      return 1
    }
    expected_index=$((expected_index + 1))
    count=$((count + 1))
  done <<< "$rows"
  (( count == 8 )) || {
    printf 'H20 inventory must contain exactly eight NVIDIA H20 GPUs.\n' >&2
    return 1
  }
  printf '%s\n' "$rows"
}

strict_gpu_pids() {
  local index="$1" pids
  pids="$(nvidia-smi -i "$index" --query-compute-apps=pid --format=csv,noheader,nounits)" || {
    printf 'Could not inspect compute processes on GPU %s.\n' "$index" >&2
    return 1
  }
  if [[ -z "$pids" ]]; then
    return 0
  fi
  [[ "$pids" =~ ^[0-9]+([[:space:]]+[0-9]+)*$ ]] || {
    printf 'GPU %s process inventory is not a strict PID list or empty output.\n' "$index" >&2
    return 1
  }
  printf '%s\n' "$pids"
}

select_idle_h20() {
  local rows index pids
  rows="$(require_h20_inventory)" || return 1
  for index in 0 1 2 3 4 5 6 7; do
    pids="$(strict_gpu_pids "$index")" || return 1
    if [[ -z "$pids" ]]; then
      printf '%s\n' "$index"
      return 0
    fi
  done
  printf 'Every NVIDIA H20 has an active compute process.\n' >&2
  return 1
}

require_selected_gpu_idle() {
  local pids
  require_h20_inventory >/dev/null || return 1
  pids="$(strict_gpu_pids "$GPU_INDEX")" || return 1
  [[ -z "$pids" ]] || {
    printf 'GPU %s is no longer strictly idle.\n' "$GPU_INDEX" >&2
    return 1
  }
}

require_initial_run_absent() {
  [[ ! -e "$RUN_ROOT" && ! -L "$RUN_ROOT" ]] || {
    printf 'Unique run root already exists: %s\n' "$RUN_ROOT" >&2
    return 1
  }
}

require_run_size() {
  [[ ! -L "$RUN_ROOT" ]] || {
    printf 'Run root may not be a symlink.\n' >&2
    return 1
  }
  [[ -e "$RUN_ROOT" ]] || return 0
  [[ -d "$RUN_ROOT" ]] || {
    printf 'Run root is not a directory.\n' >&2
    return 1
  }
  local run_kib du_output
  du_output="$(du -sk -- "$RUN_ROOT")" || return 1
  run_kib="${du_output%%[[:space:]]*}"
  [[ "$run_kib" =~ ^[0-9]+$ ]] || {
    printf 'Could not determine run-root size.\n' >&2
    return 1
  }
  (( run_kib < MAX_RUN_KIB )) || {
    printf 'The run root reached 20 GiB; refusing to continue.\n' >&2
    return 1
  }
}

path_identity() {
  stat -Lc '%d:%i' -- "$1"
}

fd_identity() {
  path_identity "/proc/$$/fd/$1"
}

require_open_fd_path_identity() {
  local path="$1" fd="$2" expected="$3" label="$4" observed=""
  [[ -f "$path" && ! -L "$path" ]] || {
    printf '%s pathname is not a no-follow regular file for its opened FD.\n' \
      "$label" >&2
    return 1
  }
  observed="$(fd_identity "$fd" 2>/dev/null || true)"
  [[ -n "$observed" && "$observed" == "$expected" ]] || {
    printf '%s opened FD identity changed.\n' "$label" >&2
    return 1
  }
  observed="$(path_identity "$path" 2>/dev/null || true)"
  [[ -n "$observed" && "$observed" == "$expected" ]] || {
    printf '%s pathname does not match its opened FD.\n' "$label" >&2
    return 1
  }
}

CONTROL_ARTIFACT_PATHS=()
CONTROL_ARTIFACT_IDENTITIES=()
CONTROL_ARTIFACT_SIZES=()
CONTROL_ARTIFACT_SHA256S=()
CONTROL_ARTIFACT_LABELS=()
REGISTERED_CONTROL_SIZE=""

register_control_artifact_from_fd() {
  local path="$1" fd="$2" identity="$3" label="$4"
  local observed_size observed_sha256
  require_open_fd_path_identity "$path" "$fd" "$identity" "$label" || return 1
  observed_size="$(stat -Lc '%s' -- "/proc/$$/fd/$fd")" || return 1
  observed_sha256="$(sha256_file "/proc/$$/fd/$fd")" || return 1
  [[ "$observed_size" =~ ^[0-9]+$ \
    && "$observed_sha256" =~ ^[0-9a-f]{64}$ ]] || return 1
  require_open_fd_path_identity "$path" "$fd" "$identity" "$label" || return 1
  CONTROL_ARTIFACT_PATHS+=("$path")
  CONTROL_ARTIFACT_IDENTITIES+=("$identity")
  CONTROL_ARTIFACT_SIZES+=("$observed_size")
  CONTROL_ARTIFACT_SHA256S+=("$observed_sha256")
  CONTROL_ARTIFACT_LABELS+=("$label")
  REGISTERED_CONTROL_SIZE="$observed_size"
}

require_control_artifacts() {
  local index path relative lexical resolved expected_control_identity=""
  local -a registered_paths=() ledger_args=() checker_env=(
    "PATH=$CHILD_PATH" "HOME=$CHILD_HOME" "USER=ubuntu" "LOGNAME=ubuntu"
    "LANG=C.UTF-8" "LC_ALL=C.UTF-8" "PYTHONNOUSERSITE=1"
  )
  (( ${#CONTROL_ARTIFACT_PATHS[@]} == ${#CONTROL_ARTIFACT_IDENTITIES[@]} \
    && ${#CONTROL_ARTIFACT_PATHS[@]} == ${#CONTROL_ARTIFACT_SIZES[@]} \
    && ${#CONTROL_ARTIFACT_PATHS[@]} == ${#CONTROL_ARTIFACT_SHA256S[@]} \
    && ${#CONTROL_ARTIFACT_PATHS[@]} == ${#CONTROL_ARTIFACT_LABELS[@]} )) || return 1
  (( ${#CONTROL_ARTIFACT_PATHS[@]} > 0 )) || return 0
  for (( index = 0; index < ${#CONTROL_ARTIFACT_PATHS[@]}; index++ )); do
    path="${CONTROL_ARTIFACT_PATHS[index]}"
    case "$path" in
      "$CONTROL_ROOT"/*) relative="${path#"$CONTROL_ROOT"/}" ;;
      *) return 1 ;;
    esac
    registered_paths+=("$path")
    ledger_args+=(
      "$relative" "${CONTROL_ARTIFACT_IDENTITIES[index]}"
      "${CONTROL_ARTIFACT_SIZES[index]}" "${CONTROL_ARTIFACT_SHA256S[index]}"
    )
  done
  lexical="$(realpath -m -s -- "${registered_paths[@]}")" || {
    printf 'Could not normalize all registered control artifact paths.\n' >&2
    return 1
  }
  resolved="$(realpath -m -- "${registered_paths[@]}")" || {
    printf 'Could not resolve all registered control artifact paths.\n' >&2
    return 1
  }
  [[ "$lexical" == "$resolved" ]] || {
    printf 'A registered control artifact path contains a symlink or junction component.\n' >&2
    return 1
  }
  if [[ "$TEST_MODE" == "1" ]]; then
    checker_env+=(
      "FIXTURE_REAL_PYTHON=${FIXTURE_REAL_PYTHON:?}"
      "FIXTURE_WINDOWS_TEMP=${FIXTURE_WINDOWS_TEMP:?}"
    )
    if [[ -n "${FIXTURE_SWAP_CONTROL_BETWEEN_STAT_AND_HASH_PATH:-}" ]]; then
      checker_env+=(
        "FIXTURE_SWAP_CONTROL_BETWEEN_STAT_AND_HASH_PATH=$FIXTURE_SWAP_CONTROL_BETWEEN_STAT_AND_HASH_PATH"
      )
    fi
  else
    expected_control_identity="$CONTROL_ROOT_IDENTITY"
  fi
  env -i "${checker_env[@]}" "$PYTHON" - control-ledger \
    "$CONTROL_ROOT" "$expected_control_identity" "${ledger_args[@]}" \
    7>&- 8>&- 9>&- <<'PY_CONTROL'
import hashlib
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys

def native(name: str) -> Path:
    windows_temp = os.environ.get("FIXTURE_WINDOWS_TEMP")
    if windows_temp and name == "/tmp":
        return Path(windows_temp)
    if windows_temp and name.startswith("/tmp/"):
        return Path(windows_temp) / name[5:]
    return Path(name)

TEST_MODE = "FIXTURE_WINDOWS_TEMP" in os.environ
SUPPORTS_SECURE_OPEN = (
    os.open in os.supports_dir_fd
    and bool(getattr(os, "O_DIRECTORY", 0))
    and bool(getattr(os, "O_NOFOLLOW", 0))
)
if not TEST_MODE and not SUPPORTS_SECURE_OPEN:
    raise RuntimeError("formal control validation requires dir_fd, O_DIRECTORY, and O_NOFOLLOW")

STABLE_FIELDS = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns")

def directory_record(info: os.stat_result) -> tuple[int, int, int]:
    if not stat.S_ISDIR(info.st_mode):
        raise ValueError("control path component is not a directory")
    return info.st_dev, info.st_ino, info.st_mode

def record_directory(records: dict[str, tuple[int, int, int]], relative: str, info: os.stat_result) -> None:
    observed = directory_record(info)
    previous = records.setdefault(relative, observed)
    if previous != observed:
        raise ValueError("control directory identity changed")

def relative_parts(relative: str) -> tuple[str, ...]:
    if not relative or "\\" in relative or "\x00" in relative:
        raise ValueError("unsafe control relative path")
    path = PurePosixPath(relative)
    if (
        path.is_absolute()
        or path.as_posix() != relative
        or not path.parts
        or any(part in ("", ".", "..") for part in path.parts)
        or ":" in path.parts[0]
    ):
        raise ValueError("unsafe control relative path")
    return path.parts

def portable_open(root: Path, parts: tuple[str, ...], directories: dict) -> int:
    current = root
    root_info = os.stat(current, follow_symlinks=False)
    if stat.S_ISLNK(root_info.st_mode):
        raise ValueError("control root may not be a symlink")
    record_directory(directories, ".", root_info)
    traversed = []
    for part in parts[:-1]:
        current = current / part
        traversed.append(part)
        info = os.stat(current, follow_symlinks=False)
        if stat.S_ISLNK(info.st_mode):
            raise ValueError("control path component may not be a symlink")
        record_directory(directories, "/".join(traversed), info)
    path = current / parts[-1]
    before = os.stat(path, follow_symlinks=False)
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ValueError("control artifact is not a no-follow regular file")
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    opened = os.fstat(descriptor)
    if (
        not stat.S_ISREG(opened.st_mode)
        or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
    ):
        os.close(descriptor)
        raise ValueError("control artifact identity changed while opening")
    return descriptor

def secure_open(root: Path, parts: tuple[str, ...], directories: dict) -> int:
    directory_flags = (
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | os.O_DIRECTORY | os.O_NOFOLLOW
    )
    file_flags = (
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0) | os.O_NOFOLLOW
    )
    directory = os.open(root, directory_flags)
    try:
        record_directory(directories, ".", os.fstat(directory))
        traversed = []
        for part in parts[:-1]:
            child = os.open(part, directory_flags, dir_fd=directory)
            traversed.append(part)
            record_directory(directories, "/".join(traversed), os.fstat(child))
            os.close(directory)
            directory = child
        descriptor = os.open(parts[-1], file_flags, dir_fd=directory)
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            os.close(descriptor)
            raise ValueError("control artifact is not a regular file")
        return descriptor
    finally:
        os.close(directory)

def open_control(root: Path, relative: str, directories: dict) -> int:
    parts = relative_parts(relative)
    if TEST_MODE:
        return portable_open(root, parts, directories)
    return secure_open(root, parts, directories)

def scan_control(root: Path, relative: str, directories: dict) -> tuple[int, int, int, str]:
    descriptor = open_control(root, relative, directories)
    try:
        before = os.fstat(descriptor)
        hasher = hashlib.sha256()
        actual_size = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
            actual_size += len(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if any(getattr(before, field) != getattr(after, field) for field in STABLE_FIELDS):
        raise ValueError(f"control artifact changed while hashing: {relative}")
    current_descriptor = open_control(root, relative, directories)
    try:
        current = os.fstat(current_descriptor)
    finally:
        os.close(current_descriptor)
    if any(getattr(after, field) != getattr(current, field) for field in STABLE_FIELDS):
        raise ValueError(f"control artifact pathname changed while hashing: {relative}")
    if actual_size != after.st_size:
        raise ValueError(f"control artifact size changed while hashing: {relative}")
    return after.st_dev, after.st_ino, after.st_size, hasher.hexdigest()

if sys.argv[1] != "control-ledger":
    raise ValueError("invalid control validation mode")
root = native(sys.argv[2])
expected_root_identity = sys.argv[3]
arguments = sys.argv[4:]
if not arguments or len(arguments) % 4:
    raise ValueError("invalid registered control ledger")
expected_records = {}
for index in range(0, len(arguments), 4):
    relative, identity, size_text, digest = arguments[index:index + 4]
    relative_parts(relative)
    dev_text, separator, ino_text = identity.partition(":")
    if (
        not separator
        or not dev_text.isdecimal()
        or not ino_text.isdecimal()
        or not size_text.isdecimal()
        or not re.fullmatch(r"[0-9a-f]{64}", digest)
        or relative in expected_records
    ):
        raise ValueError("invalid registered control record")
    expected_records[relative] = (
        int(dev_text), int(ino_text), int(size_text), digest,
    )

directories = {}
for relative, expected in expected_records.items():
    if scan_control(root, relative, directories) != expected:
        raise ValueError(f"registered control record changed: {relative}")
if expected_root_identity:
    root_record = directories["."]
    if f"{root_record[0]}:{root_record[1]}" != expected_root_identity:
        raise ValueError("registered control root identity changed")
PY_CONTROL
  local checker_status=$?
  (( checker_status == 0 )) || return "$checker_status"
  if [[ -n "${IDENTITY_PATH:-}" && -n "${IDENTITY_IDENTITY:-}" ]]; then
    require_open_fd_path_identity \
      "$IDENTITY_PATH" 7 "$IDENTITY_IDENTITY" "Identity record" || return 1
  fi
}

STATIC_PATHS=()
STATIC_IDENTITIES=""

record_static_path_identities() {
  STATIC_PATHS=(
    "$REPO_ROOT" "$REPO_ROOT/.git" "$PYTHON" "$PYTHON_REALPATH"
    "$SOURCE_RUN" "$SOURCE_RUN/manifests" "$SOURCE_COMPLETION" "$SOURCE_MANIFEST"
    "$REFERENCE_RUN" "$REFERENCE_RUN/manifests" "$REFERENCE_RUN/reports"
    "$REFERENCE_COMPLETION" "$REFERENCE_INVENTORY" "$REFERENCE_CONFIG"
    "$REFERENCE_REPORT" "$REFERENCE_LONG_MANIFEST" "$REFERENCE_TEACHER_MANIFEST"
    "$FORMAL_RUN" "$FORMAL_RUN/manifests" "$FORMAL_COMPLETION" "$FORMAL_MANIFEST"
    "$CHECKPOINT_DIR" "$CHECKPOINT_FILE"
  )
  STATIC_IDENTITIES="$(stat -Lc '%d:%i' -- "${STATIC_PATHS[@]}")" || {
    printf 'Could not record all frozen path identities.\n' >&2
    return 1
  }
  (( $(awk 'END {print NR}' <<< "$STATIC_IDENTITIES") == ${#STATIC_PATHS[@]} )) || {
    printf 'Frozen path identity inventory is incomplete.\n' >&2
    return 1
  }
}

require_static_path_identities() {
  local observed
  observed="$(stat -Lc '%d:%i' -- "${STATIC_PATHS[@]}")" || {
    printf 'Could not recheck all frozen path identities.\n' >&2
    return 1
  }
  [[ "$observed" == "$STATIC_IDENTITIES" ]] || {
    printf 'A frozen path identity changed during execution.\n' >&2
    return 1
  }
}

require_initial_control_absent() {
  [[ ! -e "$CONTROL_ROOT" && ! -L "$CONTROL_ROOT" ]] || {
    printf 'Unique control root already exists: %s\n' "$CONTROL_ROOT" >&2
    return 1
  }
}

record_output_identities() {
  RESULT_ROOT_IDENTITY="$(path_identity "$RESULT_ROOT")" || return 1
  CONTROL_PARENT_IDENTITY="$(path_identity "$CONTROL_PARENT")" || return 1
  CONTROL_ROOT_IDENTITY="$(path_identity "$CONTROL_ROOT")" || return 1
  LOG_ROOT_IDENTITY="$(path_identity "$LOG_ROOT")" || return 1
  RUN_ROOT_IDENTITY=""
}

require_output_identities() {
  local observed expected
  observed="$(stat -Lc '%d:%i' -- \
    "$RESULT_ROOT" "$CONTROL_PARENT" "$CONTROL_ROOT" "$LOG_ROOT")" || true
  expected="${RESULT_ROOT_IDENTITY}"$'\n'"${CONTROL_PARENT_IDENTITY}"$'\n'"${CONTROL_ROOT_IDENTITY}"$'\n'"${LOG_ROOT_IDENTITY}"
  [[ -n "$observed" && "$observed" == "$expected" ]] || {
    printf 'Result/control/log path identity changed during execution.\n' >&2
    return 1
  }
  if [[ -z "$RUN_ROOT_IDENTITY" ]]; then
    [[ ! -e "$RUN_ROOT" && ! -L "$RUN_ROOT" ]] || {
      printf 'Run root appeared before the preflight stage.\n' >&2
      return 1
    }
  else
    observed="$(path_identity "$RUN_ROOT")" || true
    [[ -n "$observed" && "$observed" == "$RUN_ROOT_IDENTITY" ]] || {
      printf 'Run root identity changed during execution.\n' >&2
      return 1
    }
  fi
}

require_lock_identities() {
  local observed
  require_open_fd_path_identity \
    "$RUN_LOCK_PATH" 9 "$RUN_LOCK_IDENTITY" "Run lock" || return 1
  require_open_fd_path_identity \
    "$GPU_LOCK_PATH" 8 "$GPU_LOCK_IDENTITY" "GPU lock" || return 1
}

require_system_identity
require_initial_git_state
validate_paths
require_tree_isolation
authenticate_inputs
record_static_path_identities
require_static_path_identities
require_free_space
GPU_INDEX="$(select_idle_h20)"
validate_paths
require_initial_run_absent
require_initial_control_absent
require_run_size

if (( PREFLIGHT_ONLY == 1 )); then
  jq -cn \
    --arg result_root "$RESULT_ROOT" \
    --arg gpu_index "$GPU_INDEX" \
    --arg git_commit "$GIT_COMMIT" \
    --arg python_path "$PYTHON" \
    --arg python_realpath "$PYTHON_REALPATH" \
    --arg python_sha256 "$EXPECTED_PYTHON_SHA256" \
    '{result_root:$result_root,planned_stages:["preflight","prepare","smoke","calibration","report","verify"],gpu_index:$gpu_index,git_commit:$git_commit,python_path:$python_path,python_realpath:$python_realpath,python_sha256:$python_sha256}'
  exit 0
fi

# These are the first writes: every read-only shell gate above has succeeded.
mkdir -p -- "$RESULT_ROOT"
validate_paths
mkdir -p -- "$CONTROL_PARENT"
validate_paths
mkdir -- "$CONTROL_ROOT" || {
  printf 'Control root must be atomically created and initially absent.\n' >&2
  exit 24
}
mkdir -- "$LOG_ROOT" || {
  printf 'Could not atomically create the log root.\n' >&2
  exit 24
}
validate_paths
record_output_identities

RUN_LOCK_PATH="$CONTROL_ROOT/run.lock"
GPU_LOCK_PATH="${CONTROL_PARENT}/gpu_${GPU_INDEX}.lock"
reject_symlink_components "$RUN_LOCK_PATH" "Run lock" || exit 24
reject_symlink_components "$GPU_LOCK_PATH" "GPU lock" || exit 24
require_child_of "$CONTROL_ROOT" "$RUN_LOCK_PATH" "Run lock" || exit 24
require_child_of "$CONTROL_PARENT" "$GPU_LOCK_PATH" "GPU lock" || exit 24
RUN_LOCK_IDENTITY=""
GPU_LOCK_IDENTITY=""

remove_exact_owned_lock_best_effort() {
  local path="$1" expected="$2" fd="$3"
  local fd_observed="" path_observed=""
  if [[ -n "$expected" ]]; then
    fd_observed="$(fd_identity "$fd" 2>/dev/null || true)"
    if [[ "$fd_observed" == "$expected" \
      && -f "$path" && ! -L "$path" ]]; then
      path_observed="$(path_identity "$path" 2>/dev/null || true)"
      if [[ "$path_observed" == "$expected" ]]; then
        # Unlink our exact pathname while the advisory lock is still held.
        rm -f -- "$path" || true
      fi
    fi
  fi
  flock -u "$fd" >/dev/null 2>&1 || true
  eval "exec ${fd}>&-"
}

release_locks_best_effort() {
  remove_exact_owned_lock_best_effort "$GPU_LOCK_PATH" "$GPU_LOCK_IDENTITY" 8
  remove_exact_owned_lock_best_effort "$RUN_LOCK_PATH" "$RUN_LOCK_IDENTITY" 9
  exec 7>&- || true
}
trap release_locks_best_effort EXIT

close_lock_fd() {
  case "$1" in
    8) exec 8>&- ;;
    9) exec 9>&- ;;
    *) return 1 ;;
  esac
}

unlock_lock_fd_strict() {
  if [[ "$TEST_MODE" == "1" ]]; then
    flock -u "$1"
  else
    /usr/bin/flock -u "$1"
  fi
}

remove_exact_owned_lock_strict() {
  local path="$1" expected="$2" fd="$3"
  local fd_observed="" path_observed="" current_observed=""
  [[ -n "$expected" && -f "$path" && ! -L "$path" ]] || return 1
  fd_observed="$(/usr/bin/stat -Lc '%d:%i' -- "/proc/$$/fd/$fd")" || return 1
  path_observed="$(/usr/bin/stat -Lc '%d:%i' -- "$path")" || return 1
  [[ "$fd_observed" == "$expected" && "$path_observed" == "$expected" ]] || return 1
  /usr/bin/rm -f -- "$path" || return 1
  if [[ -e "$path" || -L "$path" ]]; then
    current_observed="$(/usr/bin/stat -Lc '%d:%i' -- "$path" 2>/dev/null)" || return 1
    [[ "$current_observed" != "$expected" ]] || return 1
  fi
  unlock_lock_fd_strict "$fd" || return 1
  close_lock_fd "$fd" || return 1
  if [[ -e "$path" || -L "$path" ]]; then
    current_observed="$(/usr/bin/stat -Lc '%d:%i' -- "$path" 2>/dev/null)" || return 1
    [[ "$current_observed" != "$expected" ]] || return 1
  fi
}

release_locks_strict() {
  local status=0
  remove_exact_owned_lock_strict \
    "$GPU_LOCK_PATH" "$GPU_LOCK_IDENTITY" 8 || status=1
  remove_exact_owned_lock_strict \
    "$RUN_LOCK_PATH" "$RUN_LOCK_IDENTITY" 9 || status=1
  exec 7>&- || status=1
  return "$status"
}

set -o noclobber
if ! exec 9>"$RUN_LOCK_PATH"; then
  set +o noclobber
  printf 'Run lock exists or is stale; audit its PID, path, and inode before manual removal.\n' >&2
  exit 24
fi
set +o noclobber
RUN_LOCK_IDENTITY="$(fd_identity 9)" || {
  printf 'Could not identify the run lock opened FD.\n' >&2
  exit 24
}
require_open_fd_path_identity \
  "$RUN_LOCK_PATH" 9 "$RUN_LOCK_IDENTITY" "Run lock" || exit 24
printf '%s\n' "$$" >&9
flock -n 9 || {
  printf 'Run ID %s is already active.\n' "$RUN_ID" >&2
  exit 24
}
set -o noclobber
if ! exec 8>"$GPU_LOCK_PATH"; then
  set +o noclobber
  printf 'GPU lock exists or is stale; audit its PID, path, and inode before manual removal.\n' >&2
  exit 24
fi
set +o noclobber
GPU_LOCK_IDENTITY="$(fd_identity 8)" || {
  printf 'Could not identify the GPU lock opened FD.\n' >&2
  exit 24
}
require_open_fd_path_identity \
  "$GPU_LOCK_PATH" 8 "$GPU_LOCK_IDENTITY" "GPU lock" || exit 24
printf '%s\n' "$$" >&8
flock -n 8 || {
  printf 'GPU %s runner lock is already held.\n' "$GPU_INDEX" >&2
  exit 24
}

IDENTITY_PATH="$CONTROL_ROOT/identity.json"
identity_payload="$(jq -cn \
  --arg git_commit "$GIT_COMMIT" \
  --arg python_path "$PYTHON" \
  --arg python_realpath "$PYTHON_REALPATH" \
  --arg python_sha256 "$EXPECTED_PYTHON_SHA256" \
  '{git_commit:$git_commit,python_path:$python_path,python_realpath:$python_realpath,python_sha256:$python_sha256}')"
set -o noclobber
if ! exec 7>"$IDENTITY_PATH"; then
  set +o noclobber
  printf 'Identity record must be a create-new regular file.\n' >&2
  exit 24
fi
set +o noclobber
IDENTITY_IDENTITY="$(fd_identity 7)" || {
  printf 'Could not identify the identity record opened FD.\n' >&2
  exit 24
}
require_open_fd_path_identity \
  "$IDENTITY_PATH" 7 "$IDENTITY_IDENTITY" "Identity record" || exit 24
printf '%s\n' "$identity_payload" >&7
require_open_fd_path_identity \
  "$IDENTITY_PATH" 7 "$IDENTITY_IDENTITY" "Identity record" || exit 24
register_control_artifact_from_fd \
  "$IDENTITY_PATH" 7 "$IDENTITY_IDENTITY" "Identity record" || exit 24

OMP_THREADS="${OMP_NUM_THREADS:-8}"
[[ "$OMP_THREADS" =~ ^[1-9][0-9]*$ ]] || {
  printf 'OMP_NUM_THREADS must be a positive integer.\n' >&2
  exit 24
}
if [[ "$TEST_MODE" == "1" ]]; then
  CHILD_PATH="$PATH"
  CHILD_HOME="${HOME:-/tmp}"
else
  CHILD_PATH="/home/ubuntu/anaconda3/envs/vggt-gx/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
  CHILD_HOME="/home/ubuntu"
fi

STAGE_ENV=(
  "PATH=$CHILD_PATH"
  "HOME=$CHILD_HOME"
  "USER=ubuntu"
  "LOGNAME=ubuntu"
  "LANG=C.UTF-8"
  "LC_ALL=C.UTF-8"
  "PYTHONPATH=$REPO_ROOT"
  "PYTHONNOUSERSITE=1"
  "CUDA_VISIBLE_DEVICES=$GPU_INDEX"
  "OMP_NUM_THREADS=$OMP_THREADS"
  "HF_HUB_OFFLINE=1"
  "TRANSFORMERS_OFFLINE=1"
)
if [[ "$TEST_MODE" == "1" ]]; then
  TEST_CHILD_VARIABLES=(
    FIXTURE_PYTHON_LOG FIXTURE_PYTHON_ENV_LOG FIXTURE_REAL_PYTHON
    FIXTURE_COMPLETION_HELPER FIXTURE_WINDOWS_TEMP FIXTURE_CREATE_COMPLETION
    FIXTURE_COMPLETION_MODE FIXTURE_BLOCK_STAGE FIXTURE_BLOCK_STARTED
    FIXTURE_BLOCK_RELEASE FIXTURE_STDERR_STAGE FIXTURE_FAIL_STAGE
    FIXTURE_TAMPER_AFTER_STAGE FIXTURE_TAMPER_PATH
    FIXTURE_TAMPER_PYTHON_AFTER_STAGE FIXTURE_REPLACE_PATH_AFTER_STAGE
    FIXTURE_REPLACE_PATH FIXTURE_CONTROL_MUTATE_AFTER_STAGE
    FIXTURE_CONTROL_MUTATION_MODE FIXTURE_CONTROL_MUTATE_PATH
  )
  for variable_name in "${TEST_CHILD_VARIABLES[@]}"; do
    if [[ -n "${!variable_name:-}" ]]; then
      STAGE_ENV+=("$variable_name=${!variable_name}")
    fi
  done
fi

cd "$REPO_ROOT"

full_runtime_gate() {
  require_system_identity || return 1
  require_git_state || return 1
  validate_paths || return 1
  require_output_identities || return 1
  require_lock_identities || return 1
  require_control_artifacts || return 1
  require_static_path_identities || return 1
  authenticate_inputs || return 1
  require_static_path_identities || return 1
  require_free_space || return 1
  require_selected_gpu_idle || return 1
  require_run_size || return 1
}

create_stage_log() {
  local path="$1" label="$2" fd="$3"
  reject_symlink_components "$path" "$label" || return 1
  require_child_of "$LOG_ROOT" "$path" "$label" || return 1
  set -o noclobber
  if ! eval "exec ${fd}>\"\$path\""; then
    set +o noclobber
    printf '%s must be a create-new, no-follow regular file.\n' "$label" >&2
    return 1
  fi
  set +o noclobber
  [[ -f "$path" && ! -L "$path" ]] || return 1
}

run_stage() {
  local stage="$1" stdout_log stderr_log return_code=0
  local stdout_identity stderr_identity stderr_snapshot_size observed
  stdout_log="$LOG_ROOT/${stage}.out.log"
  stderr_log="$LOG_ROOT/${stage}.err.log"
  full_runtime_gate
  if [[ "$stage" == "preflight" ]]; then
    require_initial_run_absent
  else
    [[ -n "$RUN_ROOT_IDENTITY" ]] || exit 25
  fi
  create_stage_log "$stdout_log" "$stage stdout log" 6 || exit 25
  create_stage_log "$stderr_log" "$stage stderr log" 5 || exit 25
  stdout_identity="$(fd_identity 6)" || {
    printf 'Could not identify the %s stdout log opened FD.\n' "$stage" >&2
    exit 25
  }
  require_open_fd_path_identity \
    "$stdout_log" 6 "$stdout_identity" "$stage stdout log" || exit 25
  stderr_identity="$(fd_identity 5)" || {
    printf 'Could not identify the %s stderr log opened FD.\n' "$stage" >&2
    exit 25
  }
  require_open_fd_path_identity \
    "$stderr_log" 5 "$stderr_identity" "$stage stderr log" || exit 25
  env -i "${STAGE_ENV[@]}" "$PYTHON" \
    -m pre_experiments.camera_translation_hvrfm.stages \
    "$stage" "${COMMON_ARGS[@]}" 1>&6 2>&5 7>&- 8>&- 9>&- || return_code=$?
  require_open_fd_path_identity \
    "$stdout_log" 6 "$stdout_identity" "$stage stdout log" || exit 25
  require_open_fd_path_identity \
    "$stderr_log" 5 "$stderr_identity" "$stage stderr log" || exit 25
  register_control_artifact_from_fd \
    "$stdout_log" 6 "$stdout_identity" "$stage stdout log" || exit 25
  register_control_artifact_from_fd \
    "$stderr_log" 5 "$stderr_identity" "$stage stderr log" || exit 25
  stderr_snapshot_size="$REGISTERED_CONTROL_SIZE"
  exec 6>&-
  exec 5>&-
  observed="$(path_identity "$stdout_log" || true)"
  [[ -f "$stdout_log" && ! -L "$stdout_log" \
    && "$observed" == "$stdout_identity" ]] || {
    printf '%s stdout log identity changed.\n' "$stage" >&2
    exit 25
  }
  observed="$(path_identity "$stderr_log" || true)"
  [[ -f "$stderr_log" && ! -L "$stderr_log" \
    && "$observed" == "$stderr_identity" ]] || {
    printf '%s stderr log identity changed.\n' "$stage" >&2
    exit 25
  }
  if (( return_code != 0 )); then
    printf '%s stage failed with exit code %s; artifacts and logs were preserved.\n' \
      "$stage" "$return_code" >&2
    exit 25
  fi
  [[ "$stderr_snapshot_size" == "0" ]] || {
    printf '%s wrote stderr; refusing to continue.\n' "$stage" >&2
    exit 26
  }
  [[ -d "$RUN_ROOT" && ! -L "$RUN_ROOT" ]] || {
    printf '%s did not preserve the canonical run root.\n' "$stage" >&2
    exit 26
  }
  if [[ "$stage" == "preflight" ]]; then
    RUN_ROOT_IDENTITY="$(path_identity "$RUN_ROOT")" || exit 26
  fi
  full_runtime_gate
}

COMMON_ARGS=(
  --run-root "$RUN_ROOT"
  --git-commit "$GIT_COMMIT"
  --source-run "$SOURCE_RUN"
  --reference-run "$REFERENCE_RUN"
  --formal-run "$FORMAL_RUN"
  --checkpoint-dir "$CHECKPOINT_DIR"
  --expected-source-completion-sha256 "$EXPECTED_SOURCE_COMPLETION_SHA256"
  --expected-reference-completion-sha256 "$EXPECTED_REFERENCE_COMPLETION_SHA256"
  --expected-formal-completion-sha256 "$EXPECTED_FORMAL_COMPLETION_SHA256"
  --expected-checkpoint-sha256 "$EXPECTED_CHECKPOINT_SHA256"
  --device cuda
)

for stage in "${PLANNED_STAGES[@]}"; do
  run_stage "$stage"
done

FINAL_SNAPSHOT_PATH="$CONTROL_ROOT/final_validation_snapshot.json"
FINAL_SNAPSHOT_TOKEN=""

validate_final_completion() {
  local mode="$1" expected_token="${2:-}" output="" expected_run_identity=""
  local expected_control_identity="" index path relative
  local -a control_ledger_args=()
  local validator_env=(
    "PATH=$CHILD_PATH" "HOME=$CHILD_HOME" "USER=ubuntu" "LOGNAME=ubuntu"
    "LANG=C.UTF-8" "LC_ALL=C.UTF-8" "PYTHONNOUSERSITE=1"
  )
  if [[ "$TEST_MODE" == "1" ]]; then
    validator_env+=(
      "FIXTURE_REAL_PYTHON=${FIXTURE_REAL_PYTHON:?}"
      "FIXTURE_WINDOWS_TEMP=${FIXTURE_WINDOWS_TEMP:?}"
    )
  else
    expected_run_identity="$RUN_ROOT_IDENTITY"
    expected_control_identity="$CONTROL_ROOT_IDENTITY"
  fi
  (( ${#CONTROL_ARTIFACT_PATHS[@]} == ${#CONTROL_ARTIFACT_IDENTITIES[@]} \
    && ${#CONTROL_ARTIFACT_PATHS[@]} == ${#CONTROL_ARTIFACT_SIZES[@]} \
    && ${#CONTROL_ARTIFACT_PATHS[@]} == ${#CONTROL_ARTIFACT_SHA256S[@]} )) || return 1
  for (( index = 0; index < ${#CONTROL_ARTIFACT_PATHS[@]}; index++ )); do
    path="${CONTROL_ARTIFACT_PATHS[index]}"
    case "$path" in
      "$CONTROL_ROOT"/*) relative="${path#"$CONTROL_ROOT"/}" ;;
      *) return 1 ;;
    esac
    control_ledger_args+=(
      "$relative" "${CONTROL_ARTIFACT_IDENTITIES[index]}"
      "${CONTROL_ARTIFACT_SIZES[index]}" "${CONTROL_ARTIFACT_SHA256S[index]}"
    )
  done
  [[ "$mode" == "capture" || "$mode" == "verify" ]] || return 1
  if ! output="$(env -i "${validator_env[@]}" \
    "$PYTHON" - "$mode" "$RUN_ROOT" "$RUN_ID" "$GIT_COMMIT" \
    "$CONTROL_ROOT" "$expected_token" "$expected_run_identity" \
    "$expected_control_identity" "${control_ledger_args[@]}" \
    7>&- 8>&- 9>&- <<'PY'
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys

def native(name: str) -> Path:
    windows_temp = os.environ.get("FIXTURE_WINDOWS_TEMP")
    if windows_temp and name == "/tmp":
        return Path(windows_temp)
    if windows_temp and name.startswith("/tmp/"):
        return Path(windows_temp) / name[5:]
    return Path(name)

def reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON value: {value}")

def unique_object(pairs: list[tuple[str, object]]) -> dict:
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value

def load(payload: bytes, label: str) -> dict:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{label} is not UTF-8") from error
    value = json.loads(
        text,
        object_pairs_hook=unique_object,
        parse_constant=reject_constant,
    )
    if type(value) is not dict:
        raise ValueError(f"{label} root is not an object")
    return value

def digest(unsigned: dict) -> str:
    encoded = json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()

def require_digest(payload: dict) -> None:
    unsigned = dict(payload)
    recorded = unsigned.pop("completion_digest", None)
    if not isinstance(recorded, str) or recorded != digest(unsigned):
        raise ValueError("completion_digest mismatch")

STABLE_FIELDS = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns")
TEST_MODE = "FIXTURE_WINDOWS_TEMP" in os.environ
SUPPORTS_SECURE_OPEN = (
    os.open in os.supports_dir_fd
    and bool(getattr(os, "O_DIRECTORY", 0))
    and bool(getattr(os, "O_NOFOLLOW", 0))
)
if not TEST_MODE and not SUPPORTS_SECURE_OPEN:
    raise RuntimeError("formal validation requires dir_fd, O_DIRECTORY, and O_NOFOLLOW")

def canonical_bytes(payload: dict) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")

def file_record(info: os.stat_result, sha256: str) -> dict:
    return {
        "dev": info.st_dev,
        "ino": info.st_ino,
        "mode": info.st_mode,
        "size": info.st_size,
        "mtime_ns": info.st_mtime_ns,
        "ctime_ns": info.st_ctime_ns,
        "sha256": sha256,
    }

def directory_record(info: os.stat_result) -> dict:
    return {"dev": info.st_dev, "ino": info.st_ino, "mode": info.st_mode}

def record_directory(records: dict, relative: str, info: os.stat_result) -> None:
    if not stat.S_ISDIR(info.st_mode):
        raise ValueError("path component is not a directory")
    observed = directory_record(info)
    previous = records.setdefault(relative, observed)
    if previous != observed:
        raise ValueError("directory identity changed during validation")

def relative_parts(relative: object) -> tuple[str, ...]:
    if not isinstance(relative, str) or not relative or "\\" in relative or "\x00" in relative:
        raise ValueError("unsafe relative path")
    posix = PurePosixPath(relative)
    if (
        posix.is_absolute()
        or posix.as_posix() != relative
        or not posix.parts
        or any(part in ("", ".", "..") for part in posix.parts)
        or ":" in posix.parts[0]
    ):
        raise ValueError("unsafe relative path")
    return posix.parts

def portable_open_at(root: Path, parts: tuple[str, ...], directories: dict) -> int:
    current = root
    root_info = os.stat(current, follow_symlinks=False)
    if stat.S_ISLNK(root_info.st_mode):
        raise ValueError("root may not be a symlink")
    record_directory(directories, ".", root_info)
    traversed = []
    for part in parts[:-1]:
        traversed.append(part)
        current = current / part
        info = os.stat(current, follow_symlinks=False)
        if stat.S_ISLNK(info.st_mode):
            raise ValueError("path component may not be a symlink")
        record_directory(directories, "/".join(traversed), info)
    path = current / parts[-1]
    before = os.stat(path, follow_symlinks=False)
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ValueError("artifact is not a no-follow regular file")
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    opened = os.fstat(descriptor)
    if (
        not stat.S_ISREG(opened.st_mode)
        or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
    ):
        os.close(descriptor)
        raise ValueError("artifact identity changed while opening")
    return descriptor

def secure_open_at(root: Path, parts: tuple[str, ...], directories: dict) -> int:
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | os.O_DIRECTORY
        | os.O_NOFOLLOW
    )
    file_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0)
        | os.O_NOFOLLOW
    )
    directory = os.open(root, directory_flags)
    traversed = []
    try:
        record_directory(directories, ".", os.fstat(directory))
        for part in parts[:-1]:
            child = os.open(part, directory_flags, dir_fd=directory)
            traversed.append(part)
            record_directory(directories, "/".join(traversed), os.fstat(child))
            os.close(directory)
            directory = child
        descriptor = os.open(parts[-1], file_flags, dir_fd=directory)
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            os.close(descriptor)
            raise ValueError("artifact is not a regular file")
        return descriptor
    finally:
        os.close(directory)

def open_at(root: Path, relative: str, directories: dict) -> int:
    parts = relative_parts(relative)
    if TEST_MODE:
        return portable_open_at(root, parts, directories)
    return secure_open_at(root, parts, directories)

def snapshot_file(root: Path, relative: str, directories: dict, keep_bytes: bool = False):
    try:
        descriptor = open_at(root, relative, directories)
    except (FileNotFoundError, NotADirectoryError) as error:
        raise ValueError(f"artifact is missing: {relative}") from error
    chunks = []
    try:
        before = os.fstat(descriptor)
        hasher = hashlib.sha256()
        actual_size = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
            actual_size += len(chunk)
            if keep_bytes:
                chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if any(getattr(before, field) != getattr(after, field) for field in STABLE_FIELDS):
        raise ValueError(f"artifact changed while hashing: {relative}")
    current_descriptor = open_at(root, relative, directories)
    try:
        current = os.fstat(current_descriptor)
    finally:
        os.close(current_descriptor)
    if any(getattr(after, field) != getattr(current, field) for field in STABLE_FIELDS):
        raise ValueError(f"artifact pathname changed while hashing: {relative}")
    if actual_size != after.st_size:
        raise ValueError(f"artifact size changed while hashing: {relative}")
    return file_record(after, hasher.hexdigest()), b"".join(chunks)

def create_snapshot(
    root: Path, relative: str, payload: bytes, expected_root_record: dict,
) -> None:
    parts = relative_parts(relative)
    if len(parts) != 1:
        raise ValueError("snapshot must be a direct control-root child")
    flags = (
        os.O_WRONLY | os.O_CREAT | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_BINARY", 0)
    )
    if TEST_MODE:
        root_info = os.stat(root, follow_symlinks=False)
        if stat.S_ISLNK(root_info.st_mode) or not stat.S_ISDIR(root_info.st_mode):
            raise ValueError("control root is unsafe")
        if directory_record(root_info) != expected_root_record:
            raise ValueError("control root identity changed before snapshot creation")
        descriptor = os.open(root / parts[0], flags, 0o600)
    else:
        root_descriptor = os.open(
            root,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | os.O_DIRECTORY | os.O_NOFOLLOW,
        )
        try:
            if directory_record(os.fstat(root_descriptor)) != expected_root_record:
                raise ValueError("control root identity changed before snapshot creation")
            descriptor = os.open(parts[0], flags, 0o600, dir_fd=root_descriptor)
        finally:
            os.close(root_descriptor)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("snapshot is not a regular file")
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    if TEST_MODE:
        current_root = os.stat(root, follow_symlinks=False)
        if (
            stat.S_ISLNK(current_root.st_mode)
            or directory_record(current_root) != expected_root_record
        ):
            raise ValueError("control root identity changed during snapshot creation")

mode = sys.argv[1]
run_root = native(sys.argv[2])
run_id, git_commit = sys.argv[3:5]
control_root = native(sys.argv[5])
expected_token = sys.argv[6]
expected_run_identity, expected_control_identity = sys.argv[7:9]
ledger_arguments = sys.argv[9:]
if mode not in {"capture", "verify"}:
    raise ValueError("invalid final validation mode")
if not ledger_arguments or len(ledger_arguments) % 4:
    raise ValueError("invalid terminal control ledger")
expected_control_records = {}
for index in range(0, len(ledger_arguments), 4):
    relative, identity, size_text, sha256 = ledger_arguments[index:index + 4]
    relative_parts(relative)
    dev_text, separator, ino_text = identity.partition(":")
    if (
        not separator
        or not dev_text.isdecimal()
        or not ino_text.isdecimal()
        or not size_text.isdecimal()
        or not re.fullmatch(r"[0-9a-f]{64}", sha256)
        or relative in expected_control_records
    ):
        raise ValueError("invalid terminal control record")
    expected_control_records[relative] = {
        "dev": int(dev_text), "ino": int(ino_text),
        "size": int(size_text), "sha256": sha256,
    }
run_directories = {}
completion_record, completion_bytes = snapshot_file(
    run_root, "verified_completion.json", run_directories, keep_bytes=True,
)
inventory_record, inventory_bytes = snapshot_file(
    run_root, "manifests/verification_inventory.json", run_directories, keep_bytes=True,
)
if expected_run_identity:
    root_record = run_directories["."]
    if f"{root_record['dev']}:{root_record['ino']}" != expected_run_identity:
        raise ValueError("run root identity mismatch")
completion = load(completion_bytes, "verified completion")
inventory = load(inventory_bytes, "verification inventory")
completion_keys = {
    "schema", "run_id", "git_commit", "classification", "inventory_path",
    "inventory_sha256", "report_completion_sha256", "file_count",
    "total_bytes", "completion_digest",
}
inventory_keys = {
    "schema", "run_id", "git_commit", "classification",
    "report_completion_sha256", "calibration_completion_sha256", "files",
    "file_count", "total_bytes", "completion_digest",
}
if set(completion) != completion_keys or set(inventory) != inventory_keys:
    raise ValueError("schema keys mismatch")
for payload, schema in (
    (completion, "camera_translation_hvrfm.verified_completion.v1"),
    (inventory, "camera_translation_hvrfm.verification_inventory.v1"),
):
    if payload["schema"] != schema:
        raise ValueError("schema mismatch")
    if payload["run_id"] != run_id or payload["git_commit"] != git_commit:
        raise ValueError("run or git identity mismatch")
    if payload["classification"] != "TRANSLATION_ENDPOINTS_READY":
        raise ValueError("terminal classification mismatch")
    require_digest(payload)
if completion["inventory_path"] != "manifests/verification_inventory.json":
    raise ValueError("inventory_path mismatch")
if completion["inventory_sha256"] != hashlib.sha256(inventory_bytes).hexdigest():
    raise ValueError("inventory_sha256 mismatch")
if type(completion["file_count"]) is not int or completion["file_count"] <= 0:
    raise ValueError("completion file_count mismatch")
if type(completion["total_bytes"]) is not int or completion["total_bytes"] <= 0:
    raise ValueError("completion total_bytes mismatch")
if type(inventory["files"]) is not dict or not inventory["files"]:
    raise ValueError("inventory must contain files")
if type(inventory["file_count"]) is not int or inventory["file_count"] != len(inventory["files"]):
    raise ValueError("inventory file_count mismatch")
if type(inventory["total_bytes"]) is not int or inventory["total_bytes"] <= 0:
    raise ValueError("inventory total_bytes mismatch")

def safe_parts(relative: object) -> tuple[str, ...]:
    return relative_parts(relative)

observed_total = 0
member_records = {}
for relative, record in inventory["files"].items():
    parts = safe_parts(relative)
    if type(record) is not dict or set(record) != {"sha256", "bytes"}:
        raise ValueError("inventory record schema mismatch")
    if not isinstance(record["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", record["sha256"]):
        raise ValueError("inventory record digest mismatch")
    if type(record["bytes"]) is not int or record["bytes"] < 0:
        raise ValueError("inventory record size mismatch")
    actual_record, _ = snapshot_file(run_root, relative, run_directories)
    if actual_record["size"] != record["bytes"]:
        raise ValueError(f"verification inventory file size mismatch: {relative}")
    if actual_record["sha256"] != record["sha256"]:
        raise ValueError(f"verification inventory file SHA-256 mismatch: {relative}")
    member_records[relative] = actual_record
    observed_total += actual_record["size"]
if observed_total != inventory["total_bytes"]:
    raise ValueError("inventory byte total mismatch")
for field in ("report_completion_sha256", "calibration_completion_sha256"):
    if not isinstance(inventory[field], str) or not re.fullmatch(r"[0-9a-f]{64}", inventory[field]):
        raise ValueError(f"{field} mismatch")
for relative, field in (
    ("reports/completed.json", "report_completion_sha256"),
    ("calibration/completed.json", "calibration_completion_sha256"),
):
    record = inventory["files"].get(relative)
    if type(record) is not dict or record.get("sha256") != inventory[field]:
        raise ValueError(f"verification inventory {field} file binding mismatch")
if completion["report_completion_sha256"] != inventory["report_completion_sha256"]:
    raise ValueError("report completion mismatch")
if completion["file_count"] != inventory["file_count"]:
    raise ValueError("file_count binding mismatch")
if completion["total_bytes"] != inventory["total_bytes"]:
    raise ValueError("total_bytes binding mismatch")

control_directories = {}
control_records = {}
control_relatives = ["identity.json"]
for stage_name in ("preflight", "prepare", "smoke", "calibration", "report", "verify"):
    control_relatives.extend(
        (f"logs/{stage_name}.out.log", f"logs/{stage_name}.err.log")
    )
if set(expected_control_records) != set(control_relatives):
    raise ValueError("terminal control ledger membership mismatch")
for relative in control_relatives:
    control_records[relative], _ = snapshot_file(
        control_root, relative, control_directories,
    )
    expected_record = expected_control_records[relative]
    if any(
        control_records[relative][field] != expected_record[field]
        for field in ("dev", "ino", "size", "sha256")
    ):
        raise ValueError(f"terminal control record changed: {relative}")
if expected_control_identity:
    root_record = control_directories["."]
    if f"{root_record['dev']}:{root_record['ino']}" != expected_control_identity:
        raise ValueError("control root identity mismatch")

snapshot_payload = {
    "schema": "camera_translation_hvrfm.terminal_snapshot.v1",
    "run_id": run_id,
    "git_commit": git_commit,
    "run_directories": run_directories,
    "top_level": {
        "verified_completion.json": completion_record,
        "manifests/verification_inventory.json": inventory_record,
    },
    "inventory_files": member_records,
    "file_count": inventory["file_count"],
    "total_bytes": inventory["total_bytes"],
    "control_directories": control_directories,
    "control_files": control_records,
}
snapshot_bytes = canonical_bytes(snapshot_payload)
member_digest = hashlib.sha256(canonical_bytes(member_records)).hexdigest()
snapshot_relative = "final_validation_snapshot.json"

if mode == "capture":
    if expected_token:
        raise ValueError("capture mode may not receive a prior token")
    create_snapshot(
        control_root, snapshot_relative, snapshot_bytes, control_directories["."],
    )
else:
    if not expected_token:
        raise ValueError("verify mode requires the capture token")

snapshot_record, observed_snapshot_bytes = snapshot_file(
    control_root, snapshot_relative, control_directories, keep_bytes=True,
)
if mode == "verify":
    decoded_snapshot = load(observed_snapshot_bytes, "terminal snapshot")
    if set(decoded_snapshot) != set(snapshot_payload):
        raise ValueError("terminal snapshot schema mismatch")
    if canonical_bytes(decoded_snapshot) != observed_snapshot_bytes:
        raise ValueError("terminal snapshot is not canonical")
    if observed_snapshot_bytes != snapshot_bytes:
        raise ValueError("terminal snapshot no longer matches the full filesystem scan")

token_payload = {
    "schema": "camera_translation_hvrfm.terminal_snapshot_token.v1",
    "run_id": run_id,
    "git_commit": git_commit,
    "run_root": run_directories["."],
    "member_digest": member_digest,
    "file_count": inventory["file_count"],
    "total_bytes": inventory["total_bytes"],
    "snapshot": snapshot_record,
}
token_text = canonical_bytes(token_payload).decode("utf-8").rstrip("\n")
if mode == "verify":
    decoded_token = load((expected_token + "\n").encode("utf-8"), "snapshot token")
    if set(decoded_token) != set(token_payload):
        raise ValueError("snapshot token schema mismatch")
    if canonical_bytes(decoded_token).decode("utf-8").rstrip("\n") != expected_token:
        raise ValueError("snapshot token is not canonical")
    if token_text != expected_token:
        raise ValueError("terminal filesystem fingerprint changed")
print(token_text, end="")
PY
  )"; then
    return 1
  fi
  [[ -n "$output" && "$output" != *$'\n'* ]] || return 1
  if [[ "$mode" == "capture" ]]; then
    FINAL_SNAPSHOT_TOKEN="$output"
  else
    [[ "$output" == "$expected_token" ]] || return 1
  fi
}

full_runtime_gate
validate_final_completion capture || {
  printf 'Strict verified_completion.json and verification inventory validation failed.\n' >&2
  exit 27
}
full_runtime_gate
validate_final_completion verify "$FINAL_SNAPSHOT_TOKEN" || {
  printf 'Terminal completion/member/control identity-hash barrier failed.\n' >&2
  exit 27
}
if ! release_locks_strict; then
  builtin printf 'Strict lock cleanup failed; verified status was not emitted.\n' >&2
  exit 28
fi
trap - EXIT
builtin printf '[camera-translation-hvrfm] verified %s\n' "$RUN_ROOT"
