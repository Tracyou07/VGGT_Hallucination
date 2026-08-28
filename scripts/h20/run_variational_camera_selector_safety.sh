#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/ubuntu/yjh/vggt/.worktrees/vrfm_selector_safety}"
PYTHON="${PYTHON:-/home/ubuntu/anaconda3/envs/vggt-gx/bin/python}"
BASE_RUN="${BASE_RUN:-/data/yjh/output/variational_camera_selector/selector_A_20260827T153304Z}"
RESULT_ROOT="${RESULT_ROOT:-/data/yjh/output/variational_camera_selector_safety}"
RUN_ID="${RUN_ID:-selector_safety_$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="$RESULT_ROOT/$RUN_ID"
GPU_INDICES="${GPU_INDICES:-2 3}"
MIN_FREE_GIB="${MIN_FREE_GIB:-100}"
LOCK_DIR="$RESULT_ROOT/.locks/$RUN_ID"

[[ "$(hostname)" == "VM-0-11-ubuntu" ]] || {
  printf 'Safety selector requires H20 host VM-0-11-ubuntu.\n' >&2
  exit 1
}
[[ "$(id -un)" == "ubuntu" ]] || {
  printf 'Safety selector requires H20 user ubuntu.\n' >&2
  exit 1
}
[[ -x "$PYTHON" ]] || {
  printf 'Missing frozen Python environment: %s\n' "$PYTHON" >&2
  exit 1
}
[[ -f "$BASE_RUN/verified_completion.json" ]] || {
  printf 'Missing verified base selector: %s\n' "$BASE_RUN" >&2
  exit 1
}
[[ -d "$REPO_ROOT/.git" || -f "$REPO_ROOT/.git" ]] || {
  printf 'Missing safety selector worktree: %s\n' "$REPO_ROOT" >&2
  exit 1
}
[[ -z "$(git -C "$REPO_ROOT" status --porcelain --untracked-files=no)" ]] || {
  printf 'Safety selector worktree has tracked changes.\n' >&2
  exit 1
}

mkdir -p "$RESULT_ROOT" "$RESULT_ROOT/.locks" "$RUN_ROOT/logs"
available_kib="$(df --output=avail -k "$RESULT_ROOT" | tail -n 1 | tr -d ' ')"
required_kib="$((MIN_FREE_GIB * 1024 * 1024))"
(( available_kib >= required_kib )) || {
  printf 'Insufficient /data space: need at least %s GiB free.\n' "$MIN_FREE_GIB" >&2
  exit 1
}

read -r -a gpu_array <<< "$GPU_INDICES"
(( ${#gpu_array[@]} >= 1 )) || {
  printf 'At least one GPU index is required.\n' >&2
  exit 1
}
for gpu in "${gpu_array[@]}"; do
  [[ "$gpu" =~ ^[0-7]$ ]] || {
    printf 'Invalid H20 GPU index: %s\n' "$gpu" >&2
    exit 1
  }
  used_mib="$(nvidia-smi --id="$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
  (( used_mib < 10000 )) || {
    printf 'GPU %s is already using %s MiB; refusing to collide.\n' "$gpu" "$used_mib" >&2
    exit 1
  }
done

if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  printf 'Safety run lock already exists: %s\n' "$LOCK_DIR" >&2
  exit 1
fi
cleanup() {
  rmdir "$LOCK_DIR" 2>/dev/null || true
}
trap cleanup EXIT

unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
cd "$REPO_ROOT"
exec > >(tee -a "$RUN_ROOT/logs/runner.log") 2>&1

common=(
  --run-root "$RUN_ROOT"
  --base-run "$BASE_RUN"
  --device cuda
  --steps 800
  --batch-size 1
  --learning-rate 1e-4
  --tau 0.05
  --seed 20260828
  --d-model 128
  --checkpoint-interval 25
)

if [[ -f "$RUN_ROOT/verified_completion.json" ]]; then
  CUDA_VISIBLE_DEVICES="${gpu_array[0]}" "$PYTHON" -m \
    pre_experiments.variational_camera_selector.safety_pipeline \
    --stage verify "${common[@]}"
  printf '[safety] already verified: %s\n' "$RUN_ROOT/verified_completion.json"
  exit 0
fi

scenes=(
  scene0000_00
  scene0013_02
  scene0029_01
  scene0691_00
  scene0084_01
  scene0121_01
  scene0207_01
  scene0280_00
)

printf '[safety] run_id=%s gpus=%s free_kib=%s\n' "$RUN_ID" "$GPU_INDICES" "$available_kib"
for ((start=0; start<${#scenes[@]}; start+=${#gpu_array[@]})); do
  pids=()
  labels=()
  for ((slot=0; slot<${#gpu_array[@]} && start+slot<${#scenes[@]}; slot++)); do
    scene="${scenes[start+slot]}"
    gpu="${gpu_array[slot]}"
    labels+=("$scene")
    (
      export CUDA_VISIBLE_DEVICES="$gpu"
      "$PYTHON" -m pre_experiments.variational_camera_selector.safety_pipeline \
        --stage oof-fold --fold-scene "$scene" "${common[@]}"
    ) >"$RUN_ROOT/logs/${scene}.out.log" 2>"$RUN_ROOT/logs/${scene}.err.log" &
    pids+=("$!")
    printf '[safety] launched held=%s gpu=%s pid=%s\n' "$scene" "$gpu" "$!"
  done
  for ((slot=0; slot<${#pids[@]}; slot++)); do
    if ! wait "${pids[slot]}"; then
      scene="${labels[slot]}"
      printf '[safety] fold failed: %s\n' "$scene" >&2
      tail -80 "$RUN_ROOT/logs/${scene}.err.log" >&2 || true
      exit 1
    fi
    printf '[safety] fold complete: %s\n' "${labels[slot]}"
  done
done

export CUDA_VISIBLE_DEVICES="${gpu_array[0]}"
"$PYTHON" -m pre_experiments.variational_camera_selector.safety_pipeline \
  --stage finalize "${common[@]}"
[[ -f "$RUN_ROOT/verified_completion.json" ]] || {
  printf 'Safety selector did not publish verified_completion.json.\n' >&2
  exit 1
}
printf '[safety] verified: %s\n' "$RUN_ROOT/verified_completion.json"
