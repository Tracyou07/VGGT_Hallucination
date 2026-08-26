#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/ubuntu/yjh/vggt/.worktrees/camera_velocity_ambiguity_02_pre_experiment}"
DATA_ROOT="${DATA_ROOT:-/data/yjh/share/datasets/ScanNet}"
PROCESSED_ROOT="${PROCESSED_ROOT:-$DATA_ROOT/processed_cva02_v1}"
CKPT_DIR="${CKPT_DIR:-/data/yjh/share/pretrained/VGGT-1B}"
RESULT_ROOT="${RESULT_ROOT:-/data/output/camera_velocity_ambiguity}"
CONDA_ENV="${CONDA_ENV:-/home/ubuntu/anaconda3/envs/vggt-gx}"
DEVICE="${DEVICE:-cuda}"
GPU_INDEX="${GPU_INDEX:-1}"
SMOKE_SCENE_LIMIT="1"
CALIBRATION_SCENE_LIMIT="10"
MARKER="$DATA_ROOT/verified_completion.json"
EXPECTED_BRANCH="codex/camera_velocity_ambiguity_02_pre_experiment"
MIN_FREE_GIB="${MIN_FREE_GIB:-50}"
RUN_ID="${RUN_ID:-cva02_$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="$RESULT_ROOT/$RUN_ID"
LOG_DIR="$RUN_ROOT/logs"
LOCK_DIR="$RESULT_ROOT/.locks/$RUN_ID"

[[ "$(hostname)" == "VM-0-11-ubuntu" ]] || {
  printf 'CVA02 formal runner requires H20 host VM-0-11-ubuntu.\n' >&2
  exit 1
}
[[ "$(whoami)" == "ubuntu" ]] || {
  printf 'CVA02 formal runner requires H20 user ubuntu.\n' >&2
  exit 1
}
[[ "$DEVICE" == "cuda" ]] || {
  printf 'CVA02 formal execution requires DEVICE=cuda.\n' >&2
  exit 1
}
[[ -d "$REPO_ROOT/.git" || -f "$REPO_ROOT/.git" ]] || {
  printf 'Missing CVA02 worktree: %s\n' "$REPO_ROOT" >&2
  exit 1
}
[[ "$(cd "$REPO_ROOT" && git branch --show-current)" == "$EXPECTED_BRANCH" ]] || {
  printf 'Wrong CVA02 branch in %s.\n' "$REPO_ROOT" >&2
  exit 1
}
[[ -f "$MARKER" ]] || {
  printf 'Missing verified_completion.json: %s\n' "$MARKER" >&2
  exit 1
}
[[ -f "$CKPT_DIR/model.safetensors" ]] || {
  printf 'Missing local VGGT checkpoint: %s/model.safetensors\n' "$CKPT_DIR" >&2
  exit 1
}
[[ -x "$CONDA_ENV/bin/python" ]] || {
  printf 'Missing frozen Python environment: %s\n' "$CONDA_ENV" >&2
  exit 1
}
[[ "$GPU_INDEX" =~ ^[0-7]$ ]] || {
  printf 'GPU_INDEX must be an H20 index from 0 through 7.\n' >&2
  exit 1
}

mkdir -p "$RESULT_ROOT" "$RESULT_ROOT/.locks" "$LOG_DIR" "$PROCESSED_ROOT"
available_kib="$(df --output=avail -k "$RESULT_ROOT" | tail -n 1 | tr -d ' ')"
required_kib="$((MIN_FREE_GIB * 1024 * 1024))"
(( available_kib >= required_kib )) || {
  printf 'Insufficient /data space: need at least %s GiB free.\n' "$MIN_FREE_GIB" >&2
  exit 1
}
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader
gpu_used_mib="$(nvidia-smi --id="$GPU_INDEX" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
(( gpu_used_mib < 10000 )) || {
  printf 'GPU %s is already using %s MiB; refusing to collide.\n' "$GPU_INDEX" "$gpu_used_mib" >&2
  exit 1
}

if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  printf 'Run lock already exists: %s\n' "$LOCK_DIR" >&2
  exit 1
fi
cleanup() {
  rmdir "$LOCK_DIR" 2>/dev/null || true
}
trap cleanup EXIT

exec > >(tee -a "$LOG_DIR/run.log") 2>&1
export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
cd "$REPO_ROOT"

CALIBRATION_COMPLETE="$RUN_ROOT/manifests/calibration_complete.json"
if [[ -f "$CALIBRATION_COMPLETE" ]]; then
  printf '[cva02] already complete: %s\n' "$CALIBRATION_COMPLETE"
  exit 0
fi

printf '[cva02] run_id=%s gpu=%s free_kib=%s\n' "$RUN_ID" "$GPU_INDEX" "$available_kib"
"$CONDA_ENV/bin/python" -m pre_experiments.camera_velocity_ambiguity_02.pipeline \
  --stage smoke \
  --run-id "$RUN_ID" \
  --scene-limit "$SMOKE_SCENE_LIMIT" \
  --data-root "$DATA_ROOT" \
  --processed-root "$PROCESSED_ROOT" \
  --checkpoint-dir "$CKPT_DIR" \
  --result-root "$RESULT_ROOT" \
  --marker "$MARKER" \
  --device "$DEVICE"

SMOKE_COMPLETE="$RUN_ROOT/manifests/smoke_complete.json"
[[ -f "$SMOKE_COMPLETE" ]] || {
  printf 'One-scene production smoke did not publish smoke_complete.json.\n' >&2
  exit 1
}
"$CONDA_ENV/bin/python" - "$SMOKE_COMPLETE" "$RUN_ID" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

path, run_id = Path(sys.argv[1]), sys.argv[2]
payload = json.loads(path.read_text(encoding="utf-8"))
digest = payload.pop("completion_digest", None)
canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
if digest != hashlib.sha256(canonical).hexdigest():
    raise SystemExit("smoke completion digest mismatch")
if payload.get("run_id") != run_id or payload.get("production_frame_count") != 500:
    raise SystemExit("smoke completion identity mismatch")
if payload.get("global_runs") != 1 or payload.get("local_windows") != 9:
    raise SystemExit("smoke did not use the exact production path")
PY

printf '[cva02] smoke passed; expanding the same run to ten calibration scenes\n'
"$CONDA_ENV/bin/python" -m pre_experiments.camera_velocity_ambiguity_02.pipeline \
  --stage calibration \
  --run-id "$RUN_ID" \
  --scene-limit "$CALIBRATION_SCENE_LIMIT" \
  --data-root "$DATA_ROOT" \
  --processed-root "$PROCESSED_ROOT" \
  --checkpoint-dir "$CKPT_DIR" \
  --result-root "$RESULT_ROOT" \
  --marker "$MARKER" \
  --device "$DEVICE"

[[ -f "$CALIBRATION_COMPLETE" ]] || {
  printf 'Ten-scene calibration did not publish calibration_complete.json.\n' >&2
  exit 1
}
printf '[cva02] calibration complete: %s\n' "$CALIBRATION_COMPLETE"
