#!/usr/bin/env bash
set -euo pipefail

if [[ "${SCANNET_TOS_ACCEPTED:-0}" != "1" ]]; then
  printf 'Set SCANNET_TOS_ACCEPTED=1 only after accepting the official ScanNet terms.\n' >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
AUTODL_TMP="${AUTODL_TMP:-/root/autodl-tmp}"
CONDA_ROOT="${CONDA_ROOT:-/root/miniconda3}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-vggt}"
CONDA_SH="$CONDA_ROOT/etc/profile.d/conda.sh"
SCANNET_ROOT="${SCANNET_ROOT:-$AUTODL_TMP/datasets/scannetv2}"
PROCESS_DIR="${PROCESS_DIR:-$SCANNET_ROOT/process_scannet}"
STATE_DIR="${STATE_DIR:-$SCANNET_ROOT/adaptation200_state}"
RAW_DIR="${RAW_DIR:-$STATE_DIR/raw_sens/scans}"
STAGING_DIR="${STAGING_DIR:-$STATE_DIR/process_staging}"
QUARANTINE_DIR="${QUARANTINE_DIR:-$STATE_DIR/quarantine}"
RESULTS_ROOT="${RESULTS_ROOT:-$AUTODL_TMP/results}"
RESULT_DIR="${RESULT_DIR:-$RESULTS_ROOT/camera_refiner_data_construction/scannet_adaptation200}"
OFFICIAL_TRAIN="${OFFICIAL_TRAIN:-$REPO_ROOT/configs/scannetv2_train_official.txt}"
EXCLUDED_SCENES="${EXCLUDED_SCENES:-$REPO_ROOT/configs/fastvggt_scannet50.txt}"
CANDIDATE_MANIFEST="${CANDIDATE_MANIFEST:-$STATE_DIR/candidate_manifest.json}"
ACCEPTED_SCENES="${ACCEPTED_SCENES:-$STATE_DIR/accepted_scenes.txt}"
REJECTED_SCENES="${REJECTED_SCENES:-$STATE_DIR/rejected_scenes.tsv}"
FINAL_MANIFEST="${FINAL_MANIFEST:-$RESULT_DIR/manifest.json}"

TARGET_SCENES="${TARGET_SCENES:-200}"
REFINER_TRAIN_SCENES="${REFINER_TRAIN_SCENES:-160}"
VALIDATION_SCENES="${VALIDATION_SCENES:-20}"
SELECTOR_TRAIN_SCENES="${SELECTOR_TRAIN_SCENES:-20}"
MIN_MATCHING_FRAMES="${MIN_MATCHING_FRAMES:-500}"
MIN_FREE_GIB="${MIN_FREE_GIB:-60}"
SELECTION_SEED="${SELECTION_SEED:-33}"
DOWNLOAD_RETRIES="${DOWNLOAD_RETRIES:-10}"
SCANNET_V1_SCANS_URL="${SCANNET_V1_SCANS_URL:-http://kaldir.vc.cit.tum.de/scannet/v1/scans}"
SCANNET_V2_SCANS_URL="${SCANNET_V2_SCANS_URL:-http://kaldir.vc.cit.tum.de/scannet/v2/scans}"
SCANNET_CURL="${SCANNET_CURL:-curl}"
SCANNET_CURL_ARGS="${SCANNET_CURL_ARGS:---retry 20 --retry-delay 5 --retry-all-errors --speed-time 180 --speed-limit 1024}"

# shellcheck source=../scannet_download.sh
source "$REPO_ROOT/scripts/autodl/scannet_download.sh"

for value in \
  "$TARGET_SCENES" "$REFINER_TRAIN_SCENES" "$VALIDATION_SCENES" \
  "$SELECTOR_TRAIN_SCENES" "$MIN_MATCHING_FRAMES" "$MIN_FREE_GIB" \
  "$DOWNLOAD_RETRIES"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || {
    printf 'Scene counts, frame thresholds, retries, and disk guard must be positive integers.\n' >&2
    exit 1
  }
done
[[ "$SELECTION_SEED" =~ ^[0-9]+$ ]] || {
  printf 'SELECTION_SEED must be non-negative.\n' >&2
  exit 1
}
role_total=$((REFINER_TRAIN_SCENES + VALIDATION_SCENES + SELECTOR_TRAIN_SCENES))
[[ "$role_total" -eq "$TARGET_SCENES" ]] || {
  printf 'Role counts sum to %s, expected TARGET_SCENES=%s.\n' "$role_total" "$TARGET_SCENES" >&2
  exit 1
}
[[ -f "$CONDA_SH" ]] || { printf 'Missing Conda activation: %s\n' "$CONDA_SH" >&2; exit 1; }
[[ -f "$OFFICIAL_TRAIN" ]] || { printf 'Missing official ScanNet train split: %s\n' "$OFFICIAL_TRAIN" >&2; exit 1; }
[[ -f "$EXCLUDED_SCENES" ]] || { printf 'Missing protected ScanNet-50 list: %s\n' "$EXCLUDED_SCENES" >&2; exit 1; }

# Reuse the provisioned AutoDL environment. This script installs no packages.
# shellcheck source=/dev/null
source "$CONDA_SH"
conda activate "$CONDA_ENV_NAME"
cd "$REPO_ROOT"

mkdir -p "$STATE_DIR" "$RAW_DIR" "$STAGING_DIR" "$QUARANTINE_DIR" "$PROCESS_DIR" "$RESULT_DIR"
touch "$ACCEPTED_SCENES" "$REJECTED_SCENES"
exec 9>"$STATE_DIR/.prepare.lock"
flock --nonblock 9 || {
  printf 'Another ScanNet adaptation preparation is active: %s\n' "$STATE_DIR" >&2
  exit 1
}

python -m pre_experiments.camera_refiner_data_construction.scannet_adaptation \
  build-candidate-manifest \
  --official-train "$OFFICIAL_TRAIN" \
  --exclude-scenes "$EXCLUDED_SCENES" \
  --output "$CANDIDATE_MANIFEST" \
  --seed "$SELECTION_SEED" \
  --min-frames "$MIN_MATCHING_FRAMES" \
  --refiner-train-scenes "$REFINER_TRAIN_SCENES" \
  --validation-scenes "$VALIDATION_SCENES" \
  --selector-train-scenes "$SELECTOR_TRAIN_SCENES"

mapfile -t candidates < <(
  python -m pre_experiments.camera_refiner_data_construction.scannet_adaptation \
    list-candidates --manifest "$CANDIDATE_MANIFEST"
)
declare -A accepted=()
declare -A rejected=()
while IFS= read -r scene; do
  [[ -n "$scene" ]] && accepted["$scene"]=1
done < "$ACCEPTED_SCENES"
while IFS=$'\t' read -r scene _; do
  [[ -n "$scene" ]] && rejected["$scene"]=1
done < "$REJECTED_SCENES"

frame_count() {
  python -m pre_experiments.camera_refiner_data_construction.scannet_adaptation \
    processed-scene-frame-count --scene-dir "$1"
}

accepted_count() {
  awk 'NF {count += 1} END {print count + 0}' "$ACCEPTED_SCENES"
}

require_download_space() {
  local free_kib
  local required_kib
  free_kib="$(df -Pk "$AUTODL_TMP" | awk 'NR == 2 {print $4}')"
  required_kib=$((MIN_FREE_GIB * 1024 * 1024))
  if ((free_kib < required_kib)); then
    printf '[disk-stop] free=%s KiB, required=%s KiB before next ScanNet download\n' \
      "$free_kib" "$required_kib" >&2
    exit 75
  fi
}

cleanup_scene_download() {
  local scene="$1"
  local sens="$RAW_DIR/$scene/$scene.sens"

  rm -f -- "$sens"
  rm -f -- "$sens.partial"
  rmdir --ignore-fail-on-non-empty "$(dirname "$sens")" 2>/dev/null || true
}

for scene in "${candidates[@]}"; do
  if [[ -n "${accepted[$scene]:-}" || -n "${rejected[$scene]:-}" ]]; then
    cleanup_scene_download "$scene"
    continue
  fi
  if (( $(accepted_count) >= TARGET_SCENES )); then
    break
  fi

  destination="$PROCESS_DIR/$scene"
  existing_count="$(frame_count "$destination")"
  if ((existing_count >= MIN_MATCHING_FRAMES)); then
    printf '%s\n' "$scene" >> "$ACCEPTED_SCENES"
    accepted["$scene"]=1
    printf '[reuse] %s frames=%s accepted=%s/%s\n' \
      "$scene" "$existing_count" "$(accepted_count)" "$TARGET_SCENES"
    cleanup_scene_download "$scene"
    continue
  fi

  require_download_space
  sens="$RAW_DIR/$scene/$scene.sens"
  download_asset "$scene" .sens "$sens"

  extract_root="$(mktemp -d "$STAGING_DIR/${scene}.XXXXXX")"
  scene_list="$extract_root/scene.txt"
  printf '%s\n' "$scene" > "$scene_list"
  if ! python "$REPO_ROOT/scripts/autodl/scannet/extract_scannet_sens.py" \
    --raw-dir "$RAW_DIR" \
    --out-dir "$extract_root/processed" \
    --scene-list "$scene_list" \
    --scene-limit 0; then
    printf '[extract-failed] %s; retaining %s for retry\n' "$scene" "$sens" >&2
    rm -rf -- "$extract_root"
    exit 1
  fi

  extracted="$extract_root/processed/$scene"
  extracted_count="$(frame_count "$extracted")"
  if ((extracted_count >= MIN_MATCHING_FRAMES)); then
    if [[ -e "$destination" ]]; then
      quarantine="$QUARANTINE_DIR/${scene}.$(date +%Y%m%d%H%M%S)"
      mv -- "$destination" "$quarantine"
    fi
    mv -- "$extracted" "$destination"
    printf '%s\n' "$scene" >> "$ACCEPTED_SCENES"
    accepted["$scene"]=1
    printf '[accepted] %s frames=%s total=%s/%s\n' \
      "$scene" "$extracted_count" "$(accepted_count)" "$TARGET_SCENES"
  else
    printf '%s\t%s\tinsufficient_matching_frames\n' \
      "$scene" "$extracted_count" >> "$REJECTED_SCENES"
    rejected["$scene"]=1
    printf '[rejected] %s frames=%s need=%s\n' \
      "$scene" "$extracted_count" "$MIN_MATCHING_FRAMES"
  fi
  cleanup_scene_download "$scene"
  rm -rf -- "$extract_root"
done

actual="$(accepted_count)"
if ((actual != TARGET_SCENES)); then
  printf 'Only %s/%s valid ScanNet scenes were prepared.\n' "$actual" "$TARGET_SCENES" >&2
  exit 1
fi

python -m pre_experiments.camera_refiner_data_construction.scannet_adaptation \
  finalize \
  --candidate-manifest "$CANDIDATE_MANIFEST" \
  --accepted-scenes "$ACCEPTED_SCENES" \
  --processed-root "$PROCESS_DIR" \
  --output "$FINAL_MANIFEST"
printf '[done] ScanNet adaptation manifest: %s\n' "$FINAL_MANIFEST"
