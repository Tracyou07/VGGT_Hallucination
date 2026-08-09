#!/usr/bin/env bash
set -euo pipefail

if [[ "${SCANNET_TOS_ACCEPTED:-0}" != "1" ]]; then
  printf 'Set SCANNET_TOS_ACCEPTED=1 only after accepting the official ScanNet terms.\n' >&2
  exit 1
fi

AUTODL_TMP="${AUTODL_TMP:-/root/autodl-tmp}"
REPO_ROOT="${REPO_ROOT:-$AUTODL_TMP/VGGT_Hallucination}"
CO3D_ROOT="${CO3D_ROOT:-$AUTODL_TMP/datasets/co3dv2_2050}"
CO3D_MANIFEST="${CO3D_MANIFEST:-$CO3D_ROOT/download_manifest.json}"
CONDA_PYTHON="${CONDA_PYTHON:-/root/miniconda3/envs/vggt/bin/python}"
POLL_SECONDS="${POLL_SECONDS:-300}"
TARGET_BRANCH="016-camera-refiner-multiscale"

[[ "$POLL_SECONDS" =~ ^[1-9][0-9]*$ ]] || {
  printf 'POLL_SECONDS must be a positive integer.\n' >&2
  exit 1
}
[[ -x "$CONDA_PYTHON" ]] || { printf 'Missing vggt Python: %s\n' "$CONDA_PYTHON" >&2; exit 1; }
[[ -d "$REPO_ROOT/.git" ]] || { printf 'Missing repository: %s\n' "$REPO_ROOT" >&2; exit 1; }

co3d_is_running() {
  pgrep -af 'pre_experiments[.]camera_refiner_data_construction[.]co3d_download' >/dev/null
}

co3d_is_complete() {
  [[ -f "$CO3D_MANIFEST" ]] || return 1
  "$CONDA_PYTHON" - "$CO3D_MANIFEST" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
target = int(manifest.get("target_sequence_count", -1))
actual = int(manifest.get("sequence_count", -2))
if target != 2050 or actual != target or len(manifest.get("sequences", [])) != target:
    raise SystemExit(1)
PY
}

printf '[wait] CO3D manifest: %s\n' "$CO3D_MANIFEST"
while ! co3d_is_complete; do
  if ! co3d_is_running; then
    printf '[blocked] CO3D exited without a valid download_manifest.json\n' >&2
    exit 1
  fi
  printf '[wait] CO3D still active; checking again in %ss\n' "$POLL_SECONDS"
  sleep "$POLL_SECONDS"
done
while co3d_is_running; do
  printf '[wait] CO3D manifest is complete; waiting for downloader process to exit\n'
  sleep 5
done

cd "$REPO_ROOT"
if [[ -n "$(git status --porcelain)" ]]; then
  printf '[blocked] repository has local changes; refusing to switch branches\n' >&2
  git status --short >&2
  exit 1
fi

until git fetch origin "$TARGET_BRANCH"; do
  printf '[network] git fetch failed; retrying in %ss\n' "$POLL_SECONDS" >&2
  sleep "$POLL_SECONDS"
done
if git show-ref --verify --quiet "refs/heads/$TARGET_BRANCH"; then
  git switch 016-camera-refiner-multiscale
else
  git switch --track -c 016-camera-refiner-multiscale "origin/$TARGET_BRANCH"
fi
git merge --ff-only "origin/$TARGET_BRANCH"

printf '[start] CO3D complete; starting ScanNet adaptation acquisition\n'
exec env SCANNET_TOS_ACCEPTED="$SCANNET_TOS_ACCEPTED" \
  bash "$REPO_ROOT/scripts/autodl/camera_refiner_data_construction/prepare_scannet_adaptation200.sh"
