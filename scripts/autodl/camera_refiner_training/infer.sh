#!/usr/bin/env bash
set -euo pipefail

source /root/miniconda3/etc/profile.d/conda.sh
conda activate vggt

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

: "${DATASET_MANIFEST:?Set DATASET_MANIFEST to the frozen data-construction manifest}"
: "${DATASET_ROOT:?Set DATASET_ROOT to the directory containing scene shards}"
: "${LOCAL_RUN_DIR:?Set LOCAL_RUN_DIR to the exact local-global run directory}"
: "${FROZEN_UNITS:?Set FROZEN_UNITS to the translation-unit JSON}"

CHECKPOINT="${CHECKPOINT:-/root/autodl-tmp/results/camera_refiner_training/train/best.pt}"
OUT_DIR="${OUT_DIR:-/root/autodl-tmp/results/camera_refiner_training/inference}"
DEVICE="${DEVICE:-cuda}"
ROLE="${ROLE:-validation}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

python -m pre_experiments.camera_refiner_training.infer \
  --checkpoint "${CHECKPOINT}" \
  --dataset-manifest "${DATASET_MANIFEST}" \
  --dataset-root "${DATASET_ROOT}" \
  --local-run-dir "${LOCAL_RUN_DIR}" \
  --frozen-units "${FROZEN_UNITS}" \
  --out-dir "${OUT_DIR}" \
  --device "${DEVICE}" \
  --role "${ROLE}" \
  "$@"
