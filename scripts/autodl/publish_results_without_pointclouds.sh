#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

AUTODL_TMP="${AUTODL_TMP:-/root/autodl-tmp}"
RESULT_DIR="${RESULT_DIR:-$AUTODL_TMP/vggt_hallucination/results}"
RESULT_UPLOAD_DIR="${RESULT_UPLOAD_DIR:-$REPO_ROOT/results/scannet_hallucination}"
PUSH_RESULTS="${PUSH_RESULTS:-0}"
COMMIT_MSG="${COMMIT_MSG:-Upload ScanNet hallucination results}"

cd "$REPO_ROOT"

python "$SCRIPT_DIR/collect_results_without_pointclouds.py" \
    --src "$RESULT_DIR" \
    --dst "$RESULT_UPLOAD_DIR"

echo "[publish] collected results under $RESULT_UPLOAD_DIR"
echo "[publish] point-cloud file types are excluded by whitelist"

if [[ "$PUSH_RESULTS" != "1" ]]; then
    echo "[publish] set PUSH_RESULTS=1 to commit and push these collected files"
    exit 0
fi

git add "$RESULT_UPLOAD_DIR"
if git diff --cached --quiet; then
    echo "[publish] no result changes to commit"
    exit 0
fi

git commit -m "$COMMIT_MSG"
git push origin "$(git branch --show-current)"
