# Local-Global Consistency Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` or `superpowers:executing-plans`.

**Goal:** Build the fixed four-scene Round 2A experiment that tests whether
prediction-only local-global Camera Token and pose disagreement predicts
long-context aligned pose degradation.

**Architecture:** Reuse the published Round 1.5 500-frame artifacts as global
predictions. Run frozen camera-only VGGT on nine deterministic 100-frame local
windows, store raw high-dimensional diagnostics only in the external AutoDL run,
then perform all alignment, aggregation, threshold fitting, and GT validation in
a CPU analysis pass. VGGT's default API and weights remain unchanged.

**Tech Stack:** Python 3.10+, PyTorch 2.3, NumPy, `unittest`, Bash, existing VGGT
checkpoint and ScanNet preparation tools.

## Global Constraints

- Implement only Round 2A; do not implement scene screening, dataset splits,
  latent pair generation, token replacement, optimization, or training.
- Detection scores use predictions only. GT remains raw and appears only in
  separately named evaluation columns after predictions are scored.
- Use exact Round 1.5 500-frame IDs and fixed windows `(length=100, stride=50)`.
- Publish only CSV/JSON scalar artifacts. Local Camera Tokens remain external.
- Unit tests require no CUDA, checkpoint, network, or ScanNet files.

## Task 1: Deterministic Windows And Prediction Alignment

**Files:**

- Create `pre_experiments/local_global_consistency/windows.py`
- Create `pre_experiments/local_global_consistency/alignment.py`
- Create `tests/camera_iteration/test_local_global_consistency.py`

**Interfaces:**

- `build_sliding_windows(frame_ids, length=100, stride=50)` returns immutable
  window records with start index, frame IDs, and boundary distances.
- `align_prediction_trajectories(reference_c2w, moving_c2w)` estimates an
  orientation-preserving prediction-to-prediction Sim(3) and returns aligned
  moving poses plus per-frame translation/rotation residuals.

**Steps:**

1. Write tests proving 500 IDs produce starts `0..400`, internal coverage two,
   edge coverage one, and invalid/non-covering parameters fail.
2. Write a synthetic Sim(3) test proving prediction-only alignment removes
   global scale/rotation/translation without reading GT.
3. Implement the minimum pure NumPy helpers and run the focused tests.

## Task 2: Local Artifacts And GPU Runner

**Files:**

- Create `pre_experiments/local_global_consistency/artifacts.py`
- Create `pre_experiments/local_global_consistency/run_study.py`
- Extend `tests/camera_iteration/test_local_global_consistency.py`

**Interfaces:**

- Each external window directory contains `window_diagnostics.npz` with exactly
  frame IDs, normalized Camera Tokens, raw predicted c2w, and raw GT c2w.
- The CLI consumes processed ScanNet, an official local checkpoint, a published
  Round 1.5 run, output directory, device, scene limit, window length/stride,
  and four fixed Camera Head iterations.
- Completed windows are resumable only when run ID, frame IDs, and boundaries
  match.

**Steps:**

1. Write artifact schema, validation, parser, and resume-contract tests.
2. Implement camera-only local inference by reusing existing image loading,
   checkpoint loading, pose conversion, and metadata helpers.
3. Verify no depth/point/track head is enabled and no global inference reruns.

## Task 3: CPU Score And GT Validation Analysis

**Files:**

- Create `pre_experiments/local_global_consistency/metrics.py`
- Create `pre_experiments/local_global_consistency/analyze.py`
- Extend `tests/camera_iteration/test_local_global_consistency.py`

**Interfaces:**

- Local-local rows compare each adjacent-window overlap.
- Global-local rows compare every local observation with matching published
  global prediction.
- Per-frame aggregation reports median global-local score, local-local
  reliability, aligned global error, median independently aligned local error,
  and global-minus-local error growth.
- Stable-control p95 local-local thresholds are fitted before gated scores.
- Scene summaries report Pearson, Spearman, quartiles, and valid frame counts.

**Steps:**

1. Write synthetic tests separating prediction-only score inputs from raw-GT
   evaluation labels and rejecting aligned/missing GT contracts.
2. Implement cosine, overlap alignment, independent prediction-to-raw-GT
   evaluation, per-frame aggregation, p95 thresholds, and correlations.
3. Write deterministic CSV/JSON outputs and test the complete CPU analysis.

## Task 4: AutoDL And Scalar Publishing

**Files:**

- Create `scripts/autodl/run_local_global_consistency.sh`
- Create `scripts/autodl/local_global_consistency/export_numeric_results.py`
- Modify `.gitignore`
- Modify `tests/camera_iteration/test_autodl_scripts.py`
- Modify `tests/camera_iteration/test_contracts.py`

**Steps:**

1. Test that the runner only validates existing environment, checkpoint, data,
   and global artifacts; it performs no install or download.
2. Test an exact scalar-only exporter contract and rejection of NPY/NPZ/model
   artifacts.
3. Implement the fixed AutoDL runner, exporter, and JSON/CSV Git whitelist.

## Task 5: Reproduction Documentation And Verification

**Files:**

- Create `pre_experiments/local_global_consistency/README.md`
- Create `log/2026-07-21_round2_local_global_consistency.md`
- Modify `AGENTS.md`

**Steps:**

1. Document one-scene smoke, four-scene formal run, output interpretation, and
   the prediction/GT boundary.
2. Run `python -m unittest discover -s tests` in the AutoDL `vggt` environment.
3. Run `bash -n scripts/autodl/run_local_global_consistency.sh`,
   `git diff --check`, and CLI help/compile checks.
4. Commit and push the reproducible branch before remote execution.
