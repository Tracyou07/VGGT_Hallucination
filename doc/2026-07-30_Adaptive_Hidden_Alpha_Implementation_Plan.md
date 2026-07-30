# Adaptive Hidden Alpha Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:executing-plans` to implement this plan task-by-task.

**Goal:** Test whether prediction-only local-global consistency can select a
scene-specific hidden interpolation strength that improves over fixed
`alpha=0.02`.

**Architecture:** Calibration joins the complete fixed-alpha replacement table
with local-global prediction scores by scene. It derives an oracle label only
from calibration GT metrics, fits a standardized ridge selector over three
predeclared prediction-only features, and freezes coefficients, normalization,
alpha choices, split provenance, and source digests. Holdout loads that frozen
selector, predicts one alpha per scene before reading GT, and runs selected
units plus all five controls at the same scene alpha.

**Tech Stack:** Python, NumPy, PyTorch, existing VGGT Camera Head and strict
CSV/JSON/NPZ artifact helpers.

## Global Constraints

- Predictions are evaluated only after alignment; GT remains raw.
- Holdout alpha selection must never consume GT-derived fields.
- Candidate alphas are exactly `0.01`, `0.02`, and `0.05`.
- Features are scene medians of global-local token cosine, global-local pose
  translation, and local-local pose translation.
- Ridge regularization is fixed at `1.0`; ties choose the smaller alpha.
- Calibration reports leave-one-scene-out performance against fixed `0.02`.
- Existing result directories remain untouched and are never staged.

---

### Task 1: Frozen Selector Core

**Files:**
- Create: `pre_experiments/camera_hidden_state_attribution/adaptive_alpha.py`
- Test: `tests/camera_hidden_state_attribution/test_adaptive_alpha.py`

- [x] Write failing tests for prediction-only feature aggregation, oracle
  labels, deterministic ridge fitting, quantization, digest validation, and
  rejection of GT-named feature fields.
- [x] Run the focused test and confirm RED due to the missing module.
- [x] Implement the minimal NumPy selector and authenticated frozen manifest.
- [x] Run the focused test and confirm GREEN.

### Task 2: Calibration and Holdout Runner

**Files:**
- Create:
  `pre_experiments/camera_hidden_state_attribution/run_adaptive_replacement.py`
- Modify:
  `pre_experiments/camera_hidden_state_attribution/replacement_analyze.py`
- Test:
  `tests/camera_hidden_state_attribution/test_adaptive_replacement_runner.py`

- [x] Write failing tests for calibration joins, LOOCV output, scene-specific
  holdout alpha prediction, all-control execution, and split/digest mismatch.
- [x] Implement `calibration` as CPU-only fitting and `holdout` as GPU Camera
  Head replay using one frozen alpha per scene.
- [x] Write per-scene selected alpha, selected/control aligned deltas, alpha
  histogram, and scene-bootstrap confidence intervals.
- [x] Run focused tests and confirm GREEN.

### Task 3: AutoDL Entry Point and Verification

**Files:**
- Create: `scripts/autodl/run_camera_hidden_adaptive_alpha.sh`
- Modify: `pre_experiments/camera_hidden_state_attribution/README.md`
- Test: `tests/camera_hidden_state_attribution/test_adaptive_alpha_autodl.py`

- [x] Add ordered `calibration`, `holdout`, and numeric `export` stages with
  explicit source paths and resumable run pointers.
- [x] Document exact AutoDL environment variables and commands.
- [x] Run the hidden-state test suite, `compileall`, shell syntax validation,
  and `git diff --check`.
- [x] Commit and push only code, tests, plan, and documentation.
