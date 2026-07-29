# Camera Hidden-State Attribution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible token-replay pipeline that identifies and validates
Camera Head hidden units associated with translation, rotation, and FoV across
100-frame and 500-frame contexts.

**Architecture:** Extend the existing opt-in Camera Head trace without changing default
predictions, then replay saved normalized camera tokens through the checkpoint Camera
Head. Pure NumPy analysis computes per-unit contribution drift, freezes calibration
rankings, and evaluates holdout stability; controlled hidden-unit ablation validates
output specificity.

**Tech Stack:** Python 3.10+, PyTorch, NumPy, standard-library `unittest`, shell.

## Global Constraints

- Use the frozen 10-scene calibration and 40-scene holdout split.
- Unit identity is `(iteration, hidden_index)`; freeze 64 units per output group.
- Unit score is scene-equal mean contribution drift multiplied by static group
  specificity; ties use iteration then hidden index.
- Prediction rankings remain prediction-only. GT is raw and appears only in separately
  named validation metrics.
- Raw high-dimensional NPZ traces remain outside Git; publish only CSV/JSON.
- Do not rerun image preprocessing or Aggregator inference.

---

### Task 1: Trace and Ablation Interface

**Files:**
- Modify: `vggt/layers/mlp.py`
- Modify: `vggt/heads/camera_head.py`
- Test: `tests/camera_hidden_state_attribution/test_camera_trace.py`

**Interfaces:**
- Produces: `Mlp.forward_features(x)`, `Mlp.forward_head(hidden)`.
- Produces: trace keys `trunk_output_list` and `pose_branch_hidden_list`.
- Produces: optional `hidden_ablation_mask` with shape `[iterations, hidden_dim]`.

- [ ] Add failing tests proving traced and untraced predictions match, hidden tensors
  have expected shapes, and an all-false mask changes nothing.
- [ ] Run `python -m unittest tests.camera_hidden_state_attribution.test_camera_trace -v`
  and confirm failure before implementation.
- [ ] Split `Mlp.forward` into reusable feature/head methods and implement opt-in trace
  plus per-iteration boolean ablation in `CameraHead`.
- [ ] Remove the misleading `pose_tokens_modulated_list` trace field and update existing
  Camera Head trace tests.
- [ ] Run the focused tests and commit.

### Task 2: Contribution and Ranking Core

**Files:**
- Create: `pre_experiments/camera_hidden_state_attribution/__init__.py`
- Create: `pre_experiments/camera_hidden_state_attribution/attribution.py`
- Test: `tests/camera_hidden_state_attribution/test_attribution.py`

**Interfaces:**
- Produces: `group_weight_norms(weight) -> dict[str, ndarray]`.
- Produces: `contribution_drift(global_hidden, local_hidden, weight)`.
- Produces: `freeze_unit_sets(scene_rows, top_k=64, seed=33)`.

- [ ] Write synthetic failing tests for exact per-unit contribution, scene-equal
  aggregation, deterministic tie-breaking, disjoint output schemas, and random controls
  matched to selected iteration counts.
- [ ] Run the focused test and confirm failure.
- [ ] Implement translation `0:3`, rotation `3:7`, and FoV `7:9` contribution norms.
  Rank `(iteration, unit)` by `mean_scene_drift * group_specificity`.
- [ ] Run focused tests and commit.

### Task 3: Token Replay and Artifact Contracts

**Files:**
- Create: `pre_experiments/camera_hidden_state_attribution/artifacts.py`
- Create: `pre_experiments/camera_hidden_state_attribution/run_study.py`
- Test: `tests/camera_hidden_state_attribution/test_run_study.py`

**Interfaces:**
- Consumes: global/local `normalized_camera_tokens`, stored pose, split manifest, Camera
  Head checkpoint weights.
- Produces: per-scene `unit_statistics.npz`, `intervention_summary.json`, and
  `complete.json`.

- [ ] Write failing CPU tests using a small Camera Head and synthetic global/local
  token artifacts. Require exact frame-ID matching, replay-pose verification, resumable
  identity checks, and rejection of aligned/mutated GT as input.
- [ ] Run focused tests and confirm failure.
- [ ] Implement `smoke`, `calibration`, and `holdout` CLI stages. Calibration writes
  `frozen_units.json`; holdout requires it and refuses digest mismatches.
- [ ] Implement selected-unit and matched-random ablation. Decode final poses and report
  camera-center displacement, relative rotation change, and FoV change separately.
- [ ] Run focused tests and commit.

### Task 4: Numeric Aggregation and Export

**Files:**
- Create: `pre_experiments/camera_hidden_state_attribution/analyze.py`
- Create: `scripts/autodl/camera_hidden_state_attribution/export_numeric_results.py`
- Test: `tests/camera_hidden_state_attribution/test_analyze_export.py`

**Interfaces:**
- Produces: `per_unit.csv`, `per_scene.csv`, `frozen_units.json`, `summary.json`.
- Export accepts only authenticated CSV/JSON and rejects NPZ, images, and incomplete
  holdout runs.

- [ ] Write failing tests for scene-equal aggregation, scene-bootstrap CI with 10,000
  samples/seed 33, frozen-unit immutability, and strict numeric export.
- [ ] Run focused tests and confirm failure.
- [ ] Implement aggregation and strict export.
- [ ] Run focused tests and commit.

### Task 5: AutoDL Entry Point and Focused Cleanup

**Files:**
- Create: `scripts/autodl/run_camera_hidden_state_attribution.sh`
- Create: `pre_experiments/camera_hidden_state_attribution/README.md`
- Modify: `AGENTS.md`
- Modify: `.gitignore`
- Delete: `pre_experiments/local_global_consistency/plot_trajectory_overlay.py`
- Delete: `doc/2026-07-29_Trajectory_Overlay_Visualization_Design.md`
- Test: `tests/camera_hidden_state_attribution/test_autodl_script.py`

**Interfaces:**
- Shell stages: `smoke`, `calibration`, `holdout`, `export`, and `all`.

- [ ] Write a failing shell-contract test requiring `set -euo pipefail`, existing
  checkpoint/data validation, explicit source/local run inputs, and ordered stages.
- [ ] Run the focused test and confirm failure.
- [ ] Implement the single AutoDL entry point and concise reproduction README.
- [ ] Remove only the invalid overlay code/document; retain scalar visualization and
  shared local-global input contracts.
- [ ] Update repository guidance and result ignore rules for numeric outputs.
- [ ] Run focused tests and commit.

### Task 6: Full Verification

**Files:**
- Modify only files required by verification findings.

- [ ] Run `python -m unittest discover -s tests/camera_hidden_state_attribution -v`.
- [ ] Run `python -m unittest discover -s tests/local_global_consistency -v`.
- [ ] Run `python -m compileall pre_experiments/camera_hidden_state_attribution vggt`.
- [ ] Run `git diff --check` and verify no raw NPZ/image artifact is tracked.
- [ ] Record exact implementation and refiner decision criteria in the worktree log.
