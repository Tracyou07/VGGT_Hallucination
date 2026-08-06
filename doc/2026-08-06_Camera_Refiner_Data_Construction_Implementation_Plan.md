# Camera Refiner Data Construction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. The user has explicitly prohibited subagents.

**Goal:** Build a resumable, split-safe pipeline that compares 100/200/300-frame Camera hidden states, freezes a calibration-selected multiscale intervention, evaluates it without holdout refitting, and exports validated external training shards plus tracked manifests.

**Architecture:** Existing local-global runs provide frame-matched Camera tokens for each context length. The new package replays those tokens through the frozen Camera Head, selects the most interior hidden observation at each scale, mixes local hidden states with frozen `beta` weights, applies a bounded `alpha` residual only at the frozen hidden-unit mask, and evaluates every predicted trajectory after independent Sim(3) alignment to raw GT. Large arrays remain external; Git receives only code, schemas, configuration, and documentation.

**Tech Stack:** Python 3.10+, NumPy, PyTorch, `unittest`, JSON/CSV, POSIX shell.

## Global Constraints

- Preserve upstream VGGT APIs and default Camera Head behavior.
- Local scales are exactly 100, 200, and 300 frames with 50% overlap and a deterministic tail window.
- Public tensor boundaries document frame, scale, refinement-iteration, and hidden dimensions.
- Any prediction-versus-GT metric aligns the prediction; GT remains raw.
- Calibration may select candidates; holdout consumes one authenticated frozen policy without refitting.
- Unit tests require no CUDA, checkpoint, network, or ScanNet credentials.
- Never commit datasets, checkpoints, figures, PLY files, or generated NPZ shards.

---

### Task 1: Multiscale Protocol

**Files:**
- Create: `pre_experiments/camera_refiner_data_construction/__init__.py`
- Create: `pre_experiments/camera_refiner_data_construction/protocol.py`
- Create: `tests/camera_refiner_data_construction/__init__.py`
- Create: `tests/camera_refiner_data_construction/test_protocol.py`

**Interfaces:**
- Produces: `Candidate(alpha: float, beta: tuple[float, float, float])`.
- Produces: `default_pure_candidates()` and `default_mixture_candidates()`.
- Produces: `assemble_multiscale_hidden(global_frame_ids, scale_windows)` returning `[3, I, S, H]` hidden plus `[3, S]` window metadata.
- Produces: `mix_local_hidden(local_hidden, beta)` returning `[I, S, H]`.

- [ ] Write tests for canonical candidate names, beta simplex validation, deterministic most-interior tie-breaking, complete scale coverage, and finite mixture output.
- [ ] Run `python -m unittest tests.camera_refiner_data_construction.test_protocol -v` and verify failures because the package does not exist.
- [ ] Implement immutable candidates, exact scale constants, assembly through the existing `assemble_short_hidden`, and weighted hidden mixing.
- [ ] Re-run the focused test and verify it passes.

### Task 2: Strict External Scene Shards

**Files:**
- Create: `pre_experiments/camera_refiner_data_construction/artifacts.py`
- Create: `tests/camera_refiner_data_construction/test_artifacts.py`

**Interfaces:**
- Consumes: protocol arrays with scales `[3]`, frames `[S]`, hidden `[3, I, S, H]`, and candidates `[C]`.
- Produces: `save_scene_shard(path, payload)` and `load_scene_shard(path, scene)`.
- Produces: schema version `1` with exact NPZ member validation.

- [ ] Write round-trip tests covering frame identity, h500/h100/h200/h300 shapes, candidate pose outputs, aligned error arrays, raw GT, selected boundaries, and non-finite rejection.
- [ ] Run the artifact test and verify missing-module failure.
- [ ] Implement strict dtype, shape, uniqueness, baseline-first, beta-simplex, and finite-value validation using atomic NPZ writes.
- [ ] Re-run the focused test and verify it passes.

### Task 3: Scale-Run Provenance and Camera Head Replay

**Files:**
- Create: `pre_experiments/camera_refiner_data_construction/run_study.py`
- Create: `tests/camera_refiner_data_construction/test_run_study.py`

**Interfaces:**
- Consumes: one global 500-frame context run, three local-global run directories keyed by 100/200/300, one frozen hidden replacement manifest, and one split manifest.
- Produces: `validate_scale_runs(...)`, `run_scene_candidates(...)`, and CLI stages `smoke`, `calibration`, `holdout`.
- Produces: external `scene_shard.npz`, per-scene completion JSON, immutable run metadata, and a run-directory pointer.

- [ ] Write tests that reject wrong scale/stride/partition/split/source provenance and accept exact 100/50, 200/100, and 300/150 runs.
- [ ] Run the focused test and verify the expected missing implementation failure.
- [ ] Implement metadata authentication and deterministic local-window loading.
- [ ] Add a CPU CameraHead test that replays baseline and candidate hidden mixtures, applies only the frozen selected mask, and reports aligned translation/rotation plus FoV change.
- [ ] Run the focused test and verify it passes.

### Task 4: Calibration Freeze and Holdout Analysis

**Files:**
- Create: `pre_experiments/camera_refiner_data_construction/analyze.py`
- Create: `tests/camera_refiner_data_construction/test_analyze.py`

**Interfaces:**
- Consumes: completed scene shards and immutable run metadata.
- Produces: scene-equal candidate CSV/JSON summaries.
- Produces: `freeze_candidate_policy(...)` with candidate identity, split digest, calibration scenes, source run IDs, safety limits, and canonical digest.
- Produces: holdout analysis that authenticates and evaluates only the frozen candidate.

- [ ] Write tests for scene-equal aggregation, deterministic scene bootstrap, leave-one-scene-out robustness, rotation/FoV safety filtering, deterministic tie-breaking, and holdout no-refit enforcement.
- [ ] Run the focused test and verify failure before implementation.
- [ ] Implement primary ranking by aligned translation delta with explicit safety limits stored in the frozen manifest.
- [ ] Write numeric CSV/JSON outputs without raw tensors or GT-derived detection fields.
- [ ] Re-run the focused test and verify it passes.

### Task 5: Dataset Manifest and Validator

**Files:**
- Create: `pre_experiments/camera_refiner_data_construction/dataset.py`
- Create: `tests/camera_refiner_data_construction/test_dataset.py`

**Interfaces:**
- Consumes: external scene shards and an explicit scene-role mapping.
- Produces: a versioned JSON manifest containing relative shard paths, SHA-256 checksums, scene roles, source provenance, frame counts, and schema version.
- Produces: `validate_dataset_manifest(...)` rejecting overlap, missing scales, digest mismatch, non-finite shards, or use of existing holdout scenes as training data.

- [ ] Write tests for deterministic checksums, disjoint roles, holdout leakage rejection, shard tampering, and compact inspection counts.
- [ ] Run the focused test and verify failure before implementation.
- [ ] Implement manifest creation and full validation without copying tensor data into Git.
- [ ] Re-run the focused test and verify it passes.

### Task 6: AutoDL Entry Point and Documentation

**Files:**
- Create: `scripts/autodl/camera_refiner_data_construction/run_multiscale_study.sh`
- Create: `scripts/autodl/camera_refiner_data_construction/validate_dataset.py`
- Create: `tests/camera_refiner_data_construction/test_autodl.py`
- Modify: `README.md`
- Modify: `pre_experiments/README.md`

**Interfaces:**
- Produces: resumable `smoke`, `calibration`, and `holdout` commands using the existing Conda environment and checkpoint.
- Produces: CPU-only manifest validation command suitable for a fresh machine.

- [ ] Write static tests for `set -euo pipefail`, quoted paths, exact scale/stride pairs, required environment variables, no environment creation/download, and no holdout refit.
- [ ] Run the focused test and verify failure before creating the scripts.
- [ ] Implement the shell entry and validator wrapper.
- [ ] Document external directory layout, stage order, resume behavior, outputs, and metric policy.
- [ ] Run focused tests, `python -m compileall -q pre_experiments vggt`, and `python -m unittest discover -s tests/camera_refiner_data_construction -v`.

### Task 7: Regression Verification

**Files:**
- Verify only; no production files expected.

**Interfaces:**
- Confirms the new package does not change default VGGT behavior or inherited experiment contracts.

- [ ] Run `python -m unittest discover -s tests/camera_refiner_data_construction -v`.
- [ ] Run `python -m unittest discover -s tests/local_global_consistency -v`.
- [ ] Run `python -m unittest discover -s tests/camera_hidden_state_attribution -v` and record the inherited Windows 8.3-path visualization failure separately if it remains the only failure.
- [ ] Run `python -m compileall -q pre_experiments vggt`.
- [ ] Inspect `git status --short` and verify no generated result, NPZ, dataset, checkpoint, or figure is tracked.
