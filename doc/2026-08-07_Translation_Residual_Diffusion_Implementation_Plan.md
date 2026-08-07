# Translation Residual Diffusion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. The user explicitly prohibited subagents.

**Goal:** Build resumable training and inference for a translation-only camera-center residual diffusion model using 100-frame local context and frozen translation-unit features.

**Architecture:** A strict dataset adapter joins external multiscale scene shards with one exact 100/50 local-global run. It constructs 100-frame condition tensors in a prediction-derived scene gauge. A compact temporal Transformer predicts clean residuals and confidence; DDIM sampling generates overlapping corrections that are fused while copying global rotations exactly.

**Tech Stack:** Python 3.10+, PyTorch, NumPy, `unittest`, JSON/CSV, POSIX shell.

**Status:** Implemented on 2026-08-07. The retained suite includes an end-to-end
CPU train/infer smoke test; AutoDL GPU execution remains the next experimental step.

## Global Constraints

- Never use GT as an inference feature.
- Keep global VGGT rotations numerically unchanged.
- Use only scale 100 with stride 50 in the first implementation.
- Read translation units from an authenticated frozen JSON manifest.
- Resume only when checkpoint and run configuration identities match.
- Unit tests require no CUDA, checkpoint, network, or ScanNet credentials.
- AutoDL scripts reuse the existing `vggt` Conda environment and never install dependencies or download weights.

---

### Task 1: Dataset Contract and Coordinate Gauge

**Files:**
- Create: `pre_experiments/camera_refiner_training/data.py`
- Create: `pre_experiments/camera_refiner_training/geometry.py`
- Test: `tests/camera_refiner_training/test_data.py`
- Test: `tests/camera_refiner_training/test_geometry.py`

**Interfaces:**
- `load_translation_units(path, count, iteration)` returns authenticated unit indices and digest.
- `SceneSource.load(...)` validates one scene shard and its 100/50 local windows.
- `build_scene_windows(...)` returns condition, target residual, global centers, frame IDs, starts, and raw poses.
- `canonicalize_scene(...)` and `restore_centers(...)` are inverse operations.
- `fuse_window_corrections(...)` combines overlaps with confidence and boundary taper.

- [ ] Write tests for manifest parsing, shard/window identity, prediction-only local alignment, GT exclusion from conditions, canonical round trip, and exact rotation preservation.
- [ ] Run focused tests and verify they fail because the package is absent.
- [ ] Implement the minimal data and geometry APIs.
- [ ] Re-run focused tests and verify they pass.

### Task 2: Denoiser and Diffusion Process

**Files:**
- Create: `pre_experiments/camera_refiner_training/model.py`
- Create: `pre_experiments/camera_refiner_training/diffusion.py`
- Test: `tests/camera_refiner_training/test_model.py`
- Test: `tests/camera_refiner_training/test_diffusion.py`

**Interfaces:**
- `ResidualDiT(config)` maps noisy residuals, conditions, and timesteps to clean residual and confidence through adaptive LayerNorm blocks with zero-initialized output.
- `DiffusionSchedule.cosine(steps)` stores cumulative noise coefficients.
- `q_sample(clean, timestep, noise)` applies forward noise.
- `ddim_sample(model, condition, schedule, sample_steps, generator)` returns clean residual and confidence.

- [ ] Write failing shape, determinism, finite-value, and timestep-validation tests.
- [ ] Implement a configurable 1D DiT and clean-residual diffusion schedule, following the timestep conditioning and zero-initialization patterns in RayDiffusion and DiffusionSfM.
- [ ] Verify focused tests pass.

### Task 3: Losses and Overlap Consistency

**Files:**
- Create: `pre_experiments/camera_refiner_training/losses.py`
- Test: `tests/camera_refiner_training/test_losses.py`

**Interfaces:**
- `training_losses(...)` returns total, denoising, center, relative-motion, overlap, and conservative-gate terms.
- Overlap matching uses scene identity and absolute window starts in the shared scene gauge.

- [ ] Write failing tests showing zero loss for exact predictions, nonzero adjacent-window disagreement, and lag validation.
- [ ] Implement weighted loss terms for lags 1, 5, 10, and 25.
- [ ] Verify focused tests pass.

### Task 4: Resumable Training CLI

**Files:**
- Create: `pre_experiments/camera_refiner_training/train.py`
- Create: `pre_experiments/camera_refiner_training/checkpoint.py`
- Test: `tests/camera_refiner_training/test_train.py`

**Interfaces:**
- CLI consumes dataset manifest, dataset root, exact local run, frozen units, output directory, and model kind.
- Checkpoints contain model, optimizer, epoch, model config, normalization statistics, data/unit digests, and run config digest.
- `--resume` rejects configuration drift and continues from `last.pt`.

- [ ] Write failing tests for one CPU optimization step, checkpoint round trip, deterministic split filtering, and resume mismatch rejection.
- [ ] Implement diffusion and deterministic training modes with scene-grouped batches.
- [ ] Verify focused tests pass.

### Task 5: Inference, Fusion, and Metrics

**Files:**
- Create: `pre_experiments/camera_refiner_training/infer.py`
- Create: `pre_experiments/camera_refiner_training/metrics.py`
- Test: `tests/camera_refiner_training/test_infer.py`

**Interfaces:**
- CLI loads one frozen checkpoint and explicit scene role.
- Per-scene NPZ stores frame IDs, global/corrected c2w, raw GT, confidence, and correction magnitude.
- CSV/JSON summaries report aligned translation error and confirm zero rotation change.

- [ ] Write failing tests for overlap fusion, checkpoint inference, resumable scene completion, aligned metrics, and exact rotation copy.
- [ ] Implement deterministic DDIM and deterministic-Transformer inference paths.
- [ ] Verify focused tests pass.

### Task 6: AutoDL Entry Points and Branch Cleanup

**Files:**
- Create: `scripts/autodl/camera_refiner_training/train.sh`
- Create: `scripts/autodl/camera_refiner_training/infer.sh`
- Test: `tests/camera_refiner_training/test_autodl.py`
- Modify: `README.md`
- Modify: `AGENTS.md`
- Remove: predecessor runners, analyzers, publishers, visualizers, and tests not imported by the training pipeline.

**Interfaces:**
- Shell scripts activate `vggt`, validate explicit external paths, use `/root/autodl-tmp/results/camera_refiner_training`, and support resumable execution.

- [ ] Write failing static shell-contract tests.
- [ ] Implement scripts and documented commands.
- [ ] Remove code proven unreachable from the training/import graph.
- [ ] Run shell syntax, compile, focused tests, and the complete retained suite.

### Task 7: Final Verification

- [ ] Run `python -m unittest discover -s tests/camera_refiner_training -v`.
- [ ] Run `python -m unittest discover -s tests -v`.
- [ ] Run `python -m compileall -q pre_experiments vggt`.
- [ ] Run `bash -n` on both AutoDL scripts.
- [ ] Run `git diff --check` and confirm no generated tensor, dataset, checkpoint, image, or result file is tracked.
