# Camera Hidden Causal Preference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and validate a prediction-only `4 x 1024` end-to-end causal
preference atlas for Camera Head hidden positions.

**Architecture:** Add opt-in batch-specific hidden and pose-delta perturbations
to the frozen Camera Head API. Estimate a nine-dimensional centered
finite-difference Jacobian at every refinement iteration, then project the
shared output-layer columns through it to recover all hidden-unit effects.
Calibration freezes cross-group scales; holdout only applies and validates
them.

**Tech Stack:** Python 3.10+, PyTorch, NumPy, standard-library `unittest`,
CSV/JSON/NPZ, Bash.

## Global Constraints

- Preserve default VGGT predictions bit-for-bit when perturbations are absent.
- Unit identity is `(iteration, hidden_index)`; iterations are never pooled.
- Calibration may fit normalization; holdout must never refit it.
- The atlas is prediction-only. GT is raw whenever used elsewhere, and any
  metric containing a prediction uses aligned prediction data.
- Tests require no CUDA, checkpoint, network, or ScanNet credentials.
- Do not commit raw NPZ scene artifacts, images, datasets, or checkpoints.

---

### Task 1: Camera Head Perturbation Interface

**Files:**
- Modify: `vggt/heads/camera_head.py`
- Modify: `tests/camera_hidden_state_attribution/test_camera_trace.py`

**Interfaces:**
- Consumes: normalized pose tokens `[B, S, C]`.
- Produces: optional `hidden_additive_perturbation [I, B, H]` and
  `pose_delta_additive_perturbation [I, B, 9]` arguments on `forward`,
  `decode_pose_tokens`, and `trunk_fn`.

- [x] **Step 1: Write failing identity, batch-isolation, and validation tests**

```python
zeros = torch.zeros(2, 1, 16)
actual = head(tokens, num_iterations=2,
              hidden_additive_perturbation=zeros)
torch.testing.assert_close(actual[-1], baseline[-1])

perturbation = torch.zeros(2, 2, 16)
perturbation[0, 0, 3] = 0.25
changed = head.decode_pose_tokens(
    normalized_tokens.expand(2, -1, -1),
    num_iterations=2,
    hidden_additive_perturbation=perturbation,
)
assert not torch.equal(changed[-1][0], baseline_batch[-1][0])
torch.testing.assert_close(changed[-1][1], baseline_batch[-1][1])
```

- [x] **Step 2: Run the focused test and verify missing-argument failure**

Run:
`python -m unittest tests.camera_hidden_state_attribution.test_camera_trace -v`

Expected: failures report that the perturbation keyword arguments are
unexpected.

- [x] **Step 3: Validate and apply both perturbation tensors**

Validate exact iteration, batch, and feature dimensions plus floating dtype and
finite values. Add hidden perturbations after `forward_features`; add pose-delta
perturbations after `forward_head`. Move each iteration slice to the computed
tensor's device and dtype before addition.

- [x] **Step 4: Run the focused test and existing attribution tests**

Run:
`python -m unittest tests.camera_hidden_state_attribution.test_camera_trace -v`

Run:
`python -m unittest discover -s tests/camera_hidden_state_attribution -v`

Expected: all tests pass.

### Task 2: Pure Causal Projection and Aggregation

**Files:**
- Create: `pre_experiments/camera_hidden_state_attribution/causal_preference.py`
- Create: `tests/camera_hidden_state_attribution/test_causal_preference.py`

**Interfaces:**
- Produces:
  `activation_rms(hidden, floor_ratio, absolute_floor) -> [I, H]`,
  `central_output_jacobians(...) -> dict[str, ndarray]`,
  `project_hidden_effects(...) -> dict[str, ndarray]`,
  `fit_causal_normalization(...) -> dict[str, object]`, and
  `apply_causal_normalization(...) -> dict[str, ndarray]`.

- [x] **Step 1: Write failing hand-derived numerical tests**

Use a two-frame fixture with a known camera-center derivative, a known
z-rotation derivative, and known FoV derivatives. Assert literal RMS scales,
translation effects, rotation degrees, FoV effects, normalized values, and
preferred labels.

- [x] **Step 2: Run the new test and verify import failure**

Run:
`python -m unittest tests.camera_hidden_state_attribution.test_causal_preference -v`

Expected: import fails because `causal_preference.py` does not exist.

- [x] **Step 3: Implement shape-checked NumPy calculations**

Use centered differences `(plus - minus) / (2 * basis_step)`. Convert rotation
matrix derivatives to local angular speed with the skew part of
`R.T @ dR`. Project each standardized hidden direction
`fc2_weight[:, unit] * activation_rms[iteration, unit]` in bounded unit chunks.
Fit output-group scales with a calibration-only 0.9 quantile and a positive
floor.

- [x] **Step 4: Run the new and existing tests**

Run:
`python -m unittest tests.camera_hidden_state_attribution.test_causal_preference -v`

Run:
`python -m unittest discover -s tests/camera_hidden_state_attribution -v`

Expected: all tests pass.

### Task 3: Scene Artifact Contract and Model Replay

**Files:**
- Modify: `pre_experiments/camera_hidden_state_attribution/artifacts.py`
- Create: `pre_experiments/camera_hidden_state_attribution/run_causal_preference.py`
- Create: `tests/camera_hidden_state_attribution/test_causal_runner.py`

**Interfaces:**
- Produces:
  `measure_scene_causal_effects(camera_head, normalized_tokens, device, ...)`,
  `save_causal_scene_effects(path, effects)`, and
  `load_causal_scene_effects(path, scene)`.

- [x] **Step 1: Write failing replay and artifact round-trip tests**

Create a small CPU Camera Head (`dim_in=16`, hidden size 8), run two basis
dimensions over two iterations, and assert finite `[2, 8]` effect arrays. Save
and load them and assert exact key/shape preservation. Corrupt an NPZ member
set and assert rejection.

- [x] **Step 2: Run the focused test and verify missing-symbol failure**

Run:
`python -m unittest tests.camera_hidden_state_attribution.test_causal_runner -v`

Expected: imports fail for the new runner and artifact functions.

- [x] **Step 3: Implement batched basis replay and direct checks**

Duplicate one token sequence into positive/negative batches. Add one
pose-delta basis perturbation per sample, decode the final pose, convert w2c
outputs to camera centers and c2w rotations, and project the resulting
Jacobians. Select the highest total-effect units deterministically for optional
direct hidden perturbation checks.

- [x] **Step 4: Implement strict numeric scene artifacts**

Persist only named finite numeric arrays. Basis dimensions explicitly omitted
by smoke mode use zero Jacobians and are identified by a measured-basis mask;
formal aggregation rejects an incomplete mask. Write via the existing atomic
NPZ helper.

- [x] **Step 5: Run focused and full CPU tests**

Run:
`python -m unittest tests.camera_hidden_state_attribution.test_causal_runner -v`

Run:
`python -m unittest discover -s tests/camera_hidden_state_attribution -v`

Expected: all tests pass.

### Task 4: Calibration, Holdout, and Numeric Summary

**Files:**
- Create: `pre_experiments/camera_hidden_state_attribution/causal_analyze.py`
- Modify: `pre_experiments/camera_hidden_state_attribution/run_causal_preference.py`
- Create: `tests/camera_hidden_state_attribution/test_causal_analyze.py`

**Interfaces:**
- Produces:
  `write_causal_numeric_summary(...)`,
  `freeze_causal_normalization(...)`, and
  `validate_frozen_causal_normalization(...)`.

- [x] **Step 1: Write failing provenance and summary tests**

Use two literal scene effect dictionaries. Assert scene-equal means, one CSV
row per `(iteration, unit)`, frozen digest validation, no holdout refit, stable
preferred-group labels, Spearman values, and top-64 overlap.

- [x] **Step 2: Run the focused test and verify missing-symbol failure**

Run:
`python -m unittest tests.camera_hidden_state_attribution.test_causal_analyze -v`

Expected: import fails for the analysis module.

- [x] **Step 3: Implement calibration freeze and holdout comparison**

Freeze the split digest, ordered calibration scene list, quantile scales,
calibration mean atlas, method/schema fields, and canonical digest. Holdout
validates all provenance fields and applies the frozen scales. Write
`per_position.csv`, `direct_checks.csv`, `summary.json`, and the frozen JSON.

- [x] **Step 4: Complete CLI stages and resumability**

Support `smoke`, `calibration`, and `holdout`; require frozen normalization for
holdout. Include basis step, basis batch size, direct-check count, scene list,
split digest, checkpoint path, and source run in the run digest. A formal run
requires every partition scene and all nine basis dimensions.

- [x] **Step 5: Run focused and full CPU tests**

Run:
`python -m unittest tests.camera_hidden_state_attribution.test_causal_analyze -v`

Run:
`python -m unittest discover -s tests/camera_hidden_state_attribution -v`

Expected: all tests pass.

### Task 5: AutoDL Entry Point and Strict Export

**Files:**
- Create: `scripts/autodl/run_camera_hidden_causal_preference.sh`
- Create: `scripts/autodl/camera_hidden_state_attribution/export_causal_preference.py`
- Create: `tests/camera_hidden_state_attribution/test_causal_autodl.py`
- Modify: `pre_experiments/camera_hidden_state_attribution/README.md`
- Modify: `AGENTS.md`

**Interfaces:**
- Consumes environment variables `SOURCE_RUN_DIR`, `SPLIT_MANIFEST`,
  `CKPT_DIR`, `STAGE`, `BASIS_BATCH_SIZE`, `BASIS_STEP`, and
  `DIRECT_CHECKS_PER_ITERATION`.
- Publishes only the six authenticated numeric causal files.

- [x] **Step 1: Write failing exporter behavior tests**

Construct a complete temporary holdout run, export it, and assert all allowed
files are copied. Add `raw_basis.npz` at the run root and assert export
rejection. Set `protocol_complete=false` and assert rejection.

- [x] **Step 2: Run the focused test and verify missing-symbol failure**

Run:
`python -m unittest tests.camera_hidden_state_attribution.test_causal_autodl -v`

Expected: import or file-not-found failure for the new exporter/entry point.

- [x] **Step 3: Implement ordered smoke/calibration/holdout/export stages**

Use `set -euo pipefail`, fail before model loading when required paths are
missing, persist run-directory state files, pass calibration frozen
normalization to holdout, and export the completed holdout run.

- [x] **Step 4: Update reproduction documentation**

Document the exact environment variables, stage-specific resume commands,
output policy, low-rank Jacobian method, and the fact that GT is unused.

- [x] **Step 5: Run syntax and full test verification**

Run: `bash -n scripts/autodl/run_camera_hidden_causal_preference.sh`

Run:
`python -m unittest discover -s tests/camera_hidden_state_attribution -v`

Expected: Bash syntax succeeds and all tests pass.

### Task 6: Final Verification and Branch Delivery

**Files:**
- Review all modified files and the branch diff.

**Interfaces:**
- Produces one independently reproducible commit on
  `camera-hidden-state-attribution-preexperiment`.

- [x] **Step 1: Run complete relevant test suites**

Run:
`python -m unittest discover -s tests/camera_hidden_state_attribution -v`

Run: `python -m unittest discover -s tests/local_global_consistency -v`

- [x] **Step 2: Verify repository hygiene**

Run: `git status --short`

Run:
`git diff --check`

Confirm raw `results/camera_hidden_state_attribution/` remains untracked and is
not staged.

- [x] **Step 3: Review the exact staged diff**

Run:
`git diff --stat`

Run:
`git diff -- AGENTS.md vggt/heads/camera_head.py pre_experiments scripts tests doc`

- [ ] **Step 4: Commit and push**

```bash
git add AGENTS.md vggt/heads/camera_head.py \
  pre_experiments/camera_hidden_state_attribution \
  scripts/autodl tests/camera_hidden_state_attribution doc
git commit -m "Add camera hidden causal preference atlas"
git push origin camera-hidden-state-attribution-preexperiment
```
