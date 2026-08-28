# Privileged Latent Lifting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the already verified short-window camera advantage into four compact, whole-500 native Camera-token residual targets per scene, then run one-scene smoke and ten-scene calibration on H20.

**Architecture:** Reuse authenticated long/short Camera-token source shards and the formal long/short privileged labels. Decode and align each short teacher under one frozen baseline gauge, build four deterministic quality-weighted teacher variants, and optimize 32-frequency DCT residual coefficients through the frozen Camera Head. Store only compact coefficients and strict provenance; do not train the posterior or conditional prior until this latent-lifting gate passes.

**Tech Stack:** Python 3.10+, NumPy 1.26, PyTorch 2.3, repository `unittest`, frozen VGGT Camera Head, Bash H20 runner.

**Spec:** `docs/superpowers/specs/2026-08-28-privileged-conditional-hierarchical-vrfm-design.md`

## Global Constraints

- Formal inference remains one 500-frame VGGT backbone forward plus exactly four lightweight residual/Camera Head decodes; Stage A itself is offline training-label construction.
- Reuse `/data/yjh/output/vggt/variational_camera_latent/vrfm_camera_20260827T044926Z` and `/data/yjh/output/vggt/long_short_camera_head/long_short_head_formal_20260828T072407Z`; verify hashes and never mutate them.
- Prediction-only long inputs and short predictions remain separate from GT, quality weights, fused teachers, and optimized residual labels.
- Any artifact derived from GT is privileged even when it does not contain a raw GT array.
- Use one fixed `[500,32]` orthonormal DCT basis; coefficients have shape `[4,32,2048]`; frequencies 0–3 are global and 4–31 are local.
- Raw short tokens are never endpoints in long-token coordinates.
- The Camera Head, VGGT checkpoint, frozen baseline-to-GT Sim(3), scene split, four variant masks, and optimizer configuration are immutable within a run.
- CPU tests require no CUDA, checkpoint, network, ScanNet credential, or large artifact.
- Formal GPU work runs only on H20 after identity/GPU/process/disk/worktree checks. Refuse to start below 100 GiB free on `/data` or if the run root exceeds 20 GiB.
- Do not use the exposed H20 Hugging Face token and do not pull large H20 artifacts to Windows.
- Preserve atomic writes, exact member schemas, SHA-256 manifests, idempotent resume, and fail-closed verification.

---

### Task 1: Fixed Hierarchical DCT Basis and Strict Target Schema

**Files:**
- Create: `pre_experiments/conditional_hierarchical_vrfm/__init__.py`
- Create: `pre_experiments/conditional_hierarchical_vrfm/basis.py`
- Create: `pre_experiments/conditional_hierarchical_vrfm/artifacts.py`
- Create: `tests/conditional_hierarchical_vrfm/__init__.py`
- Create: `tests/conditional_hierarchical_vrfm/test_basis.py`
- Create: `tests/conditional_hierarchical_vrfm/test_artifacts.py`

**Interfaces:**
- Produces: `temporal_dct_basis(frame_count: int = 500, rank: int = 32, *, device: torch.device | None = None, dtype: torch.dtype = torch.float32) -> Tensor`.
- Produces: `expand_residual(coefficients: Tensor, basis: Tensor) -> Tensor` for `[B,32,2048] -> [B,500,2048]`.
- Produces: `split_hierarchical_coefficients(coefficients: Tensor, global_rank: int = 4) -> tuple[Tensor,Tensor]`.
- Produces: `save_latent_targets(path: Path, arrays: Mapping[str,np.ndarray]) -> str` and `load_latent_targets(path: Path) -> dict[str,np.ndarray]`.

- [ ] **Step 1: Write failing basis tests**

```python
class TemporalBasisTests(unittest.TestCase):
    def test_basis_is_deterministic_and_orthonormal(self):
        first = temporal_dct_basis()
        second = temporal_dct_basis()
        self.assertTrue(torch.equal(first, second))
        torch.testing.assert_close(first.T @ first, torch.eye(32), atol=1e-5, rtol=1e-5)

    def test_expand_and_hierarchical_split_have_exact_shapes(self):
        coefficients = torch.zeros(4, 32, 2048)
        residual = expand_residual(coefficients, temporal_dct_basis())
        self.assertEqual(residual.shape, (4, 500, 2048))
        global_part, local_part = split_hierarchical_coefficients(coefficients)
        self.assertEqual(global_part.shape, (4, 4, 2048))
        self.assertEqual(local_part.shape, (4, 28, 2048))
```

- [ ] **Step 2: Run the focused test and verify failure**

Run: `python -m unittest tests.conditional_hierarchical_vrfm.test_basis -v`

Expected: import failure for `pre_experiments.conditional_hierarchical_vrfm.basis`.

- [ ] **Step 3: Implement the fixed orthonormal basis**

```python
def temporal_dct_basis(
    frame_count: int = 500,
    rank: int = 32,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    if frame_count < 2 or rank < 1 or rank > frame_count:
        raise ValueError("rank must lie in [1, frame_count]")
    n = torch.arange(frame_count, device=device, dtype=torch.float64)[:, None]
    k = torch.arange(rank, device=device, dtype=torch.float64)[None, :]
    basis = torch.cos(torch.pi * (n + 0.5) * k / frame_count)
    basis[:, 0] *= frame_count ** -0.5
    if rank > 1:
        basis[:, 1:] *= (2.0 / frame_count) ** 0.5
    return basis.to(dtype=dtype)

def expand_residual(coefficients: Tensor, basis: Tensor) -> Tensor:
    if coefficients.ndim != 3 or coefficients.shape[1:] != (32, 2048):
        raise ValueError("coefficients must have shape [batch,32,2048]")
    if basis.shape != (500, 32):
        raise ValueError("basis must have shape [500,32]")
    return torch.einsum("fr,brc->bfc", basis, coefficients)
```

- [ ] **Step 4: Write failing artifact firewall tests**

```python
def test_latent_target_round_trip_uses_exact_schema(self):
    digest = save_latent_targets(self.path, self.valid_arrays())
    self.assertEqual(len(digest), 64)
    loaded = load_latent_targets(self.path)
    self.assertEqual(loaded["residual_coefficients"].shape, (4, 32, 2048))

def test_latent_target_rejects_missing_binding_and_nonfinite_coefficients(self):
    arrays = self.valid_arrays()
    del arrays["teacher_sha256"]
    with self.assertRaisesRegex(ValueError, "exact schema"):
        save_latent_targets(self.path, arrays)
    arrays = self.valid_arrays()
    arrays["residual_coefficients"][0, 0, 0] = np.nan
    with self.assertRaisesRegex(ValueError, "finite"):
        save_latent_targets(self.path, arrays)
```

- [ ] **Step 5: Implement exact atomic artifacts**

The latent-target schema is exactly:

```python
LATENT_TARGET_MEMBERS = {
    "scene", "frame_ids", "teacher_variant_ids", "teacher_window_masks",
    "coverage_masks", "residual_coefficients", "decoded_c2w_raw",
    "optimization_steps", "initial_losses", "final_losses", "basis_sha256",
    "source_sha256", "teacher_sha256", "checkpoint_sha256", "git_commit",
}
```

Enforce shapes `()`, `[500]`, `[4]`, `[4,9]`, `[4,500]`, `[4,32,2048]`,
`[4,500,4,4]`, `[4]`, `[4]`, `[4]`, and scalar Unicode digests respectively.
Reject object dtypes, extra members, nonfinite numeric arrays, non-homogeneous poses,
duplicate variant IDs, malformed SHA-256 strings, and non-binary masks. Save through a
`.tmp` sibling followed by `Path.replace`.

- [ ] **Step 6: Run focused tests and commit**

Run:

```powershell
python -m unittest tests.conditional_hierarchical_vrfm.test_basis -v
python -m unittest tests.conditional_hierarchical_vrfm.test_artifacts -v
```

Expected: all tests pass.

Commit:

```bash
git add pre_experiments/conditional_hierarchical_vrfm tests/conditional_hierarchical_vrfm
git commit -m "Add hierarchical residual basis and target schema"
```

### Task 2: Reproducible Short-Teacher Variants and Upper-Bound Replay

**Files:**
- Create: `pre_experiments/conditional_hierarchical_vrfm/teacher.py`
- Create: `tests/conditional_hierarchical_vrfm/test_teacher.py`

**Interfaces:**
- Consumes: authenticated `variational_camera_latent.source.v1` shard, prepared GT poses, frozen Camera Head, and checkpoint digest.
- Produces: `TeacherVariantSet(scene: str, frame_ids: np.ndarray, aligned_short_c2w: np.ndarray, window_weights: np.ndarray, window_masks: np.ndarray, fused_c2w: np.ndarray, coverage_weights: np.ndarray, oracle: FrozenOracle)`.
- Produces: `build_teacher_variants(source_path: Path, prepared_scene: Path, camera_head: nn.Module, *, checkpoint_sha256: str, device: torch.device, variant_count: int = 4) -> TeacherVariantSet`.
- Produces: `summarize_teacher_upper_bound(teachers: Sequence[TeacherVariantSet]) -> dict[str,object]`.

- [ ] **Step 1: Write failing deterministic-mask and leakage tests**

```python
def test_variant_zero_uses_all_positive_windows_and_other_masks_are_stable(self):
    weights = np.array([0.2, 0.0, 0.1, 0.4, 0.3, 0.0, 0.2, 0.1, 0.5])
    first = build_variant_window_masks("scene0000_00", weights)
    second = build_variant_window_masks("scene0000_00", weights)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first[0], weights > 0)
    self.assertEqual(len({row.tobytes() for row in first}), 4)

def test_teacher_builder_never_mutates_the_authenticated_prediction_source(self):
    before = sha256_file(self.source_path)
    build_teacher_variants(
        self.source_path,
        self.prepared_scene,
        self.fake_camera_head,
        checkpoint_sha256="a" * 64,
        device=torch.device("cpu"),
    )
    self.assertEqual(sha256_file(self.source_path), before)
```

- [ ] **Step 2: Run the focused test and verify failure**

Run: `python -m unittest tests.conditional_hierarchical_vrfm.test_teacher -v`

Expected: import failure for `teacher.py`.

- [ ] **Step 3: Implement short decoding, one frozen gauge, and four masks**

Use `load_source_shard`, `decode_camera_tokens`, `pose_encoding_to_c2w`,
`load_prepared_gt`, `fit_frozen_oracle`, `align_local_to_global`, and
`apply_frozen_oracle`. Fit the oracle once from the unmodified 500-frame baseline. For
window start positions `0,50,...,400`, align each decoded 100-frame short prediction to
the matching baseline segment, apply the frozen oracle, calculate baseline and teacher
RMS under that same oracle, and set

```python
weight = np.clip((baseline_rms - teacher_rms) / max(baseline_rms, 1e-12), 0.0, 1.0)
```

Create variant 0 from every positive window. Derive variants 1–3 from
`sha256(f"{scene}:teacher_variant:{index}")`; draw a stable 0.75-inclusion mask over only
positive windows, reject empty/duplicate masks, and deterministically flip the highest
remaining positive-weight window when a retry is required. Fuse translations by literal
positive weights and rotations by maximum contributing weight, matching the existing
verified teacher semantics.

- [ ] **Step 4: Add an exact replay fixture for the formal labels**

The unit fixture uses synthetic poses with known window utilities. The H20 pipeline later
adds a data-bound replay requirement: formal ten-scene variant-0 summary must reproduce
mean coverage `0.89`, mean utility `0.1293578271441714`, and positive scene count `10`
within `1e-10`. A mismatch means source/label semantics changed and stops the run.

- [ ] **Step 5: Run tests and commit**

Run: `python -m unittest tests.conditional_hierarchical_vrfm.test_teacher -v`

Expected: all tests pass.

Commit:

```bash
git add pre_experiments/conditional_hierarchical_vrfm/teacher.py tests/conditional_hierarchical_vrfm/test_teacher.py
git commit -m "Build reproducible privileged short teachers"
```

### Task 3: Differentiable Whole-500 Latent Lifting

**Files:**
- Create: `pre_experiments/conditional_hierarchical_vrfm/lift.py`
- Create: `tests/conditional_hierarchical_vrfm/test_lift.py`

**Interfaces:**
- Produces: frozen dataclass `LiftConfig(rank=32, global_rank=4, max_steps=250, learning_rate=5e-3, teacher_translation=1.0, relative_translation=0.5, rotation=0.1, uncovered_anchor=0.2, smoothness=0.05, residual_norm=1e-4, gradient_clip=1.0)`.
- Produces: `LiftResult(coefficients: Tensor, decoded_c2w_raw: Tensor, initial_loss: float, final_loss: float, completed_steps: int, finite: bool)`.
- Produces: `latent_lift_loss(...) -> dict[str,Tensor]` and `optimize_latent_target(...) -> LiftResult`.

- [ ] **Step 1: Write failing loss and optimization tests**

```python
def test_zero_coefficients_reproduce_the_baseline_exactly(self):
    decoded = decode_coefficients(self.fake_head, self.long_tokens, torch.zeros(1, 32, 2048))
    torch.testing.assert_close(decoded, self.baseline_c2w, atol=0.0, rtol=0.0)

def test_uncovered_frames_are_anchored_and_nan_teacher_frames_are_never_read(self):
    losses = latent_lift_loss(
        corrected_c2w_raw=self.corrected,
        baseline_c2w_raw=self.baseline,
        teacher_c2w_gt_gauge=self.teacher_with_nan_gaps,
        coverage_weight=self.coverage,
        oracle=self.oracle,
        residual=self.residual,
        config=LiftConfig(max_steps=2),
    )
    self.assertTrue(all(torch.isfinite(value) for value in losses.values()))

def test_tiny_optimizer_reduces_loss_and_keeps_head_frozen(self):
    before = {name: value.detach().clone() for name, value in self.fake_head.named_parameters()}
    result = optimize_latent_target(self.fake_head, self.long_tokens, self.teacher, self.oracle, LiftConfig(max_steps=20))
    self.assertLess(result.final_loss, result.initial_loss)
    for name, value in self.fake_head.named_parameters():
        torch.testing.assert_close(value, before[name])
```

- [ ] **Step 2: Run the focused test and verify failure**

Run: `python -m unittest tests.conditional_hierarchical_vrfm.test_lift -v`

Expected: import failure for `lift.py`.

- [ ] **Step 3: Implement zero-initialized coefficient optimization**

For one variant, create `nn.Parameter(torch.zeros(1,32,2048,float32))`, expand with the
fixed basis, add to normalized long Camera tokens, and call the frozen
`CameraHead.decode_pose_tokens(..., num_iterations=4)`. Convert to C2W differentiably and
apply the frozen Sim(3) with `apply_sim3_torch`.

Compute the exact total:

```python
total = (
    config.teacher_translation * teacher_center_loss
    + config.relative_translation * relative_motion_loss
    + config.rotation * covered_rotation_loss
    + config.uncovered_anchor * uncovered_center_anchor
    + config.smoothness * second_difference_loss
    + config.residual_norm * residual.square().mean()
)
```

Mask before indexing teacher arrays so all-NaN uncovered entries never enter arithmetic.
Use AdamW with zero weight decay, BF16 autocast only around Camera Head decode, float32
loss accumulation, gradient clipping at one, and best-finite-state retention. Abort on a
nonfinite loss/gradient, changed Camera Head parameter, non-homogeneous decoded pose, or
failure to reduce loss. Optimize four variants sequentially to bound memory.

- [ ] **Step 4: Add deterministic resume payloads**

`save_lift_checkpoint(path, *, coefficients, optimizer, variant_index, next_step,
config_digest, source_sha256, teacher_sha256, rng_state, best_coefficients, best_loss,
cuda_rng_state=None, device_type=None, loss_trace=(), initial_loss=0.0)` writes atomically.
The payload preserves the current AdamW state and the independently retained best-finite
state; the latter cannot be reconstructed after a nonmonotonic run.  It also records the
complete loss history, initial loss, CPU RNG state, and (for CUDA) the RNG state of the
coefficient device.  Resume rejects any changed digest/config/variant/device, noncanonical
installed-AdamW state, malformed RNG state, or inconsistent history, then restores the
exact next step transactionally. Add a test comparing a 20-step uninterrupted run to 8
steps plus resume to 20 with identical coefficients and loss trace, plus corruption tests
for optimizer and RNG state.

- [ ] **Step 5: Run tests and commit**

Run: `python -m unittest tests.conditional_hierarchical_vrfm.test_lift -v`

Expected: all tests pass.

Commit:

```bash
git add pre_experiments/conditional_hierarchical_vrfm/lift.py tests/conditional_hierarchical_vrfm/test_lift.py
git commit -m "Lift short teachers into native camera latents"
```

### Task 4: Evaluation, Gates, Resumable Pipeline, and Verification

**Files:**
- Create: `pre_experiments/conditional_hierarchical_vrfm/evaluate.py`
- Create: `pre_experiments/conditional_hierarchical_vrfm/report.py`
- Create: `pre_experiments/conditional_hierarchical_vrfm/pipeline.py`
- Create: `tests/conditional_hierarchical_vrfm/test_evaluate.py`
- Create: `tests/conditional_hierarchical_vrfm/test_pipeline.py`
- Create: `tests/conditional_hierarchical_vrfm/test_report.py`

**Interfaces:**
- Produces: `evaluate_latent_targets(...) -> dict[str,object]` with per-scene/per-variant covered utility, full-scene utility, rotation delta, uncovered drift, residual norm, and teacher-retention ratio.
- Produces: `classify_stage_a(scene_metrics: Sequence[Mapping[str,object]]) -> dict[str,object]`.
- Produces CLI stages `prepare`, `smoke`, `calibration`, `report`, and `verify`.

- [ ] **Step 1: Write failing metric and gate tests**

```python
def test_metrics_use_one_baseline_frozen_oracle_for_every_variant(self):
    metrics = evaluate_latent_targets(self.baseline, self.targets, self.labels)
    self.assertEqual(metrics["alignment_fit_count"], 0)
    self.assertEqual(metrics["variant_count"], 4)

def test_stage_a_requires_every_frozen_gate(self):
    result = classify_stage_a(self.passing_scenes())
    self.assertEqual(result["classification"], "LATENT_TARGETS_READY")
    harmed = self.passing_scenes()
    harmed[0]["mean_full_scene_utility"] = -0.0101
    result = classify_stage_a(harmed)
    self.assertEqual(result["classification"], "LATENT_LIFT_FAILED")
    self.assertIn("per_scene_harm", result["failed_gates"])
```

- [ ] **Step 2: Implement fixed evaluation and Stage A gates**

For every variant compare baseline and corrected poses under the oracle already stored in
the teacher sidecar. Never fit an alignment to corrected output. Aggregate variants within
scene, then scenes with equal scene weight. Implement these exact gates:

```python
gates = {
    "finite": all(scene["all_finite"] for scene in scenes),
    "teacher_retention": mean(scene["teacher_retention"] for scene in scenes) >= 0.70,
    "positive_mean": mean(scene["mean_full_scene_utility"] for scene in scenes) > 0.0,
    "positive_scene_count": sum(scene["mean_full_scene_utility"] > 0.0 for scene in scenes) >= 8,
    "per_scene_harm": min(scene["mean_full_scene_utility"] for scene in scenes) >= -0.01,
    "rotation_guard": mean(scene["mean_rotation_delta_deg"] for scene in scenes) <= 0.1,
    "uncovered_anchor": max(scene["uncovered_drift_fraction"] for scene in scenes) <= 0.005,
    "leakage_audit": all(scene["prediction_contract_clean"] for scene in scenes),
}
```

`LATENT_TARGETS_READY` requires every gate; otherwise return `LATENT_LIFT_FAILED` and all
failed gate names.

- [ ] **Step 3: Write pipeline barrier tests**

```python
def test_smoke_must_complete_before_calibration(self):
    with self.assertRaisesRegex(ValueError, "smoke completion"):
        run_calibration(self.args)

def test_verify_rehashes_every_artifact_and_rejects_extra_files(self):
    self.build_complete_fixture()
    verify_completed_run(self.run_root)
    (self.run_root / "privileged_labels/latent_targets/extra.npz").write_bytes(b"x")
    with self.assertRaisesRegex(ValueError, "exact directory"):
        verify_completed_run(self.run_root)

def test_prediction_only_manifest_contains_no_short_or_privileged_path(self):
    manifest = build_long_context_manifest(self.source_manifest)
    serialized = json.dumps(manifest).lower()
    for forbidden in ("short", "teacher", "privileged", "gt", "prepared"):
        self.assertNotIn(forbidden, serialized)
```

- [ ] **Step 4: Implement idempotent stages and signed completions**

`prepare` validates the clean Git commit, authenticated source manifests, formal labels,
checkpoint, scene roles, and replayed teacher upper bound. It writes immutable
`config.json` and manifests without copying source tensors.

`smoke` processes only `scene0000_00`, four variants, 20 optimization steps, and requires
finite decreasing losses plus exact checkpoint resume. `calibration` requires the signed
smoke completion and processes all ten manifest scenes at 250 steps per variant. Existing
valid target shards resume by exact digest; mismatched shards fail rather than overwrite.

`report` writes JSON and Markdown. `verify` reloads every NPZ with `allow_pickle=False`,
checks exact member sets/shapes/finiteness, rehashes all inputs/outputs, confirms the run is
below 20 GiB, validates test evidence, and publishes `verified_completion.json` only when
all ten scenes and four variants are present.

- [ ] **Step 5: Run focused and full CPU tests**

Run:

```powershell
python -m unittest discover -s tests/conditional_hierarchical_vrfm -v
python -m unittest discover -s tests/variational_camera_latent -v
python -m unittest discover -s tests/long_short_camera_head -v
python -m compileall -q pre_experiments
```

Expected: all tests pass; the pre-existing Windows symlink test may remain skipped for
lack of symlink privilege.

- [ ] **Step 6: Commit**

```bash
git add pre_experiments/conditional_hierarchical_vrfm tests/conditional_hierarchical_vrfm
git commit -m "Add fail-closed latent lifting pipeline"
```

### Task 5: H20 Calibration-First Runner

**Files:**
- Create: `scripts/h20/run_privileged_conditional_hvrfm_teacher_lift.sh`
- Create: `tests/conditional_hierarchical_vrfm/test_h20_runner.py`

**Interfaces:**
- Runs the complete Stage A protocol on H20 with one invocation.
- Produces `/data/yjh/output/vggt/privileged_conditional_hvrfm/<run_id>/verified_completion.json` or exits nonzero with all partial/checkpoint artifacts preserved.

- [ ] **Step 1: Write failing behavioral runner-contract tests**

```python
def test_preflight_only_succeeds_with_controlled_h20_facts(self):
    result = run_runner_fixture(
        self.runner,
        hostname="VM-0-11-ubuntu",
        user="ubuntu",
        free_gib=150,
        gpu_name="NVIDIA H20",
        gpu_compute_pids="",
        branch="codex/privileged-conditional-hvrfm",
        dirty_status="",
        arguments=("--preflight-only",),
    )
    self.assertEqual(result.returncode, 0, result.stderr)
    payload = json.loads(result.stdout)
    self.assertEqual(payload["result_root"], "/data/yjh/output/vggt/privileged_conditional_hvrfm")
    self.assertEqual(payload["planned_stages"], ["prepare", "smoke", "calibration", "report", "verify"])

def test_preflight_only_fails_before_compute_for_busy_gpu(self):
    result = run_runner_fixture(
        self.runner,
        hostname="VM-0-11-ubuntu",
        user="ubuntu",
        free_gib=150,
        gpu_name="NVIDIA H20",
        gpu_compute_pids="8123",
        branch="codex/privileged-conditional-hvrfm",
        dirty_status="",
        arguments=("--preflight-only",),
    )
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("active compute process", result.stderr)
```

- [ ] **Step 2: Implement the fail-closed runner**

Use `set -Eeuo pipefail` and accept only the optional literal argument
`--preflight-only`. Default to the dedicated remote worktree
`/home/ubuntu/yjh/vggt/.worktrees/privileged_conditional_hvrfm`, branch
`codex/privileged-conditional-hvrfm`, Python
`/home/ubuntu/anaconda3/envs/vggt-gx/bin/python`, output root
`/data/yjh/output/vggt/privileged_conditional_hvrfm`, and a dynamically selected idle H20
GPU. Verify host/user, clean exact branch, local checkpoint, both source completions,
verified ScanNet marker, at least 100 GiB free, no compute PID on the selected GPU, and
less than 20 GiB in the run root after every stage.

Run `prepare`, `smoke`, `calibration`, `report`, and `verify` serially. Capture stdout and
stderr separately; any nonempty stderr fails closed. Use `flock` for one run ID. Preserve
checkpoints and partial artifacts on failure. `--preflight-only` performs every read-only
gate, prints one JSON object containing the resolved result root and planned stage list,
and exits before creating a run directory or invoking Python. The test fixture provides
temporary executable stand-ins for `hostname`, `id`, `df`, `nvidia-smi`, `git`, and `du`,
so it exercises the real shell control flow rather than inspecting source text.

- [ ] **Step 3: Run syntax and contract tests**

Run:

```powershell
python -m unittest tests.conditional_hierarchical_vrfm.test_h20_runner -v
bash -n scripts/h20/run_privileged_conditional_hvrfm_teacher_lift.sh
```

Expected: all tests and Bash syntax pass.

- [ ] **Step 4: Commit**

```bash
git add scripts/h20/run_privileged_conditional_hvrfm_teacher_lift.sh tests/conditional_hierarchical_vrfm/test_h20_runner.py
git commit -m "Add H20 latent lifting runner"
```

### Task 6: H20 Execution and Evidence Publication

**Files:**
- Create on H20 only: `/data/yjh/output/vggt/privileged_conditional_hvrfm/<run_id>/...`
- Modify after the run: `docs/reports/2026-08-28-privileged-latent-lifting-report.md`

**Interfaces:**
- Consumes the clean pushed `codex/privileged-conditional-hvrfm` commit and authenticated existing data.
- Produces the first verified latent-target batch or a precise fail-closed scientific result.

- [ ] **Step 1: Recheck H20 immediately before compute**

Run read-only identity, `nvidia-smi`, `/data` free-space, active-job, source-completion,
checkpoint, worktree, branch, commit, and dirty-status checks. Select an actually idle H20;
do not assume GPU 3 remains idle.

- [ ] **Step 2: Launch the one-call calibration-first runner**

Run from the clean dedicated worktree:

```bash
bash scripts/h20/run_privileged_conditional_hvrfm_teacher_lift.sh
```

The runner performs one-scene smoke and automatically expands to ten scenes only after the
signed smoke gate passes.

- [ ] **Step 3: Verify output without pulling large artifacts**

On H20, rerun the pipeline `verify` stage, inspect stderr logs, recompute SHA-256 for the
report/completion, check exact scene/variant counts, and confirm output size is below
20 GiB. Copy back only small scalar JSON/Markdown if a later user request explicitly asks
for it.

- [ ] **Step 4: Publish the repository report**

Record the run ID, commit, source/checkpoint digests, teacher replay, per-scene metrics,
all Stage A gates, output path, and either `LATENT_TARGETS_READY` or
`LATENT_LIFT_FAILED`. If ready, the next plan is posterior upper-bound training; if failed,
stop the VRFM path and report the failing latent-lift assumption.

- [ ] **Step 5: Commit the evidence report**

```bash
git add docs/reports/2026-08-28-privileged-latent-lifting-report.md
git commit -m "Document privileged latent lifting result"
```
