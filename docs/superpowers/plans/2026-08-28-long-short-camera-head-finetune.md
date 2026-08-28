# Long–Short Camera Head Fine-Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fine-tune VGGT's native Camera Head with training-only short-window/GT supervision and verify long-only inference on H20.

**Architecture:** Reuse authenticated VRFM source shards to publish strict long-only inference shards and separate privileged teacher/GT sidecars. Train matched `gt_only` and `long_short` Camera Heads from the same VGGT checkpoint, then evaluate both with the baseline-frozen scene alignment and fail closed unless the long–short model improves locked replay without material harm.

**Tech Stack:** Python 3.11, PyTorch, NumPy, unittest/pytest, Bash, local VGGT-1B safetensors, NVIDIA H20.

**Spec:** `docs/superpowers/specs/2026-08-28-long-short-camera-head-finetune-design.md`

## Global Constraints

- Formal output root is `/data/yjh/output/vggt/long_short_camera_head/<run_id>`.
- Student inference consumes only `[500,2048]` long-window Camera tokens and a Camera Head checkpoint.
- Short-window tokens, GT poses, teacher weights, and oracle transforms are training/evaluation only.
- The scene split is exactly eight manifest `train` scenes and two manifest `validation` scenes.
- Alignment is fitted once from the original 500-frame baseline and never refitted to student predictions.
- Initialize from `/data/yjh/share/pretrained/VGGT-1B`; do not download weights or use credentials.
- Formal GPU computation runs only on H20 after identity, disk, process, GPU, source, and clean-worktree gates.
- Do not copy checkpoints, tensor shards, or other large H20 artifacts back to Windows.

---

### Task 1: Strict Data Contracts and Path Rebinding

**Files:**
- Create: `pre_experiments/long_short_camera_head/__init__.py`
- Create: `pre_experiments/long_short_camera_head/data.py`
- Test: `tests/long_short_camera_head/__init__.py`
- Test: `tests/long_short_camera_head/test_data.py`

**Interfaces:**
- Produces: `load_source_records(source_run: Path) -> tuple[SceneRecord, ...]`
- Produces: `publish_long_context(record: SceneRecord, destination: Path) -> LongContextRecord`
- Produces: `load_long_context(path: Path) -> dict[str, np.ndarray]`
- Produces: `load_prepared_gt(prepared_scene: Path, frame_ids: np.ndarray) -> np.ndarray`

- [ ] **Step 1: Write failing contract tests**

```python
def test_rebases_stale_manifest_path_and_checks_digest(self):
    records = load_source_records(self.source_run)
    self.assertEqual(records[0].path, self.source_run / "prediction_only/source/scene0000_00.npz")

def test_long_context_excludes_short_and_gt_members(self):
    publish_long_context(self.record, self.long_path)
    arrays = load_long_context(self.long_path)
    self.assertEqual(set(arrays), {"scene", "frame_ids", "camera_tokens", "baseline_c2w", "source_sha256"})
```

- [ ] **Step 2: Run tests and verify the module is absent**

Run: `python -m pytest tests/long_short_camera_head/test_data.py -v`

Expected: FAIL with `ModuleNotFoundError: pre_experiments.long_short_camera_head`.

- [ ] **Step 3: Implement immutable records, manifest rebinding, strict NPZ validation, and atomic writes**

```python
@dataclass(frozen=True)
class SceneRecord:
    scene: str
    role: str
    path: Path
    sha256: str

LONG_MEMBERS = {"scene", "frame_ids", "camera_tokens", "baseline_c2w", "source_sha256"}

def resolve_record_path(source_run: Path, row: dict[str, object]) -> Path:
    path = source_run / "prediction_only" / "source" / f"{row['scene']}.npz"
    if sha256_file(path) != row["sha256"]:
        raise ValueError("source shard digest mismatch")
    return path
```

- [ ] **Step 4: Run the focused tests**

Run: `python -m pytest tests/long_short_camera_head/test_data.py -v`

Expected: PASS.

- [ ] **Step 5: Commit the contract layer**

```bash
git add pre_experiments/long_short_camera_head tests/long_short_camera_head
git commit -m "Add long-only Camera Head data contracts"
```

### Task 2: Privileged Teacher Labels and Differentiable Geometry

**Files:**
- Create: `pre_experiments/long_short_camera_head/geometry.py`
- Create: `pre_experiments/long_short_camera_head/labels.py`
- Test: `tests/long_short_camera_head/test_geometry.py`
- Test: `tests/long_short_camera_head/test_labels.py`

**Interfaces:**
- Produces: `apply_sim3_torch(c2w: Tensor, scale: Tensor, rotation: Tensor, translation: Tensor) -> Tensor`
- Produces: `rotation_matrix_loss(predicted: Tensor, target: Tensor) -> Tensor`
- Produces: `build_privileged_labels(source_path: Path, prepared_scene: Path, camera_head: nn.Module, destination: Path, device: torch.device) -> PrivilegedRecord`
- Produces: `load_privileged_labels(path: Path) -> dict[str, np.ndarray]`

- [ ] **Step 1: Write failing geometry and quality-weight tests**

```python
def test_apply_sim3_is_differentiable(self):
    poses = torch.eye(4).repeat(3, 1, 1).requires_grad_()
    aligned = apply_sim3_torch(poses, torch.tensor(2.0), torch.eye(3), torch.ones(3))
    aligned[:, :3, 3].sum().backward()
    self.assertIsNotNone(poses.grad)

def test_bad_short_teacher_gets_zero_weight(self):
    labels = build_fixture_labels(baseline_rms=1.0, teacher_rms=(0.5, 1.2))
    self.assertGreater(labels["window_teacher_weight"][0], 0.0)
    self.assertEqual(labels["window_teacher_weight"][1], 0.0)
```

- [ ] **Step 2: Run tests and confirm missing functions fail**

Run: `python -m pytest tests/long_short_camera_head/test_geometry.py tests/long_short_camera_head/test_labels.py -v`

Expected: FAIL on imports.

- [ ] **Step 3: Implement frozen-oracle application and teacher construction**

Decode the frozen global and nine short token sequences, verify the decoded global trajectory matches `global_pred_c2w`, fit the scene oracle once, align each 100-frame teacher to its matching global segment with `align_local_to_global`, compute positive utility, and fuse only positive teachers. Store exact arrays:

```python
PRIVILEGED_MEMBERS = {
    "scene", "frame_ids", "gt_c2w", "oracle_scale", "oracle_rotation",
    "oracle_translation", "oracle_digest", "gt_scene_scale", "baseline_pose_encoding",
    "teacher_c2w_gt_gauge", "teacher_weight", "window_teacher_weight",
    "window_baseline_rms", "window_teacher_rms", "source_sha256", "checkpoint_sha256",
}
```

- [ ] **Step 4: Run focused tests and existing geometry regressions**

Run: `python -m pytest tests/long_short_camera_head/test_geometry.py tests/long_short_camera_head/test_labels.py tests/camera_velocity_ambiguity_02/test_frozen_oracle.py tests/camera_velocity_ambiguity_02/test_pipeline_science.py -v`

Expected: PASS.

- [ ] **Step 5: Commit privileged label construction**

```bash
git add pre_experiments/long_short_camera_head tests/long_short_camera_head
git commit -m "Build quality-weighted short-window teacher labels"
```

### Task 3: Native Camera Head Loss and Matched Training

**Files:**
- Create: `pre_experiments/long_short_camera_head/losses.py`
- Create: `pre_experiments/long_short_camera_head/train.py`
- Test: `tests/long_short_camera_head/test_losses.py`
- Test: `tests/long_short_camera_head/test_train.py`

**Interfaces:**
- Produces: `LossWeights`
- Produces: `camera_head_losses(student_pose: Tensor, baseline_pose: Tensor, labels: TrainingLabels, *, teacher_coefficient: float, weights: LossWeights) -> dict[str, Tensor]`
- Produces: `TrainConfig`
- Produces: `train_camera_head(config: TrainConfig) -> TrainingResult`
- Produces: `load_camera_head_checkpoint(path: Path, checkpoint_dir: Path, device: torch.device) -> nn.Module`

- [ ] **Step 1: Write failing loss and checkpoint tests**

```python
def test_teacher_term_is_exactly_disabled_for_gt_only(self):
    losses = camera_head_losses(self.pred, self.base, self.labels, teacher_coefficient=0.0)
    self.assertEqual(float(losses["teacher"]), 0.0)

def test_only_declared_camera_head_parameters_train(self):
    names = configure_trainable_scope(self.head)
    self.assertTrue(any(name.startswith("trunk.3") for name in names))
    self.assertFalse(any(name.startswith("trunk.0") for name in names))
    self.assertFalse(self.head.token_norm.weight.requires_grad)
```

- [ ] **Step 2: Run tests and confirm imports fail**

Run: `python -m pytest tests/long_short_camera_head/test_losses.py tests/long_short_camera_head/test_train.py -v`

Expected: FAIL on imports.

- [ ] **Step 3: Implement robust normalized losses and the conservative trainable scope**

```python
total = (
    weights.gt_translation * gt_translation
    + weights.relative_translation * relative_translation
    + weights.rotation * rotation
    + weights.anchor * anchor
    + teacher_coefficient * weights.teacher * teacher
)
```

Use lags `(1, 5, 10, 25)`, Smooth L1 translation losses, float32 pose conversion outside BF16-sensitive linear algebra, AdamW, gradient clipping, deterministic scene cycling, atomic checkpoints, exact resume digests, and best-checkpoint selection on locked-replay mean RMS.

- [ ] **Step 4: Run CPU training tests**

Run: `python -m pytest tests/long_short_camera_head/test_losses.py tests/long_short_camera_head/test_train.py -v`

Expected: PASS, including a tiny fake-head overfit and exact checkpoint reload.

- [ ] **Step 5: Commit native Camera Head training**

```bash
git add pre_experiments/long_short_camera_head tests/long_short_camera_head
git commit -m "Train VGGT Camera Head with long-short consistency"
```

### Task 4: Long-Only Inference, Evaluation, and Reporting

**Files:**
- Create: `pre_experiments/long_short_camera_head/evaluate.py`
- Create: `pre_experiments/long_short_camera_head/report.py`
- Test: `tests/long_short_camera_head/test_evaluate.py`
- Test: `tests/long_short_camera_head/test_report.py`

**Interfaces:**
- Produces: `run_long_only_inference(long_context_path: Path, checkpoint_path: Path, checkpoint_dir: Path, destination: Path, device: torch.device) -> PredictionRecord`
- Produces: `evaluate_prediction(prediction_path: Path, privileged_path: Path, destination: Path) -> EvaluationRecord`
- Produces: `write_report(run_root: Path) -> Path`

- [ ] **Step 1: Write leakage, metric, and classification tests**

```python
def test_inference_signature_has_no_privileged_or_short_argument(self):
    names = set(inspect.signature(run_long_only_inference).parameters)
    self.assertFalse(names & {"gt", "prepared_root", "short_tokens", "privileged"})

def test_report_fails_promising_when_one_scene_worsens_over_one_percent(self):
    report = classify(make_metrics(utilities=(0.05, -0.02)))
    self.assertNotEqual(report["classification"], "PROMISING")
```

- [ ] **Step 2: Run tests and verify missing modules fail**

Run: `python -m pytest tests/long_short_camera_head/test_evaluate.py tests/long_short_camera_head/test_report.py -v`

Expected: FAIL on imports.

- [ ] **Step 3: Implement strict long-only prediction and frozen-oracle metrics**

Prediction NPZ members are exactly:

```python
{"scene", "frame_ids", "pose_encoding", "predicted_c2w", "source_sha256", "checkpoint_sha256"}
```

Evaluation writes translation RMS, overlap RMS, relative-translation errors, rotation error, correction magnitude, and utility. Reporting compares baseline, `gt_only`, and `long_short`, labels the two held scenes `locked_replay`, and applies the exact spec gates.

- [ ] **Step 4: Run focused tests**

Run: `python -m pytest tests/long_short_camera_head/test_evaluate.py tests/long_short_camera_head/test_report.py -v`

Expected: PASS.

- [ ] **Step 5: Commit evaluation and reporting**

```bash
git add pre_experiments/long_short_camera_head tests/long_short_camera_head
git commit -m "Add long-only Camera Head evaluation"
```

### Task 5: Fail-Closed Pipeline, H20 Runner, and Completion Verification

**Files:**
- Create: `pre_experiments/long_short_camera_head/pipeline.py`
- Create: `scripts/h20/run_long_short_camera_head.sh`
- Modify: `pre_experiments/README.md`
- Test: `tests/long_short_camera_head/test_pipeline.py`
- Test: `tests/long_short_camera_head/test_h20_runner.py`

**Interfaces:**
- Produces CLI stages `prepare`, `smoke`, `calibration`, `evaluate`, `report`, and `verify`
- Produces `verify_completed_run(run_root: Path) -> Path`

- [ ] **Step 1: Write failing pipeline and runner gates**

```python
def test_default_result_root_is_under_vggt(self):
    text = self.runner.read_text()
    self.assertIn("/data/yjh/output/vggt/long_short_camera_head", text)

def test_runner_requires_clean_branch_h20_and_100_gib(self):
    text = self.runner.read_text()
    for required in ("VM-0-11-ubuntu", "NVIDIA H20", "status --short", "-ge 100"):
        self.assertIn(required, text)
```

- [ ] **Step 2: Run tests and confirm the runner is absent**

Run: `python -m pytest tests/long_short_camera_head/test_pipeline.py tests/long_short_camera_head/test_h20_runner.py -v`

Expected: FAIL.

- [ ] **Step 3: Implement resumable stages and exact completion verification**

The runner executes preparation, one-scene smoke, matched calibration variants,
long-only inference, privileged evaluation, reporting, and verification. It records
stdout/stderr separately, uses a run lock, selects explicit idle GPUs, and never sources
tokens or downloads weights.

- [ ] **Step 4: Run the complete local regression suite**

Run: `python -m pytest tests/long_short_camera_head tests/variational_camera_latent tests/variational_camera_selector -q`

Expected: PASS with only documented platform skips.

- [ ] **Step 5: Commit the pipeline**

```bash
git add pre_experiments/long_short_camera_head scripts/h20/run_long_short_camera_head.sh pre_experiments/README.md tests/long_short_camera_head
git commit -m "Add fail-closed H20 Camera Head pipeline"
```

### Task 6: H20 Smoke, Calibration, Independent Verification, and Final Report

**Files:**
- Create: `docs/reports/2026-08-28-long-short-camera-head-report.md`

**Interfaces:**
- Consumes the committed runner and clean remote worktree.
- Produces one verified H20 run and a concise repository report containing only scalar results and artifact paths.

- [ ] **Step 1: Push the implementation branch and update a dedicated H20 worktree**

Run locally: `git push -u origin codex/long-short-camera-head-finetune`

Run remotely: `git -C /home/ubuntu/yjh/vggt worktree add /home/ubuntu/yjh/vggt/.worktrees/long_short_camera_head origin/codex/long-short-camera-head-finetune`

- [ ] **Step 2: Recheck H20 identity, disk, GPU ownership, active tasks, checkpoint, source run, and worktree cleanliness**

Expected: H20 identity matches; `/data` has at least 100 GiB; selected GPUs have no compute processes; checkpoint/source markers exist; worktree is clean.

- [ ] **Step 3: Run smoke and inspect its separate stderr and completion marker**

Run: `RUN_ID=long_short_head_<UTC> GPU_GT_ONLY=2 GPU_LONG_SHORT=3 bash scripts/h20/run_long_short_camera_head.sh`

Expected: the one-scene loss decreases, output is finite, and smoke verification passes before calibration begins.

- [ ] **Step 4: Allow matched calibration and evaluation to complete**

Expected: both variants finish or fail closed with preserved checkpoints/logs. No unrelated GPU process is stopped or modified.

- [ ] **Step 5: Independently rerun verification**

Run: `python -m pre_experiments.long_short_camera_head.pipeline --stage verify --run-root /data/yjh/output/vggt/long_short_camera_head/<run_id>`

Expected: `verified_completion.json` is reproduced without mismatch and every stderr log is empty.

- [ ] **Step 6: Write the scalar report and run verification-before-completion checks**

Document the exact commit/run root, split status, baseline/gt-only/long-short scene metrics, classification, checkpoint hashes, inference leakage audit, test commands, and limitations. Do not claim improvement unless the formal gates pass.

- [ ] **Step 7: Commit and push the report**

```bash
git add docs/reports/2026-08-28-long-short-camera-head-report.md
git commit -m "Report long-short Camera Head result"
git push
```
