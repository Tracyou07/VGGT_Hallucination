# Variational Camera Candidate Selector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a prediction-only ranker that selects a VRFM latent correction direction and continuous step size from long-window Camera tokens, while using GT utility only through a physically separate training/evaluation sidecar.

**Architecture:** Repackage the sealed Phase 1 run into a small long-context-only shard plus references to the existing raw VRFM candidate shards. Train matched full-context and residual-only listwise rankers on the fixed eight train scenes, score candidates without privileged inputs, then attach the sealed sidecar only in a separate evaluation stage on the fixed two validation scenes.

**Tech Stack:** Python 3.10+, NumPy, PyTorch, repository `unittest`, Bash H20 runner.

**Spec:** `docs/superpowers/specs/2026-08-27-variational-camera-candidate-selector-design.md`

## Global Constraints

- Formal inputs are immutable under `/data/yjh/output/variational_camera_latent/vrfm_camera_20260827T044926Z` and bind producer commit `e7f178587b9d80ebf730e9f6e3c2266d49b8b64b`.
- Formal outputs are new files under `/data/yjh/output/variational_camera_selector/<run_id>/`; never mutate the Phase 1 run.
- Train scenes are the first eight source-manifest records; validation scenes are exactly `scene0325_01` and `scene0675_00`.
- Prediction/inference may read only long-context shards and existing VRFM candidate shards. It must not receive a source shard, short-window tensor, GT, depth, error, quality, or privileged path.
- Training labels come only from the sealed VRFM residual privileged sidecar joined by exact scene, sample ID, sample seed, alpha, role, and SHA-256 bindings.
- Collapse alpha zero to one no-op; every overlap has exactly 225 choices: one no-op plus 32 directions times seven nonzero alphas.
- The 20-Q random-control family is never a training input or augmentation.
- Train matched full-context and residual-only rankers with `tau=0.05`; fit no dataset statistic on validation scenes.
- One-scene smoke must pass before automatic fixed 8/2 calibration.
- Formal compute runs only on H20 after identity, GPU, disk, active-process, worktree, upstream-completion, and input-digest preflight.
- Never use the exposed H20 Hugging Face token and never pull large H20 outputs to Windows.
- Follow RED-GREEN TDD for every production behavior. All 64 existing `variational_camera_latent` tests remain green; the five documented unrelated Windows downloader baseline failures are not modified by this plan.

---

### Task 1: Long-Context-Only Schema and Input Binding

**Files:**
- Create: `pre_experiments/variational_camera_selector/__init__.py`
- Create: `pre_experiments/variational_camera_selector/contracts.py`
- Create: `pre_experiments/variational_camera_selector/schema.py`
- Create: `tests/variational_camera_selector/__init__.py`
- Create: `tests/variational_camera_selector/test_schema.py`

**Interfaces:**
- Consumes: existing `load_source_shard`, `load_candidate_shard`, and the sealed source/candidate manifests.
- Produces: `LongContextRecord`, `build_long_context_shard(source_path, candidate_path, destination, *, role)`, `load_long_context_shard(path)`, and `write_prediction_binding_manifest(...)`.

- [ ] **Step 1: Write the failing firewall and shape tests**

```python
class LongContextSchemaTests(unittest.TestCase):
    def test_build_copies_only_long_context_and_binds_candidate(self):
        record = build_long_context_shard(
            self.source_path, self.candidate_path, self.output, role="train"
        )
        arrays = load_long_context_shard(record.path)
        self.assertEqual(arrays["global_camera_tokens"].shape, (500, 2048))
        self.assertEqual(arrays["overlap_long_tokens"].shape, (8, 50, 2048))
        self.assertEqual(str(arrays["role"]), "train")
        self.assertEqual(str(arrays["candidate_shard_sha256"]), record.candidate_sha256)
        self.assertFalse(any("short" in name for name in arrays))

    def test_loader_rejects_short_gt_quality_and_digest_tampering(self):
        for forbidden in ("short_camera_tokens", "gt_c2w", "quality", "error"):
            with self.subTest(forbidden=forbidden), self.assertRaises(ValueError):
                load_long_context_shard(self.write_tampered_member(forbidden))
        with self.assertRaisesRegex(ValueError, "candidate.*SHA-256"):
            build_long_context_shard(
                self.source_path, self.other_candidate, self.output, role="train"
            )
```

- [ ] **Step 2: Run RED and confirm the package is absent**

Run: `python -m unittest tests.variational_camera_selector.test_schema -v`

Expected: import failure for `pre_experiments.variational_camera_selector`.

- [ ] **Step 3: Implement strict contracts and atomic NPZ/JSON writes**

```python
@dataclass(frozen=True)
class LongContextRecord:
    scene: str
    role: str
    path: Path
    sha256: str
    source_sha256: str
    candidate_sha256: str

LONG_CONTEXT_MEMBERS = {
    "global_frame_ids", "global_camera_tokens", "overlap_frame_ids",
    "overlap_long_tokens", "span_starts", "source_sample_ids", "scene",
    "role", "source_shard_sha256", "candidate_shard_sha256",
    "producer_git_commit",
}

def load_long_context_shard(path: Path) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]).copy() for name in archive.files}
    validate_long_context_shard(arrays)
    return arrays
```

Validate exact members, finite float32 `[500,2048]`/`[8,50,2048]`, aligned IDs,
eight stable sample IDs, role in `{train,validation}`, lowercase digests, and candidate
sample IDs/source-long tokens exactly matching the source shard. Write `.tmp` then replace.

- [ ] **Step 4: Run GREEN and the existing source/candidate tests**

Run:

```bash
python -m unittest tests.variational_camera_selector.test_schema -v
python -m unittest tests.variational_camera_latent.test_source_schema tests.variational_camera_latent.test_candidates -v
```

Expected: all pass.

- [ ] **Step 5: Commit the schema slice**

```bash
git add pre_experiments/variational_camera_selector tests/variational_camera_selector
git commit -m "Add long-only selector input schema"
```

---

### Task 2: Candidate Groups, No-Op Deduplication, and Privileged Join

**Files:**
- Create: `pre_experiments/variational_camera_selector/dataset.py`
- Create: `tests/variational_camera_selector/test_dataset.py`

**Interfaces:**
- Consumes: long-context binding records, candidate shards, residual-scan prediction records, and residual privileged sidecars.
- Produces: `PredictionCandidateDataset`, `SelectorTrainingDataset`, `CandidateGroup`, and `join_training_group(...)`.

- [ ] **Step 1: Write failing group and leakage tests**

```python
class SelectorDatasetTests(unittest.TestCase):
    def test_prediction_group_has_one_noop_and_224_nonzero_choices(self):
        group = self.prediction_dataset[0]
        self.assertEqual(group.delta_tokens.shape, (225, 50, 2048))
        self.assertEqual(group.alphas.shape, (225,))
        self.assertEqual(int(np.count_nonzero(group.alphas == 0.0)), 1)
        np.testing.assert_array_equal(group.delta_tokens[0], 0.0)

    def test_prediction_dataset_cannot_accept_privileged_path(self):
        signature = inspect.signature(PredictionCandidateDataset)
        self.assertFalse(any("privileged" in name for name in signature.parameters))

    def test_training_join_rejects_scene_seed_alpha_role_and_digest_mismatch(self):
        for mutation in ("scene", "sample_seed", "alpha", "role", "prediction_sha256"):
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                SelectorTrainingDataset(self.prediction_manifest, self.mutate_sidecar(mutation))
```

- [ ] **Step 2: Run RED and verify missing APIs**

Run: `python -m unittest tests.variational_camera_selector.test_dataset -v`

Expected: import/name failure for `PredictionCandidateDataset`.

- [ ] **Step 3: Implement exact 225-choice construction**

```python
@dataclass(frozen=True)
class CandidateGroup:
    scene: str
    role: str
    sample_id: str
    span_start: int
    global_tokens: np.ndarray
    x0: np.ndarray
    delta_tokens: np.ndarray
    alphas: np.ndarray
    z: np.ndarray
    sample_seeds: np.ndarray
    choice_ids: np.ndarray
    utilities: np.ndarray | None = None

def expand_candidate_grid(x0, corrected, z, sample_seeds, alphas):
    nonzero = np.asarray(alphas)[1:]
    residual = np.asarray(corrected) - np.asarray(x0)[None]
    scaled = (nonzero[:, None, None, None] * residual[None]).reshape(-1, 50, 2048)
    return np.concatenate((np.zeros((1, 50, 2048), np.float32), scaled), axis=0)
```

Use alpha-major then sample-major order and repeat z/seeds in that exact order; the no-op
uses zero z, seed `-1`, and stable choice ID `<sample_id>:noop`. `SelectorTrainingDataset`
loads utilities only after prediction artifacts validate and then collapses sidecar alpha zero
to one exact zero utility.

- [ ] **Step 4: Run GREEN plus firewall regressions**

Run:

```bash
python -m unittest tests.variational_camera_selector.test_dataset -v
python -m unittest tests.variational_camera_latent.test_privileged_report tests.variational_camera_latent.test_vrfm_residual_scan -v
```

Expected: all pass.

- [ ] **Step 5: Commit the dataset slice**

```bash
git add pre_experiments/variational_camera_selector/dataset.py tests/variational_camera_selector/test_dataset.py
git commit -m "Add sealed selector training joins"
```

---

### Task 3: Full-Context and Residual-Only Rankers

**Files:**
- Create: `pre_experiments/variational_camera_selector/model.py`
- Create: `pre_experiments/variational_camera_selector/loss.py`
- Create: `tests/variational_camera_selector/test_model.py`

**Interfaces:**
- Consumes: batched `CandidateGroup` tensors.
- Produces: `CandidateRanker`, `summarize_sequence`, and `listwise_quality_loss`.

- [ ] **Step 1: Write failing model, context-ablation, and loss tests**

```python
class CandidateRankerTests(unittest.TestCase):
    def test_ranker_returns_one_finite_score_per_choice(self):
        scores = self.full_model(
            self.global_tokens, self.x0, self.delta, self.alpha,
            self.span_starts, self.z,
        )
        self.assertEqual(scores.shape, (2, 5))
        self.assertTrue(torch.isfinite(scores).all())

    def test_residual_only_ranker_is_invariant_to_global_context(self):
        first = self.residual_model(self.global_tokens, *self.candidate_inputs)
        second = self.residual_model(self.global_tokens + 100.0, *self.candidate_inputs)
        torch.testing.assert_close(first, second)

    def test_listwise_target_prefers_higher_utility_without_hard_one_hot(self):
        scores = torch.zeros(1, 3, requires_grad=True)
        utilities = torch.tensor([[0.0, 0.10, 0.09]])
        loss, target = listwise_quality_loss(scores, utilities, tau=0.05, return_target=True)
        self.assertGreater(float(target[0, 1]), float(target[0, 2]))
        self.assertGreater(float(target[0, 2]), 0.0)
        loss.backward()
        self.assertTrue(torch.isfinite(scores.grad).all())
```

- [ ] **Step 2: Run RED**

Run: `python -m unittest tests.variational_camera_selector.test_model -v`

Expected: import failure for `CandidateRanker`.

- [ ] **Step 3: Implement shared projection and temporal summaries**

```python
def summarize_sequence(projected: Tensor) -> Tensor:
    delta = projected[..., 1:, :] - projected[..., :-1, :]
    return torch.cat(
        (
            projected.mean(dim=-2), projected.std(dim=-2, unbiased=False),
            delta.mean(dim=-2), delta.std(dim=-2, unbiased=False),
        ),
        dim=-1,
    )

def listwise_quality_loss(
    scores: Tensor,
    utilities: Tensor,
    *,
    tau: float = 0.05,
    return_target: bool = False,
):
    if tau <= 0 or scores.shape != utilities.shape or not torch.isfinite(utilities).all():
        raise ValueError("scores/utilities must be matching finite groups and tau must be positive")
    target = torch.softmax(utilities / tau, dim=-1)
    loss = -(target * torch.log_softmax(scores, dim=-1)).sum(dim=-1).mean()
    return (loss, target) if return_target else loss
```

`CandidateRanker` uses a shared `nn.Linear(2048,128)`, summary vectors for `G`, `X0`, and
scaled residuals, learned span/alpha/z embeddings, residual RMS, LayerNorm, and a two-layer
MLP scalar head. `include_global_context=False` must completely skip the global summary.

- [ ] **Step 4: Run GREEN and gradient/shape regressions**

Run: `python -m unittest tests.variational_camera_selector.test_model -v`

Expected: all pass with finite gradients.

- [ ] **Step 5: Commit the model slice**

```bash
git add pre_experiments/variational_camera_selector/model.py pre_experiments/variational_camera_selector/loss.py tests/variational_camera_selector/test_model.py
git commit -m "Add latent candidate rankers"
```

---

### Task 4: Matched Training and Exact Resume

**Files:**
- Create: `pre_experiments/variational_camera_selector/train.py`
- Create: `tests/variational_camera_selector/test_train.py`

**Interfaces:**
- Consumes: `SelectorTrainingDataset` and `SelectorTrainConfig`.
- Produces: `train_selectors(config) -> SelectorTrainingResult`, matched checkpoints, JSONL metrics, and exact training state.

- [ ] **Step 1: Write failing train/resume/split tests**

```python
class SelectorTrainingTests(unittest.TestCase):
    def test_one_step_trains_both_models_with_finite_loss(self):
        result = train_selectors(self.config(max_steps=1, device="cpu"))
        self.assertEqual(result.completed_step, 1)
        rows = [json.loads(line) for line in result.metrics_path.read_text().splitlines()]
        self.assertTrue(np.isfinite(rows[0]["full_context_loss"]))
        self.assertTrue(np.isfinite(rows[0]["residual_only_loss"]))

    def test_resume_starts_at_exact_next_step_and_rejects_changed_input(self):
        first = train_selectors(self.config(max_steps=2, checkpoint_interval=1))
        second = train_selectors(self.config(max_steps=4, checkpoint_interval=1))
        self.assertEqual((first.start_step, second.start_step), (0, 2))
        with self.assertRaisesRegex(ValueError, "config or input digest"):
            train_selectors(self.config(max_steps=4, tau=0.1))

    def test_validation_scenes_are_never_constructed_by_training_dataset(self):
        self.assertEqual(set(self.training_dataset.scenes), set(FROZEN_TRAIN_SCENES))
        self.assertTrue(set(self.training_dataset.scenes).isdisjoint(FROZEN_VALIDATION_SCENES))
```

- [ ] **Step 2: Run RED**

Run: `python -m unittest tests.variational_camera_selector.test_train -v`

Expected: missing `train_selectors`.

- [ ] **Step 3: Implement deterministic matched training**

```python
@dataclass(frozen=True)
class SelectorTrainConfig:
    prediction_manifest: Path
    privileged_manifest: Path
    run_root: Path
    max_steps: int
    batch_size: int = 1
    learning_rate: float = 1e-4
    tau: float = 0.05
    seed: int = 20260827
    d_model: int = 128
    device: str = "cuda"
    checkpoint_interval: int = 50
    git_commit: str = "unknown"
```

Seed Python/NumPy/Torch, use a deterministic overlap order, train full-context and
residual-only models on the exact same group each step, reject nonfinite loss/gradients, and
atomically checkpoint both models, both optimizers, RNG states, input/config digests, step,
and model config. A changed `max_steps` is allowed for resume; immutable fields are not.

- [ ] **Step 4: Run GREEN and existing training regression**

Run:

```bash
python -m unittest tests.variational_camera_selector.test_train -v
python -m unittest tests.variational_camera_latent.test_train -v
```

Expected: all pass.

- [ ] **Step 5: Commit the training slice**

```bash
git add pre_experiments/variational_camera_selector/train.py tests/variational_camera_selector/test_train.py
git commit -m "Add resumable selector training"
```

---

### Task 5: Prediction-Only Scores and Privileged Evaluation

**Files:**
- Create: `pre_experiments/variational_camera_selector/evaluate.py`
- Create: `tests/variational_camera_selector/test_evaluate.py`

**Interfaces:**
- Consumes: prediction groups and a selector checkpoint; separately consumes score shards plus privileged utility sidecars.
- Produces: `score_scene_candidates`, `load_score_shard`, `evaluate_scene_scores`, and `summarize_calibration`.

- [ ] **Step 1: Write failing prediction firewall and metric-unit tests**

```python
class SelectorEvaluationTests(unittest.TestCase):
    def test_score_api_has_no_privileged_parameter_and_output_has_no_labels(self):
        self.assertFalse(any("privileged" in name for name in inspect.signature(score_scene_candidates).parameters))
        path = score_scene_candidates(self.prediction_record, self.checkpoint, self.output, device="cpu")
        arrays = load_score_shard(path)
        self.assertEqual(arrays["full_context_scores"].shape, (8, 225))
        self.assertFalse(any(fragment in name.lower() for name in arrays for fragment in ("gt", "quality", "error", "depth")))

    def test_privileged_summary_uses_overlap_then_scene_units(self):
        report = summarize_calibration(self.two_scene_fixtures, random_seed=20260827)
        self.assertEqual(report["scene_count"], 2)
        self.assertEqual(report["overlap_count"], 16)
        self.assertEqual(report["inference_unit"], "overlap")
        self.assertEqual(report["aggregate_unit"], "scene")
```

- [ ] **Step 2: Run RED**

Run: `python -m unittest tests.variational_camera_selector.test_evaluate -v`

Expected: missing score/evaluation APIs.

- [ ] **Step 3: Implement atomic scoring and separate label attachment**

Prediction score shards store exact choice IDs, alpha, z, seeds, predicted scores, selected
indices, input/checkpoint digests, and no label. The privileged evaluator validates those
bindings before indexing `relative_improvement` and writes selected utility, no-op, fixed-seed
uniform random, residual-only, full-context, and oracle values to a separate sidecar.

```python
def classify_signal(scene_rows: Sequence[dict[str, float]]) -> str:
    full_beats_noop = all(row["full_context_mean"] > 0.0 for row in scene_rows)
    full_beats_controls = np.mean([
        row["full_context_mean"] - max(row["residual_only_mean"], row["random_mean"])
        for row in scene_rows
    ]) > 0.0
    if full_beats_noop and full_beats_controls:
        return "LEARNABLE_SIGNAL"
    if any(row["full_context_mean"] > 0.0 for row in scene_rows):
        return "WEAK_SIGNAL"
    return "NO_GENERALIZATION"
```

Also report positive-over-1% count, oracle regret, top-1/4/8 coverage, Spearman rank
correlation computed without SciPy, and every per-scene value.

- [ ] **Step 4: Run GREEN and no-leakage regressions**

Run:

```bash
python -m unittest tests.variational_camera_selector.test_evaluate -v
python -m unittest tests.variational_camera_latent.test_privileged_report tests.variational_camera_latent.test_vrfm_residual_scan -v
```

Expected: all pass.

- [ ] **Step 5: Commit the evaluation slice**

```bash
git add pre_experiments/variational_camera_selector/evaluate.py tests/variational_camera_selector/test_evaluate.py
git commit -m "Add separated selector evaluation"
```

---

### Task 6: Fail-Closed Pipeline, Verification, and H20 Runner

**Files:**
- Create: `pre_experiments/variational_camera_selector/pipeline.py`
- Create: `scripts/h20/run_variational_camera_selector.sh`
- Create: `tests/variational_camera_selector/test_pipeline.py`
- Create: `tests/variational_camera_selector/test_h20_runner.py`
- Modify: `pre_experiments/README.md`

**Interfaces:**
- Consumes: sealed Phase 1 root, selector run root, frozen config, and H20 environment.
- Produces: CLI stages `prepare`, `smoke`, `calibration`, `score`, `privileged`, `report`, and `verify`; signed completion markers; one-command H20 orchestration.

- [ ] **Step 1: Write failing stage-order, provenance, and runner tests**

```python
class SelectorPipelineTests(unittest.TestCase):
    def test_privileged_stage_requires_signed_prediction_barrier(self):
        with self.assertRaisesRegex(ValueError, "prediction barrier"):
            run_stage(self.args(stage="privileged"))

    def test_verify_requires_exact_two_validation_scenes_and_all_digests(self):
        completion = verify_completed_run(self.valid_run)
        self.assertEqual(completion["validation_scenes"], ["scene0325_01", "scene0675_00"])
        with self.assertRaises(ValueError):
            verify_completed_run(self.tamper_score_digest())

class SelectorH20RunnerTests(unittest.TestCase):
    def test_wrong_identity_fails_before_python_is_called(self):
        result, calls = self.run_with_fake_path(hostname="wrong-host", user="ubuntu")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(any(call.startswith("python ") for call in calls))

    def test_valid_preflight_calls_one_long_only_auto_stage(self):
        result, calls = self.run_with_fake_path(
            hostname="VM-0-11-ubuntu", user="ubuntu", free_gpu_mib=90000
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        python_calls = [call for call in calls if call.startswith("python ")]
        self.assertEqual(len(python_calls), 1)
        self.assertIn("--stage auto", python_calls[0])
        self.assertIn("/data/yjh/output/variational_camera_selector", python_calls[0])
        self.assertNotIn("matched_random", python_calls[0])
```

- [ ] **Step 2: Run RED**

Run:

```bash
python -m unittest tests.variational_camera_selector.test_pipeline tests.variational_camera_selector.test_h20_runner -v
bash -n scripts/h20/run_variational_camera_selector.sh
```

Expected: missing pipeline and runner.

- [ ] **Step 3: Implement exact stage barriers and CLI**

Use atomic JSON/NPZ/checkpoint writes and immutable payload comparison. `prepare` seals long
contexts/bindings; `smoke` trains one scene and writes a technical completion; `calibration`
requires that completion and trains eight scenes; `score` writes prediction-only outputs;
`privileged` cannot import/load labels before all score files and their aggregate manifest are
sealed; `report` writes exploratory results; `verify` recomputes every digest and exact count.

The H20 runner must check:

```bash
test "$(hostname)" = "VM-0-11-ubuntu"
test "$(id -un)" = "ubuntu"
test -f "$INPUT_ROOT/verified_completion.json"
test "$(git status --porcelain)" = ""
nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits
df -Pk /data
```

It selects one sufficiently free GPU without stopping other jobs, uses the existing Python
environment, runs smoke then calibration through verify, and never reads/exports an HF token.

- [ ] **Step 4: Run GREEN and all relevant tests**

Run:

```bash
python -m unittest discover -s tests/variational_camera_selector -v
python -m unittest discover -s tests/variational_camera_latent -v
bash -n scripts/h20/run_variational_camera_selector.sh
```

Expected: all selector and all 64 existing variational tests pass; one Windows symlink test may
remain skipped.

- [ ] **Step 5: Commit the integrated CPU pipeline**

```bash
git add pre_experiments/variational_camera_selector scripts/h20/run_variational_camera_selector.sh tests/variational_camera_selector pre_experiments/README.md
git commit -m "Add H20 latent selector pipeline"
```

---

### Task 7: Local Verification, Push, and H20 Preflight

**Files:**
- Verify only; no production file is added unless a failing selector test first demonstrates a defect.

**Interfaces:**
- Consumes: the completed selector branch.
- Produces: a clean pushed commit and an authenticated H20 worktree ready to run.

- [ ] **Step 1: Run focused and historical relevant suites from a clean checkout**

Run:

```bash
python -m unittest discover -s tests/variational_camera_selector -v
python -m unittest discover -s tests/variational_camera_latent -v
python -m unittest tests.camera_velocity_ambiguity_02.test_no_gpu_before_integrity tests.camera_velocity_ambiguity_02.test_input_gate -v
git diff --check
git status --short
```

Expected: all selected tests pass; only user-owned `.superpowers/` may remain untracked.

- [ ] **Step 2: Commit any test-first corrections, push the branch, and record HEAD**

Run:

```bash
git push -u origin codex/vrfm-candidate-selector
git rev-parse HEAD
```

Expected: remote branch matches local clean HEAD.

- [ ] **Step 3: Create/update the H20 worktree without touching unrelated worktrees**

Run remotely:

```bash
git -C /home/ubuntu/yjh/vggt fetch origin codex/vrfm-candidate-selector
git -C /home/ubuntu/yjh/vggt worktree add \
  /home/ubuntu/yjh/vggt/.worktrees/vrfm_candidate_selector \
  origin/codex/vrfm-candidate-selector
```

If the worktree already exists, require its clean HEAD to match the pushed commit; do not reset
or overwrite a dirty worktree.

- [ ] **Step 4: Recheck H20 identity and available resources**

Run remotely:

```bash
hostname
id -un
nvidia-smi
df -h /data
pgrep -af 'python|torchrun|deepspeed|run_variational' || true
git -C /home/ubuntu/yjh/vggt/.worktrees/vrfm_candidate_selector status --short --branch
```

Expected: authenticated H20 identity, enough disk, at least one non-conflicting GPU, and clean
selector worktree. If not, fail closed and report the exact blocker.

---

### Task 8: H20 Smoke, Automatic Calibration, and Final Audit

**Files:**
- Formal outputs only under `/data/yjh/output/variational_camera_selector/<run_id>/`.

**Interfaces:**
- Consumes: clean H20 selector worktree and sealed Phase 1 root.
- Produces: smoke completion, 8/2 calibration checkpoint, prediction-only scores/selections, physically separate privileged evaluation, report, and `verified_completion.json`.

- [ ] **Step 1: Launch one-scene smoke with a fresh run ID**

Run remotely:

```bash
cd /home/ubuntu/yjh/vggt/.worktrees/vrfm_candidate_selector
RUN_ID="selector_$(date -u +%Y%m%dT%H%M%SZ)"
RUN_ROOT="/data/yjh/output/variational_camera_selector/$RUN_ID"
bash scripts/h20/run_variational_camera_selector.sh "$RUN_ROOT"
```

Expected: runner validates Phase 1, overfits `scene0000_00` without nonfinite values, seals the
smoke completion, and automatically continues to calibration.

- [ ] **Step 2: Monitor without restarting healthy work**

Check log growth, PID/GPU ownership, disk, checkpoint step, stderr, and completion markers.
Do not delete partial outputs or restart on a single slow sample. Stop and diagnose on nonfinite,
digest/provenance mismatch, disk below 20 GiB, GPU conflict/OOM, nonempty fatal stderr, or process
exit without a valid marker.

- [ ] **Step 3: Verify the completed run twice for idempotence**

Run remotely:

```bash
python -m pre_experiments.variational_camera_selector.pipeline --stage verify --run-root "$RUN_ROOT"
python -m pre_experiments.variational_camera_selector.pipeline --stage verify --run-root "$RUN_ROOT"
sha256sum "$RUN_ROOT/verified_completion.json" "$RUN_ROOT/reports/calibration_summary.json"
```

Expected: both verifies exit zero and keep byte-identical completion/report digests.

- [ ] **Step 4: Perform an independent artifact audit**

Independently enumerate exact directories, reject symlinks/extras, validate every manifest/file
SHA-256, load every NPZ with `allow_pickle=False`, verify finite arrays/counts/roles, confirm score
shards contain no forbidden label names, recompute selection indices and scene-level metrics from
the physically separate sidecars, and confirm the report classification.

- [ ] **Step 5: Publish a concise result report**

Report run root, code commit, GPU, step count, exact scene split, score/report/completion digests,
full-context/residual-only/random/no-op/oracle held-out results, classification, limitations, and
the next decision. Do not pull candidate tensors or checkpoints to Windows.
