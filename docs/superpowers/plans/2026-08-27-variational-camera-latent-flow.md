# Variational Camera Latent Flow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a long-window-only Camera-token VRFM experiment that uses offline short-window tokens as equal-weight teachers, exports prediction-only candidate latents plus a physically separate privileged sidecar, and completes one-scene smoke followed by ten-scene calibration on H20.

**Architecture:** Reuse authenticated CVA02 global/local prediction artifacts. For every adjacent local pair, construct one shared 50-frame global source and two equal-weight local endpoints while retaining the full 500-frame global token sequence as cross-attention context. Train a latent-conditioned velocity network and a capacity-matched deterministic baseline, sample 32 fixed-z trajectories per overlap, decode through the frozen Camera Head, and publish resumable hashed shards under `/data/yjh/output/variational_camera_latent/$RUN_ID/`.

**Tech Stack:** Python 3.10+, NumPy, PyTorch, repository `unittest`, official VGGT Camera Head, Bash H20 runner.

**Spec:** `docs/superpowers/specs/2026-08-27-variational-camera-latent-flow-design.md`

## Global Constraints

- Formal compute runs only on H20 as `ubuntu@VM-0-11-ubuntu`; re-check GPU, disk, active jobs, worktree, checkpoint, and verified ScanNet marker first.
- Reuse the authenticated source run `/data/yjh/output/camera_velocity_ambiguity/cva02_20260826T2319CST_7e6fd06`; verify its completion/provenance before building new shards and never mutate it.
- Formal inference receives only full long-window Camera tokens; short-window tokens exist only in offline source shards and training.
- GT pose/depth/error data lives only under `privileged_labels`; prediction-only loaders and model APIs must not accept a privileged path.
- The flow state is `[B,50,2048]`; full long context is `[B,N,2048]`; left/right teachers are equal weight and have no GT quality weighting.
- One 16-D z is fixed across all 50 frames and all ODE steps; formal sampling uses 32 z values and 16-step Heun integration.
- VGGT and Camera Head remain frozen; do not modify upstream `vggt/` APIs.
- Use atomic files, SHA-256 manifests, exact provenance, idempotent resume, and fail closed on schema, alignment, nonfinite, or leakage errors.
- Do not use the exposed H20 Hugging Face token and do not pull large H20 outputs to Windows.
- Use `unittest`; CPU tests must not require CUDA, checkpoints, network, ScanNet credentials, or large artifacts.

---

### Task 1: Prediction-only Source Shards and Manifest

**Files:**
- Create: `pre_experiments/variational_camera_latent/__init__.py`
- Create: `pre_experiments/variational_camera_latent/contracts.py`
- Create: `pre_experiments/variational_camera_latent/schema.py`
- Create: `pre_experiments/variational_camera_latent/source.py`
- Create: `tests/variational_camera_latent/__init__.py`
- Create: `tests/variational_camera_latent/test_source_schema.py`

**Interfaces:**
- Consumes: authenticated CVA02 scene roots containing `global/prediction.npz` and `local/window_NNN/prediction.npz`.
- Produces: `build_scene_source_shard(prediction_root: Path, destination: Path, *, role: str) -> SourceShardRecord`, `load_source_shard(path: Path) -> dict[str,np.ndarray]`, and `write_source_manifest(path: Path, *, dataset_root: Path, records: Sequence[SourceShardRecord], source_run_digest: str) -> dict[str,object]`.

- [ ] **Step 1: Write failing alignment and firewall tests**

```python
class SourceShardTests(unittest.TestCase):
    def test_builds_eight_shared_50_pairs_from_nine_windows(self):
        record = build_scene_source_shard(self.predictions, self.output, role="train")
        arrays = load_source_shard(record.path)
        self.assertEqual(arrays["global_camera_tokens"].shape, (500, 2048))
        self.assertEqual(arrays["short_camera_tokens"].shape, (9, 100, 2048))
        self.assertEqual(arrays["overlap_long_tokens"].shape, (8, 50, 2048))
        np.testing.assert_array_equal(
            arrays["overlap_left_tokens"][0], arrays["short_camera_tokens"][0, 50:]
        )
        np.testing.assert_array_equal(
            arrays["overlap_right_tokens"][0], arrays["short_camera_tokens"][1, :50]
        )

    def test_rejects_frame_misalignment_and_object_arrays(self):
        with self.assertRaisesRegex(ValueError, "frame IDs"):
            build_scene_source_shard(self.misaligned, self.output, role="train")
```

- [ ] **Step 2: Run tests and verify the module is absent**

Run: `python -m unittest tests.variational_camera_latent.test_source_schema -v`

Expected: import failure for `pre_experiments.variational_camera_latent`.

- [ ] **Step 3: Implement focused contracts and atomic schema**

```python
@dataclass(frozen=True)
class SourceShardRecord:
    scene: str
    role: str
    path: Path
    overlap_count: int
    sha256: str

def validate_source_shard(arrays: Mapping[str, np.ndarray]) -> None:
    required = SOURCE_REQUIRED_MEMBERS - set(arrays)
    if required:
        raise ValueError(f"source shard is missing members: {sorted(required)}")
    if any(np.asarray(value).dtype.hasobject for value in arrays.values()):
        raise ValueError("source shard may not contain object arrays")
    validate_source_shapes_and_alignment(arrays)

def save_source_shard(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    normalized = {name: np.asarray(value) for name, value in arrays.items()}
    validate_source_shard(normalized)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **normalized)
    temporary.replace(path)

def load_source_shard(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    validate_source_shard(arrays)
    return arrays
```

Validation must enforce finite float arrays, exact 2048 channels, 100-frame local windows, 50-frame primary overlaps, unique Unicode sample IDs, strictly increasing frame IDs, exact left/right frame identity, and no GT-named members.

- [ ] **Step 4: Implement the CVA02 artifact adapter**

```python
def build_scene_source_shard(
    prediction_root: Path,
    destination: Path,
    *,
    role: str,
) -> SourceShardRecord:
    global_arrays = load_prediction_arrays(prediction_root / "global/prediction.npz")
    locals_ = load_ordered_local_predictions(prediction_root / "local")
    return save_aligned_primary_overlaps(global_arrays, locals_, destination, role=role)
```

Reuse the repository's completed-artifact validation where possible; never silently read an incomplete `.npz` without its authenticated completion sidecar.

- [ ] **Step 5: Run focused tests and commit**

Run: `python -m unittest tests.variational_camera_latent.test_source_schema -v`

Expected: all tests pass.

```bash
git add pre_experiments/variational_camera_latent tests/variational_camera_latent
git commit -m "Add variational camera latent source schema"
```

### Task 2: Latent Compatibility Preflight and Camera Decode Adapter

**Files:**
- Create: `pre_experiments/variational_camera_latent/camera.py`
- Create: `tests/variational_camera_latent/test_camera.py`

**Interfaces:**
- Consumes: frozen `model.camera_head`, `[B,S,2048]` normalized tokens, and source shards.
- Produces: `decode_camera_tokens(camera_head: nn.Module, tokens: Tensor, *, iterations: int = 4) -> Tensor`, `run_latent_preflight(camera_head: nn.Module, long_tokens: Tensor, left_tokens: Tensor, right_tokens: Tensor) -> dict[str,object]`.

- [ ] **Step 1: Write failing decode and fixed-alpha preflight tests**

```python
def test_preflight_checks_both_paths_at_fixed_alphas(self):
    report = run_latent_preflight(self.fake_head, self.long, self.left, self.right)
    self.assertEqual(report["alphas"], [0.0, 0.25, 0.5, 0.75, 1.0])
    self.assertTrue(report["all_finite"])
    self.assertEqual(self.fake_head.calls, 13)  # 3 endpoints + 2 * 5 path points

def test_decode_rejects_nonfinite_pose(self):
    with self.assertRaisesRegex(ValueError, "non-finite"):
        decode_camera_tokens(self.bad_head, self.long)
```

- [ ] **Step 2: Verify failure**

Run: `python -m unittest tests.variational_camera_latent.test_camera -v`

Expected: import failure for `camera.py`.

- [ ] **Step 3: Implement frozen Camera Head decoding and preflight**

```python
def decode_camera_tokens(camera_head: nn.Module, tokens: Tensor) -> Tensor:
    with torch.no_grad():
        raw = camera_head.decode_pose_tokens(tokens, num_iterations=4)[-1]
    if raw.shape[-1] != 9 or not torch.isfinite(raw).all():
        raise ValueError("Camera Head produced non-finite or malformed pose encoding")
    return raw
```

Do not align tokens. Record decoded endpoint equality and finite interpolation diagnostics; poor intermediate scientific quality is reported, not raised.

- [ ] **Step 4: Run tests and commit**

```bash
python -m unittest tests.variational_camera_latent.test_camera -v
git add pre_experiments/variational_camera_latent/camera.py tests/variational_camera_latent/test_camera.py
git commit -m "Add Camera token compatibility preflight"
```

### Task 3: Contextual VRFM and Deterministic Baseline

**Files:**
- Create: `pre_experiments/variational_camera_latent/model.py`
- Create: `pre_experiments/variational_camera_latent/flow.py`
- Create: `tests/variational_camera_latent/test_model_flow.py`

**Interfaces:**
- Produces: `VRFMModel`, `DeterministicRFMModel`, `RecognitionPosterior`, `vrfm_loss(model: VRFMModel, posterior: RecognitionPosterior, batch: TrainingBatch, *, progress: float, beta_max: float = 1e-4) -> LossOutput`, and `heun_sample(model: VRFMModel, x0: Tensor, context: Tensor, span: Tensor, z: Tensor, *, steps: int = 16) -> Tensor`.

- [ ] **Step 1: Write failing shape, z-scope, and loss tests**

```python
def test_vrfm_shapes_and_single_segment_z(self):
    model = VRFMModel(d_model=32, z_dim=4, layers=1, heads=4)
    out = model(self.x_t, self.t, self.z, self.global_tokens, self.span_starts)
    self.assertEqual(out.shape, self.x_t.shape)

def test_heun_reuses_one_z_for_all_steps(self):
    recorder = RecordingVelocity()
    heun_sample(recorder, self.x0, self.context, self.span, self.z, steps=4)
    self.assertTrue(all(torch.equal(call.z, self.z) for call in recorder.calls))

def test_left_and_right_pairs_are_equal_weight(self):
    batch = make_training_pairs(self.long, self.left, self.right)
    self.assertEqual(batch.endpoint_side.tolist(), [0, 1])
    self.assertTrue(torch.equal(batch.weights, torch.ones(2)))
```

- [ ] **Step 2: Verify failure**

Run: `python -m unittest tests.variational_camera_latent.test_model_flow -v`

Expected: missing model/flow modules.

- [ ] **Step 3: Implement the direct Camera-token models**

```python
class VRFMModel(nn.Module):
    def forward(
        self,
        x_t: Tensor,
        t: Tensor,
        z: Tensor,
        global_tokens: Tensor,
        span_starts: Tensor,
    ) -> Tensor:
        state = self.input_adapter(x_t) + self.time_embedding(t)[:, None, :]
        state = state + self.z_embedding(z)[:, None, :] + self.span_embedding(span_starts)[:, None, :]
        context = self.context_adapter(global_tokens)
        for block in self.blocks:
            state = block(state, context)
        return self.output_adapter(state)

class RecognitionPosterior(nn.Module):
    def forward(
        self, global_tokens: Tensor, x0: Tensor, x1: Tensor, span_starts: Tensor
    ) -> tuple[Tensor, Tensor]:
        delta = self.delta_adapter(x1 - x0).mean(dim=1)
        context = self.context_adapter(global_tokens).mean(dim=1)
        stats = self.output(torch.cat((delta, context, self.span_embedding(span_starts)), dim=-1))
        return stats.chunk(2, dim=-1)
```

Use 2048→256 adapters, four self/cross-attention blocks with eight heads, a 16-D z embedding, learned span embedding, and 256→2048 velocity output. Tiny constructor values used by CPU tests must follow the same code path.

- [ ] **Step 4: Implement interpolation, KL warm-up, and Heun integration**

```python
@dataclass(frozen=True)
class LossOutput:
    total: Tensor
    velocity_mse: Tensor
    kl: Tensor
    beta: float

def vrfm_loss(model, posterior, batch, *, progress: float, beta_max: float = 1e-4):
    t = torch.rand(batch.x0.shape[0], device=batch.x0.device)
    x_t = torch.lerp(batch.x0, batch.x1, t[:, None, None])
    target = batch.x1 - batch.x0
    mu, log_var = posterior(batch.context, batch.x0, batch.x1, batch.span_starts)
    z = mu + torch.exp(0.5 * log_var) * torch.randn_like(mu)
    prediction = model(x_t, t, z, batch.context, batch.span_starts)
    mse = torch.mean((prediction - target) ** 2)
    kl = -0.5 * torch.mean(1.0 + log_var - mu.square() - log_var.exp())
    beta = beta_max * min(max(progress / 0.2, 0.0), 1.0)
    return LossOutput(total=mse + beta * kl, velocity_mse=mse, kl=kl, beta=beta)

def heun_sample(model, x0, context, span, z, *, steps: int = 16):
    state = x0
    dt = 1.0 / steps
    for index in range(steps):
        t0 = torch.full((x0.shape[0],), index * dt, device=x0.device)
        first = model(state, t0, z, context, span)
        proposal = state + dt * first
        second = model(proposal, t0 + dt, z, context, span)
        state = state + 0.5 * dt * (first + second)
    return state
```

Sampling must not call the recognition posterior. Deterministic RFM shares the velocity backbone but omits z and KL.

- [ ] **Step 5: Run tests and commit**

```bash
python -m unittest tests.variational_camera_latent.test_model_flow -v
git add pre_experiments/variational_camera_latent/model.py pre_experiments/variational_camera_latent/flow.py tests/variational_camera_latent/test_model_flow.py
git commit -m "Add contextual variational Camera token flow"
```

### Task 4: Trainer, Checkpoint, and Resume

**Files:**
- Create: `pre_experiments/variational_camera_latent/train.py`
- Create: `tests/variational_camera_latent/test_train.py`

**Interfaces:**
- Consumes: source manifest and train scene records.
- Produces: `train_models(config: TrainConfig) -> TrainingResult`, atomic VRFM/deterministic checkpoints, `training_state.json`, and resumable metric JSONL.

- [ ] **Step 1: Write failing dataset, firewall, and resume tests**

```python
def test_training_dataset_never_accepts_privileged_root(self):
    self.assertNotIn("privileged", inspect.signature(OverlapDataset).parameters)

def test_resume_restores_exact_next_step(self):
    first = train_models(self.config(max_steps=3))
    resumed = train_models(self.config(max_steps=5))
    self.assertEqual(resumed.start_step, 3)
    self.assertEqual(resumed.completed_step, 5)
```

- [ ] **Step 2: Verify failure**

Run: `python -m unittest tests.variational_camera_latent.test_train -v`

- [ ] **Step 3: Implement deterministic scene split and pair dataset**

```python
class OverlapDataset(Dataset):
    def __getitem__(self, index: int) -> TrainingExample:
        pair = index // 2
        side = index % 2
        endpoint = self.left[pair] if side == 0 else self.right[pair]
        return TrainingExample(self.global_tokens, self.long[pair], endpoint, self.starts[pair])
```

Use the first eight calibration scenes as train and last two as validation; keep role in manifests and logs.

- [ ] **Step 4: Implement matched training and atomic resume**

Train VRFM and deterministic models with separate optimizers but identical steps, batch order, learning-rate schedule, and precision. Save model, posterior, optimizer, scheduler, RNG, step, config digest, source manifest digest, and git commit to a temporary checkpoint before atomic replace.

- [ ] **Step 5: Run tests and commit**

```bash
python -m unittest tests.variational_camera_latent.test_train -v
git add pre_experiments/variational_camera_latent/train.py tests/variational_camera_latent/test_train.py
git commit -m "Add resumable VRFM training"
```

### Task 5: Candidate Shards and Exploratory Clustering

**Files:**
- Create: `pre_experiments/variational_camera_latent/candidates.py`
- Create: `pre_experiments/variational_camera_latent/clustering.py`
- Create: `tests/variational_camera_latent/test_candidates.py`

**Interfaces:**
- Produces: `generate_scene_candidates(source_path: Path, checkpoint_path: Path, destination: Path, *, samples: int = 32, steps: int = 16) -> CandidateShardRecord`, `two_means(features: np.ndarray) -> ClusterResult`, and candidate manifest entries.

- [ ] **Step 1: Write failing candidate shape and analysis-replay tests**

```python
def test_candidate_shard_keeps_raw_samples(self):
    arrays = load_candidate_shard(generate_scene_candidates(self.fixture))
    self.assertEqual(arrays["z"].shape, (8, 32, 16))
    self.assertEqual(arrays["corrected_camera_tokens"].shape, (8, 32, 50, 2048))
    self.assertEqual(arrays["latent_cluster_ids"].shape, (8, 32))

def test_clustering_can_be_replayed_without_model(self):
    first = analyze_candidate_shard(self.path)
    second = analyze_candidate_shard(self.path)
    self.assertEqual(first, second)
```

- [ ] **Step 2: Verify failure**

Run: `python -m unittest tests.variational_camera_latent.test_candidates -v`

- [ ] **Step 3: Implement deterministic two-means and summaries**

Use farthest-pair initialization, bounded iterations, deterministic tie-breaking, one-cluster/two-cluster SSE ratio, and no scikit-learn dependency. Cluster flattened centered token deltas for latent IDs and aligned decoded trajectory features for camera IDs.

- [ ] **Step 4: Implement 32-sample candidate generation**

Generate VRFM and deterministic candidate shards atomically. The deterministic shard repeats no artificial z dimension; it records one candidate per overlap plus matching metadata. Store float32 tensors, decoded raw/activated cameras, seeds, step count, checkpoint digest, and source sample IDs.

- [ ] **Step 5: Run tests and commit**

```bash
python -m unittest tests.variational_camera_latent.test_candidates -v
git add pre_experiments/variational_camera_latent/candidates.py pre_experiments/variational_camera_latent/clustering.py tests/variational_camera_latent/test_candidates.py
git commit -m "Add VRFM latent candidate export"
```

### Task 6: Privileged Sidecar and Exploration Report

**Files:**
- Create: `pre_experiments/variational_camera_latent/privileged.py`
- Create: `pre_experiments/variational_camera_latent/report.py`
- Create: `tests/variational_camera_latent/test_privileged_report.py`

**Interfaces:**
- Consumes: candidate shards plus prepared ScanNet GT only inside the privileged command path.
- Produces: `write_privileged_scene_sidecar(source_path: Path, candidate_path: Path, prepared_scene_root: Path, destination: Path) -> PrivilegedShardRecord`, `summarize_run(prediction_manifest: Path, privileged_manifest: Path, destination: Path) -> dict[str,object]`, and `PROMISING|WEAK_SIGNAL|NO_SIGNAL` report.

- [ ] **Step 1: Write failing isolation and classification tests**

```python
def test_prediction_only_candidate_generation_cannot_receive_gt(self):
    self.assertNotIn("gt", inspect.signature(generate_scene_candidates).parameters)

def test_report_keeps_weak_signal_as_successful_completion(self):
    report = summarize_run(self.weak_fixture)
    self.assertEqual(report["signal"], "WEAK_SIGNAL")
    self.assertTrue(report["technically_complete"])
```

- [ ] **Step 2: Verify failure**

Run: `python -m unittest tests.variational_camera_latent.test_privileged_report -v`

- [ ] **Step 3: Implement sidecar-only candidate evaluation**

Reuse frozen raw-GT and prediction-only gauge alignment helpers from CVA02. Link by sample ID, save teacher/candidate errors and improvement over global baseline, and reject any sidecar whose source/candidate manifest digest differs.

- [ ] **Step 4: Implement exploratory summaries**

Report z sensitivity, latent/camera pairwise distances, cluster summaries, teacher coverage, best-of-32 improvement, KL/posterior statistics, deterministic comparison, and technical failures. Use soft signal labels; do not turn weak scientific evidence into process failure.

- [ ] **Step 5: Run tests and commit**

```bash
python -m unittest tests.variational_camera_latent.test_privileged_report -v
git add pre_experiments/variational_camera_latent/privileged.py pre_experiments/variational_camera_latent/report.py tests/variational_camera_latent/test_privileged_report.py
git commit -m "Add privileged VRFM evaluation sidecar"
```

### Task 7: Orchestrator and H20 Runner

**Files:**
- Create: `pre_experiments/variational_camera_latent/pipeline.py`
- Create: `scripts/h20/run_variational_camera_latent.sh`
- Create: `tests/variational_camera_latent/test_pipeline_runner.py`
- Modify: `docs/camera_velocity_ambiguity_02_status.md`

**Interfaces:**
- Produces CLI stages `source`, `smoke`, `calibration`, `privileged`, `report`, `verify`, and a formal H20 runner that performs smoke then automatically expands to ten scenes.

- [ ] **Step 1: Write failing CLI and runner guard tests**

```python
def test_runner_uses_h20_and_new_output_root(self):
    text = RUNNER.read_text(encoding="utf-8")
    self.assertIn('RESULT_ROOT="${RESULT_ROOT:-/data/yjh/output/variational_camera_latent}"', text)
    self.assertIn('[[ "$(hostname)" == "VM-0-11-ubuntu" ]]', text)
    self.assertIn('SMOKE_SCENE_LIMIT="1"', text)
    self.assertIn('CALIBRATION_SCENE_LIMIT="10"', text)
    self.assertNotIn("HF_TOKEN", text)
```

- [ ] **Step 2: Verify failure**

Run: `python -m unittest tests.variational_camera_latent.test_pipeline_runner -v`

- [ ] **Step 3: Implement the resumable stage orchestrator**

```python
def run_stage(args: argparse.Namespace) -> Path:
    verify_inputs_and_provenance(args)
    if args.stage == "source": return build_sources(args)
    if args.stage == "smoke": return train_and_sample_smoke(args)
    if args.stage == "calibration": return train_and_sample_calibration(args)
    if args.stage == "privileged": return build_privileged_sidecars(args)
    if args.stage == "report": return publish_report(args)
    if args.stage == "verify": return verify_completed_run(args.run_root)
    raise ValueError(f"unsupported stage: {args.stage}")
```

Stage completion JSON must include a canonical digest and all upstream manifest/checkpoint digests. Existing exact completions resume; mismatches fail closed.

- [ ] **Step 4: Implement the formal H20 shell runner**

Use `set -euo pipefail`, exact host/user/branch/GPU/disk/checkpoint/verified-marker checks, a per-run lock, logs under the run root, `CUDA_VISIBLE_DEVICES`, one-scene smoke, digest validation, and automatic ten-scene calibration. Never source credential files.

- [ ] **Step 5: Run focused and full CPU tests**

```bash
python -m unittest discover -s tests/variational_camera_latent -v
python -m unittest discover -s tests -v
```

Expected: all tests pass without CUDA/network/data.

- [ ] **Step 6: Commit**

```bash
git add pre_experiments/variational_camera_latent scripts/h20/run_variational_camera_latent.sh tests/variational_camera_latent docs/camera_velocity_ambiguity_02_status.md
git commit -m "Add H20 variational Camera latent pipeline"
```

### Task 8: H20 Sync, Smoke, Calibration, and Final Verification

**Files:**
- Remote worktree: `/home/ubuntu/yjh/vggt/.worktrees/camera_velocity_ambiguity_02_pre_experiment`
- Remote output: `/data/yjh/output/variational_camera_latent/$RUN_ID/`

**Interfaces:**
- Consumes the committed branch and existing authenticated CVA02 prediction run.
- Produces complete source/candidate/privileged manifests, checkpoints, logs, and summaries.

- [ ] **Step 1: Verify H20 before mutation**

```bash
ssh h20 'hostname; whoami; nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader; df -h /data; pgrep -af "python|run_variational_camera_latent" || true; git -C /home/ubuntu/yjh/vggt/.worktrees/camera_velocity_ambiguity_02_pre_experiment status --short --branch'
```

Expected: `VM-0-11-ubuntu`, `ubuntu`, an available H20 GPU, sufficient `/data`, no conflicting run, correct clean worktree.

- [ ] **Step 2: Push and fast-forward the isolated remote worktree**

```bash
git push origin codex/camera_velocity_ambiguity_02_pre_experiment
ssh h20 'git -C /home/ubuntu/yjh/vggt/.worktrees/camera_velocity_ambiguity_02_pre_experiment pull --ff-only'
```

Do not overwrite remote changes; stop on a dirty or divergent worktree.

- [ ] **Step 3: Run the formal pipeline**

```powershell
$runId = "vrfm_camera_$((Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ'))"
$gpuIndex = (ssh h20 "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2n | head -n1 | cut -d, -f1").Trim()
ssh h20 "cd /home/ubuntu/yjh/vggt/.worktrees/camera_velocity_ambiguity_02_pre_experiment && RUN_ID=$runId GPU_INDEX=$gpuIndex bash scripts/h20/run_variational_camera_latent.sh"
```

The runner must complete smoke before launching ten-scene calibration. Monitor logs, GPU utilization, disk, completion manifests, and candidate shard growth without pulling large files.

- [ ] **Step 4: Verify artifacts and scientific summary remotely**

```powershell
ssh h20 "cd /home/ubuntu/yjh/vggt/.worktrees/camera_velocity_ambiguity_02_pre_experiment && /home/ubuntu/anaconda3/envs/vggt-gx/bin/python -m pre_experiments.variational_camera_latent.pipeline --stage verify --run-root /data/yjh/output/variational_camera_latent/$runId"
```

Verify SHA-256 manifests, exact 10-scene roles, candidate shapes, finite tensors/cameras, separated privileged tree, checkpoint resume metadata, and final signal label.

- [ ] **Step 5: Commit any evidence-only fixes, push, and report**

Run full CPU tests after any fix, commit each coherent correction, fast-forward H20, resume from exact stage, and report the final run ID, output path, counts, signal label, z sensitivity, best-of-32 improvement, deterministic comparison, and retained limitations. Do not claim discrete multimodality unless the data actually supports it.
