# CO3D Pretraining Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. The user explicitly prohibited subagent execution.

**Goal:** Add a resumable CO3Dv2 prediction-cache pipeline and compare CO3D-pretrained camera refiners against ScanNet-only controls without changing VGGT rotations.

**Architecture:** A dependency-free CO3D reader creates frozen ordered-clip manifests from official annotations. A camera-only VGGT runner exports compact, versioned refiner shards shared with ScanNet. A streaming, length-bucketed trainer pretrains on 50/75/100-frame CO3D clips and initializes the existing 100-frame ScanNet refiner through an explicit parent checkpoint.

**Tech Stack:** Python 3.10+, NumPy, Pillow, PyTorch, existing VGGT modules, `unittest`, Bash, JSON/NPZ artifacts.

## Global Constraints

- Do not add PyTorch3D, create Conda environments, download VGGT weights, or require network access in tests.
- Use CO3Dv2 RGB and camera annotations only; ignore depth, masks, and point clouds.
- Preserve annotation frame order and never concatenate or loop unrelated sequences.
- GT constructs labels and metrics only; conditions and alignment at inference remain prediction-only.
- Predict camera-center translation residuals only and copy global VGGT rotations exactly.
- Cache only the frozen iteration-0 translation units, never full Camera Head traces.
- Keep datasets under `/root/autodl-tmp/datasets/co3dv2` and generated artifacts under `/root/autodl-tmp/results`.
- Keep existing ScanNet manifests and training commands readable until their canonical cache has been exported.
- Tests use `unittest`, run on CPU, and require no dataset, checkpoint, CUDA, or credentials.

---

### Task 1: CO3D Annotation Reader and Frozen Clip Manifest

**Files:**
- Create: `configs/co3d_raydiffusion_category_split.json`
- Create: `pre_experiments/camera_refiner_training/co3d.py`
- Create: `pre_experiments/camera_refiner_training/build_co3d_manifest.py`
- Create: `tests/camera_refiner_training/test_co3d.py`

**Interfaces:**
- Produces: `Co3DFrame`, `ClipSpec`, `load_category_sequences(root, category, min_quality)`, `pytorch3d_viewpoint_to_c2w(rotation, translation)`, `select_ordered_clips(...)`, and `write_clip_manifest(...)`.
- Consumes later: a JSON manifest containing `schema_version`, split digest, and clips with absolute-data-root-relative image paths.

- [ ] **Step 1: Write camera-conversion and annotation tests**

```python
def test_pytorch3d_viewpoint_conversion_preserves_camera_center(self):
    rotation = np.eye(3)
    translation = np.array([1.0, 2.0, 3.0])
    c2w = pytorch3d_viewpoint_to_c2w(rotation, translation)
    np.testing.assert_allclose(c2w[:3, 3], [-1.0, -2.0, -3.0])
    np.testing.assert_allclose(c2w[:3, :3], np.diag([-1.0, -1.0, 1.0]))

def test_reader_sorts_frame_number_not_filename(self):
    sequences = load_category_sequences(self.root, "apple", min_quality=0.5)
    self.assertEqual([frame.frame_number for frame in sequences["sequence_a"]], [2, 10])
```

- [ ] **Step 2: Run the tests and verify missing-module failure**

Run: `python -m unittest tests.camera_refiner_training.test_co3d -v`

Expected: FAIL because `co3d.py` does not exist.

- [ ] **Step 3: Implement the dependency-free CO3D reader and pose conversion**

Parse `frame_annotations.jgz` and `sequence_annotations.jgz` with `gzip` and `json`.
Reject sequences when `viewpoint_quality_score <= 0.5`, an image is missing, a
viewpoint is absent, a rotation is non-finite or has determinant outside
`[0.99, 1.01]`, or a translation is non-finite. Convert PyTorch3D row-vector
world-to-view cameras to OpenCV-style camera-to-world as:

```python
axis_flip = np.diag([-1.0, -1.0, 1.0])
c2w[:3, :3] = rotation @ axis_flip
c2w[:3, 3] = -(rotation @ translation)
```

- [ ] **Step 4: Test deterministic, ordered clip selection**

```python
def test_clip_selection_is_seeded_ordered_and_sequence_disjoint(self):
    first = select_ordered_clips(frames, lengths=(50, 75, 100), max_clips=4, seed=33)
    second = select_ordered_clips(frames, lengths=(50, 75, 100), max_clips=4, seed=33)
    self.assertEqual(first, second)
    for clip in first:
        self.assertEqual(tuple(sorted(clip.frame_numbers)), clip.frame_numbers)
        self.assertIn(clip.temporal_stride, (1, 2))
```

Use a SHA-256 hash of category and sequence to assign training-category sequences
90/10 to `pretrain` and `validation`. Categories in the frozen 10-category list use
role `category_holdout`. Select at most four contiguous clips per sequence, preferring
100, then 75, then 50 valid frames. Record rejected sequences and exact reasons.

- [ ] **Step 5: Implement and test the manifest CLI**

Run:

```bash
python -m pre_experiments.camera_refiner_training.build_co3d_manifest \
  --data-root /root/autodl-tmp/datasets/co3dv2 \
  --category-split configs/co3d_raydiffusion_category_split.json \
  --out /root/autodl-tmp/results/camera_refiner_data_construction/co3d/manifest.json \
  --seed 33 --max-clips-per-sequence 4
```

The CLI must write atomically, include a canonical digest, and fail when an image
path escapes `--data-root`.

- [ ] **Step 6: Commit**

```bash
git add configs/co3d_raydiffusion_category_split.json \
  pre_experiments/camera_refiner_training/co3d.py \
  pre_experiments/camera_refiner_training/build_co3d_manifest.py \
  tests/camera_refiner_training/test_co3d.py
git commit -m "Add CO3D clip manifest builder"
```

### Task 2: Canonical Refiner Shard Contract

**Files:**
- Create: `pre_experiments/camera_refiner_training/shards.py`
- Create: `tests/camera_refiner_training/test_shards.py`
- Modify: `pre_experiments/camera_refiner_training/data.py`
- Modify: `tests/camera_refiner_training/test_data.py`

**Interfaces:**
- Produces: `RefinerShard`, `RefinerManifestEntry`, `save_refiner_shard(path, shard)`, `load_refiner_shard(path)`, `write_refiner_manifest(...)`, and `load_refiner_manifest(...)`.
- The shard carries one or more same-length windows so ScanNet can retain overlapping windows while each CO3D clip uses one window.

- [ ] **Step 1: Write strict round-trip tests**

```python
def test_refiner_shard_round_trip_rejects_rotation_or_digest_corruption(self):
    shard = make_shard(source="co3d", window_count=1, frame_count=50, condition_dim=136)
    save_refiner_shard(self.path, shard)
    restored = load_refiner_shard(self.path)
    np.testing.assert_array_equal(restored.frame_ids, shard.frame_ids)
    self.assertEqual(restored.source, "co3d")
```

Also test duplicate sample IDs, path traversal, checksum mismatch, non-increasing
frame IDs, non-finite arrays, malformed homogeneous rows, and mismatched condition
dimensions.

- [ ] **Step 2: Run the tests and verify failure**

Run: `python -m unittest tests.camera_refiner_training.test_shards -v`

Expected: FAIL because `shards.py` does not exist.

- [ ] **Step 3: Implement schema version 2**

Use a frozen NPZ member set:

```python
REFINER_SHARD_MEMBERS = {
    "schema_version", "sample_id", "group_id", "source", "role",
    "frame_ids", "starts", "condition", "target_residual",
    "global_centers", "alignment_residual", "global_c2w",
    "gt_c2w_raw", "gauge_origin", "gauge_rotation", "gauge_scale",
}
```

Shapes are `frame_ids [W,S]`, `condition [W,S,D]`, residual/centers `[W,S,3]`,
alignment `[W,S]`, starts `[W]`, and full trajectories `[F,4,4]`. Save through a
temporary sibling and atomic rename. Manifest entries include `sample_id`,
`group_id`, `source`, `role`, `path`, `sha256`, `window_count`, `window_length`, and
`condition_dim`.

- [ ] **Step 4: Preserve the existing legacy adapter**

Keep `build_scene_windows(...)` and legacy `load_dataset_manifest(...)` intact for
existing ScanNet artifacts. Add canonical loaders beside them; do not silently treat
legacy files as schema version 2.

- [ ] **Step 5: Run focused tests**

Run: `python -m unittest tests.camera_refiner_training.test_shards tests.camera_refiner_training.test_data -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add pre_experiments/camera_refiner_training/shards.py \
  pre_experiments/camera_refiner_training/data.py \
  tests/camera_refiner_training/test_shards.py \
  tests/camera_refiner_training/test_data.py
git commit -m "Define canonical refiner shards"
```

### Task 3: Shared Feature Construction and ScanNet Export

**Files:**
- Create: `pre_experiments/camera_refiner_training/features.py`
- Create: `pre_experiments/camera_refiner_training/export_scannet_cache.py`
- Create: `tests/camera_refiner_training/test_features.py`
- Create: `tests/camera_refiner_training/test_export_scannet_cache.py`
- Modify: `pre_experiments/camera_refiner_training/data.py`

**Interfaces:**
- Produces: `build_refiner_features(...) -> RefinerFeatures` and a CLI that materializes legacy ScanNet scene data as canonical shards.
- Consumes: global/local poses, selected global/local unit arrays, boundary distance, observation validity, frame IDs, and raw GT.

- [ ] **Step 1: Write a GT-isolation regression test**

```python
def test_gt_changes_targets_but_never_conditions(self):
    first = build_refiner_features(gt_c2w=poses(offset=0.1), **prediction_inputs)
    second = build_refiner_features(gt_c2w=poses(offset=0.3), **prediction_inputs)
    np.testing.assert_array_equal(first.condition, second.condition)
    self.assertFalse(np.array_equal(first.target_residual, second.target_residual))
```

Also assert a condition dimension of `13 + 3 * unit_count`, prediction-derived scene
gauge, finite arrays, and unchanged source pose rotations.

- [ ] **Step 2: Extract feature construction from `build_scene_windows`**

`build_refiner_features` must concatenate canonical global centers, canonical local
centers, their difference, global units, local units, unit difference, normalized
position, normalized boundary distance, local-alignment residual, and validity. GT
alignment returns only `target_residual`.

- [ ] **Step 3: Write the ScanNet exporter test**

Create a temporary legacy shard plus local diagnostics, invoke
`export_scannet_cache.main(...)`, and assert that the canonical result has source
`scannet`, nine 100-frame windows for 500 frames at stride 50, the frozen unit digest,
and a manifest checksum that reloads.

- [ ] **Step 4: Implement resumable ScanNet materialization**

The CLI accepts the existing `--dataset-manifest`, `--dataset-root`,
`--local-run-dir`, `--frozen-units`, `--out-dir`, `--window-length`, and `--stride`.
Skip a scene only when its shard checksum and `complete.json` invocation digest match.

- [ ] **Step 5: Run focused tests**

Run: `python -m unittest tests.camera_refiner_training.test_features tests.camera_refiner_training.test_export_scannet_cache tests.camera_refiner_training.test_data -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add pre_experiments/camera_refiner_training/features.py \
  pre_experiments/camera_refiner_training/export_scannet_cache.py \
  pre_experiments/camera_refiner_training/data.py \
  tests/camera_refiner_training/test_features.py \
  tests/camera_refiner_training/test_export_scannet_cache.py
git commit -m "Export canonical ScanNet refiner data"
```

### Task 4: Camera-Only VGGT CO3D Cache Builder

**Files:**
- Create: `pre_experiments/camera_refiner_training/vggt_camera.py`
- Create: `pre_experiments/camera_refiner_training/build_co3d_cache.py`
- Create: `tests/camera_refiner_training/test_vggt_camera.py`
- Create: `tests/camera_refiner_training/test_build_co3d_cache.py`

**Interfaces:**
- Produces: `CameraPrediction(c2w, selected_units)`, `load_camera_only_model(...)`, `predict_camera_clip(...)`, and `build_co3d_cache.main(...)`.
- Consumes: Task 1 clip manifest, frozen unit manifest, local VGGT checkpoint, and Task 2/3 shard APIs.

- [ ] **Step 1: Test selected-unit extraction without CUDA**

```python
def test_prediction_keeps_only_requested_iteration_zero_units(self):
    prediction = predict_camera_clip(
        model=FakeVGGT(), image_paths=self.images,
        unit_indices=(1, 3), device=torch.device("cpu"), image_loader=fake_loader,
    )
    self.assertEqual(prediction.selected_units.shape, (len(self.images), 2))
    self.assertEqual(prediction.c2w.shape, (len(self.images), 4, 4))
```

The fake model must assert `return_camera_trace=True` and
`camera_trace_pose_tokens=True`. The implementation reads
`camera_trace["pose_branch_hidden_list"][0]` and never serializes other units.

- [ ] **Step 2: Implement local-only checkpoint loading**

Support `model.safetensors` and `model.pt`, instantiate
`VGGT(enable_track=False, enable_point=False, enable_depth=False)`, load without a
network fallback, set evaluation mode, and convert pose encodings with
`pose_encoding_to_extri_intri` followed by homogeneous inversion.

- [ ] **Step 3: Test global/local assembly and resumability**

Use an injected fake predictor for a 50-frame clip. Assert one global call, canonical
25-frame local windows with half overlap, one valid canonical shard, and no calls on
a second invocation with matching `complete.json`. Change the unit digest and assert
the cache is recomputed.

- [ ] **Step 4: Implement the cache CLI**

For each clip, run full-context VGGT, run local windows of length
`ceil(context_length / 2)` and stride `floor(local_length / 2)`, assemble local poses
and units by maximum distance from a local boundary, then call
`build_refiner_features`. Write one shard per clip and an output manifest only after
all requested clips are complete. Catch CUDA OOM per clip, record failure, clear the
allocator, and continue so reruns can resume.

- [ ] **Step 5: Run focused tests**

Run: `python -m unittest tests.camera_refiner_training.test_vggt_camera tests.camera_refiner_training.test_build_co3d_cache -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add pre_experiments/camera_refiner_training/vggt_camera.py \
  pre_experiments/camera_refiner_training/build_co3d_cache.py \
  tests/camera_refiner_training/test_vggt_camera.py \
  tests/camera_refiner_training/test_build_co3d_cache.py
git commit -m "Build resumable CO3D VGGT caches"
```

### Task 5: Streaming Length-Bucketed Training Data

**Files:**
- Create: `pre_experiments/camera_refiner_training/training_data.py`
- Create: `tests/camera_refiner_training/test_training_data.py`
- Modify: `pre_experiments/camera_refiner_training/losses.py`
- Modify: `tests/camera_refiner_training/test_losses.py`
- Modify: `pre_experiments/camera_refiner_training/model.py`
- Modify: `tests/camera_refiner_training/test_model.py`

**Interfaces:**
- Produces: `CanonicalWindowDataset`, `GroupedLengthBatchSampler`, `collate_windows`, and `fit_source_condition_stats`.
- Batches contain same-length windows, `source`, `group_ids`, `starts`, frame IDs, conditions, targets, and global centers.

- [ ] **Step 1: Test lazy NPZ loading and homogeneous-length batches**

```python
def test_sampler_is_deterministic_and_never_mixes_lengths(self):
    sampler = GroupedLengthBatchSampler(index, batch_size=4, seed=33)
    first = list(sampler.batches(epoch=2))
    second = list(sampler.batches(epoch=2))
    self.assertEqual(first, second)
    for batch in first:
        self.assertEqual(len({index[item].window_length for item in batch}), 1)
```

Patch `np.load` and assert dataset construction opens no shard; `__getitem__` opens
only the requested shard.

- [ ] **Step 2: Implement flat window indexing and grouped buckets**

Expand manifest metadata into `(entry_index, window_index)` records without opening
NPZ files. Shuffle groups deterministically by `seed + epoch`, retain windows from a
group together when possible, and batch only equal lengths. This preserves overlap
loss opportunities without loading the dataset into memory.

- [ ] **Step 3: Add per-source statistics**

Stream training windows once and compute float64 running mean/variance keyed by
`source`; return float32 tensors. Validation and held-out roles must never contribute.
`collate_windows` selects the matching frozen source statistics.

- [ ] **Step 4: Verify variable-length model and losses**

Run the same `ResidualDiT(max_frames=100)` at lengths 50, 75, and 100. Update
`training_losses` to use only configured lags shorter than the current sequence and
to reject a sequence shorter than every lag. Keep the exact rotation invariant
outside the model unchanged.

- [ ] **Step 5: Run focused tests**

Run: `python -m unittest tests.camera_refiner_training.test_training_data tests.camera_refiner_training.test_model tests.camera_refiner_training.test_losses -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add pre_experiments/camera_refiner_training/training_data.py \
  pre_experiments/camera_refiner_training/losses.py \
  pre_experiments/camera_refiner_training/model.py \
  tests/camera_refiner_training/test_training_data.py \
  tests/camera_refiner_training/test_losses.py \
  tests/camera_refiner_training/test_model.py
git commit -m "Stream variable-length refiner batches"
```

### Task 6: Step-Based Pretraining, Fine-Tuning, and Checkpoint Lineage

**Files:**
- Modify: `pre_experiments/camera_refiner_training/checkpoint.py`
- Modify: `pre_experiments/camera_refiner_training/train.py`
- Modify: `tests/camera_refiner_training/test_train.py`
- Modify: `tests/camera_refiner_training/test_cli_smoke.py`

**Interfaces:**
- Adds CLI arguments `--stage`, `--max-steps`, `--batch-size`, `--validation-interval`, `--init-checkpoint`, and canonical `--dataset-manifest`/`--dataset-root`.
- `--resume` restores the exact run; `--init-checkpoint` loads model weights only and starts a new run with recorded parent digest.

- [ ] **Step 1: Write checkpoint-lineage tests**

```python
def test_finetune_initialization_loads_weights_but_not_optimizer_or_step(self):
    parent = save_test_checkpoint(global_step=12, source="co3d")
    state = initialize_from_checkpoint(parent, child_model, map_location="cpu")
    self.assertEqual(state.parent_digest, read_checkpoint_payload(parent)["run_digest"])
    self.assertEqual(state.global_step, 0)
```

Also test strict resume rejection after manifest, stage, model, unit, or source-stat
changes.

- [ ] **Step 2: Upgrade checkpoints to schema version 2**

Store `global_step`, `epoch`, `step_in_epoch`, model/optimizer state, source-keyed
condition statistics, run config and digest, parent digest, Python/NumPy/Torch RNG
states, and sampler seed. Read schema version 1 for existing inference, but new
training writes version 2 only.

- [ ] **Step 3: Replace eager epoch training with bounded steps**

Use `CanonicalWindowDataset` and `GroupedLengthBatchSampler`. Validate every
`--validation-interval` steps, atomically update `last.pt`, and update `best.pt` only
on lower validation loss. The primary stages are:

```text
co3d_pretrain    roles: pretrain -> validation
scannet_only     roles: train -> validation
scannet_finetune roles: train -> validation, requires --init-checkpoint
```

- [ ] **Step 4: Preserve a legacy ScanNet compatibility path**

When a schema-version-1 manifest and `--local-run-dir` are supplied, retain the
existing eager scene adapter. Emit a deprecation message and preserve existing CLI
smoke behavior. Canonical manifests never require `--local-run-dir` or
`--frozen-units` because their digests are authenticated in the manifest.

- [ ] **Step 5: Extend the CPU CLI smoke**

Train one deterministic CO3D step, initialize a ScanNet fine-tune run from its
checkpoint, resume the child for one additional step, and assert parent lineage,
source statistics, and monotonically increasing child global steps.

- [ ] **Step 6: Run focused tests**

Run: `python -m unittest tests.camera_refiner_training.test_train tests.camera_refiner_training.test_cli_smoke -v`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add pre_experiments/camera_refiner_training/checkpoint.py \
  pre_experiments/camera_refiner_training/train.py \
  tests/camera_refiner_training/test_train.py \
  tests/camera_refiner_training/test_cli_smoke.py
git commit -m "Add staged refiner pretraining"
```

### Task 7: Canonical Inference and Transfer Comparison

**Files:**
- Modify: `pre_experiments/camera_refiner_training/infer.py`
- Create: `pre_experiments/camera_refiner_training/compare_transfer.py`
- Modify: `tests/camera_refiner_training/test_infer.py`
- Create: `tests/camera_refiner_training/test_compare_transfer.py`
- Modify: `pre_experiments/camera_refiner_training/visualize.py`

**Interfaces:**
- Canonical inference groups shards by `group_id`, fuses overlapping corrections, and writes the existing per-scene camera/metric outputs.
- `compare_transfer` consumes ScanNet summary JSON/CSV from named runs and emits paired transfer metrics plus a machine-readable gate decision.

- [ ] **Step 1: Test canonical inference and rotation preservation**

Construct two overlapping canonical windows from one ScanNet scene, run a fake
refiner, and assert fused frame coverage, unchanged 3-by-3 rotations, and fallback to
the global center where confidence is zero.

- [ ] **Step 2: Implement canonical inference alongside legacy inference**

Select source-specific checkpoint statistics, reject an unknown source, group by
`group_id`, and retain current aligned prediction metrics. Write `source`, manifest
digest, checkpoint parent digest, and rotation maximum absolute change into every
summary row.

- [ ] **Step 3: Write paired-transfer gate tests**

```python
def test_transfer_gate_requires_mean_median_win_rate_lags_and_rotation(self):
    report = compare_runs(scannet_only, compute_matched, co3d_then_scannet)
    self.assertTrue(report["gate"]["passed"])
    self.assertGreaterEqual(report["paired"]["scene_win_rate"], 0.60)
```

Include failure cases for a 59-percent win rate, lag degradation, nonzero rotation
change, and missing scene identities.

- [ ] **Step 4: Implement scalar report and visualization**

Write `transfer_summary.json`, `paired_scenes.csv`, and a PNG containing paired
per-scene translation differences and aggregate confidence intervals. Labels must
distinguish `co3d_zero_shot`, `scannet_only`, `scannet_compute_matched`, and
`co3d_then_scannet`.

- [ ] **Step 5: Run focused tests**

Run: `python -m unittest tests.camera_refiner_training.test_infer tests.camera_refiner_training.test_compare_transfer tests.camera_refiner_training.test_visualize -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add pre_experiments/camera_refiner_training/infer.py \
  pre_experiments/camera_refiner_training/compare_transfer.py \
  pre_experiments/camera_refiner_training/visualize.py \
  tests/camera_refiner_training/test_infer.py \
  tests/camera_refiner_training/test_compare_transfer.py
git commit -m "Evaluate CO3D transfer on ScanNet"
```

### Task 8: AutoDL Entry Points and Reproduction Documentation

**Files:**
- Create: `scripts/autodl/camera_refiner_training/prepare_co3d.sh`
- Create: `scripts/autodl/camera_refiner_training/build_co3d_cache.sh`
- Create: `scripts/autodl/camera_refiner_training/export_scannet_cache.sh`
- Create: `scripts/autodl/camera_refiner_training/run_transfer_study.sh`
- Modify: `scripts/autodl/camera_refiner_training/train.sh`
- Modify: `scripts/autodl/camera_refiner_training/infer.sh`
- Modify: `tests/camera_refiner_training/test_autodl_scripts.py`
- Modify: `README.md`
- Modify: `pre_experiments/README.md`

**Interfaces:**
- Produces shell commands for dataset download, manifest/cache construction, CO3D pretraining, ScanNet export/fine-tuning, inference, and paired comparison.

- [ ] **Step 1: Write shell-contract tests**

Assert every script uses `set -euo pipefail`, activates `vggt`, sets a positive
integer `OMP_NUM_THREADS`, writes only below `/root/autodl-tmp/results`, never invokes
`conda create` or weight download, and accepts explicit path overrides.

- [ ] **Step 2: Implement official CO3D subset preparation**

`prepare_co3d.sh` accepts `CO3D_TOOLS_DIR`, `CO3D_ROOT`, `CO3D_REPO_URL`, and
`DOWNLOAD_MODE=single_sequence|categories`. It shallow-clones the official tooling
only when absent and invokes:

```bash
python "${CO3D_TOOLS_DIR}/co3d/download_dataset.py" \
  --download_folder "${CO3D_ROOT}" --single_sequence_subset
```

Category mode requires `CO3D_CATEGORIES` and passes it through
`--download_categories`. It does not install the CO3D package or PyTorch3D.

- [ ] **Step 3: Implement cache/export wrappers**

Defaults:

```text
CO3D_ROOT=/root/autodl-tmp/datasets/co3dv2
CKPT_DIR=/root/autodl-tmp/ckpt/VGGT-1B
CO3D_RESULTS=/root/autodl-tmp/results/camera_refiner_data_construction/co3d
SCANNET_RESULTS=/root/autodl-tmp/results/camera_refiner_data_construction/scannet
TRAIN_RESULTS=/root/autodl-tmp/results/camera_refiner_training
```

Every long-running command supports per-shard resume and prints its final manifest or
run directory.

- [ ] **Step 4: Implement the transfer-study wrapper**

Require frozen CO3D and ScanNet manifests. Run `co3d_pretrain`, `scannet_only`,
compute-matched `scannet_only`, and `scannet_finetune` as separate output
directories. `RESUME=1` resumes each independently. Do not start full-category data
preparation automatically.

- [ ] **Step 5: Document exact remote workflow**

README commands must cover official subset download, cache build, ScanNet canonical
export, one-step smoke, pilot pretraining, fine-tuning, inference, and transfer
comparison. State that 5.5 TB full CO3Dv2 is outside the pilot and requires explicit
storage provisioning.

- [ ] **Step 6: Validate shell and CPU tests**

Run:

```bash
bash -n scripts/autodl/camera_refiner_training/*.sh
python -m unittest tests.camera_refiner_training.test_autodl_scripts -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add scripts/autodl/camera_refiner_training README.md pre_experiments/README.md \
  tests/camera_refiner_training/test_autodl_scripts.py
git commit -m "Add AutoDL CO3D transfer workflow"
```

### Task 9: End-to-End Verification

**Files:**
- Modify only files required by failures found in this task.

**Interfaces:**
- Produces a clean, committed branch with CPU evidence for all data contracts and one end-to-end staged train/infer smoke.

- [ ] **Step 1: Run syntax checks**

Run:

```bash
python -m compileall -q pre_experiments
bash -n scripts/autodl/camera_refiner_training/*.sh
```

Expected: both exit zero.

- [ ] **Step 2: Run the complete retained test suite**

Run: `python -m unittest discover -s tests -v`

Expected: all tests pass without CUDA, network, checkpoint, or dataset access.

- [ ] **Step 3: Check repository hygiene**

Run:

```bash
git diff --check
git status --short
git ls-files | grep -E '\.(npz|pt|pth|ply|jpg|png)$'
```

Expected: no uncommitted implementation changes and no new generated dataset,
checkpoint, image, point-cloud, or raw cache files. Existing committed paper PDFs are
outside this filter.

- [ ] **Step 4: Record implementation completion**

Update this plan's completed checkboxes, then commit only that bookkeeping change:

```bash
git add doc/2026-08-07_CO3D_Pretraining_Implementation_Plan.md
git commit -m "Record CO3D workflow verification"
```
