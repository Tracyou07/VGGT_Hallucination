# ScanNet-50 Local-Global Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:executing-plans` to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking. Do not dispatch subagents.

**Goal:** Build a leakage-controlled ScanNet-50 workflow that fits reliability
thresholds on a difficulty-stratified 10-scene calibration set and evaluates
them unchanged on a 40-scene holdout set.

**Architecture:** The Camera Context branch supplies one explicit 50-scene
global source run. This worktree constructs and freezes a raw-GT-only split,
runs the existing 100/50 local-window protocol in separate calibration and
holdout runs, writes a provenance-rich threshold artifact, and produces
holdout-only aggregate statistics.

**Tech Stack:** Python 3.10+, NumPy, PyTorch, standard-library `unittest`,
POSIX Bash, JSON/CSV, existing VGGT and ScanNet helpers.

## Global Constraints

- All implementation belongs to `local-global-consistency-preexperiment`.
- Prediction-only scores never use GT.
- Prediction-versus-GT metrics use aligned predictions against raw GT.
- GT is never aligned, overwritten, or ambiguously named.
- The split builder must not load VGGT prediction arrays.
- Calibration and holdout have separate run IDs and output roots.
- Holdout analysis must consume an explicit frozen threshold artifact and must
  not fit thresholds.
- Raw window NPZ files remain outside Git.
- Existing four-scene results and frozen source artifacts remain unchanged.

---

## File Map

- `pre_experiments/local_global_consistency/split.py`: raw-motion difficulty,
  deterministic 10/40 split construction, manifest validation, and CLI.
- `pre_experiments/local_global_consistency/context_source.py`: exact
  prediction-free frame-ID loading and exact 50-scene global-source contract.
- `pre_experiments/local_global_consistency/thresholds.py`: frozen threshold
  schema, fitting, hashing, loading, and validation.
- `pre_experiments/local_global_consistency/aggregate.py`: holdout-only
  scene-level aggregation and deterministic bootstrap confidence intervals.
- `pre_experiments/local_global_consistency/run_study.py`: partition-aware
  local inference with preflight validation.
- `pre_experiments/local_global_consistency/analyze.py`: explicit calibration
  and holdout analysis modes.
- `scripts/autodl/run_scannet50_local_global.sh`: sequential AutoDL entry point.
- `scripts/autodl/local_global_consistency/export_numeric_results.py`: strict
  publication of split, threshold, CSV, JSON, and manifest outputs.
- `configs/scannet50_local_global_split.json`: generated and frozen split.
- `tests/local_global_consistency/`: focused CPU tests for every contract.

---

### Task 1: Raw-Motion Stratified Split

**Files:**
- Create: `pre_experiments/local_global_consistency/split.py`
- Create: `pre_experiments/local_global_consistency/context_source.py`
- Create: `tests/local_global_consistency/test_split.py`
- Create: `tests/local_global_consistency/test_context_source.py`
- Modify: `pre_experiments/common/scannet.py`
- Modify: `tests/local_global_consistency/test_scannet.py`

**Interfaces:**
- Produces:
  - `uniform_frame_ids(valid_ids: list[int], count: int) -> list[int]`
  - `load_context_frame_ids(path: Path) -> np.ndarray`
  - `motion_features(raw_c2w: np.ndarray) -> dict[str, float]`
  - `build_split_manifest(...) -> dict[str, object]`
  - `load_split_manifest(path: Path, expected_scenes: list[str]) -> dict`
- Consumes only processed image IDs, raw GT poses, scene IDs, and seed 33.

- [ ] **Step 1: Add failing uniform-selection tests**

Test that 500 IDs reproduce NumPy integer `linspace` selection, insufficient
scenes fail, and no duplicate IDs are accepted:

```python
def test_uniform_frame_ids_matches_context_protocol(self):
    valid = list(range(0, 2000, 2))
    selected = uniform_frame_ids(valid, 500)
    expected = [valid[i] for i in np.linspace(0, 999, 500, dtype=np.int64)]
    self.assertEqual(selected, expected)
```

- [ ] **Step 2: Run the test and verify the expected import failure**

```bash
python -m unittest tests.local_global_consistency.test_scannet -v
```

Expected: fail because `uniform_frame_ids` is absent.

- [ ] **Step 3: Implement deterministic frame selection**

Add strict positive-count, ordered-unique-ID, and sufficient-length checks.
Return exactly `count` IDs selected with integer `np.linspace`.

- [ ] **Step 4: Add failing motion and split tests**

Construct synthetic raw `c2w` trajectories with increasing translation,
rotation, and step discontinuities. Assert:

```python
self.assertEqual(len(manifest["calibration_scenes"]), 10)
self.assertEqual(len(manifest["holdout_scenes"]), 40)
self.assertEqual(
    set(manifest["calibration_scenes"]) | set(manifest["holdout_scenes"]),
    set(scannet50),
)
self.assertEqual(
    Counter(manifest["new_calibration_strata"].values()),
    {"easy": 2, "medium": 2, "hard": 2},
)
```

Patch any prediction loader to raise immediately and prove split construction
still succeeds.

- [ ] **Step 5: Implement split construction and CLI**

For every candidate, compute cumulative translation, cumulative geodesic
rotation, P95 translation step, and P95 rotation step from raw `c2w`. Convert
each feature to an average-tie percentile rank, average the ranks, sort by
`(difficulty_score, scene)`, and divide the 46 candidates with
`np.array_split(..., 3)` into easy/medium/hard. Select two per stratum by
ascending SHA-256 of `f"33:{scene}"`.

The CLI must require:

```text
--data-dir
--scene-list
--source-run-dir
--output
--seed 33
```

It writes atomically and records feature values, percentile ranks, strata,
selection hashes, fixed observed scenes, source run ID, and a canonical
`split_digest`.

Add `--validate MANIFEST --scene-list SCENE_LIST` as a mutually exclusive CLI
mode that calls `load_split_manifest` and exits nonzero on any contract
violation.

- [ ] **Step 6: Run focused tests**

```bash
python -m unittest \
  tests.local_global_consistency.test_context_source \
  tests.local_global_consistency.test_scannet \
  tests.local_global_consistency.test_split -v
```

Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add pre_experiments/common/scannet.py \
  pre_experiments/local_global_consistency/context_source.py \
  pre_experiments/local_global_consistency/split.py \
  tests/local_global_consistency/test_context_source.py \
  tests/local_global_consistency/test_scannet.py \
  tests/local_global_consistency/test_split.py
git commit -m "Add stratified ScanNet-50 split builder"
```

---

### Task 2: Exact 50-Scene Context Source Contract

**Files:**
- Modify: `pre_experiments/local_global_consistency/context_source.py`
- Modify: `tests/local_global_consistency/test_context_source.py`
- Modify: `pre_experiments/local_global_consistency/run_study.py`
- Modify: `tests/local_global_consistency/test_local_global_consistency.py`

**Interfaces:**
- Consumes `load_split_manifest`.
- Produces:
  - `load_context_frame_ids(path: Path) -> np.ndarray`
  - `validate_context_source(source: Path, split: dict, data_dir: Path) -> dict`
  - runner flags `--split-manifest` and
    `--partition {calibration,holdout}`.

- [ ] **Step 1: Add failing source-contract tests**

Use temporary NPZ fixtures and metadata. Require exactly 50 metadata scenes,
`frame_counts == [500]`, `iterations == [4]`, `sampling == "nested_uniform"`,
`preprocess_mode == "pad"`, and one exact diagnostics artifact per scene.
Reject missing scenes, extra scenes, wrong protocols, wrong frame IDs, and raw
GT mismatches.

Instrument NPZ access so `load_context_frame_ids` fails if it reads
`normalized_camera_tokens`, `pred_c2w_raw`, or any other prediction member.

- [ ] **Step 2: Verify tests fail**

```bash
python -m unittest tests.local_global_consistency.test_context_source -v
```

Expected: fail because `context_source.py` does not exist.

- [ ] **Step 3: Implement source validation**

Load only `frame_ids` during split preflight. Recompute the deterministic 500
IDs from processed image/raw-pose intersections and require exact equality.
During local inference, continue using the strict full global artifact loader
and require its `gt_c2w_raw` to equal processed raw GT.

- [ ] **Step 4: Make the local runner partition-aware**

Replace metadata-driven scene slicing with the explicit split manifest and
partition. Include `split_digest`, partition, ordered partition scenes, and
source run ID in the run invocation and run ID. Before loading the model,
validate every selected scene and artifact so an incomplete 50-scene source
fails before GPU work.

Retain `--scene-limit` only as an explicitly marked smoke-test option. A run
with a nonzero limit records `protocol_complete=false` and cannot be analyzed
or exported as formal calibration/holdout evidence.

- [ ] **Step 5: Add resume and run-ID tests**

Prove calibration and holdout produce different IDs, split changes alter IDs,
source changes alter IDs, and an existing window is skipped only when its
partition, split digest, boundaries, and frame IDs match.

- [ ] **Step 6: Run focused tests**

```bash
python -m unittest \
  tests.local_global_consistency.test_context_source \
  tests.local_global_consistency.test_local_global_consistency -v
```

Expected: all pass in the full `vggt` environment.

- [ ] **Step 7: Commit**

```bash
git add pre_experiments/local_global_consistency/context_source.py \
  pre_experiments/local_global_consistency/run_study.py \
  tests/local_global_consistency/test_context_source.py \
  tests/local_global_consistency/test_local_global_consistency.py
git commit -m "Validate partitioned ScanNet-50 context inputs"
```

---

### Task 3: Frozen Calibration Threshold Artifact

**Files:**
- Create: `pre_experiments/local_global_consistency/thresholds.py`
- Create: `tests/local_global_consistency/test_thresholds.py`
- Modify: `pre_experiments/local_global_consistency/analyze.py`
- Modify: `pre_experiments/local_global_consistency/metrics.py`
- Modify: `tests/local_global_consistency/test_local_global_consistency.py`

**Interfaces:**
- Produces:
  - `fit_frozen_thresholds(score_rows, provenance) -> dict[str, object]`
  - `load_frozen_thresholds(path, expected_split_digest,
    expected_source_run_id) -> dict`
  - analyzer flag `--mode calibration`.

- [ ] **Step 1: Add failing threshold-schema tests**

Build score rows for ten scenes and assert P95 values use only non-null
local-local prediction fields. Require sample counts, all ten scene IDs,
source/calibration run IDs, split digest, code commit, and a SHA-256
`threshold_digest`.

Reject a calibration input containing a holdout scene or a GT-named field in
the threshold fitting mapping.

- [ ] **Step 2: Verify tests fail**

```bash
python -m unittest tests.local_global_consistency.test_thresholds -v
```

Expected: fail because the threshold module is absent.

- [ ] **Step 3: Implement calibration-only fitting**

Move threshold fitting out of the implicit stable-scene path. Calibration mode
must require all ten calibration scenes and a complete 90-window run. Write:

```text
frozen_reliability_thresholds.json
calibration_prediction_scores_per_frame.csv
calibration_gt_validation_per_frame.csv
calibration_summary.csv
calibration_summary.json
```

Canonicalize the threshold payload without `threshold_digest`, hash it, append
the digest, and write atomically.

- [ ] **Step 4: Remove implicit threshold fitting**

Delete `DEFAULT_STABLE_SCENES` and any analyzer behavior that silently fits from
available scenes. Calling calibration mode on holdout metadata or holdout mode
without an external threshold file must raise `ValueError`.

- [ ] **Step 5: Run focused tests**

```bash
python -m unittest \
  tests.local_global_consistency.test_thresholds \
  tests.local_global_consistency.test_local_global_consistency -v
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add pre_experiments/local_global_consistency/thresholds.py \
  pre_experiments/local_global_consistency/analyze.py \
  pre_experiments/local_global_consistency/metrics.py \
  tests/local_global_consistency/test_thresholds.py \
  tests/local_global_consistency/test_local_global_consistency.py
git commit -m "Freeze calibration reliability thresholds"
```

---

### Task 4: Holdout-Only Evaluation and Bootstrap

**Files:**
- Create: `pre_experiments/local_global_consistency/aggregate.py`
- Create: `tests/local_global_consistency/test_aggregate.py`
- Modify: `pre_experiments/local_global_consistency/analyze.py`
- Modify: `tests/local_global_consistency/test_local_global_consistency.py`

**Interfaces:**
- Consumes `load_frozen_thresholds`.
- Produces:
  - `bootstrap_holdout(scene_rows, *, samples=10000, seed=33) -> list[dict]`
  - analyzer flags `--mode holdout` and `--thresholds PATH`.

- [ ] **Step 1: Add failing holdout immutability tests**

Pass a frozen threshold fixture into holdout analysis, patch every fitting
function to raise, and prove analysis still succeeds without modifying the
threshold file. Reject threshold digests, split digests, source IDs, or scene
sets that do not match holdout metadata.

- [ ] **Step 2: Add failing aggregate-statistics tests**

Use four synthetic scenes with known global/local translation and rotation
errors. Assert aggregate output includes mean, median, positive-growth
fraction, frozen reliability coverage, Pearson/Spearman values, and finite
`ci95_low <= estimate <= ci95_high`.

Run bootstrap twice and require byte-identical JSON with seed 33.

- [ ] **Step 3: Verify tests fail**

```bash
python -m unittest tests.local_global_consistency.test_aggregate -v
```

Expected: fail because `aggregate.py` does not exist.

- [ ] **Step 4: Implement scene-level bootstrap**

First reduce frame rows to one summary per scene. For each of 10,000 iterations,
sample 40 scene summaries with replacement and recompute the aggregate metric.
Use NumPy `default_rng(33)` and percentile bounds 2.5/97.5. Do not bootstrap
frames independently because frames within one scene are correlated.

- [ ] **Step 5: Implement strict holdout analysis**

Require exactly 40 holdout scenes and 360 complete windows. Apply the frozen
threshold values unchanged. Write:

```text
holdout_prediction_scores_per_frame.csv
holdout_gt_validation_per_frame.csv
holdout_per_scene_summary.csv
holdout_aggregate_summary.csv
holdout_aggregate_summary.json
holdout_complete.json
```

Record threshold path and digest in every holdout completion/summary manifest.

- [ ] **Step 6: Run focused tests**

```bash
python -m unittest \
  tests.local_global_consistency.test_aggregate \
  tests.local_global_consistency.test_thresholds \
  tests.local_global_consistency.test_local_global_consistency -v
```

Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add pre_experiments/local_global_consistency/aggregate.py \
  pre_experiments/local_global_consistency/analyze.py \
  tests/local_global_consistency/test_aggregate.py \
  tests/local_global_consistency/test_local_global_consistency.py
git commit -m "Evaluate frozen thresholds on holdout scenes"
```

---

### Task 5: AutoDL Orchestration and Numeric Export

**Files:**
- Create: `scripts/autodl/run_scannet50_local_global.sh`
- Modify: `scripts/autodl/local_global_consistency/export_numeric_results.py`
- Modify: `tests/local_global_consistency/test_autodl_scripts.py`
- Modify: `tests/local_global_consistency/test_export_local_global_consistency.py`
- Modify: `README.md`
- Modify: `AGENTS.md`
- Modify: `pre_experiments/local_global_consistency/README.md`

**Interfaces:**
- Requires explicit environment variable `SOURCE_RUN_DIR`.
- Produces calibration and holdout run-directory pointer files plus one frozen
  threshold path.

- [ ] **Step 1: Add failing shell-contract tests**

Require the wrapper to:

- activate the existing `vggt` environment;
- reject an unset/nonexistent `SOURCE_RUN_DIR`;
- require `configs/scannet50_local_global_split.json`;
- run calibration inference and analysis before holdout;
- pass the exact frozen threshold path into holdout analysis;
- never invoke context inference, environment creation, weight download, or
  automatic newest-directory discovery.

- [ ] **Step 2: Verify shell tests fail**

```bash
python -m unittest \
  tests.local_global_consistency.test_autodl_scripts.AutoDLScriptsTest -v
```

Expected: fail because the wrapper is absent.

- [ ] **Step 3: Implement the sequential wrapper**

Support `STAGE=calibration`, `STAGE=holdout`, and `STAGE=all`. `holdout` requires
`CALIBRATION_RUN_DIR/frozen_reliability_thresholds.json`. Formal mode fixes
window length 100, stride 50, Camera iterations 4, preprocessing `pad`, and
device `cuda`.

Write logs and pointer files under:

```text
/root/autodl-tmp/local_global_consistency/scannet50/
```

Raw runs remain under partition-specific subdirectories.

- [ ] **Step 4: Add failing exporter tests**

Allow only the frozen split manifest, frozen thresholds, CSV/JSON summaries,
run metadata, completion files, and numeric window manifests. Reject every NPZ,
image, PLY, checkpoint, path traversal, incomplete run, or holdout result whose
threshold digest is missing.

- [ ] **Step 5: Implement exporter and documentation**

Export calibration and holdout into separate repository subdirectories under:

```text
  results/local_global_consistency/scannet50/
  {calibration_run_id}__{holdout_run_id}/
```

Document source generation, split generation, one-scene smoke commands, formal
calibration, formal holdout, resume, export, and the aligned-prediction/raw-GT
rule.

- [ ] **Step 6: Run shell and exporter tests**

```bash
python -m unittest \
  tests.local_global_consistency.test_autodl_scripts \
  tests.local_global_consistency.test_export_local_global_consistency -v
bash -n scripts/autodl/run_scannet50_local_global.sh
```

Expected: all pass with no shell syntax errors.

- [ ] **Step 7: Commit**

```bash
git add scripts/autodl/run_scannet50_local_global.sh \
  scripts/autodl/local_global_consistency/export_numeric_results.py \
  tests/local_global_consistency/test_autodl_scripts.py \
  tests/local_global_consistency/test_export_local_global_consistency.py \
  README.md AGENTS.md pre_experiments/local_global_consistency/README.md
git commit -m "Automate ScanNet-50 local-global validation"
```

---

### Task 6: Generate the Frozen Split and Verify End to End

**Files:**
- Create: `configs/scannet50_local_global_split.json`
- Modify only if verification exposes a defect in files owned by Tasks 1-5.

**Interfaces:**
- Consumes the completed 50-scene context run and processed ScanNet-50 data.
- Produces the committed split manifest and verified AutoDL commands.

- [ ] **Step 1: Generate the split on AutoDL before local inference**

```bash
conda activate vggt
git switch local-global-consistency-preexperiment
git pull --ff-only origin local-global-consistency-preexperiment

SOURCE_RUN_DIR="$(
  sed -n 's/^\[done\] results=//p' \
    /root/autodl-tmp/camera_context/scannet50_context.log | tail -n 1
)"
[[ -n "$SOURCE_RUN_DIR" && -d "$SOURCE_RUN_DIR" ]]
export SOURCE_RUN_DIR

python -m pre_experiments.local_global_consistency.split \
  --data-dir /root/autodl-tmp/datasets/scannetv2/process_scannet \
  --scene-list configs/fastvggt_scannet50.txt \
  --source-run-dir "$SOURCE_RUN_DIR" \
  --output configs/scannet50_local_global_split.json \
  --seed 33
```

- [ ] **Step 2: Inspect and freeze the manifest**

Verify counts and strata through the module's validation CLI, then commit and
push only the JSON manifest:

```bash
python -m pre_experiments.local_global_consistency.split \
  --validate configs/scannet50_local_global_split.json \
  --scene-list configs/fastvggt_scannet50.txt
git add configs/scannet50_local_global_split.json
git commit -m "Freeze ScanNet-50 calibration holdout split"
git push origin local-global-consistency-preexperiment
```

- [ ] **Step 3: Run the complete CPU suite locally or on AutoDL**

```bash
python -m unittest discover -s tests/local_global_consistency -v
python -m compileall -q \
  pre_experiments/local_global_consistency \
  pre_experiments/common \
  scripts/autodl/scannet \
  tests/local_global_consistency
git diff --check
```

Expected: zero failures and zero whitespace errors.

- [ ] **Step 4: Run one-scene GPU smoke tests**

Run calibration and holdout inference with `--scene-limit 1`. Confirm each
produces nine windows, approximately 15 GB peak allocated memory, no analysis
completion marker, and `protocol_complete=false`.

- [ ] **Step 5: Run formal calibration then holdout**

```bash
STAGE=all \
bash scripts/autodl/run_scannet50_local_global.sh
```

Expected: 90 calibration windows, one frozen threshold artifact, 360 holdout
windows, and holdout aggregate outputs referencing the frozen threshold digest.

- [ ] **Step 6: Export numeric evidence and inspect repository policy**

Run the strict exporter against both completed run directories. Confirm:

```bash
git status --short
git check-ignore -v \
  results/local_global_consistency/scannet50/example/window_diagnostics.npz
```

Only expected split/CSV/JSON files may be staged.

- [ ] **Step 7: Verify repository status**

Run `git status --short --branch`. The implementation and split commits must be
present, raw NPZ files must remain ignored, and only intentionally exported
numeric evidence may remain uncommitted for result review.
