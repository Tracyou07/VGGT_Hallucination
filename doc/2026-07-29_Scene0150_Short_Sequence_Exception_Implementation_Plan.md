# Scene0150 Short-Sequence Exception Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:executing-plans` to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Admit the real 430-frame `scene0150_00` source while keeping every
other ScanNet-50 scene strictly 500-frame and preserving complete formal
window validation.

**Architecture:** Centralize the one-scene frame-count policy in
`context_source.py` and reuse it during split construction and full source
preflight. Let row collection derive expected window counts from each validated
source artifact, then require formal analysis to match that count exactly.

**Tech Stack:** Python 3.10+, NumPy, `unittest`, existing ScanNet-50 CLIs.

## Global Constraints

- Only `scene0150_00` may contain 430 selected frames.
- Every other scene must contain 500 selected frames.
- Window length remains 100; stride remains 50; the final tail window is kept.
- Prediction metrics use aligned predictions; GT always remains raw.
- No Camera Context NPZ file is committed.

---

### Task 1: Centralize the Source Frame-Count Policy

**Files:**
- Modify: `pre_experiments/local_global_consistency/context_source.py`
- Modify: `pre_experiments/local_global_consistency/split.py`
- Test: `tests/local_global_consistency/test_context_source.py`
- Test: `tests/local_global_consistency/test_split.py`

**Interfaces:**
- Produces: `expected_context_frame_count(scene: str) -> int`
- Consumes: ordered valid ScanNet frame IDs and source `frame_ids`

- [ ] **Step 1: Write failing source-contract tests**

Add cases proving:

```python
self.assertEqual(expected_context_frame_count("scene0150_00"), 430)
self.assertEqual(expected_context_frame_count("scene0000_00"), 500)
```

Build a 430-frame `scene0150_00` fixture and require
`validate_context_source()` to accept it. Add 429- and 431-frame variants and a
430-frame non-exception scene; each must raise `ValueError`.

- [ ] **Step 2: Run the focused tests and observe the expected failure**

```bash
python -m unittest \
  tests.local_global_consistency.test_context_source \
  tests.local_global_consistency.test_split -v
```

Expected: failure because `expected_context_frame_count` is absent and both
source paths still request 500 IDs for every scene.

- [ ] **Step 3: Implement the exact exception**

Add:

```python
DEFAULT_CONTEXT_FRAME_COUNT = 500
CONTEXT_FRAME_COUNT_EXCEPTIONS = {"scene0150_00": 430}


def expected_context_frame_count(scene: str) -> int:
    return CONTEXT_FRAME_COUNT_EXCEPTIONS.get(
        scene, DEFAULT_CONTEXT_FRAME_COUNT
    )
```

In both `validate_context_source()` and `split._build_from_paths()`, select
`expected_context_frame_count(scene)` IDs with `uniform_frame_ids()`, compare
the artifact IDs exactly, and reject any mismatched length or ordering.

- [ ] **Step 4: Run focused tests**

Run the command from Step 2. Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add pre_experiments/local_global_consistency/context_source.py \
  pre_experiments/local_global_consistency/split.py \
  tests/local_global_consistency/test_context_source.py \
  tests/local_global_consistency/test_split.py
git commit -m "Allow exact scene0150 context length"
```

### Task 2: Derive Formal Window Counts

**Files:**
- Modify: `pre_experiments/local_global_consistency/analyze.py`
- Test: `tests/local_global_consistency/test_local_global_consistency.py`

**Interfaces:**
- Produces: `collect_run_rows(...)[expected_window_count]`
- Consumes: exact windows returned by `build_sliding_windows()`

- [ ] **Step 1: Write failing analyzer tests**

Add tests where calibration contains the exception and supplies:

```python
collected = {
    "scores": score_rows,
    "validation": validation_rows,
    "window_count": 89,
    "expected_window_count": 89,
}
```

Require calibration analysis to pass. Change `window_count` to 90 while keeping
`expected_window_count` at 89 and require a `ValueError`. Mirror this for
holdout counts 359 and 360.

- [ ] **Step 2: Run focused analyzer tests and observe failure**

```bash
python -m unittest \
  tests.local_global_consistency.test_local_global_consistency -v
```

Expected: the 89/359 valid cases fail because analysis still hard-codes 90/360.

- [ ] **Step 3: Implement derived count validation**

During `collect_run_rows()`, sum `len(expected_windows)` for every scene and
return it as `expected_window_count`. Replace the two hard-coded checks with a
shared exact comparison:

```python
if collected.get("window_count") != collected.get("expected_window_count"):
    raise ValueError("formal analysis requires the complete expected window set")
```

Record both actual and expected counts in completion metadata. Scene partition
sizes remain strictly 10 and 40.

- [ ] **Step 4: Run focused analyzer tests**

Run the command from Step 2. Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add pre_experiments/local_global_consistency/analyze.py \
  tests/local_global_consistency/test_local_global_consistency.py
git commit -m "Derive formal local window counts"
```

### Task 3: Update Protocol Documentation and Validate Real Data

**Files:**
- Modify: `README.md`
- Modify: `pre_experiments/local_global_consistency/README.md`
- Modify: `doc/2026-07-28_ScanNet50_Local_Global_Validation_Design.md`
- Modify:
  `doc/2026-07-28_ScanNet50_Local_Global_Validation_Implementation_Plan.md`

**Interfaces:**
- Consumes: source run `d33d98b_309a9a586242`
- Produces: documented 449-window formal protocol

- [ ] **Step 1: Update active protocol text**

Document 49×500 plus `scene0150_00`×430, eight tail-covering windows for the
exception, partition-dependent 89/90 and 359/360 counts, and 449 total windows.
Remove unconditional claims of 90 calibration plus 360 holdout windows.

- [ ] **Step 2: Run the complete CPU suite**

```bash
python -m unittest discover -s tests/local_global_consistency -v
python -m compileall -q pre_experiments/local_global_consistency
bash -n scripts/autodl/run_scannet50_local_global.sh
```

Expected: all commands exit zero.

- [ ] **Step 3: Validate the real source structure**

Run a dependency-free scan over
`results/camera_context/d33d98b_309a9a586242` and require:

```text
49 scenes with 500 frames
scene0150_00 with 430 frames
50 identical NPZ schemas
no non-finite values
```

Full raw-GT/frame-ID validation remains part of split construction on AutoDL,
where processed ScanNet is available.

- [ ] **Step 4: Commit**

```bash
git add README.md pre_experiments/local_global_consistency/README.md \
  doc/2026-07-28_ScanNet50_Local_Global_Validation_Design.md \
  doc/2026-07-28_ScanNet50_Local_Global_Validation_Implementation_Plan.md
git commit -m "Document scene0150 formal exception"
```

### Task 4: Generate and Freeze the Split on AutoDL

**Files:**
- Create on AutoDL, then commit:
  `configs/scannet50_local_global_split.json`

**Interfaces:**
- Consumes: processed ScanNet-50 and source run `d33d98b_309a9a586242`
- Produces: authenticated 10/40 split manifest

- [ ] **Step 1: Generate the split**

```bash
export SOURCE_RUN_DIR=/root/autodl-tmp/camera_context/results/d33d98b_309a9a586242
python -m pre_experiments.local_global_consistency.split \
  --data-dir /root/autodl-tmp/datasets/scannetv2/process_scannet \
  --scene-list configs/fastvggt_scannet50.txt \
  --source-run-dir "$SOURCE_RUN_DIR" \
  --output configs/scannet50_local_global_split.json \
  --seed 33
```

- [ ] **Step 2: Validate and commit the frozen split**

```bash
python -m pre_experiments.local_global_consistency.split \
  --validate configs/scannet50_local_global_split.json \
  --scene-list configs/fastvggt_scannet50.txt
git add configs/scannet50_local_global_split.json
git commit -m "Freeze ScanNet-50 calibration holdout split"
git push origin local-global-consistency-preexperiment
```

Expected: 10 calibration scenes, 40 holdout scenes, no overlap, and exact raw
GT/frame-ID validation including the 430-frame exception.
