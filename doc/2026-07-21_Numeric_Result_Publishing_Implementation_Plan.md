# Camera Iteration Numeric Result Publishing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Export only compact reproducible Camera Iteration numeric artifacts from AutoDL into a Git-trackable run directory.

**Architecture:** A pure Python exporter validates run metadata, completion markers, file sizes, and NPZ member names before copying a fixed whitelist into an immutable destination. Git ignore rules expose only JSON, CSV, and NPZ files in the dedicated published-results namespace; documentation keeps commit and push explicit.

**Tech Stack:** Python 3.10+ standard library, Bash usage examples, `unittest`, Git.

## Global Constraints

- Source runs remain under external AutoDL storage and are never modified.
- Publish only run/selection JSON, CSV, and compact pose-trace NPZ files.
- Reject Camera Token arrays and any allowed file larger than 50 MiB by default.
- Never copy images, point clouds, depth maps, checkpoints, or datasets.
- Never run `git commit` or `git push` from the exporter.
- Tests require no Torch, CUDA, network, checkpoint, NumPy, or ScanNet.

---

### Task 1: Exporter Contracts and Implementation

**Files:**
- Create: `scripts/autodl/camera_iteration/export_numeric_results.py`
- Create: `tests/camera_iteration/test_export_numeric_results.py`

**Interfaces:**
- Produces: `export_numeric_results(source: Path, destination_root: Path, max_file_bytes: int) -> Path`
- CLI: `--source`, `--destination-root`, `--max-file-mb`
- Output: `results/camera_iteration/<run_id>/publish_manifest.json`

- [ ] **Step 1: Write failing temporary-tree tests**

Create fixtures containing valid root summaries, one complete selection, a
small zip-compatible NPZ, and forbidden `.ply`/`.jpg` files. Assert that the
whitelist is copied, forbidden files are absent, the manifest hashes every
copy, and an existing destination is rejected. Add rejection tests for an
oversized allowed file, mismatched completion run ID, and NPZ members named
`normalized_camera_tokens.npy` or `pose_tokens_modulated.npy`.

- [ ] **Step 2: Verify tests fail**

```powershell
python -m unittest tests.camera_iteration.test_export_numeric_results -v
```

Expected: import failure because the exporter does not exist.

- [ ] **Step 3: Implement validation and atomic export**

Use exact root and selection filename sets, `json`, `zipfile`, `hashlib`,
`tempfile`, and `shutil`. Validate `run_metadata.json` contains a non-empty
`run_id`; validate every copied selection has all five required files and a
matching `complete.json`. Copy into a temporary sibling directory, write a
sorted manifest, then rename to `<destination-root>/<run_id>`.

- [ ] **Step 4: Verify focused tests**

```powershell
python -m unittest tests.camera_iteration.test_export_numeric_results -v
python -m py_compile scripts/autodl/camera_iteration/export_numeric_results.py
```

Expected: all exporter tests pass and compilation exits zero.

### Task 2: Git Publication Boundary and Documentation

**Files:**
- Modify: `.gitignore`
- Modify: `README.md`
- Modify: `pre_experiments/camera_iteration/README.md`
- Modify: `AGENTS.md`
- Modify: `tests/camera_iteration/test_contracts.py`
- Create: `log/2026-07-21_numeric_result_publishing.md`

**Interfaces:**
- Git namespace: `results/camera_iteration/<run_id>/`
- AutoDL command: `python scripts/autodl/camera_iteration/export_numeric_results.py --source "$RESULT_DIR/<run_id>"`

- [ ] **Step 1: Write failing ignore-boundary test**

Use `git check-ignore` to assert ordinary `/results` artifacts and `.ply`
files remain ignored while JSON, CSV, and NPZ under the publication namespace
are trackable.

- [ ] **Step 2: Verify boundary test fails**

```powershell
python -m unittest tests.camera_iteration.test_contracts -v
```

Expected: published numeric paths are still ignored by `/results/`.

- [ ] **Step 3: Add allowlist and instructions**

Keep `/results/*` ignored, re-include `results/camera_iteration/` directories,
then allow only `*.json`, `*.csv`, and `*.npz`. Document export, size review,
explicit `git add`, commit, and push. State that high-dimensional token traces,
point clouds, images, weights, and datasets remain prohibited.

- [ ] **Step 4: Run complete available verification**

```powershell
python -m unittest tests.camera_iteration.test_export_numeric_results tests.camera_iteration.test_contracts tests.camera_iteration.test_autodl_scripts -v
python -m py_compile scripts/autodl/camera_iteration/export_numeric_results.py
git diff --check
```

Expected: zero failures and zero diff errors. Record that checkpoint-backed
execution is not rerun because publishing does not alter inference.

- [ ] **Step 5: Commit and push**

```powershell
git add .gitignore AGENTS.md README.md pre_experiments/camera_iteration/README.md scripts/autodl/camera_iteration/export_numeric_results.py tests/camera_iteration log doc
git commit -m "Add numeric result publishing workflow"
git push origin camera-iteration-preexperiment
```
