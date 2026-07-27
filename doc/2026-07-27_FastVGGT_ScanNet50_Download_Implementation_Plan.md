# FastVGGT ScanNet-50 Download Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` (recommended) or
> `superpowers:executing-plans` to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prepare FastVGGT's exact ScanNet-50 scenes through the repository's
existing official ScanNet download and extraction workflow.

**Architecture:** A committed text file defines the dataset identity. The
existing AutoDL Bash entry point reads any scene list, validates it, downloads
`.sens` and optional GT PLY assets with bounded retries, and delegates frame
extraction to the existing Python utility.

**Tech Stack:** Bash, Python 3.10+, `unittest`, official ScanNet downloader.

## Global Constraints

- Require `SCANNET_TOS_ACCEPTED=1`; never bypass ScanNet authorization.
- Supply confirmation input automatically after the explicit ToS guard.
- Reuse valid nonempty final files and never accept `.tmp` files as complete.
- Keep `DOWNLOAD_GT_PLY=0` as the backward-compatible default.
- Do not install environments or download model weights in the data script.

---

### Task 1: Lock The FastVGGT ScanNet-50 Identity

**Files:**
- Create: `configs/fastvggt_scannet50.txt`
- Modify: `tests/camera_iteration/test_autodl_scripts.py`

**Interfaces:**
- Consumes: FastVGGT `eval/scannet_50.yaml`.
- Produces: newline-separated scene IDs selected with `SCENE_LIMIT=0`.

- [ ] **Step 1: Write the failing configuration test**

Add a test that reads `configs/fastvggt_scannet50.txt`, asserts 50 unique
entries, validates `sceneNNNN_NN`, and compares all entries in order with the
official FastVGGT list.

- [ ] **Step 2: Run the test and verify the missing file failure**

Run:

```bash
python -m unittest tests.camera_iteration.test_autodl_scripts.AutoDLScriptsTest.test_fastvggt_scannet50_list
```

Expected: `FileNotFoundError`.

- [ ] **Step 3: Add the exact 50 scene IDs**

Create one scene ID per line, beginning with `scene0000_00` and ending with
`scene0691_00`.

- [ ] **Step 4: Run the focused test**

Expected: one passing test.

### Task 2: Generalize The Existing ScanNet Preparation Script

**Files:**
- Modify: `scripts/autodl/prepare_scannet_camera_iteration.sh`
- Modify: `tests/camera_iteration/test_autodl_scripts.py`

**Interfaces:**
- Consumes: `SCENE_LIST`, `SCENE_LIMIT`, `DOWNLOAD_RETRIES`,
  `DOWNLOAD_GT_PLY`, and `SCANNET_TOS_ACCEPTED`.
- Produces: exact nonempty `.sens`, optional `_vh_clean_2.ply`, and extracted
  `process_scannet/<scene>/{color,pose}`.

- [ ] **Step 1: Write failing shell-contract tests**

Assert the script contains `DOWNLOAD_RETRIES`, `DOWNLOAD_GT_PLY`, scene-ID
validation, `printf '\n\n\n\n'`, both official `--type` values, retry attempts,
exact expected paths, and temporary-file cleanup. Assert its defaults still
use the existing 10-scene list and do not install dependencies or weights.

- [ ] **Step 2: Run the focused tests and verify failure**

Run:

```bash
python -m unittest tests.camera_iteration.test_autodl_scripts
```

Expected: failures for missing retry and PLY behavior.

- [ ] **Step 3: Implement bounded per-asset downloads**

Add:

```bash
DOWNLOAD_RETRIES="${DOWNLOAD_RETRIES:-5}"
DOWNLOAD_GT_PLY="${DOWNLOAD_GT_PLY:-0}"
GT_DOWNLOAD_ROOT="${GT_DOWNLOAD_ROOT:-$SCANNET_ROOT/raw}"
```

Validate positive retry count and boolean PLY mode. Implement a
`download_asset(scene, file_type, download_root, expected)` function that:

1. returns immediately for a nonempty `expected`;
2. invokes the official downloader with automated confirmation;
3. retries only when the command fails or the exact expected file is empty;
4. removes `expected.tmp` before the next attempt;
5. exits after `DOWNLOAD_RETRIES` failures.

Call it for `.sens` in every selected scene and for `_vh_clean_2.ply` only
when `DOWNLOAD_GT_PLY=1`.

- [ ] **Step 4: Preserve extraction and final validation**

Pass the selected list and limit to `extract_scannet_sens.py`, then keep
`missing_processed_scenes` as the final readiness check.

- [ ] **Step 5: Run the focused test module**

Expected: all `AutoDLScriptsTest` tests pass.

### Task 3: Verify The Unified Workflow

**Files:**
- Modify: `doc/2026-07-27_FastVGGT_ScanNet50_Download_Design.md` only if
  implementation details differ from the approved design.

**Interfaces:**
- Consumes: Tasks 1 and 2.
- Produces: a committed, reproducible CPU-validated data-preparation workflow.

- [ ] **Step 1: Run all camera-iteration CPU tests**

```bash
python -m unittest discover -s tests/camera_iteration -v
```

- [ ] **Step 2: Run shell syntax and whitespace validation**

```bash
bash -n scripts/autodl/prepare_scannet_camera_iteration.sh
git diff --check
```

- [ ] **Step 3: Inspect the final diff and commit**

Commit the implementation, configuration, tests, and this plan with:

```bash
git commit -m "Prepare FastVGGT ScanNet-50 data"
```
