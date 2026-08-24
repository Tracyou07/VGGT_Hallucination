# 3D Reconstruction Diffusion Documentation Organization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the repository's documentation navigable from one project entry point while preserving historical experiments and results unchanged.

**Architecture:** Keep `README.md` as the project entry point and add `docs/README.md` as the complete index. Add one CVA02 status page that links the frozen design and execution plan. Mark `doc/`, `log/`, and audit worktrees as historical or auxiliary sources instead of moving their files.

**Tech Stack:** Markdown, relative links, PowerShell validation, Git.

---

### Task 1: Replace the repository README with the project entry point

**Files:**
- Modify: `README.md`
- Read: `docs/README.md` after Task 2 for the final index link

- [ ] **Step 1: Write the new README sections**

Create these sections in this order: project purpose, research evolution, current CVA02 status, authoritative paths, experiment gates, historical material, quick navigation, and next actions. State explicitly that `cva02_dev` is the primary worktree and that old local/global numbers are historical evidence, not CVA02 results.

- [ ] **Step 2: Verify path and status claims**

Run `git branch --show-current`, `git rev-parse HEAD`, and `Test-Path` for the CVA02 design and plan. Copy only verified paths and commit identities into the README.

- [ ] **Step 3: Check Markdown structure**

Run `git diff --check` and ensure every link target exists with `Test-Path` or `rg --files`.

- [ ] **Step 4: Commit**

```powershell
git add README.md
git commit -m "Organize project README around current CVA02"
```

### Task 2: Add the complete documentation index

**Files:**
- Create: `docs/README.md`

- [ ] **Step 1: Build the index categories**

Include four tables: current CVA02 documents, historical experiments, data/evaluation tools, and process/audit records. Each row must include path, purpose, status, and whether it may support current conclusions.

- [ ] **Step 2: Add the source-of-truth rule**

State that current protocol and progress come from CVA02 design/status pages; historical `doc/` and `log/` files remain immutable evidence; `cva02_dev` is the main development worktree; the other worktrees are auxiliary copies.

- [ ] **Step 3: Validate every listed path**

Run a PowerShell loop over all paths listed in the index and fail if any target is missing. Run `git diff --check`.

- [ ] **Step 4: Commit**

```powershell
git add docs/README.md
git commit -m "Add documentation index and source-of-truth rules"
```

### Task 3: Add the current CVA02 status page

**Files:**
- Create: `docs/camera_velocity_ambiguity_02_status.md`

- [ ] **Step 1: Record frozen identity and scope**

Document the branch, base commit, FastVGGT pin, H20 code/data/weight/output paths, and the statement that no formal GPU result exists until the ScanNet integrity marker passes.

- [ ] **Step 2: Record the experiment state machine**

Describe `calibration -> freeze -> development evaluation -> decision`, the 10/40 split, the 50-scene/449-window/399-pair counts, and the four possible interpretations. Mark the fourth interpretation as requiring independent RGB-D observation energy.

- [ ] **Step 3: Record implementation ownership and gates**

Link Person A/B/C task ranges, the ScanNet 100/100 gate, the FastVGGT plot reproduction-only rule, and the no-training rule for the pre-experiment.

- [ ] **Step 4: Add an explicit update protocol**

Specify that only this page receives current progress updates, with timestamp, branch HEAD, ScanNet state, and next action. Historical logs are linked but not treated as live status.

- [ ] **Step 5: Validate links and terminology**

Run `git diff --check` and search the page for unresolved placeholder markers and contradictory claims such as `fresh holdout`.

- [ ] **Step 6: Commit**

```powershell
git add docs/camera_velocity_ambiguity_02_status.md
git commit -m "Add CVA02 current status and execution gates"
```

### Task 4: Add navigation for historical and auxiliary material

**Files:**
- Create: `doc/README.md`
- Create: `log/README.md`
- Modify: `pre_experiments/README.md`

- [ ] **Step 1: Add `doc/README.md`**

Explain that `doc/` contains historical designs and implementation plans. Group entries by local/global, camera iteration, ScanNet/FastVGGT, and publication/result handling. Link representative files and state that these files are not the CVA02 source of truth.

- [ ] **Step 2: Add `log/README.md`**

Explain the difference between completed historical evidence and live status. Link the stitching/rotation log and the round history in chronological order.

- [ ] **Step 3: Update `pre_experiments/README.md`**

Add a “current entry point” section linking to the CVA02 status page, design, plan, and package path, while preserving the existing Round 2 semantics.

- [ ] **Step 4: Validate the navigation**

Run `rg --files` for every linked target and `git diff --check`.

- [ ] **Step 5: Commit**

```powershell
git add doc/README.md log/README.md pre_experiments/README.md
git commit -m "Add historical documentation navigation"
```

### Task 5: Final consistency audit

**Files:**
- Modify: `docs/README.md` or `docs/camera_velocity_ambiguity_02_status.md` only if validation finds a stale path or claim

- [ ] **Step 1: Validate repository state**

Run `git status --short`, `git diff --check`, and `git log -5 --oneline`.

- [ ] **Step 2: Validate links and forbidden ambiguity**

Check that all Markdown links point to existing files and that current pages do not call the 40-scene historical evaluation a fresh holdout.

- [ ] **Step 3: Validate preservation**

Use `git diff --stat` and `git diff --name-status` to confirm that only the planned documentation files changed; no code, dataset, weight, or result file may be modified.

- [ ] **Step 4: Commit any final wording correction**

```powershell
git add docs doc log pre_experiments/README.md README.md
git commit -m "Finalize project documentation organization"
```
