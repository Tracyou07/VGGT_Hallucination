# Unified AutoDL Results Root Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. The user explicitly prohibited subagents. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate existing experiment roots into `/root/autodl-tmp/results` and make every active branch write that canonical root directly.

**Architecture:** A standard-library migration CLI owns an explicit experiment allowlist, performs a complete collision preflight, then atomically moves or safely merges each legacy root before creating a historical compatibility link. Each branch independently adopts `RESULTS_ROOT`, preserving its existing internal `results`, `state`, and pointer layout.

**Tech Stack:** Python 3.10+, POSIX symlinks, SHA-256, `unittest`, Bash, Git worktrees.

## Global Constraints

- Never move or rewrite `datasets`, `ckpt`, `hf_home`, Git checkouts, or Conda environments.
- Never overwrite a destination file or force-push a branch.
- A content or filesystem-type conflict aborts before the first migration mutation.
- Updated scripts write canonical paths directly; compatibility links serve historical absolute paths only.
- Preserve explicit environment overrides and every branch's existing experiment protocol.
- Tests require no CUDA, checkpoint, network access, or ScanNet credentials.

---

### Task 1: Idempotent Migration CLI

**Files:**
- Create: `scripts/autodl/migrate_results_root.py`
- Create: `tests/results_root/__init__.py`
- Create: `tests/results_root/test_migrate_results_root.py`

**Interfaces:**
- Produces: `EXPERIMENT_ROOTS: tuple[str, ...]` with the eleven approved roots.
- Produces: `plan_migration(autodl_tmp, results_root) -> MigrationPlan`.
- Produces: `execute_migration(plan, *, dry_run=False, create_links=True) -> dict[str, object]`.
- Produces: CLI flags `--autodl-tmp`, `--results-root`, and `--dry-run`.

- [ ] Write failing tests for the exact allowlist and exclusion of protected roots.
- [ ] Write failing tests for clean move, destination-only recovery, partial merge, identical duplicate removal, and second-run idempotence.
- [ ] Write a failing test proving one differing collision prevents every planned root from moving.
- [ ] Run `python -m unittest discover -s tests/results_root -v`; verify missing-module failure.
- [ ] Implement immutable plan records, chunked SHA-256 comparison, symlink/type validation, whole-plan preflight, interruption-safe merge, and JSON CLI reporting.
- [ ] Re-run the focused suite and `python -m compileall -q scripts/autodl/migrate_results_root.py`.
- [ ] Commit with `git commit -m "Add idempotent AutoDL results migration"`.

### Task 2: Current Refiner Branch Canonical Paths

**Files:**
- Modify: `scripts/autodl/camera_refiner_data_construction/run_multiscale_study.sh`
- Modify: `pre_experiments/camera_refiner_data_construction/run_study.py`
- Modify: `pre_experiments/local_global_consistency/run_study.py`
- Modify: `README.md`
- Modify: `tests/camera_refiner_data_construction/test_autodl.py`

**Interfaces:**
- Consumes: `RESULTS_ROOT`, defaulting to `$AUTODL_TMP/results`.
- Produces: refiner work root `$RESULTS_ROOT/camera_refiner_data_construction`.
- Produces: direct local-global default `$RESULTS_ROOT/local_global_consistency/results`.

- [ ] Add failing assertions that defaults contain `RESULTS_ROOT` and reject legacy output expressions.
- [ ] Run the focused AutoDL test and confirm failure.
- [ ] Add `RESULTS_ROOT` to Shell and Python defaults without changing data/checkpoint paths.
- [ ] Update all copyable commands and output-layout documentation.
- [ ] Run Camera Refiner and Local-Global focused suites plus Bash syntax checks.
- [ ] Commit with `git commit -m "Route refiner outputs through unified results root"`.

### Task 3: Prepare Independent Branch Worktrees

**Files:**
- No tracked files expected.

**Interfaces:**
- Consumes: freshly fetched `origin/*` refs.
- Produces: one clean named worktree per rollout branch under `.worktrees/results-root/`.

- [ ] Verify every existing worktree is clean; stop rather than overwrite user changes.
- [ ] Fast-forward clean local branches that are behind their upstream.
- [ ] Create tracking branches and worktrees for missing branches.
- [ ] Record each starting commit before edits.

### Task 4: Update Every Active Branch

**Files by branch:**
- `camera-context-consistency-preexperiment`: Camera Context/Iteration shell defaults, Python defaults if present, README, path tests.
- `camera-iteration-preexperiment`: Camera Iteration shell defaults, Python defaults if present, README, path tests.
- `camera-head-amplification-preexperiment`: amplification shell and `run_replay.py` defaults, README, tests.
- `local-global-consistency-preexperiment`: local-global `run_study.py`, README, tests.
- `camera-hidden-state-attribution-preexperiment`: four hidden-study shell runners, four Python defaults, inherited local-global default, README, tests.
- `phenomenon-characterization`: ScanNet runner and publisher defaults, README, tests.
- `camera-refiner-training`: every retained output default and README path, tests.
- `main`: all result-producing shell/Python defaults and top-level documentation/tests present at that revision.

**Interfaces:**
- Consumes: the Task 1 migration commit through cherry-pick.
- Produces: direct `$RESULTS_ROOT/{experiment}` output defaults on each branch.

- [ ] For each branch, cherry-pick the tested migration CLI commit.
- [ ] Add a failing branch-local path-contract test or update the existing static contract to require `RESULTS_ROOT`.
- [ ] Replace only output/state/pointer roots; leave dataset, checkpoint, cache, and repository defaults unchanged.
- [ ] Update exact README commands on that branch.
- [ ] Run its focused CPU tests, shell syntax checks, and `git diff --check`.
- [ ] Commit path changes with `git commit -m "Route experiment outputs through unified results root"`.
- [ ] Push the named branch normally and verify local/remote commit equality.

### Task 5: Final Cross-Branch Audit

**Files:**
- Verify only.

**Interfaces:**
- Confirms all eight active experiment branches plus `main` expose the migration CLI and contain no legacy default output expressions.

- [ ] Fetch all remote refs after pushes.
- [ ] Use `git grep` on every remote branch for legacy `$AUTODL_TMP/{experiment}` output defaults; require no matches outside migration compatibility data/tests.
- [ ] Verify every remote branch contains `scripts/autodl/migrate_results_root.py`.
- [ ] Re-run the migration suite and current branch focused tests from clean HEAD.
- [ ] Confirm no dataset, checkpoint, NPZ, PLY, image, or result artifact was committed.
- [ ] Publish the exact AutoDL dry-run, migration, inspection, and rollback-safe diagnostic commands.
