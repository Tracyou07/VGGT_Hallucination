# Local-Global Worktree Cleanup Plan

**Goal:** Reduce this worktree to the code and frozen inputs required to run
the Round 2A local-global consistency experiment with an existing `vggt`
Conda environment and VGGT checkpoint.

## Keep

- Baseline `vggt/` plus camera-trace observability changes.
- `pre_experiments/local_global_consistency/`.
- Shared camera utilities for checkpoint loading, raw ScanNet input, pose
  alignment, and atomic metadata writes.
- ScanNet-50 download/extraction scripts and the local-global AutoDL runner.
- Local-global numeric exporter, focused tests, all `doc/`, and all `log/`.
- The frozen four-scene Round 1.5 input reduced to `run_metadata.json` and
  each scene's `frames_500/context_diagnostics.npz`.

## Remove

- Environment creation and checkpoint download scripts.
- Round 1 camera-iteration runner, metrics, exporter, published results, and
  tests that only cover that retired workflow.
- Round 1.5 analysis/runtime code after retaining its minimal frozen input.
- Round 1.6 Camera Head amplification code, runner, exporter, results, and
  tests.
- Superseded 10-scene and context scene lists.

## Update

- Make `configs/fastvggt_scannet50.txt` the ScanNet preparation default with
  all scenes selected.
- Keep the local-global runner at the frozen four-scene protocol.
- Rewrite active README and `AGENTS.md` guidance around Round 2A only.
- Remove stale test assertions and error messages that instruct users to
  create environments or download checkpoints.

## Verification

Run focused AutoDL, ScanNet, local-global, exporter, contract, and VGGT camera
option tests; run Bash syntax checks; scan imports and documentation for
deleted runtime paths; and require a clean `git diff --check`.
