# Local-Global Worktree Cleanup

## Scope

The `local-global-consistency-preexperiment` worktree is now focused on the
Round 2A local-global consistency experiment. All `doc/` and `log/` content is
retained.

## Removed

- Environment creation and VGGT checkpoint download scripts.
- Round 1 camera-iteration runtime, exporter, tests, and published results.
- Round 1.5 context-analysis runtime and redundant result artifacts.
- Round 1.6 camera-head amplification runtime, exporter, tests, and results.
- Superseded ScanNet scene lists and obsolete helper APIs.

## Retained

- Round 2A implementation, runner, numeric exporter, and focused tests.
- ScanNet-50 download and extraction workflow.
- Existing-environment and existing-checkpoint runtime checks.
- Four frozen Round 1.5 `frames_500/context_diagnostics.npz` inputs plus their
  shared `run_metadata.json`.
- VGGT camera-token tracing support required by Round 2A.

## Verification

- Active source and tests contain no imports of deleted experiment modules.
- `git diff --check` reports no whitespace errors.
- 20 focused dependency-free tests pass.
- The complete local test discovery loads 23 tests successfully; four modules
  cannot import locally because the portable Python runtime lacks `numpy` and
  `torch`.

Run the complete suite on AutoDL inside the existing `vggt` Conda environment:

```bash
conda activate vggt
python -m unittest discover -s tests/local_global_consistency -v
```
