# Camera Head Amplification Implementation Plan

**Goal:** Implement a reproducible frozen Camera Head replay that separates
Aggregator token drift from downstream Camera Head amplification.

## Round 1: Replay Core

- Add tests for matching 200-frame IDs into the 500-frame artifact.
- Add hook-based capture for AdaLN input, four trunk blocks, trunk norm, pose
  delta, and accumulated raw pose across four refinement iterations.
- Load only the Camera Head parameters from the local VGGT checkpoint.

## Round 2: Metrics And Contracts

- Test and implement RMS drift and input-relative amplification ratios.
- Validate `H200(Z200)` and `H500(Z500)` against saved raw predictions.
- Decode 9D outputs to camera poses and independently align every prediction
  to raw GT; never align GT.
- Persist only per-frame scalars and aggregate numeric summaries.

## Round 3: Reproduction And Publishing

- Add a CLI consuming one published Round 1.5 run directory.
- Add an AutoDL runner with fixed contexts 200/500 and four iterations.
- Extend the numeric exporter with an exact Round 1.6 artifact whitelist.
- Update worktree README, `AGENTS.md`, and the dated experiment log.

## Verification

- Run focused tests after each failing-test/implementation cycle.
- Run `python -m unittest discover -s tests` in the AutoDL `vggt` environment.
- Run `bash -n scripts/autodl/run_camera_head_amplification.sh`.
- Run a one-scene replay before the four-scene protocol and verify both
  baseline replay checks pass.
