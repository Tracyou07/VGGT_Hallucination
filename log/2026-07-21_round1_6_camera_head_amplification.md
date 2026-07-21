# 2026-07-21 Round 1.6 Camera Head Amplification

## Why This Round Was Added

Round 1.5 showed that long-context failures coincide with drift in normalized
Aggregator Camera Tokens, but it did not exclude amplification by the Camera
Head's four transformer blocks. Round 1.6 isolates that downstream path before
the project enters geometry-residual validation.

## Implementation Completed

- Created branch/worktree `camera-head-amplification-preexperiment` from the
  completed Round 1.5 result commit.
- Added selective Camera Head checkpoint loading; the Aggregator and image
  encoder are not constructed or executed.
- Added hook-based capture for AdaLN input, four trunk blocks, trunk norm, 9D
  delta, and accumulated raw pose across refinement iterations.
- Added fixed-length token-perturbation and extra-head-context comparisons.
- Added strict replay validation against saved 200- and 500-frame activated
  pose encodings.
- Added independently aligned pose metrics against raw GT, a fixed AutoDL
  runner, CPU unit tests, and scalar-only result publishing.

## Local Verification Constraint

The Windows host currently exposes no `python`, `python3`, or `py` executable,
so the unit suite cannot run locally. The AutoDL `vggt` environment must run
`python -m unittest discover -s tests` before the four-scene replay. Bash syntax
and repository-level static checks are performed locally.

## Result Status

No Round 1.6 scientific result is recorded yet. Implementation completion is
not evidence that Camera Head amplification occurs; conclusions require the
remote replay and successful baseline gates.
