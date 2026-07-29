# 2026-07-29 Local-Global Stitching and Rotation Analysis

## Work Completed

- Added prediction-only local-window stitching utilities and synthetic tests.
- Evaluated sequential overlap Sim(3), global-anchored local assembly, and a
  translation hybrid on the 40-scene ScanNet holdout.
- Separated the independently aligned local-window oracle from realizable complete
  trajectories.
- Rechecked rotation with gauge-invariant relative rotation error and
  orientation-only SO(3) alignment.
- Removed misleading trajectory figures produced from independently GT-aligned local
  windows.

## Main Findings

Naive sequential overlap stitching increases translation error from `0.08806` to
`0.15749`, showing strong accumulated gauge error. Global-anchored local camera centers
reduce it to `0.08278`, improving 33 of 40 scenes. This recovers only about 12% of the
large independent-window oracle gap, but provides a realizable training-free baseline.

The prior claim that global context improves rotation was caused by the evaluation
method. Camera-center Sim(3) alignment gives the same global prediction `3.98 deg` as
one full trajectory and `9.53 deg` when evaluated in 100-frame segments. Under RRE,
local windows are better by about `0.12-0.19 deg` at lags 1-25. Under equal-scale
orientation-only SO(3) alignment, local and global are statistically indistinguishable.

## Decision

Use global rotations and long-range gauge while replacing camera centers with
global-anchored local estimates. Future rotation evaluation must use RRE or
orientation-only SO(3) alignment. GT remains raw in every evaluation; predictions are
aligned where the metric requires alignment.
