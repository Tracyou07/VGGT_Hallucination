# Camera Refiner Pure-Scale Calibration Results

## Protocol and Integrity

Run `b4920c1_548d8da82129` evaluated the frozen VGGT Camera Head on the fixed
10-scene calibration split. Each scene contributes 500 target frames. The
candidate grid contains pure local hidden states from 100-, 200-, and 300-frame
contexts, each interpolated toward the 500-frame global hidden state with
`alpha` in `{0.01, 0.02, 0.05, 0.10}`.

The transferred result contains 90 `100/50`, 40 `200/100`, and 30 `300/150`
local windows. All 160 window diagnostics and all 10 scene shards pass their
strict repository loaders. The source commit is `b4920c1` and is an ancestor of
the current branch.

## Calibration Result

Lower translation delta is better. No candidate passes the frozen-policy
criteria because every 95% bootstrap interval crosses zero.

| Candidate | Scale | Alpha | Mean delta | CI95 upper | Scenes improved | Frame fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `a0p02_b1_0_0` | 100 | 0.02 | -0.0001867 | +0.0001150 | 40% | 59.72% |
| `a0p01_b1_0_0` | 100 | 0.01 | -0.0001404 | +0.0000171 | 60% | 61.24% |
| `a0p02_b0_0_1` | 300 | 0.02 | -0.0001228 | +0.0000830 | 60% | 52.82% |
| `a0p02_b0_1_0` | 200 | 0.02 | -0.0000975 | +0.0000610 | 60% | 55.26% |
| `a0p01_b0_1_0` | 200 | 0.01 | -0.0000827 | +0.0000198 | 80% | 55.76% |

All candidates remain below the configured rotation and FOV safety limits.
The failure is statistical robustness, not a safety regression.

## Interpretation

The best mean result, 100 frames at `alpha=0.02`, is driven by a minority of
scenes: its median delta is positive and only 4/10 scenes improve. At
`alpha=0.01`, 100 frames gives the strongest mean and per-frame effect, while
200 frames gives the broadest scene coverage at 8/10. These are weak but
repeatable scale preferences, not sufficient evidence for a fixed refiner.

Larger `alpha` generally increases variance and regression risk. Local hidden
displacement also decreases as context length approaches 500 frames, but a
smaller hidden displacement does not guarantee better camera output.

## Decision

Do not run holdout and do not weaken the frozen gates after seeing calibration.
This run tested only pure scales and produced no `frozen_candidate_policy.json`.
The next predeclared step is a new calibration run with
`CANDIDATE_FAMILY=all`, reusing the validated local-window runs and adding the
three mixtures from the design. Holdout remains untouched until one candidate
passes the original gates.
