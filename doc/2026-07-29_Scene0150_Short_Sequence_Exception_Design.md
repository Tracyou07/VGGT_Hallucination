# Scene0150 Short-Sequence Exception Design

## Problem

The formal ScanNet-50 Camera Context source requests 500 frames for every
scene. `scene0150_00` contains only 430 valid RGB/raw-pose pairs, while the
other 49 scenes contain 500. The existing source contract therefore rejects
the otherwise complete FastVGGT scene set.

## Decision

Keep the original FastVGGT-50 scene set and add one exact exception:

- `scene0150_00` must contain exactly 430 selected frames.
- Every other scene must contain exactly 500 selected frames.
- The source metadata must still declare `frame_counts == [500]`; this records
  the requested protocol length, not guaranteed availability.
- No other short scene is accepted.

The exception changes neither prediction alignment nor GT handling. Source
frame IDs must still equal deterministic uniform selection from processed
ScanNet, and `gt_c2w_raw` must still match raw dataset poses exactly.

## Window Protocol

Window length remains 100 and stride remains 50. The existing tail-coverage
rule appends a final window when the sequence end is not stride-aligned.
`scene0150_00` therefore uses eight windows:

```text
[0,100), [50,150), [100,200), [150,250),
[200,300), [250,350), [300,400), [330,430)
```

All 500-frame scenes retain nine windows. The final short-scene window overlaps
the preceding window by 70 frames so all 430 frames remain represented.

## Formal Completion

Calibration and holdout still require exactly 10 and 40 declared scenes.
Expected window counts are derived from the frozen split and validated source:

- If `scene0150_00` is in calibration: calibration 89, holdout 360.
- If it is in holdout: calibration 90, holdout 359.
- The complete ScanNet-50 run always contains 449 windows.

Analysis must compare the collected count with this derived expectation rather
than accepting an arbitrary number. Threshold fitting remains calibration-only,
and holdout continues to consume the frozen threshold artifact unchanged.

## Verification

Tests must prove that the source contract accepts exactly the 430-frame
exception, rejects 429/431 frames for that scene, and rejects 430 frames for
every other scene. Analyzer tests must cover both possible split placements and
reject missing or extra windows. Existing 500-frame behavior must remain
unchanged.
