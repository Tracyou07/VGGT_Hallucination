# Trajectory Overlay Visualization Design

## Goal

Use existing ScanNet-50 artifacts to compare raw GT, 500-frame global
prediction, and 100-frame local-window predictions before and after alignment.
The first visualization targets `scene0409_01`, which has the largest mean
global-minus-local translation error growth in the holdout set.

## Metric Contract

GT is always read from `gt_c2w_raw` and is never transformed. The unaligned row
shows raw prediction coordinates. In the aligned row, the complete global
trajectory receives one Sim(3) alignment to raw GT, while each local window is
independently aligned to its matching raw GT segment. This exactly matches the
existing validation protocol.

Independently aligned local windows must remain separate colored segments. The
figure must not imply that they form a stitched 500-frame trajectory.

## Layout

Create one 2-by-3 PNG:

- rows: raw predictions, then aligned predictions;
- columns: XY view, XZ view, and 3D view;
- raw GT: black solid line;
- global prediction: red solid line;
- local windows: continuous categorical color sequence;
- start and end positions: distinct markers;
- axes: equal spatial scaling where supported.

Titles and legends identify alignment state and trajectory source. A figure
note states that local segments are independently aligned and not stitched.

## Inputs and Output

Read the global artifact from
`results/camera_context/d33d98b_309a9a586242/<scene>/frames_500/context_diagnostics.npz`
and local artifacts from the formal holdout run
`results/local_global_consistency/scannet50/runs/holdout/3d5de75_35564d765bb5/`.

Write the PNG under the formal holdout run's `visualizations/` directory. The
visualization script should accept a scene argument so the same rendering can
be reproduced for other scenes without changing code.

## Verification

Verify that all frame IDs and local GT segments match the global raw GT,
coordinates are finite, the expected eight or nine local windows are present,
and the generated PNG is nonempty. Inspect the image to confirm readable
legends, equal-scale 2D axes, and no implication that local segments are one
continuous trajectory.
