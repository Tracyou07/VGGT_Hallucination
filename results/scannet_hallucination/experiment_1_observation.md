0.# Experiment 1 Observation: Frame-Length Trend

Date: 2026-07-01

This note records the first ScanNet VGGT hallucination observation. It only
summarizes the trend of VGGT camera pose, depth, and point-cloud outputs across
different input lengths. Detailed comparison against GT pose/depth/mesh is left
for the next analysis.

## Setup

- Dataset: ScanNet scenes `scene0000_00`, `scene0013_02`, `scene0029_01`,
  `scene0042_02`, and `scene0056_00`.
- Sampling: prefix sequence.
- Frame counts: 100, 200, 300, 400, 500.
- Metrics source: `results/scannet_hallucination/**/metrics.json`.
- Aggregated table: `results/scannet_hallucination/aggregate_from_metrics.csv`.

## Aggregate Trend

| Frames | Pose ATE RMSE | Pose RPE Rot | Depth AbsRel aligned | Derived PCD Chamfer aligned | Native PCD Chamfer aligned |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 0.0143 | 0.1217 | 0.0202 | 0.5852 | 0.5858 |
| 200 | 0.0304 | 0.1379 | 0.0221 | 0.5297 | 0.5391 |
| 300 | 0.0313 | 0.1514 | 0.0232 | 0.5151 | 0.5071 |
| 400 | 0.0436 | 0.1707 | 0.0217 | 0.4891 | 0.4772 |
| 500 | 0.0518 | 0.2108 | 0.0223 | 0.4597 | 0.4768 |

## Observations

1. Camera pose error increases with frame length. Mean aligned ATE rises from
   0.0143 at 100 frames to 0.0518 at 500 frames, and mean RPE rotation rises
   from 0.1217 to 0.2108. This is the clearest long-frame accumulation signal.

2. Depth does not show a clear long-frame hallucination trend after scale
   alignment. Aligned AbsRel remains around 0.020-0.023 across all frame counts.
   Raw depth error is much higher, so the current depth issue is mainly a scale
   bias rather than a frame-length-amplified shape error.

3. Point-cloud Chamfer does not worsen with longer inputs. Both derived
   point clouds (`pred depth + pred pose`) and native `world_points` improve in
   aligned Chamfer as frames increase. This suggests the point-cloud metric is
   influenced by increased scene coverage, sampling, and alignment, and may hide
   the pose degradation observed directly in camera metrics.

4. Native `world_points` and derived point clouds behave similarly. Their
   Chamfer values are close across all frame counts, so this experiment does not
   show a strong independent failure mode for the native point head.

## Current Interpretation

The first experiment supports a pose-first hallucination hypothesis: long input
sequences amplify camera-pose error, while depth remains relatively stable after
scale alignment. Point-cloud Chamfer is not a sensitive indicator of this pose
accumulation in the current setup; it improves with frame count instead of
degrading.

GT-based attribution is intentionally not concluded here. The next analysis
should compare VGGT outputs against GT pose, GT depth, and GT mesh using the
counterfactual point-cloud metrics.
