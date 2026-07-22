# Experiment 2 Observation: GT Comparison

Date: 2026-07-02

This note records the GT comparison for the ScanNet VGGT hallucination run. It
uses the 25 per-run `metrics.json` files under
`results/scannet_hallucination/**/metrics.json` and does not use derived delta
columns.

## Metric Directions

| Metric group | Compared against | Direction |
| --- | --- | --- |
| Pose ATE / ARE / RPE | predicted pose vs GT pose | lower is better |
| Depth AbsRel / RMSE | predicted depth vs GT depth | lower is better |
| Depth delta1 | predicted depth vs GT depth | higher is better |
| Point-cloud Chamfer | candidate point cloud vs GT ScanNet mesh | lower is better |

For point clouds, `GT depth + GT pose` is not expected to be zero. It is a
partial RGB-D reconstruction baseline built from the selected GT depth frames
and GT poses, then compared against the ScanNet mesh. It differs from directly
comparing the GT mesh with itself because it is affected by frame coverage,
depth validity, sampling, and reconstruction geometry.

## Pose and Depth vs GT

| Frames | Pose ATE RMSE | Pose ARE deg | Pose RPE rot | Depth AbsRel raw | Depth AbsRel aligned | Depth delta1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 0.0143 | 12.4501 | 0.1217 | 0.4399 | 0.0202 | 0.9891 |
| 200 | 0.0304 | 6.0397 | 0.1379 | 0.4560 | 0.0221 | 0.9872 |
| 300 | 0.0313 | 6.8920 | 0.1514 | 0.4633 | 0.0232 | 0.9859 |
| 400 | 0.0436 | 4.0524 | 0.1707 | 0.4665 | 0.0217 | 0.9871 |
| 500 | 0.0518 | 5.4723 | 0.2108 | 0.4647 | 0.0223 | 0.9873 |

## Point Cloud vs GT Mesh: Raw Chamfer

| Frames | GT depth + GT pose | GT depth + Pred pose | Pred depth + GT pose | Pred depth + Pred pose | Native world_points |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 0.3922 | 0.4184 | 0.8031 | 0.7902 | 0.7887 |
| 200 | 0.3284 | 0.3878 | 0.8084 | 0.7954 | 0.7949 |
| 300 | 0.2963 | 0.3891 | 0.7954 | 0.7898 | 0.7894 |
| 400 | 0.2742 | 0.3834 | 0.7929 | 0.7852 | 0.7846 |
| 500 | 0.2463 | 0.3805 | 0.7840 | 0.7803 | 0.7789 |

## Point Cloud vs GT Mesh: Aligned Chamfer
`
| Frames | GT depth + GT pose | GT depth + Pred pose | Pred depth + GT pose | Pred depth + Pred pose | Native world_points |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 0.6223 | 0.5863 | 0.5535 | 0.5852 | 0.5858 |
| 200 | 0.6350 | 0.6459 | 0.4984 | 0.5297 | 0.5391 |
| 300 | 0.5753 | 0.5467 | 0.4812 | 0.5151 | 0.5071 |
| 400 | 0.5426 | 0.5557 | 0.4659 | 0.4891 | 0.4772 |
| 500 | 0.5867 | 0.5588 | 0.4367 | 0.4597 | 0.4768 |

## Observations

1. Pose error against GT increases with frame count. ATE RMSE rises from 0.0143
   at 100 frames to 0.0518 at 500 frames, and RPE rotation rises from 0.1217 to
   0.2108.

2. Depth has a large raw scale error but remains stable after scale alignment.
   Aligned AbsRel stays around 0.020-0.023, and delta1 remains near 0.986-0.989.

3. The raw `GT depth + GT pose` RGB-D baseline improves with more frames because
   scene coverage increases. It is not a zero-error GT-vs-GT comparison.

4. VGGT final point clouds do not worsen with longer inputs under Chamfer.
   `Pred depth + Pred pose` and native `world_points` are close to each other
   and improve slightly in raw Chamfer and more clearly in aligned Chamfer.

5. Aligned Chamfer should be interpreted cautiously. Because it includes
   alignment, it can hide scale and pose errors that are visible in raw metrics
   or direct camera-pose evaluation.

## Current Interpretation

The GT comparison reinforces the pose-first finding: pose has a clear
long-frame degradation against GT, while depth remains stable after scale
alignment. Point-cloud Chamfer does not directly expose this pose degradation;
it is affected by coverage and alignment and can improve even when camera pose
metrics get worse.
