# Round 1 Camera Iteration Results

## Scope

The checkpoint-backed study ran on 10 ScanNet scenes with nested selections at
25, 50, 100, 200, and 500 frames. Each selection used one Aggregator forward
and retained Camera Head outputs at iterations 1, 2, 4, 8, and 16. This produced
50 scene/length selections and 250 aggregate metric rows. Prediction quality is
reported only after Sim(3) alignment; GT poses remain raw.

## Confirmed Findings

- Camera Head refinement converges quickly. Mean raw 9D update norm is `1.8702`
  at iteration 1, `0.004624` at iteration 2, `0.000439` at iteration 4, and
  `0.0000785` at iteration 16.
- Iteration 1 is the initial full 9D pose prediction because no previous pose
  exists. Iterations 2 and later are additive residual updates. All iterations
  share one four-block Transformer trunk and one pose MLP; the normalized Camera
  Tokens remain fixed across iterations.
- No non-default iteration satisfies the planned gate of at least 5% median
  aligned ATE improvement over iteration 4 while remaining non-worse on at
  least 7 of 10 scenes. Small per-case preferences are inconsistent. The fixed
  baseline remains iteration 4; an iteration selector is not justified.
- Long-sequence degradation is concentrated in outliers. At iteration 4, the
  aligned ATE mean/median is `0.0848/0.0665` at 200 frames and
  `0.2104/0.0690` at 500 frames. `scene0000_00` changes from `0.1933` to
  `1.3663`; `scene0691_00` changes from `0.0743` to `0.1336`. Most other scenes
  remain close to their shorter-context values.
- The same frame can receive a very different Camera update under a longer
  context. For example, `scene0109_00` frame 1075 has iteration-4 update norm
  `0.00165` at 200 frames and `0.08948` at 500 frames.

## Interpretation Boundary

Round 1 proves that additional Camera Head iterations do not solve the observed
long-sequence failures. It also proves context sensitivity of internal pose
updates. It does not prove that Camera Token drift causes pose error: the run did
not save matched per-frame Camera Tokens or per-frame aligned errors. Large
update norm is not itself a camera-error metric.

The working hypothesis is that long-context attention changes selected Camera
representations, after which the shared pose decoder converges to a biased 9D
pose. Round 1.5 is responsible for testing this chain with matched frame IDs,
normalized Camera Tokens, independently aligned predictions, raw GT, and
per-frame translation/rotation errors.

## Decision

Fix `camera_num_iterations=4`. Do not spend further work on 8/16-iteration
refinement or geometry-aware iteration selection. Run the targeted Round 1.5
context-consistency diagnosis before implementing GT-free geometry residuals.
