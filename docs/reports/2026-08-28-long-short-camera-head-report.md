# Long–Short Native Camera Head Fine-Tuning Report

## Outcome

The protocol-compliant H20 experiment is classified `NO_SOURCE_HEAD_SIGNAL`.
The native Camera Head learned both objectives, but the quality-weighted
short-window supervision did not generalize reliably when inference received
only the 500-frame long-window Camera tokens.

This rejects the current **Camera-Head-only** source solution. It does not show
that short windows are useless. It shows that a shared decoder cannot recover
their useful local correction reliably from the frozen long-window token
representation.

## Run identity

- Formal run: `/data/yjh/output/vggt/long_short_camera_head/long_short_head_formal_20260828T072407Z`
- Git revision: `2476a59f583ce4c39bbe66dc65d6a8e5cddfb52e`
- Base VGGT checkpoint SHA-256:
  `f164acf60724910d8fe1578bb499d800850c7bb0948db7555c413f9fbe60467e`
- Split: eight training scenes and two locked-replay scenes
- Evaluation: both variants on all ten scenes; only the two locked-replay scenes
  participate in the acceptance decision
- Long-only inference audit: passed
- Repeated `verified_completion.json` SHA-256:
  `54e72876c7596ba5a02b31b7c089ce60dad81a64f6a5402a6e7aa5113410ba9b`
- Artifact size: 7.8 GiB; no large artifact was copied from H20

The earlier run `long_short_head_20260828T064208Z` is superseded and must be
treated as provisional because it used a shortened 4/40-step schedule.

## Locked protocol

The student consumed only cached `[500, 2048]` long-window Camera tokens. GT
poses, the frozen scene Sim(3), short-window teacher poses, and quality weights
lived in separate privileged sidecars used only during training and scoring.

- Smoke: `scene0000_00`, 20 updates, BF16 autocast, AdamW, learning rate
  `2e-6`, weight decay `1e-4`, gradient clipping at one.
- Calibration: at most 400 updates, validation every 25, patience 100, identical
  seed/order/optimizer for `gt_only` and `long_short`.
- Trainable scope: final native Camera Head transformer block and native pose
  decoder only.
- Inference: 500-frame long context only; no GT, short token, teacher, or
  privileged-label input.

The smoke gate passed strictly: loss fell from `0.06355449` to `0.05972434`,
the best checkpoint reloaded tensor-for-tensor exactly, and its long-only output
contained 500 finite poses.

Both calibration variants selected step 25 and stopped normally at step 125
after 100 updates without validation improvement:

- `gt_only` training loss: `0.05610651 → 0.04929767`;
- `long_short` training loss: `0.06355449 → 0.05596327`.

## Locked-replay results

| Scene | Original long RMS | GT-only RMS | Long–short RMS | Long–short utility |
|---|---:|---:|---:|---:|
| `scene0325_01` | 0.08137197 | 0.08443880 | 0.08461124 | -3.980% |
| `scene0675_00` | 0.11056502 | 0.10999312 | 0.10985244 | +0.645% |
| Mean | 0.09596850 | 0.09721596 | 0.09723184 | -1.668% (mean scene utility) |

The long–short model was also slightly worse than the matched GT-only control
in mean locked-replay RMS. Mean rotation error increased by `0.0406°`, so the
rotation guard passed. Positive utility, per-scene harm, and the GT-only
comparison gates failed.

The ten-scene diagnostics make the failure mode clearer. On most training
scenes both models improve the original long prediction by a small amount, and
the two models remain extremely close. On the untouched scenes, one improves
slightly while the other degrades by about four percent. This is overfitting to
small scene-dependent corrections, not evidence of a stable short-window rule
that the frozen long tokens expose.

## Source-level interpretation

Longer Camera Head training is not justified: validation was best at the first
25-step checkpoint and worsened afterward, while training loss kept falling.
The bottleneck is upstream of the decoder.

The next source-level experiment should unfreeze the final native Aggregator
block(s) together with the Camera Head. Short-window supervision would then
shape the 500-frame forward pass so its Camera tokens retain locally reliable
motion evidence. The same GT-only control, long-only inference boundary,
ten-scene diagnostics, and two locked-replay gates should remain unchanged.

## Verification

- Local Camera Head tests: 38 passed.
- Compatibility tests: 64 VRFM tests passed (one Windows symlink skip) and 55
  selector tests passed.
- H20 preflight: all 38 Camera Head tests passed and were hash-bound to the run.
- All stderr logs are empty.
- The verifier checked the formal configuration, data/config digests, smoke
  decrease and exact reload, checkpoint metadata, 20 predictions, 20 metric
  files, all stage completion hashes, report hash, and long-only signature.
- Independent H20 verification ran twice and produced the identical completion
  hash shown above.
