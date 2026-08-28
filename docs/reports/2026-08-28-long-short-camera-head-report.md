# Long–Short Native Camera Head Fine-Tuning Report

## Outcome

The source-level Camera Head experiment completed on H20 and is classified
`NO_SOURCE_HEAD_SIGNAL`. The native Camera Head learned the training objective,
but the quality-weighted short-window variant did not improve the two locked
replay scenes when inference received only the 500-frame Camera tokens.

This checkpoint is therefore **not deployable**. The negative result points to
the frozen long-window token representation, rather than optimization failure,
as the next component to test.

## Run identity

- Formal run: `/data/yjh/output/vggt/long_short_camera_head/long_short_head_20260828T064208Z`
- Training commit recorded by the manifest: `352765a4a906b768d7cf011522f81d58c3d6c9dc`
- Reporting fix and final verifier commit: `59a8083`
- Base VGGT checkpoint SHA-256:
  `f164acf60724910d8fe1578bb499d800850c7bb0948db7555c413f9fbe60467e`
- Data split: eight training scenes and two locked-replay scenes
- Long-only inference audit: passed
- Repeated `verified_completion.json` SHA-256:
  `b6385283039cd4ac11078123df151e7cc8e3a562547384621974f3f0d3efc8d5`
- Formal artifact size: 7.8 GiB; no large artifact was copied from H20.

## Protocol

The student consumed only cached `[500, 2048]` long-window Camera tokens. GT
poses, the frozen scene Sim(3), short-window teacher poses, and teacher weights
were stored in separate privileged sidecars. The two matched variants began
from the same VGGT checkpoint:

- `gt_only`: GT translation, relative-translation, rotation, and anchor losses;
- `long_short`: the same losses plus positive-utility, quality-weighted
  short-window teacher consistency.

Only the final native Camera Head transformer block and its native pose decoder
were trainable. Inference had no GT, short token, teacher, or privileged-label
argument. Teacher coverage was substantial (300–500 of 500 frames per scene),
so the result is not explained by an empty teacher mask.

The one-scene smoke gate passed: its objective decreased monotonically from
`0.0641164` to `0.0626640` in four steps. Both matched calibrations stopped at
step 25 after their locked-replay selection metric stopped improving.

## Locked-replay results

| Scene | Original long RMS | GT-only RMS | Long–short RMS | Long–short utility |
|---|---:|---:|---:|---:|
| `scene0325_01` | 0.0813720 | 0.0826863 | 0.0827762 | -1.7257% |
| `scene0675_00` | 0.1105650 | 0.1100993 | 0.1100902 | +0.4295% |
| Mean | 0.0959685 | 0.0963928 | 0.0964332 | -0.6481% (mean scene utility) |

The long–short variant was also approximately 0.042% worse in mean RMS than
the matched GT-only control. Mean rotation error increased by only 0.0214°, so
the rotation guard passed. It failed the positive-utility, per-scene-harm, and
GT-only comparison gates.

Training itself behaved normally:

- GT-only objective: `0.0565422 → 0.0526214`;
- long–short objective: `0.0641164 → 0.0593664`.

Thus the experiment learned a small correction, but that correction was
scene-dependent: it helped `scene0675_00` slightly and harmed `scene0325_01`
more strongly. A Camera Head shared across scenes could not reliably infer the
short-window correction from the frozen long-window tokens.

## Interpretation and next source-level step

This result rejects the current **head-only** solution; it does not establish
that short-window supervision is useless. The earlier evidence says short
windows contain useful local information, while this experiment says that the
frozen 500-frame Camera tokens do not expose that information to a shared
decoder reliably enough.

The next source-level experiment should move supervision upstream: train the
last native Aggregator blocks together with the Camera Head so that a 500-frame
forward pass produces Camera tokens that preserve the locally reliable motion
evidence. It should retain the same matched GT-only control, strict long-only
inference, and locked-replay gates. Simply training this Camera Head longer is
not justified by these results.

## Verification

- Local tests: 30 long–short tests passed.
- Compatibility tests: 64 VRFM tests passed (one Windows symlink skip) and 55
  selector tests passed.
- H20 tests: all 30 long–short tests passed before launch.
- Independent H20 verification was run twice; the completion hash was
  identical both times.
- The first report attempt failed because a stage marker was included in a JSON
  glob. The exception is preserved under `diagnostics/report_attempt1.err.log`;
  a regression test was added, only the report stage was rerun, and no training
  artifact was changed or recomputed.
