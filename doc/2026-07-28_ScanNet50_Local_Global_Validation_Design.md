# ScanNet-50 Local-Global Validation Design

## Goal

Extend Round 2A from four observed scenes to a leakage-controlled ScanNet-50
validation. The Camera Context branch produces one compatible global artifact
per scene under a requested 500-frame protocol. All split handling,
local-window inference, threshold fitting, and evaluation belong to the
`local-global-consistency-preexperiment` worktree.

## Leakage-Controlled Stratified Split

The calibration set always contains the four previously observed scenes:

```text
scene0000_00  scene0013_02  scene0029_01  scene0691_00
```

These currently represent two easy, one medium, and one difficult observed
case. Select the other six calibration scenes before inspecting any new VGGT
predictions. For each of the remaining 46 scenes, use the exact deterministic
source-frame selection and raw GT poses to calculate:

- cumulative translation;
- cumulative geodesic rotation;
- P95 consecutive-frame translation;
- P95 consecutive-frame rotation.

Convert each measure to a percentile rank across the 46 candidates and average
the four ranks into a raw-motion difficulty proxy. Split candidates into
equal-size easy, medium, and hard thirds. Select two scenes per third using a
stable SHA-256 ordering keyed by seed 33. This produces a ten-scene calibration
set with an intended approximate 4/3/3 easy/medium/hard composition without
using predictions or GT error outcomes from the holdout candidates.

The holdout set is the other 40 entries in
`configs/fastvggt_scannet50.txt`. Generate and commit a structured split
manifest before local inference. It records the raw proxy components, strata,
selection seed, selected calibration scenes, and ordered holdout scenes. Tests
require 50 unique scenes, a 10/40 partition, no overlap, exact equality with the
ScanNet-50 list, two newly selected scenes per stratum, and no prediction input
to split construction.

## Inputs

The workflow requires an explicit `SOURCE_RUN_DIR` containing:

- metadata declaring all 50 scenes and the requested 500-frame,
  four-iteration, `nested_uniform`, `pad` protocol;
- one `frames_500/context_diagnostics.npz` per scene;
- processed ScanNet RGB frames and raw GT poses;
- the existing VGGT checkpoint.

The runner must not select the newest context directory automatically. Missing
or incompatible metadata and artifacts are hard failures. Each source artifact
must contain exactly 500 frames except `scene0150_00`, which must contain
exactly its 430 available frames; no other short sequence is accepted.

## Two-Stage Workflow

### Calibration

Run length-100 local windows at stride 50. Normal scenes produce nine windows;
`scene0150_00` produces eight, with the final tail-coverage window `[330, 430)`.
Analyze prediction-only local-local and local-global scores, then fit the three
P95 reliability thresholds from all ten calibration scenes. Write a frozen
threshold artifact containing metric names, values, contributing sample counts,
split digest, source run ID, calibration run ID, and code commit.

### Holdout Validation

Run the same local-window protocol on the 40 holdout scenes. The analyzer must
receive the frozen threshold artifact explicitly. Holdout mode is not allowed
to fit or modify thresholds. It writes per-frame prediction scores, separate
raw-GT validation labels, per-scene summaries, and aggregate 40-scene
statistics.

Calibration and holdout use separate run IDs and output directories. A wrapper
may execute both stages sequentially, but the holdout stage only starts after a
complete calibration threshold artifact exists.

## Metric Contract

Prediction-only detection scores never use GT. Prediction-to-prediction pose
comparisons may use Sim(3) alignment. Any metric involving prediction and GT
uses aligned predictions against raw GT. GT arrays are never aligned,
overwritten, or exported under an ambiguous name.

Primary validation reports include:

- mean and median global-minus-local translation and rotation growth;
- fraction of frames with positive growth;
- Pearson and Spearman association between frozen prediction scores and GT
  growth;
- reliable-frame coverage under frozen thresholds;
- per-scene distributions and aggregate 95% confidence intervals computed by
  10,000 deterministic bootstrap resamples over the 40 scene-level summaries
  with seed 33.

Calibration metrics are reported separately and never pooled into holdout
claims.

## Resume and Outputs

Window artifacts retain the existing atomic NPZ-then-completion-marker
protocol. Resume checks validate run ID, scene, window boundaries, and frame
IDs. The expected workload is 449 windows. Depending on whether
`scene0150_00` belongs to calibration or holdout, the partition counts are
89/360 or 90/359, respectively.

Raw window NPZ files remain outside Git. The numeric exporter permits only
completed CSV/JSON summaries and manifests, rejects high-dimensional arrays,
and records the threshold artifact used by holdout evaluation.

## Verification

CPU tests cover split identity, source-run validation, calibration-only fitting,
holdout threshold immutability, missing-window rejection, resume behavior,
metric naming, and exporter allowlists. GPU smoke tests run one calibration
scene and one holdout scene before launching the full protocol.
