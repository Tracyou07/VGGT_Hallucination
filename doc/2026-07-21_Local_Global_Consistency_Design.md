# Round 2 Local-Global Consistency Design

## Objective

Round 2A tests whether prediction-only disagreement between overlapping local
windows and a 500-frame global run can detect long-context Camera
representation failures. It replaces the original iteration-selection Round 2,
which was not activated because Round 1 found no useful scene-dependent Camera
iteration choice.

The experiment does not correct tokens, optimize poses, or train a model. GT is
used only after scoring to evaluate whether a score predicts increased aligned
pose error. Every GT trajectory remains raw; every metric containing a
prediction uses the appropriate aligned prediction.

## Fixed Protocol

Use the Round 1.5 four-scene panel:

- observed failures: `scene0000_00`, `scene0691_00`;
- stable controls: `scene0013_02`, `scene0029_01`.

For each scene, reuse the exact ordered 500 frame IDs from the published Round
1.5 artifact. Run nine local windows of 100 frames with stride 50, starting at
indices `0, 50, ..., 400`. Internal frames have two local observations; the
first and last 50 frames have one and are marked low-confidence for local-local
analysis. Camera Head iterations remain fixed at four. The model is frozen and
only Camera output and normalized Camera Tokens are requested.

## Prediction-Only Scores

For a shared frame, Camera Tokens from two local windows are compared directly
with cosine distance because they occupy the same learned feature space.
Local-local token disagreement measures whether a local reference is stable.
Global-local token disagreement is the median distance between the published
global token and its available local observations.

Absolute poses from different windows cannot be compared directly. For every
overlap, estimate an orientation-preserving Sim(3) using only the two predicted
camera trajectories, transform one prediction into the other's coordinate
system, and report translation and rotation residuals. Apply the same
prediction-to-prediction alignment between each local window and the matching
segment of the global trajectory. No GT participates in these scores.

The primary per-frame score candidates are:

- local-local token cosine distance;
- local-local aligned pose disagreement;
- global-local token cosine distance;
- global-local aligned pose disagreement;
- a reliability-gated global-local score that is valid only when local-local
  disagreement is below a threshold estimated from stable controls.

The first version does not combine these values into a manually weighted scalar.
It evaluates each candidate separately to avoid hiding which signal works.

## Evaluation With Raw GT

For evaluation only, independently align every global or local predicted
trajectory to the corresponding raw GT frames. Define per-frame degradation as
global aligned error minus the median local aligned error. This label never
enters score computation.

Report Pearson and Spearman correlation, score/error quartiles, and scene-level
means. Because frames are temporally correlated and there are only four scenes,
do not claim dataset-level statistical generalization. A candidate advances if:

1. its direction is consistent in both observed failure scenes;
2. failure-scene scores separate from both stable controls;
3. high-score frames have larger aligned error growth than low-score frames;
4. local-local reliability filtering strengthens rather than reverses the
   relationship.

## Data-Pair Gate

Dataset selection, splitting, versioning, and pair-release rules are defined in
`doc/2026-07-21_Local_Global_Dataset_Construction_Design.md`.

Round 2 does not yet publish `global -> local` training pairs. A frame becomes a
candidate using only prediction-side conditions: two local windows agree and
the global prediction differs. With two local observations, choose the token
from the window where the frame is farther from a boundary; do not average
tokens or call either token GT. Raw GT validates only at the aggregate protocol
level that this prediction-only rule usually selects a better local result; GT
must not accept or reject individual training pairs.

Before constructing a dataset, Round 3 must assemble these selected local tokens
into a sequence and show that frozen-Camera-Head replacement or blending improves
the trajectory. This prevents training a corrector toward a locally consistent
but unusable latent target.

## Components And Artifacts

Create `pre_experiments/local_global_consistency/` for deterministic windows,
prediction-only alignment, score computation, analysis, and a CLI. Reuse
ScanNet loading, raw-GT pose metrics, local checkpoint loading, and atomic
metadata helpers from existing pre-experiment packages. Do not change VGGT's
default forward API.

Raw local Camera Tokens remain in the external AutoDL run directory for later
data-pair work. Git publishing permits only compact CSV/JSON summaries and
per-frame scalar rows under
`results/local_global_consistency/<run_id>/`; images, checkpoints, point clouds,
and high-dimensional token arrays are excluded.

Required outputs are run metadata, window definitions, local-local and
global-local per-frame rows, scene summaries, GT-based validation tables, and a
completion record. A one-scene smoke run must remain distinguishable from the
fixed four-scene protocol.

## Reproduction And Tests

The AutoDL runner reuses the existing `vggt` conda environment, official local
checkpoint, processed ScanNet data, and published Round 1.5 global artifacts.
It performs no downloads or installation. CPU unit tests cover window coverage,
frame matching, prediction-only Sim(3), score gating, GT separation, output
contracts, and exporter whitelists without requiring CUDA, weights, or ScanNet.
