# Camera Refiner Data Construction Design

## Objective

This worktree determines how 100-, 200-, and 300-frame local contexts should contribute to camera-hidden refinement, then exports the evidence needed to train and evaluate a camera refiner. It inherits the frozen hidden extraction and Camera Head replay pipeline from `camera-hidden-state-attribution-preexperiment`.

The work has two ordered stages:

1. Characterize each local context scale and candidate multiscale mixtures.
2. Freeze the useful candidates and export a reusable, split-aware training dataset.

The worktree does not train the final refiner. Training remains owned by `camera-refiner-training`.

## Experimental Variables

For every target frame, extract the global hidden state `h500` and local hidden states `h100`, `h200`, and `h300`. Each local scale uses 50% overlapping windows plus a tail-anchored final window. If a frame occurs in multiple windows at one scale, select the window where it is farthest from either boundary; retain the earlier window on a tie.

Keep three variables separate:

- `L`: local context length, one of 100, 200, or 300 frames.
- `beta`: multiscale mixture weights. The three weights are non-negative and sum to one.
- `alpha`: total interpolation strength from the global hidden state toward the local mixture.

The refined hidden state is computed as:

```text
h_local = beta100*h100 + beta200*h200 + beta300*h300
h_refined = h500 + alpha*(h_local - h500)
```

Every candidate must be replayed through the actual Camera Head. Hidden distance alone is diagnostic evidence, not an outcome metric.

## Scale Study

Run the scale study in two phases to avoid an unnecessary full grid.

First, evaluate pure-scale mixtures `(1,0,0)`, `(0,1,0)`, and `(0,0,1)` using candidate alphas `{0.01, 0.02, 0.05, 0.10}`. Record hidden displacement, translation, rotation, FOV, ATE, per-frame improvement, and scene-level stability.

Second, retain only useful scales and evaluate a small frozen set of mixtures. Initial candidates are `(0.5,0.5,0)`, `(0,0.5,0.5)`, and `(0.2,0.3,0.5)`. Calibration may eliminate candidates but must not add post-hoc candidates after holdout evaluation begins.

Primary selection uses scene-level aligned translation error. Rotation and FOV are safety metrics. A candidate is rejected if its mean gain is driven by a single scene, its confidence interval is incompatible with improvement, or it causes a material rotation/FOV regression.

## Dataset Schema

Store one logical record per scene and target frame with:

- scene ID, frame ID, split, source checkpoint, and code commit;
- global and selected local window boundaries for every scale;
- `h500`, `h100`, `h200`, and `h300` at the frozen intervention site;
- prediction-only global-local and local-local consistency features;
- candidate `alpha`, `beta`, and actual hidden displacement;
- Camera Head translation, rotation, and FOV outputs;
- raw GT camera data;
- aligned prediction metrics computed against raw GT;
- candidate rank, best candidate, improvement magnitude, and validity flags.

Large tensors are stored as sharded NPZ or another explicitly versioned binary format outside Git. Git tracks the schema, manifests, numeric summaries, checksums, and reproduction scripts. Derived preference pairs are generated from the canonical candidate records rather than stored as the only representation.

## Metric and Split Contract

Any metric containing a prediction must use the aligned prediction. Ground truth must always remain raw; aligned GT is forbidden.

Scene roles are immutable in a frozen manifest:

- calibration selects scales, mixtures, features, and thresholds;
- holdout evaluates a frozen policy and never supplies labels for selection;
- future train scenes provide refiner supervision;
- future validation scenes select training checkpoints and hyperparameters.

Existing holdout scenes must not enter refiner training. Dataset generation must fail on duplicate scene IDs across splits, missing scale coverage, incomplete Camera Head outputs, non-finite tensors, or provenance mismatches.

## Reproducibility and Outputs

Every run writes an immutable metadata file containing scene manifest digest, checkpoint digest, source commit, window parameters, intervention location, candidate grid, alignment implementation, and random seeds. Interrupted runs resume only when metadata and artifact checksums match.

Expected tracked outputs are:

- a frozen scale-study configuration;
- calibration and holdout numeric summaries;
- the versioned dataset schema and split manifests;
- a dataset validator and compact inspection report;
- AutoDL commands for extraction, Camera Head replay, export, and verification.

## Acceptance Criteria

The worktree is complete when:

1. The 100/200/300 single-scale comparison is reproducible.
2. Any selected multiscale mixture beats or meaningfully complements the best fixed scale on calibration without violating safety metrics.
3. The frozen decision is evaluated once on untouched holdout scenes.
4. Dataset shards pass schema, checksum, split-leakage, provenance, and finite-value validation.
5. A fresh machine can reproduce a small dataset shard and its numeric summary from documented commands.
