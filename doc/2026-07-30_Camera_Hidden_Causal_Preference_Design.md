# Camera Hidden Causal Preference Design

## Question

The existing attribution experiment shows which Camera Head hidden units are
stable and important under zero ablation. It does not tell us whether a unit
primarily affects translation, rotation, or FoV after all four refinement
iterations. This experiment builds an end-to-end causal preference atlas for
all `(iteration, hidden_index)` positions.

The result is diagnostic, not yet a correction rule. A high translation
preference means that a local perturbation mainly changes the final camera
centers; it does not prove that replacing that unit with a local-context value
will improve GT error.

## Efficient Causal Measurement

At each refinement iteration, the 1024-dimensional post-GELU hidden vector is
mapped to the 9-dimensional pose delta by the shared linear `fc2` layer. A
hidden-unit perturbation therefore enters the remaining Camera Head through
one 9D direction: the corresponding `fc2` weight column.

Instead of replaying two trajectories for every one of 4096 positions, the
experiment:

1. Replays the unmodified Camera Head and records each unit's RMS activation.
2. Adds positive and negative perturbations to each of the nine pose-delta
   basis directions at each of four iterations.
3. Uses centered finite differences to estimate the final-output Jacobian.
4. Projects every `fc2` column, scaled by that unit's RMS activation, through
   the Jacobian.

This reduces one scene from 8192 unit-specific perturbation samples to 72
basis samples. A small deterministic set of direct hidden perturbations checks
the first-order projection against actual forward passes.

## Output Effects

Each position receives three non-negative, trajectory-level effects:

- **Translation:** mean norm of the final camera-center derivative.
- **Rotation:** mean local angular speed of the final camera rotation, in
  degrees per one RMS hidden activation.
- **FoV:** mean norm of the final activated FoV derivative.

Perturbations are applied to a unit in every frame of the sequence. The atlas
therefore measures channel-level sequence effects, not frame-specific effects.

## Calibration and Holdout

The ten frozen calibration scenes fit one 90th-percentile scale per output
group, because meters, degrees, and FoV values are not directly comparable.
Normalized effects are divided by these frozen scales. The three normalized
effects are then divided by their sum to form translation, rotation, and FoV
preferences.

The forty holdout scenes reuse the calibration scales without refitting.
Validation reports full-rank Spearman correlation, top-64 overlap, preferred
group agreement, and direct-projection error. Ranking and preference remain
prediction-only; GT is not used by this experiment.

## Artifacts

Per-scene Jacobian projections remain as NPZ files under
`/root/autodl-tmp/camera_hidden_causal_preference/`. Formal export contains
only authenticated numeric CSV/JSON files:

- `per_position.csv`: one row for each of the 4096 positions.
- `direct_checks.csv`: projected versus directly measured spot checks.
- `frozen_causal_normalization.json`: calibration scales and reference atlas.
- `summary.json`, `run_metadata.json`, and `complete.json`.

The next correction experiment may join `per_position.csv` with the existing
local-global drift table by `(iteration, unit)`. That join is deliberately not
used to fit this causal atlas.
