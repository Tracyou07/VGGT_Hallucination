# Adaptive Alpha Calibration Analysis

## Oracle Alpha Definition

The oracle alpha is the candidate in `{0.01, 0.02, 0.05}` that minimizes the final aligned translation error for each calibration scene. It is an empirical optimum over this discrete candidate set, not the theoretical optimum over a continuous alpha range.

## First Ridge Selector

The first prediction-only ridge selector did not pass the calibration acceptance threshold. Under leave-one-out cross-validation (LOOCV), its mean translation-error delta was `-0.0002039647`, compared with `-0.0001866527` for the fixed `alpha=0.02` policy. However, the paired scene outcome was only one win, seven ties, and two losses. The paired bootstrap 95% confidence interval was `[-0.00031313, +0.00022090]`. After removing `scene0000_00`, adaptive selection was worse than the fixed policy by `+0.00010713`. The selector matched the oracle alpha in only 2 of 10 scenes.

## Scene Difficulty

Raw-motion difficulty was available for six calibration scenes and did not support the hypothesis that harder scenes require larger alpha. Its Spearman correlation with oracle alpha was `-0.621` (`p=0.188`). Raw trajectory motion therefore should not be used directly as the adaptive gate condition.

## Relative Consistency Signal

The ratio of global-local translation discrepancy to local-local translation inconsistency (GL/LL) showed a stronger relationship with oracle alpha:

- Spearman correlation: `0.631` (`p=0.0504`)
- Pearson correlation: `0.757` (`p=0.0113`)
- Mean GL/LL ratio for oracle `alpha=0.01`: `2.11`
- Mean GL/LL ratio for oracle `alpha=0.05`: `5.58`

This supports the interpretation that a larger gate may be appropriate when short-window predictions agree with one another but jointly disagree with the long-context prediction. A large long-short discrepancy alone is insufficient if the short windows are also mutually inconsistent.

## Status and Next Step

The GL/LL relationship is a post-calibration hypothesis and must not be treated as validated evidence. The feature definition and selector must be frozen before evaluation on the untouched holdout set.

The next selector should use a stabilized log-ratio feature rather than relying only on separate raw medians:

```text
log((global_local_translation + epsilon)
    / (local_local_translation + epsilon))
```
