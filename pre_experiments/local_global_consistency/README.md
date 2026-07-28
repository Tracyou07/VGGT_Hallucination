# ScanNet-50 Local-Global Consistency

This training-free study asks whether context-dependent Camera Token and Camera
Pose disagreement predicts VGGT long-context degradation.

## Protocol

- Global reference: one explicit 50-scene Camera Context run with 500 frames,
  four Camera Head iterations, `nested_uniform` sampling, and `pad`
  preprocessing.
- Local inference: nine 100-frame windows per scene at stride 50.
- Calibration: 10 scenes and 90 windows fit three prediction-only P95
  Local-Local reliability thresholds.
- Holdout: 40 disjoint scenes and 360 windows consume those thresholds without
  refitting.
- Statistics: frames are reduced to scene summaries before 10,000 deterministic
  bootstrap resamples with seed 33.

The split is selected from raw GT motion statistics before new VGGT outcomes
are inspected. Detection tables contain no GT values. Validation separately
aligns predictions to `gt_c2w_raw`; GT remains unchanged.

## Modules

- `split.py`: deterministic raw-motion split and manifest validation.
- `context_source.py`: exact 50-scene source preflight.
- `run_study.py`: resumable calibration or holdout local inference.
- `metrics.py`: prediction-only disagreement and separate GT labels.
- `thresholds.py`: authenticated frozen calibration thresholds.
- `aggregate.py`: per-scene summaries and scene bootstrap intervals.
- `analyze.py`: strict calibration and holdout output modes.
- `visualize.py`: PNG diagnostics derived only from completed CSV/JSON.

## Outputs

Calibration produces `frozen_reliability_thresholds.json`, frame-level
prediction and GT-validation tables, and calibration summaries. Holdout
produces separate frame tables, per-scene summaries, aggregate confidence
intervals, and `holdout_complete.json`. PNGs are written under
`visualizations/`; neither figures nor raw `window_diagnostics.npz` files are
eligible for repository export.

Use `scripts/autodl/run_scannet50_local_global.sh` for AutoDL execution and
`scripts/autodl/local_global_consistency/export_numeric_results.py` for strict
numeric publication. The top-level README contains exact commands.
