# Round 2A Local-Global Consistency

This pre-experiment tests whether prediction-only disagreement between local
and 500-frame global inference identifies long-context camera degradation. It
does not train VGGT, construct a dataset, or modify model weights.

## Fixed Protocol

- Reuse the exact frame IDs and global artifacts from Round 1.5 run
  `911b598_f4577f584448`.
- Run frozen camera-only VGGT with four Camera Head iterations.
- Split each 500-frame sequence into nine 100-frame windows at stride 50.
- Compare Camera Tokens directly and align predicted trajectories to each
  other with Sim(3) before measuring pose disagreement.
- Fit local-local reliability thresholds only from stable control scenes
  `scene0013_02` and `scene0029_01`.

Detection tables contain no GT-derived score. Validation independently aligns
global and local predictions to `gt_c2w_raw`; GT is never aligned or replaced.

## AutoDL Runs

The runner assumes the `vggt` conda environment, processed ScanNet scenes,
official checkpoint, and published Round 1.5 artifacts already exist.

```bash
# One-scene pipeline smoke; thresholds remain intentionally unfitted.
SCENE_LIMIT=1 bash scripts/autodl/run_local_global_consistency.sh

# Fixed four-scene formal pre-experiment.
bash scripts/autodl/run_local_global_consistency.sh
```

Raw per-window `window_diagnostics.npz` files stay under
`/root/autodl-tmp/local_global_consistency/results/<run_id>/`. After reviewing
the analysis, publish only scalar CSV/JSON outputs:

```bash
python scripts/autodl/local_global_consistency/export_numeric_results.py \
  --source /root/autodl-tmp/local_global_consistency/results/<run_id>
```

`prediction_scores_per_frame.csv` is the deployable prediction-only signal.
`gt_validation_per_frame.csv` supplies separate aligned-prediction versus raw-GT
labels. `local_global_summary.csv` reports score/error-growth correlations;
these remain pre-experiment evidence until the formal multi-scene run finishes.
