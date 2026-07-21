# Camera Context Consistency Diagnosis

Round 1.5 tests whether the same ScanNet frame receives a different Camera
representation or pose when VGGT sees a longer context. It fixes Camera Head
iterations at 4 and compares nested selections at 25, 50, 100, 200, and 500
frames.

The default four-scene panel contains two observed failures
(`scene0000_00`, `scene0691_00`) and two stable controls (`scene0013_02`,
`scene0029_01`). The model pass remains camera-only: Depth, Point, and Track
Heads are disabled so 500-frame memory stays comparable to Round 1.

## AutoDL Run

Use the existing `vggt` conda environment, checkpoint, and processed ScanNet
data:

```bash
git switch camera-iteration-preexperiment
bash scripts/autodl/run_camera_context.sh
```

The GPU is required only for the VGGT forward passes. The script then runs the
matched-frame analysis on CPU. Override `FRAME_COUNTS`, `SCENE_LIST`,
`SCENE_LIMIT`, or `RESULT_DIR` through environment variables. Iterations are
intentionally fixed at 4.

For a smoke test:

```bash
SCENE_LIMIT=1 FRAME_COUNTS="25 50" \
  RESULT_DIR=/root/autodl-tmp/camera_context/smoke \
  bash scripts/autodl/run_camera_context.sh
```

## Artifacts

Each selection adds `context_diagnostics.npz` with:

- `frame_ids` and final `normalized_camera_tokens`;
- `pred_c2w_raw` and independently `pred_c2w_aligned`;
- `gt_c2w_raw` only, never aligned GT;
- per-frame aligned translation/rotation errors;
- iteration-4 `delta_norm` and the fitted Sim(3) diagnostic.

At the run root, `context_per_frame.csv` contains shared-frame changes for
every ordered context pair. `context_summary.{csv,json}` reports token cosine
drift, pairwise-affinity drift, aligned pose/error changes, delta changes, and
their correlations. Raw token MSE is not treated as a physical geometry loss.

To rerun only the CPU analysis:

```bash
python -m pre_experiments.camera_context.analyze \
  --run-dir /root/autodl-tmp/camera_context/results/<run_id>
```

Publish the validated numeric artifacts with the existing exporter:

```bash
python scripts/autodl/camera_iteration/export_numeric_results.py \
  --source /root/autodl-tmp/camera_context/results/<run_id> \
  --destination-root results/camera_context
```

For a context run, the exporter includes the compact diagnostics and analysis
tables only when `run_metadata.json` declares `save_context_diagnostics=true`.
The 50 MiB per-file limit and exact NPZ member whitelist still apply.
