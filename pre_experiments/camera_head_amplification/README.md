# Camera Head Amplification Replay

Round 1.6 replays the frozen four-block VGGT Camera Head from normalized Camera
Tokens saved by Round 1.5. It does not load images, ScanNet `.sens` files, the
image encoder, or the Aggregator.

## Comparisons

- `H200(Z200)` versus `H200(Z500_shared)` isolates propagation of Aggregator
  token drift through a fixed-length Camera Head.
- `H200(Z500_shared)` versus `H500(Z500)_shared` holds shared-frame token values
  fixed and measures the effect of 300 additional Camera Head context tokens.
- `H500(Z500)` reproduces the actual long-context prediction.

Hooks capture the AdaLN-modulated trunk input, every transformer block, trunk
normalization, 9D delta, and accumulated raw 9D pose for each refinement
iteration. Only scalar per-frame drifts and aggregate metrics are written.

## AutoDL

The branch already contains the published Round 1.5 input. With the existing
`vggt` conda environment and official checkpoint:

```bash
git switch camera-head-amplification-preexperiment
bash scripts/autodl/run_camera_head_amplification.sh
```

For a one-scene smoke test:

```bash
SCENE_LIMIT=1 bash scripts/autodl/run_camera_head_amplification.sh
```

Override `SOURCE_RUN_DIR`, `CKPT_DIR`, `RESULT_DIR`, or `DEVICE` when needed.
The script fails unless both 200- and 500-frame baseline replays reproduce the
saved activated 9D pose encodings within `BASELINE_ATOL`/`BASELINE_RTOL`.

Publish a completed scalar-only run with:

```bash
python scripts/autodl/camera_head_amplification/export_numeric_results.py \
  --source /root/autodl-tmp/camera_head_amplification/results/<run_id>
git add results/camera_head_amplification/<run_id>
```

Prediction metrics are independently Sim(3)-aligned to raw GT. GT is never
aligned or replaced by an aligned copy.
