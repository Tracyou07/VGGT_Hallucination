# VGGT Camera Head Amplification Pre-experiment

The `camera-head-amplification-preexperiment` branch implements Round 1.6. It
replays the frozen four-block Camera Head from published Round 1.5 Camera
Tokens to test whether the head amplifies upstream representation drift.

## AutoDL Run

This replay does not load ScanNet images or rerun the Aggregator. It expects
the existing `vggt` conda environment, VGGT-1B checkpoint, and the committed
Round 1.5 numeric input:

```bash
git switch camera-head-amplification-preexperiment
bash scripts/autodl/run_camera_head_amplification.sh
```

Defaults are:

- Input: `results/camera_context/911b598_f4577f584448`
- Checkpoint: `/root/autodl-tmp/ckpt/VGGT-1B`
- Output: `/root/autodl-tmp/camera_head_amplification/results`
- Conda environment: `vggt`

Use `SCENE_LIMIT=1` for a smoke run. `SOURCE_RUN_DIR`, `CKPT_DIR`,
`RESULT_DIR`, `DEVICE`, and replay tolerances can be overridden through
environment variables.

## Publish and Verify

Only scalar CSV/JSON artifacts are publishable:

```bash
python scripts/autodl/camera_head_amplification/export_numeric_results.py \
  --source /root/autodl-tmp/camera_head_amplification/results/<run_id>
python -m unittest discover -s tests
```

Round 1.6 implementation is under
`pre_experiments/camera_head_amplification/`. Shared checkpoint, pose, and
artifact-contract helpers are the only retained files under
`pre_experiments/camera_iteration/`. Detailed method documentation remains in
`doc/`; repository-wide research guides remain on `main`.
