# VGGT Camera Context Consistency Pre-experiment

The `camera-context-consistency-preexperiment` branch implements Round 1.5:
compare the same frame's Camera Token and aligned pose under nested context
lengths while fixing Camera Head iterations at 4.

## AutoDL Run

The runner expects the existing `vggt` conda environment, VGGT-1B checkpoint,
and processed ScanNet data:

```bash
git switch camera-context-consistency-preexperiment
bash scripts/autodl/run_camera_context.sh
```

Default external paths are:

- ScanNet: `/root/autodl-tmp/datasets/scannetv2`
- VGGT-1B: `/root/autodl-tmp/ckpt/VGGT-1B`
- Results: `/root/autodl-tmp/camera_context/results`
- Conda environment: `vggt`

The context runner delegates GPU inference to the shared camera-iteration
runner, then performs matched-frame analysis on CPU. To repair or extend the
authorized ScanNet subset:

```bash
SCANNET_TOS_ACCEPTED=1 bash scripts/autodl/prepare_scannet_camera_iteration.sh
```

For a smoke run:

```bash
SCENE_LIMIT=1 FRAME_COUNTS="25 50" \
  RESULT_DIR=/root/autodl-tmp/camera_context/smoke \
  bash scripts/autodl/run_camera_context.sh
```

## Results and Development

Published Round 1.5 numeric artifacts live under
`results/camera_context/911b598_f4577f584448/`. Regenerate analysis with:

```bash
python -m pre_experiments.camera_context.analyze \
  --run-dir /root/autodl-tmp/camera_context/results/<run_id>
python -m unittest discover -s tests
```

Core shared inference code is in `pre_experiments/camera_iteration/`; matched
context artifacts and metrics are in `pre_experiments/camera_context/`.
Repository-wide research guides remain on `main`.
