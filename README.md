# VGGT Translation Refiner

This branch develops a translation-only residual refiner for long-context VGGT
camera trajectories. The model uses overlapping 100-frame local predictions and
translation-preferred Camera Head features to correct camera centers from a
500-frame global prediction. Global VGGT rotations remain unchanged.

The approved experiment specifications are:

- [`doc/2026-08-07_Translation_Only_Residual_Diffusion_Design.md`](doc/2026-08-07_Translation_Only_Residual_Diffusion_Design.md)
- [`doc/2026-08-07_CO3D_Pretraining_and_ScanNet_Finetuning_Design.md`](doc/2026-08-07_CO3D_Pretraining_and_ScanNet_Finetuning_Design.md)

## Repository Layout

- `vggt/`: upstream VGGT with opt-in Camera Head tracing.
- `pre_experiments/camera_refiner_training/`: dataset adapter, residual DiT,
  prediction-only windowing, geometry, diffusion, losses, checkpoints, training,
  inference, metrics, and visualization.
- `scripts/autodl/camera_refiner_training/`: one-command AutoDL entry points.
- `paper/`: frozen papers used to justify the implementation choices.
- `tmp/references/`: ignored, locally curated source snapshots for inspection.

## Environment

AutoDL is expected to provide the existing `vggt` Conda environment, CUDA-compatible
PyTorch, the VGGT checkpoint, and processed ScanNet data under
`/root/autodl-tmp`. This branch must not create environments or download weights.

Install the checkout into the active environment with:

```bash
pip install -e .
```

Run CPU tests, including an end-to-end train/infer smoke, with:

```bash
python -m unittest discover -s tests -v
```

Large training caches and checkpoints stay under
`/root/autodl-tmp/results/camera_refiner_training/` and must not be committed.

## AutoDL Training and Inference

Set the four external artifact paths, then run:

```bash
export DATASET_MANIFEST=/root/autodl-tmp/results/camera_refiner_data_construction/RUN_ID/dataset_manifest.json
export DATASET_ROOT=/root/autodl-tmp/results/camera_refiner_data_construction/RUN_ID
export LOCAL_RUN_DIR=/root/autodl-tmp/results/local_global_consistency/RUN_ID
export FROZEN_UNITS=/root/autodl-tmp/results/camera_hidden_state_attribution/RUN_ID/frozen_units.json
bash scripts/autodl/camera_refiner_training/train.sh
bash scripts/autodl/camera_refiner_training/infer.sh
```

Use `RESUME=1` for strict checkpoint continuation. Override defaults through
environment variables such as `EPOCHS`, `OUT_DIR`, `DEVICE`, and `MODEL_KIND`.
