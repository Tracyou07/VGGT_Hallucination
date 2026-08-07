# VGGT Translation Refiner

This branch develops a translation-only residual refiner for long-context VGGT
camera trajectories. The model uses overlapping 100-frame local predictions and
translation-preferred Camera Head features to correct camera centers from a
500-frame global prediction. Global VGGT rotations remain unchanged.

The approved experiment specification is
[`doc/2026-08-07_Translation_Only_Residual_Diffusion_Design.md`](doc/2026-08-07_Translation_Only_Residual_Diffusion_Design.md).

## Repository Layout

- `vggt/`: upstream VGGT with opt-in Camera Head tracing.
- `pre_experiments/common/`: shared model, ScanNet, metadata, and pose helpers.
- `pre_experiments/local_global_consistency/`: reusable window, alignment,
  trajectory, and metric primitives.
- `pre_experiments/camera_hidden_state_attribution/`: reusable hidden trace and
  frozen-unit utilities inherited from the attribution study.
- `scripts/autodl/scannet/`: ScanNet `.sens` extraction support.
- `paper/`: the DiffusionSfM reference paper.

Training implementation and its focused tests will be added only after the design is
reviewed and an implementation plan is approved. Historical experiment documents,
logs, outputs, and run-only entry points belong to their original branches and are
intentionally absent here.

## Environment

AutoDL is expected to provide the existing `vggt` Conda environment, CUDA-compatible
PyTorch, the VGGT checkpoint, and processed ScanNet data under
`/root/autodl-tmp`. This branch must not create environments or download weights.

Install the checkout into the active environment with:

```bash
pip install -e .
```

Run the retained CPU regression tests with:

```bash
python -m unittest discover -s tests -v
```

Large training caches and checkpoints stay under
`/root/autodl-tmp/results/camera_refiner_training/` and must not be committed.
