# Camera Refiner Dependencies

This branch retains one self-contained experiment package:

- `camera_refiner_training/` for data loading, model, losses, checkpoints, training,
  prediction-only windowing, geometry, sampling, inference, and evaluation.

Frozen translation-unit files and data-construction shards are external inputs. Do
not copy their predecessor study implementations back into this branch.
