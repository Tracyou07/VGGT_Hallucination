# Camera Refiner Dependencies

This branch retains only the predecessor modules needed by the translation refiner:

- `common/` for model loading, ScanNet input, run contracts, and pose operations;
- `local_global_consistency/` for local windows, prediction-only alignment, trajectory
  fusion, and evaluation metrics;
- `camera_hidden_state_attribution/` for Camera Head replay and frozen-unit feature
  extraction.

New model, dataset, training, sampling, and evaluation code belongs in
`camera_refiner_training/`. Do not add new studies to the retained predecessor
packages.
