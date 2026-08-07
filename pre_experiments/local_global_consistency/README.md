# Local-Global Trajectory Utilities

This package supplies reusable 100-frame window construction, prediction-only
similarity alignment, trajectory fusion, and pose metrics for the translation
refiner. It is no longer an active experiment entry point on this branch.

Keep prediction-only transforms separate from GT-derived training labels. Evaluation
aligns predictions to raw GT; GT arrays are never overwritten. New refiner-specific
logic belongs in `pre_experiments/camera_refiner_training/` rather than this package.
