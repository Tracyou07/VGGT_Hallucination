# Camera Refiner Data Construction

This package area contains reusable infrastructure inherited from the completed
local-global and hidden-attribution studies. New multiscale extraction,
Camera Head replay, dataset schema, and export code belongs in
`camera_refiner_data_construction/`.

Metric contract:

- predictions are aligned before any prediction-versus-GT metric;
- GT remains raw;
- calibration selects candidates;
- holdout evaluates a frozen decision and never refits it.

Historical result publication and experiment launchers are intentionally not
part of this worktree.
