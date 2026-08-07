# Camera Refiner Data Construction

This package area combines retained local-global and hidden-attribution
infrastructure with the implemented multiscale data-construction pipeline in
`camera_refiner_data_construction/`.

Key modules:

- `protocol.py` defines the fixed 100/200/300 scales, candidate identities,
  most-interior window selection, and hidden mixtures.
- `run_study.py` authenticates all source runs and executes resumable Camera
  Head replay for smoke, calibration, or holdout.
- `artifacts.py` owns the exact external `scene_shard.npz` schema.
- `analyze.py` performs scene-equal aggregation, calibration freezing, and
  frozen holdout evaluation.
- `dataset.py` creates and validates portable, checksum-authenticated manifests
  without copying large shards into Git.
- `co3d_download.py` builds the resumable 2050-sequence CO3Dv2 RGB and GT-pose
  subset directly under the AutoDL dataset root.

Metric contract:

- predictions are aligned before any prediction-versus-GT metric;
- GT remains raw;
- calibration selects candidates;
- holdout evaluates a frozen decision and never refits it.

Historical result publication and experiment launchers are intentionally not
part of this worktree.
