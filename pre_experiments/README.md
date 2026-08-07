# CO3Dv2 Data Construction

`camera_refiner_data_construction/co3d_download.py` is the only experiment
module retained on this branch. It owns category filtering, GT-pose
validation, deterministic sequence selection, resumable archive downloads,
RGB-only extraction, per-category state, and the final authenticated manifest.

Multiscale hidden-state replay, ScanNet processing, and their analysis code
live on the `01-camera-refiner-multiscale` branch.
