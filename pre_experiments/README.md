# CO3Dv2 Data Construction

`camera_refiner_data_construction/co3d_download.py` owns category filtering,
deterministic sequence selection, resumable archive downloads, RGB-only
extraction, and the authenticated download manifest.

`co3d_manifest.py` converts that selection into ordered, sequence-disjoint clips.
`build_co3d_cache.py` runs the frozen camera-only VGGT path on long clips and
short windows. `geometry.py` performs prediction-only Sim(3) alignment, while
`cache_schema.py` writes authenticated `full_hidden_sequence_refiner` shards.
Generated data lives under `/root/autodl-tmp/results`, never in the checkout.

Multiscale hidden-state replay, ScanNet processing, and their analysis code
live on the `016-camera-refiner-multiscale` branch.
