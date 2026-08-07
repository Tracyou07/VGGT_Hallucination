# Camera Refiner Data Construction

This branch builds a compact CO3Dv2 training subset for the camera refiner. It
contains no multiscale Camera Head experiments or ScanNet result pipeline;
those live on the `01-camera-refiner-multiscale` branch and worktree.

## Dataset Protocol

The fixed split contains the 41 PoseDiffusion training categories listed in
`configs/co3d_train41.txt`. The downloader selects 50 sequences per category,
for 2050 sequences total. Every selected sequence must provide at least 50 RGB
frames with finite GT rotation and translation annotations and
`viewpoint_quality_score >= 0.5`.

Only selected `images/` files and category-level `frame_annotations.jgz` and
`sequence_annotations.jgz` metadata are retained. Depth maps, masks, point
clouds, and completed large ZIP chunks are discarded.

## AutoDL Download

The AutoDL machine must already contain the `vggt` Conda environment. The
script does not create environments, install packages, or download weights.

```bash
cd /root/autodl-tmp/VGGT_Hallucination
git switch 02-camera-refiner-data-construction
git pull --ff-only origin 02-camera-refiner-data-construction

nohup bash scripts/autodl/camera_refiner_data_construction/download_co3d_2050.sh \
  > /root/autodl-tmp/co3d_2050.log 2>&1 &
tail -f /root/autodl-tmp/co3d_2050.log
```

The default output is `/root/autodl-tmp/datasets/co3dv2_2050`. Re-running the
same command resumes `.part` archives and validated category state. The final
selection and its SHA-256 identity are stored in `download_manifest.json`.

## Development

```bash
python -m unittest discover -s tests/camera_refiner_data_construction -v
python -m compileall -q pre_experiments
bash -n scripts/autodl/camera_refiner_data_construction/download_co3d_2050.sh
```

Datasets, archives, checkpoints, generated results, and Python environments
must remain outside Git.
