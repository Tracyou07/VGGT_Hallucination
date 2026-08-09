# Camera Refiner Data Construction

This branch builds a compact CO3Dv2 training subset for the camera refiner. It
contains no multiscale Camera Head experiments or ScanNet result pipeline;
those live on the `016-camera-refiner-multiscale` branch and worktree.

## Dataset Protocol

The fixed split contains the 41 PoseDiffusion training categories listed in
`configs/co3d_train41.txt`. The downloader targets 2050 sequences with a base quota
of 50 per category. Before downloading data archives it reads all category metadata
and measures eligible capacity. If a category has fewer than 50 valid sequences,
its shortfall is deterministically redistributed to categories with spare capacity,
preferentially those not downloaded yet. For example, `parkingmeter=47` causes
three later categories to receive 51 sequences each. Every selected sequence must
provide at least 50 RGB frames with finite GT rotation and translation annotations
and `viewpoint_quality_score >= 0.5`.

Only selected `images/` files and category-level `frame_annotations.jgz` and
`sequence_annotations.jgz` metadata are retained. Depth maps, masks, point
clouds, and completed large ZIP chunks are discarded.

## AutoDL Download

The AutoDL machine must already contain the `vggt` Conda environment. The
script does not create environments, install packages, or download weights.

```bash
cd /root/autodl-tmp/VGGT_Hallucination
git switch 021-camera-refiner-data-construction
git pull --ff-only origin 021-camera-refiner-data-construction

nohup bash scripts/autodl/camera_refiner_data_construction/download_co3d_2050.sh \
  > /root/autodl-tmp/co3d_2050.log 2>&1 &
tail -f /root/autodl-tmp/co3d_2050.log
```

The default output is `/root/autodl-tmp/datasets/co3dv2_2050`. Re-running the
same command resumes `.part` archives and validated category state. The final
selection, per-category quotas, and SHA-256 identity are stored in
`download_manifest.json` schema 2. Existing extracted sequences are retained if a
quota grows; only the additional sequences are selected.

## Build Training Shards

After `download_manifest.json` exists, run a one-sequence GPU smoke test:

```bash
bash scripts/autodl/camera_refiner_data_construction/build_co3d_training_data.sh smoke
```

Then start or resume the complete cache:

```bash
nohup bash scripts/autodl/camera_refiner_data_construction/build_co3d_training_data.sh full \
  > /root/autodl-tmp/co3d_training_data.log 2>&1 &
tail -f /root/autodl-tmp/co3d_training_data.log
```

The default protocol uses one ordered 100-frame clip per sequence. VGGT runs once
on all 100 frames and on 50-frame windows with stride 25. Sequences shorter than
100 valid frames are recorded in `clip_manifest.json` as `insufficient_frames`;
frames are never repeated or looped. Override `CLIP_LENGTH`, `SHORT_WINDOW`,
`SHORT_STRIDE`, or `MAX_CLIPS_PER_SEQUENCE` only for a separately named run.

Each shard stores final long-run Camera Head hidden state `[S,1024]`, normalized
camera tokens `[S,2048]`, raw and activated baseline pose, iteration diagnostics,
the aligned short-window consensus pose, every overlapping aligned short-pose
observation, and raw CO3D cameras. It never stores RGB tensors or short hidden
states. The extra short targets support the later diffusion-admission audit;
deterministic training consumes the consensus target.

Outputs are written under
`/root/autodl-tmp/results/camera_refiner_data_construction/co3d/<run_id>/`.
`manifest.json` is directly consumable by the full-hidden latent refiner. A
sequence is skipped on restart only when its completion marker, build digest,
shard checksum, and tensor contract all validate.

## Development

```bash
python -m unittest discover -s tests/camera_refiner_data_construction -v
python -m compileall -q pre_experiments
bash -n scripts/autodl/camera_refiner_data_construction/download_co3d_2050.sh
bash -n scripts/autodl/camera_refiner_data_construction/build_co3d_training_data.sh
```

Datasets, archives, checkpoints, generated results, and Python environments
must remain outside Git.
