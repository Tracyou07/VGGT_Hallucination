# Camera Refiner Data Construction

This branch studies how 100-, 200-, and 300-frame local contexts affect the
frozen VGGT Camera Head hidden state, then exports split-aware evidence for
training a camera refiner. It does not train the final refiner.

## Scope

The pipeline:

1. extract frame-matched `h100`, `h200`, `h300`, and `h500` hidden states;
2. replay single-scale and frozen multiscale candidates through Camera Head;
3. compare candidates using aligned predictions against raw GT;
4. export versioned manifests, numeric summaries, and external tensor shards.

The exact protocol and acceptance criteria are defined in
`doc/2026-07-30_Camera_Refiner_Data_Construction_Design.md`.

## Repository Layout

- `vggt/`: frozen VGGT model plus opt-in Camera Head tracing hooks.
- `pre_experiments/common/`: shared artifact, model, ScanNet, and pose helpers.
- `pre_experiments/camera_hidden_state_attribution/`: retained hidden capture
  and Camera Head replay infrastructure.
- `pre_experiments/local_global_consistency/`: retained windowing, alignment,
  split, and context-source infrastructure.
- `configs/`: scene lists and immutable split manifests.
- `pre_experiments/camera_refiner_data_construction/`: multiscale protocol,
  strict scene shards, Camera Head replay, calibration, and dataset manifests.
- `scripts/autodl/camera_refiner_data_construction/`: resumable AutoDL runner
  and CPU-only dataset validator.
- `tests/camera_refiner_data_construction/`: focused CPU regression tests.

## AutoDL Study

The machine must already contain the `vggt` Conda environment, processed
ScanNet-50 data, VGGT checkpoint, frozen Camera Context run, and calibrated
hidden-unit manifest. The script activates the existing environment; it never
installs packages or downloads assets.

```bash
cd /root/autodl-tmp/VGGT_Hallucination
export RESULTS_ROOT=/root/autodl-tmp/results
export SOURCE_RUN_DIR="$RESULTS_ROOT/camera_context/results/d33d98b_309a9a586242"
REPLACEMENT_CALIBRATION_RUN="$(< "$RESULTS_ROOT/camera_hidden_replacement/state/calibration_run.txt")"
export FROZEN_UNITS="$REPLACEMENT_CALIBRATION_RUN/frozen_replacement.json"

test -d "$SOURCE_RUN_DIR"
test -f "$FROZEN_UNITS"

bash scripts/autodl/camera_refiner_data_construction/run_multiscale_study.sh smoke
bash scripts/autodl/camera_refiner_data_construction/run_multiscale_study.sh calibration

CALIBRATION_RUN="$(< "$RESULTS_ROOT/camera_refiner_data_construction/pointers/calibration/multiscale.txt")"
export FROZEN_POLICY="$CALIBRATION_RUN/frozen_candidate_policy.json"
bash scripts/autodl/camera_refiner_data_construction/run_multiscale_study.sh holdout
```

Set `CANDIDATE_FAMILY=all` before calibration to include the three predefined
mixtures in addition to pure scales. Keep the same value when resuming. Each
stage first creates exact `100/50`, `200/100`, and `300/150` local-window runs,
then replays frame-matched hidden candidates. Re-running a command resumes the
same immutable run through pointer files under
`/root/autodl-tmp/results/camera_refiner_data_construction/pointers/`.

Calibration writes `candidate_summary.{csv,json}` and, only when all robustness
gates pass, `frozen_candidate_policy.json`. Holdout accepts exactly that one
authenticated candidate and cannot search or refit candidates. Large
`scene_shard.npz` files remain under `/root/autodl-tmp`; do not commit them.

Validate a portable external dataset manifest with:

```bash
python scripts/autodl/camera_refiner_data_construction/validate_dataset.py \
  --manifest /root/autodl-tmp/refiner_dataset/manifest.json \
  --dataset-root /root/autodl-tmp/refiner_dataset
```

## Development

```bash
python -m unittest discover -s tests/camera_refiner_data_construction -v
python -m compileall -q pre_experiments vggt
```

Large hidden tensors, Camera Head replay artifacts, checkpoints, datasets, and
figures stay outside Git. The remote machine is expected to provide the `vggt`
Conda environment, VGGT checkpoint, and processed ScanNet scenes.

## AutoDL CO3Dv2 Training Subset

The CO3Dv2 builder runs only on AutoDL and reuses the existing `vggt` Conda
environment. It selects the fixed 41-category training split with 50 sequences
per category (2050 total). A sequence must have at least 50 RGB frames with GT
camera annotations and `viewpoint_quality_score >= 0.5`.

```bash
cd /root/autodl-tmp/VGGT_Hallucination
git switch camera-refiner-data-construction
git pull --ff-only origin camera-refiner-data-construction

nohup bash scripts/autodl/camera_refiner_data_construction/download_co3d_2050.sh \
  > /root/autodl-tmp/co3d_2050.log 2>&1 &
tail -f /root/autodl-tmp/co3d_2050.log
```

Data is written to `/root/autodl-tmp/datasets/co3dv2_2050`. Re-running the
same command resumes `.part` downloads and completed categories. The official
server exposes category ZIP chunks rather than arbitrary sequences, so network
transfer is larger than the final 2050-sequence subset. Only selected
`images/` members and category annotations are retained; large ZIP files are
deleted after each processed chunk. The authenticated final selection is
recorded in `download_manifest.json`.
