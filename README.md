# Camera Refiner Data Construction

This branch studies how 100-, 200-, and 300-frame local contexts affect the
frozen VGGT Camera Head hidden state, then exports split-aware evidence for
training a camera refiner. It does not train the final refiner.

## Scope

The pipeline will:

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
- `scripts/autodl/`: ScanNet preparation only until the multiscale runner is
  implemented.
- `tests/`: CPU-only regression tests for retained infrastructure.

## Development

```bash
python -m pytest -q
python -m compileall -q pre_experiments vggt
```

Large hidden tensors, Camera Head replay artifacts, checkpoints, datasets, and
figures stay outside Git. The remote machine is expected to provide the `vggt`
Conda environment, VGGT checkpoint, and processed ScanNet scenes.
