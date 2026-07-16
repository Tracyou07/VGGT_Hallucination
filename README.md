# VGGT Camera Iteration Pre-experiment

This branch studies the intermediate refinement iterations produced by VGGT's
Camera Head. It is self-contained and does not depend on the phenomenon-
characterization evaluator or results.

## AutoDL Quick Start

Weights and ScanNet data are expected to exist before the run:

```bash
git clone https://github.com/Tracyou07/VGGT_Hallucination.git
cd VGGT_Hallucination
git switch camera-iteration-preexperiment
bash scripts/autodl/run_camera_iteration.sh
```

Default external paths are:

- ScanNet: `/root/autodl-tmp/datasets/scannetv2`
- VGGT-1B: `/root/autodl-tmp/ckpt/VGGT-1B`
- Results: `/root/autodl-tmp/camera_iteration/results`
- Conda environment: `vggt_camera_iteration`

The runner never downloads weights or ScanNet. It uses an existing
`process_scannet/` tree, or extracts requested scenes when only `.sens` files
are present. Every path and experiment size can be overridden with environment
variables documented in `pre_experiments/camera_iteration/README.md`.

## Development

```bash
pip install -r requirements.txt
pip install -r requirements-camera-iteration.txt
pip install -e .
python -m unittest discover -s tests
```

Core model changes live in `vggt/`. The method package lives in
`pre_experiments/camera_iteration/`, CPU tests in `tests/camera_iteration/`,
and the executable plan in `doc/VGGT_DiT_Implementation_Plan.md`.

No camera-iteration experiment has been run on this branch yet.
