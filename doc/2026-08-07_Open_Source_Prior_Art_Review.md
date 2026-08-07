# Open-Source Prior Art for Camera Residual Diffusion

## Scope

This review covers open-source methods that directly inform the translation-only
VGGT refiner. Third-party repositories are inspected under `tmp/references/` and are
not committed. Papers are stored under `paper/`.

## PoseDiffusion

Repository: <https://github.com/facebookresearch/PoseDiffusion>

PoseDiffusion predicts an absolute 9D camera encoding containing translation,
quaternion rotation, and focal length. It normalizes cameras, fixes the first camera
as a pivot, conditions an encoder-only Transformer on image features, and supports
direct clean-pose prediction. Its released training setup samples between 3 and 51
views and can use geometry-guided sampling.

Useful here: scene normalization, first-camera gauge, set-level temporal attention,
and direct clean-target prediction. Not adopted: full-pose generation, image feature
extraction, and geometry-guided sampling in the first experiment.

## RayDiffusion and DiffusionSfM

Repositories:

- <https://github.com/jasonyzhang/RayDiffusion>
- <https://github.com/QitaoZhao/DiffusionSfM>

Both methods avoid a single global pose vector by diffusing spatial ray
representations. Their released models use Transformer diffusion backbones,
sinusoidal timestep embeddings, and adaptive LayerNorm conditioning. DiffusionSfM
uses 100 training timesteps, predicts the clean target, normalizes the first camera,
and provides DDIM inference.

Useful here: a small 1D DiT with adaptive LayerNorm, zero-initialized residual output,
100 training timesteps, clean-residual prediction, and deterministic DDIM sampling.
Not adopted: dense ray origins/endpoints, image encoders, or joint rotation and
intrinsics. Those variables would defeat the first experiment's translation-only
causal test and greatly increase memory.

## VGGT-SLAM and VGGT-Long

Repositories:

- <https://github.com/MIT-SPARK/VGGT-SLAM>
- <https://github.com/DengKaiCQ/VGGT-Long>

These systems process long videos as overlapping VGGT submaps and recover global
consistency with explicit alignment and graph optimization. VGGT-SLAM 1.0 uses small
submaps and optimizes projective SL(4) transforms because Sim(3) can be insufficient
for uncalibrated cameras. VGGT-Long combines overlapping chunk alignment with loop
closure and supports Sim(3) or SE(3) depending on whether metric scale is available.

Useful here: exact overlap identity, boundary-aware fusion, explicit global gauge,
and a non-learned alignment baseline. They also show that local correction alone does
not guarantee long-range consistency. The first refiner therefore reports overlap
disagreement and remains a camera-center correction experiment; loop closure or graph
optimization is a later stage.

## Decision for This Branch

The implemented model will be a compact 1D DiT, not a generic Transformer encoder:

- diffusion state: normalized camera-center residual with shape `[B, 100, 3]`;
- conditions: global/local centers, their difference, frozen translation-unit
  activations, boundary position, and local alignment reliability;
- objective: direct clean-residual prediction;
- schedule: 100 cosine timesteps;
- sampler: deterministic DDIM with 10 steps;
- output: confidence-gated translation correction only;
- safety invariant: final rotation is copied exactly from global VGGT.

This is related to prior camera diffusion work but not equivalent to it. The novel
test is whether a learned residual prior can use VGGT's local-global and hidden-unit
disagreement to correct long-context translation without relearning rotation.

## Downloaded Papers

- `paper/PoseDiffusion_ICCV2023.pdf`
- `paper/RayDiffusion_ICLR2024.pdf`
- `paper/DiffusionSfM_CVPR2025.pdf`
- `paper/VGGT_SLAM_2025.pdf`
- `paper/VGGT_Long_2025.pdf`
