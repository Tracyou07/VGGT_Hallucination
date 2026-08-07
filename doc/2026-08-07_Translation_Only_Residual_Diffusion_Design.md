# Translation-Only Camera Residual Diffusion Design

## Objective

Train a small conditional diffusion model that corrects long-context VGGT camera
centers while preserving VGGT rotations exactly. The first experiment uses a
500-frame global prediction and overlapping 100-frame local predictions. Its purpose
is to test whether local context and translation-preferred Camera Head features can
recover translation quality lost in the long sequence.

This is a correction model, not a replacement camera estimator. It must preserve
good global predictions, make small changes when evidence is weak, and produce one
coherent 500-frame trajectory after overlapping local corrections are fused.

## Evidence and Design Rationale

Current results establish the following starting point:

- Independently aligned 100-frame predictions have substantially lower translation
  error than the 500-frame prediction, but naive sequential stitching is unstable.
- Aligning local camera centers into the global prediction gauge gives a smaller but
  consistent training-free improvement on most scenes.
- Global rotations remain competitive; the useful hybrid keeps global rotations and
  replaces only camera centers.
- Weak intervention on 41 frozen translation-preferred units slightly improves
  translation, while strong hidden replacement damages both translation and rotation.

Therefore, the model will read hidden-unit evidence but act only in output space. It
will never overwrite Camera Head activations or predict rotation corrections.

## Scope and Non-Goals

The first version will:

- use 100-frame windows with stride 50 inside a 500-frame sequence;
- predict one three-dimensional camera-center residual per frame;
- use global and prediction-only aligned local trajectories as conditions;
- condition on the frozen 41-unit translation-preferred feature set;
- fuse overlap predictions into a single corrected global trajectory;
- compare diffusion against deterministic and training-free baselines.

The first version will not refine rotation, intrinsics, depth, point maps, or Camera
Head weights. It will not add bundle adjustment, Sampson loss, tracks, or depth-based
geometry. Those additions remain a second-stage experiment after the residual model
shows a measurable advantage.

## Data and Split Integrity

CO3Dv2 pretraining is an additive stage defined in
[`2026-08-07_CO3D_Pretraining_and_ScanNet_Finetuning_Design.md`](2026-08-07_CO3D_Pretraining_and_ScanNet_Finetuning_Design.md).
It learns a short multiview camera-motion prior but does not replace the ScanNet
500-frame fine-tuning and evaluation protocol below.

The previously inspected ScanNet-50 results are development evidence, not a fully
untouched final test set for this learned method. Model training must use additional
ScanNet training scenes that were not used in the earlier 10-scene calibration or
40-scene holdout analyses.

Use the splits as follows:

- training: new ScanNet training scenes outside the frozen ScanNet-50 list;
- validation: a fixed subset of new training scenes, separated by scene;
- development comparison: the existing ScanNet-50 protocol;
- final test: a newly frozen set of scenes not inspected during method design.

A one- or two-scene overfit run is allowed only as an engineering smoke test. Its
numbers must not be reported as model quality.

Each training scene requires the 500-frame global prediction plus 100-frame local
windows at stride 50. Cache only the selected iteration-0 hidden values needed by the
model. Raw images, full hidden traces, datasets, and checkpoints remain outside Git.

## Coordinate and Label Construction

The model operates in a window coordinate system derived only from the global VGGT
prediction:

1. Extract the matching 100-frame segment from the 500-frame global trajectory.
2. Align the local 100-frame camera centers to that global segment with a
   prediction-only similarity transform.
3. Canonicalize the global segment by its first predicted camera pose and a robust
   trajectory scale.
4. Apply the same prediction-derived canonicalization to the aligned local centers.
5. For supervision only, align raw GT centers into the global prediction gauge and
   store the resulting derived training target separately.

GT alignment is label construction, never an inference input. Raw GT is retained
unchanged. Final evaluation follows repository policy: predictions are aligned for
prediction metrics, while GT remains raw.

The clean diffusion target is the per-frame difference between the derived target
center and the canonical global center. The prediction-derived scene scale keeps
targets comparable across scenes. Training-set statistics standardize condition
features and are frozen for validation and test.

## Model Inputs and Outputs

For each of the 100 frames, construct condition features from:

- canonical global camera center;
- canonical, globally aligned local camera center;
- local-minus-global center difference;
- selected global translation-unit activations;
- selected local translation-unit activations;
- local-minus-global unit activation difference;
- normalized position within the window;
- distance to the nearest window boundary;
- prediction-only local alignment residual and validity flags.

The primary unit manifest is the frozen 41-unit set used by the successful weak
replacement experiment. The manifest path, source run ID, unit indices, Camera Head
iteration, and digest must be recorded in every run.

The model outputs:

- a clean camera-center residual with shape `[batch, 100, 3]`;
- a confidence gate with shape `[batch, 100, 1]` and values from zero to one.

The applied correction is the predicted residual multiplied by the confidence gate.
All global VGGT rotation matrices are copied unchanged to the final trajectory.

## Diffusion Network

Use a compact one-dimensional DiT rather than full 500-frame attention. Following
the open-source RayDiffusion and DiffusionSfM implementations, diffusion time enters
each block through adaptive LayerNorm modulation and the residual output layer starts
at zero:

- 6 Transformer encoder blocks;
- model width 256;
- 8 attention heads;
- learned frame-position embeddings;
- sinusoidal diffusion-timestep embedding;
- separate residual and confidence output heads.

Train with 100 cosine-scheduled diffusion steps. At each update, add noise to the
normalized clean residual at a randomly sampled timestep. The denoiser predicts the
clean residual directly so trajectory losses can be evaluated on its output. Primary
inference uses deterministic DDIM sampling with 10 steps and a fixed seed.

The architecture and sampler settings are initial defaults, not conclusions. They
may be changed using only the validation split and must be frozen before final test.

## Training Losses

The total objective contains five terms:

- denoising loss: reconstruct the clean camera-center residual at the sampled noise
  level;
- absolute center loss: keep corrected centers close to the derived GT target;
- relative motion loss: match center displacement over lags 1, 5, 10, and 25;
- overlap consistency loss: make two windows predict compatible corrections for the
  same frame;
- conservative gate loss: penalize correction magnitude as a small regularizer.

Loss weights are selected on validation and then frozen. Translation-unit magnitude
does not directly weight the loss because attribution evidence is stable at the unit
level but weak as a per-frame error detector. The unit values remain learned
conditioning signals.

## Window Fusion and Reconstruction

Map each corrected window from canonical coordinates back into the original global
VGGT gauge. Frames covered by two windows receive a weighted average of their camera
center corrections. The weight combines:

- predicted confidence;
- a fixed boundary taper that favors the center of each window;
- inverse prediction-only local alignment residual.

Frames with invalid local alignment or zero total confidence keep the original global
camera center. The corrected extrinsics are reconstructed from the corrected centers
and the untouched global rotation matrices. Fusion must never alter rotation.

## Baselines and Ablations

Report the following under identical scene and frame selections:

- raw 500-frame VGGT;
- global-anchored local-center fusion without learning;
- deterministic Transformer with the same conditions and output;
- residual diffusion without hidden-unit features;
- residual diffusion with the frozen translation-unit features;
- diffusion with its confidence gate forced to one.

The deterministic Transformer determines whether diffusion itself adds value. The
no-unit ablation determines whether Camera Head attribution adds information beyond
the two predicted trajectories.

## Metrics and Acceptance Gates

Primary metrics are aligned camera-center translation error, translation relative
pose error at lags 1, 5, 10, and 25, and per-scene win rate against raw global VGGT.
Report mean, median, scene-bootstrap 95 percent confidence intervals, and paired
per-scene differences. Also report overlap disagreement and correction magnitude.

Rotation is a safety metric. Because rotations are copied unchanged, reconstructed
rotations must match the global VGGT rotations within numerical tolerance. Any larger
difference is an implementation failure.

Proceed beyond the first experiment only when the diffusion model:

- improves mean translation error over raw global VGGT;
- improves at least 60 percent of validation scenes;
- beats both the training-free fusion and deterministic Transformer baselines;
- has a paired confidence interval that does not cross zero;
- introduces no measurable rotation change.

Failure to beat the deterministic model means the useful result is conditional
translation refinement, not diffusion. Failure to beat training-free fusion means the
learned refiner is not justified.

## Artifacts and Reproduction

Store remote outputs under `/root/autodl-tmp/results/camera_refiner_training/<run_id>`.
Each run contains immutable configuration, Git commit, split manifests, unit-manifest
digest, normalization statistics, checkpoints, per-scene metrics, and completion
markers. Scene/window caches are resumable and large tensors remain uncommitted.

Only scalar CSV/JSON summaries, frozen manifests, and concise analysis documents may
be exported to Git. Training entry points must assume the existing `vggt` Conda
environment and checkpoint; they must not recreate environments or download weights.

## Staged Execution

1. Extend data construction to cache compact global/local selected-unit features for
   new training scenes.
2. Validate coordinate conversion, GT-derived labels, overlap pairing, and exact
   rotation preservation with CPU tests.
3. Overfit one scene and verify that loss and translation error decrease.
4. Train the deterministic baseline and residual diffusion model on the frozen
   training split.
5. Select settings on validation, freeze one configuration, and run development and
   final evaluations once.
6. Consider smaller windows or geometry guidance only after the 100-frame experiment
   passes its acceptance gates.
