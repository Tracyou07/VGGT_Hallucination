# CO3D Pretraining and ScanNet Fine-Tuning Design

## Decision

Use CO3Dv2 to pretrain the existing translation-only camera residual refiner, then
fine-tune and evaluate it on ScanNet. CO3D does not replace ScanNet: it supplies
camera-motion diversity, while ScanNet remains the evidence for 500-frame indoor
VGGT hallucination and the primary reporting domain.

The refiner still changes camera centers only. Global VGGT rotations are copied
exactly, and the frozen translation-preferred Camera Head units remain conditions
rather than intervention targets.

## Scientific Controls

Run three models with the same architecture and ScanNet fine-tuning schedule:

- `scannet_only`: random initialization followed by ScanNet training;
- `co3d_then_scannet`: CO3D pretraining followed by ScanNet training;
- `co3d_zero_shot`: the frozen CO3D checkpoint evaluated on ScanNet.

Also run a compute-matched `scannet_only` control when reporting the value of extra
pretraining updates. CO3D is useful only if `co3d_then_scannet` beats both ScanNet
controls on the frozen ScanNet validation split. CO3D-only results cannot establish
that 500-frame hallucination has been corrected.

## CO3D Data Scope and Splits

Use CO3Dv2 RGB frames and camera annotations. Depth maps, masks, and point clouds are
not required for the primary translation target and are ignored by the cache builder.
The official single-sequence archive may package unused modalities; none are copied
into experiment results.

Adopt the 41 training-category and 10 held-out-category partition used by
RayDiffusion. Within the 41 training categories, freeze sequence-disjoint pretrain
and validation manifests before running VGGT. No sequence may cross a split.

Use four execution scales:

- smoke: 20 sequences from the official single-sequence subset;
- pilot: the complete official single-sequence subset, approximately 100 sequences
  and 8.9 GB of downloaded archives;
- category scale-up: at least 1,000 accepted sequences distributed across ten
  training categories;
- full: all accepted sequences from the 41 training categories, started only after
  the category-scale experiment passes and storage has been explicitly provisioned.

Manifests record category, sequence, frame IDs, source annotation digest, split,
sampling seed, and rejection reason. Sampling is category-balanced so large
categories do not dominate training.

## Ordered Clip Construction

Unlike the random view sets used by prior pose-diffusion work, retain acquisition
order because this project studies accumulated error. Reject missing images,
non-finite cameras, duplicate frame IDs, invalid rotations, and trajectories with
negligible translation baseline.

For each accepted sequence, deterministically construct up to four contiguous clips:

- global context lengths: 50, 75, or 100 valid frames;
- temporal stride: 1 or 2, recorded in the manifest;
- local VGGT window: half the global context, with half-window overlap;
- refiner length: the full 50-, 75-, or 100-frame clip.

Run VGGT once on the full clip and again on its local windows. Assemble the local
predictions in the full-clip prediction gauge. This creates the same kind of
local-global disagreement used by ScanNet without fabricating a 500-frame CO3D
trajectory. Do not concatenate unrelated object videos or loop frames.

## Camera Conversion and Labels

Convert CO3D camera annotations to the repository's 4-by-4 camera-to-world convention
in one isolated, tested adapter. Tests must verify camera centers, rotation
orthogonality, handedness, frame order, and projection consistency against the
source annotation. Preserve source poses in the cache for auditability.

All model coordinates are derived from VGGT predictions. Canonicalize each full-clip
prediction by its first predicted camera and robust trajectory scale. Align local
predictions to the matching global segment using prediction-only similarity
alignment. Use CO3D GT only to construct the supervised camera-center residual and
evaluation metrics. No GT value may enter an inference condition.

Primary CO3D training uses actual VGGT errors. Synthetic drift is excluded initially
because it would obscure whether the model learned real VGGT failure modes. It may
be tested later as a separately named ablation if the pilot contains too few
meaningful residuals.

## Canonical Prediction Cache

Both datasets must export the same versioned refiner-shard schema. Each clip stores:

- `source`, category or scene, clip ID, ordered frame IDs, and context lengths;
- raw global VGGT camera-to-world poses and raw GT camera-to-world poses;
- assembled prediction-only local poses and local-alignment diagnostics;
- iteration-0 activations for only the frozen translation-unit indices;
- canonical centers, GT-derived residual labels, validity flags, and schema digest.

Do not store RGB images, full hidden states, depth, point clouds, or VGGT weights in
the result tree. Cache generation is resumable per clip and writes a completion
marker only after validation and an atomic rename.

## Loader and Normalization Changes

Replace the current eager scene loader with a streaming dataset over canonical
shards. Bucket batches by refiner length so 50-, 75-, and 100-frame clips do not need
padding. The DiT keeps `max_frames=100`; loss lags are selected from those shorter
than the current clip.

Condition statistics are computed from each source's training split only and stored
under source-specific keys. CO3D pretraining uses CO3D statistics; ScanNet
fine-tuning and inference use frozen ScanNet statistics. Checkpoints record both
statistics, both manifest digests, the source stage, and the parent checkpoint.

## Training Curriculum

1. Overfit a small CO3D cache and verify decreasing loss, nonzero corrections, and
   exact rotation preservation.
2. Train deterministic and diffusion models on the CO3D pilot using identical data.
3. Evaluate the frozen checkpoints on held-out CO3D categories and ScanNet zero-shot.
4. Fine-tune each CO3D checkpoint on ScanNet with no CO3D replay in the primary run.
5. Train random-initialized and compute-matched ScanNet controls.
6. Scale to all 41 CO3D training categories only if pretraining improves frozen
   ScanNet validation metrics.

Training should be step-based rather than loading every sequence into memory. Resume
state includes optimizer, scheduler, sampler epoch, random generators, and consumed
clip IDs.

## Evaluation and Transfer Gate

CO3D reports aligned camera-center error, relative translation error, correction
magnitude, and category-level win rate. ScanNet retains the established 500-frame
global and 100-frame local protocol, including raw VGGT, training-free fusion,
deterministic refiner, and diffusion refiner baselines.

Proceed to full CO3D pretraining only when the pilot:

- improves ScanNet mean and median aligned translation error over `scannet_only`;
- wins on at least 60 percent of frozen ScanNet validation scenes;
- does not degrade relative translation at lags 1, 5, 10, or 25;
- changes no global rotation beyond numerical tolerance;
- shows a benefit beyond the compute-matched ScanNet control.

If only CO3D metrics improve, treat the result as domain-specific pretraining failure
and keep the ScanNet-only model.

## Artifacts

Keep datasets under `/root/autodl-tmp/datasets/co3dv2/`. Store compact prediction
caches under `/root/autodl-tmp/results/camera_refiner_data_construction/co3d/<run_id>`
and training runs under `/root/autodl-tmp/results/camera_refiner_training/<run_id>`.
Only manifests, scalar summaries, and concise analyses may be committed.
