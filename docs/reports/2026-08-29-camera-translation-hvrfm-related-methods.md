# Camera-translation H-VRFM: related methods and executable design

Date: 2026-08-29

## 1. Scope and evidence boundary

The deployment contract is fixed: inference may run VGGT once on the 500-frame
window, run the frozen Camera Head once, and then produce about four corrections
with a lightweight head.  It may not run short windows, a second backbone pass,
test-time optimization, chunk alignment, loop closure, GT, or privileged labels.
Short-window predictions are privileged training evidence only.

Two kinds of statements are kept separate below:

- **Paper fact** describes a mechanism or result reported by a cited primary
  source.
- **Project inference** is our proposed adaptation or a conclusion from our own
  calibration evidence.  It is not a claim made by the cited paper.

Our present evidence does **not** support a prior assumption of four discrete
modes.  The Phase-1 experiment found that 45/80 overlaps had a useful
short-window direction, but 42 were useful only after a small continuous step;
left and right teachers won 20 and 25 times, without stable separated branches.
The same report explicitly concludes that the evidence resembles scene-dependent
continuous direction-and-step selection rather than fixed left/right solutions:
[Phase-1 report](2026-08-27-variational-camera-latent-phase1-report.md).
The formal token-DCT Stage A run was scientifically classified
`LATENT_LIFT_FAILED`.  Within that failure analysis, a separate CPU-only
translation counterfactual copied authenticated short-teacher centers in the raw
long gauge and obtained positive full-scene translation utility on 10/10
calibration scenes while preserving long rotation/FOV and uncovered frames:
[Stage A decision](2026-08-29-camera-translation-hvrfm-stage-a-decision.md).
That counterfactual motivates translation A-prime; it is not an A-prime formal
result, and translation A-prime has not yet been run.

**Project inference.**  The first learned prior should therefore be a unimodal,
condition-dependent, low-rank correlated Gaussian.  A categorical four-mode or
four-component GMM prior would turn an unverified scientific hypothesis into an
architectural assumption.  It is justified only if the Stage B posterior later
exhibits stable, held-out, separated clusters.

## 2. Primary-source method map

### Variational and hierarchical flow matching

- **Paper fact — VRFM.** [Variational Rectified Flow Matching
  (2025)](https://arxiv.org/abs/2502.09616) introduces a latent variable into the
  velocity field so different velocities may coexist at the same data/time
  location.  Its approximate posterior sees the source, target, interpolant, and
  time during training; inference samples a standard-normal latent.  Random
  source/target pairing can induce multiple velocity targets at the same
  data/time location, while squared-error regression learns their conditional
  mean.  **Project inference:** in our camera-correction setting, ordinary MSE may
  therefore average useful but different short-teacher correction directions.
  Retain the latent-conditioned velocity field, but replace the fixed inference
  prior with a learned long-window conditional prior `p(z | H_long)`.

- **Paper fact — endpoint-posterior VFM.** [Variational Flow Matching for Graph
  Generation (NeurIPS 2024)](https://papers.neurips.cc/paper_files/paper/2024/file/15b780350b302a1bf9a3bd273f5c15a4-Paper-Conference.pdf)
  reformulates flow matching as approximation of the posterior probability path
  over possible endpoints rather than direct regression to one expected path.
  **Project inference:** this supports learning a distribution over
  short-teacher correction endpoints, but CatFlow itself is categorical and does
  not provide our continuous camera model, privileged posterior, or K-candidate
  sampler.

- **Paper fact — HRF.** [Towards Hierarchical Rectified Flow (ICLR
  2025)](https://proceedings.iclr.cc/paper_files/paper/2025/file/66d1ebf0d4fc408aad4c7cc2f1a654bc-Paper-Conference.pdf)
  couples ODEs in location, velocity, acceleration, and higher domains to model
  random velocity fields and obtain straighter paths with few function
  evaluations.  **Project inference:** a global latent plus local temporal
  correction is compatible with this motivation, but our 10-by-50 temporal
  hierarchy is not mathematically the same as HRF's derivative-order hierarchy.
  A nested HRF sampler is unnecessary for the first experiment.

- **Paper fact — S-VFM.** [Learning Straight Flows: Variational Flow Matching for
  Efficient Generation (CVPR 2026)](https://openaccess.thecvf.com/content/CVPR2026/papers/Ma_Learning_Straight_Flows_Variational_Flow_Matching_for_Efficient_Generation_CVPR_2026_paper.pdf)
  uses a variational latent as a global generation overview and adds a straightness
  objective for efficient few-step generation.  **Project inference:** a
  velocity-consistency loss is appropriate only after the latent has separated
  useful directions; without that separation, straightness can preserve the same
  mean-direction failure.

- **Paper fact — MM-FM.** [Flow Matching for Multimodal Distributions (CVPR
  2026)](https://mm-flow.github.io/) co-designs a mixture source distribution and
  mode-dependent coupling for a multimodal target.  **Project inference:** a
  conditional GMM source is a later ablation if posterior evidence demonstrates
  discrete modes, not the Stage B default.

### Privileged posterior, distillation, and K-candidate generation

- **Paper fact — MUSE-VAE.** [MUSE-VAE (CVPR
  2022)](https://openaccess.thecvf.com/content/CVPR2022/papers/Lee_MUSE-VAE_Multi-Scale_VAE_for_Environment-Aware_Long_Term_Trajectory_Prediction_CVPR_2022_paper.pdf)
  learns a condition-only prior to approximate a posterior that additionally sees
  the future, and includes reconstruction from the prior to reduce the
  train/test gap.  **Project inference:** this is the closest structural analogue
  to `q(z | H_long, short teachers)` and `p(z | H_long)`.  We additionally need a
  permutation-invariant set posterior and joint K-sample alignment.

- **Paper fact — privileged-modality KD.** [Multi-modal Knowledge
  Distillation-based Human Trajectory Forecasting (CVPR
  2025)](https://openaccess.thecvf.com/content/CVPR2025/html/Jeong_Multi-modal_Knowledge_Distillation-based_Human_Trajectory_Forecasting_CVPR_2025_paper.html)
  trains a full-modality teacher and distils local and global knowledge into a
  limited-modality student.  **Project inference:** short windows play the role of
  the unavailable modality.  Pure feature L2 is insufficient here because long
  and short windows have different temporal topology; distil the posterior and
  correction distribution instead.

- **Paper fact — MoFlow.** [MoFlow (CVPR
  2025)](https://openaccess.thecvf.com/content/CVPR2025/html/Fu_MoFlow_One-Step_Flow_Matching_for_Human_Trajectory_Forecasting_via_Implicit_CVPR_2025_paper.html)
  jointly predicts K trajectory sets, balances at-least-one accuracy, diversity,
  and plausibility, and uses IMLE to distil a flow teacher into a one-step student.
  **Project inference:** joint K prediction and sample-based distillation are
  useful for Stage C, but direct IMLE is high-risk with only ten calibration
  scenes and a small finite teacher set; analytic KL and set reconstruction should
  be the primary objectives.

- **Paper fact — TrajFlow.** [TrajFlow
  (2025)](https://arxiv.org/abs/2506.08541) emits multiple coherent trajectories in
  one pass and uses a Plackett--Luce listwise ranking loss for confidence
  calibration.  **Project inference:** use the same listwise principle to train a
  long-only candidate scorer with privileged utility labels, while keeping those
  labels physically absent at inference.

- **Paper fact — DLow.** [DLow (ECCV
  2020)](https://arxiv.org/abs/2003.08386) maps one shared random variable through
  multiple learned transforms into correlated latent codes, with an energy-based
  diversity prior and KL constraints for likelihood.  **Project inference:** this
  is the strongest precedent for four correlated candidates.  Diversity must be
  capped by the observed teacher spread; unconditional repulsion could invent
  branches that the data does not contain.

- **Paper fact — DSF/DPP.** [Diverse Trajectory Forecasting with Determinantal
  Point Processes (ICLR 2020)](https://arxiv.org/abs/1907.04967) learns a
  context-conditioned deterministic function that emits a fixed-cardinality set
  of latent codes.  Its DPP kernel combines sample quality/likelihood and
  diversity, and training maximizes a differentiable expected-cardinality
  objective; it is not merely a pairwise-repulsion loss.  **Project inference:**
  DPP is an optional diversity ablation only.  Applying it as unconditional
  repulsion, without teacher-spread and quality constraints, could invent branches
  that are poorly matched to the currently observed continuous correction family.

## 3. Why the default is a low-rank correlated Gaussian

For K=4 joint latents, use

```text
Z[1:K] = mu(H_long)
       + L(H_long) * epsilon_shared
       + D(H_long) * epsilon_private,
```

where `L` is low rank and `D` is diagonal.  This choice is a **project
inference**, not a theorem from DLow:

1. A shared factor represents scene-level direction/scale uncertainty common to
   all four corrections.
2. Private diagonal noise allows local alternatives without four independent
   backbone-conditioned generators.
3. The full joint Gaussian admits stable analytic KL and exact K=4 permutation
   comparison.
4. It can represent continuous, anisotropic, correlated uncertainty without
   asserting discrete branches.
5. Batched decoding preserves the one-backbone/one-Camera-Head contract.

Escalate to a conditional mixture only if posterior samples form reproducible
clusters across seeds and held-out scenes, and a mixture improves held-out
likelihood/coverage rather than only best-of-K oracle utility.

## 4. Stage B: target-aware posterior upper bound

### Architecture

```text
frozen VGGT(500 frames), exactly once
  -> H_long[500,D] and long camera[500,9]

training-only authenticated short teachers
  -> permutation-invariant set encoder
  -> q_phi(Z | H_long, teacher set)

batched lightweight v_theta(X_t, t, Z, H_long)
  -> K=4 translation-only corrections[4,500,3]
```

Use a lossless 10-by-50 temporal representation, a small global scene token, and
masked frame tokens.  Decode in the frozen raw-long W2C gauge.  Quaternion and FOV
bytes are copied from the long prediction, and corrections outside the published
coverage mask remain positive zero.

### Losses

- `L_fm`: ratio-of-means masked velocity matching, scene-balanced, with stored
  teacher quality weights normalized within scene.  Quality weighting is a noise
  control, not the source of multimodality.
- `L_set`: bidirectional soft-Chamfer.  Teacher-to-candidate supplies coverage;
  candidate-to-teacher prevents implausible diversity.
- `L_kl_perm`: joint posterior/prior KL minimized over all 24 K=4 permutations,
  evaluated with a float64 dense Cholesky.  Use KL warm-up and free bits.
- `L_geometry`: covered translation endpoint loss in the frozen long gauge.
- `L_zero`: exact zero outside coverage; no rotation or FOV objective is allowed
  to move those copied components.
- `L_rank`: privileged listwise ranking target for a scorer whose inputs remain
  prediction-only.
- `L_div_capped`: penalize collapse only when predicted pairwise spread is below
  authenticated teacher spread; never reward spread beyond that reference.

Stage B asks whether a short-aware posterior can encode the useful correction
family at all.  Do not start Stage C if posterior oracle cannot beat the
deterministic mean-correction control on held-out scenes.

## 5. Stage C: long-only conditional prior

Train `p_psi(Z | H_long)` with the low-rank-plus-diagonal joint parameterization.
Freeze or slowly update the validated posterior/decoder first, then use:

- exact permutation-aware `KL(q_phi || p_psi)`;
- posterior-to-prior and prior-to-posterior sample-set matching;
- reconstruction/FM loss on samples drawn from the prior, preventing a prior that
  matches moments but decodes poorly;
- optional MoFlow-style IMLE distillation as a secondary ablation;
- long-only Plackett--Luce candidate scoring;
- only after mode separation, S-VFM-style velocity consistency across two times
  on the same `(z, endpoint)` path.

Begin with four lightweight Euler steps, then ablate 2 and 1.  All K candidates
are decoded as one batch after the single frozen backbone and Camera Head calls.

## 6. Decision matrix

| Observation | Conclusion | Next action |
|---|---|---|
| Posterior oracle does not beat deterministic mean | Teacher/coordinate target is inadequate; VRFM complexity is not justified | Stop before Stage C and repair teacher construction |
| Posterior oracle improves, posterior uniform does not | Candidate generation works but most posterior mass is poor | Fix set likelihood/quality weighting |
| Posterior improves, prior does not | Short windows contain useful information not yet predictable from long context | Improve posterior-to-prior distillation; do not claim deployability |
| Prior oracle improves, selected does not | Generator works; scorer/calibration fails | Focus on ranking and prediction-only scoring |
| Prior uniform and selected improve | Deployable long-only signal exists | Proceed to full-data training |
| K candidates remain close but improve baseline | Continuous correction/noise reduction, not discrete branches | Retain correlated Gaussian; do not add GMM |
| Stable held-out posterior clusters and GMM improves likelihood/coverage | Evidence for discrete modes | Add mixture prior as a separately reported model |

Always report the chain `posterior oracle -> posterior uniform -> prior oracle ->
prior uniform -> prior selected`.  Best-of-K alone cannot distinguish generation,
distillation, and selection failures.

## 7. Long-video methods: useful controls but invalid deployment paths

- [VGGT](https://arxiv.org/abs/2503.11651) establishes the single feed-forward,
  hundreds-of-views backbone used by this project.
- [VGGT-Long](https://arxiv.org/abs/2507.16443) uses chunks, overlap alignment,
  and loop closure.
- [StreamVGGT](https://github.com/wzzheng/streamvggt) changes the backbone to
  causal attention with cached memory tokens.
- [TALO (CVPR 2026)](https://arxiv.org/abs/2512.02341) uses overlapping submaps,
  propagated global control points, and Thin Plate Spline alignment.
- [VGGT-Motion (2026)](https://arxiv.org/abs/2602.05508) uses motion-aware
  submaps, anchor-driven Sim(3), loop closure, and pose-graph optimization.

**Paper fact.**  The four long-sequence adaptations after base VGGT explicitly
address submap context, static redundancy, scale, or cross-window registration
with the mechanisms listed above.  **Project inference.**  Those four
adaptations are useful only as offline upper bounds and failure-diagnosis
controls for this project, not as formal deployment baselines, because each
requires chunk/short-window inference, modified recurrent state, alignment,
loop closure, or optimization forbidden by the one-500-frame contract.  Base
VGGT remains the permitted single-pass backbone.

## 8. Minimum ablation set

Run the following on the fixed calibration split before scaling:

1. raw long baseline;
2. deterministic mean correction;
3. posterior K=4 oracle and uniform;
4. prior K=4 oracle, uniform, and learned selection;
5. independent K Gaussians versus shared low-rank correlated Gaussian;
6. no KL, fixed KL weight, and warm-up plus free bits;
7. no set loss, one-way set loss, and bidirectional set loss;
8. no quality weights, stored weights, and capped/normalized weights;
9. no diversity, unconditional repulsion, and teacher-spread-capped diversity;
10. K in `{1,2,4,8}` and flow NFE in `{1,2,4,8}`;
11. best single short teacher versus the full teacher set posterior;
12. Gaussian versus GMM only after the predeclared clustering gate.

For every ablation, report equal-scene mean and worst-scene utility, coverage,
precision, teacher/predicted pairwise spread, ranking calibration, uncovered
drift, rotation delta, and q/FOV byte preservation.  This prevents a large scene,
an oracle selector, or artificial diversity from masquerading as a useful learned
prior.
