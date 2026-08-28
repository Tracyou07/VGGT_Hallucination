# Privileged Conditional Hierarchical VRFM Design

## Objective

Build a source-level long-window camera correction system in which training may use nine
overlapping 100-frame VGGT predictions as a privileged-compute teacher, while formal
inference runs the 500-frame VGGT backbone exactly once. Inference may draw exactly four
lightweight residual candidates from that one long representation and may repeat only the
residual flow and frozen Camera Head decode.

The immediate deliverable is a compact, verified batch of native Camera-token residual
targets. Later phases use those targets to train a short-aware posterior and a long-only
conditional prior. Formal outputs live below
`/data/yjh/output/vggt/privileged_conditional_hvrfm/<run_id>/` on H20. Large artifacts stay
on H20.

## Accepted Inference Contract

The user-approved deployment boundary is:

1. load one 500-frame image sequence;
2. run the VGGT backbone once and retain its long-window representation;
3. derive `K=4` residual candidates from that same representation;
4. decode those candidates through the final lightweight correction module and frozen
   Camera Head;
5. choose one candidate using prediction-only geometric evidence or retain the exact
   no-op baseline.

No image, short-window token, GT pose, depth label, error label, or teacher quality may be
loaded by the formal inference API.

## Evidence That Fixes the Design

Completed experiments impose four non-negotiable constraints:

- Short-window pose directions are often useful, but mostly at small continuous step
  sizes: 45/80 overlaps had at least one useful short direction and 42 were rescued only
  by a small alpha.
- The original raw-token VRFM direction was not special: it ranked 18/21 against twenty
  matched random orthogonal transforms, with 17/20 random transforms scoring better.
- A post-hoc selector reached Spearman 0.60175 but harmed held-out utility; a conservative
  gate reduced harm only by collapsing toward no-op.
- Native Camera Head-only fine-tuning had mean locked-replay utility `-1.668%`, localizing
  the missing signal outside a pure decoder-weight update.

A read-only replay on the existing formal long/short labels adds a fifth fact. The
GT-quality-weighted short teacher improves translation RMS on its covered frames in all
10/10 scenes, with mean relative utility `12.9358%` and mean frame coverage `89%`. This is
a privileged-compute upper bound, not a deployable result, because GT-derived scalar
weights select helpful windows.

Therefore the project must not flow directly from long tokens to raw short tokens, must
not use a fixed unconditional Gaussian prior, and must not repair eight overlaps
independently.

## Relation to Current Methods

The design keeps the useful VRFM principle—one condition can support a distribution of
velocity fields through a latent variable—but replaces its fixed prior with a learned
long-conditioned prior. This addresses the conditional train/inference mismatch described
by condition-aware flow-matching work. Hierarchical rectified flow motivates separating
low-frequency whole-trajectory correction from local temporal detail. VGGT-Long,
streaming VGGT variants, Mamba-VGGT, and InfiniteVGGT all place long-context repair in
representation or memory space rather than only in the final camera decoder. VGGT and
TrajVG motivate using depth, point, track, and pose consistency as long-only evidence.

Primary references:

- VRFM: <https://arxiv.org/abs/2502.09616>
- C2OT: <https://arxiv.org/abs/2503.10636>
- Conditional Variable Flow Matching: <https://arxiv.org/abs/2411.08314>
- Hierarchical Rectified Flow: <https://arxiv.org/abs/2502.17436>
- Riemannian Flow Matching: <https://arxiv.org/abs/2302.03660>
- VGGT: <https://arxiv.org/abs/2503.11651>
- TrajVG: <https://arxiv.org/abs/2602.04439>
- VGGT-Long: <https://arxiv.org/abs/2507.16443>
- Streaming 4D Visual Geometry Transformer: <https://arxiv.org/abs/2507.11539>
- InfiniteVGGT: <https://arxiv.org/abs/2601.02281>

## Architecture

### 1. Canonical latent target by optimization, not raw-token pairing

Let `H` be the frozen normalized Camera tokens from the single 500-frame prediction,
with shape `[500,2048]`. A native residual is represented by fixed temporal DCT basis
`B in R^[500,32]` and learned coefficients `C in R^[32,2048]`:

`R = B C`, `H_corrected = H + R`.

`B` is deterministic and orthonormal. Frequencies 0–3 are the global component and
frequencies 4–31 are the local component. This gives a compact whole-500 correction,
avoids eight independent seams, and stores 32/500 of a dense residual.

For each training scene and deterministic teacher bootstrap, optimize `C` through the
frozen Camera Head so the decoded trajectory matches the fused short-window teacher.
The optimization starts from exact zero and minimizes:

- teacher camera-center loss on frames with positive teacher coverage;
- relative-translation loss at lags 1, 5, 10, and 25;
- rotation loss on covered frames;
- second-difference smoothness of the corrected trajectory;
- residual norm regularization;
- baseline anchor loss on uncovered frames.

The decoded Camera Head and baseline-to-GT Sim(3) are frozen. No student-specific
alignment may be fitted. The resulting minimum-regularized `C` is the canonical latent
endpoint. Raw short tokens are never treated as coordinates in the long-token space.

### 2. Multiple teacher endpoints for the same long condition

Each scene produces exactly four deterministic teacher variants. Variant 0 uses all
positive-quality windows. Variants 1–3 use SHA-256-seeded Bernoulli masks over positive
windows with inclusion probability 0.75, while forcing every covered frame to retain at
least one contributing window whenever possible. Empty or duplicate masks are rejected.

Each variant is fused independently and lifted to one coefficient tensor. This produces
four valid endpoints for the same long condition without inventing left/right categories.
Their differences represent continuous teacher ambiguity and redundancy.

### 3. Privileged posterior

After latent lifting passes, train
`q_phi(z_global,z_local | H_long,S_1:9)` using the long prediction plus nine short
prediction sequences. GT arrays never enter `q_phi`; GT-derived quality weights affect
only the training loss through a physically separate sidecar.

The posterior conditions a whole-500 velocity model that flows from zero coefficients to
one of the four canonical coefficient endpoints. `z_global` controls frequencies 0–3 and
`z_local` controls frequencies 4–31. One sampled pair is fixed through every ODE step.

### 4. Long-only conditional prior

Train `p_psi(z_global,z_local | H_long,G_long)` to match the posterior, where `G_long`
contains only prediction-time summaries from the same long forward pass: Camera-token
statistics, Camera Head confidence/iteration traces, and compact depth/point/track
consistency features. The KL is `KL(q_phi || p_psi)`, with warmup and free bits; it is not
`KL(q_phi || N(0,I))`.

The prior signature must not accept short tokens, prepared-scene paths, GT, teacher
weights, or utilities. Checkpoints from the old unconditional VRFM are schema-incompatible
and must fail closed.

### 5. Prediction-only candidate choice

Formal inference samples exactly four latent pairs from `p_psi`, integrates four compact
coefficient trajectories, expands them through `B`, and decodes four Camera trajectories.
Selection uses a frozen prediction-only geometric score. It must report first, uniform
random, geometry-selected, and GT-oracle results separately. GT-oracle is an upper bound,
never the deployable score. If no candidate clears the frozen no-op margin, output the
unmodified long prediction.

## Data Separation

Artifacts are physically separated:

- `prediction_only/long_context/<scene>.npz`: long tokens, long prediction summaries,
  frame IDs, baseline trajectory, and provenance only;
- `prediction_only/short_teacher_source/<scene>.npz`: nine short predictions and their
  frame bindings, usable only by the offline posterior/target builder;
- `privileged_labels/teacher/<scene>.npz`: raw GT, frozen alignment, per-window utilities,
  fused teacher variants, coverage masks, and source hashes;
- `privileged_labels/latent_targets/<scene>.npz`: four optimized DCT coefficient targets,
  decoded diagnostics, loss traces, and teacher bindings;
- `prediction_only/candidates/<scene>.npz`: long-only posterior-free prior samples and
  decoded candidate metadata;
- `privileged_labels/evaluation/<scene>.npz`: GT utilities joined strictly by sample ID.

Anything derived from GT remains privileged even if the raw GT array is absent. Main
training examples and inference examples join labels only by immutable scene/sample IDs
and SHA-256 digests.

## Staged, Fail-Closed Protocol

### Stage A: teacher replay and native latent lifting

Reuse the authenticated ten-scene source and formal long/short labels. First reproduce the
`12.9358%` covered-frame teacher upper bound. Then optimize DCT coefficients for one smoke
scene and, on success, all ten calibration scenes.

Stage A passes only if:

- all four variants decode to finite homogeneous poses;
- mean covered-frame decoded utility retains at least 70% of the fused-teacher utility;
- mean full-scene utility is positive;
- at least 8/10 scene means are positive;
- no scene is worse than baseline by more than 1%;
- mean rotation increase is at most 0.1 degree;
- uncovered-frame anchor drift is at most 0.5% of frozen scene scale;
- every artifact passes schema, digest, provenance, and leakage verification.

Failure stops before training a posterior or prior. Passing Stage A yields the first useful
batch of native latent training data.

### Stage B: privileged posterior upper bound

Train the posterior/flow on eight train scenes and evaluate the two locked replay scenes.
Report posterior mean, first, random, geometry-selected, and oracle `K=4` utility. Continue
only if posterior-mean scene-level paired bootstrap 95% CI has a lower bound above zero.

### Stage C: conditional-prior distillation

Replace posterior samples with long-only prior samples under exactly the same candidate
budget and decoder. Continue only if geometry-selected prior utility is positive and at
least 70% of geometry-selected posterior utility. Do not use prior oracle utility to hide a
failed deployable selector.

### Stage D: fresh-scene evaluation

Freeze architecture, seeds, `K=4`, thresholds, and geometry score before generating new
long/short caches. Run on disjoint ScanNet scenes. The final system must beat no-op, a
deterministic residual control, the old fixed-prior VRFM, and matched random residuals.
It must also reverse the observed `-1.668%` Camera Head-only loss by a pre-registered
practical margin.

## Resource and Safety Constraints

- Formal compute uses H20 only after rechecking identity, GPUs, active processes, worktree,
  checkpoint, dataset marker, and disk.
- `/data` currently has limited free space; Stage A references existing source shards and
  stores DCT coefficients rather than dense residuals. The runner refuses to start below
  100 GiB free and has a 20 GiB run-root budget.
- Never use the previously exposed H20 Hugging Face token.
- Never copy large H20 artifacts to Windows.
- A dirty worktree, stderr output, nonfinite tensor, schema mismatch, source digest change,
  GPU conflict, or disk-budget violation stops the run without deleting prior artifacts.

## Stage A Artifact Contract

```text
/data/yjh/output/vggt/privileged_conditional_hvrfm/<run_id>/
  config.json
  logs/
  manifests/
  prediction_only/long_context_manifest.json
  privileged_labels/teacher/
  privileged_labels/latent_targets/
  smoke/
  calibration/
  reports/teacher_upper_bound.json
  reports/latent_lift.json
  reports/human_report.md
  verified_completion.json
```

`verified_completion.json` binds the Git commit, source and checkpoint hashes, exact scene
roles, fixed DCT basis digest, four variant masks, optimizer configuration, checkpoint-free
target hashes, decoded metric hashes, test evidence, and final Stage A classification.
