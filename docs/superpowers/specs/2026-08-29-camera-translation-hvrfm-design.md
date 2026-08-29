# Camera-Translation Hierarchical VRFM Design

## Decision

The privileged short-window signal is useful, but the current correction interface is not.
The verified token-DCT lift changed `[500,2048]` Camera tokens and then decoded them through
the nonlinear Camera Head.  It achieved positive mean translation utility, but coupled that
gain to bad short-window rotations and to motion outside the teacher support.

The replacement keeps long and short Camera tokens as the conditioning information, but moves
the flow state to the normalized translation part of the Camera Head output.  Rotation and FOV
remain exactly equal to the long-window baseline.

Verified evidence from the ten-scene calibration cohort:

- token-DCT lift: mean full-scene utility `+7.0175%`, `9/10` positive scenes,
  teacher retention `0.6968`, mean rotation delta `+2.2291 deg`, and maximum uncovered drift
  `0.1202`;
- direct translation counterfactual: mean full-scene utility `+10.6546%`, `10/10` positive
  scenes, worst scene `+0.8438%`, teacher retention `1.0`, rotation delta `0`, and uncovered
  drift `0`;
- the short teacher's covered-frame rotation is itself `+2.2971 deg` worse than the long
  baseline, and the token lift reproduces that failure.  Rotation is therefore not a valid
  target for this phase.

## Frozen inference contract

Formal inference must:

1. execute the VGGT aggregator exactly once on one 500-frame input;
2. execute the normal frozen Camera Head once to obtain long Camera tokens and the baseline
   pose encoding;
3. run no short windows, chunks, loop closure, GT, privileged sidecar, posterior, test-time
   optimization, or second backbone pass;
4. sample one correlated set of exactly four lightweight translation trajectories;
5. keep quaternion and FOV bytes from the baseline pose encoding unchanged;
6. select using prediction-only evidence, with an exact zero/no-op fallback.

Formal compute runs only on H20 and publishes under
`/data/yjh/output/vggt/camera_translation_hvrfm/<run_id>/`.

## Translation endpoint

For frame `t`, let the long Camera Head output be

```text
e_long[t] = [T_long_w2c[t], q_long[t], fov_long[t]] in R^9.
```

Let `c_long[t]` be the corresponding long camera center, and let `c_teacher[k,t]` be the
prediction-to-prediction aligned short-window teacher center for endpoint `k`.  The
prediction-only normalization scale is

```text
s_long = RMS(c_long - mean(c_long)).
```

For coverage mask `m[k,t]`, define the privileged endpoint with an explicit branch (ordinary
multiplication is forbidden because an uncovered fused teacher may be NaN):

```text
X_target[k,t] = where(
    m[k,t],
    (T_teacher_hybrid_w2c[k,t] - T_long_w2c[t]) / s_long,
    0
)
T_teacher_hybrid_w2c[k,t] = -R_long_w2c[t] @ c_teacher[k,t].
```

Equivalently,

```text
X_target[k,t] = where(
    m[k,t],
    -R_long_w2c[t] @ (c_teacher[k,t] - c_long[t]) / s_long,
    0
).
```

Decode only translation:

```text
T_corrected_w2c = T_long_w2c + s_long * X
q_corrected = q_long
fov_corrected = fov_long.
```

This endpoint is dense `[K=4,500,3]` float32.  It is smaller than `[K,32,2048]`, is lossless,
has exact zero as no-op, is exactly zero outside teacher coverage, preserves long rotation and
FOV, and is invariant to a global Sim(3) change after division by `s_long`.

The numeric teacher center is built in the raw long prediction gauge.  Raw GT coordinates and
the frozen baseline-to-GT oracle must never enter its coordinate, alignment, or gauge
calculation.  They may affect it only indirectly through pre-registered, digest-bound
privileged window weights/masks used to select and fuse short predictions.  The target and
those weights/masks are therefore privileged.  Holding registered weights/masks fixed while
mutating raw GT/oracle diagnostics must leave the endpoint unchanged.

## Physical data separation

The main sample is prediction-only:

```text
prediction_only/long_context/<scene>.npz
```

It contains the long Camera tokens, baseline pose encoding/C2W, frame IDs, prediction scale,
and immutable provenance.

Training-only short predictions are separate:

```text
privileged_training/short_context/<scene>.npz
```

Numeric targets and GT-derived labels are separate and linked only by stable sample ID:

```text
privileged_labels/translation_targets/<scene>.npz
privileged_labels/quality/<scene>.npz
```

The quality sidecar contains coverage/quality weights, GT, frozen oracle, utilities, and error
diagnostics.  None of those arrays may enter formal inference.

## Posterior, prior, and correlated candidate set

Training uses a target-aware privileged posterior:

```text
q_phi(Z_set | H_long, S_1:9, X_target_set, X_t, t, endpoint_mask).
```

Deployment uses a long-only conditional prior:

```text
p_psi(Z_set | H_long, e_long, s_long, G_long).
```

`G_long` may contain only summaries actually produced by the same retained long forward.  The
initial implementation uses the authenticated long Camera tokens, baseline pose encoding, and
prediction scale; it must not fabricate depth, point, track, or iteration traces.

The four latents are one correlated joint random variable:

```text
z_k = mu_k + A_k eps_shared + sigma_k eps_k,
eps_shared ~ N(0,I), eps_k ~ N(0,I), sigma_k > 0.
```

The resulting low-rank-plus-diagonal covariance is positive definite, so posterior-to-prior KL
is computed on the exact stacked Gaussian.  Because the endpoint set is unordered while a
stacked Gaussian is slot-sensitive, enumerate all `4! = 24` block permutations of the
posterior.  For permutation matrix `P_pi`, compute

```text
KL_pi = KL(N(P_pi mu_q, P_pi Sigma_q P_pi^T) || N(mu_p, Sigma_p))
L_KL = min_pi KL_pi.
```

Use lexicographic permutation order as the deterministic tie-break and test invariance to any
input endpoint permutation.  Four independent samples, an unmatched slotwise KL, and a
singular shared-noise-only covariance are forbidden.

Endpoint order is randomized during training.  The model and losses are permutation
equivariant/invariant; slot IDs do not mean left/right or fixed modes.

## Hierarchical flow

Use rectified flow in endpoint space:

```text
X_0 = 0
X_t = t * X_target
v_target = X_target.
```

One sampled `Z_set` remains fixed for all ODE steps.  The temporal hierarchy uses ten
50-frame coarse block means plus within-block zero-mean local residuals.  The coarse branch is
decoded first; the local branch is conditioned on coarse state.  This local support replaces
global DCT modes and prevents a correction from leaking into an uncovered region.

Losses are:

- coverage- and quality-weighted Huber flow matching;
- bidirectional soft-Chamfer set matching between four predicted and four teacher endpoints;
- exact-zero uncovered anchor;
- relative-motion losses at lags `1,5,10,25`;
- correction-only second-difference and magnitude trust region;
- exact joint Gaussian KL with warm-up/free bits during prior distillation;
- optional weak diversity only if measured collapse is below the teacher set's actual spread.

A prediction-only ranking head is optional.  It cannot rescue an unsafe candidate generator:
uniform/random four-candidate utility must pass before selector gains count.

## Stage gates

### A-prime: endpoint construction

Start with `scene0029_01`, then all ten calibration scenes.  Require:

- exact schemas, provenance, finite arrays, hashes, and physical leakage audit;
- `X_target[~mask]` is bitwise zero;
- maximum covered center round-trip error divided by `s_long` is `< 1e-5`;
- maximum uncovered decoded-center drift divided by `s_long` is `< 1e-8`;
- maximum corrected-vs-baseline SO(3) geodesic delta is `<= 1e-6 deg`, and raw pose-encoding
  quaternion bytes `[3:7]` and FOV bytes `[7:9]` are exactly equal;
- per endpoint, covered utility uses frozen-oracle translation RMS on its own coverage mask;
  per-scene retention is the mean corrected covered utility divided by the mean raw-teacher
  covered utility, with a finite strictly positive denominator; the equal-scene mean retention
  must be `>= 0.95`;
- per-scene full utility is the mean of its four endpoint full-scene utilities; all `10/10`
  scenes must be positive, their equal-scene mean must be positive, and the minimum must be
  `>= 0`.

The target sidecar stores finite baseline-filled raw-gauge teacher centers `[4,500,3]`, the
coverage mask, endpoints, and the authenticated teacher-reference digest.  The independent
verifier re-decodes the authenticated long/short Camera tokens, repeats raw-gauge alignment
and fusion from the registered weights/masks, and proves that the stored centers/endpoints are
the unique result.  It compares pose-encoding quaternion `[3:7]` and FOV `[7:9]` directly;
C2W replay alone is insufficient to verify FOV.

### B: posterior upper bound

First memorize the four endpoints for `scene0029_01`, then train on eight scenes and evaluate
on two replay scenes.  Require matched endpoint utility retention `>= 0.90`, positive uniform
four-candidate utility, per-scene harm `>= -0.01`, and a bidirectional set metric better than a
deterministic mean/no-latent control.

### C: long-only prior

Distill only after B passes.  Require positive prior uniform and selected utility, selected
utility no worse than uniform, per-scene harm `>= -0.01`, and prior selected utility at least
`70%` of posterior selected utility.  Report first, uniform, prediction-only selected, oracle,
and exact no-op.

## Method relationship and boundary

- [VRFM](https://arxiv.org/abs/2502.09616): target-aware latent disambiguates ambiguous
  velocity fields; one latent sample stays fixed along the ODE.
- [VFP](https://arxiv.org/abs/2508.01622): the long-conditioned prior learns the privileged
  posterior rather than assuming a fixed standard Gaussian.
- [DLow](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/794_ECCV_2020_paper.php):
  one shared random source creates a correlated candidate set.
- [TrajFlow](https://arxiv.org/abs/2506.08541): jointly interacting K outputs and optional
  ranking, without making the ranker a safety gate.
- [Hierarchical Rectified Flow](https://arxiv.org/abs/2502.17436): coarse/local velocity
  hierarchy, not its full multi-model recipe.
- [IMLE flow distillation](https://arxiv.org/abs/2603.09415): bidirectional set matching for
  teacher coverage and candidate fidelity.

VGGT-Long chunk/loop processing and VGGT-Align multi-chunk/test-time adaptation violate the
single-backbone contract.  They are used only as evidence that long-sequence scale and global
consistency are important; they are not deployment components.
