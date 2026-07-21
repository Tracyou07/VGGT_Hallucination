# Local-Global Dataset Construction Design

## Purpose And Scope

This protocol defines one reproducible data pipeline for three different
artifacts:

1. a ScanNet scene-screening table for finding long-context degradation;
2. a Round 2 benchmark for validating prediction-only local-global scores;
3. a future DiT dataset of global-to-local Camera latent correction pairs.

Only the first two artifacts are implemented in Round 2. Latent pairs remain
blocked until a frozen-Camera-Head replacement or blending experiment proves
that the proposed local target improves trajectory quality.

## Data Identity And Eligibility

The atomic split unit is a physical ScanNet scene, identified by the
`sceneXXXX` prefix. All scans such as `scene0000_00` and `scene0000_01` must
remain in the same train, validation, or test split. A scan is eligible when it
has at least 500 ordered frame IDs with both an image and a finite raw 4x4 GT
pose.

Construct one deterministic 500-frame sequence by uniform index selection over
the eligible ordered IDs. Every later global run, 200-frame subset, and local
window references this exact sequence. A local window is contiguous in this
selected sequence, not necessarily contiguous in the original video frame
numbers. Store both selected-sequence indices and original frame IDs.

Every artifact records dataset version, source scan, physical-scene group,
frame IDs, selection algorithm, checkpoint hash, code commit, preprocessing
mode, and random seed. Changing any of these creates a new dataset version.

## Stage A: Global Screening

Run frozen, camera-only VGGT at four Camera Head iterations on nested 200- and
500-frame selections. Prediction metrics are independently Sim(3)-aligned to
raw GT. GT remains raw and is used only to construct research labels.

For each scan, record at least:

- aligned ATE and ARE at 200 and 500 frames;
- absolute and relative ATE growth;
- aligned per-frame translation and rotation error growth on shared frames;
- Camera Token drift on shared frames;
- Sim(3) scale and runtime diagnostics.

Assign screening strata from the distribution within the construction pool,
not from manually selected examples:

- `long_context_failure`: high positive 200-to-500 degradation;
- `stable_easy`: low degradation and low absolute error;
- `stable_hard`: low degradation but high absolute error;
- `ambiguous`: conflicting ATE/ARE direction, unstable validity, or values near
  stratum boundaries.

Quantile boundaries and minimum absolute-change guards are fitted on the
training partition only, then frozen. Manual observations such as
`scene0000_00` may seed development but never define test membership.

Screen candidates in batches until the benchmark contains at least 20 physical
scenes in each of the three primary strata. Preserve at least 20 additional
physical scenes as a held-out mixed test set. If failures are rare, expand the
screening pool instead of weakening the positive definition.

## Stage B: Round 2 Consistency Benchmark

For every selected 500-frame scan, run nine frozen local windows of length 100
and stride 50. Retain external raw artifacts needed for analysis: normalized
Camera Tokens, activated camera encodings, raw predicted poses, frame IDs, and
window boundaries. Git receives only scalar CSV/JSON exports.

Create one row per frame and local observation. Prediction-only columns include:

- local-local token cosine distance;
- local-local pose residual after prediction-to-prediction Sim(3);
- global-local token cosine distance;
- global-local pose residual after prediction-to-prediction Sim(3);
- number of local observations and distance from the nearest window boundary.

GT evaluation columns are stored separately and never passed to a detector:

- independently aligned global and local translation/rotation errors;
- global error minus median local error;
- screening stratum.

Fit local reliability thresholds using only prediction-side values in the
training/validation stable controls. Freeze thresholds before opening the test
labels. Test scenes must not influence score choice, threshold choice, window
size, or weighting.

## Prediction-Only Pair States

After Round 2, assign each internal frame one state using frozen prediction-only
rules:

- `positive_candidate`: at least two local observations agree, while the global
  result differs from them;
- `stable_zero`: local observations agree and the global result also agrees;
- `ambiguous`: local observations disagree, coverage is insufficient, or a
  score falls inside an uncertainty margin.

No raw or aligned GT value may change a frame state. GT is allowed only to
measure, over an entire split, whether `positive_candidate` frames usually have
better local predictions and whether `stable_zero` frames require no change.

When two reliable local tokens are available, choose the observation whose
frame lies farther from its window boundary. Resolve exact ties by lower window
start index. Do not average tokens. The selected token is a local reference,
never a GT latent.

## Stage C: Future Latent Pair Dataset

Stage C starts only if Round 3 demonstrates that selected local-token
replacement or residual blending improves aligned trajectory metrics across
held-out scenes. Build sequence records containing:

```text
global_tokens        [S, 2048]
target_tokens        [S, 2048]
pair_state           [S]
correction_mask      [S]
training_weight      [S]
prediction_conditions[S, K]
frame_ids            [S]
window_source        [S]
```

For `positive_candidate`, the target is the deterministically selected local
token. For `stable_zero`, the target equals the global token and the residual
target is zero. `ambiguous` frames have zero training weight. Conditions may
include frozen local-local and global-local scores but never GT errors or
screening labels.

Store high-dimensional tensors outside Git in sharded safetensors files with a
JSONL or Parquet index. Each shard has a SHA-256 digest and references an
immutable manifest. Scalar benchmark exports remain in Git. Dataset generation
must be resumable and must reject mixed checkpoint, code, preprocessing, or
frame-selection identities.

## Split And Sampling Policy

Freeze physical-scene splits before tuning detector thresholds or model
hyperparameters. Development observations belong only to train/validation.
Never move a scene after inspecting test performance.

Frame count is not effective sample count. During future training:

- sample physical scenes approximately uniformly;
- balance positive candidates and stable-zero examples;
- cap temporally adjacent frames from one scene per batch or epoch;
- preserve ambiguous frames for audit but exclude them from the loss;
- report results by scene and stratum, not only pooled frame averages.

An initial DiT feasibility run may use roughly 100 physical scenes, but a
generalization claim requires expansion to several hundred training scenes and
at least 50 untouched test scenes. Additional windows from one scene do not
replace scene diversity.

## Quality Gates

The pipeline advances only when all applicable gates pass:

1. **Identity gate:** no physical-scene leakage and every artifact matches its
   manifest, checkpoint, commit, preprocessing, and frame IDs.
2. **Replay gate:** saved global outputs reproduce the published Round 1.5
   global result for matching scenes and frames.
3. **Detection gate:** at least one prediction-only score separates both
   failure scenes from stable controls and preserves direction on held-out data.
4. **Reliability gate:** local-local filtering improves the relationship between
   global-local score and aligned error growth.
5. **Intervention gate:** frozen-head local replacement or blending improves
   held-out trajectories without materially degrading stable controls.
6. **Pair-release gate:** only after gate 5 may `latent-pairs-v1` be generated.

Version the outputs independently as `screen-v1`, `consistency-v1`, and
`latent-pairs-v1`. A failed gate remains a recorded negative result; it must not
be bypassed by selecting different test scenes.
