# Long–Short Camera Head Fine-Tuning Design

## Objective

Fine-tune VGGT's native Camera Head so that a single 500-frame long-window Camera-token
sequence decodes with the local accuracy exposed by overlapping 100-frame predictions.
Short-window tokens and ground-truth geometry are training-only supervision. Formal
inference accepts only the long-window Camera tokens and the fine-tuned Camera Head.

All formal outputs live below
`/data/yjh/output/vggt/long_short_camera_head/<run_id>/` on H20. Large artifacts remain
on H20 and are not copied back to the Windows machine.

## Motivation and Scientific Claim

The completed VRFM studies found useful candidate corrections but did not find reliable
evidence for a small number of discrete latent branches. VRFM directions were not special
under the matched random-basis control, and post-hoc ranking could not avoid rare,
high-confidence catastrophic choices. The most economical source-level hypothesis is
therefore that long-context decoding trades away local accuracy, rather than that inference
must choose among several discrete solutions.

This experiment changes the native Camera Head itself. It does not add a candidate
generator, selector, safety gate, or runtime short-window pass. A positive result means
that the frozen long-window Camera tokens already contain enough information for a better
decoder. A negative result localizes the remaining problem upstream in the Aggregator.

## Scope

The experiment will:

- initialize a student Camera Head from the verified local VGGT-1B checkpoint;
- keep the VGGT Aggregator and every non-camera head frozen;
- train the final Camera Head transformer block and decoder/modulation layers;
- use cached 500-frame Camera tokens as the only student input;
- use nine cached 100-frame Camera-token sequences and raw ScanNet poses only while
  constructing training labels and losses;
- compare `gt_only` and `long_short` objectives under the same initialization, scene
  order, update count, optimizer, and evaluation protocol;
- run a one-scene engineering smoke before an eight-train/two-validation calibration;
- export a loadable Camera Head checkpoint, long-only predictions, scalar metrics,
  provenance, and a concise human report.

The experiment will not fine-tune the 1B-parameter Aggregator, run RGB/depth heads, add
an external residual network, sample a latent `z`, refit alignment per candidate, or claim
fresh final-test generalization from the two previously observed validation scenes.

## Inputs and Data Separation

The authenticated ten-scene source run is
`/data/yjh/output/vggt/variational_camera_latent/vrfm_camera_20260827T044926Z`.
Its source manifest defines eight `train` scenes and two `validation` scenes. Manifest
paths may contain the pre-migration output root; loaders rebase each record by scene name
under the explicitly supplied source run, then verify the recorded SHA-256.

Prediction-only inputs contain frame IDs, 500-frame Camera tokens, nine 100-frame Camera
token sequences, and the frozen baseline camera trajectory. Raw GT poses are read from
`/data/yjh/share/datasets/ScanNet/processed_cva02_v1/<scene>/` only by the privileged
label builder and evaluator.

The new run publishes two physically separate families:

- `prediction_only/long_context/<scene>.npz`: only long tokens, frame IDs, baseline
  trajectory, and prediction provenance;
- `privileged_labels/training/<scene>.npz`: raw GT poses, frozen scene alignment,
  quality-weighted local-teacher targets, masks, and diagnostics.

The inference API accepts a long-context shard and checkpoint only. It must reject the
training source schema and must not accept a prepared/GT path or short-window tensor.

## Teacher Target Construction

A frozen copy of the original Camera Head decodes all nine short-window token sequences.
Each 100-frame teacher trajectory is aligned to the matching segment of the frozen global
prediction with a prediction-only orientation-preserving Sim(3). The transform is fitted
on the complete 100-frame window and then applied unchanged.

For evaluation and supervised loss, one frozen scene Sim(3) is fitted once from the
complete 500-frame baseline prediction to raw GT. It is never refitted to student output.

For each local window:

1. compute baseline and aligned-teacher RMS translation error under the frozen scene
   transform;
2. define positive teacher utility as
   `max(0, (baseline_rms - teacher_rms) / max(baseline_rms, eps))`;
3. use that utility as the window's teacher weight, capped at one;
4. blend overlapping teacher poses using their positive weights;
5. mark frames with no positive teacher weight as teacher-invalid.

This uses GT only to decide how strongly a training example should teach the student.
Bad short-window predictions do not become targets. The raw GT, weights, and teacher
targets remain privileged and never enter inference.

## Trainable Model

The student is the repository's native `vggt.heads.camera_head.CameraHead`. It is loaded
from the verified local checkpoint. By default the trainable scope is:

- the final transformer block in `camera_head.trunk`;
- `trunk_norm`;
- `embed_pose` and `poseLN_modulation`;
- `pose_branch` and `empty_pose_tokens`.

`token_norm` and earlier trunk blocks stay frozen. The checkpoint records the exact
trainable parameter names and count. The student always decodes four Camera Head
iterations, matching the authenticated baseline.

## Losses

The student output is converted differentiably to camera-to-world matrices and transformed
by the frozen baseline-to-GT Sim(3). All translation losses are normalized by the frozen
GT trajectory scale.

Both variants use:

- `L_gt_translation`: robust per-frame camera-center loss to raw GT;
- `L_relative_translation`: robust displacement loss at lags 1, 5, 10, and 25;
- `L_rotation`: rotation-matrix consistency to raw GT after the frozen scene rotation;
- `L_anchor`: a small pose-encoding penalty to the original Camera Head output.

The `long_short` variant additionally uses:

- `L_teacher`: robust camera-center loss to the quality-weighted aligned short-window
  target, evaluated only where teacher weight is positive.

The `gt_only` control sets the teacher coefficient to zero. Default weights are
`1.0, 0.5, 0.1, 0.01, 0.5` for GT translation, relative translation, rotation, anchor,
and teacher respectively. No loss coefficient is tuned on the two validation scenes.

## Training Protocol

The one-scene smoke uses `scene0000_00`, 20 updates, batch size one, BF16 autocast, AdamW,
learning rate `2e-6`, weight decay `1e-4`, and gradient clipping at one. It passes only if
all losses and gradients remain finite, the final loss is below the initial loss, the
checkpoint reloads exactly, and long-only inference produces finite 500-frame output.

Calibration trains two matched variants on the eight manifest `train` scenes for at most
400 updates. Every 25 updates it evaluates the two locked replay scenes without gradients.
The best checkpoint is selected by mean frozen-oracle validation RMS; patience is 100
updates. Random seeds, scene order, and optimizer settings are identical between variants.

GPU work runs only on currently idle H20 devices after checking identity, disk, active
processes, and memory. The runner fails closed below 100 GiB free on `/data`, on a dirty
worktree, on a non-H20 device, or when the verified ScanNet/source markers are absent.

## Evaluation and Acceptance

Every checkpoint is evaluated with long-only inputs against the untouched original Camera
Head on all ten scenes. The evaluator applies the baseline-frozen scene transform to both
baseline and student output; it never fits a student-specific transform.

Report full-scene and overlap translation RMS, relative-translation error at lags
1/5/10/25, rotation error, per-scene utility, and correction magnitude. Training scenes
are diagnostic. The two validation scenes are explicitly labeled `locked_replay`, not
fresh final validation.

The source-level hypothesis is `PROMISING` only if `long_short`:

- has positive mean utility on the two locked-replay validation scenes;
- does not worsen either validation scene by more than 1%;
- beats the matched `gt_only` control in mean validation RMS;
- has no non-finite output and no rotation-error increase above 0.1 degree mean;
- uses no short-window or privileged input in its inference artifact.

Otherwise the result is `NO_SOURCE_HEAD_SIGNAL` or `HEAD_ONLY_INSUFFICIENT`. A failed
gate does not delete checkpoints or metrics, but no checkpoint is labeled deployable.

## Artifacts and Completion

Each run contains:

```text
/data/yjh/output/vggt/long_short_camera_head/<run_id>/
  config.json
  logs/
  manifests/
  prediction_only/long_context/
  privileged_labels/training/
  smoke/
  calibration/gt_only/
  calibration/long_short/
  evaluation/prediction_only/
  evaluation/privileged_labels/
  reports/report.json
  reports/human_report.md
  verified_completion.json
```

`verified_completion.json` binds the Git commit, source/checkpoint digests, split,
configuration, checkpoint hashes, prediction/evaluation manifests, report hash, test
results, and final classification. Completion verification reloads every NPZ with
`allow_pickle=False`, checks exact member sets/shapes/finiteness, verifies every SHA-256,
and confirms that inference provenance contains only long-context fields.
