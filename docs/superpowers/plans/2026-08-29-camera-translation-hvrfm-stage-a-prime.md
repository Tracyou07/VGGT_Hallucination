# Camera-Translation H-VRFM Stage A-prime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use `superpowers:subagent-driven-development`
> or `superpowers:executing-plans`.  Implement one task at a time, run the named RED/GREEN
> commands, and obtain fresh independent review before advancing.

**Goal:** Replace the failed high-dimensional Camera-token lift with a lossless, strictly
separated batch of four normalized 500-frame translation endpoints per scene, then verify the
hardest one-scene smoke and the frozen ten-scene calibration on H20.

**Architecture:** Long and short Camera tokens remain immutable conditions.  Decode and align
short predictions in the raw long gauge, fuse only camera centers using the existing signed
privileged quality/mask sidecar, and encode the result as normalized corrections to the first three
`absT_quaR_FoV` Camera Head outputs.  Quaternion and FOV are copied from the long baseline.

**Spec:** `docs/superpowers/specs/2026-08-29-camera-translation-hvrfm-design.md`

**Output:** `/data/yjh/output/vggt/camera_translation_hvrfm/<run_id>/`

## Frozen inputs and constraints

- Read the verified failed run
  `/data/yjh/output/vggt/privileged_conditional_hvrfm/privileged_teacher_lift_20260829T012716Z_tolfix`
  only for authenticated provenance, role split, formal quality masks, and comparison metrics.
- Do not mutate that run or treat its token residuals as training targets.
- Use the authenticated source run and frozen VGGT Camera Head already bound by its config.
- Smoke scene is `scene0029_01`; calibration is exactly the existing eight train and two
  validation scenes.
- Formal target construction may use the frozen Camera Head on H20.  Formal inference design
  remains one 500-frame backbone and no short window.
- Refuse below 100 GiB free, on dirty/wrong worktree, wrong host/user/commit, unavailable GPU,
  changed checkpoint/source/formal hashes, active conflicting job, or nonempty stage stderr.

## Task 1: Translation geometry and strict artifact schemas

**Files**

- Create: `pre_experiments/camera_translation_hvrfm/__init__.py`
- Create: `pre_experiments/camera_translation_hvrfm/geometry.py`
- Create: `pre_experiments/camera_translation_hvrfm/artifacts.py`
- Create: `tests/camera_translation_hvrfm/__init__.py`
- Create: `tests/camera_translation_hvrfm/test_geometry.py`
- Create: `tests/camera_translation_hvrfm/test_artifacts.py`

**RED tests**

- `prediction_scale` is finite, positive, and computed only from baseline C2W.
- `build_translation_endpoint` returns exact `[4,500,3]` float32, zeros uncovered entries,
  and rejects wrong masks, frames, rotations, scales, NaN/Inf, or reflections.
- Uncovered teacher entries may be NaN; construction must use `where`/indexed assignment, never
  `mask * value`, and must emit bitwise-zero uncovered endpoints plus finite baseline-filled
  raw-gauge teacher centers.
- `apply_translation_endpoint` changes only pose-encoding `[...,0:3]`; quaternion/FOV are
  bitwise equal to baseline; zero endpoint is bitwise no-op.
- Round-trip recovered centers match hybrid teacher centers on covered frames and baseline
  centers elsewhere.
- A common global Sim(3) transformation leaves the normalized endpoint invariant within
  explicit float64 tolerance.  The test transforms the full camera system using
  `c'=a Q c+b`, `R_w2c'=R_w2c Q^T`, and `T_w2c'=a*T_w2c-R_w2c Q^T b`, not centers alone.
- Each NPZ schema uses exact members/shapes/dtypes, rejects object arrays/extra keys/symlinks,
  binds sample IDs and SHA-256 provenance, and writes atomically.

Run RED:

```bash
python -m unittest tests.camera_translation_hvrfm.test_geometry \
  tests.camera_translation_hvrfm.test_artifacts -v
```

Implement the smallest geometry/schema code, then run GREEN, `compileall`, and `git diff
--check`.  Commit only after fresh review.

## Task 2: Raw-gauge short teacher and physical separation

**Files**

- Create: `pre_experiments/camera_translation_hvrfm/teacher.py`
- Create: `pre_experiments/camera_translation_hvrfm/data.py`
- Create: `tests/camera_translation_hvrfm/test_teacher.py`
- Create: `tests/camera_translation_hvrfm/test_data.py`

**Behavior**

- Load the authenticated source shard; decode global and nine short Camera-token predictions
  with the frozen Camera Head while retaining the baseline pose encoding.
- Align each short prediction to the long prediction using prediction-to-prediction Sim(3).
- Consume only the signed, digest-bound window weights/masks from the quality sidecar to fuse
  raw-gauge teacher centers.  Raw GT/oracle coordinates cannot enter coordinate/alignment/gauge
  calculations; their only allowed indirect influence is through those privileged weights and
  masks.  The numeric endpoint builder must not accept GT or oracle arrays.
- Publish prediction-only long context, training-only short context, translation target, and
  independent quality sidecar under separate roots with stable sample IDs.
- Mutation tests must prove the dependency boundary: changing GT/oracle diagnostic arrays while
  holding registered masks/weights fixed does not change the numeric endpoint, while changing
  masks/weights does.
- Re-decode/authenticate the baseline against the source and refuse any provenance mismatch.

Run RED/GREEN:

```bash
python -m unittest tests.camera_translation_hvrfm.test_teacher \
  tests.camera_translation_hvrfm.test_data -v
```

Run Tasks 1-2 together, legacy conditional tests, compile, and diff check before review.

## Task 3: Evaluation, classification, and signed report

**Files**

- Create: `pre_experiments/camera_translation_hvrfm/evaluate.py`
- Create: `pre_experiments/camera_translation_hvrfm/report.py`
- Create: `tests/camera_translation_hvrfm/test_evaluate.py`
- Create: `tests/camera_translation_hvrfm/test_report.py`

**Behavior**

- Replay in the one frozen baseline-to-GT gauge without refitting alignment.
- Report per scene/endpoint covered and full translation utility, teacher retention, maximum
  normalized covered center round-trip, maximum normalized uncovered drift, maximum rotation
  delta, raw pose quaternion/FOV byte equality, endpoint RMS, coverage, and provenance.
- Freeze aggregation exactly as the spec: endpoint utilities use their own masks, per-scene
  retention is the ratio of mean corrected/mean teacher covered utilities, and per-scene full
  utility is the mean of four endpoint full-scene utilities.  Reject nonpositive teacher
  denominators.
- Gate exactly the Stage A-prime thresholds in the spec.  Unit tests independently trip every
  gate and reject missing/extra scenes or endpoints.
- Emit deterministic JSON and concise Markdown plus a digest-bound completion record.

Run:

```bash
python -m unittest tests.camera_translation_hvrfm.test_evaluate \
  tests.camera_translation_hvrfm.test_report -v
```

## Task 4: Fail-closed pipeline and independent verifier

**Files**

- Create: `pre_experiments/camera_translation_hvrfm/pipeline.py`
- Create: `tests/camera_translation_hvrfm/test_pipeline.py`

**Stages**

```text
preflight -> prepare -> smoke -> calibration -> report -> verify
```

- `prepare` authenticates every upstream manifest/file and publishes immutable separated
  inputs without constructing the result root before preflight succeeds.
- `smoke` is only `scene0029_01` and must pass structural round-trip before calibration.
- `calibration` is exactly ten scenes/four endpoints.
- `verify` inventories/hashes every output, re-decodes the authenticated long and short Camera
  tokens, repeats raw-gauge alignment/fusion from registered weights/masks, compares the stored
  finite baseline-filled teacher centers and teacher-reference digest, replays endpoints through
  the real frozen pose conversion, compares raw quaternion/FOV slices, recomputes all
  metrics/gates, audits prediction-only members, and writes
  `verified_completion.json` only after success.
- Resume is stage-signed and refuses partial/foreign/corrupt artifacts.  Failed runs are
  preserved.

Run focused and grouped tests plus both predecessor suites.

## Task 5: H20 runner

**Files**

- Create: `scripts/h20/run_camera_translation_hvrfm_targets.sh`
- Create: `tests/camera_translation_hvrfm/test_h20_runner.py`

The real `--preflight-only` path must behaviorally prove host/user, exact clean commit, H20
availability, selected free GPU, disk threshold, checkpoint/source/formal files and hashes,
ordered stages, output root, and zero secret/token use.  The formal path logs every stage,
captures stderr separately, refuses concurrent duplicate runs, and never deletes partials.

Run shell syntax plus behavioral fixtures.  Independently review the final runner blob.

## Task 6: Formal H20 execution

1. Re-run all new/legacy CPU tests locally and on H20 at the exact pushed commit.
2. Run read-only preflight.
3. Launch one formal run with a unique run ID.
4. Monitor process, selected GPU, disk, stage completion, logs, and artifact counts without
   stopping/restarting a healthy run.
5. Require smoke pass before automatic ten-scene calibration.
6. After runner RC 0 and empty stderr, independently execute `verify` again.
7. Inspect `verified_completion.json`, report JSON/Markdown, every gate, hashes, and total size.
8. Preserve all inputs and failed runs; do not pull large artifacts to the Windows host.

When A-prime passes, write a separate reviewed implementation plan for posterior/flow Stage B;
do not improvise Stage B inside this target-construction plan.  That follow-on plan must test
the spec's `4!` permutation-minimized exact joint Gaussian KL and lexicographic tie-break.
