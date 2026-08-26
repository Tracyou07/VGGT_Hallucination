# Camera Velocity Ambiguity 02 Implementation Plan

> **For implementers:** execute this plan task by task with test-driven development.
> Never start H20 GPU inference until the ScanNet 100/100 integrity marker passes.

**Goal:** Build a calibration-first ScanNet-50 pre-experiment that tests whether
adjacent local VGGT windows provide one useful repair direction, a selector
problem, continuous repair redundancy, or independently supported multimodal
camera velocity.

**Architecture:** A new `camera_velocity_ambiguity_02` package consumes a frozen
protocol and either generates or validates global/local VGGT camera artifacts.
Prediction-only geometry is isolated from privileged GT evaluation, RGB-D
observation energy, and presentation-only FastVGGT plots. An authenticated state
machine enforces calibration -> freeze -> development -> decision.

**Tech stack:** Python 3.10, NumPy, PyTorch/VGGT for H20 inference, unittest,
Matplotlib, optional evo/scipy for the exact FastVGGT reproduction plot, Bash.

**Frozen identities:**

- base commit: `a85bcba9356be72d00f970e948ffc461f58c95e8`
- branch: `codex/camera_velocity_ambiguity_02_pre_experiment`
- FastVGGT source: `6526e275a29572653a034762bb3c6c9ce280ff55`
- parent split digest:
  `69c283245c4f220965e6fde3b96192de298e292eb8ca625c94851fe8932cdb8a`
- output root: `/data/output/camera_velocity_ambiguity/<run_id>/`

## Ownership and integration order

- **Person A — Data/protocol/prediction:** Tasks 1–4.
- **Person B — Geometry/evidence/decision:** Tasks 5–9.
- **Person C — FastVGGT reproduction/reporting:** Tasks 10–12.
- **Joint integration:** Tasks 13–14, with a non-author reviewing each prior part.

The three owners work in disjoint files after Task 1 freezes interfaces. Merge
in order A -> B -> C. Every production change starts with a failing CPU test.

## Task 1: Freeze protocol and authenticated split v2

**Owner:** Person A

**Files:**

- Create: `pre_experiments/camera_velocity_ambiguity_02/__init__.py`
- Create: `pre_experiments/camera_velocity_ambiguity_02/contracts.py`
- Create: `pre_experiments/camera_velocity_ambiguity_02/protocol.py`
- Create: `configs/scannet50_camera_velocity_ambiguity_02_split_v2.json`
- Test: `tests/camera_velocity_ambiguity_02/test_protocol.py`

**Step 1: Write failing tests**

Cover exact scene order, parent v1 digest, unchanged 10/40 membership,
`development_evaluation` naming, frame counts 500/430, windows 449, adjacent
pairs 399, primary 398, secondary 1, calibration primary 80, development
primary 318, and alpha tuple `(0, .25, .5, .75, 1)`.

Add tamper tests for any changed count, scene, parent digest, frame-selection
policy or config digest.

**Step 2: Run RED**

```bash
python -m unittest tests.camera_velocity_ambiguity_02.test_protocol -v
```

Expected: import/config failures.

**Step 3: Implement minimally**

Implement canonical JSON hashing, `EvidenceSource`, `ProtocolViolation`, typed
cohort names and `load_protocol_v2()`. The loader must call the existing v1
split validator, then verify that v2 copies membership/order exactly.

Mechanically derive counts with existing `build_sliding_windows`; never trust
declared count fields without recomputation.

**Step 4: Run GREEN and full baseline**

```bash
python -m unittest tests.camera_velocity_ambiguity_02.test_protocol -v
python -m unittest discover -s tests -v
```

**Step 5: Commit**

```bash
git add pre_experiments/camera_velocity_ambiguity_02 configs/scannet50_camera_velocity_ambiguity_02_split_v2.json tests/camera_velocity_ambiguity_02
git commit -m "Add CVA02 frozen protocol"
```

## Task 2: Add data-integrity and GPU launch gate

**Owner:** Person A

**Files:**

- Create: `pre_experiments/camera_velocity_ambiguity_02/input_gate.py`
- Test: `tests/camera_velocity_ambiguity_02/test_input_gate.py`
- Test: `tests/camera_velocity_ambiguity_02/test_no_gpu_before_integrity.py`

**Step 1: Write failing tests**

Use temporary files only. Require an authenticated `verified_completion.json`
covering 100 assets, official scene-list digest, Content-Length, local SHA-256,
H20 SHA-256 and remote root. Test missing marker, wrong root, missing asset,
changed digest, stale marker and non-finite/invalid metadata.

Monkeypatch model/device loading to raise and prove it is never reached when
the marker fails.

**Step 2: Run RED**

```bash
python -m unittest tests.camera_velocity_ambiguity_02.test_input_gate tests.camera_velocity_ambiguity_02.test_no_gpu_before_integrity -v
```

**Step 3: Implement minimally**

Return an immutable verified-input contract. Do not duplicate the downloader
or verifier. The gate consumes their final marker and exact path identity.

**Step 4: Run GREEN and commit**

```bash
python -m unittest tests.camera_velocity_ambiguity_02.test_input_gate tests.camera_velocity_ambiguity_02.test_no_gpu_before_integrity -v
git add pre_experiments/camera_velocity_ambiguity_02/input_gate.py tests/camera_velocity_ambiguity_02
git commit -m "Gate CVA02 on verified ScanNet inputs"
```

## Task 3: Implement FastVGGT-compatible frame selection and artifact schemas

**Owner:** Person A

**Files:**

- Create: `pre_experiments/camera_velocity_ambiguity_02/frames.py`
- Create: `pre_experiments/camera_velocity_ambiguity_02/artifacts.py`
- Test: `tests/camera_velocity_ambiguity_02/test_frames.py`
- Test: `tests/camera_velocity_ambiguity_02/test_artifacts.py`

**Step 1: Write failing tests**

Test valid image/finite-pose intersection; preserved first frame; floor-stride
selection; exact 500/430 counts; strictly increasing IDs; non-contiguous raw
IDs; missing/duplicate/extra IDs; normal 9-window and exceptional 8-window
layouts; exact NPZ members; finite arrays; atomic completion semantics; and
schema/provenance mismatch on resume.

**Step 2: Run RED**

```bash
python -m unittest tests.camera_velocity_ambiguity_02.test_frames tests.camera_velocity_ambiguity_02.test_artifacts -v
```

**Step 3: Implement minimally**

Port only the behavior of FastVGGT `build_frame_selection`; do not import its
large evaluation module. Save prediction-only NPZ members separately from raw
GT/oracle inputs. Completion sidecars bind run ID, frame digest, checkpoint,
commit, preprocess=`crop`, camera iterations and protocol digest.

**Step 4: Run GREEN and commit**

```bash
python -m unittest tests.camera_velocity_ambiguity_02.test_frames tests.camera_velocity_ambiguity_02.test_artifacts -v
git add pre_experiments/camera_velocity_ambiguity_02 tests/camera_velocity_ambiguity_02
git commit -m "Add CVA02 frame and artifact contracts"
```

## Task 4: Build resumable global/local camera prediction runner

**Owner:** Person A

**Files:**

- Create: `pre_experiments/camera_velocity_ambiguity_02/predict.py`
- Test: `tests/camera_velocity_ambiguity_02/test_predict.py`

**Step 1: Write failing fake-model tests**

Verify camera-only configuration, `crop` preprocessing, 4 Camera Head
iterations, exact model call shapes, global once per scene, local once per
window, frame identity, raw c2w conversion, Camera Token extraction, shard
selection, resume, `.tmp` rejection and no network fallback.

The fake model must run on CPU and require no checkpoint, CUDA or ScanNet.

**Step 2: Run RED**

```bash
python -m unittest tests.camera_velocity_ambiguity_02.test_predict -v
```

**Step 3: Implement minimally**

Reuse existing checkpoint/device helpers and VGGT pose decoding. Keep all model
logic in this module; scientific modules must never import torch or VGGT.

Write per-scene global and window outputs incrementally. A completed artifact
is reusable only if its exact completion sidecar validates.

**Step 4: Run GREEN and commit**

```bash
python -m unittest tests.camera_velocity_ambiguity_02.test_predict -v
git add pre_experiments/camera_velocity_ambiguity_02/predict.py tests/camera_velocity_ambiguity_02/test_predict.py
git commit -m "Add resumable CVA02 camera predictions"
```

## Task 5: Build overlap units and prediction-only alignment

**Owner:** Person B

**Files:**

- Create: `pre_experiments/camera_velocity_ambiguity_02/units.py`
- Create: `pre_experiments/camera_velocity_ambiguity_02/geometry.py`
- Test: `tests/camera_velocity_ambiguity_02/test_units.py`
- Test: `tests/camera_velocity_ambiguity_02/test_prediction_alignment.py`

**Step 1: Write failing tests**

Cover full-window L/global and R/global alignment; independent gauges; pure
gauge-copy residual zero; fit over all 100 frames rather than shared frames;
signed residuals; cosine/separation/direction agreement; zero-speed
not-evaluable; rank/condition/scale/RMS gates; exact pair identity; and the
398+1 routing including the 20 frames shared by three exceptional windows.

Use a GT sentinel that raises on access and prove this path never touches it.

**Step 2: Run RED**

```bash
python -m unittest tests.camera_velocity_ambiguity_02.test_units tests.camera_velocity_ambiguity_02.test_prediction_alignment -v
```

**Step 3: Implement minimally**

Wrap the existing prediction-to-prediction alignment with stronger diagnostics.
The public alignment signature must not accept GT. Preserve adjacent pair
identity instead of keying only by frame ID.

**Step 4: Run GREEN and commit**

```bash
python -m unittest tests.camera_velocity_ambiguity_02.test_units tests.camera_velocity_ambiguity_02.test_prediction_alignment -v
git add pre_experiments/camera_velocity_ambiguity_02 tests/camera_velocity_ambiguity_02
git commit -m "Add CVA02 prediction-only overlap geometry"
```

## Task 6: Implement one-time frozen global-to-GT oracle

**Owner:** Person B

**Files:**

- Create: `pre_experiments/camera_velocity_ambiguity_02/frozen_oracle.py`
- Test: `tests/camera_velocity_ambiguity_02/test_frozen_oracle.py`

**Step 1: Write failing tests**

Recover known Sim(3) from synthetic 500 and 430 trajectories. Bind transform to
scene/full-frame digest/fit count. Reject subset fitting, candidate-specific
fitting, zero variance, poor condition, invalid scale and non-finite values.
Verify GT remains bytewise unchanged.

Monkeypatch the fitter to raise during candidate evaluation, proving the
already-frozen object is reused for baseline, endpoints and every alpha.

**Step 2: Run RED, implement, run GREEN**

```bash
python -m unittest tests.camera_velocity_ambiguity_02.test_frozen_oracle -v
```

Reuse the existing correct c2w camera-center Umeyama implementation, but expose
an immutable typed transform and strict apply-only API.

**Step 3: Commit**

```bash
git add pre_experiments/camera_velocity_ambiguity_02/frozen_oracle.py tests/camera_velocity_ambiguity_02/test_frozen_oracle.py
git commit -m "Freeze one scene-level CVA02 oracle transform"
```

## Task 7: Implement interpolation, translation metrics and convexity guard

**Owner:** Person B

**Files:**

- Create: `pre_experiments/camera_velocity_ambiguity_02/interpolation.py`
- Test: `tests/camera_velocity_ambiguity_02/test_interpolation.py`

**Step 1: Write failing tests**

Assert alpha endpoints/midpoint, global rotation and FoV preservation, no raw
9D interpolation, frozen-transform identity, ATE/RTE, and convexity for
per-frame L2, mean L2 and RMS. A synthetic violated curve must raise
`ProtocolViolation`, never classify as multimodal.

**Step 2: Run RED, implement, run GREEN**

```bash
python -m unittest tests.camera_velocity_ambiguity_02.test_interpolation -v
```

**Step 3: Commit**

```bash
git add pre_experiments/camera_velocity_ambiguity_02/interpolation.py tests/camera_velocity_ambiguity_02/test_interpolation.py
git commit -m "Add CVA02 residual interpolation and convexity guard"
```

## Task 8: Implement independent RGB-D observation energy

**Owner:** Person B

**Files:**

- Create: `pre_experiments/camera_velocity_ambiguity_02/rgbd_gate.py`
- Test: `tests/camera_velocity_ambiguity_02/test_rgbd_gate.py`

**Step 1: Write failing synthetic tests**

Cover a planar RGB-D scene, wrong translation, bidirectional projection,
free-space penalty, behind-surface occlusion, coverage penalty, insufficient
correspondence, flat energy, boundary scale, fixed scene scale and a synthetic
two-endpoint/interior-barrier case.

Reject any API payload containing GT pose. Changing GT or FastVGGT plot metrics
must leave RGB-D energy unchanged.

**Step 2: Run RED, implement, run GREEN**

```bash
python -m unittest tests.camera_velocity_ambiguity_02.test_rgbd_gate -v
```

Start with a deterministic sparse pixel grid and fixed adjacent-frame edges;
do not optimize thresholds on development data.

**Step 3: Commit**

```bash
git add pre_experiments/camera_velocity_ambiguity_02/rgbd_gate.py tests/camera_velocity_ambiguity_02/test_rgbd_gate.py
git commit -m "Add CVA02 RGB-D observation gate"
```

## Task 9: Add event classifier, controls, freeze state and scene bootstrap

**Owner:** Person B

**Files:**

- Create: `pre_experiments/camera_velocity_ambiguity_02/events.py`
- Create: `pre_experiments/camera_velocity_ambiguity_02/controls.py`
- Create: `pre_experiments/camera_velocity_ambiguity_02/state.py`
- Create: `pre_experiments/camera_velocity_ambiguity_02/statistics.py`
- Test: `tests/camera_velocity_ambiguity_02/test_events.py`
- Test: `tests/camera_velocity_ambiguity_02/test_controls.py`
- Test: `tests/camera_velocity_ambiguity_02/test_state_and_statistics.py`

**Step 1: Write failing tests**

Create synthetic evidence for all four event classes. Require a valid
`OBSERVATION_RGBD` barrier for class four; missing/invalid/presentation-only
barriers make it unidentifiable. Test self, gauge-copy, random wrong-window,
sign inversion, epsilon and degenerate controls.

Test calibration completeness, immutable policy hash, no overwrite,
development-before-freeze rejection, no threshold overrides, exact 10/40 scene
sets, primary-only aggregation and deterministic paired scene bootstrap with
seed 33 and 10,000 samples.

**Step 2: Run RED, implement, run GREEN**

```bash
python -m unittest tests.camera_velocity_ambiguity_02.test_events tests.camera_velocity_ambiguity_02.test_controls tests.camera_velocity_ambiguity_02.test_state_and_statistics -v
```

Unknown metric fields fail closed. Presentation metrics passed to the decision
API raise instead of being silently ignored.

**Step 3: Commit**

```bash
git add pre_experiments/camera_velocity_ambiguity_02 tests/camera_velocity_ambiguity_02
git commit -m "Add CVA02 evidence decisions and frozen state"
```

## Task 10: Vendor FastVGGT trajectory plotting unchanged

**Owner:** Person C

**Files:**

- Create: `pre_experiments/camera_velocity_ambiguity_02/vendor/__init__.py`
- Create: `pre_experiments/camera_velocity_ambiguity_02/vendor/fastvggt_eval_trajectory.py`
- Create: `pre_experiments/camera_velocity_ambiguity_02/vendor/PROVENANCE.md`
- Create: `pre_experiments/camera_velocity_ambiguity_02/fastvggt_plot_adapter.py`
- Modify: `.gitattributes`
- Test: `tests/camera_velocity_ambiguity_02/test_fastvggt_plot_parity.py`
- Test: `tests/camera_velocity_ambiguity_02/test_fastvggt_plot_adapter.py`

**Step 1: Write failing parity tests**

AST-extract the two vendored functions using `lineno/end_lineno`, canonical LF
and no final newline. Require hashes:

- `umeyama_alignment`:
  `5d649d11137eb492fae461f2cc66befadd56a8c05552fd9d26373e83da9aa318`
- `eval_trajectory`:
  `e356c5452c4a92c9ec1f127280c7dd1e58e870ee6818d66ceb8a7aac33ccd81b`

The default test must not import evo or require the FastVGGT audit checkout.

**Step 2: Copy exactly**

Copy upstream lines 54–115 and 174–349 without modifying function bodies,
comments, strings, parameters or error handling. Imports live outside those
functions. Record repository, commit, source path, blob, hashes and identical
root license in provenance.

**Step 3: Add adapter tests and implementation**

The adapter passes upstream-style pred w2c, first-frame-relative GT w2c and
frame IDs with `align=True`; it only saves the returned PIL image and a
sidecar. Fixed fields:

```text
reproduction_only=true
eligible_for_primary_metrics=false
known_coordinate_issue=true
```

Do not watermark or post-process the image.

**Step 4: Run GREEN and commit**

```bash
python -m unittest tests.camera_velocity_ambiguity_02.test_fastvggt_plot_parity tests.camera_velocity_ambiguity_02.test_fastvggt_plot_adapter -v
git add .gitattributes pre_experiments/camera_velocity_ambiguity_02 tests/camera_velocity_ambiguity_02
git commit -m "Vendor FastVGGT trajectory reproduction"
```

## Task 11: Add analysis CLI and metric firewall

**Owner:** Person C

**Files:**

- Create: `pre_experiments/camera_velocity_ambiguity_02/analyze.py`
- Create: `pre_experiments/camera_velocity_ambiguity_02/pipeline.py`
- Test: `tests/camera_velocity_ambiguity_02/test_analysis_pipeline.py`
- Test: `tests/camera_velocity_ambiguity_02/test_metric_firewall.py`

**Step 1: Write failing tests**

Test exact input manifests, separate prediction/oracle/RGB-D/presentation files,
calibration freeze, development consumption, scalar CSV/JSON schemas, primary
and secondary separation, interrupted resume and plot independence.

Inject `fastvggt_*`, `plot_*`, independent-alignment and unknown columns into
decision inputs; each must fail closed.

**Step 2: Run RED, implement, run GREEN**

```bash
python -m unittest tests.camera_velocity_ambiguity_02.test_analysis_pipeline tests.camera_velocity_ambiguity_02.test_metric_firewall -v
```

**Step 3: Commit**

```bash
git add pre_experiments/camera_velocity_ambiguity_02 tests/camera_velocity_ambiguity_02
git commit -m "Add CVA02 calibrated analysis pipeline"
```

## Task 12: Add primary visualizations and reporting

**Owner:** Person C

**Files:**

- Create: `pre_experiments/camera_velocity_ambiguity_02/visualize.py`
- Create: `pre_experiments/camera_velocity_ambiguity_02/README.md`
- Create: `docs/camera_velocity_ambiguity_02_report_template.md`
- Test: `tests/camera_velocity_ambiguity_02/test_visualize.py`

**Step 1: Write failing tests**

Using tiny scalar CSV/JSON fixtures, require direction-similarity,
interpolation-energy and scene-prevalence PNGs plus the separate reproduction
directory. Delete all PNGs and prove decision hashes remain unchanged.

**Step 2: Run RED, implement, run GREEN**

```bash
python -m unittest tests.camera_velocity_ambiguity_02.test_visualize -v
```

The report template must force one of four conclusions and a GO/NO-GO table;
it must display excluded/invalid/control counts and privileged-oracle caveats.

**Step 3: Commit**

```bash
git add pre_experiments/camera_velocity_ambiguity_02 docs tests/camera_velocity_ambiguity_02
git commit -m "Add CVA02 visualizations and report template"
```

## Task 13: Add H20 orchestration and optional plot dependencies

**Owner:** Joint, Person C authors

**Files:**

- Create: `scripts/h20/run_camera_velocity_ambiguity_02.sh`
- Modify: `pyproject.toml`
- Test: `tests/camera_velocity_ambiguity_02/test_h20_runner.py`

**Step 1: Write failing tests**

Static-test `set -euo pipefail`, exact H20 default paths, quoted variables,
integrity marker gate, calibration-before-development, pointer/log directories,
no environment creation, no checkpoint download, no CPU formal mode and no
large artifact export.

Also require the runner to enforce a one-scene smoke gate before the full
calibration cohort.  The smoke scene is the first frozen calibration scene and
uses the exact production frame selection, global/local inference, artifact,
geometry, oracle, RGB-D and control paths.  A matching authenticated smoke
completion sidecar is required before the runner may expand to all ten
calibration scenes.  The calibration run must resume and reuse the completed
smoke scene rather than infer it a second time.

**Step 2: Implement**

Defaults:

```text
REPO_ROOT=/home/ubuntu/yjh/vggt/.worktrees/camera_velocity_ambiguity_02_pre_experiment
DATA_ROOT=/data/yjh/share/datasets/ScanNet
CKPT_DIR=/data/yjh/share/pretrained/VGGT-1B
RESULT_ROOT=/data/output/camera_velocity_ambiguity
CONDA_ENV=/home/ubuntu/anaconda3/envs/vggt-gx
DEVICE=cuda
SMOKE_SCENE_LIMIT=1
CALIBRATION_SCENE_LIMIT=10
```

Use a dedicated optional dependency extra for evo/scipy/matplotlib. Document
that evo is GPLv3 and reproduction-only.

**Step 3: Verify and commit**

```bash
bash -n scripts/h20/run_camera_velocity_ambiguity_02.sh
python -m unittest tests.camera_velocity_ambiguity_02.test_h20_runner -v
git add scripts/h20 pyproject.toml tests/camera_velocity_ambiguity_02/test_h20_runner.py
git commit -m "Add H20 CVA02 orchestration"
```

## Task 14: Full CPU verification, review and H20 execution gates

**Owner:** Joint

**Step 1: Run focused and complete CPU suites**

```bash
python -m unittest discover -s tests/camera_velocity_ambiguity_02 -v
python -m unittest discover -s tests -v
bash -n scripts/h20/run_camera_velocity_ambiguity_02.sh
git diff --check
git status --short
```

**Step 2: Independent reviews**

- Person A reviews GT/RGB-D/presentation isolation.
- Person B reviews input integrity, frame identity and resume behavior.
- Person C reviews numerical schemas, provenance and all result labels.
- A final reviewer checks the complete branch against the frozen design.

Resolve findings through the original task owner, then rerun all commands.

**Step 3: Wait for real-data gate**

Do not run inference until:

```text
ScanNet assets = 100/100
official/local/H20 integrity = PASS
verified_completion.json = authenticated
H20 disk/GPU/process preflight = PASS
```

**Step 4: One-scene production smoke**

Run the first frozen calibration scene through the complete production path.
Require exact global/local call counts, reloadable artifacts, matching frame and
pair identities, finite Camera Tokens/poses/energies, passing controls, valid
provenance and an authenticated smoke completion sidecar.  Any failure stops
the runner before the other nine scenes are opened.

**Step 5: Calibration only**

Resume from the smoke output and run the exact 10-scene calibration stage,
inspect manifests and controls, then freeze the policy.  The first scene must
be reused without duplicate inference.  No development scene may be opened
during threshold selection.

**Step 6: Apply GO/STOP gate**

Continue to the 40-scene development evaluation only if calibration artifacts
are complete, all negative controls pass, frozen policy validation succeeds and
no protocol violation occurred.

**Step 7: Development and decision**

Run development without threshold overrides, generate scalar outputs and all
figures, select exactly one conclusion, and state V-RFM `GO / NO-GO`.

**Step 8: Final verification**

Record commit, config hash, data manifest, split-v2, input run IDs, scene counts,
pair counts, exclusion reasons, bootstrap seed/sample count and the fact that
FastVGGT plots are reproduction-only. Never pull large artifacts to the local
machine.
