# Camera Solution Space 01 — Stage 1 Implementation Tasks

> Worktree branch: `codex/camera_solution_space_01_stage1`
> Spec: `doc/camera_solution_space_01_theory_foundation/scannet_fixed_observation_experiment_plan.md`
> Runtime: `/home/ubuntu/anaconda3/envs/vggt-gx/bin/python`

## Global constraints

- All tracked implementation lives inside `/home/ubuntu/yjh/vggt` or its
  `.worktrees/` child. Raw data stays under `/data/yjh/share/datasets/ScanNet`,
  weights under `/data/yjh/share/pretrained`, derived artifacts under
  `/data/output/camera_solution_space_01`.
- Never modify, delete, rename, or extract into the raw dataset tree.
- Tests use synthetic fixtures and must not download data or load model weights.
- Real ScanNet/VGGT runs happen only on H20. Small CPU unit tests are correctness
  tests, not a formal CPU smoke experiment.
- Use `unittest`, test-first RED then GREEN, Python 3.10+, NumPy `<2`.
- No raw 9D pose distance/interpolation. `pose_enc_list` is not a sample set.
- Sensor GT pose and PLY are audit-only and cannot enter candidate acceptance
  energy.
- Every artifact contract is strict, versioned, hash-addressed, and fail-closed.
  No overwrite flag, automatic resampling, or silent schema upgrade.
- Keep commits scoped. Do not push from the implementation worktree.

## Task 1: Canonical configs, contracts, and random-access SENS index

Create:

```text
configs/camera_solution_space_01/fastvggt_scannet50.txt
configs/camera_solution_space_01/scannet50_split_v1.json
pre_experiments/camera_solution_space_01/__init__.py
pre_experiments/camera_solution_space_01/contracts.py
pre_experiments/camera_solution_space_01/sens_index.py
tests/camera_solution_space_01/__init__.py
tests/camera_solution_space_01/test_contracts.py
tests/camera_solution_space_01/test_sens_index.py
```

Requirements:

1. The scene list is exactly the 50 lines in the FastVGGT official
   `eval/scannet_50.yaml`, in the same order, no duplicates.
2. The split algorithm is recorded as sorting SHA-256 of
   `camera_solution_space_01:v1:<scene_id>` then scene ID; the first 12 are
   calibration and the other 38 evaluation. The exact calibration list must
   match the detailed spec.
3. `contracts.py` provides canonical compact JSON bytes and SHA-256 helpers.
   Reject NaN/Infinity, unsupported objects, non-string mapping keys, and
   schema mismatch. Hashes are lowercase 64-character hex.
4. `sens_index.py` supports official ScanNet SENS version 4, explicit
   little-endian parsing, bounded reads, and these compression codes:
   color `{0: raw, 1: png, 2: jpeg}`; depth
   `{0: raw_ushort, 1: zlib_ushort, 2: occi_ushort}`.
5. Parse sensor name, four 4x4 float32 calibration matrices, color/depth
   compression, dimensions, depth shift, frame count, and for every frame:
   camera-to-world, both timestamps, payload sizes, payload offsets, and next
   record offset. Indexing must seek over payloads without decoding them or
   retaining payload bytes.
6. Validate version, dimensions, finite matrices, nonnegative bounded string
   and payload sizes, and record/file bounds. After the RGB-D frame table,
   parse the canonical SENS v4 `uint64` IMU count and its fixed 128-byte record
   range without materializing records; require exact EOF after that IMU
   section. Fail with typed, actionable exceptions including field/frame
   context. Never accept arbitrary trailing bytes.
7. The public API is read-only and accepts a file path; no network and no
   output deletion. Use immutable dataclasses for parsed metadata.

Tests must first fail, then cover:

- a synthetic two-frame SENS fixture with JPEG-like and zlib payload bytes;
- exact offsets and metadata without any decode call;
- truncated header, invalid version/compression/dimensions/matrix;
- payload size beyond EOF and truncated RGB-D frame;
- zero/nonzero IMU sections, missing/truncated IMU count or records, oversized
  count, and trailing undeclared bytes after an otherwise valid IMU section;
- canonical JSON determinism and non-finite rejection;
- exact 50 scene list and 12/38 split reconstruction from the recorded rule.

Verification:

```bash
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
/home/ubuntu/anaconda3/envs/vggt-gx/bin/python -m unittest \
  tests.camera_solution_space_01.test_contracts \
  tests.camera_solution_space_01.test_sens_index -v
```

Commit message: `Add strict ScanNet SENS index contracts`

## Task 2: Fixed eight-frame observation planning and sealing

Create:

```text
pre_experiments/camera_solution_space_01/observation.py
scripts/camera_solution_space_01/plan_observations.py
scripts/camera_solution_space_01/seal_observations.py
scripts/camera_solution_space_01/validate_observations.py
tests/camera_solution_space_01/test_observation.py
```

Requirements:

1. Planning and sealing are separate APIs. Planning fixes exactly eight original
   frame indices `[f, f+15, ..., f+105]`; sealing cannot select or replace a
   frame.
2. Plan schema records source path/size/SHA-256, scene/split, selection version
   `fixed8_stride15_v1`, ordered frame IDs/timestamps, header fingerprint, and
   selection diagnostics. Canonical hash is the plan ID.
3. The first implementation accepts a precomputed eligibility callback/cache;
   it must deterministically choose the lowest eligible `f`. Do not implement
   matcher thresholds in this task.
4. Random-access extraction only reads planned payload offsets. Implement JPEG
   color decode and zlib uint16 depth decode; reject unsupported active
   compression instead of guessing. Validate decoded shape exactly.
5. Seal with same-directory temp files then atomic rename. Existing output is
   accepted only if deep validation exactly matches; otherwise fail without
   overwriting. Write `complete.json` last.
6. Manifest lists every output relative path, size, SHA-256, ordered model input,
   source fingerprint, plan ID, and a deterministic artifact Merkle hash.
7. Validation fails on missing/extra/tampered files, changed source fingerprint,
   wrong order/schema/hash, or data path escaping the observation root.

Tests cover deterministic lowest-window choice, no resampling, random-access
reads only for selected frames, uint16 depth round trip, JPEG dimension check,
atomic incomplete output behavior, idempotent validation, tamper/extra/path
traversal rejection, and proof that the raw source tree is unchanged.

Verification:

```bash
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
/home/ubuntu/anaconda3/envs/vggt-gx/bin/python -m unittest \
  tests.camera_solution_space_01.test_observation -v
```

Commit message: `Seal fixed ScanNet observations`

## Task 3: Gauge-fixed SE(3) trajectories and distance

Create:

```text
pre_experiments/camera_solution_space_01/se3.py
pre_experiments/camera_solution_space_01/trajectory.py
tests/camera_solution_space_01/test_se3.py
tests/camera_solution_space_01/test_trajectory.py
```

Requirements:

1. Float64 NumPy and Torch-compatible interfaces for SO(3)/SE(3) exp, log,
   inverse, compose, and geodesic interpolation. Stable small-angle branches;
   reject non-finite or invalid homogeneous matrices.
2. World-to-camera gauge fix is exactly `P_i = W_i @ inv(W_0)`, with `P_0=I`.
3. Trajectory dimension for eight frames is `6*(8-1)=42`; first frame is never
   optimized.
4. Distance uses left difference `log(P'_i @ inv(P_i))`, translation scale
   0.10 m, rotation scale 5 degrees, RMS over frames 1..7. `delta=1`, dedup
   threshold 0.10.
5. Product geodesic is
   `exp(t * log(P_B @ inv(P_A))) @ P_A` per non-anchor frame.
6. Do not accept or expose raw VGGT 9D pose encodings in this package.

Tests cover exp/log round trips at zero, tiny, near-pi, random valid transforms;
inverse/compose; invalid inputs; global gauge copies; quaternion sign-equivalent
rotations after matrix conversion; exact scale behavior of distance; endpoint
and midpoint geodesic validity; and 42-dimensional pack/unpack round trips.

Verification:

```bash
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
/home/ubuntu/anaconda3/envs/vggt-gx/bin/python -m unittest \
  tests.camera_solution_space_01.test_se3 \
  tests.camera_solution_space_01.test_trajectory -v
```

Commit message: `Add gauge-fixed trajectory geometry`

## Task 4: Frozen RGB-D energy and calibration controls

Create the matching, `rgbd_energy.py`, calibration/control modules, configs,
CLIs, and tests named in the detailed spec. Use the exact SIFT/MAGSAC values,
15 fixed edges, residual scales, Huber `c=2.5`, pair weights 0.7/0.3,
behind-camera penalty 10, distance thresholds, `m_eval`, `h`, gray zone, and
95% negative-control gate from the spec. The objective API accepts only a
sealed observation and trajectory; it must not accept GT pose, PLY, or VGGT
predictions. Complete all analytic and synthetic controls before any evaluation
scene result is visible.

Commit message: `Add independent RGB-D solution energy`

## Task 5: Proposals, local continuation, path search, and result cards

Implement the strict local VGGT proposal, independent pose-graph proposal,
Sobol multistart/refit registry, local Hessian and finite-difference checks,
profile continuation, product-SE(3) string/NEB, and reporting described in the
detailed spec. Candidate/path acceptance reads frozen observation, objective,
and calibration cards only. Reports must encode the asymmetric conclusion
language: search failure never becomes a disconnectedness claim.

The official VGGT proposal preprocessing is frozen as
`official_vggt_pad518_bicubic_v1`: decoded native RGB, aspect-preserving resize
with the largest side 518, the other side rounded to a multiple of patch size
14, Pillow bicubic resampling, and centered white padding to 518x518. Tests must
compare the array adapter against `vggt.utils.load_fn.load_and_preprocess_images`
on lossless fixtures. Do not use the obsolete long-side-640/bilinear text, the
Omega 512/256 transform, or treat four `pose_enc_list` refinements as samples.

Commit message: `Add camera solution-space probes`
