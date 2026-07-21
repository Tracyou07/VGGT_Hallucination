# Camera Iteration Study

This training-free probe compares Camera Head iterations `1, 2, 4, 8, 16`.
For each scene/frame selection, VGGT runs once: the Aggregator runs once and
the Camera Head runs through the largest requested iteration. Iteration `k` is
one-based and evaluates `pose_enc_list[k - 1]` from that shared forward pass.

## Required Local Inputs

The checkpoint directory must contain `model.safetensors` or `model.pt`.
ScanNet must use one of these layouts:

```text
SCANNET_ROOT/process_scannet/<scene>/{color,pose}/
SCANNET_ROOT/raw_sens/scans/<scene>/<scene>.sens
```

Files in `pose/` are raw ScanNet GT camera-to-world matrices. Invalid or
non-finite poses are skipped. GT is used only for final evaluation, never for
iteration selection or model updates.

## AutoDL Run

On a new AutoDL Miniconda machine whose `base` environment already has working
Torch and CUDA, prepare and run each stage independently:

```bash
git switch camera-iteration-preexperiment
bash scripts/autodl/setup_vggt_env.sh
bash scripts/autodl/download_vggt_weights.sh
SCANNET_TOS_ACCEPTED=1 bash scripts/autodl/prepare_scannet_camera_iteration.sh
bash scripts/autodl/run_camera_iteration.sh
```

Defaults are `SCANNET_ROOT=/root/autodl-tmp/datasets/scannetv2`,
`CKPT_DIR=/root/autodl-tmp/ckpt/VGGT-1B`, and
`RESULT_DIR=/root/autodl-tmp/camera_iteration/results`. The environment is
`vggt`, cloned from `base` without replacing Torch/CUDA. The data stage requires
prior acceptance of the official ScanNet terms, downloads only configured
`.sens` scenes, and extracts only color/raw-pose data. Stages skip complete,
non-empty artifacts. The runner only validates and executes the study.

Override any protocol setting through the environment:

```bash
SCENE_LIMIT=1 FRAME_COUNTS="25 50" ITERATIONS="1 2 4 8" \
SAMPLING=nested_uniform PREPROCESS_MODE=pad \
RESULT_DIR=/root/autodl-tmp/camera_iteration/smoke \
  bash scripts/autodl/run_camera_iteration.sh
```

Other overrides are `AUTODL_TMP`, `CONDA_ROOT`, `CONDA_ENV_NAME`, `CKPT_DIR`,
`SCANNET_ROOT`, `SCENE_LIST`, `SEED`, `SAVE_CAMERA_TOKENS=1`, and
`SAVE_CONTEXT_DIAGNOSTICS=1`.

## Manual Run

```bash
python -m pre_experiments.camera_iteration.run_study \
  --data-dir /path/to/scannetv2/process_scannet \
  --scene-list configs/camera_iteration_scannet.txt \
  --scene-limit 1 --frame-counts 25 \
  --iterations 1 2 4 8 16 --sampling nested_uniform \
  --ckpt-dir /path/to/VGGT-1B --device cuda \
  --out-dir /absolute/path/to/results
```

Repository-local output is restricted to
`results/pre_experiments/camera_iteration/`. External output paths must be
absolute. A deterministic `<commit>_<invocation-hash>` directory makes an
identical invocation resumable.

## Output Contract

Each run contains `run_metadata.json`, `summary.json`, and `summary.csv`.
Each `<scene>/frames_<requested>/` directory contains:

- `iteration_metrics.{json,csv}`: one row per requested iteration.
- `camera_trace.npz`: frame IDs, activated and raw 9D pose encodings, raw 9D
  updates, and update norms for all iterations through `max(iterations)`.
- `selected_frame_ids.json`: exact ordered raw GT/image frame IDs.
- `complete.json`: written last and used for resume validation.

`--save-camera-tokens` additionally stores normalized and per-iteration
modulated Camera Tokens. It is opt-in because storage grows as `O(KSC)`.
`--save-context-diagnostics` instead writes a compact
`context_diagnostics.npz` with only the final normalized Camera Token, raw and
aligned predictions, raw GT, and per-frame aligned errors. It does not retain
per-iteration modulated tokens.
Incomplete selections are rerun; a selection is skipped only when all required
artifacts exist and its run ID, frame IDs, and iterations match.

## Publishing Numeric Artifacts

Keep the full runtime tree on AutoDL, then export only Git-safe measurements:

```bash
python scripts/autodl/camera_iteration/export_numeric_results.py \
  --source /root/autodl-tmp/camera_iteration/results/<run_id>
```

The destination is `results/camera_iteration/<run_id>/`. It contains root and
per-selection JSON/CSV files, selected frame IDs, completion metadata, and
pose-only `camera_trace.npz` files. NPZ traces containing
`normalized_camera_tokens` or `pose_tokens_modulated`, and any allowed file
larger than 50 MiB, are rejected. Review `publish_manifest.json` and `du -sh`
before explicitly running `git add`, `git commit`, and `git push`.
For a declared Round 1.5 run, the same exporter additionally validates and
copies `context_diagnostics.npz`, `context_per_frame.csv`, and
`context_summary.{csv,json}`.

## Metric Semantics

Primary conclusions use aligned prediction metrics:
`pose_ate_rmse_aligned`, `pose_are_mean_deg_aligned`,
`pose_rpe_rot_mean_deg`, and `pose_rpe_trans_mean_aligned`.
`pose_sim3_scale` diagnoses scale/gauge behavior and is not a quality result.
Translation RPE is the translation norm of the aligned relative-transform
error; it preserves direction and is not a difference between step lengths.
`delta_norm_{mean,p95,max}` is the L2 norm of the raw 9D Camera Head update,
not an SE(3) distance. Pure GT analyses use raw GT only.

## Verification

```bash
python -m unittest discover -s tests
python -m pre_experiments.camera_iteration.run_study --help
bash -n scripts/autodl/run_camera_iteration.sh
```

The branch-local contract is recorded in
`doc/2026-07-16_Camera_Iteration_Worktree_Design.md`. The repository-wide
research guide and implementation plan are maintained only on `main`, not
duplicated in this worktree. No checkpoint-backed experiment has run yet.
