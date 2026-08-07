"""Run translation-only camera refinement and export per-scene results."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch

from pre_experiments.camera_refiner_training.checkpoint import (
    load_checkpoint,
    read_checkpoint_payload,
)
from pre_experiments.camera_refiner_training.data import (
    SceneWindows,
    build_scene_windows,
    load_dataset_manifest,
    load_translation_units,
)
from pre_experiments.camera_refiner_training.diffusion import DiffusionSchedule, ddim_sample
from pre_experiments.camera_refiner_training.geometry import (
    apply_center_corrections,
    fuse_window_corrections,
)
from pre_experiments.camera_refiner_training.metrics import translation_metrics
from pre_experiments.camera_refiner_training.model import ModelConfig, ResidualDiT
from pre_experiments.camera_refiner_training.visualize import write_trajectory_plot
from pre_experiments.camera_refiner_training.io import atomic_write_json


@dataclass(frozen=True)
class RefinementResult:
    refined_c2w: np.ndarray
    correction_canonical: np.ndarray
    confidence: np.ndarray


def scene_frame_ids(
    window_frame_ids: np.ndarray,
    starts: np.ndarray,
    *,
    total_frames: int,
) -> np.ndarray:
    windows = np.asarray(window_frame_ids, dtype=np.int64)
    offsets = np.asarray(starts, dtype=np.int64)
    if windows.ndim != 2 or offsets.shape != (len(windows),):
        raise ValueError("window frame IDs and starts have incompatible shapes")
    result = np.full(total_frames, -1, dtype=np.int64)
    for values, start in zip(windows, offsets):
        target = slice(int(start), int(start) + len(values))
        existing = result[target]
        conflict = (existing >= 0) & (existing != values)
        if np.any(conflict):
            raise ValueError("overlapping windows disagree on frame IDs")
        result[target] = values
    if np.any(result < 0):
        raise ValueError("window frame IDs do not cover the full scene")
    return result


@torch.no_grad()
def refine_scene(
    model: torch.nn.Module,
    scene: SceneWindows,
    *,
    condition_mean: torch.Tensor,
    condition_std: torch.Tensor,
    model_kind: str,
    diffusion_steps: int,
    sample_steps: int,
    seed: int,
    device: torch.device,
) -> RefinementResult:
    model.eval()
    condition = torch.from_numpy(scene.condition).float()
    condition = ((condition - condition_mean) / condition_std).to(device)
    if model_kind == "diffusion":
        generator = torch.Generator(device=device.type).manual_seed(seed)
        correction, confidence = ddim_sample(
            model,
            condition,
            DiffusionSchedule.cosine(diffusion_steps),
            sample_steps=sample_steps,
            generator=generator,
        )
    elif model_kind == "deterministic":
        timestep = torch.zeros(len(condition), dtype=torch.long, device=device)
        correction, confidence = model(
            torch.zeros((*condition.shape[:2], 3), device=device), condition, timestep
        )
    else:
        raise ValueError("model_kind must be diffusion or deterministic")
    correction_np = correction.cpu().numpy()
    confidence_np = confidence.squeeze(-1).cpu().numpy()
    fused, fused_confidence = fuse_window_corrections(
        correction_np,
        confidence_np,
        starts=scene.starts,
        total_frames=len(scene.global_c2w),
        alignment_residual=scene.alignment_residual,
    )
    global_centers = scene.global_c2w[:, :3, 3]
    canonical = scene.gauge.canonicalize(global_centers)
    refined_centers = scene.gauge.restore(canonical + fused)
    refined = apply_center_corrections(scene.global_c2w, refined_centers - global_centers)
    return RefinementResult(refined, fused, fused_confidence)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--local-run-dir", type=Path, required=True)
    parser.add_argument("--frozen-units", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--role", default="validation")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument("--unit-count", type=int)
    parser.add_argument("--iteration", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    payload = read_checkpoint_payload(args.checkpoint, device)
    run_config = payload["run_config"]
    model_config = ModelConfig(**payload["model_config"])
    model = ResidualDiT(model_config).to(device)
    state = load_checkpoint(args.checkpoint, model=model, map_location=device)
    checkpoint_unit_count = int(run_config.get("unit_count", 41))
    checkpoint_iteration = int(run_config.get("iteration", 0))
    if args.unit_count is not None and args.unit_count != checkpoint_unit_count:
        parser.error("unit-count does not match checkpoint")
    if args.iteration is not None and args.iteration != checkpoint_iteration:
        parser.error("iteration does not match checkpoint")
    units = load_translation_units(
        args.frozen_units,
        count=checkpoint_unit_count,
        iteration=checkpoint_iteration,
    )
    if units.digest != run_config["unit_digest"]:
        parser.error("frozen unit manifest does not match checkpoint")
    manifest = load_dataset_manifest(
        args.dataset_manifest, args.dataset_root, roles={args.role}
    )
    if manifest.digest != run_config["dataset_digest"]:
        parser.error("dataset manifest does not match checkpoint")
    rows = []
    for entry in manifest.entries:
        scene_dir = args.out_dir / entry.scene
        completion = scene_dir / "complete.json"
        if completion.is_file() and not args.overwrite:
            existing = json.loads(completion.read_text(encoding="utf-8"))
            if existing.get("checkpoint_digest") != state.run_digest:
                parser.error(f"existing output uses another checkpoint: {scene_dir}")
            rows.append(existing)
            continue
        scene = build_scene_windows(
            entry.shard,
            args.local_run_dir / entry.scene,
            units,
            window_length=int(run_config["window_length"]),
            stride=int(run_config["stride"]),
        )
        result = refine_scene(
            model,
            scene,
            condition_mean=state.condition_mean,
            condition_std=state.condition_std,
            model_kind=str(run_config["model_kind"]),
            diffusion_steps=int(run_config["diffusion_steps"]),
            sample_steps=args.sample_steps,
            seed=args.seed,
            device=device,
        )
        scene_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            scene_dir / "refined_camera.npz",
            frame_ids=scene_frame_ids(
                scene.frame_ids, scene.starts, total_frames=len(scene.global_c2w)
            ),
            baseline_c2w=scene.global_c2w,
            refined_c2w=result.refined_c2w,
            gt_c2w_raw=scene.gt_c2w_raw,
            correction_canonical=result.correction_canonical,
            confidence=result.confidence,
        )
        row = {
            "scene": entry.scene,
            "checkpoint_digest": state.run_digest,
            "rotation_max_abs_change": float(
                np.max(
                    np.abs(
                        result.refined_c2w[:, :3, :3]
                        - scene.global_c2w[:, :3, :3]
                    )
                )
            ),
        }
        row.update({f"baseline_{key}": value for key, value in translation_metrics(scene.global_c2w, scene.gt_c2w_raw).items()})
        row.update({f"refined_{key}": value for key, value in translation_metrics(result.refined_c2w, scene.gt_c2w_raw).items()})
        atomic_write_json(completion, row)
        write_trajectory_plot(
            scene_dir / "trajectory.png",
            baseline_c2w=scene.global_c2w,
            refined_c2w=result.refined_c2w,
            gt_c2w=scene.gt_c2w_raw,
        )
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    atomic_write_json(args.out_dir / "summary.json", rows)
    if rows:
        summary_csv = args.out_dir / "summary.csv"
        temporary = summary_csv.with_suffix(".csv.tmp")
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(summary_csv)


if __name__ == "__main__":
    main()
