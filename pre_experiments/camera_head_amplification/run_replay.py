"""Replay frozen VGGT Camera Head on Round 1.5 normalized Camera Tokens."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np
import torch

from pre_experiments.camera_head_amplification.checkpoint import (
    load_local_camera_head,
)
from pre_experiments.camera_head_amplification.metrics import (
    build_stage_per_frame_rows,
    build_stage_summary_rows,
    match_frame_indices,
    validate_replay_baseline,
)
from pre_experiments.camera_head_amplification.replay import (
    CameraHeadReplay,
    replay_camera_head,
)
from pre_experiments.camera_iteration.contracts import (
    atomic_write_json,
    read_git_commit,
)
from pre_experiments.camera_iteration.model_io import resolve_device
from pre_experiments.camera_iteration.pose_metrics import (
    align_pose_sequence,
    evaluate_pose,
    to_homogeneous,
)
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "results" / "camera_context" / "911b598_f4577f584448"
DEFAULT_CHECKPOINT = Path("/root/autodl-tmp/ckpt/VGGT-1B")
DEFAULT_OUTPUT = Path("/root/autodl-tmp/camera_head_amplification/results")
CONTEXT_MEMBERS = {
    "frame_ids",
    "normalized_camera_tokens",
    "pred_c2w_raw",
    "pred_c2w_aligned",
    "gt_c2w_raw",
    "translation_error_aligned",
    "rotation_error_deg_aligned",
    "delta_norm",
    "sim3_scale",
    "sim3_rotation",
    "sim3_translation",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--ckpt-dir", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--short-frames", type=int, default=200)
    parser.add_argument("--long-frames", type=int, default=500)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--scene-limit", type=int, default=0)
    parser.add_argument("--baseline-atol", type=float, default=1e-5)
    parser.add_argument("--baseline-rtol", type=float, default=1e-5)
    args = parser.parse_args(argv)
    if args.short_frames < 2 or args.long_frames <= args.short_frames:
        parser.error("contexts must satisfy 2 <= short-frames < long-frames")
    if args.iterations < 1:
        parser.error("--iterations must be at least 1")
    if args.scene_limit < 0:
        parser.error("--scene-limit must be non-negative")
    if args.baseline_atol < 0 or args.baseline_rtol < 0:
        parser.error("baseline tolerances must be non-negative")
    return args


def _read_json_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON object: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_saved_pose(selection_dir: Path, iterations: int) -> np.ndarray:
    path = selection_dir / "camera_trace.npz"
    if not path.is_file():
        raise FileNotFoundError(f"missing Round 1.5 camera trace: {path}")
    with np.load(path, allow_pickle=False) as archive:
        if "pose_enc_by_iteration" not in archive:
            raise ValueError(f"pose_enc_by_iteration missing from {path}")
        pose = np.asarray(archive["pose_enc_by_iteration"], dtype=np.float32)
    if pose.ndim != 3 or pose.shape[-1] != 9 or pose.shape[0] < iterations:
        raise ValueError(f"invalid pose_enc_by_iteration shape in {path}: {pose.shape}")
    return pose[:iterations]


def _load_context_artifact(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"missing Round 1.5 context diagnostics: {path}")
    with np.load(path, allow_pickle=False) as archive:
        members = set(archive.files)
        if members != CONTEXT_MEMBERS:
            raise ValueError(
                f"unexpected context diagnostics members in {path}: {sorted(members)}"
            )
        arrays = {name: np.asarray(archive[name]).copy() for name in members}
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise ValueError(f"context diagnostics contain non-finite values: {path}")
    return arrays


def _load_selection(
    source: Path,
    scene: str,
    frames: int,
    iterations: int,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    directory = source / scene / f"frames_{frames}"
    context = _load_context_artifact(directory / "context_diagnostics.npz")
    return context, _read_saved_pose(directory, iterations)


def _to_numpy(replay: CameraHeadReplay) -> dict[str, object]:
    representations = {
        name: value[:, 0].detach().float().cpu().numpy()
        for name, value in replay.representations.items()
    }
    representations["pose_delta_9d"] = (
        replay.pose_delta[:, 0].detach().float().cpu().numpy()
    )
    representations["raw_pose_9d"] = (
        replay.raw_pose[:, 0].detach().float().cpu().numpy()
    )
    return {
        "activated_pose": replay.activated_pose[:, 0]
        .detach()
        .float()
        .cpu()
        .numpy(),
        "representations": representations,
    }


def _pose_rows(
    *,
    scene: str,
    variant: str,
    activated_pose: np.ndarray,
    frame_ids: np.ndarray,
    gt_c2w_raw: np.ndarray,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    final_pose = torch.from_numpy(activated_pose[-1]).unsqueeze(0)
    extrinsic, _ = pose_encoding_to_extri_intri(
        final_pose,
        build_intrinsics=False,
    )
    pred_w2c = to_homogeneous(extrinsic[0].numpy())
    metrics = evaluate_pose(pred_w2c, gt_c2w_raw)
    alignment = align_pose_sequence(pred_w2c, gt_c2w_raw)
    summary: dict[str, object] = {
        "scene": scene,
        "variant": variant,
        "frame_count": int(len(frame_ids)),
        **metrics,
    }
    per_frame = [
        {
            "scene": scene,
            "variant": variant,
            "frame_id": int(frame_id),
            "translation_error_aligned": float(translation_error),
            "rotation_error_deg_aligned": float(rotation_error),
        }
        for frame_id, translation_error, rotation_error in zip(
            frame_ids,
            alignment["translation_error_aligned"],
            alignment["rotation_error_deg_aligned"],
        )
    ]
    return summary, per_frame


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _scene_names(source: Path, limit: int) -> list[str]:
    metadata = _read_json_object(source / "run_metadata.json")
    invocation = metadata.get("invocation")
    scenes = invocation.get("scenes") if isinstance(invocation, dict) else None
    if not isinstance(scenes, list) or not all(isinstance(item, str) for item in scenes):
        raise ValueError("source run metadata must declare invocation.scenes")
    return scenes[:limit] if limit else scenes


def _make_run_id(commit: str, invocation: dict[str, object]) -> str:
    canonical = json.dumps(invocation, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]
    return f"{commit[:7]}_{digest}"


def run(args: argparse.Namespace) -> Path:
    source = args.source_run_dir.resolve()
    checkpoint = args.ckpt_dir.resolve()
    output = args.out_dir.resolve()
    source_metadata = _read_json_object(source / "run_metadata.json")
    scenes = _scene_names(source, args.scene_limit)
    if not scenes:
        raise ValueError("source run contains no selected scenes")
    device = resolve_device(args.device)
    commit = read_git_commit(ROOT)
    invocation = {
        "source_run_dir": source.as_posix(),
        "source_run_id": source_metadata.get("run_id"),
        "checkpoint_dir": checkpoint.as_posix(),
        "device": str(device),
        "short_frames": args.short_frames,
        "long_frames": args.long_frames,
        "iterations": args.iterations,
        "scenes": scenes,
        "baseline_atol": args.baseline_atol,
        "baseline_rtol": args.baseline_rtol,
    }
    run_id = _make_run_id(commit, invocation)
    run_dir = output / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "study_type": "method_pre_experiment",
        "study_name": "camera_head_amplification",
        "git_commit": commit,
        "run_id": run_id,
        "source_run_id": source_metadata.get("run_id"),
        "invocation": invocation,
        "argv": list(sys.argv),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "primary_metric_policy": "prediction metrics use aligned values; GT remains raw",
    }
    atomic_write_json(run_dir / "run_metadata.json", metadata)

    head = load_local_camera_head(checkpoint).to(device).eval()
    stage_summary: list[dict[str, object]] = []
    stage_per_frame: list[dict[str, object]] = []
    pose_summary: list[dict[str, object]] = []
    pose_per_frame: list[dict[str, object]] = []
    baseline_checks: list[dict[str, object]] = []

    for scene in scenes:
        short, short_saved = _load_selection(
            source, scene, args.short_frames, args.iterations
        )
        long, long_saved = _load_selection(
            source, scene, args.long_frames, args.iterations
        )
        indices = match_frame_indices(short["frame_ids"], long["frame_ids"])
        if not np.allclose(
            short["gt_c2w_raw"], long["gt_c2w_raw"][indices], atol=1e-10, rtol=0
        ):
            raise ValueError(f"raw GT mismatch between contexts for {scene}")

        short_tokens = torch.from_numpy(short["normalized_camera_tokens"]).unsqueeze(0).to(device)
        shared_tokens = torch.from_numpy(
            long["normalized_camera_tokens"][indices]
        ).unsqueeze(0).to(device)
        long_tokens = torch.from_numpy(long["normalized_camera_tokens"]).unsqueeze(0).to(device)

        short_replay = _to_numpy(
            replay_camera_head(head, short_tokens, num_iterations=args.iterations)
        )
        shared_replay = _to_numpy(
            replay_camera_head(head, shared_tokens, num_iterations=args.iterations)
        )
        long_replay = _to_numpy(
            replay_camera_head(
                head,
                long_tokens,
                num_iterations=args.iterations,
            )
        )
        for context_name, actual, expected in (
            (f"H{args.short_frames}(Z{args.short_frames})", short_replay["activated_pose"], short_saved),
            (f"H{args.long_frames}(Z{args.long_frames})", long_replay["activated_pose"], long_saved),
        ):
            diagnostics = validate_replay_baseline(
                actual,
                expected,
                atol=args.baseline_atol,
                rtol=args.baseline_rtol,
            )
            baseline_checks.append(
                {"scene": scene, "variant": context_name, "passed": True, **diagnostics}
            )

        comparison = f"H{args.short_frames}(Z{args.short_frames})_vs_H{args.short_frames}(Z{args.long_frames}_shared)"
        stage_summary.extend(
            build_stage_summary_rows(
                scene=scene,
                comparison=comparison,
                baseline_tokens=short["normalized_camera_tokens"],
                perturbed_tokens=long["normalized_camera_tokens"][indices],
                baseline_stages=short_replay["representations"],
                perturbed_stages=shared_replay["representations"],
            )
        )
        stage_per_frame.extend(
            build_stage_per_frame_rows(
                scene=scene,
                comparison=comparison,
                frame_ids=short["frame_ids"],
                baseline_tokens=short["normalized_camera_tokens"],
                perturbed_tokens=long["normalized_camera_tokens"][indices],
                baseline_stages=short_replay["representations"],
                perturbed_stages=shared_replay["representations"],
            )
        )

        long_shared_stages = {
            name: values[:, indices]
            for name, values in long_replay["representations"].items()
        }
        head_context_comparison = (
            f"H{args.short_frames}(Z{args.long_frames}_shared)_vs_"
            f"H{args.long_frames}(Z{args.long_frames})_shared"
        )
        stage_summary.extend(
            build_stage_summary_rows(
                scene=scene,
                comparison=head_context_comparison,
                baseline_tokens=long["normalized_camera_tokens"][indices],
                perturbed_tokens=long["normalized_camera_tokens"][indices],
                baseline_stages=shared_replay["representations"],
                perturbed_stages=long_shared_stages,
                allow_zero_input=True,
            )
        )
        stage_per_frame.extend(
            build_stage_per_frame_rows(
                scene=scene,
                comparison=head_context_comparison,
                frame_ids=short["frame_ids"],
                baseline_tokens=long["normalized_camera_tokens"][indices],
                perturbed_tokens=long["normalized_camera_tokens"][indices],
                baseline_stages=shared_replay["representations"],
                perturbed_stages=long_shared_stages,
            )
        )

        variants = (
            (f"H{args.short_frames}(Z{args.short_frames})", short_replay["activated_pose"], short["frame_ids"], short["gt_c2w_raw"]),
            (f"H{args.short_frames}(Z{args.long_frames}_shared)", shared_replay["activated_pose"], short["frame_ids"], short["gt_c2w_raw"]),
            (f"H{args.long_frames}(Z{args.long_frames})", long_replay["activated_pose"], long["frame_ids"], long["gt_c2w_raw"]),
            (f"H{args.long_frames}(Z{args.long_frames})_shared", long_replay["activated_pose"][:, indices], short["frame_ids"], short["gt_c2w_raw"]),
        )
        for variant, activated, frame_ids, gt_raw in variants:
            summary, per_frame = _pose_rows(
                scene=scene,
                variant=variant,
                activated_pose=activated,
                frame_ids=frame_ids,
                gt_c2w_raw=gt_raw,
            )
            pose_summary.append(summary)
            pose_per_frame.extend(per_frame)

        del short_replay, shared_replay, long_replay
        if device.type == "cuda":
            torch.cuda.empty_cache()

    atomic_write_json(run_dir / "baseline_checks.json", baseline_checks)
    atomic_write_json(run_dir / "amplification_summary.json", stage_summary)
    atomic_write_json(run_dir / "pose_metrics.json", pose_summary)
    _write_csv(run_dir / "amplification_summary.csv", stage_summary)
    _write_csv(run_dir / "amplification_per_frame.csv", stage_per_frame)
    _write_csv(run_dir / "pose_metrics.csv", pose_summary)
    _write_csv(run_dir / "pose_per_frame.csv", pose_per_frame)
    atomic_write_json(
        run_dir / "complete.json",
        {"run_id": run_id, "scenes": scenes, "baseline_checks_passed": True},
    )
    return run_dir


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_dir = run(args)
    print(f"[done] results={run_dir}")


if __name__ == "__main__":
    main()
