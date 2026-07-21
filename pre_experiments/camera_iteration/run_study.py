"""Run the self-contained ScanNet Camera Head iteration study."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Callable

import numpy as np
import torch

from pre_experiments.camera_context.artifacts import build_context_diagnostics
from pre_experiments.camera_iteration.contracts import (
    atomic_write_json,
    build_run_metadata,
    read_git_commit,
    validate_output_root,
)
from pre_experiments.camera_iteration.metrics import MetricRow, build_iteration_rows, validate_iterations
from pre_experiments.camera_iteration.model_io import load_local_model, resolve_device
from pre_experiments.camera_iteration.pose_metrics import to_homogeneous
from pre_experiments.camera_iteration.scannet import (
    load_scene_frames,
    make_frame_selections,
    read_scene_list,
)
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


ROOT = Path(__file__).resolve().parents[2]
AUTODL_TMP = Path(os.environ.get("AUTODL_TMP", "/root/autodl-tmp"))
DEFAULT_SCANNET_ROOT = Path(
    os.environ.get("SCANNET_ROOT", str(AUTODL_TMP / "datasets" / "scannetv2"))
)
DEFAULT_CHECKPOINT_DIR = Path(
    os.environ.get("CKPT_DIR", str(AUTODL_TMP / "ckpt" / "VGGT-1B"))
)


def _resolve_repo_path(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse and validate the camera-iteration study command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_SCANNET_ROOT / "process_scannet")
    parser.add_argument(
        "--scene-list",
        type=Path,
        default=ROOT / "configs" / "camera_iteration_scannet.txt",
    )
    parser.add_argument("--scene-limit", type=int, default=10)
    parser.add_argument("--frame-counts", type=int, nargs="+", default=[25, 50, 100, 200, 500])
    parser.add_argument("--iterations", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument(
        "--sampling",
        choices=["prefix", "uniform", "nested_uniform", "regime_step"],
        default="nested_uniform",
    )
    parser.add_argument("--ckpt-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--preprocess-mode", choices=["pad", "crop"], default="pad")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "results" / "pre_experiments" / "camera_iteration",
    )
    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument("--save-camera-tokens", action="store_true")
    parser.add_argument("--save-context-diagnostics", action="store_true")
    args = parser.parse_args(argv)

    if args.scene_limit < 0:
        parser.error("--scene-limit must be non-negative")
    if (
        not args.frame_counts
        or any(count < 1 for count in args.frame_counts)
        or any(a >= b for a, b in zip(args.frame_counts, args.frame_counts[1:]))
    ):
        parser.error("--frame-counts must be positive and strictly increasing")
    try:
        validate_iterations(args.iterations, max(args.iterations, default=0))
    except ValueError as error:
        parser.error(str(error))
    return args


def selection_is_complete(
    output_dir: Path,
    run_id: str,
    selected_ids: list[int],
    iterations: list[int],
    require_context_diagnostics: bool = False,
) -> bool:
    """Return true only for a complete selection matching the current invocation."""
    completion_path = output_dir / "complete.json"
    required = [
        output_dir / "iteration_metrics.json",
        output_dir / "iteration_metrics.csv",
        output_dir / "camera_trace.npz",
        output_dir / "selected_frame_ids.json",
        completion_path,
    ]
    if require_context_diagnostics:
        required.append(output_dir / "context_diagnostics.npz")
    if not all(path.is_file() for path in required):
        return False
    try:
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        completion.get("run_id") == run_id
        and completion.get("selected_frame_ids") == selected_ids
        and completion.get("iterations") == iterations
    )


def _atomic_write_csv(path: Path, rows: list[MetricRow]) -> None:
    if not rows:
        raise ValueError("cannot write an empty metric table")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _atomic_save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _tensor_stack(values: list[torch.Tensor]) -> np.ndarray:
    return np.stack([value[0].detach().float().cpu().numpy() for value in values])


def run_selection(
    *,
    model: object,
    scene: str,
    requested_frame_count: int,
    selected_ids: list[int],
    image_by_id: dict[int, Path],
    poses_by_id: dict[int, np.ndarray],
    iterations: list[int],
    device: torch.device,
    preprocess_mode: str,
    output_dir: Path,
    run_id: str,
    save_camera_tokens: bool,
    save_context_diagnostics: bool = False,
    image_loader: Callable[[list[str], str], torch.Tensor] = load_and_preprocess_images,
) -> list[MetricRow]:
    """Run one maximum-iteration forward pass and persist selected iteration rows."""
    if not selected_ids:
        raise ValueError("selected_ids must not be empty")
    image_paths = [str(image_by_id[frame_id]) for frame_id in selected_ids]
    ground_truth_c2w = np.stack([poses_by_id[frame_id] for frame_id in selected_ids])
    images = image_loader(image_paths, preprocess_mode)
    image_hw = (int(images.shape[-2]), int(images.shape[-1]))

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.no_grad():
        with torch.cuda.amp.autocast(
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            predictions = model(
                images.to(device),
                camera_num_iterations=max(iterations),
                return_camera_trace=True,
                camera_trace_pose_tokens=save_camera_tokens,
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inference_time_ms = (time.perf_counter() - started) * 1000.0
    peak_memory_mb = (
        float(torch.cuda.max_memory_allocated(device) / (1024**2))
        if device.type == "cuda"
        else 0.0
    )

    trace = predictions["camera_trace"]
    rows = build_iteration_rows(
        scene=scene,
        frame_count=len(selected_ids),
        requested_iterations=iterations,
        pose_enc_list=predictions["pose_enc_list"],
        delta_norm=trace["delta_norm"],
        gt_c2w=ground_truth_c2w,
        image_hw=image_hw,
    )
    for row in rows:
        row["frame_count_requested"] = requested_frame_count
        row["inference_time_ms"] = inference_time_ms
        row["peak_cuda_memory_mb"] = peak_memory_mb

    arrays = {
        "frame_ids": np.asarray(selected_ids, dtype=np.int64),
        "pose_enc_by_iteration": _tensor_stack(predictions["pose_enc_list"]),
        "raw_pose_enc_by_iteration": _tensor_stack(trace["raw_pose_enc_list"]),
        "pose_delta_by_iteration": _tensor_stack(trace["pose_delta_list"]),
        "delta_norm": trace["delta_norm"].detach().float().cpu().numpy(),
    }
    if save_camera_tokens:
        arrays["normalized_camera_tokens"] = (
            trace["normalized_camera_tokens"][0].detach().float().cpu().numpy()
        )
        arrays["pose_tokens_modulated"] = _tensor_stack(
            trace["pose_tokens_modulated_list"]
        )

    if save_context_diagnostics:
        final_extrinsic, _ = pose_encoding_to_extri_intri(
            predictions["pose_enc_list"][-1],
            image_hw,
            build_intrinsics=False,
        )
        diagnostics = build_context_diagnostics(
            frame_ids=np.asarray(selected_ids, dtype=np.int64),
            normalized_camera_tokens=trace["normalized_camera_tokens"][0]
            .detach()
            .float()
            .cpu()
            .numpy(),
            pred_w2c=to_homogeneous(
                final_extrinsic[0].detach().float().cpu().numpy()
            ),
            gt_c2w_raw=ground_truth_c2w,
            delta_norm=trace["delta_norm"][-1, 0]
            .detach()
            .float()
            .cpu()
            .numpy(),
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "iteration_metrics.json", rows)
    _atomic_write_csv(output_dir / "iteration_metrics.csv", rows)
    _atomic_save_npz(output_dir / "camera_trace.npz", arrays)
    if save_context_diagnostics:
        _atomic_save_npz(output_dir / "context_diagnostics.npz", diagnostics)
    atomic_write_json(output_dir / "selected_frame_ids.json", selected_ids)
    atomic_write_json(
        output_dir / "complete.json",
        {
            "run_id": run_id,
            "scene": scene,
            "frame_count_requested": requested_frame_count,
            "frame_count_actual": len(selected_ids),
            "selected_frame_ids": selected_ids,
            "iterations": iterations,
        },
    )
    return rows


def _read_metric_rows(path: Path) -> list[MetricRow]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not all(isinstance(row, dict) for row in data):
        raise ValueError(f"invalid metric rows in {path}")
    return data


def _write_summary(run_dir: Path, rows: list[MetricRow]) -> None:
    atomic_write_json(run_dir / "summary.json", rows)
    if rows:
        _atomic_write_csv(run_dir / "summary.csv", rows)


def _effective_invocation(
    args: argparse.Namespace,
    scenes: list[str],
    data_dir: Path,
    scene_list: Path,
    checkpoint_dir: Path,
    output_root: Path,
) -> dict[str, object]:
    return {
        "scenes": scenes,
        "data_dir": data_dir.as_posix(),
        "scene_list": scene_list.as_posix(),
        "checkpoint_dir": checkpoint_dir.as_posix(),
        "output_root": output_root.as_posix(),
        "frame_counts": args.frame_counts,
        "iterations": args.iterations,
        "sampling": args.sampling,
        "device": args.device,
        "preprocess_mode": args.preprocess_mode,
        "seed": args.seed,
        "save_camera_tokens": args.save_camera_tokens,
        "save_context_diagnostics": args.save_context_diagnostics,
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    data_dir = _resolve_repo_path(args.data_dir)
    scene_list = _resolve_repo_path(args.scene_list)
    checkpoint_dir = _resolve_repo_path(args.ckpt_dir)
    output_root = validate_output_root(args.out_dir, ROOT)
    scenes = read_scene_list(scene_list, args.scene_limit)
    if not scenes:
        raise ValueError(f"no scenes selected from {scene_list}")

    git_commit = read_git_commit(ROOT)
    invocation = _effective_invocation(
        args,
        scenes,
        data_dir,
        scene_list,
        checkpoint_dir,
        output_root,
    )
    metadata = build_run_metadata(git_commit, invocation)
    metadata.update(
        {
            "argv": list(sys.argv if argv is None else argv),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        }
    )
    run_id = str(metadata["run_id"])
    run_dir = output_root / run_id
    atomic_write_json(run_dir / "run_metadata.json", metadata)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = resolve_device(args.device)
    model = None
    all_rows: list[MetricRow] = []

    for scene in scenes:
        image_by_id, poses_by_id, valid_ids = load_scene_frames(data_dir, scene)
        selections = make_frame_selections(valid_ids, args.frame_counts, args.sampling)
        for requested_count, selected_ids in selections.items():
            selection_dir = run_dir / scene / f"frames_{requested_count}"
            if selection_is_complete(
                selection_dir,
                run_id,
                selected_ids,
                args.iterations,
                require_context_diagnostics=args.save_context_diagnostics,
            ):
                print(f"[resume] {scene} frames={requested_count}")
                all_rows.extend(_read_metric_rows(selection_dir / "iteration_metrics.json"))
                continue

            if model is None:
                model = load_local_model(checkpoint_dir).to(device).eval()
            print(
                f"[run] {scene} requested={requested_count} "
                f"actual={len(selected_ids)} iterations={args.iterations}"
            )
            rows = run_selection(
                model=model,
                scene=scene,
                requested_frame_count=requested_count,
                selected_ids=selected_ids,
                image_by_id=image_by_id,
                poses_by_id=poses_by_id,
                iterations=args.iterations,
                device=device,
                preprocess_mode=args.preprocess_mode,
                output_dir=selection_dir,
                run_id=run_id,
                save_camera_tokens=args.save_camera_tokens,
                save_context_diagnostics=args.save_context_diagnostics,
            )
            all_rows.extend(rows)
            _write_summary(run_dir, all_rows)

    _write_summary(run_dir, all_rows)
    print(f"[done] results={run_dir}")


if __name__ == "__main__":
    main()
