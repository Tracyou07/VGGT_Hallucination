"""Run frozen VGGT on Round 2A overlapping local windows."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Callable

import numpy as np
import torch

from pre_experiments.camera_iteration.contracts import atomic_write_json, read_git_commit
from pre_experiments.camera_iteration.model_io import load_local_model, resolve_device
from pre_experiments.camera_iteration.pose_metrics import to_homogeneous
from pre_experiments.camera_iteration.scannet import load_scene_frames
from pre_experiments.local_global_consistency.artifacts import (
    atomic_save_npz,
    build_window_diagnostics,
    load_global_context,
)
from pre_experiments.local_global_consistency.windows import FrameWindow, build_sliding_windows
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


ROOT = Path(__file__).resolve().parents[2]
AUTODL_TMP = Path(os.environ.get("AUTODL_TMP", "/root/autodl-tmp"))
DEFAULT_DATA = AUTODL_TMP / "datasets" / "scannetv2" / "process_scannet"
DEFAULT_CHECKPOINT = AUTODL_TMP / "ckpt" / "VGGT-1B"
DEFAULT_GLOBAL_RUN = ROOT / "results" / "camera_context" / "911b598_f4577f584448"
DEFAULT_OUTPUT = AUTODL_TMP / "local_global_consistency" / "results"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--source-run-dir", type=Path, default=DEFAULT_GLOBAL_RUN)
    parser.add_argument("--ckpt-dir", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-dir-file", type=Path)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--scene-limit", type=int, default=4)
    parser.add_argument("--window-length", type=int, default=100)
    parser.add_argument("--window-stride", type=int, default=50)
    parser.add_argument("--camera-iterations", type=int, choices=[4], default=4)
    parser.add_argument("--preprocess-mode", choices=["pad", "crop"], default="pad")
    args = parser.parse_args(argv)
    if args.scene_limit < 0:
        parser.error("--scene-limit must be non-negative")
    if args.window_length < 2 or args.window_stride < 1:
        parser.error("window length/stride must be positive and length at least 2")
    if args.window_stride > args.window_length:
        parser.error("--window-stride must not exceed --window-length")
    return args


def _json_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON object: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _source_scenes(source: Path, limit: int) -> tuple[dict[str, object], list[str]]:
    metadata = _json_object(source / "run_metadata.json")
    invocation = metadata.get("invocation")
    scenes = invocation.get("scenes") if isinstance(invocation, dict) else None
    if not isinstance(scenes, list) or not scenes or not all(isinstance(x, str) for x in scenes):
        raise ValueError("source run metadata must declare invocation.scenes")
    return metadata, scenes[:limit] if limit else scenes


def _run_id(commit: str, invocation: dict[str, object]) -> str:
    canonical = json.dumps(invocation, sort_keys=True, separators=(",", ":"))
    return f"{commit[:7]}_{hashlib.sha256(canonical.encode()).hexdigest()[:12]}"


def configure_camera_only(model: object) -> object:
    """Retain Aggregator and Camera Head while releasing unused prediction heads."""
    if getattr(model, "camera_head", None) is None:
        raise ValueError("loaded model must provide a Camera Head")
    for name in ("depth_head", "point_head", "track_head"):
        if hasattr(model, name):
            setattr(model, name, None)
    return model


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def window_is_complete(
    directory: Path,
    *,
    run_id: str,
    scene: str,
    window_index: int,
    start: int,
    stop: int,
    frame_ids: list[int],
) -> bool:
    try:
        artifact = load_window_diagnostics(directory / "window_diagnostics.npz")
        complete = _json_object(directory / "complete.json")
    except (ValueError, FileNotFoundError, OSError):
        return False
    return np.array_equal(artifact["frame_ids"], np.asarray(frame_ids)) and all(
        complete.get(key) == value
        for key, value in {
            "frame_ids": frame_ids,
            "run_id": run_id,
            "scene": scene,
            "start": start,
            "stop": stop,
            "window_index": window_index,
        }.items()
    )


def run_window(
    *,
    model: object,
    window: FrameWindow,
    scene: str,
    image_by_id: dict[int, Path],
    gt_c2w_raw: np.ndarray,
    device: torch.device,
    preprocess_mode: str,
    output_dir: Path,
    run_id: str,
    camera_iterations: int,
    image_loader: Callable[[list[str], str], torch.Tensor] = load_and_preprocess_images,
) -> dict[str, object]:
    image_paths = [str(image_by_id[frame_id]) for frame_id in window.frame_ids]
    images = image_loader(image_paths, preprocess_mode)
    image_hw = (int(images.shape[-2]), int(images.shape[-1]))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=device.type == "cuda"):
            predictions = model(
                images.to(device),
                camera_num_iterations=camera_iterations,
                return_camera_trace=True,
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    trace = predictions["camera_trace"]
    extrinsic, _ = pose_encoding_to_extri_intri(
        predictions["pose_enc_list"][-1], image_hw, build_intrinsics=False
    )
    artifact = build_window_diagnostics(
        frame_ids=np.asarray(window.frame_ids, dtype=np.int64),
        normalized_camera_tokens=trace["normalized_camera_tokens"][0]
        .detach()
        .float()
        .cpu()
        .numpy(),
        pred_w2c=to_homogeneous(extrinsic[0].detach().float().cpu().numpy()),
        gt_c2w_raw=gt_c2w_raw,
    )
    atomic_save_npz(output_dir / "window_diagnostics.npz", artifact)
    completion: dict[str, object] = {
        "run_id": run_id,
        "scene": scene,
        "window_index": window.index,
        "start": window.start,
        "stop": window.stop,
        "frame_ids": list(window.frame_ids),
        "inference_time_ms": elapsed_ms,
        "peak_cuda_memory_mb": (
            float(torch.cuda.max_memory_allocated(device) / (1024**2))
            if device.type == "cuda"
            else 0.0
        ),
    }
    atomic_write_json(output_dir / "complete.json", completion)
    return completion


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    source = args.source_run_dir.resolve()
    data_dir = args.data_dir.resolve()
    checkpoint = args.ckpt_dir.resolve()
    output = args.out_dir.resolve()
    source_metadata, scenes = _source_scenes(source, args.scene_limit)
    commit = read_git_commit(ROOT)
    device = resolve_device(args.device)
    invocation = {
        "source_run_dir": source.as_posix(),
        "source_run_id": source_metadata.get("run_id"),
        "data_dir": data_dir.as_posix(),
        "checkpoint_dir": checkpoint.as_posix(),
        "device": str(device),
        "scenes": scenes,
        "window_length": args.window_length,
        "window_stride": args.window_stride,
        "camera_iterations": args.camera_iterations,
        "preprocess_mode": args.preprocess_mode,
    }
    run_id = _run_id(commit, invocation)
    run_dir = output / run_id
    atomic_write_json(
        run_dir / "run_metadata.json",
        {
            "study_type": "method_pre_experiment",
            "study_name": "local_global_consistency",
            "git_commit": commit,
            "run_id": run_id,
            "source_run_id": source_metadata.get("run_id"),
            "invocation": invocation,
            "argv": list(sys.argv if argv is None else argv),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "score_policy": "all detection scores are prediction-only",
            "metric_policy": "prediction metrics are aligned to raw GT; GT remains raw",
        },
    )

    model = None
    manifest_rows: list[dict[str, object]] = []
    for scene in scenes:
        global_artifact = load_global_context(
            source / scene / "frames_500" / "context_diagnostics.npz"
        )
        frame_ids = global_artifact["frame_ids"]
        windows = build_sliding_windows(
            frame_ids, length=args.window_length, stride=args.window_stride
        )
        image_by_id, poses_by_id, _ = load_scene_frames(data_dir, scene)
        data_gt = np.stack([poses_by_id[int(frame_id)] for frame_id in frame_ids])
        if not np.allclose(data_gt, global_artifact["gt_c2w_raw"], atol=1e-10, rtol=0):
            raise ValueError(f"raw GT mismatch with published global artifact: {scene}")

        for window in windows:
            directory = run_dir / scene / f"window_{window.index:03d}"
            ids = list(window.frame_ids)
            if window_is_complete(
                directory,
                run_id=run_id,
                scene=scene,
                window_index=window.index,
                start=window.start,
                stop=window.stop,
                frame_ids=ids,
            ):
                print(f"[resume] {scene} window={window.index}")
                completion = _json_object(directory / "complete.json")
                manifest_rows.append(completion)
                continue
            if model is None:
                model = configure_camera_only(load_local_model(checkpoint)).to(device).eval()
            print(f"[run] {scene} window={window.index} start={window.start} stop={window.stop}")
            manifest_rows.append(
                run_window(
                    model=model,
                    window=window,
                    scene=scene,
                    image_by_id=image_by_id,
                    gt_c2w_raw=global_artifact["gt_c2w_raw"][window.start : window.stop],
                    device=device,
                    preprocess_mode=args.preprocess_mode,
                    output_dir=directory,
                    run_id=run_id,
                    camera_iterations=args.camera_iterations,
                )
            )
            atomic_write_json(run_dir / "window_manifest.json", manifest_rows)

    atomic_write_json(run_dir / "window_manifest.json", manifest_rows)
    if args.run_dir_file is not None:
        _atomic_write_text(args.run_dir_file.resolve(), f"{run_dir}\n")
    print(f"[done] local_windows={run_dir}")


if __name__ == "__main__":
    main()
