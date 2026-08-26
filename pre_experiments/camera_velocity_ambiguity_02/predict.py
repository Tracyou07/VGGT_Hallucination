"""Resumable prediction-only global/local Camera Token extraction for CVA02."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch

from pre_experiments.camera_velocity_ambiguity_02.artifacts import (
    IncompletePredictionArtifact,
    PredictionIdentity,
    build_prediction_identity,
    load_completed_prediction,
    save_completed_prediction,
)
from pre_experiments.camera_velocity_ambiguity_02.frames import (
    FrameSelection,
    build_protocol_windows,
)
from pre_experiments.common.model_io import load_local_model
from pre_experiments.common.pose_metrics import to_homogeneous
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


ImageLoader = Callable[[list[str], str], torch.Tensor]
PoseDecoder = Callable[..., tuple[torch.Tensor, object]]


@dataclass(frozen=True)
class PredictionContext:
    """Run-level provenance shared by every prediction unit."""

    run_id: str
    checkpoint_sha256: str
    git_commit: str
    protocol_digest: str
    preprocess: str = "crop"
    camera_iterations: int = 4


def select_scene_shard(
    scenes: Sequence[str], *, shard_index: int, shard_count: int
) -> tuple[str, ...]:
    """Return an order-preserving, disjoint strided scene shard."""
    if isinstance(shard_count, bool) or not isinstance(shard_count, int) or shard_count < 1:
        raise ValueError("shard_count must be a positive integer")
    if (
        isinstance(shard_index, bool)
        or not isinstance(shard_index, int)
        or shard_index < 0
        or shard_index >= shard_count
    ):
        raise ValueError("shard_index must satisfy 0 <= shard_index < shard_count")
    if len(set(scenes)) != len(scenes):
        raise ValueError("scene order must be unique")
    return tuple(scenes[shard_index::shard_count])


def configure_camera_only(model: object) -> object:
    """Retain the Aggregator and Camera Head and disable every unused head."""
    if getattr(model, "camera_head", None) is None:
        raise ValueError("loaded model must provide a Camera Head")
    for name in ("depth_head", "point_head", "track_head"):
        if hasattr(model, name):
            setattr(model, name, None)
    return model


def load_local_camera_model(checkpoint_dir: Path, device: torch.device) -> object:
    """Load only from a local checkpoint; this function has no network fallback."""
    model = configure_camera_only(load_local_model(Path(checkpoint_dir)))
    return model.to(device).eval()


def _identity(
    context: PredictionContext,
    *,
    scene: str,
    artifact_kind: str,
    window_index: int | None,
    frame_ids: np.ndarray,
) -> PredictionIdentity:
    return build_prediction_identity(
        run_id=context.run_id,
        scene=scene,
        artifact_kind=artifact_kind,
        window_index=window_index,
        frame_ids=frame_ids,
        checkpoint_sha256=context.checkpoint_sha256,
        git_commit=context.git_commit,
        protocol_digest=context.protocol_digest,
        preprocess=context.preprocess,
        camera_iterations=context.camera_iterations,
    )


def _predict_sequence(
    *,
    model: object,
    frame_ids: np.ndarray,
    image_paths: Sequence[Path],
    context: PredictionContext,
    device: torch.device,
    image_loader: ImageLoader,
    pose_decoder: PoseDecoder,
) -> dict[str, np.ndarray]:
    if len(image_paths) != len(frame_ids):
        raise ValueError("image paths must match frame IDs")
    images = image_loader([str(path) for path in image_paths], context.preprocess)
    if not isinstance(images, torch.Tensor) or images.ndim != 4:
        raise ValueError("image loader must return [frames, 3, height, width] tensor")
    if images.shape[0] != len(frame_ids) or images.shape[1] != 3:
        raise ValueError("image loader returned a tensor with wrong frame or channel count")
    if images.shape[-2] < 1 or images.shape[-1] < 1 or not torch.isfinite(images).all():
        raise ValueError("image loader returned invalid image values or dimensions")

    image_hw = (int(images.shape[-2]), int(images.shape[-1]))
    with torch.no_grad():
        with torch.amp.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            predictions = model(
                images.to(device),
                camera_num_iterations=context.camera_iterations,
                return_camera_trace=True,
            )
    if not isinstance(predictions, dict):
        raise ValueError("VGGT prediction must be a dictionary")
    pose_enc_list = predictions.get("pose_enc_list")
    trace = predictions.get("camera_trace")
    if not isinstance(pose_enc_list, list) or not pose_enc_list:
        raise ValueError("VGGT prediction must contain a non-empty pose_enc_list")
    if not isinstance(trace, dict) or "normalized_camera_tokens" not in trace:
        raise ValueError("VGGT prediction must contain normalized Camera Tokens")
    pose_encoding = pose_enc_list[-1]
    tokens = trace["normalized_camera_tokens"]
    count = len(frame_ids)
    if not isinstance(pose_encoding, torch.Tensor) or pose_encoding.shape != (1, count, 9):
        raise ValueError("final pose encoding must have shape [1, frames, 9]")
    if not isinstance(tokens, torch.Tensor) or tokens.ndim != 3 or tokens.shape[:2] != (1, count):
        raise ValueError("normalized Camera Tokens must have shape [1, frames, channels]")

    extrinsic, _ = pose_decoder(
        pose_encoding,
        image_hw,
        build_intrinsics=False,
    )
    if not isinstance(extrinsic, torch.Tensor) or extrinsic.shape != (1, count, 3, 4):
        raise ValueError("decoded world-to-camera extrinsics must have shape [1, frames, 3, 4]")
    token_array = tokens[0].detach().float().cpu().numpy()
    w2c = to_homogeneous(extrinsic[0].detach().float().cpu().numpy())
    try:
        c2w = np.linalg.inv(w2c)
    except np.linalg.LinAlgError as error:
        raise ValueError("decoded world-to-camera extrinsics must be invertible") from error
    return {
        "frame_ids": np.asarray(frame_ids, dtype=np.int64),
        "normalized_camera_tokens": token_array,
        "pred_c2w_raw": c2w,
    }


def _run_or_resume(
    *,
    model: object,
    frame_ids: np.ndarray,
    image_paths: Sequence[Path],
    directory: Path,
    identity: PredictionIdentity,
    context: PredictionContext,
    device: torch.device,
    image_loader: ImageLoader,
    pose_decoder: PoseDecoder,
) -> bool:
    artifact_path = directory / "prediction.npz"
    completion_path = directory / "complete.json"
    try:
        load_completed_prediction(artifact_path, completion_path, identity)
        return False
    except IncompletePredictionArtifact:
        pass
    arrays = _predict_sequence(
        model=model,
        frame_ids=frame_ids,
        image_paths=image_paths,
        context=context,
        device=device,
        image_loader=image_loader,
        pose_decoder=pose_decoder,
    )
    save_completed_prediction(artifact_path, completion_path, arrays, identity)
    return True


def run_scene_predictions(
    *,
    model: object,
    scene: str,
    selection: FrameSelection,
    output_dir: Path,
    context: PredictionContext,
    device: torch.device,
    window_length: int = 100,
    window_stride: int = 50,
    image_loader: ImageLoader = load_and_preprocess_images,
    pose_decoder: PoseDecoder = pose_encoding_to_extri_intri,
) -> dict[str, int]:
    """Run global once and every local window, resuming only exact completions."""
    configure_camera_only(model)
    if not (
        len(selection.frame_ids)
        == len(selection.image_paths)
        == len(selection.pose_indices)
    ):
        raise ValueError("frame selection members must have matching lengths")
    frame_ids = np.asarray(selection.frame_ids, dtype=np.int64)
    windows = build_protocol_windows(
        selection,
        length=window_length,
        stride=window_stride,
    )
    root = Path(output_dir)
    summary = {"global_ran": 0, "local_ran": 0, "resumed": 0}

    global_identity = _identity(
        context,
        scene=scene,
        artifact_kind="global",
        window_index=None,
        frame_ids=frame_ids,
    )
    if _run_or_resume(
        model=model,
        frame_ids=frame_ids,
        image_paths=selection.image_paths,
        directory=root / "global",
        identity=global_identity,
        context=context,
        device=device,
        image_loader=image_loader,
        pose_decoder=pose_decoder,
    ):
        summary["global_ran"] += 1
    else:
        summary["resumed"] += 1

    for window in windows:
        local_ids = np.asarray(window.frame_ids, dtype=np.int64)
        local_identity = _identity(
            context,
            scene=scene,
            artifact_kind="local",
            window_index=window.index,
            frame_ids=local_ids,
        )
        if _run_or_resume(
            model=model,
            frame_ids=local_ids,
            image_paths=selection.image_paths[window.start : window.stop],
            directory=root / "local" / f"window_{window.index:03d}",
            identity=local_identity,
            context=context,
            device=device,
            image_loader=image_loader,
            pose_decoder=pose_decoder,
        ):
            summary["local_ran"] += 1
        else:
            summary["resumed"] += 1
    return summary
