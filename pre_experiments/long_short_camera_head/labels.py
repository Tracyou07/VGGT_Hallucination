from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn

from pre_experiments.camera_velocity_ambiguity_02.frozen_oracle import (
    apply_frozen_oracle,
    evaluate_with_frozen_oracle,
    fit_frozen_oracle,
)
from pre_experiments.camera_velocity_ambiguity_02.geometry import (
    align_local_to_global,
    global_scene_scale,
)
from pre_experiments.variational_camera_latent.camera import (
    decode_camera_tokens,
    pose_encoding_to_c2w,
)
from pre_experiments.variational_camera_latent.source import load_source_shard

from .data import load_prepared_gt, sha256_file


PRIVILEGED_MEMBERS = {
    "scene",
    "frame_ids",
    "gt_c2w",
    "oracle_scale",
    "oracle_rotation",
    "oracle_translation",
    "oracle_digest",
    "gt_scene_scale",
    "baseline_pose_encoding",
    "teacher_c2w_gt_gauge",
    "teacher_weight",
    "window_teacher_weight",
    "window_baseline_rms",
    "window_teacher_rms",
    "source_sha256",
    "checkpoint_sha256",
}


@dataclass(frozen=True)
class PrivilegedRecord:
    scene: str
    path: Path
    sha256: str
    teacher_frame_count: int


def positive_teacher_weight(baseline_rms: float, teacher_rms: float) -> float:
    values = np.asarray([baseline_rms, teacher_rms], dtype=np.float64)
    if not np.isfinite(values).all() or baseline_rms <= 0.0 or teacher_rms < 0.0:
        raise ValueError("teacher errors must be finite with positive baseline")
    return float(np.clip((baseline_rms - teacher_rms) / baseline_rms, 0.0, 1.0))


def fuse_teacher_trajectories(
    *,
    frame_count: int,
    windows: Iterable[tuple[int, np.ndarray, float]],
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(frame_count, bool) or frame_count < 1:
        raise ValueError("frame_count must be positive")
    centers = np.zeros((frame_count, 3), dtype=np.float64)
    weights = np.zeros(frame_count, dtype=np.float64)
    best_weight = np.full(frame_count, -1.0, dtype=np.float64)
    best_rotation = np.full((frame_count, 3, 3), np.nan, dtype=np.float64)
    for start, poses, weight in windows:
        poses = np.asarray(poses, dtype=np.float64)
        if (
            isinstance(start, bool)
            or start < 0
            or poses.ndim != 3
            or poses.shape[1:] != (4, 4)
            or start + len(poses) > frame_count
            or not np.isfinite(poses).all()
            or not np.isfinite(weight)
            or weight < 0.0
            or weight > 1.0
        ):
            raise ValueError("teacher window is malformed")
        if weight == 0.0:
            continue
        stop = start + len(poses)
        centers[start:stop] += weight * poses[:, :3, 3]
        weights[start:stop] += weight
        replace = weight > best_weight[start:stop]
        local_indices = np.flatnonzero(replace)
        best_rotation[start + local_indices] = poses[local_indices, :3, :3]
        best_weight[start + local_indices] = weight
    output = np.full((frame_count, 4, 4), np.nan, dtype=np.float64)
    valid = weights > 0.0
    if np.any(valid):
        output[valid, :3, :3] = best_rotation[valid]
        output[valid, :3, 3] = centers[valid] / weights[valid, None]
        output[valid, 3, :] = [0.0, 0.0, 0.0, 1.0]
    return output, weights


def construct_privileged_arrays(
    *,
    scene: str,
    frame_ids: np.ndarray,
    source_sha256: str,
    checkpoint_sha256: str,
    baseline_pose_encoding: np.ndarray,
    baseline_c2w: np.ndarray,
    short_c2w: np.ndarray,
    gt_c2w: np.ndarray,
) -> dict[str, np.ndarray]:
    """Construct quality-weighted short teachers using one frozen scene alignment."""
    frame_ids = np.asarray(frame_ids)
    baseline_pose_encoding = np.asarray(baseline_pose_encoding, dtype=np.float32)
    baseline_c2w = np.asarray(baseline_c2w, dtype=np.float64)
    short_c2w = np.asarray(short_c2w, dtype=np.float64)
    gt_c2w = np.asarray(gt_c2w, dtype=np.float64)
    if frame_ids.shape != (500,) or not np.issubdtype(frame_ids.dtype, np.integer):
        raise ValueError("label frame IDs must be an integer vector with shape [500]")
    if baseline_pose_encoding.shape != (500, 9):
        raise ValueError("baseline pose encoding must have shape [500,9]")
    if baseline_c2w.shape != (500, 4, 4) or gt_c2w.shape != (500, 4, 4):
        raise ValueError("baseline and GT poses must have shape [500,4,4]")
    if short_c2w.shape != (9, 100, 4, 4):
        raise ValueError("short-window poses must have shape [9,100,4,4]")
    if not all(
        np.isfinite(value).all()
        for value in (baseline_pose_encoding, baseline_c2w, short_c2w, gt_c2w)
    ):
        raise ValueError("label construction inputs must be finite")
    if len(source_sha256) != 64 or len(checkpoint_sha256) != 64:
        raise ValueError("label construction digests must be SHA-256 strings")

    oracle = fit_frozen_oracle(scene, frame_ids, baseline_c2w, gt_c2w)
    prediction_scale = global_scene_scale(baseline_c2w)
    gt_centers = gt_c2w[:, :3, 3]
    gt_scale = float(
        np.sqrt(np.mean(np.sum((gt_centers - gt_centers.mean(axis=0)) ** 2, axis=1)))
    )
    if not np.isfinite(gt_scale) or gt_scale <= 1e-12:
        raise ValueError("GT trajectory has insufficient scale")

    teacher_windows: list[tuple[int, np.ndarray, float]] = []
    baseline_rms = np.empty(9, dtype=np.float64)
    teacher_rms = np.empty(9, dtype=np.float64)
    teacher_weight = np.zeros(9, dtype=np.float64)
    for index, start in enumerate(range(0, 401, 50)):
        stop = start + 100
        baseline_segment = baseline_c2w[start:stop]
        gt_segment = gt_c2w[start:stop]
        baseline_rms[index] = evaluate_with_frozen_oracle(
            oracle, baseline_segment, gt_segment
        ).rms_translation_error
        aligned = align_local_to_global(
            baseline_segment,
            short_c2w[index],
            scene_scale=prediction_scale,
        )
        if not aligned.valid or aligned.aligned_c2w is None:
            teacher_rms[index] = baseline_rms[index]
            continue
        teacher_rms[index] = evaluate_with_frozen_oracle(
            oracle, aligned.aligned_c2w, gt_segment
        ).rms_translation_error
        teacher_weight[index] = positive_teacher_weight(
            baseline_rms[index], teacher_rms[index]
        )
        if teacher_weight[index] > 0.0:
            teacher_windows.append(
                (
                    start,
                    apply_frozen_oracle(oracle, aligned.aligned_c2w),
                    float(teacher_weight[index]),
                )
            )

    fused_teacher, frame_weights = fuse_teacher_trajectories(
        frame_count=500,
        windows=teacher_windows,
    )
    arrays = {
        "scene": np.asarray(scene, dtype="U32"),
        "frame_ids": frame_ids.astype(np.int64, copy=True),
        "gt_c2w": gt_c2w.copy(),
        "oracle_scale": np.asarray(oracle.scale, dtype=np.float64),
        "oracle_rotation": np.asarray(oracle.rotation, dtype=np.float64),
        "oracle_translation": np.asarray(oracle.translation, dtype=np.float64),
        "oracle_digest": np.asarray(oracle.transform_digest, dtype="U64"),
        "gt_scene_scale": np.asarray(gt_scale, dtype=np.float64),
        "baseline_pose_encoding": baseline_pose_encoding.copy(),
        "teacher_c2w_gt_gauge": fused_teacher,
        "teacher_weight": frame_weights,
        "window_teacher_weight": teacher_weight,
        "window_baseline_rms": baseline_rms,
        "window_teacher_rms": teacher_rms,
        "source_sha256": np.asarray(source_sha256, dtype="U64"),
        "checkpoint_sha256": np.asarray(checkpoint_sha256, dtype="U64"),
    }
    _validate_privileged(arrays)
    return arrays


def build_privileged_labels(
    source_path: Path,
    prepared_scene: Path,
    camera_head: nn.Module | object,
    destination: Path,
    *,
    checkpoint_sha256: str,
    device: torch.device,
) -> PrivilegedRecord:
    """Decode the frozen teacher once and publish a training-only label shard."""
    source_path = Path(source_path)
    source = load_source_shard(source_path)
    if "global_pred_c2w" not in source:
        raise ValueError("source shard lacks authenticated baseline camera poses")
    scene = str(source["sample_ids"][0]).split(":", 1)[0]
    if any(not str(sample).startswith(scene + ":") for sample in source["sample_ids"]):
        raise ValueError("source sample IDs do not share one scene")
    global_tokens = torch.from_numpy(source["global_camera_tokens"]).unsqueeze(0).to(device)
    short_tokens = torch.from_numpy(source["short_camera_tokens"]).to(device)
    baseline_pose = decode_camera_tokens(camera_head, global_tokens)[0]
    short_pose = decode_camera_tokens(camera_head, short_tokens)
    baseline_c2w = pose_encoding_to_c2w(baseline_pose.unsqueeze(0))[0]
    short_c2w = pose_encoding_to_c2w(short_pose)
    baseline_pose_np = baseline_pose.float().cpu().numpy()
    baseline_c2w_np = baseline_c2w.double().cpu().numpy()
    short_c2w_np = short_c2w.double().cpu().numpy()
    if not np.allclose(
        baseline_c2w_np,
        source["global_pred_c2w"],
        atol=2e-4,
        rtol=2e-4,
    ):
        raise ValueError("frozen Camera Head does not reproduce authenticated baseline")
    gt_c2w = load_prepared_gt(prepared_scene, source["global_frame_ids"])
    arrays = construct_privileged_arrays(
        scene=scene,
        frame_ids=source["global_frame_ids"],
        source_sha256=sha256_file(source_path),
        checkpoint_sha256=checkpoint_sha256,
        baseline_pose_encoding=baseline_pose_np,
        baseline_c2w=baseline_c2w_np,
        short_c2w=short_c2w_np,
        gt_c2w=gt_c2w,
    )
    destination = Path(destination)
    digest = save_privileged_labels(destination, arrays)
    return PrivilegedRecord(
        scene=scene,
        path=destination,
        sha256=digest,
        teacher_frame_count=int(np.count_nonzero(arrays["teacher_weight"])),
    )


def _validate_privileged(arrays: dict[str, np.ndarray]) -> None:
    if set(arrays) != PRIVILEGED_MEMBERS:
        raise ValueError("privileged members do not match the strict schema")
    if any(value.dtype.hasobject for value in arrays.values()):
        raise ValueError("privileged arrays may not use object dtype")
    expected = {
        "scene": (),
        "frame_ids": (500,),
        "gt_c2w": (500, 4, 4),
        "oracle_scale": (),
        "oracle_rotation": (3, 3),
        "oracle_translation": (3,),
        "oracle_digest": (),
        "gt_scene_scale": (),
        "baseline_pose_encoding": (500, 9),
        "teacher_c2w_gt_gauge": (500, 4, 4),
        "teacher_weight": (500,),
        "window_teacher_weight": (9,),
        "window_baseline_rms": (9,),
        "window_teacher_rms": (9,),
        "source_sha256": (),
        "checkpoint_sha256": (),
    }
    for name, shape in expected.items():
        if arrays[name].shape != shape:
            raise ValueError(f"privileged member {name} has invalid shape")
    for name in ("scene", "oracle_digest", "source_sha256", "checkpoint_sha256"):
        if arrays[name].dtype.kind != "U":
            raise ValueError(f"privileged member {name} must be Unicode")
    weights = arrays["teacher_weight"]
    if not np.isfinite(weights).all() or np.any(weights < 0.0):
        raise ValueError("teacher weights must be finite and nonnegative")
    teacher = arrays["teacher_c2w_gt_gauge"]
    valid = weights > 0.0
    if np.any(valid) and not np.isfinite(teacher[valid]).all():
        raise ValueError("valid teacher poses must be finite")
    if np.any(~valid) and not np.isnan(teacher[~valid]).all():
        raise ValueError("invalid teacher poses must be all-NaN")
    for name, value in arrays.items():
        if name == "teacher_c2w_gt_gauge":
            continue
        if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
            raise ValueError(f"privileged member {name} must be finite")
    if not np.allclose(arrays["gt_c2w"][:, 3, :], [0.0, 0.0, 0.0, 1.0]):
        raise ValueError("GT poses must be homogeneous")


def save_privileged_labels(path: Path, arrays: dict[str, np.ndarray]) -> str:
    normalized = {name: np.asarray(value) for name, value in arrays.items()}
    _validate_privileged(normalized)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **normalized)
    temporary.replace(path)
    return sha256_file(path)


def load_privileged_labels(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(Path(path), allow_pickle=False) as archive:
            arrays = {name: np.asarray(archive[name]).copy() for name in archive.files}
    except (OSError, ValueError, KeyError) as error:
        raise ValueError(f"invalid privileged label shard: {path}") from error
    _validate_privileged(arrays)
    return arrays
