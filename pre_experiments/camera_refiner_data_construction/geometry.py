"""Prediction-only pose alignment for long/short Camera Head supervision."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from vggt.utils.rotation import mat_to_quat, quat_to_mat


@dataclass(frozen=True)
class PoseAlignment:
    aligned_pose: np.ndarray
    translation_residual: np.ndarray
    rotation_residual_deg: np.ndarray
    scale: float
    rotation: np.ndarray
    translation: np.ndarray


@dataclass(frozen=True)
class ConsensusShortPose:
    pose: np.ndarray
    selected_window: np.ndarray
    observation_count: np.ndarray
    selected_boundary_distance: np.ndarray


def _pose_array(name: str, value: np.ndarray) -> np.ndarray:
    pose = np.asarray(value, dtype=np.float64)
    if pose.ndim != 2 or pose.shape[1] != 9 or len(pose) < 2:
        raise ValueError(f"{name} must have shape [S, 9] with S >= 2")
    if not np.isfinite(pose).all():
        raise ValueError(f"{name} must contain only finite values")
    if np.any(np.linalg.norm(pose[:, 3:7], axis=1) <= 1e-8):
        raise ValueError(f"{name} contains a zero quaternion")
    if np.any(pose[:, 7:] < 0):
        raise ValueError(f"{name} contains a negative field of view")
    return pose


def _c2w_array(value: np.ndarray) -> np.ndarray:
    poses = np.asarray(value, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or len(poses) < 2:
        raise ValueError("c2w must have shape [S, 4, 4] with S >= 2")
    if not np.isfinite(poses).all():
        raise ValueError("c2w must contain only finite values")
    if not np.allclose(poses[:, 3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise ValueError("c2w homogeneous rows are invalid")
    rotations = poses[:, :3, :3]
    identity = np.eye(3, dtype=np.float64)
    if not np.allclose(
        np.einsum("sji,sjk->sik", rotations, rotations), identity, atol=1e-5
    ) or not np.allclose(np.linalg.det(rotations), 1.0, atol=1e-5):
        raise ValueError("c2w rotations must be proper orthonormal matrices")
    return poses


def pose_encoding_to_c2w(pose_encoding: np.ndarray) -> np.ndarray:
    """Convert activated VGGT absT_quaR_FoV values to OpenCV c2w matrices."""
    pose = _pose_array("pose_encoding", pose_encoding)
    quaternion = torch.from_numpy(pose[:, 3:7])
    rotation = quat_to_mat(quaternion).cpu().numpy()
    w2c = np.tile(np.eye(4, dtype=np.float64), (len(pose), 1, 1))
    w2c[:, :3, :3] = rotation
    w2c[:, :3, 3] = pose[:, :3]
    return np.linalg.inv(w2c)


def c2w_to_pose_encoding(c2w: np.ndarray, fov: np.ndarray) -> np.ndarray:
    """Encode c2w matrices while preserving externally supplied VGGT FOV values."""
    poses = _c2w_array(c2w)
    field_of_view = np.asarray(fov, dtype=np.float64)
    if field_of_view.shape != (len(poses), 2):
        raise ValueError("fov must have shape [S, 2]")
    if not np.isfinite(field_of_view).all() or np.any(field_of_view < 0):
        raise ValueError("fov must be finite and non-negative")
    w2c = np.linalg.inv(poses)
    quaternion = mat_to_quat(torch.from_numpy(w2c[:, :3, :3])).cpu().numpy()
    return np.concatenate((w2c[:, :3, 3], quaternion, field_of_view), axis=1).astype(
        np.float32
    )


def _umeyama(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("alignment points must have matching shape [S, 3]")
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    variance = float(np.mean(np.sum(source_centered * source_centered, axis=1)))
    if variance <= 1e-12:
        raise ValueError("moving trajectory has insufficient translation variance")
    covariance = (target_centered.T @ source_centered) / len(source)
    left, singular_values, right_transpose = np.linalg.svd(covariance)
    signs = np.ones(3, dtype=np.float64)
    if np.linalg.det(left @ right_transpose) < 0:
        signs[-1] = -1.0
    rotation = left @ np.diag(signs) @ right_transpose
    scale = float(np.sum(singular_values * signs) / variance)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("pose alignment produced a non-positive scale")
    translation = target_mean - scale * (rotation @ source_mean)
    return scale, rotation, translation


def _rotation_angle_deg(value: np.ndarray) -> float:
    cosine = float(np.clip((np.trace(value) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def align_pose_to_reference(
    reference_pose: np.ndarray,
    moving_pose: np.ndarray,
) -> PoseAlignment:
    """Transform one predicted trajectory into another prediction's Sim(3) gauge."""
    reference = _pose_array("reference_pose", reference_pose)
    moving = _pose_array("moving_pose", moving_pose)
    if reference.shape != moving.shape:
        raise ValueError("reference and moving poses must have the same shape")
    reference_c2w = pose_encoding_to_c2w(reference)
    moving_c2w = pose_encoding_to_c2w(moving)
    scale, rotation, translation = _umeyama(
        moving_c2w[:, :3, 3], reference_c2w[:, :3, 3]
    )
    aligned = moving_c2w.copy()
    aligned[:, :3, :3] = np.einsum(
        "ij,sjk->sik", rotation, moving_c2w[:, :3, :3]
    )
    aligned[:, :3, 3] = (
        scale * (moving_c2w[:, :3, 3] @ rotation.T) + translation
    )
    translation_residual = np.linalg.norm(
        aligned[:, :3, 3] - reference_c2w[:, :3, 3], axis=1
    )
    rotation_residual = np.asarray(
        [
            _rotation_angle_deg(candidate[:3, :3].T @ expected[:3, :3])
            for candidate, expected in zip(aligned, reference_c2w)
        ],
        dtype=np.float32,
    )
    return PoseAlignment(
        aligned_pose=c2w_to_pose_encoding(aligned, moving[:, 7:]),
        translation_residual=translation_residual.astype(np.float32),
        rotation_residual_deg=rotation_residual,
        scale=scale,
        rotation=rotation,
        translation=translation,
    )


def select_consensus_short_pose(
    *,
    frame_count: int,
    frame_indices: tuple[np.ndarray, ...],
    aligned_poses: tuple[np.ndarray, ...],
) -> ConsensusShortPose:
    """Select the least boundary-affected short-window observation per frame."""
    if frame_count < 2 or not frame_indices or len(frame_indices) != len(aligned_poses):
        raise ValueError("short-pose observations are incomplete")
    selected = np.full(frame_count, -1, dtype=np.int64)
    selected_distance = np.full(frame_count, -1, dtype=np.int64)
    observation_count = np.zeros(frame_count, dtype=np.int64)
    pose = np.zeros((frame_count, 9), dtype=np.float32)
    for window_index, (raw_indices, raw_pose) in enumerate(
        zip(frame_indices, aligned_poses)
    ):
        indices = np.asarray(raw_indices, dtype=np.int64)
        values = _pose_array("aligned_short_pose", raw_pose).astype(np.float32)
        if indices.ndim != 1 or len(indices) != len(values):
            raise ValueError("short-window frame indices and poses must match")
        if (
            np.any(indices < 0)
            or np.any(indices >= frame_count)
            or np.any(np.diff(indices) <= 0)
        ):
            raise ValueError("short-window frame indices are invalid")
        boundary = np.minimum(np.arange(len(indices)), np.arange(len(indices))[::-1])
        for local_index, frame_index in enumerate(indices):
            observation_count[frame_index] += 1
            if boundary[local_index] > selected_distance[frame_index]:
                selected[frame_index] = window_index
                selected_distance[frame_index] = boundary[local_index]
                pose[frame_index] = values[local_index]
    if np.any(selected < 0):
        missing = np.flatnonzero(selected < 0).tolist()
        raise ValueError(f"short windows do not cover frames: {missing}")
    return ConsensusShortPose(
        pose=pose,
        selected_window=selected,
        observation_count=observation_count,
        selected_boundary_distance=selected_distance,
    )
