"""Self-contained aligned camera-pose metrics for the iteration study."""

from __future__ import annotations

import math

import numpy as np


def _validate_pose_stack(name: str, poses: np.ndarray) -> np.ndarray:
    array = np.asarray(poses, dtype=np.float64)
    if array.ndim != 3 or array.shape[1:] != (4, 4):
        raise ValueError(f"{name} must have shape [S, 4, 4]")
    if len(array) < 2:
        raise ValueError(f"{name} must contain at least two poses")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def to_homogeneous(extrinsic: np.ndarray) -> np.ndarray:
    """Convert a stack of OpenCV world-to-camera [R|t] matrices to 4x4."""
    array = np.asarray(extrinsic, dtype=np.float64)
    if array.ndim != 3 or array.shape[1:] != (3, 4):
        raise ValueError("extrinsic must have shape [S, 3, 4]")
    matrices = np.tile(np.eye(4, dtype=np.float64), (len(array), 1, 1))
    matrices[:, :3, :4] = array
    return matrices


def invert_poses(poses: np.ndarray) -> np.ndarray:
    """Invert a validated stack of homogeneous camera poses."""
    return np.linalg.inv(_validate_pose_stack("poses", poses))


def rotation_angle_deg(rotation: np.ndarray) -> float:
    """Return the geodesic angle of a 3x3 rotation matrix in degrees."""
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError("rotation must have shape [3, 3]")
    value = float(np.clip((np.trace(matrix) - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(value))


def umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Estimate the orientation-preserving Sim(3) mapping src points to dst."""
    source = np.asarray(src, dtype=np.float64)
    target = np.asarray(dst, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("src and dst must have matching shape [S, 3]")
    if len(source) < 2 or not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("src and dst must contain at least two finite points")

    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    source_variance = float(np.mean(np.sum(source_centered * source_centered, axis=1)))
    if source_variance <= 1e-12:
        raise ValueError("src trajectory has insufficient translation variance")

    covariance = (target_centered.T @ source_centered) / len(source)
    left, singular_values, right_transpose = np.linalg.svd(covariance)
    signs = np.ones(3, dtype=np.float64)
    if np.linalg.det(left @ right_transpose) < 0:
        signs[-1] = -1.0
    rotation = left @ np.diag(signs) @ right_transpose
    scale = float(np.sum(singular_values * signs) / source_variance)
    translation = target_mean - scale * (rotation @ source_mean)
    return scale, rotation, translation


def evaluate_pose(pred_w2c: np.ndarray, gt_c2w: np.ndarray) -> dict[str, float]:
    """Evaluate predicted poses after Sim(3) alignment to raw GT camera poses."""
    predicted_w2c = _validate_pose_stack("pred_w2c", pred_w2c)
    ground_truth_c2w = _validate_pose_stack("gt_c2w", gt_c2w)
    if len(predicted_w2c) != len(ground_truth_c2w):
        raise ValueError("pred_w2c and gt_c2w must contain the same number of poses")

    predicted_c2w = np.linalg.inv(predicted_w2c)
    predicted_centers = predicted_c2w[:, :3, 3]
    ground_truth_centers = ground_truth_c2w[:, :3, 3]
    scale, alignment_rotation, translation = umeyama(predicted_centers, ground_truth_centers)
    aligned_centers = scale * (predicted_centers @ alignment_rotation.T) + translation
    translation_errors = np.linalg.norm(aligned_centers - ground_truth_centers, axis=1)

    aligned_c2w = predicted_c2w.copy()
    aligned_c2w[:, :3, :3] = np.einsum(
        "ij,sjk->sik",
        alignment_rotation,
        predicted_c2w[:, :3, :3],
    )
    aligned_c2w[:, :3, 3] = aligned_centers

    rotation_errors = []
    for aligned_pose, ground_truth_pose in zip(aligned_c2w, ground_truth_c2w):
        rotation_errors.append(
            rotation_angle_deg(aligned_pose[:3, :3].T @ ground_truth_pose[:3, :3])
        )

    relative_rotation_errors = []
    relative_translation_errors = []
    for index in range(1, len(ground_truth_c2w)):
        predicted_relative = np.linalg.inv(aligned_c2w[index - 1]) @ aligned_c2w[index]
        ground_truth_relative = (
            np.linalg.inv(ground_truth_c2w[index - 1]) @ ground_truth_c2w[index]
        )
        relative_error = np.linalg.inv(ground_truth_relative) @ predicted_relative
        relative_rotation_errors.append(rotation_angle_deg(relative_error[:3, :3]))
        relative_translation_errors.append(np.linalg.norm(relative_error[:3, 3]))

    return {
        "pose_ate_rmse_aligned": float(np.sqrt(np.mean(translation_errors**2))),
        "pose_ate_mean_aligned": float(np.mean(translation_errors)),
        "pose_ate_max_aligned": float(np.max(translation_errors)),
        "pose_are_mean_deg_aligned": float(np.mean(rotation_errors)),
        "pose_are_max_deg_aligned": float(np.max(rotation_errors)),
        "pose_rpe_rot_mean_deg": float(np.mean(relative_rotation_errors)),
        "pose_rpe_trans_mean_aligned": float(np.mean(relative_translation_errors)),
        "pose_sim3_scale": scale,
    }
