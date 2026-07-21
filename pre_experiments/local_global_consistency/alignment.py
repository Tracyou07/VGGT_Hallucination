"""Prediction-to-prediction trajectory alignment without ground truth."""

from __future__ import annotations

import numpy as np

from pre_experiments.camera_iteration.pose_metrics import rotation_angle_deg, umeyama


def _pose_stack(name: str, value: np.ndarray) -> np.ndarray:
    poses = np.asarray(value, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or len(poses) < 2:
        raise ValueError(f"{name} must have shape [S, 4, 4] with S >= 2")
    if not np.isfinite(poses).all():
        raise ValueError(f"{name} must contain only finite values")
    return poses


def align_prediction_trajectories(
    reference_c2w: np.ndarray,
    moving_c2w: np.ndarray,
) -> dict[str, np.ndarray | float]:
    """Align one predicted c2w trajectory to another predicted trajectory."""
    reference = _pose_stack("reference_c2w", reference_c2w)
    moving = _pose_stack("moving_c2w", moving_c2w)
    if reference.shape != moving.shape:
        raise ValueError("prediction trajectories must have the same shape")

    scale, rotation, translation = umeyama(
        moving[:, :3, 3],
        reference[:, :3, 3],
    )
    aligned = moving.copy()
    aligned[:, :3, :3] = np.einsum("ij,sjk->sik", rotation, moving[:, :3, :3])
    aligned[:, :3, 3] = scale * (moving[:, :3, 3] @ rotation.T) + translation
    translation_residual = np.linalg.norm(
        aligned[:, :3, 3] - reference[:, :3, 3], axis=1
    )
    rotation_residual = np.asarray(
        [
            rotation_angle_deg(aligned_pose[:3, :3].T @ reference_pose[:3, :3])
            for aligned_pose, reference_pose in zip(aligned, reference)
        ],
        dtype=np.float64,
    )
    return {
        "aligned_c2w": aligned,
        "translation_residual": translation_residual,
        "rotation_residual_deg": rotation_residual,
        "sim3_scale": scale,
        "sim3_rotation": rotation,
        "sim3_translation": translation,
    }
