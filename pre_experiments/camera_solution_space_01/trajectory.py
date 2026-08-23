"""Gauge-fixed eight-frame w2c trajectory geometry."""

from __future__ import annotations

import numpy as np
import torch

from .se3 import compose, geodesic_interpolate, inverse, se3_exp, se3_log


FRAME_COUNT = 8
TRAJECTORY_DIMENSION = 42
TRANSLATION_SCALE = 0.10
ROTATION_SCALE = np.deg2rad(5.0)
FAR_DISTANCE = 1.0
DEDUP_DISTANCE = 0.10


def _is_torch(value) -> bool:
    return isinstance(value, torch.Tensor)


def _require_eight_transforms(value, name: str):
    if not isinstance(value, (np.ndarray, torch.Tensor)):
        raise TypeError(f"{name} must be a NumPy array or torch tensor")
    if tuple(value.shape) != (FRAME_COUNT, 4, 4):
        raise ValueError(f"{name} must contain exactly eight transforms with shape (8, 4, 4)")
    se3_log(value)
    return value


def _identity(like):
    if _is_torch(like):
        return torch.eye(4, dtype=torch.float64, device=like.device)
    return np.eye(4, dtype=np.float64)


def _require_identity_anchor(trajectory, name: str):
    identity = _identity(trajectory)
    if _is_torch(trajectory):
        valid = torch.allclose(trajectory[0], identity, atol=1e-10, rtol=0.0)
    else:
        valid = np.allclose(trajectory[0], identity, atol=1e-10, rtol=0.0)
    if not valid:
        raise ValueError(f"{name} first frame/anchor must be identity")


def validate_trajectory(trajectory):
    """Validate and return one float64 gauge-fixed ``(8,4,4)`` trajectory."""
    _require_eight_transforms(trajectory, "trajectory")
    _require_identity_anchor(trajectory, "trajectory")
    return trajectory


def gauge_fix_w2c(world_to_camera):
    """Return ``P_i = W_i @ inv(W_0)`` for exactly eight OpenCV w2c poses."""
    _require_eight_transforms(world_to_camera, "world-to-camera poses")
    fixed = compose(world_to_camera, inverse(world_to_camera[0]))
    identity = _identity(world_to_camera)[None]
    if _is_torch(world_to_camera):
        return torch.cat((identity, fixed[1:]), dim=0)
    return np.concatenate((identity, fixed[1:]), axis=0)


def pack_trajectory(trajectory):
    """Pack frames 1..7 as 42 translation-first logarithm coordinates."""
    validate_trajectory(trajectory)
    return se3_log(trajectory[1:]).reshape(TRAJECTORY_DIMENSION)


def unpack_trajectory(packed):
    """Unpack exactly 42 coordinates and prepend a fixed identity anchor."""
    if not isinstance(packed, (np.ndarray, torch.Tensor)):
        raise TypeError("packed trajectory must be a NumPy array or torch tensor")
    if tuple(packed.shape) != (TRAJECTORY_DIMENSION,):
        raise ValueError("packed trajectory must have shape (42,)")
    transforms = se3_exp(packed.reshape(FRAME_COUNT - 1, 6))
    identity = _identity(packed)[None]
    if _is_torch(packed):
        return torch.cat((identity, transforms), dim=0)
    return np.concatenate((identity, transforms), axis=0)


def _require_matching_backends(first, second):
    if _is_torch(first) != _is_torch(second):
        raise TypeError("trajectories must use the same NumPy or torch backend")
    if _is_torch(first) and first.device != second.device:
        raise ValueError("torch trajectories must use the same device")


def trajectory_distance(first, second):
    """Return the frozen RMS left-difference distance over frames 1..7."""
    validate_trajectory(first)
    validate_trajectory(second)
    _require_matching_backends(first, second)
    if _is_torch(first):
        if torch.equal(first, second):
            return (first.sum() + second.sum()) * 0.0
    elif np.array_equal(first, second):
        return np.float64(0.0)
    difference = compose(second[1:], inverse(first[1:]))
    twist = se3_log(difference)
    if _is_torch(twist):
        translation2 = (twist[..., :3] ** 2).sum(dim=-1)
        rotation2 = (twist[..., 3:] ** 2).sum(dim=-1)
        return torch.sqrt((translation2 / TRANSLATION_SCALE**2 + rotation2 / ROTATION_SCALE**2).mean())
    translation2 = (twist[..., :3] ** 2).sum(axis=-1)
    rotation2 = (twist[..., 3:] ** 2).sum(axis=-1)
    return np.sqrt(np.mean(translation2 / TRANSLATION_SCALE**2 + rotation2 / ROTATION_SCALE**2))


def product_geodesic(first, second, t):
    """Interpolate each non-anchor frame on the frozen left SE(3) geodesic."""
    validate_trajectory(first)
    validate_trajectory(second)
    _require_matching_backends(first, second)
    frames = geodesic_interpolate(first[1:], second[1:], t)
    identity = _identity(first)[None]
    if _is_torch(first):
        return torch.cat((identity, frames), dim=0)
    return np.concatenate((identity, frames), axis=0)


__all__ = [
    "FAR_DISTANCE",
    "DEDUP_DISTANCE",
    "validate_trajectory",
    "gauge_fix_w2c",
    "pack_trajectory",
    "unpack_trajectory",
    "trajectory_distance",
    "product_geodesic",
]
