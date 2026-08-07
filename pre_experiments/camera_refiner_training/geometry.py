"""Prediction-gauge geometry and overlap fusion for camera-center refinement."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np



def umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Estimate an orientation-preserving Sim(3) mapping src to dst."""
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
    variance = float(np.mean(np.sum(source_centered * source_centered, axis=1)))
    if variance <= 1e-12:
        raise ValueError("src trajectory has insufficient translation variance")
    covariance = (target_centered.T @ source_centered) / len(source)
    left, singular_values, right_transpose = np.linalg.svd(covariance)
    signs = np.ones(3, dtype=np.float64)
    if np.linalg.det(left @ right_transpose) < 0:
        signs[-1] = -1.0
    rotation = left @ np.diag(signs) @ right_transpose
    scale = float(np.sum(singular_values * signs) / variance)
    translation = target_mean - scale * (rotation @ source_mean)
    return scale, rotation, translation


def _centers(name: str, value: np.ndarray, *, minimum: int = 2) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3 or len(array) < minimum:
        raise ValueError(f"{name} must have shape [S, 3] with S >= {minimum}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite values")
    return array


def _poses(name: str, value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 3 or array.shape[1:] != (4, 4) or len(array) < 2:
        raise ValueError(f"{name} must have shape [S, 4, 4]")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite values")
    return array


@dataclass(frozen=True)
class SceneGauge:
    """Scene-wide canonical coordinates derived only from global prediction."""

    origin: np.ndarray
    basis: np.ndarray
    scale: float

    @classmethod
    def from_c2w(cls, global_c2w: np.ndarray) -> "SceneGauge":
        poses = _poses("global_c2w", global_c2w)
        centers = poses[:, :3, 3]
        origin = centers[0].copy()
        basis = poses[0, :3, :3].copy()
        centered = centers - origin
        scale = float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))
        if not np.isfinite(scale) or scale <= 1e-8:
            raise ValueError("global trajectory has insufficient translation variance")
        return cls(origin=origin, basis=basis, scale=scale)

    def canonicalize(self, centers: np.ndarray) -> np.ndarray:
        values = _centers("centers", centers)
        return ((values - self.origin) @ self.basis) / self.scale

    def restore(self, canonical_centers: np.ndarray) -> np.ndarray:
        values = _centers("canonical_centers", canonical_centers)
        return (values * self.scale) @ self.basis.T + self.origin


def align_centers_to_reference(
    moving_centers: np.ndarray,
    reference_centers: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Map moving centers to reference using an orientation-preserving Sim(3)."""
    moving = _centers("moving_centers", moving_centers)
    reference = _centers("reference_centers", reference_centers)
    if moving.shape != reference.shape:
        raise ValueError("moving and reference centers must have matching shape")
    scale, rotation, translation = umeyama(moving, reference)
    aligned = scale * (moving @ rotation.T) + translation
    residual = np.linalg.norm(aligned - reference, axis=1)
    return aligned, residual


def boundary_taper(window_length: int) -> np.ndarray:
    if window_length < 2:
        raise ValueError("window_length must be at least two")
    indices = np.arange(window_length)
    distance = np.minimum(indices + 1, window_length - indices).astype(np.float64)
    return distance / distance.max()


def fuse_window_corrections(
    corrections: np.ndarray,
    confidence: np.ndarray,
    *,
    starts: np.ndarray,
    total_frames: int,
    alignment_residual: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fuse canonical corrections from overlapping windows."""
    values = np.asarray(corrections, dtype=np.float64)
    gates = np.asarray(confidence, dtype=np.float64)
    offsets = np.asarray(starts, dtype=np.int64)
    if values.ndim != 3 or values.shape[2] != 3:
        raise ValueError("corrections must have shape [N, W, 3]")
    if gates.shape != values.shape[:2] or offsets.shape != (len(values),):
        raise ValueError("confidence and starts do not match corrections")
    if total_frames < values.shape[1] or np.any(offsets < 0):
        raise ValueError("invalid total_frames or window starts")
    if np.any(offsets + values.shape[1] > total_frames):
        raise ValueError("a window exceeds total_frames")
    if not np.isfinite(values).all() or not np.isfinite(gates).all():
        raise ValueError("corrections and confidence must be finite")
    gates = np.clip(gates, 0.0, 1.0)
    reliability = np.ones_like(gates)
    if alignment_residual is not None:
        residual = np.asarray(alignment_residual, dtype=np.float64)
        if residual.shape != gates.shape or np.any(residual < 0) or not np.isfinite(residual).all():
            raise ValueError("alignment_residual must be finite non-negative [N, W]")
        reliability = 1.0 / (1.0 + residual)
    taper = boundary_taper(values.shape[1])[None]
    weights = gates * reliability * taper
    sums = np.zeros((total_frames, 3), dtype=np.float64)
    weight_sums = np.zeros(total_frames, dtype=np.float64)
    confidence_sums = np.zeros(total_frames, dtype=np.float64)
    taper_sums = np.zeros(total_frames, dtype=np.float64)
    for index, start in enumerate(offsets):
        target = slice(int(start), int(start) + values.shape[1])
        sums[target] += values[index] * weights[index, :, None]
        weight_sums[target] += weights[index]
        confidence_sums[target] += gates[index] * taper[0]
        taper_sums[target] += taper[0]
    fused = np.zeros_like(sums)
    valid = weight_sums > 0
    fused[valid] = sums[valid] / weight_sums[valid, None]
    fused_confidence = np.zeros(total_frames, dtype=np.float64)
    covered = taper_sums > 0
    fused_confidence[covered] = confidence_sums[covered] / taper_sums[covered]
    return fused, fused_confidence


def apply_center_corrections(
    global_c2w: np.ndarray,
    center_corrections: np.ndarray,
) -> np.ndarray:
    """Add center corrections while copying every non-translation pose value."""
    poses = _poses("global_c2w", global_c2w)
    corrections = _centers("center_corrections", center_corrections)
    if len(corrections) != len(poses):
        raise ValueError("center corrections must match global poses")
    corrected = poses.copy()
    corrected[:, :3, 3] += corrections
    return corrected
