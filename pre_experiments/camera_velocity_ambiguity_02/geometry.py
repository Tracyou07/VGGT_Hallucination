"""Prediction-only local/global geometry and signed repair directions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pre_experiments.camera_velocity_ambiguity_02.units import OverlapUnit
from pre_experiments.common.pose_metrics import umeyama


@dataclass(frozen=True)
class AlignmentLimits:
    """Fail-closed numerical limits for prediction-to-prediction Sim(3)."""

    min_rank: int = 2
    max_condition: float = 1e6
    min_scale: float = 1e-4
    max_scale: float = 1e4
    max_normalized_rms: float = 0.5
    rank_relative_tolerance: float = 1e-12

    def __post_init__(self) -> None:
        if self.min_rank not in {1, 2, 3}:
            raise ValueError("min_rank must be 1, 2, or 3")
        numeric = (
            self.max_condition,
            self.min_scale,
            self.max_scale,
            self.max_normalized_rms,
            self.rank_relative_tolerance,
        )
        if not np.isfinite(numeric).all() or any(value <= 0 for value in numeric):
            raise ValueError("alignment limits must be finite and positive")
        if self.min_scale >= self.max_scale:
            raise ValueError("min_scale must be smaller than max_scale")


@dataclass(frozen=True)
class AlignmentDiagnostics:
    """A fitted alignment plus the reason it is or is not scientifically usable."""

    valid: bool
    exclusion_reason: str | None
    fit_count: int
    rank: int
    condition: float
    scale: float | None
    rotation_determinant: float | None
    rms: float | None
    normalized_rms: float | None
    rotation: np.ndarray | None
    translation: np.ndarray | None
    aligned_c2w: np.ndarray | None


@dataclass(frozen=True)
class ResidualDirectionMetrics:
    """Signed left/right repair vectors and scale-normalized comparisons."""

    left_residual: np.ndarray
    right_residual: np.ndarray
    left_normalized_rms: float
    right_normalized_rms: float
    normalized_rms_separation: float
    direction_evaluable: bool
    flattened_cosine: float | None
    per_frame_cosine: np.ndarray
    per_frame_direction_agreement: float | None


@dataclass(frozen=True)
class PairGeometry:
    """Independent full-window alignments and shared-frame repair directions."""

    pair_id: str
    shared_frame_ids: tuple[int, ...]
    left_alignment: AlignmentDiagnostics
    right_alignment: AlignmentDiagnostics
    metrics: ResidualDirectionMetrics | None


def _pose_stack(name: str, value: np.ndarray) -> np.ndarray:
    poses = np.asarray(value, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or len(poses) < 2:
        raise ValueError(f"{name} must have shape [frames, 4, 4] with at least two poses")
    if not np.isfinite(poses).all():
        raise ValueError(f"{name} must contain only finite values")
    expected_bottom = np.asarray([0.0, 0.0, 0.0, 1.0])
    if not np.allclose(poses[:, 3, :], expected_bottom, atol=1e-10, rtol=0):
        raise ValueError(f"{name} must contain homogeneous camera poses")
    return poses


def global_scene_scale(global_c2w: np.ndarray) -> float:
    """Freeze prediction-only scene scale as center RMS about the global centroid."""
    poses = _pose_stack("global_c2w", global_c2w)
    centers = poses[:, :3, 3]
    centered = centers - centers.mean(axis=0)
    scale = float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))
    if not np.isfinite(scale) or scale <= 1e-12:
        raise ValueError("global prediction has insufficient scene scale")
    return scale


def _invalid_alignment(
    *,
    reason: str,
    fit_count: int,
    rank: int,
    condition: float,
    scale: float | None = None,
    rotation_determinant: float | None = None,
    rms: float | None = None,
    normalized_rms: float | None = None,
    rotation: np.ndarray | None = None,
    translation: np.ndarray | None = None,
    aligned_c2w: np.ndarray | None = None,
) -> AlignmentDiagnostics:
    return AlignmentDiagnostics(
        valid=False,
        exclusion_reason=reason,
        fit_count=fit_count,
        rank=rank,
        condition=condition,
        scale=scale,
        rotation_determinant=rotation_determinant,
        rms=rms,
        normalized_rms=normalized_rms,
        rotation=rotation,
        translation=translation,
        aligned_c2w=aligned_c2w,
    )


def align_local_to_global(
    global_segment_c2w: np.ndarray,
    local_c2w: np.ndarray,
    *,
    scene_scale: float,
    limits: AlignmentLimits = AlignmentLimits(),
) -> AlignmentDiagnostics:
    """Fit one local trajectory to its full corresponding global segment."""
    reference = _pose_stack("global_segment_c2w", global_segment_c2w)
    moving = _pose_stack("local_c2w", local_c2w)
    if reference.shape != moving.shape:
        raise ValueError("global and local prediction trajectories must match exactly")
    if not np.isfinite(scene_scale) or scene_scale <= 1e-12:
        raise ValueError("scene_scale must be finite and positive")

    moving_centers = moving[:, :3, 3]
    reference_centers = reference[:, :3, 3]
    centered = moving_centers - moving_centers.mean(axis=0)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    largest = float(singular_values[0])
    threshold = largest * limits.rank_relative_tolerance
    rank = int(np.count_nonzero(singular_values > threshold)) if largest > 0 else 0
    condition = (
        float(largest / singular_values[rank - 1])
        if rank > 0 and singular_values[rank - 1] > 0
        else float("inf")
    )
    if rank < limits.min_rank:
        return _invalid_alignment(
            reason="rank_below_minimum",
            fit_count=len(moving),
            rank=rank,
            condition=condition,
        )
    if condition > limits.max_condition:
        return _invalid_alignment(
            reason="condition_above_maximum",
            fit_count=len(moving),
            rank=rank,
            condition=condition,
        )

    scale, rotation, translation = umeyama(moving_centers, reference_centers)
    determinant = float(np.linalg.det(rotation))
    if (
        not np.isfinite(scale)
        or scale < limits.min_scale
        or scale > limits.max_scale
        or not np.isfinite(determinant)
        or abs(determinant - 1.0) > 1e-8
    ):
        return _invalid_alignment(
            reason="scale_out_of_range" if np.isfinite(determinant) and abs(determinant - 1.0) <= 1e-8 else "rotation_not_proper",
            fit_count=len(moving),
            rank=rank,
            condition=condition,
            scale=float(scale),
            rotation_determinant=determinant,
            rotation=rotation,
            translation=translation,
        )

    aligned = moving.copy()
    aligned[:, :3, :3] = np.einsum("ij,sjk->sik", rotation, moving[:, :3, :3])
    aligned[:, :3, 3] = scale * (moving_centers @ rotation.T) + translation
    residual = aligned[:, :3, 3] - reference_centers
    rms = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
    normalized_rms = rms / float(scene_scale)
    if not np.isfinite(normalized_rms) or normalized_rms > limits.max_normalized_rms:
        return _invalid_alignment(
            reason="normalized_rms_above_maximum",
            fit_count=len(moving),
            rank=rank,
            condition=condition,
            scale=float(scale),
            rotation_determinant=determinant,
            rms=rms,
            normalized_rms=normalized_rms,
            rotation=rotation,
            translation=translation,
            aligned_c2w=aligned,
        )
    return AlignmentDiagnostics(
        valid=True,
        exclusion_reason=None,
        fit_count=len(moving),
        rank=rank,
        condition=condition,
        scale=float(scale),
        rotation_determinant=determinant,
        rms=rms,
        normalized_rms=normalized_rms,
        rotation=rotation,
        translation=translation,
        aligned_c2w=aligned,
    )


def compute_residual_direction_metrics(
    global_shared_centers: np.ndarray,
    left_aligned_shared_centers: np.ndarray,
    right_aligned_shared_centers: np.ndarray,
    *,
    scene_scale: float,
) -> ResidualDirectionMetrics:
    """Compare signed left/right repair vectors without accessing ground truth."""
    global_centers = np.asarray(global_shared_centers, dtype=np.float64)
    left_centers = np.asarray(left_aligned_shared_centers, dtype=np.float64)
    right_centers = np.asarray(right_aligned_shared_centers, dtype=np.float64)
    if (
        global_centers.shape != left_centers.shape
        or global_centers.shape != right_centers.shape
        or global_centers.ndim != 2
        or global_centers.shape[1] != 3
        or len(global_centers) < 2
    ):
        raise ValueError("shared centers must have matching shape [frames, 3]")
    if not all(np.isfinite(value).all() for value in (global_centers, left_centers, right_centers)):
        raise ValueError("shared centers must contain only finite values")
    if not np.isfinite(scene_scale) or scene_scale <= 1e-12:
        raise ValueError("scene_scale must be finite and positive")

    left = left_centers - global_centers
    right = right_centers - global_centers
    difference = left - right
    left_rms = float(np.sqrt(np.mean(np.sum(left * left, axis=1)))) / scene_scale
    right_rms = float(np.sqrt(np.mean(np.sum(right * right, axis=1)))) / scene_scale
    separation = float(np.sqrt(np.mean(np.sum(difference * difference, axis=1)))) / scene_scale
    left_flat_norm = float(np.linalg.norm(left))
    right_flat_norm = float(np.linalg.norm(right))
    epsilon = 1e-12 * scene_scale * np.sqrt(len(left))
    evaluable = left_flat_norm > epsilon and right_flat_norm > epsilon
    if evaluable:
        flattened_cosine = float(
            np.clip(np.sum(left * right) / (left_flat_norm * right_flat_norm), -1.0, 1.0)
        )
        left_norm = np.linalg.norm(left, axis=1)
        right_norm = np.linalg.norm(right, axis=1)
        per_frame_valid = (left_norm > 1e-12 * scene_scale) & (
            right_norm > 1e-12 * scene_scale
        )
        per_frame_cosine = np.full(len(left), np.nan, dtype=np.float64)
        per_frame_cosine[per_frame_valid] = np.clip(
            np.sum(left[per_frame_valid] * right[per_frame_valid], axis=1)
            / (left_norm[per_frame_valid] * right_norm[per_frame_valid]),
            -1.0,
            1.0,
        )
        if np.any(per_frame_valid):
            agreement = float(np.mean(per_frame_cosine[per_frame_valid] > 0.0))
        else:
            evaluable = False
            flattened_cosine = None
            agreement = None
    else:
        flattened_cosine = None
        per_frame_cosine = np.full(len(left), np.nan, dtype=np.float64)
        agreement = None
    return ResidualDirectionMetrics(
        left_residual=left,
        right_residual=right,
        left_normalized_rms=left_rms,
        right_normalized_rms=right_rms,
        normalized_rms_separation=separation,
        direction_evaluable=evaluable,
        flattened_cosine=flattened_cosine,
        per_frame_cosine=per_frame_cosine,
        per_frame_direction_agreement=agreement,
    )


def build_pair_geometry(
    unit: OverlapUnit,
    *,
    global_c2w: np.ndarray,
    left_local_c2w: np.ndarray,
    right_local_c2w: np.ndarray,
    scene_scale: float,
    limits: AlignmentLimits = AlignmentLimits(),
) -> PairGeometry:
    """Align both windows independently over all frames, then compare overlap."""
    global_poses = _pose_stack("global_c2w", global_c2w)
    left = _pose_stack("left_local_c2w", left_local_c2w)
    right = _pose_stack("right_local_c2w", right_local_c2w)
    if left.shape[0] != unit.left_stop - unit.left_start:
        raise ValueError("left local prediction does not match its full window")
    if right.shape[0] != unit.right_stop - unit.right_start:
        raise ValueError("right local prediction does not match its full window")
    if unit.left_stop > len(global_poses) or unit.right_stop > len(global_poses):
        raise ValueError("overlap unit lies outside the global prediction")

    left_alignment = align_local_to_global(
        global_poses[unit.left_start : unit.left_stop],
        left,
        scene_scale=scene_scale,
        limits=limits,
    )
    right_alignment = align_local_to_global(
        global_poses[unit.right_start : unit.right_stop],
        right,
        scene_scale=scene_scale,
        limits=limits,
    )
    metrics = None
    if left_alignment.valid and right_alignment.valid:
        assert left_alignment.aligned_c2w is not None
        assert right_alignment.aligned_c2w is not None
        metrics = compute_residual_direction_metrics(
            global_poses[np.asarray(unit.global_shared_indices), :3, 3],
            left_alignment.aligned_c2w[np.asarray(unit.left_shared_indices), :3, 3],
            right_alignment.aligned_c2w[np.asarray(unit.right_shared_indices), :3, 3],
            scene_scale=scene_scale,
        )
    return PairGeometry(
        pair_id=unit.pair_id,
        shared_frame_ids=unit.shared_frame_ids,
        left_alignment=left_alignment,
        right_alignment=right_alignment,
        metrics=metrics,
    )
