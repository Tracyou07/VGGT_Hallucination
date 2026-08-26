"""Independent sparse RGB-D observation energy for CVA02 candidate paths."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import numpy as np

from pre_experiments.camera_velocity_ambiguity_02.contracts import canonical_json_digest
from pre_experiments.camera_velocity_ambiguity_02.interpolation import TranslationCandidate


@dataclass(frozen=True)
class RGBDConfig:
    pixel_stride: int = 8
    pixel_margin: int = 8
    min_correspondences: int = 64
    photometric_weight: float = 1.0
    depth_weight: float = 1.0
    free_space_weight: float = 1.0
    occlusion_weight: float = 0.5
    coverage_weight: float = 1.0
    depth_tolerance: float = 0.02
    flat_energy_tolerance: float = 1e-5
    scale_candidates: tuple[float, ...] = (
        0.25,
        0.3535533905932738,
        0.5,
        0.7071067811865476,
        1.0,
        1.4142135623730951,
        2.0,
        2.8284271247461903,
        4.0,
    )

    def __post_init__(self) -> None:
        if self.pixel_stride < 1 or self.pixel_margin < 0 or self.min_correspondences < 1:
            raise ValueError("RGB-D grid and correspondence settings are invalid")
        numeric = (
            self.photometric_weight,
            self.depth_weight,
            self.free_space_weight,
            self.occlusion_weight,
            self.coverage_weight,
            self.depth_tolerance,
            self.flat_energy_tolerance,
            *self.scale_candidates,
        )
        if not np.isfinite(numeric).all() or any(value < 0 for value in numeric):
            raise ValueError("RGB-D weights, tolerances, and scales must be finite and non-negative")
        if len(self.scale_candidates) < 3 or any(
            right <= left for left, right in zip(self.scale_candidates, self.scale_candidates[1:])
        ):
            raise ValueError("scale_candidates must contain at least three increasing values")
        if self.scale_candidates[0] <= 0:
            raise ValueError("scale candidates must be positive")


@dataclass(frozen=True)
class RGBDObservations:
    frame_ids: np.ndarray
    rgb: np.ndarray
    depth: np.ndarray
    intrinsics: np.ndarray


@dataclass(frozen=True)
class RGBDEnergy:
    valid: bool
    reason: str | None
    total: float
    photometric: float
    depth: float
    free_space: float
    occlusion: float
    coverage: float
    correspondence_count: int
    attempted_count: int
    direction_count: int


@dataclass(frozen=True)
class FrozenObservationScale:
    scale: float
    valid: bool
    reason: str | None
    candidate_scales: tuple[float, ...]
    candidate_energies: tuple[float, ...]
    digest: str


@dataclass(frozen=True)
class RGBDPathResult:
    valid: bool
    reason: str | None
    alphas: tuple[float, ...]
    energies: tuple[float, ...]
    correspondence_counts: tuple[int, ...]
    interior_barrier: float
    scale_digest: str


def build_rgbd_observations(payload: Mapping[str, object]) -> RGBDObservations:
    """Accept only the independent observation fields; reject GT and plot leakage."""
    expected = {"frame_ids", "rgb", "depth", "intrinsics"}
    if set(payload) != expected:
        raise ValueError(f"RGB-D observation payload must contain exactly {sorted(expected)}")
    frame_ids = np.asarray(payload["frame_ids"])
    rgb = np.asarray(payload["rgb"], dtype=np.float64)
    depth = np.asarray(payload["depth"], dtype=np.float64)
    intrinsics = np.asarray(payload["intrinsics"], dtype=np.float64)
    if frame_ids.ndim != 1 or len(frame_ids) < 2:
        raise ValueError("RGB-D frame_ids must contain at least two entries")
    integer_ids = frame_ids.astype(np.int64, copy=False)
    if not np.array_equal(frame_ids, integer_ids) or any(
        left >= right for left, right in zip(integer_ids, integer_ids[1:])
    ):
        raise ValueError("RGB-D frame_ids must be strictly increasing integers")
    if rgb.ndim != 4 or rgb.shape[-1] != 3:
        raise ValueError("rgb must have shape [frames, height, width, 3]")
    if depth.shape != rgb.shape[:3]:
        raise ValueError("depth must match RGB frame and image dimensions")
    if intrinsics.shape == (3, 3):
        intrinsics = np.repeat(intrinsics[None], len(frame_ids), axis=0)
    if intrinsics.shape != (len(frame_ids), 3, 3) or len(rgb) != len(frame_ids):
        raise ValueError("intrinsics and RGB must match frame count")
    if not all(np.isfinite(value).all() for value in (rgb, depth, intrinsics)):
        raise ValueError("RGB-D observations must contain only finite values")
    if np.any(rgb < 0) or np.any(rgb > 1) or np.any(depth < 0):
        raise ValueError("RGB must be in [0,1] and depth must be non-negative")
    if np.any(np.abs(np.linalg.det(intrinsics)) <= 1e-12):
        raise ValueError("intrinsics must be invertible")
    return RGBDObservations(
        frame_ids=integer_ids.copy(),
        rgb=rgb.copy(),
        depth=depth.copy(),
        intrinsics=intrinsics.copy(),
    )


def _candidate_poses(c2w: np.ndarray, count: int) -> np.ndarray:
    poses = np.asarray(c2w, dtype=np.float64)
    if poses.shape != (count, 4, 4) or not np.isfinite(poses).all():
        raise ValueError("candidate c2w must have finite shape [frames, 4, 4]")
    if not np.allclose(poses[:, 3, :], [0.0, 0.0, 0.0, 1.0], atol=1e-10, rtol=0):
        raise ValueError("candidate c2w must contain homogeneous poses")
    return poses


def _direction_penalties(
    observations: RGBDObservations,
    poses: np.ndarray,
    source_index: int,
    target_index: int,
    config: RGBDConfig,
) -> tuple[float, float, float, float, int, int]:
    height, width = observations.depth.shape[1:]
    ys = np.arange(config.pixel_margin, height - config.pixel_margin, config.pixel_stride)
    xs = np.arange(config.pixel_margin, width - config.pixel_margin, config.pixel_stride)
    if len(xs) == 0 or len(ys) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0, 0
    grid_x, grid_y = np.meshgrid(xs, ys)
    x = grid_x.ravel().astype(np.int64)
    y = grid_y.ravel().astype(np.int64)
    source_depth = observations.depth[source_index, y, x]
    source_valid = source_depth > 0
    x = x[source_valid]
    y = y[source_valid]
    source_depth = source_depth[source_valid]
    attempted = len(source_depth)
    if attempted == 0:
        return 0.0, 0.0, 0.0, 0.0, 0, 0

    pixels = np.stack((x, y, np.ones_like(x)), axis=1).astype(np.float64)
    camera_points = (
        (np.linalg.inv(observations.intrinsics[source_index]) @ pixels.T).T
        * source_depth[:, None]
    )
    source_pose = poses[source_index]
    target_pose = poses[target_index]
    world = camera_points @ source_pose[:3, :3].T + source_pose[:3, 3]
    target_camera = (world - target_pose[:3, 3]) @ target_pose[:3, :3]
    in_front = target_camera[:, 2] > 1e-12
    projected = target_camera @ observations.intrinsics[target_index].T
    projected_xy = projected[:, :2] / np.maximum(projected[:, 2:3], 1e-12)
    target_x = np.rint(projected_xy[:, 0]).astype(np.int64)
    target_y = np.rint(projected_xy[:, 1]).astype(np.int64)
    inside = (
        in_front
        & (target_x >= 0)
        & (target_x < width)
        & (target_y >= 0)
        & (target_y < height)
    )
    indices = np.flatnonzero(inside)
    if len(indices) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0, attempted
    observed_depth = observations.depth[
        target_index, target_y[indices], target_x[indices]
    ]
    valid_depth = observed_depth > 0
    indices = indices[valid_depth]
    observed_depth = observed_depth[valid_depth]
    correspondence = len(indices)
    if correspondence == 0:
        return 0.0, 0.0, 0.0, 0.0, 0, attempted

    source_color = observations.rgb[source_index, y[indices], x[indices]]
    target_color = observations.rgb[
        target_index, target_y[indices], target_x[indices]
    ]
    predicted_depth = target_camera[indices, 2]
    normalized_gap = (predicted_depth - observed_depth) / np.maximum(observed_depth, 1e-12)
    photometric = float(np.sum(np.mean(np.abs(source_color - target_color), axis=1)))
    depth_penalty = float(np.sum(np.abs(normalized_gap)))
    tolerance = config.depth_tolerance
    free_space = float(np.sum(np.maximum(-normalized_gap - tolerance, 0.0)))
    occlusion = float(np.sum(np.maximum(normalized_gap - tolerance, 0.0)))
    return photometric, depth_penalty, free_space, occlusion, correspondence, attempted


def compute_rgbd_energy(
    observations: RGBDObservations,
    candidate_c2w: np.ndarray,
    config: RGBDConfig = RGBDConfig(),
) -> RGBDEnergy:
    """Compute deterministic bidirectional adjacent-frame observation energy."""
    poses = _candidate_poses(candidate_c2w, len(observations.frame_ids))
    totals = np.zeros(4, dtype=np.float64)
    correspondences = 0
    attempted = 0
    directions = 0
    for left in range(len(poses) - 1):
        right = left + 1
        for source, target in ((left, right), (right, left)):
            values = _direction_penalties(
                observations, poses, source, target, config
            )
            totals += values[:4]
            correspondences += values[4]
            attempted += values[5]
            directions += 1
    denominator = max(correspondences, 1)
    photometric, depth_penalty, free_space, occlusion = (
        float(value / denominator) for value in totals
    )
    coverage = float(1.0 - correspondences / max(attempted, 1))
    total = (
        config.photometric_weight * photometric
        + config.depth_weight * depth_penalty
        + config.free_space_weight * free_space
        + config.occlusion_weight * occlusion
        + config.coverage_weight * coverage
    )
    valid = correspondences >= config.min_correspondences
    return RGBDEnergy(
        valid=valid,
        reason=None if valid else "insufficient_correspondence",
        total=float(total),
        photometric=photometric,
        depth=depth_penalty,
        free_space=free_space,
        occlusion=occlusion,
        coverage=coverage,
        correspondence_count=correspondences,
        attempted_count=attempted,
        direction_count=directions,
    )


def _scale_translations(c2w: np.ndarray, scale: float) -> np.ndarray:
    poses = np.asarray(c2w, dtype=np.float64).copy()
    origin = poses[0, :3, 3].copy()
    poses[:, :3, 3] = origin + scale * (poses[:, :3, 3] - origin)
    return poses


def freeze_observation_scale(
    observations: RGBDObservations,
    global_c2w: np.ndarray,
    config: RGBDConfig = RGBDConfig(),
) -> FrozenObservationScale:
    """Select scene scale once from the global observation energy grid."""
    poses = _candidate_poses(global_c2w, len(observations.frame_ids))
    scales = tuple(float(value) for value in config.scale_candidates)
    results = tuple(
        compute_rgbd_energy(observations, _scale_translations(poses, scale), config)
        for scale in scales
    )
    energies = tuple(result.total for result in results)
    index = int(np.argmin(energies))
    reason = None
    valid = results[index].valid
    if not valid:
        reason = results[index].reason
    elif index in {0, len(scales) - 1}:
        valid = False
        reason = "scale_at_search_boundary"
    payload = {
        "scale": scales[index],
        "valid": valid,
        "reason": reason,
        "candidate_scales": scales,
        "candidate_energies": energies,
        "config": asdict(config),
    }
    return FrozenObservationScale(
        scale=scales[index],
        valid=valid,
        reason=reason,
        candidate_scales=scales,
        candidate_energies=energies,
        digest=canonical_json_digest(payload),
    )


def evaluate_rgbd_path(
    observations: RGBDObservations,
    candidates: Sequence[TranslationCandidate],
    frozen_scale: FrozenObservationScale,
    config: RGBDConfig = RGBDConfig(),
) -> RGBDPathResult:
    """Evaluate an alpha path using one already-frozen observation scale."""
    if len(candidates) < 3:
        raise ValueError("RGB-D path requires endpoints and at least one interior alpha")
    alphas = tuple(float(candidate.alpha) for candidate in candidates)
    if alphas[0] != 0.0 or alphas[-1] != 1.0 or any(
        right <= left for left, right in zip(alphas, alphas[1:])
    ):
        raise ValueError("RGB-D candidate alphas must increase exactly from 0 to 1")
    if not frozen_scale.valid:
        return RGBDPathResult(
            valid=False,
            reason=frozen_scale.reason or "invalid_frozen_scale",
            alphas=alphas,
            energies=tuple(),
            correspondence_counts=tuple(),
            interior_barrier=0.0,
            scale_digest=frozen_scale.digest,
        )
    results = tuple(
        compute_rgbd_energy(
            observations,
            _scale_translations(candidate.c2w, frozen_scale.scale),
            config,
        )
        for candidate in candidates
    )
    energies = tuple(result.total for result in results)
    counts = tuple(result.correspondence_count for result in results)
    reason = None
    valid = all(result.valid for result in results)
    if not valid:
        reason = "insufficient_correspondence"
    elif max(energies) - min(energies) <= config.flat_energy_tolerance:
        valid = False
        reason = "flat_energy_curve"
    endpoint_ceiling = max(energies[0], energies[-1])
    barrier = max(energies[1:-1]) - endpoint_ceiling
    return RGBDPathResult(
        valid=valid,
        reason=reason,
        alphas=alphas,
        energies=energies,
        correspondence_counts=counts,
        interior_barrier=float(barrier),
        scale_digest=frozen_scale.digest,
    )
