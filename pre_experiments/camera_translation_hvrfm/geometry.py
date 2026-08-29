"""Strict translation-only endpoint geometry in the raw long-prediction gauge."""

from __future__ import annotations

from numbers import Real

import numpy as np
import torch

from pre_experiments.variational_camera_latent.camera import pose_encoding_to_c2w


_FRAMES = 500
_ENDPOINTS = 4
_RAW_SO3_ATOL = 2e-6
_HOMOGENEOUS_ATOL = 1e-10
_POSE_REPLAY_NORMALIZED_ATOL = 5e-6


def _exact_array(
    value: np.ndarray,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: np.dtype[object] | type[np.generic],
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise ValueError(f"{name} must be a NumPy array")
    expected_dtype = np.dtype(dtype)
    if value.shape != shape or value.dtype != expected_dtype:
        raise ValueError(
            f"{name} must have exact shape {shape} and dtype {expected_dtype}"
        )
    return value


def _validate_so3(rotations: np.ndarray, *, name: str, atol: float) -> None:
    gram = np.einsum("...ji,...jk->...ik", rotations, rotations)
    determinants = np.linalg.det(rotations)
    if (
        not np.isfinite(rotations).all()
        or not np.allclose(gram, np.eye(3), atol=atol, rtol=0.0)
        or not np.allclose(determinants, 1.0, atol=atol, rtol=0.0)
        or np.any(determinants <= 0.0)
    ):
        raise ValueError(f"{name} must contain proper SO(3) rotations")


def _validate_baseline_c2w(baseline_c2w: np.ndarray) -> np.ndarray:
    poses = _exact_array(
        baseline_c2w,
        name="baseline_c2w",
        shape=(_FRAMES, 4, 4),
        dtype=np.float64,
    )
    if not np.isfinite(poses).all():
        raise ValueError("baseline_c2w must contain only finite values")
    if not np.allclose(
        poses[:, 3, :],
        np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
        atol=_HOMOGENEOUS_ATOL,
        rtol=0.0,
    ):
        raise ValueError("baseline_c2w must contain homogeneous poses")
    _validate_so3(poses[:, :3, :3], name="baseline_c2w", atol=_RAW_SO3_ATOL)
    return poses


def _validate_frame_ids(frame_ids: np.ndarray, *, name: str) -> np.ndarray:
    values = _exact_array(
        frame_ids, name=name, shape=(_FRAMES,), dtype=np.int64
    )
    if np.any(values[1:] <= values[:-1]):
        raise ValueError(f"{name} must be strictly increasing and unique")
    return values


def _center_rms_scale(poses: np.ndarray) -> float:
    centers = poses[:, :3, 3]
    centered = centers - centers.mean(axis=0)
    scale = float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))
    if not np.isfinite(scale) or scale <= 1e-12:
        raise ValueError("baseline_c2w must have sufficient finite prediction scale")
    return scale


def _maximum_normalized_vector_error(
    actual: np.ndarray, expected: np.ndarray, *, scale: float
) -> float:
    errors = np.linalg.norm(actual - expected, axis=-1)
    return float(np.max(errors) / scale)


def _validate_coverage_mask(coverage_mask: np.ndarray) -> np.ndarray:
    if not isinstance(coverage_mask, np.ndarray):
        raise ValueError("coverage_mask must be a NumPy array")
    if coverage_mask.shape != (_ENDPOINTS, _FRAMES) or coverage_mask.dtype not in (
        np.dtype(np.bool_),
        np.dtype(np.uint8),
    ):
        raise ValueError("coverage_mask must be bool/uint8 with shape [4, 500]")
    if coverage_mask.dtype == np.dtype(np.uint8) and not np.isin(
        coverage_mask, (0, 1)
    ).all():
        raise ValueError("coverage_mask must be binary")
    return coverage_mask.astype(np.bool_, copy=False)


def _validate_pose_encoding_matches_c2w(
    baseline_pose_encoding: np.ndarray, baseline_c2w: np.ndarray
) -> np.ndarray:
    pose = _exact_array(
        baseline_pose_encoding,
        name="baseline_pose_encoding",
        shape=(_FRAMES, 9),
        dtype=np.float32,
    )
    if not np.isfinite(pose).all():
        raise ValueError("baseline_pose_encoding must contain only finite values")

    rotations_w2c = np.swapaxes(baseline_c2w[:, :3, :3], -1, -2)
    scale = _center_rms_scale(baseline_c2w)
    expected_translation = -np.einsum(
        "tij,tj->ti", rotations_w2c, baseline_c2w[:, :3, 3]
    )
    translation_error = _maximum_normalized_vector_error(
        pose[:, :3].astype(np.float64), expected_translation, scale=scale
    )
    if (
        not np.isfinite(translation_error)
        or translation_error > _POSE_REPLAY_NORMALIZED_ATOL
    ):
        raise ValueError("baseline pose translation does not match baseline_c2w")

    try:
        with torch.no_grad():
            replay = (
                pose_encoding_to_c2w(torch.from_numpy(pose[None]))[0]
                .to(dtype=torch.float64, device="cpu")
                .numpy()
            )
    except (RuntimeError, ValueError) as error:
        raise ValueError("baseline pose encoding cannot be replayed") from error
    if not np.allclose(
        replay[:, :3, :3],
        baseline_c2w[:, :3, :3],
        atol=_RAW_SO3_ATOL,
        rtol=0.0,
    ):
        raise ValueError("baseline pose quaternion does not match baseline_c2w")
    center_error = _maximum_normalized_vector_error(
        replay[:, :3, 3], baseline_c2w[:, :3, 3], scale=scale
    )
    if not np.isfinite(center_error) or center_error > _POSE_REPLAY_NORMALIZED_ATOL:
        raise ValueError("baseline pose replay does not match baseline_c2w")
    return pose


def prediction_scale(baseline_c2w: np.ndarray) -> float:
    """Return prediction-only RMS camera-center scale from one 500-frame C2W."""
    poses = _validate_baseline_c2w(baseline_c2w)
    return _center_rms_scale(poses)


def baseline_fill_teacher_centers(
    *,
    long_frame_ids: np.ndarray,
    teacher_frame_ids: np.ndarray,
    baseline_c2w: np.ndarray,
    teacher_centers: np.ndarray,
    coverage_mask: np.ndarray,
) -> np.ndarray:
    """Fill uncovered all-NaN teachers with the matching raw long centers."""
    long_ids = _validate_frame_ids(long_frame_ids, name="long_frame_ids")
    teacher_ids = _validate_frame_ids(teacher_frame_ids, name="teacher_frame_ids")
    if not np.array_equal(long_ids, teacher_ids):
        raise ValueError("long and teacher frame IDs must match exactly")
    poses = _validate_baseline_c2w(baseline_c2w)
    teachers = _exact_array(
        teacher_centers,
        name="teacher_centers",
        shape=(_ENDPOINTS, _FRAMES, 3),
        dtype=np.float64,
    )
    covered = _validate_coverage_mask(coverage_mask)
    if np.isinf(teachers).any():
        raise ValueError("teacher_centers may not contain infinity")
    if np.any(covered) and not np.isfinite(teachers[covered]).all():
        raise ValueError("covered teacher centers must be finite")
    if np.any(~covered):
        uncovered_all_nan = np.all(np.isnan(teachers[~covered]), axis=-1)
        if not np.all(uncovered_all_nan):
            raise ValueError("each uncovered teacher center must be all-NaN")

    baseline_centers = poses[:, :3, 3]
    filled = np.broadcast_to(
        baseline_centers[None, :, :], (_ENDPOINTS, _FRAMES, 3)
    ).copy()
    filled[covered] = teachers[covered]
    if not np.isfinite(filled).all():
        raise ValueError("baseline-filled teacher centers must be finite")
    return filled


def build_translation_endpoint(
    *,
    long_frame_ids: np.ndarray,
    teacher_frame_ids: np.ndarray,
    baseline_c2w: np.ndarray,
    baseline_pose_encoding: np.ndarray,
    teacher_centers: np.ndarray,
    coverage_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Build four normalized translation endpoints without evaluating masked NaNs."""
    poses = _validate_baseline_c2w(baseline_c2w)
    _validate_pose_encoding_matches_c2w(baseline_pose_encoding, poses)
    filled = baseline_fill_teacher_centers(
        long_frame_ids=long_frame_ids,
        teacher_frame_ids=teacher_frame_ids,
        baseline_c2w=poses,
        teacher_centers=teacher_centers,
        coverage_mask=coverage_mask,
    )
    covered = _validate_coverage_mask(coverage_mask)
    scale = prediction_scale(poses)

    endpoint = np.zeros((_ENDPOINTS, _FRAMES, 3), dtype=np.float32)
    endpoint_indices, frame_indices = np.nonzero(covered)
    if len(frame_indices):
        rotations_w2c = np.swapaxes(poses[:, :3, :3], -1, -2)
        deltas = (
            teacher_centers[endpoint_indices, frame_indices]
            - poses[frame_indices, :3, 3]
        )
        covered_values = -np.einsum(
            "nij,nj->ni", rotations_w2c[frame_indices], deltas
        ) / scale
        with np.errstate(over="ignore", invalid="ignore"):
            covered_float32 = covered_values.astype(np.float32)
        if not np.isfinite(covered_float32).all():
            raise ValueError("translation endpoint overflows float32")
        endpoint[endpoint_indices, frame_indices] = covered_float32
    if not np.all(endpoint.view(np.uint32)[~covered] == 0):
        raise AssertionError("uncovered endpoint construction lost bitwise zero")
    return endpoint, filled, scale


def apply_translation_endpoint(
    baseline_pose_encoding: np.ndarray,
    translation_endpoints: np.ndarray,
    *,
    scale: float,
) -> np.ndarray:
    """Apply endpoint translation while copying quaternion/FOV bytes unchanged."""
    baseline = _exact_array(
        baseline_pose_encoding,
        name="baseline_pose_encoding",
        shape=(_FRAMES, 9),
        dtype=np.float32,
    )
    endpoints = _exact_array(
        translation_endpoints,
        name="translation_endpoints",
        shape=(_ENDPOINTS, _FRAMES, 3),
        dtype=np.float32,
    )
    if not np.isfinite(baseline).all() or not np.isfinite(endpoints).all():
        raise ValueError("pose encoding and endpoints must be finite")
    if isinstance(scale, (bool, np.bool_)) or not isinstance(scale, Real):
        raise ValueError("scale must be a finite positive real scalar")
    scale_value = float(scale)
    if not np.isfinite(scale_value) or scale_value <= 0.0:
        raise ValueError("scale must be a finite positive real scalar")

    corrected = np.broadcast_to(
        baseline[None, :, :], (_ENDPOINTS, _FRAMES, 9)
    ).copy()
    active = endpoints != np.float32(0.0)
    if not np.any(active):
        return corrected
    baseline_translations = np.broadcast_to(
        baseline[None, :, :3], endpoints.shape
    )
    translations = (
        baseline_translations[active].astype(np.float64)
        + scale_value * endpoints[active].astype(np.float64)
    )
    with np.errstate(over="ignore", invalid="ignore"):
        translated_float32 = translations.astype(np.float32)
    if not np.isfinite(translated_float32).all():
        raise ValueError("corrected translation overflows float32")
    corrected[..., :3][active] = translated_float32
    return corrected
