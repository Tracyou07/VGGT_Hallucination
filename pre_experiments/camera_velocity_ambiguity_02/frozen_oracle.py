"""One immutable scene-level global-to-GT transform for privileged evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pre_experiments.camera_velocity_ambiguity_02.artifacts import frame_digest
from pre_experiments.camera_velocity_ambiguity_02.contracts import canonical_json_digest
from pre_experiments.common.pose_metrics import umeyama


@dataclass(frozen=True)
class OracleLimits:
    min_rank: int = 2
    max_condition: float = 1e8
    min_scale: float = 1e-4
    max_scale: float = 1e4
    rank_relative_tolerance: float = 1e-15


@dataclass(frozen=True)
class FrozenOracle:
    """Tuple-backed Sim(3) fitted once to the complete global trajectory."""

    scene: str
    frame_digest: str
    fit_count: int
    scale: float
    rotation: tuple[tuple[float, float, float], ...]
    translation: tuple[float, float, float]
    rank: int
    condition: float
    transform_digest: str


@dataclass(frozen=True)
class OracleEvaluation:
    aligned_c2w: np.ndarray
    translation_error: np.ndarray
    mean_translation_error: float
    rms_translation_error: float
    fit_transform_digest: str


def _pose_stack(name: str, value: np.ndarray, *, minimum: int = 2) -> np.ndarray:
    poses = np.asarray(value, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or len(poses) < minimum:
        raise ValueError(f"{name} must have shape [frames, 4, 4]")
    if not np.isfinite(poses).all():
        raise ValueError(f"{name} must contain only finite values")
    if not np.allclose(poses[:, 3, :], [0.0, 0.0, 0.0, 1.0], atol=1e-10, rtol=0):
        raise ValueError(f"{name} must contain homogeneous camera poses")
    return poses


def _expected_count(scene: str) -> int:
    if scene == "scene0150_00":
        return 430
    if not isinstance(scene, str) or len(scene) != len("scene0000_00") or not scene.startswith("scene"):
        raise ValueError("scene must use ScanNet sceneNNNN_NN identity")
    return 500


def _validate_limits(limits: OracleLimits) -> None:
    if limits.min_rank not in {1, 2, 3}:
        raise ValueError("oracle min_rank must be 1, 2, or 3")
    values = (
        limits.max_condition,
        limits.min_scale,
        limits.max_scale,
        limits.rank_relative_tolerance,
    )
    if not np.isfinite(values).all() or any(value <= 0 for value in values):
        raise ValueError("oracle limits must be finite and positive")
    if limits.min_scale >= limits.max_scale:
        raise ValueError("oracle min_scale must be smaller than max_scale")


def fit_frozen_oracle(
    scene: str,
    full_frame_ids: np.ndarray,
    global_prediction_c2w: np.ndarray,
    raw_gt_c2w: np.ndarray,
    *,
    limits: OracleLimits = OracleLimits(),
) -> FrozenOracle:
    """Fit exactly once using the full frozen 500/430-frame scene trajectory."""
    _validate_limits(limits)
    expected = _expected_count(scene)
    frame_ids = np.asarray(full_frame_ids)
    prediction = _pose_stack("global_prediction_c2w", global_prediction_c2w)
    ground_truth = _pose_stack("raw_gt_c2w", raw_gt_c2w)
    if len(frame_ids) != expected or len(prediction) != expected or len(ground_truth) != expected:
        raise ValueError(f"oracle fit must use exactly {expected} full-scene frames")
    if prediction.shape != ground_truth.shape:
        raise ValueError("global prediction and raw GT must have matching shapes")
    identity_digest = frame_digest(frame_ids)

    source = prediction[:, :3, 3]
    target = ground_truth[:, :3, 3]
    centered = source - source.mean(axis=0)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    largest = float(singular_values[0])
    threshold = largest * limits.rank_relative_tolerance
    rank = int(np.count_nonzero(singular_values > threshold)) if largest > 0 else 0
    if rank < limits.min_rank:
        raise ValueError("oracle source trajectory rank is below minimum")
    condition = float(largest / singular_values[rank - 1])
    if not np.isfinite(condition) or condition > limits.max_condition:
        raise ValueError("oracle source trajectory condition is above maximum")

    scale, rotation, translation = umeyama(source, target)
    if not np.isfinite(scale) or scale < limits.min_scale or scale > limits.max_scale:
        raise ValueError("oracle scale is outside the allowed range")
    determinant = float(np.linalg.det(rotation))
    if not np.isfinite(determinant) or abs(determinant - 1.0) > 1e-8:
        raise ValueError("oracle rotation must be proper")
    rotation_tuple = tuple(tuple(float(value) for value in row) for row in rotation)
    translation_tuple = tuple(float(value) for value in translation)
    transform_payload = {
        "scene": scene,
        "frame_digest": identity_digest,
        "fit_count": expected,
        "scale": float(scale),
        "rotation": rotation_tuple,
        "translation": translation_tuple,
    }
    return FrozenOracle(
        scene=scene,
        frame_digest=identity_digest,
        fit_count=expected,
        scale=float(scale),
        rotation=rotation_tuple,
        translation=translation_tuple,
        rank=rank,
        condition=condition,
        transform_digest=canonical_json_digest(transform_payload),
    )


def apply_frozen_oracle(oracle: FrozenOracle, candidate_c2w: np.ndarray) -> np.ndarray:
    """Apply an existing oracle transform without any fitting or GT access."""
    candidate = _pose_stack("candidate_c2w", candidate_c2w)
    rotation = np.asarray(oracle.rotation, dtype=np.float64)
    translation = np.asarray(oracle.translation, dtype=np.float64)
    aligned = candidate.copy()
    aligned[:, :3, :3] = np.einsum("ij,sjk->sik", rotation, candidate[:, :3, :3])
    aligned[:, :3, 3] = (
        oracle.scale * (candidate[:, :3, 3] @ rotation.T) + translation
    )
    return aligned


def evaluate_with_frozen_oracle(
    oracle: FrozenOracle,
    candidate_c2w: np.ndarray,
    raw_gt_c2w: np.ndarray,
) -> OracleEvaluation:
    """Evaluate any candidate with the already-frozen transform only."""
    candidate = _pose_stack("candidate_c2w", candidate_c2w)
    ground_truth = _pose_stack("raw_gt_c2w", raw_gt_c2w)
    if candidate.shape != ground_truth.shape:
        raise ValueError("candidate and raw GT must have matching shapes")
    aligned = apply_frozen_oracle(oracle, candidate)
    differences = aligned[:, :3, 3] - ground_truth[:, :3, 3]
    errors = np.linalg.norm(differences, axis=1)
    return OracleEvaluation(
        aligned_c2w=aligned,
        translation_error=errors,
        mean_translation_error=float(np.mean(errors)),
        rms_translation_error=float(np.sqrt(np.mean(errors * errors))),
        fit_transform_digest=oracle.transform_digest,
    )
