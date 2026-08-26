"""Translation-only residual interpolation and mandatory convexity checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from pre_experiments.camera_velocity_ambiguity_02.contracts import ProtocolViolation
from pre_experiments.camera_velocity_ambiguity_02.frozen_oracle import (
    FrozenOracle,
    evaluate_with_frozen_oracle,
)


@dataclass(frozen=True)
class TranslationCandidate:
    alpha: float
    c2w: np.ndarray
    fov: np.ndarray | None


@dataclass(frozen=True)
class TranslationCurve:
    alphas: np.ndarray
    per_frame_l2: np.ndarray
    mean_l2: np.ndarray
    rms_l2: np.ndarray
    rte_rms: np.ndarray
    transform_digest: str


def _pose_stack(value: np.ndarray) -> np.ndarray:
    poses = np.asarray(value, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or len(poses) < 2:
        raise ValueError("global_c2w must have shape [frames, 4, 4]")
    if not np.isfinite(poses).all():
        raise ValueError("global_c2w must contain only finite values")
    if not np.allclose(poses[:, 3, :], [0.0, 0.0, 0.0, 1.0], atol=1e-10, rtol=0):
        raise ValueError("global_c2w must contain homogeneous poses")
    return poses


def build_translation_candidates(
    global_c2w: np.ndarray,
    left_residual: np.ndarray,
    right_residual: np.ndarray,
    *,
    alphas: Sequence[float],
    global_fov: np.ndarray | None = None,
) -> tuple[TranslationCandidate, ...]:
    """Interpolate signed centers while copying global rotation and FoV verbatim."""
    poses = _pose_stack(global_c2w)
    left = np.asarray(left_residual, dtype=np.float64)
    right = np.asarray(right_residual, dtype=np.float64)
    expected = (len(poses), 3)
    if left.shape != expected or right.shape != expected:
        raise ValueError("left/right residuals must have shape [frames, 3]")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("left/right residuals must contain only finite values")
    alpha_values = np.asarray(tuple(alphas), dtype=np.float64)
    if (
        alpha_values.ndim != 1
        or len(alpha_values) < 2
        or not np.isfinite(alpha_values).all()
        or np.any(np.diff(alpha_values) <= 0)
        or alpha_values[0] != 0.0
        or alpha_values[-1] != 1.0
    ):
        raise ValueError("alphas must be finite, increasing, and span exactly 0 to 1")
    fov = None
    if global_fov is not None:
        fov = np.asarray(global_fov)
        if fov.shape != (len(poses), 2) or not np.isfinite(fov).all():
            raise ValueError("global_fov must have finite shape [frames, 2]")

    candidates: list[TranslationCandidate] = []
    global_centers = poses[:, :3, 3]
    for value in alpha_values:
        candidate = poses.copy()
        correction = (1.0 - value) * left + value * right
        candidate[:, :3, 3] = global_centers + correction
        candidates.append(
            TranslationCandidate(
                alpha=float(value),
                c2w=candidate,
                fov=None if fov is None else fov.copy(),
            )
        )
    return tuple(candidates)


def evaluate_translation_candidates(
    oracle: FrozenOracle,
    candidates: Sequence[TranslationCandidate],
    raw_gt_c2w: np.ndarray,
) -> TranslationCurve:
    """Evaluate all alphas through one frozen transform, never candidate refits."""
    if len(candidates) < 2:
        raise ValueError("at least two translation candidates are required")
    ground_truth = _pose_stack(raw_gt_c2w)
    alphas = np.asarray([candidate.alpha for candidate in candidates], dtype=np.float64)
    if not np.isfinite(alphas).all() or np.any(np.diff(alphas) <= 0):
        raise ValueError("candidate alphas must be finite and increasing")
    per_frame: list[np.ndarray] = []
    mean: list[float] = []
    rms: list[float] = []
    rte: list[float] = []
    target_steps = np.diff(ground_truth[:, :3, 3], axis=0)
    for candidate in candidates:
        evaluation = evaluate_with_frozen_oracle(oracle, candidate.c2w, ground_truth)
        if evaluation.fit_transform_digest != oracle.transform_digest:
            raise ProtocolViolation("candidate evaluation changed the frozen transform")
        per_frame.append(evaluation.translation_error)
        mean.append(evaluation.mean_translation_error)
        rms.append(evaluation.rms_translation_error)
        candidate_steps = np.diff(evaluation.aligned_c2w[:, :3, 3], axis=0)
        step_error = np.linalg.norm(candidate_steps - target_steps, axis=1)
        rte.append(float(np.sqrt(np.mean(step_error * step_error))))
    return TranslationCurve(
        alphas=alphas,
        per_frame_l2=np.stack(per_frame),
        mean_l2=np.asarray(mean, dtype=np.float64),
        rms_l2=np.asarray(rms, dtype=np.float64),
        rte_rms=np.asarray(rte, dtype=np.float64),
        transform_digest=oracle.transform_digest,
    )


def _assert_convex(name: str, alphas: np.ndarray, values: np.ndarray, tolerance: float) -> None:
    slopes = np.diff(values, axis=0) / np.diff(alphas).reshape((-1,) + (1,) * (values.ndim - 1))
    scale = max(1.0, float(np.max(np.abs(values))))
    if np.any(np.diff(slopes, axis=0) < -tolerance * scale):
        raise ProtocolViolation(f"translation-only {name} curve violated convexity")


def assert_translation_curve_convex(
    curve: TranslationCurve,
    *,
    tolerance: float = 1e-9,
) -> None:
    """Fail closed if an affine translation path appears to create a barrier."""
    alphas = np.asarray(curve.alphas, dtype=np.float64)
    per_frame = np.asarray(curve.per_frame_l2, dtype=np.float64)
    mean = np.asarray(curve.mean_l2, dtype=np.float64)
    rms = np.asarray(curve.rms_l2, dtype=np.float64)
    rte = np.asarray(curve.rte_rms, dtype=np.float64)
    count = len(alphas)
    if (
        alphas.ndim != 1
        or count < 3
        or np.any(np.diff(alphas) <= 0)
        or per_frame.ndim != 2
        or per_frame.shape[0] != count
        or mean.shape != (count,)
        or rms.shape != (count,)
        or rte.shape != (count,)
        or not all(np.isfinite(value).all() for value in (alphas, per_frame, mean, rms, rte))
    ):
        raise ProtocolViolation("translation curve has invalid shapes or non-finite values")
    if tolerance < 0 or not np.isfinite(tolerance):
        raise ValueError("convexity tolerance must be finite and non-negative")
    _assert_convex("per-frame L2", alphas, per_frame, tolerance)
    _assert_convex("mean L2", alphas, mean, tolerance)
    _assert_convex("RMS", alphas, rms, tolerance)
