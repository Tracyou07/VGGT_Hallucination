"""Frozen negative controls for signed CVA02 repair directions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pre_experiments.camera_velocity_ambiguity_02.geometry import (
    ResidualDirectionMetrics,
    compute_residual_direction_metrics,
)


CONTROL_NAMES = frozenset(
    {
        "self",
        "gauge_copy",
        "random_wrong_window",
        "sign_inversion",
        "epsilon",
        "degenerate_alignment",
    }
)


@dataclass(frozen=True)
class ControlCase:
    name: str
    alignment_valid: bool
    metrics: ResidualDirectionMetrics


def _residuals(name: str, value: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must have finite shape {shape}")
    return array


def _case(
    name: str,
    left: np.ndarray,
    right: np.ndarray,
    *,
    scene_scale: float,
    alignment_valid: bool = True,
) -> ControlCase:
    zeros = np.zeros_like(left)
    return ControlCase(
        name=name,
        alignment_valid=alignment_valid,
        metrics=compute_residual_direction_metrics(
            zeros,
            left,
            right,
            scene_scale=scene_scale,
        ),
    )


def build_negative_controls(
    left_residual: np.ndarray,
    right_residual: np.ndarray,
    *,
    wrong_window_residual: np.ndarray,
    scene_scale: float,
    seed: int = 33,
) -> dict[str, ControlCase]:
    """Build all controls with the same signed residual metric implementation."""
    left = np.asarray(left_residual, dtype=np.float64)
    if left.ndim != 2 or left.shape[1] != 3 or len(left) < 2 or not np.isfinite(left).all():
        raise ValueError("left_residual must have finite shape [frames, 3]")
    right = _residuals("right_residual", right_residual, left.shape)
    wrong = _residuals("wrong_window_residual", wrong_window_residual, left.shape)
    if not np.isfinite(scene_scale) or scene_scale <= 1e-12:
        raise ValueError("scene_scale must be finite and positive")
    rng = np.random.default_rng(seed)
    shuffled_wrong = wrong[rng.permutation(len(wrong))]
    epsilon = left + rng.normal(scale=1e-6 * scene_scale, size=left.shape)
    zeros = np.zeros_like(left)
    return {
        "self": _case("self", left, left, scene_scale=scene_scale),
        "gauge_copy": _case("gauge_copy", zeros, zeros, scene_scale=scene_scale),
        "random_wrong_window": _case(
            "random_wrong_window", left, shuffled_wrong, scene_scale=scene_scale
        ),
        "sign_inversion": _case(
            "sign_inversion", left, -left, scene_scale=scene_scale
        ),
        "epsilon": _case("epsilon", left, epsilon, scene_scale=scene_scale),
        "degenerate_alignment": _case(
            "degenerate_alignment",
            zeros,
            zeros,
            scene_scale=scene_scale,
            alignment_valid=False,
        ),
    }
