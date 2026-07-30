"""Pure calculations for the Camera Head hidden causal preference atlas."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


GROUPS = ("translation", "rotation", "fov")
OUTPUT_KEYS = ("camera_center", "rotation", "fov")


def activation_rms(
    hidden: np.ndarray,
    *,
    floor_ratio: float = 0.05,
    absolute_floor: float = 1e-6,
) -> np.ndarray:
    """Return per-iteration/unit RMS activation with a robust positive floor."""
    array = np.asarray(hidden, dtype=np.float64)
    if array.ndim != 3 or min(array.shape) < 1:
        raise ValueError("hidden must have shape [iteration, frame, unit]")
    if not np.isfinite(array).all():
        raise ValueError("hidden values must be finite")
    if not np.isfinite(floor_ratio) or floor_ratio < 0:
        raise ValueError("floor_ratio must be finite and non-negative")
    if not np.isfinite(absolute_floor) or absolute_floor <= 0:
        raise ValueError("absolute_floor must be finite and positive")

    rms = np.sqrt(np.mean(np.square(array), axis=1))
    iteration_floor = np.maximum(
        np.median(rms, axis=1) * floor_ratio,
        absolute_floor,
    )
    return np.maximum(rms, iteration_floor[:, None])


def central_output_jacobians(
    positive: Mapping[str, np.ndarray],
    negative: Mapping[str, np.ndarray],
    *,
    basis_step: float,
) -> dict[str, np.ndarray]:
    """Recover final-output Jacobians from centered basis perturbations."""
    if not np.isfinite(basis_step) or basis_step <= 0:
        raise ValueError("basis_step must be finite and positive")
    if set(positive) != set(OUTPUT_KEYS) or set(negative) != set(OUTPUT_KEYS):
        raise ValueError(f"outputs must contain exactly {OUTPUT_KEYS}")

    expected_suffixes = {
        "camera_center": (3,),
        "rotation": (3, 3),
        "fov": (2,),
    }
    leading_shape: tuple[int, int, int] | None = None
    result: dict[str, np.ndarray] = {}
    for name in OUTPUT_KEYS:
        plus = np.asarray(positive[name], dtype=np.float64)
        minus = np.asarray(negative[name], dtype=np.float64)
        suffix = expected_suffixes[name]
        expected_ndim = 3 + len(suffix)
        if (
            plus.shape != minus.shape
            or plus.ndim != expected_ndim
            or tuple(plus.shape[-len(suffix) :]) != suffix
        ):
            raise ValueError(
                f"{name} outputs must share shape [iteration, basis, frame, ...]"
            )
        current_leading = tuple(plus.shape[:3])
        if leading_shape is None:
            leading_shape = current_leading
        elif current_leading != leading_shape:
            raise ValueError("all output groups must share iteration/basis/frame axes")
        if not np.isfinite(plus).all() or not np.isfinite(minus).all():
            raise ValueError(f"{name} outputs must be finite")
        result[name] = (plus - minus) / (2.0 * basis_step)
    return result


def project_hidden_effects(
    jacobians: Mapping[str, np.ndarray],
    *,
    baseline_rotations: np.ndarray,
    output_weight: np.ndarray,
    activation_scales: np.ndarray,
    unit_chunk_size: int = 256,
) -> dict[str, np.ndarray]:
    """Project 9D output Jacobians onto standardized hidden-unit directions."""
    if set(jacobians) != set(OUTPUT_KEYS):
        raise ValueError(f"jacobians must contain exactly {OUTPUT_KEYS}")
    center = np.asarray(jacobians["camera_center"], dtype=np.float64)
    rotation = np.asarray(jacobians["rotation"], dtype=np.float64)
    fov = np.asarray(jacobians["fov"], dtype=np.float64)
    if (
        center.ndim != 4
        or center.shape[-1] != 3
        or rotation.shape != center.shape[:3] + (3, 3)
        or fov.shape != center.shape[:3] + (2,)
    ):
        raise ValueError("invalid causal Jacobian shapes")
    if not all(np.isfinite(values).all() for values in (center, rotation, fov)):
        raise ValueError("causal Jacobians must be finite")

    iterations, basis_dimensions, frames, _ = center.shape
    baseline = np.asarray(baseline_rotations, dtype=np.float64)
    weight = np.asarray(output_weight, dtype=np.float64)
    scales = np.asarray(activation_scales, dtype=np.float64)
    if baseline.shape != (frames, 3, 3) or not np.isfinite(baseline).all():
        raise ValueError("baseline_rotations must be finite with shape [frame, 3, 3]")
    if weight.ndim != 2 or weight.shape[0] != basis_dimensions:
        raise ValueError("output_weight must have shape [basis, unit]")
    hidden_dim = weight.shape[1]
    if scales.shape != (iterations, hidden_dim):
        raise ValueError("activation_scales must have shape [iteration, unit]")
    if (
        not np.isfinite(weight).all()
        or not np.isfinite(scales).all()
        or np.any(scales < 0)
    ):
        raise ValueError("weights and activation scales must be finite and non-negative")
    if unit_chunk_size < 1:
        raise ValueError("unit_chunk_size must be positive")

    effects = {
        group: np.empty((iterations, hidden_dim), dtype=np.float64)
        for group in GROUPS
    }
    for iteration in range(iterations):
        for start in range(0, hidden_dim, unit_chunk_size):
            stop = min(start + unit_chunk_size, hidden_dim)
            directions = (
                weight[:, start:stop] * scales[iteration, start:stop][None, :]
            )

            center_derivative = np.einsum(
                "dsx,dk->ksx",
                center[iteration],
                directions,
                optimize=True,
            )
            effects["translation"][iteration, start:stop] = np.linalg.norm(
                center_derivative,
                axis=-1,
            ).mean(axis=1)

            rotation_derivative = np.einsum(
                "dsab,dk->ksab",
                rotation[iteration],
                directions,
                optimize=True,
            )
            local_derivative = np.einsum(
                "sba,ksac->ksbc",
                baseline,
                rotation_derivative,
                optimize=True,
            )
            skew = 0.5 * (
                local_derivative
                - np.swapaxes(local_derivative, -1, -2)
            )
            angular_speed = np.sqrt(
                np.maximum(
                    0.0,
                    0.5 * np.square(skew).sum(axis=(-2, -1)),
                )
            )
            effects["rotation"][iteration, start:stop] = np.degrees(
                angular_speed
            ).mean(axis=1)

            fov_derivative = np.einsum(
                "dsx,dk->ksx",
                fov[iteration],
                directions,
                optimize=True,
            )
            effects["fov"][iteration, start:stop] = np.linalg.norm(
                fov_derivative,
                axis=-1,
            ).mean(axis=1)
    return effects


def fit_causal_normalization(
    calibration_effects: Mapping[str, np.ndarray],
    *,
    quantile: float = 0.9,
    minimum_scale: float = 1e-12,
) -> dict[str, float]:
    """Fit one robust calibration scale per output group."""
    arrays = _validate_group_arrays(calibration_effects)
    if not np.isfinite(quantile) or not 0 < quantile <= 1:
        raise ValueError("quantile must be in (0, 1]")
    if not np.isfinite(minimum_scale) or minimum_scale <= 0:
        raise ValueError("minimum_scale must be finite and positive")
    return {
        group: float(
            max(np.quantile(arrays[group], quantile), minimum_scale)
        )
        for group in GROUPS
    }


def apply_causal_normalization(
    effects: Mapping[str, np.ndarray],
    scales: Mapping[str, float],
) -> dict[str, object]:
    """Apply frozen output scales and derive per-position group preferences."""
    arrays = _validate_group_arrays(effects)
    if set(scales) != set(GROUPS):
        raise ValueError(f"scales must contain exactly {GROUPS}")
    numeric_scales = {group: float(scales[group]) for group in GROUPS}
    if any(
        not np.isfinite(value) or value <= 0
        for value in numeric_scales.values()
    ):
        raise ValueError("normalization scales must be finite and positive")

    normalized = {
        group: arrays[group] / numeric_scales[group]
        for group in GROUPS
    }
    stacked = np.stack([normalized[group] for group in GROUPS], axis=-1)
    denominator = stacked.sum(axis=-1, keepdims=True)
    preferences_stacked = np.divide(
        stacked,
        denominator,
        out=np.zeros_like(stacked),
        where=denominator > np.finfo(np.float64).eps,
    )
    group_names = np.asarray(GROUPS)
    preferred_group = group_names[np.argmax(preferences_stacked, axis=-1)]
    return {
        "normalized": {
            group: preferences_source
            for group, preferences_source in zip(
                GROUPS,
                np.moveaxis(stacked, -1, 0),
            )
        },
        "preferences": {
            group: preference
            for group, preference in zip(
                GROUPS,
                np.moveaxis(preferences_stacked, -1, 0),
            )
        },
        "preferred_group": preferred_group,
    }


def _validate_group_arrays(
    values: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    if set(values) != set(GROUPS):
        raise ValueError(f"group arrays must contain exactly {GROUPS}")
    arrays = {
        group: np.asarray(values[group], dtype=np.float64)
        for group in GROUPS
    }
    shape = arrays[GROUPS[0]].shape
    if len(shape) != 2 or min(shape) < 1:
        raise ValueError("group effects must have shape [iteration, unit]")
    for group, array in arrays.items():
        if array.shape != shape:
            raise ValueError("group effect arrays must share one shape")
        if not np.isfinite(array).all() or np.any(array < 0):
            raise ValueError(f"{group} effects must be finite and non-negative")
    return arrays
