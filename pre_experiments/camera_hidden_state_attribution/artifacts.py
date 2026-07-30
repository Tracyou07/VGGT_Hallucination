"""Artifact helpers for hidden-state attribution runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from pre_experiments.local_global_consistency.artifacts import atomic_save_npz


CAUSAL_SCENE_ARRAYS = (
    "activation_scale",
    "translation_effect",
    "rotation_effect_deg",
    "fov_effect",
    "measured_basis_mask",
    "direct_iteration",
    "direct_unit",
    "direct_projected_translation",
    "direct_measured_translation",
    "direct_projected_rotation_deg",
    "direct_measured_rotation_deg",
    "direct_projected_fov",
    "direct_measured_fov",
)


def canonical_digest(payload: dict[str, object]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def save_scene_statistics(path: Path, statistics: dict[str, object]) -> None:
    drift = statistics["drift"]
    specificity = statistics["specificity"]
    if not isinstance(drift, dict) or not isinstance(specificity, dict):
        raise ValueError("statistics must contain drift and specificity dictionaries")
    arrays = {
        "translation_drift": np.asarray(drift["translation"], dtype=np.float64),
        "rotation_drift": np.asarray(drift["rotation"], dtype=np.float64),
        "fov_drift": np.asarray(drift["fov"], dtype=np.float64),
        "translation_specificity": np.asarray(
            specificity["translation"], dtype=np.float64
        ),
        "rotation_specificity": np.asarray(
            specificity["rotation"], dtype=np.float64
        ),
        "fov_specificity": np.asarray(specificity["fov"], dtype=np.float64),
        "matched_observation_count": np.asarray(
            [statistics["matched_observation_count"]], dtype=np.int64
        ),
    }
    boundary = statistics["boundary_drift"]
    if not isinstance(boundary, dict):
        raise ValueError("statistics must contain boundary_drift")
    for stratum in ("edge", "interior"):
        arrays[f"{stratum}_observation_count"] = np.asarray(
            [statistics["boundary_counts"][stratum]], dtype=np.int64
        )
        for group in ("translation", "rotation", "fov"):
            arrays[f"{stratum}_{group}_drift"] = np.asarray(
                boundary[stratum][group], dtype=np.float64
            )
    atomic_save_npz(path, arrays)


def load_scene_statistics(path: Path, scene: str) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "translation_drift",
            "rotation_drift",
            "fov_drift",
            "translation_specificity",
            "rotation_specificity",
            "fov_specificity",
            "matched_observation_count",
            "edge_translation_drift",
            "edge_rotation_drift",
            "edge_fov_drift",
            "interior_translation_drift",
            "interior_rotation_drift",
            "interior_fov_drift",
            "edge_observation_count",
            "interior_observation_count",
        }
        if set(archive.files) != required:
            raise ValueError(f"invalid scene statistics members: {path}")
        arrays = {name: np.asarray(archive[name]).copy() for name in required}
    return {
        "scene": scene,
        "drift": {
            "translation": arrays["translation_drift"],
            "rotation": arrays["rotation_drift"],
            "fov": arrays["fov_drift"],
        },
        "specificity": {
            "translation": arrays["translation_specificity"],
            "rotation": arrays["rotation_specificity"],
            "fov": arrays["fov_specificity"],
        },
        "boundary_drift": {
            stratum: {
                group: arrays[f"{stratum}_{group}_drift"]
                for group in ("translation", "rotation", "fov")
            }
            for stratum in ("edge", "interior")
        },
        "boundary_counts": {
            stratum: int(arrays[f"{stratum}_observation_count"][0])
            for stratum in ("edge", "interior")
        },
        "matched_observation_count": int(arrays["matched_observation_count"][0]),
    }


def save_causal_scene_effects(
    path: Path,
    effects: dict[str, np.ndarray],
) -> None:
    """Atomically save one scene's strict numeric causal-effect artifact."""
    atomic_save_npz(path, _validated_causal_arrays(effects))


def load_causal_scene_effects(
    path: Path,
    scene: str,
) -> dict[str, object]:
    """Load and validate one scene's causal-effect artifact."""
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(CAUSAL_SCENE_ARRAYS):
            raise ValueError(f"invalid causal scene artifact members: {path}")
        arrays = {
            name: np.asarray(archive[name]).copy()
            for name in CAUSAL_SCENE_ARRAYS
        }
    return {"scene": scene, **_validated_causal_arrays(arrays)}


def _validated_causal_arrays(
    effects: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    if set(effects) != set(CAUSAL_SCENE_ARRAYS):
        raise ValueError("invalid causal scene artifact members")
    arrays = {
        name: np.asarray(effects[name])
        for name in CAUSAL_SCENE_ARRAYS
    }
    activation = np.asarray(arrays["activation_scale"], dtype=np.float64)
    if activation.ndim != 2 or min(activation.shape) < 1:
        raise ValueError("activation_scale must have shape [iteration, unit]")
    iterations, hidden_dim = activation.shape
    if not np.isfinite(activation).all() or np.any(activation <= 0):
        raise ValueError("activation_scale must be finite and positive")

    result: dict[str, np.ndarray] = {"activation_scale": activation}
    for name in (
        "translation_effect",
        "rotation_effect_deg",
        "fov_effect",
    ):
        values = np.asarray(arrays[name], dtype=np.float64)
        if (
            values.shape != (iterations, hidden_dim)
            or not np.isfinite(values).all()
            or np.any(values < 0)
        ):
            raise ValueError(
                f"{name} must be finite and non-negative with shape "
                "[iteration, unit]"
            )
        result[name] = values

    measured = arrays["measured_basis_mask"]
    if measured.dtype != np.bool_ or measured.ndim != 2:
        raise ValueError(
            "measured_basis_mask must be boolean with shape [iteration, basis]"
        )
    if measured.shape[0] != iterations or measured.shape[1] < 1:
        raise ValueError("measured_basis_mask iteration shape mismatch")
    result["measured_basis_mask"] = measured.astype(bool, copy=False)

    direct_iteration = arrays["direct_iteration"]
    direct_unit = arrays["direct_unit"]
    if direct_iteration.dtype.kind not in "iu" or direct_unit.dtype.kind not in "iu":
        raise ValueError("direct check indices must be integers")
    direct_iteration = direct_iteration.astype(np.int64, copy=False)
    direct_unit = direct_unit.astype(np.int64, copy=False)
    count = len(direct_iteration)
    if (
        direct_iteration.ndim != 1
        or direct_unit.shape != (count,)
        or np.any(direct_iteration < 0)
        or np.any(direct_iteration >= iterations)
        or np.any(direct_unit < 0)
        or np.any(direct_unit >= hidden_dim)
    ):
        raise ValueError("direct check indices are out of range")
    result["direct_iteration"] = direct_iteration
    result["direct_unit"] = direct_unit

    for name in CAUSAL_SCENE_ARRAYS[7:]:
        values = np.asarray(arrays[name], dtype=np.float64)
        if (
            values.shape != (count,)
            or not np.isfinite(values).all()
            or np.any(values < 0)
        ):
            raise ValueError(
                f"{name} must be finite and non-negative with direct-check shape"
            )
        result[name] = values
    return result
