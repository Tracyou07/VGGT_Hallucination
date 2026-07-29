"""Artifact helpers for hidden-state attribution runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from pre_experiments.local_global_consistency.artifacts import atomic_save_npz


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
