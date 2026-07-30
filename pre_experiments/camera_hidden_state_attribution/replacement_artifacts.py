"""Strict per-scene artifacts for short-to-long hidden replacement."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np

from pre_experiments.local_global_consistency.artifacts import (
    atomic_save_npz,
)


REPLACEMENT_SCENE_ARRAYS = (
    "condition_names",
    "condition_family",
    "condition_alpha",
    "replacement_count",
    "frame_ids",
    "selected_window_index",
    "selected_boundary_distance",
    "local_observation_count",
    "pred_c2w_raw",
    "pose_enc",
    "translation_error_aligned",
    "rotation_error_deg_aligned",
)


def _validated_arrays(
    result: Mapping[str, object],
) -> dict[str, np.ndarray]:
    if not set(REPLACEMENT_SCENE_ARRAYS).issubset(result):
        raise ValueError("replacement scene artifact members are missing")
    names = np.asarray(result["condition_names"])
    if (
        names.ndim != 1
        or names.dtype.kind not in "US"
        or len(names) < 2
        or names[0] != "baseline"
        or len(np.unique(names)) != len(names)
    ):
        raise ValueError(
            "condition_names must start with a unique baseline"
        )
    condition_count = len(names)
    families = np.asarray(result["condition_family"])
    if (
        families.ndim != 1
        or families.dtype.kind not in "US"
        or families.shape != (condition_count,)
        or families[0] != "baseline"
        or np.count_nonzero(families == "baseline") != 1
        or np.count_nonzero(families == "selected") < 1
        or not set(families.tolist()).issubset(
            {"baseline", "selected", "control"}
        )
    ):
        raise ValueError("condition_family has invalid values")
    alphas = np.asarray(result["condition_alpha"], dtype=np.float64)
    if (
        alphas.shape != (condition_count,)
        or not np.isfinite(alphas).all()
        or alphas[0] != 0.0
        or np.any(alphas < 0)
        or np.any(alphas > 1)
        or np.any(alphas[1:] <= 0)
    ):
        raise ValueError("condition_alpha has invalid values")

    replacement_count = np.asarray(result["replacement_count"])
    if (
        replacement_count.dtype.kind not in "iu"
        or replacement_count.shape != (condition_count,)
        or replacement_count[0] != 0
        or np.any(replacement_count < 0)
    ):
        raise ValueError("replacement_count has invalid values")
    frame_ids = np.asarray(result["frame_ids"])
    if (
        frame_ids.dtype.kind not in "iu"
        or frame_ids.ndim != 1
        or len(frame_ids) < 2
        or len(np.unique(frame_ids)) != len(frame_ids)
    ):
        raise ValueError("frame_ids must contain unique integer values")
    frame_count = len(frame_ids)

    integer_arrays = {
        name: np.asarray(result[name])
        for name in (
            "selected_window_index",
            "selected_boundary_distance",
            "local_observation_count",
        )
    }
    for name, values in integer_arrays.items():
        if (
            values.dtype.kind not in "iu"
            or values.shape != (frame_count,)
            or np.any(values < 0)
        ):
            raise ValueError(f"{name} must be non-negative per-frame integers")
    if np.any(integer_arrays["local_observation_count"] < 1):
        raise ValueError("local_observation_count must be positive")

    floating_shapes = {
        "pred_c2w_raw": (condition_count, frame_count, 4, 4),
        "pose_enc": (condition_count, frame_count, 9),
        "translation_error_aligned": (condition_count, frame_count),
        "rotation_error_deg_aligned": (condition_count, frame_count),
    }
    floating_arrays = {}
    for name, shape in floating_shapes.items():
        values = np.asarray(result[name], dtype=np.float64)
        if values.shape != shape or not np.isfinite(values).all():
            raise ValueError(f"{name} must contain finite values with shape {shape}")
        if name.endswith("error_aligned") and np.any(values < 0):
            raise ValueError(f"{name} must be non-negative")
        floating_arrays[name] = values

    return {
        "condition_names": names.astype(str),
        "condition_family": families.astype(str),
        "condition_alpha": alphas,
        "replacement_count": replacement_count.astype(np.int64),
        "frame_ids": frame_ids.astype(np.int64),
        **{
            name: values.astype(np.int64)
            for name, values in integer_arrays.items()
        },
        **floating_arrays,
    }


def save_replacement_scene(
    path: Path,
    result: Mapping[str, object],
) -> None:
    atomic_save_npz(path, _validated_arrays(result))


def load_replacement_scene(
    path: Path,
    scene: str,
) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(REPLACEMENT_SCENE_ARRAYS):
            raise ValueError(f"invalid replacement scene members: {path}")
        arrays = {
            name: np.asarray(archive[name]).copy()
            for name in REPLACEMENT_SCENE_ARRAYS
        }
    return {"scene": scene, **_validated_arrays(arrays)}
