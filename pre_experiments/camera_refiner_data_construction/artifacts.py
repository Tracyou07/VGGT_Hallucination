"""Strict external shards for multiscale Camera hidden experiments."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np

from pre_experiments.camera_refiner_data_construction.protocol import (
    LOCAL_SCALES,
)
from pre_experiments.local_global_consistency.artifacts import atomic_save_npz


SCHEMA_VERSION = 1
SCENE_SHARD_ARRAYS = (
    "schema_version",
    "scene_name",
    "frame_ids",
    "scales",
    "candidate_names",
    "candidate_alpha",
    "candidate_beta",
    "global_hidden",
    "local_hidden",
    "selected_window_index",
    "selected_boundary_distance",
    "selected_window_start",
    "selected_window_stop",
    "local_observation_count",
    "pred_c2w_raw",
    "pose_enc",
    "gt_c2w_raw",
    "translation_error_aligned",
    "rotation_error_deg_aligned",
    "hidden_displacement_rms",
    "camera_center_displacement_mean",
    "rotation_change_deg_mean",
    "fov_change_mean",
)


def _finite_array(
    payload: Mapping[str, object],
    name: str,
    shape: tuple[int, ...],
    *,
    non_negative: bool = False,
) -> np.ndarray:
    values = np.asarray(payload[name], dtype=np.float64)
    if values.shape != shape or not np.isfinite(values).all():
        raise ValueError(f"{name} must contain finite values with shape {shape}")
    if non_negative and np.any(values < 0.0):
        raise ValueError(f"{name} must be non-negative")
    return values


def _validated_arrays(payload: Mapping[str, object]) -> dict[str, np.ndarray]:
    required = set(SCENE_SHARD_ARRAYS) - {"schema_version", "scene_name"}
    if not required.issubset(payload):
        raise ValueError("scene shard members are missing")
    scene = payload.get("scene")
    if not isinstance(scene, str) or not scene:
        raise ValueError("scene identity must be a non-empty string")

    frame_ids = np.asarray(payload["frame_ids"])
    if (
        frame_ids.ndim != 1
        or frame_ids.dtype.kind not in "iu"
        or len(frame_ids) < 2
        or len(np.unique(frame_ids)) != len(frame_ids)
        or np.any(np.diff(frame_ids.astype(np.int64)) <= 0)
    ):
        raise ValueError("frame_ids must be unique increasing integers")
    frame_ids = frame_ids.astype(np.int64)
    frame_count = len(frame_ids)

    scales = np.asarray(payload["scales"])
    if scales.dtype.kind not in "iu" or not np.array_equal(scales, LOCAL_SCALES):
        raise ValueError(f"scales must equal {LOCAL_SCALES}")
    scale_count = len(LOCAL_SCALES)

    names = np.asarray(payload["candidate_names"])
    if (
        names.ndim != 1
        or names.dtype.kind not in "US"
        or len(names) < 2
        or names[0] != "baseline"
        or len(np.unique(names)) != len(names)
    ):
        raise ValueError("candidate_names must start with one unique baseline")
    names = names.astype(str)
    candidate_count = len(names)

    alpha = _finite_array(payload, "candidate_alpha", (candidate_count,))
    if (
        alpha[0] != 0.0
        or np.any(alpha[1:] <= 0.0)
        or np.any(alpha[1:] > 1.0)
    ):
        raise ValueError("candidate_alpha must contain baseline zero then (0, 1]")
    beta = _finite_array(
        payload,
        "candidate_beta",
        (candidate_count, scale_count),
        non_negative=True,
    )
    if not np.array_equal(beta[0], np.zeros(scale_count)) or not np.allclose(
        beta[1:].sum(axis=1), 1.0, atol=1e-8, rtol=0.0
    ):
        raise ValueError("candidate_beta must contain baseline zeros then simplex rows")

    global_hidden = np.asarray(payload["global_hidden"], dtype=np.float32)
    if (
        global_hidden.ndim != 3
        or global_hidden.shape[1] != frame_count
        or global_hidden.shape[0] < 1
        or global_hidden.shape[2] < 1
        or not np.isfinite(global_hidden).all()
    ):
        raise ValueError("global_hidden must be finite [iteration, frame, hidden]")
    iterations, _, hidden_dim = global_hidden.shape
    local_hidden = np.asarray(payload["local_hidden"], dtype=np.float32)
    expected_local = (scale_count, iterations, frame_count, hidden_dim)
    if local_hidden.shape != expected_local or not np.isfinite(local_hidden).all():
        raise ValueError(f"local_hidden must be finite with shape {expected_local}")

    integer_arrays = {}
    for name in (
        "selected_window_index",
        "selected_boundary_distance",
        "selected_window_start",
        "selected_window_stop",
        "local_observation_count",
    ):
        values = np.asarray(payload[name])
        if (
            values.dtype.kind not in "iu"
            or values.shape != (scale_count, frame_count)
            or np.any(values < 0)
        ):
            raise ValueError(
                f"{name} must contain non-negative integers with shape "
                f"{(scale_count, frame_count)}"
            )
        integer_arrays[name] = values.astype(np.int64)
    if np.any(integer_arrays["local_observation_count"] < 1):
        raise ValueError("local_observation_count must be positive")
    lengths = (
        integer_arrays["selected_window_stop"]
        - integer_arrays["selected_window_start"]
    )
    if not np.array_equal(
        lengths,
        np.broadcast_to(np.asarray(LOCAL_SCALES)[:, None], lengths.shape),
    ):
        raise ValueError("selected window boundaries do not match their scales")

    pred_c2w = _finite_array(
        payload,
        "pred_c2w_raw",
        (candidate_count, frame_count, 4, 4),
    )
    pose_enc = _finite_array(
        payload,
        "pose_enc",
        (candidate_count, frame_count, 9),
    )
    gt_c2w = _finite_array(payload, "gt_c2w_raw", (frame_count, 4, 4))
    translation = _finite_array(
        payload,
        "translation_error_aligned",
        (candidate_count, frame_count),
        non_negative=True,
    )
    rotation = _finite_array(
        payload,
        "rotation_error_deg_aligned",
        (candidate_count, frame_count),
        non_negative=True,
    )
    scalar_metrics = {
        name: _finite_array(
            payload,
            name,
            (candidate_count,),
            non_negative=True,
        )
        for name in (
            "hidden_displacement_rms",
            "camera_center_displacement_mean",
            "rotation_change_deg_mean",
            "fov_change_mean",
        )
    }
    if any(values[0] != 0.0 for values in scalar_metrics.values()):
        raise ValueError("baseline displacement metrics must be zero")

    return {
        "schema_version": np.asarray(SCHEMA_VERSION, dtype=np.int64),
        "scene_name": np.asarray(scene),
        "frame_ids": frame_ids,
        "scales": scales.astype(np.int64),
        "candidate_names": names,
        "candidate_alpha": alpha,
        "candidate_beta": beta,
        "global_hidden": global_hidden,
        "local_hidden": local_hidden,
        **integer_arrays,
        "pred_c2w_raw": pred_c2w,
        "pose_enc": pose_enc,
        "gt_c2w_raw": gt_c2w,
        "translation_error_aligned": translation,
        "rotation_error_deg_aligned": rotation,
        **scalar_metrics,
    }


def save_scene_shard(path: Path, payload: Mapping[str, object]) -> None:
    """Validate and atomically save one external multiscale scene shard."""
    atomic_save_npz(path, _validated_arrays(payload))


def load_scene_shard(path: Path, scene: str) -> dict[str, object]:
    """Load one exact shard and verify its requested scene identity."""
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(SCENE_SHARD_ARRAYS):
            raise ValueError(f"invalid scene shard members: {path}")
        if int(np.asarray(archive["schema_version"])) != SCHEMA_VERSION:
            raise ValueError(f"unsupported scene shard schema: {path}")
        arrays = {
            name: np.asarray(archive[name]).copy()
            for name in SCENE_SHARD_ARRAYS
            if name not in {"schema_version", "scene_name"}
        }
        stored_scene = str(np.asarray(archive["scene_name"]).item())
    if stored_scene != scene:
        raise ValueError(
            f"scene identity mismatch: expected {scene}, found {stored_scene}"
        )
    return {"scene": scene, **_validated_arrays({"scene": scene, **arrays})}
