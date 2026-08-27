from __future__ import annotations

from collections.abc import Mapping

import numpy as np


SOURCE_SCHEMA = "variational_camera_latent.source.v1"
SOURCE_REQUIRED_MEMBERS = {
    "global_frame_ids",
    "global_camera_tokens",
    "short_frame_ids",
    "short_camera_tokens",
    "overlap_frame_ids",
    "overlap_long_tokens",
    "overlap_left_tokens",
    "overlap_right_tokens",
    "span_starts",
    "sample_ids",
}
_FORBIDDEN_NAME_PARTS = ("gt", "ground_truth", "privileged", "depth", "error")


def _strictly_increasing(values: np.ndarray, label: str) -> None:
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError(f"{label} frame IDs must be a one-dimensional integer vector")
    if np.any(values[1:] <= values[:-1]):
        raise ValueError(f"{label} frame IDs must be strictly increasing")


def validate_source_shard(arrays: Mapping[str, np.ndarray]) -> None:
    """Validate the prediction-only 500/100/50 source contract."""
    names = set(arrays)
    forbidden = sorted(
        name for name in names if any(part in name.lower() for part in _FORBIDDEN_NAME_PARTS)
    )
    if forbidden:
        raise ValueError(f"source shard may not contain GT or privileged members: {forbidden}")
    missing = SOURCE_REQUIRED_MEMBERS - names
    extra = names - SOURCE_REQUIRED_MEMBERS
    if missing or extra:
        raise ValueError(
            f"source shard members mismatch; missing={sorted(missing)}, extra={sorted(extra)}"
        )

    normalized = {name: np.asarray(value) for name, value in arrays.items()}
    if any(value.dtype.hasobject for value in normalized.values()):
        raise ValueError("source shard may not contain object arrays")

    global_ids = normalized["global_frame_ids"]
    short_ids = normalized["short_frame_ids"]
    overlap_ids = normalized["overlap_frame_ids"]
    global_tokens = normalized["global_camera_tokens"]
    short_tokens = normalized["short_camera_tokens"]
    overlap_long = normalized["overlap_long_tokens"]
    overlap_left = normalized["overlap_left_tokens"]
    overlap_right = normalized["overlap_right_tokens"]
    span_starts = normalized["span_starts"]
    sample_ids = normalized["sample_ids"]

    if global_ids.shape != (500,):
        raise ValueError("global frame IDs must have shape [500]")
    if short_ids.shape != (9, 100):
        raise ValueError("short frame IDs must have shape [9, 100]")
    if overlap_ids.shape != (8, 50):
        raise ValueError("overlap frame IDs must have shape [8, 50]")
    _strictly_increasing(global_ids, "global")
    for index, values in enumerate(short_ids):
        _strictly_increasing(values, f"short window {index}")
    for index, values in enumerate(overlap_ids):
        _strictly_increasing(values, f"overlap {index}")

    expected_shapes = {
        "global_camera_tokens": (500, 2048),
        "short_camera_tokens": (9, 100, 2048),
        "overlap_long_tokens": (8, 50, 2048),
        "overlap_left_tokens": (8, 50, 2048),
        "overlap_right_tokens": (8, 50, 2048),
    }
    for name, expected in expected_shapes.items():
        value = normalized[name]
        if value.shape != expected:
            raise ValueError(f"{name} must have shape {expected}")
        if not np.issubdtype(value.dtype, np.floating) or not np.isfinite(value).all():
            raise ValueError(f"{name} must contain finite floating-point values")

    expected_starts = np.arange(0, 400, 50, dtype=np.int64)
    if span_starts.shape != (8,) or not np.array_equal(span_starts, expected_starts):
        raise ValueError("span_starts must be [0, 50, ..., 350]")
    if sample_ids.shape != (8,) or sample_ids.dtype.kind != "U":
        raise ValueError("sample IDs must be a Unicode vector with shape [8]")
    if len(set(sample_ids.tolist())) != 8 or any(not value for value in sample_ids.tolist()):
        raise ValueError("sample IDs must be non-empty and unique")

    for index, start in enumerate(range(0, 401, 50)):
        if not np.array_equal(short_ids[index], global_ids[start : start + 100]):
            raise ValueError(f"short window {index} frame IDs do not align with global frame IDs")
    for index, start in enumerate(range(50, 401, 50)):
        if not np.array_equal(short_ids[index, 50:], short_ids[index + 1, :50]):
            raise ValueError(f"adjacent short-window frame IDs disagree at overlap {index}")
        if not np.array_equal(overlap_ids[index], global_ids[start : start + 50]):
            raise ValueError(f"overlap {index} frame IDs do not align with global frame IDs")
        if not np.array_equal(overlap_left[index], short_tokens[index, 50:]):
            raise ValueError(f"overlap {index} left tokens do not match the left teacher")
        if not np.array_equal(overlap_right[index], short_tokens[index + 1, :50]):
            raise ValueError(f"overlap {index} right tokens do not match the right teacher")
        if not np.array_equal(overlap_long[index], global_tokens[start : start + 50]):
            raise ValueError(f"overlap {index} long tokens do not match global context")
