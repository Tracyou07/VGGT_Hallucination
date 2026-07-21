"""Matched-frame metrics for comparing two Camera context lengths."""

from __future__ import annotations

import numpy as np

from pre_experiments.camera_iteration.pose_metrics import rotation_angle_deg


ContextRow = dict[str, float | int]
ContextSummary = dict[str, float | int | None]


def _validated_context(context: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    required = {
        "frame_ids",
        "normalized_camera_tokens",
        "pred_c2w_aligned",
        "translation_error_aligned",
        "rotation_error_deg_aligned",
        "delta_norm",
    }
    missing = required.difference(context)
    if missing:
        raise ValueError(f"context is missing arrays: {sorted(missing)}")

    arrays = {name: np.asarray(context[name]) for name in required}
    frame_ids = arrays["frame_ids"]
    if frame_ids.ndim != 1 or len(frame_ids) == 0:
        raise ValueError("frame_ids must be a non-empty vector")
    if len(np.unique(frame_ids)) != len(frame_ids):
        raise ValueError("frame_ids must be unique")
    expected = len(frame_ids)
    if arrays["normalized_camera_tokens"].ndim != 2:
        raise ValueError("normalized_camera_tokens must have shape [S, C]")
    if arrays["pred_c2w_aligned"].shape != (expected, 4, 4):
        raise ValueError("pred_c2w_aligned must have shape [S, 4, 4]")
    for name in (
        "normalized_camera_tokens",
        "translation_error_aligned",
        "rotation_error_deg_aligned",
        "delta_norm",
    ):
        if len(arrays[name]) != expected:
            raise ValueError(f"{name} must match frame_ids")
        if not np.isfinite(arrays[name]).all():
            raise ValueError(f"{name} must contain only finite values")
    return arrays


def _normalize_rows(tokens: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(tokens, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("normalized_camera_tokens must have non-zero row norms")
    return tokens / norms


def _pearson_or_none(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 2 or np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def compare_contexts(
    short: dict[str, np.ndarray],
    long: dict[str, np.ndarray],
    *,
    short_frames: int,
    long_frames: int,
) -> tuple[list[ContextRow], ContextSummary]:
    """Compare matched frames after each context was independently aligned."""
    if short_frames >= long_frames:
        raise ValueError("short_frames must be smaller than long_frames")
    short_arrays = _validated_context(short)
    long_arrays = _validated_context(long)
    long_index = {
        int(frame_id): index
        for index, frame_id in enumerate(long_arrays["frame_ids"].tolist())
    }
    matches = [
        (short_index, long_index[int(frame_id)])
        for short_index, frame_id in enumerate(short_arrays["frame_ids"].tolist())
        if int(frame_id) in long_index
    ]
    if not matches:
        raise ValueError("contexts must contain at least one shared frame")

    short_indices = np.asarray([pair[0] for pair in matches], dtype=np.int64)
    long_indices = np.asarray([pair[1] for pair in matches], dtype=np.int64)
    short_tokens = _normalize_rows(
        short_arrays["normalized_camera_tokens"][short_indices].astype(np.float64)
    )
    long_tokens = _normalize_rows(
        long_arrays["normalized_camera_tokens"][long_indices].astype(np.float64)
    )
    cosine_distance = 1.0 - np.sum(short_tokens * long_tokens, axis=1)
    affinity_drift = float(
        np.mean(np.abs(short_tokens @ short_tokens.T - long_tokens @ long_tokens.T))
    )

    rows: list[ContextRow] = []
    for match_index, (short_index, long_index_value) in enumerate(matches):
        short_pose = short_arrays["pred_c2w_aligned"][short_index]
        long_pose = long_arrays["pred_c2w_aligned"][long_index_value]
        short_translation_error = float(
            short_arrays["translation_error_aligned"][short_index]
        )
        long_translation_error = float(
            long_arrays["translation_error_aligned"][long_index_value]
        )
        short_rotation_error = float(
            short_arrays["rotation_error_deg_aligned"][short_index]
        )
        long_rotation_error = float(
            long_arrays["rotation_error_deg_aligned"][long_index_value]
        )
        short_delta = float(short_arrays["delta_norm"][short_index])
        long_delta = float(long_arrays["delta_norm"][long_index_value])
        rows.append(
            {
                "short_frames": short_frames,
                "long_frames": long_frames,
                "frame_id": int(short_arrays["frame_ids"][short_index]),
                "token_cosine_distance": float(cosine_distance[match_index]),
                "aligned_center_drift": float(
                    np.linalg.norm(short_pose[:3, 3] - long_pose[:3, 3])
                ),
                "aligned_rotation_drift_deg": rotation_angle_deg(
                    short_pose[:3, :3].T @ long_pose[:3, :3]
                ),
                "translation_error_short": short_translation_error,
                "translation_error_long": long_translation_error,
                "translation_error_change": long_translation_error
                - short_translation_error,
                "rotation_error_short_deg": short_rotation_error,
                "rotation_error_long_deg": long_rotation_error,
                "rotation_error_change_deg": long_rotation_error
                - short_rotation_error,
                "delta_norm_short": short_delta,
                "delta_norm_long": long_delta,
                "delta_norm_change": long_delta - short_delta,
            }
        )

    translation_error_changes = np.asarray(
        [row["translation_error_change"] for row in rows], dtype=np.float64
    )
    delta_changes = np.asarray(
        [row["delta_norm_change"] for row in rows], dtype=np.float64
    )
    summary: ContextSummary = {
        "short_frames": short_frames,
        "long_frames": long_frames,
        "shared_frame_count": len(rows),
        "token_cosine_distance_mean": float(np.mean(cosine_distance)),
        "token_cosine_distance_p95": float(np.quantile(cosine_distance, 0.95)),
        "token_pairwise_affinity_drift": affinity_drift,
        "aligned_center_drift_mean": float(
            np.mean([row["aligned_center_drift"] for row in rows])
        ),
        "translation_error_change_mean": float(
            np.mean([row["translation_error_change"] for row in rows])
        ),
        "rotation_error_change_mean_deg": float(
            np.mean([row["rotation_error_change_deg"] for row in rows])
        ),
        "delta_norm_change_mean": float(
            np.mean(delta_changes)
        ),
        "token_translation_error_change_pearson": _pearson_or_none(
            cosine_distance, translation_error_changes
        ),
        "delta_translation_error_change_pearson": _pearson_or_none(
            delta_changes, translation_error_changes
        ),
    }
    return rows, summary
