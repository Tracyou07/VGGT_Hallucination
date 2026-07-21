"""Compact numeric artifact construction for one context selection."""

from __future__ import annotations

import numpy as np

from pre_experiments.camera_iteration.pose_metrics import align_pose_sequence


def build_context_diagnostics(
    *,
    frame_ids: np.ndarray,
    normalized_camera_tokens: np.ndarray,
    pred_w2c: np.ndarray,
    gt_c2w_raw: np.ndarray,
    delta_norm: np.ndarray,
) -> dict[str, np.ndarray]:
    """Build aligned prediction diagnostics while preserving raw GT poses."""
    ids = np.asarray(frame_ids, dtype=np.int64)
    tokens = np.asarray(normalized_camera_tokens, dtype=np.float32)
    predicted_w2c = np.asarray(pred_w2c, dtype=np.float64)
    ground_truth = np.asarray(gt_c2w_raw, dtype=np.float64)
    deltas = np.asarray(delta_norm, dtype=np.float32)
    if ids.ndim != 1 or len(ids) < 2:
        raise ValueError("frame_ids must contain at least two frames")
    sequence_length = len(ids)
    if (
        tokens.ndim != 2
        or predicted_w2c.shape != (sequence_length, 4, 4)
        or ground_truth.shape != (sequence_length, 4, 4)
        or deltas.shape != (sequence_length,)
    ):
        raise ValueError("all context arrays must match the frame sequence length")
    if len(np.unique(ids)) != sequence_length:
        raise ValueError("frame_ids must be unique")
    for name, array in (
        ("normalized_camera_tokens", tokens),
        ("pred_w2c", predicted_w2c),
        ("gt_c2w_raw", ground_truth),
        ("delta_norm", deltas),
    ):
        if not np.isfinite(array).all():
            raise ValueError(f"{name} must contain only finite values")

    alignment = align_pose_sequence(predicted_w2c, ground_truth)
    return {
        "frame_ids": ids,
        "normalized_camera_tokens": tokens,
        "pred_c2w_raw": np.linalg.inv(predicted_w2c),
        "pred_c2w_aligned": np.asarray(alignment["aligned_c2w"], dtype=np.float64),
        "gt_c2w_raw": ground_truth.copy(),
        "translation_error_aligned": np.asarray(
            alignment["translation_error_aligned"], dtype=np.float64
        ),
        "rotation_error_deg_aligned": np.asarray(
            alignment["rotation_error_deg_aligned"], dtype=np.float64
        ),
        "delta_norm": deltas,
        "sim3_scale": np.asarray(float(alignment["sim3_scale"]), dtype=np.float64),
        "sim3_rotation": np.asarray(alignment["sim3_rotation"], dtype=np.float64),
        "sim3_translation": np.asarray(
            alignment["sim3_translation"], dtype=np.float64
        ),
    }
