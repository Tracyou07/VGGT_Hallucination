"""Strict raw artifacts for global and local Camera consistency analysis."""

from __future__ import annotations

from pathlib import Path

import numpy as np


GLOBAL_CONTEXT_MEMBERS = {
    "frame_ids",
    "normalized_camera_tokens",
    "pred_c2w_raw",
    "pred_c2w_aligned",
    "gt_c2w_raw",
    "translation_error_aligned",
    "rotation_error_deg_aligned",
    "delta_norm",
    "sim3_scale",
    "sim3_rotation",
    "sim3_translation",
}
WINDOW_MEMBERS = {
    "frame_ids",
    "normalized_camera_tokens",
    "pred_c2w_raw",
    "gt_c2w_raw",
}


def _load_exact_npz(path: Path, expected: set[str]) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"required diagnostics artifact is missing: {path}")
    with np.load(path, allow_pickle=False) as archive:
        members = set(archive.files)
        if members != expected:
            raise ValueError(f"unexpected NPZ members in {path}: {sorted(members)}")
        arrays = {name: np.asarray(archive[name]).copy() for name in members}
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise ValueError(f"diagnostics contain non-finite values: {path}")
    return arrays


def load_global_context(path: Path) -> dict[str, np.ndarray]:
    """Load an exact published Round 1.5 context artifact."""
    return _load_exact_npz(path, GLOBAL_CONTEXT_MEMBERS)


def load_window_diagnostics(path: Path) -> dict[str, np.ndarray]:
    """Load one exact local-window raw artifact."""
    return _load_exact_npz(path, WINDOW_MEMBERS)


def build_window_diagnostics(
    *,
    frame_ids: np.ndarray,
    normalized_camera_tokens: np.ndarray,
    pred_w2c: np.ndarray,
    gt_c2w_raw: np.ndarray,
) -> dict[str, np.ndarray]:
    """Build raw local diagnostics while preserving raw GT exactly."""
    ids = np.asarray(frame_ids, dtype=np.int64)
    tokens = np.asarray(normalized_camera_tokens, dtype=np.float32)
    predicted_w2c = np.asarray(pred_w2c, dtype=np.float64)
    ground_truth = np.asarray(gt_c2w_raw, dtype=np.float64)
    if ids.ndim != 1 or len(ids) < 2 or len(np.unique(ids)) != len(ids):
        raise ValueError("frame_ids must contain at least two unique frames")
    sequence_length = len(ids)
    if (
        tokens.ndim != 2
        or len(tokens) != sequence_length
        or predicted_w2c.shape != (sequence_length, 4, 4)
        or ground_truth.shape != (sequence_length, 4, 4)
    ):
        raise ValueError("all window arrays must match the frame sequence length")
    for name, value in (
        ("normalized_camera_tokens", tokens),
        ("pred_w2c", predicted_w2c),
        ("gt_c2w_raw", ground_truth),
    ):
        if not np.isfinite(value).all():
            raise ValueError(f"{name} must contain only finite values")
    return {
        "frame_ids": ids,
        "normalized_camera_tokens": tokens,
        "pred_c2w_raw": np.linalg.inv(predicted_w2c),
        "gt_c2w_raw": ground_truth.copy(),
    }


def atomic_save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)
