"""Prediction-free access to Camera Context source frame identities."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_context_frame_ids(path: Path) -> np.ndarray:
    """Read only frame IDs from a context artifact.

    Split construction must remain independent of every VGGT prediction array.
    """
    if not path.is_file():
        raise FileNotFoundError(f"context diagnostics artifact is missing: {path}")
    with np.load(path, allow_pickle=False) as archive:
        if "frame_ids" not in archive.files:
            raise ValueError(f"context diagnostics has no frame_ids member: {path}")
        raw_ids = np.asarray(archive["frame_ids"])
    if raw_ids.ndim != 1 or len(raw_ids) < 2:
        raise ValueError("context frame_ids must be a one-dimensional sequence")
    if not np.issubdtype(raw_ids.dtype, np.integer):
        raise ValueError("context frame_ids must have an integer dtype")
    frame_ids = raw_ids.astype(np.int64, copy=True)
    if np.any(np.diff(frame_ids) <= 0):
        raise ValueError("context frame_ids must be strictly increasing and unique")
    return frame_ids
