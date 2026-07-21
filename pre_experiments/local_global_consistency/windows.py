"""Deterministic overlapping windows over an ordered selected sequence."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FrameWindow:
    index: int
    start: int
    stop: int
    frame_ids: tuple[int, ...]
    boundary_distance: tuple[int, ...]


def build_sliding_windows(
    frame_ids: np.ndarray,
    *,
    length: int = 100,
    stride: int = 50,
) -> list[FrameWindow]:
    """Cover ordered frame IDs with deterministic, gap-free windows."""
    ids = np.asarray(frame_ids)
    if ids.ndim != 1 or len(ids) < 2:
        raise ValueError("frame_ids must be a vector with at least two entries")
    if not np.isfinite(ids).all():
        raise ValueError("frame_ids must contain only finite values")
    integer_ids = ids.astype(np.int64, copy=False)
    if not np.array_equal(ids, integer_ids):
        raise ValueError("frame_ids must contain integers")
    if len(np.unique(integer_ids)) != len(integer_ids):
        raise ValueError("frame_ids must be unique")
    if length < 2 or length > len(integer_ids):
        raise ValueError("length must be between 2 and the sequence length")
    if stride < 1 or stride > length:
        raise ValueError("stride must be between 1 and length")

    final_start = len(integer_ids) - length
    starts = list(range(0, final_start + 1, stride))
    if starts[-1] != final_start:
        starts.append(final_start)

    windows = []
    for index, start in enumerate(starts):
        stop = start + length
        local_ids = tuple(int(value) for value in integer_ids[start:stop])
        boundary = tuple(min(position, length - 1 - position) for position in range(length))
        windows.append(
            FrameWindow(
                index=index,
                start=start,
                stop=stop,
                frame_ids=local_ids,
                boundary_distance=boundary,
            )
        )
    return windows
