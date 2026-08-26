"""FastVGGT-compatible ScanNet frame selection for CVA02."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from pre_experiments.common.scannet import frame_id
from pre_experiments.local_global_consistency.windows import (
    FrameWindow,
    build_sliding_windows,
)


@dataclass(frozen=True)
class FrameSelection:
    """One ordered selection and its identities in the finite-pose sequence."""

    frame_ids: tuple[int, ...]
    image_paths: tuple[Path, ...]
    pose_indices: tuple[int, ...]


def _finite_pose_ids(values: Iterable[int]) -> tuple[int, ...]:
    raw = tuple(values)
    if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) for value in raw):
        raise ValueError("finite_pose_frame_ids must contain integers")
    normalized = tuple(int(value) for value in raw)
    if len(set(normalized)) != len(normalized):
        raise ValueError("finite_pose_frame_ids must be unique")
    return tuple(sorted(normalized))


def build_fastvggt_frame_selection(
    image_paths: Sequence[Path],
    finite_pose_frame_ids: Iterable[int],
    *,
    input_frames: int,
) -> FrameSelection:
    """Port FastVGGT's preserve-first, floor-stride ScanNet selection exactly."""
    if isinstance(input_frames, bool) or not isinstance(input_frames, (int, np.integer)):
        raise ValueError("input_frames must be a positive integer")
    count = int(input_frames)
    if count <= 0:
        raise ValueError("input_frames must be a positive integer")

    image_by_id: dict[int, Path] = {}
    for raw_path in image_paths:
        path = Path(raw_path)
        identifier = frame_id(path)
        if identifier in image_by_id:
            raise ValueError(f"duplicate image frame ID: {identifier}")
        image_by_id[identifier] = path

    pose_ids = _finite_pose_ids(finite_pose_frame_ids)
    pose_index_by_id = {identifier: index for index, identifier in enumerate(pose_ids)}
    valid_ids = sorted(set(image_by_id).intersection(pose_index_by_id))
    if len(valid_ids) < count:
        raise ValueError(f"need at least {count} valid image/finite-pose frames, found {len(valid_ids)}")

    if len(valid_ids) > count:
        if count == 1:
            selected = valid_ids[:1]
        else:
            first = valid_ids[0]
            remaining = valid_ids[1:]
            step = max(1, len(remaining) // (count - 1))
            selected = [first, *remaining[::step][: count - 1]]
    else:
        selected = valid_ids

    if len(selected) != count or any(left >= right for left, right in zip(selected, selected[1:])):
        raise RuntimeError("FastVGGT frame selection failed to produce the requested ordered count")
    return FrameSelection(
        frame_ids=tuple(selected),
        image_paths=tuple(image_by_id[identifier] for identifier in selected),
        pose_indices=tuple(pose_index_by_id[identifier] for identifier in selected),
    )


def build_protocol_windows(
    selection: FrameSelection,
    *,
    length: int = 100,
    stride: int = 50,
) -> tuple[FrameWindow, ...]:
    """Build the frozen overlapping windows over selected, not raw, positions."""
    windows = build_sliding_windows(
        np.asarray(selection.frame_ids, dtype=np.int64),
        length=length,
        stride=stride,
    )
    return tuple(windows)
