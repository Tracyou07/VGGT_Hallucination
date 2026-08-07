"""Window coverage and prediction-only local trajectory assembly."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from pre_experiments.camera_refiner_training.geometry import umeyama


@dataclass(frozen=True)
class FrameWindow:
    index: int
    start: int
    stop: int
    frame_ids: tuple[int, ...]


def build_sliding_windows(
    frame_ids: np.ndarray,
    *,
    length: int = 100,
    stride: int = 50,
) -> list[FrameWindow]:
    ids = np.asarray(frame_ids)
    if ids.ndim != 1 or len(ids) < 2 or not np.isfinite(ids).all():
        raise ValueError("frame_ids must contain at least two finite values")
    integer_ids = ids.astype(np.int64, copy=False)
    if not np.array_equal(ids, integer_ids) or len(np.unique(integer_ids)) != len(ids):
        raise ValueError("frame_ids must contain unique integers")
    if length < 2 or length > len(ids) or stride < 1 or stride > length:
        raise ValueError("invalid window length or stride")
    final_start = len(ids) - length
    starts = list(range(0, final_start + 1, stride))
    if starts[-1] != final_start:
        starts.append(final_start)
    return [
        FrameWindow(
            index=index,
            start=start,
            stop=start + length,
            frame_ids=tuple(int(value) for value in integer_ids[start : start + length]),
        )
        for index, start in enumerate(starts)
    ]


def _window_arrays(
    window: dict[str, np.ndarray], index: int
) -> tuple[np.ndarray, np.ndarray]:
    try:
        frame_ids = np.asarray(window["frame_ids"], dtype=np.int64)
        poses = np.asarray(window["pred_c2w_raw"], dtype=np.float64)
    except KeyError as error:
        raise ValueError(f"window {index} is missing {error.args[0]}") from error
    if frame_ids.ndim != 1 or len(frame_ids) < 3 or len(np.unique(frame_ids)) != len(frame_ids):
        raise ValueError(f"window {index} frame_ids must contain at least three unique IDs")
    if poses.shape != (len(frame_ids), 4, 4) or not np.isfinite(poses).all():
        raise ValueError(f"window {index} poses must be finite with shape [S, 4, 4]")
    return frame_ids, poses


def assemble_windows_in_reference_gauge(
    windows: Sequence[dict[str, np.ndarray]],
    *,
    reference_frame_ids: np.ndarray,
    reference_c2w: np.ndarray,
) -> dict[str, np.ndarray]:
    frame_ids = np.asarray(reference_frame_ids, dtype=np.int64)
    reference = np.asarray(reference_c2w, dtype=np.float64)
    if frame_ids.ndim != 1 or len(frame_ids) < 3 or len(np.unique(frame_ids)) != len(frame_ids):
        raise ValueError("reference_frame_ids must contain at least three unique IDs")
    if reference.shape != (len(frame_ids), 4, 4) or not np.isfinite(reference).all():
        raise ValueError("reference_c2w must be finite with shape [S, 4, 4]")
    if not windows:
        raise ValueError("at least one local window is required")
    reference_index = {int(frame_id): index for index, frame_id in enumerate(frame_ids)}
    assembled: dict[int, np.ndarray] = {}
    boundary: dict[int, int] = {}
    counts = {int(frame_id): 0 for frame_id in frame_ids}
    for window_index, window in enumerate(windows):
        local_ids, poses = _window_arrays(window, window_index)
        try:
            indices = np.asarray([reference_index[int(value)] for value in local_ids])
        except KeyError as error:
            raise ValueError(f"window {window_index} contains an unknown frame") from error
        reference_segment = reference[indices]
        scale, rotation, translation = umeyama(
            poses[:, :3, 3], reference_segment[:, :3, 3]
        )
        aligned = poses.copy()
        aligned[:, :3, :3] = np.einsum("ij,sjk->sik", rotation, poses[:, :3, :3])
        aligned[:, :3, 3] = scale * (poses[:, :3, 3] @ rotation.T) + translation
        for local_index, value in enumerate(local_ids):
            frame_id = int(value)
            counts[frame_id] += 1
            distance = min(local_index, len(local_ids) - 1 - local_index)
            if frame_id not in assembled or distance > boundary[frame_id]:
                assembled[frame_id] = aligned[local_index]
                boundary[frame_id] = distance
    missing = [int(value) for value in frame_ids if int(value) not in assembled]
    if missing:
        raise ValueError(f"local windows do not cover reference frames: {missing[:5]}")
    return {
        "frame_ids": frame_ids.copy(),
        "assembled_c2w": np.stack([assembled[int(value)] for value in frame_ids]),
        "observation_count": np.asarray([counts[int(value)] for value in frame_ids]),
    }
