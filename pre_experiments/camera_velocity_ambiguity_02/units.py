"""Adjacent local-window overlap identities for CVA02."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Sequence

from pre_experiments.local_global_consistency.windows import FrameWindow


_SCENE_PATTERN = re.compile(r"scene\d{4}_\d{2}")


@dataclass(frozen=True)
class OverlapUnit:
    """One adjacent-window pair; frames may legitimately occur in other pairs."""

    scene: str
    pair_id: str
    left_window_index: int
    right_window_index: int
    left_start: int
    left_stop: int
    right_start: int
    right_stop: int
    shared_frame_ids: tuple[int, ...]
    left_shared_indices: tuple[int, ...]
    right_shared_indices: tuple[int, ...]
    global_shared_indices: tuple[int, ...]
    route: str


def _validate_window(window: FrameWindow) -> None:
    ids = tuple(window.frame_ids)
    if window.start < 0 or window.stop <= window.start or len(ids) != window.stop - window.start:
        raise ValueError("window boundaries must match frame count")
    if any(not isinstance(value, int) for value in ids):
        raise ValueError("window frame IDs must contain integers")
    if any(left >= right for left, right in zip(ids, ids[1:])):
        raise ValueError("window frame IDs must be strictly increasing and unique")


def build_overlap_units(
    scene: str,
    windows: Sequence[FrameWindow],
    *,
    primary_overlap: int,
) -> tuple[OverlapUnit, ...]:
    """Build ordered pair units without collapsing identity by shared frame ID."""
    if not isinstance(scene, str) or _SCENE_PATTERN.fullmatch(scene) is None:
        raise ValueError("scene must use ScanNet sceneNNNN_NN identity")
    if not isinstance(primary_overlap, int) or isinstance(primary_overlap, bool) or primary_overlap < 2:
        raise ValueError("primary_overlap must be an integer of at least two")
    for window in windows:
        _validate_window(window)

    units: list[OverlapUnit] = []
    for left, right in zip(windows, windows[1:]):
        if right.index != left.index + 1:
            raise ValueError("windows must have adjacent window indices")
        if right.start <= left.start or right.stop <= left.stop or right.start >= left.stop:
            raise ValueError("adjacent window boundaries must form a forward overlap")
        positional_overlap = left.stop - right.start
        left_start = right.start - left.start
        left_ids = tuple(left.frame_ids[left_start:])
        right_ids = tuple(right.frame_ids[:positional_overlap])
        if left_ids != right_ids or len(left_ids) != positional_overlap:
            raise ValueError("window boundaries and shared frame identities are inconsistent")
        if len(left_ids) < 2:
            raise ValueError("adjacent windows must share at least two frames")

        left_indices = tuple(range(left_start, len(left.frame_ids)))
        right_indices = tuple(range(positional_overlap))
        global_indices = tuple(range(right.start, left.stop))
        route = "primary" if positional_overlap == primary_overlap else "secondary"
        units.append(
            OverlapUnit(
                scene=scene,
                pair_id=(
                    f"{scene}/window_{left.index:03d}__window_{right.index:03d}"
                ),
                left_window_index=left.index,
                right_window_index=right.index,
                left_start=left.start,
                left_stop=left.stop,
                right_start=right.start,
                right_stop=right.stop,
                shared_frame_ids=left_ids,
                left_shared_indices=left_indices,
                right_shared_indices=right_indices,
                global_shared_indices=global_indices,
                route=route,
            )
        )
    return tuple(units)
