"""Minimal ScanNet frame and raw GT pose loading for the camera study."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def read_scene_list(path: Path, limit: int = 0) -> list[str]:
    """Read ordered, non-comment scene IDs from a UTF-8 text file."""
    if limit < 0:
        raise ValueError("limit must be non-negative")
    scenes = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    return scenes[:limit] if limit else scenes


def frame_id(path: Path) -> int:
    """Parse the integer frame ID used by extracted ScanNet files."""
    try:
        return int(path.stem)
    except ValueError as error:
        raise ValueError(f"frame filename must have a numeric stem: {path.name}") from error


def get_sorted_image_paths(color_dir: Path) -> list[Path]:
    """Return supported color images sorted by numeric frame ID."""
    paths: list[Path] = []
    for suffix in ("*.jpg", "*.jpeg", "*.png"):
        paths.extend(color_dir.glob(suffix))
    return sorted(paths, key=frame_id)


def load_poses(pose_dir: Path) -> dict[int, np.ndarray]:
    """Load finite 4x4 raw GT camera-to-world poses, skipping invalid files."""
    poses: dict[int, np.ndarray] = {}
    for path in sorted(pose_dir.glob("*.txt"), key=frame_id):
        try:
            pose = np.loadtxt(path, dtype=np.float64)
        except (OSError, ValueError):
            continue
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            continue
        poses[frame_id(path)] = pose
    return poses


def linspace_indices(total: int, count: int) -> np.ndarray:
    """Select up to count stable indices spanning an ordered sequence."""
    if total < 1 or count < 1:
        raise ValueError("total and count must be positive")
    if count >= total:
        return np.arange(total, dtype=np.int64)
    return np.linspace(0, total - 1, count, dtype=np.int64)


def _regime_step_ids(valid_ids: list[int], count: int) -> list[int]:
    if len(valid_ids) <= count:
        return list(valid_ids)
    if count == 1:
        return [valid_ids[0]]
    remaining = valid_ids[1:]
    step = max(1, len(remaining) // (count - 1))
    return [valid_ids[0], *remaining[::step][: count - 1]]


def make_frame_selections(
    valid_ids: list[int],
    frame_counts: list[int],
    sampling: str,
) -> dict[int, list[int]]:
    """Create deterministic frame selections keyed by requested frame count."""
    if not valid_ids:
        raise ValueError("valid_ids must not be empty")
    if not frame_counts or any(type(count) is not int or count < 1 for count in frame_counts):
        raise ValueError("frame_counts must contain positive integers")
    if len(set(frame_counts)) != len(frame_counts):
        raise ValueError("frame_counts must be unique")
    if sampling not in {"prefix", "uniform", "nested_uniform", "regime_step"}:
        raise ValueError(f"unknown sampling mode: {sampling}")

    selections: dict[int, list[int]] = {}
    if sampling == "nested_uniform":
        largest_count = max(frame_counts)
        largest = [valid_ids[index] for index in linspace_indices(len(valid_ids), largest_count)]
        for count in frame_counts:
            selections[count] = [
                largest[index] for index in linspace_indices(len(largest), count)
            ]
        return selections

    for count in frame_counts:
        if sampling == "prefix":
            selections[count] = valid_ids[:count]
        elif sampling == "uniform":
            selections[count] = [
                valid_ids[index] for index in linspace_indices(len(valid_ids), count)
            ]
        else:
            selections[count] = _regime_step_ids(valid_ids, count)
    return selections


def load_scene_frames(
    data_dir: Path,
    scene: str,
) -> tuple[dict[int, Path], dict[int, np.ndarray], list[int]]:
    """Load image and raw GT pose maps and return their ordered ID intersection."""
    scene_dir = data_dir / scene
    image_paths = get_sorted_image_paths(scene_dir / "color")
    image_by_id = {frame_id(path): path for path in image_paths}
    poses_by_id = load_poses(scene_dir / "pose")
    valid_ids = sorted(set(image_by_id).intersection(poses_by_id))
    if not valid_ids:
        raise FileNotFoundError(f"no matching color frames and finite poses for {scene}")
    return image_by_id, poses_by_id, valid_ids
