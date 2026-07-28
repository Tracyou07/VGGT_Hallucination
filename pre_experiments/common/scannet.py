"""Minimal ScanNet frame and raw GT pose loading for Round 2A."""

from __future__ import annotations

from pathlib import Path

import numpy as np


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


def uniform_frame_ids(valid_ids: list[int], count: int) -> list[int]:
    """Match the Camera Context protocol's deterministic uniform selection."""
    if count <= 0:
        raise ValueError("count must be positive")
    if len(valid_ids) < count:
        raise ValueError(f"need at least {count} valid frames, found {len(valid_ids)}")
    if any(not isinstance(value, (int, np.integer)) for value in valid_ids):
        raise ValueError("valid_ids must contain integers")
    if any(left >= right for left, right in zip(valid_ids, valid_ids[1:])):
        raise ValueError("valid_ids must be strictly increasing and unique")
    indices = np.linspace(0, len(valid_ids) - 1, count, dtype=np.int64)
    return [int(valid_ids[int(index)]) for index in indices]
