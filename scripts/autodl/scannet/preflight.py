"""Validate configured ScanNet inputs without importing VGGT or PyTorch."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal, Sequence


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def read_scene_list(path: Path, limit: int) -> list[str]:
    """Read configured scenes without importing the experiment package."""
    if limit < 0:
        raise ValueError("scene limit must be non-negative")
    scenes = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    return scenes[:limit] if limit else scenes


def _image_stems(color_dir: Path) -> set[str]:
    if not color_dir.is_dir():
        return set()
    return {
        path.stem
        for path in color_dir.iterdir()
        if path.is_file()
        and path.stat().st_size > 0
        and path.suffix.lower() in IMAGE_SUFFIXES
    }


def _finite_pose_stems(pose_dir: Path) -> set[str]:
    if not pose_dir.is_dir():
        return set()
    valid: set[str] = set()
    for path in pose_dir.glob("*.txt"):
        try:
            values = [float(value) for value in path.read_text(encoding="utf-8").split()]
        except (OSError, ValueError):
            continue
        if len(values) == 16 and all(math.isfinite(value) for value in values):
            valid.add(path.stem)
    return valid


def processed_scene_is_complete(scene_dir: Path) -> bool:
    """Return whether a scene has a matching non-empty image and finite pose."""
    return bool(
        _image_stems(scene_dir / "color").intersection(
            _finite_pose_stems(scene_dir / "pose")
        )
    )


def missing_processed_scenes(root: Path, scenes: Sequence[str]) -> list[str]:
    process_root = root / "process_scannet"
    return [
        scene
        for scene in scenes
        if not processed_scene_is_complete(process_root / scene)
    ]


def _raw_scene_is_complete(scans_root: Path, scene: str) -> bool:
    return scans_root.is_dir() and any(
        path.is_file() and path.stat().st_size > 0
        for path in scans_root.rglob(f"{scene}.sens")
    )


def detect_scannet_layout(
    scannet_root: Path,
    scenes: Sequence[str] | None = None,
) -> Literal["processed", "raw"]:
    """Prefer extracted color/pose data, then accept local raw .sens files."""
    if scenes:
        missing_processed = missing_processed_scenes(scannet_root, scenes)
        if not missing_processed:
            return "processed"
        scans_root = scannet_root / "raw_sens" / "scans"
        missing_raw = [
            scene for scene in scenes if not _raw_scene_is_complete(scans_root, scene)
        ]
        if not missing_raw:
            return "raw"
        raise FileNotFoundError(
            "SCANNET_ROOT is incomplete; missing processed scenes "
            f"{missing_processed} and raw scenes {missing_raw}: "
            f"{scannet_root.resolve()}"
        )

    process_root = scannet_root / "process_scannet"
    if process_root.is_dir() and any(
        processed_scene_is_complete(path) for path in process_root.iterdir()
        if path.is_dir()
    ):
        return "processed"
    scans_root = scannet_root / "raw_sens" / "scans"
    if scans_root.is_dir() and any(
        path.is_file() and path.stat().st_size > 0
        for path in scans_root.rglob("*.sens")
    ):
        return "raw"
    raise FileNotFoundError(
        "SCANNET_ROOT has neither process_scannet/<scene>/{color,pose} "
        f"nor raw_sens/scans/**/*.sens: {scannet_root.resolve()}"
    )
