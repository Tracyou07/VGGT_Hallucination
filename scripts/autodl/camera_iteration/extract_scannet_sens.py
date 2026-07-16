"""Extract configured ScanNet color frames and raw camera poses from .sens."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Protocol


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.autodl.camera_iteration.preflight import read_scene_list


class SensorDataLike(Protocol):
    def __init__(self, sens_file: str) -> None: ...

    def export_color_images(self, output_dir: str) -> None: ...

    def export_poses(self, output_dir: str) -> None: ...


def find_sens_file(raw_dir: Path, scene: str) -> Path:
    """Find one scene's local .sens file using standard layouts first."""
    candidates = (
        raw_dir / scene / f"{scene}.sens",
        raw_dir / f"{scene}.sens",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    recursive = sorted(raw_dir.rglob(f"{scene}.sens")) if raw_dir.is_dir() else []
    if recursive:
        return recursive[0]
    raise FileNotFoundError(f"Missing .sens file for {scene} under {raw_dir.resolve()}")


def _frame_stems(directory: Path, suffixes: set[str]) -> set[str]:
    if not directory.is_dir():
        return set()
    return {
        path.stem
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in suffixes
    }


def scene_is_complete(scene_dir: Path) -> bool:
    """Require at least one frame with both an image and raw GT pose."""
    color_ids = _frame_stems(scene_dir / "color", {".jpg", ".jpeg", ".png"})
    pose_ids = _frame_stems(scene_dir / "pose", {".txt"})
    return bool(color_ids.intersection(pose_ids))


def extract_scene(
    sensor_data_type: type[SensorDataLike],
    sens_file: Path,
    output_dir: Path,
) -> bool:
    """Extract one scene unless a matching color/pose pair already exists."""
    if scene_is_complete(output_dir):
        print(f"[extract] skip complete {output_dir.name}")
        return False

    print(f"[extract] {sens_file} -> {output_dir}")
    sensor_data = sensor_data_type(str(sens_file))
    sensor_data.export_color_images(str(output_dir / "color"))
    sensor_data.export_poses(str(output_dir / "pose"))
    if not scene_is_complete(output_dir):
        raise RuntimeError(f"Extraction produced no matching color/pose frames: {output_dir}")
    return True


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--scene-limit", type=int, default=10)
    args = parser.parse_args(argv)
    if args.scene_limit < 0:
        parser.error("--scene-limit must be non-negative")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    scenes = read_scene_list(args.scene_list, args.scene_limit)
    if not scenes:
        raise ValueError(f"No scenes selected from {args.scene_list}")

    from scripts.autodl.camera_iteration.sensreader_py3.SensorData import SensorData

    for scene in scenes:
        sens_file = find_sens_file(args.raw_dir, scene)
        extract_scene(SensorData, sens_file, args.out_dir / scene)


if __name__ == "__main__":
    main()
