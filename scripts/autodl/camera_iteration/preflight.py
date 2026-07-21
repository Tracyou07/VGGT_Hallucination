"""Validate local AutoDL inputs without importing VGGT or PyTorch."""

from __future__ import annotations

import argparse
import importlib.util
import math
from pathlib import Path
from typing import Literal, Sequence


MODULE_SPECS = (
    ("numpy", "numpy<2"),
    ("PIL", "Pillow"),
    ("huggingface_hub", "huggingface_hub"),
    ("einops", "einops"),
    ("safetensors", "safetensors"),
    ("cv2", "opencv-python-headless==4.11.0.86"),
    ("imageio", "imageio"),
)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def missing_package_specs() -> list[str]:
    """Return pip requirement specs for unavailable study dependencies."""
    return [
        package_spec
        for module_name, package_spec in MODULE_SPECS
        if importlib.util.find_spec(module_name) is None
    ]


def find_checkpoint(checkpoint_dir: Path) -> Path:
    """Resolve a supported checkpoint without a network fallback."""
    for filename in ("model.safetensors", "model.pt"):
        candidate = checkpoint_dir / filename
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate.resolve()
    raise FileNotFoundError(
        "CKPT_DIR must contain non-empty model.safetensors or model.pt: "
        f"{checkpoint_dir.resolve()}"
    )


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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scannet-root", type=Path)
    parser.add_argument("--ckpt-dir", type=Path)
    parser.add_argument("--scene-list", type=Path)
    parser.add_argument("--scene-limit", type=int, default=10)
    parser.add_argument("--print-missing", action="store_true")
    parser.add_argument("--print-layout", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.print_missing:
        for package_spec in missing_package_specs():
            print(package_spec)
        return

    if args.scannet_root is None or args.ckpt_dir is None:
        raise SystemExit("--scannet-root and --ckpt-dir are required")
    if args.scene_limit < 0:
        raise SystemExit("--scene-limit must be non-negative")

    missing = missing_package_specs()
    if missing:
        raise RuntimeError("Missing Python packages: " + ", ".join(missing))
    scenes = (
        read_scene_list(args.scene_list, args.scene_limit)
        if args.scene_list is not None
        else None
    )
    checkpoint = find_checkpoint(args.ckpt_dir)
    layout = detect_scannet_layout(args.scannet_root, scenes)
    if args.print_layout:
        print(layout)
        return
    print(f"checkpoint={checkpoint}")
    print(f"scannet_layout={layout}")


if __name__ == "__main__":
    main()
