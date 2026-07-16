"""Validate local AutoDL inputs without importing VGGT or PyTorch."""

from __future__ import annotations

import argparse
import importlib.util
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
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "CKPT_DIR must contain model.safetensors or model.pt: "
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


def _processed_layout_exists(root: Path, scenes: Sequence[str] | None) -> bool:
    process_root = root / "process_scannet"
    candidates = (
        [process_root / scene for scene in scenes]
        if scenes
        else [path for path in process_root.iterdir() if path.is_dir()]
        if process_root.is_dir()
        else []
    )
    return any(
        (scene_dir / "color").is_dir() and (scene_dir / "pose").is_dir()
        for scene_dir in candidates
    )


def _raw_layout_exists(root: Path, scenes: Sequence[str] | None) -> bool:
    scans_root = root / "raw_sens" / "scans"
    if not scans_root.is_dir():
        return False
    configured = set(scenes or [])
    return any(
        not configured or sens_file.stem in configured
        for sens_file in scans_root.rglob("*.sens")
    )


def detect_scannet_layout(
    scannet_root: Path,
    scenes: Sequence[str] | None = None,
) -> Literal["processed", "raw"]:
    """Prefer extracted color/pose data, then accept local raw .sens files."""
    if _processed_layout_exists(scannet_root, scenes):
        return "processed"
    if _raw_layout_exists(scannet_root, scenes):
        return "raw"
    scene_text = f" for configured scenes {list(scenes)}" if scenes else ""
    raise FileNotFoundError(
        "SCANNET_ROOT has neither process_scannet/<scene>/{color,pose} "
        f"nor raw_sens/scans/**/*.sens{scene_text}: {scannet_root.resolve()}"
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
