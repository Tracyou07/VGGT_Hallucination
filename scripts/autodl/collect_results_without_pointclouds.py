"""Collect lightweight VGGT hallucination results without point-cloud files."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


ALLOWED_FILENAMES = {
    "metrics.json",
    "selected_frame_ids.json",
    "predicted_cameras.npz",
    "trajectory.png",
    "summary.csv",
    "summary.json",
}

POINTCLOUD_SUFFIXES = {
    ".ply",
    ".pcd",
    ".pts",
    ".xyz",
    ".las",
    ".laz",
    ".obj",
    ".off",
    ".glb",
    ".gltf",
}


def should_copy(path: Path) -> bool:
    if path.suffix.lower() in POINTCLOUD_SUFFIXES:
        return False
    return path.name in ALLOWED_FILENAMES


def collect_results(src: Path, dst: Path) -> dict[str, Any]:
    src = src.resolve()
    dst = dst.resolve()
    if not src.exists():
        raise FileNotFoundError(f"Result source does not exist: {src}")

    copied: list[str] = []
    skipped_pointcloud: list[str] = []
    skipped_other = 0
    for path in sorted(src.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(src)
        if path.suffix.lower() in POINTCLOUD_SUFFIXES:
            skipped_pointcloud.append(rel.as_posix())
            continue
        if not should_copy(path):
            skipped_other += 1
            continue
        out_path = dst / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, out_path)
        copied.append(rel.as_posix())

    manifest = {
        "source": str(src),
        "destination": str(dst),
        "copied_files": len(copied),
        "skipped_pointcloud_files": len(skipped_pointcloud),
        "skipped_other_files": skipped_other,
        "copied": copied,
        "skipped_pointcloud": skipped_pointcloud,
    }
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "manifest_without_pointclouds.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--dst", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = collect_results(args.src, args.dst)
    print(
        "[collect] copied={copied_files} skipped_pointcloud={skipped_pointcloud_files} "
        "skipped_other={skipped_other_files}".format(**manifest)
    )
    print(f"[collect] destination={manifest['destination']}")


if __name__ == "__main__":
    main()
