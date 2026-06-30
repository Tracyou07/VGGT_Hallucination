"""Extract ScanNet .sens files into the layout used by hallucination eval.

Output per scene:
  color/*.jpg
  depth/*.png
  pose/*.txt
  intrinsic/*.txt
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path


def read_scenes(scene_list: Path, scene_limit: int) -> list[str]:
    scenes = []
    for line in scene_list.read_text(encoding="utf-8").splitlines():
        item = line.strip()
        if item and not item.startswith("#"):
            scenes.append(item)
    if scene_limit > 0:
        scenes = scenes[:scene_limit]
    return scenes


def ensure_scannet_repo(scannet_repo: Path) -> Path:
    vendored_dir = Path(__file__).resolve().parent / "sensreader_py3"
    if (vendored_dir / "SensorData.py").exists():
        print(f"Using vendored SensReader: {vendored_dir}")
        return vendored_dir

    sensreader_dir = scannet_repo / "SensReader" / "python"
    if sensreader_dir.exists():
        return sensreader_dir
    scannet_repo.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "git",
            "clone",
            "https://github.com/ScanNet/ScanNet.git",
            "--depth",
            "1",
            str(scannet_repo),
        ],
        check=True,
    )
    return sensreader_dir


def patch_sensreader(src_dir: Path, dst_dir: Path) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for py_file in src_dir.glob("*.py"):
        content = py_file.read_text(encoding="utf-8", errors="replace")
        content = re.sub(
            r"^(\s*)print ([^(].+)$",
            lambda m: m.group(1) + "print(" + m.group(2).rstrip() + ")",
            content,
            flags=re.MULTILINE,
        )
        content = re.sub(
            r"''.join\(struct\.unpack\('c'\*(\w+),\s*f\.read\(\1\)\)\)",
            r"f.read(\1).decode('utf-8')",
            content,
        )
        content = content.replace("self.color_data = ''", "self.color_data = b''")
        content = content.replace("self.depth_data = ''", "self.depth_data = b''")
        (dst_dir / py_file.name).write_text(content, encoding="utf-8")
    (dst_dir / "__init__.py").write_text("", encoding="utf-8")
    return dst_dir


def count_files(path: Path, suffixes: tuple[str, ...]) -> int:
    if not path.exists():
        return 0
    return sum(1 for p in path.iterdir() if p.suffix.lower() in suffixes)


def extract_scene(sensor_data_cls, sens_file: Path, out_scene: Path, export_depth: bool) -> None:
    color_dir = out_scene / "color"
    depth_dir = out_scene / "depth"
    pose_dir = out_scene / "pose"
    intrinsic_dir = out_scene / "intrinsic"

    if count_files(color_dir, (".jpg", ".png")) > 0 and count_files(pose_dir, (".txt",)) > 0:
        if not export_depth or count_files(depth_dir, (".png",)) > 0:
            print(f"[extract] skip existing {out_scene.name}")
            return

    out_scene.mkdir(parents=True, exist_ok=True)
    sd = sensor_data_cls(str(sens_file))
    print(f"[extract] {out_scene.name}: frames={len(sd.frames)}")

    sd.export_color_images(str(color_dir))
    sd.export_poses(str(pose_dir))
    try:
        sd.export_intrinsics(str(intrinsic_dir))
    except Exception as exc:
        print(f"[extract] warning: intrinsic export failed for {out_scene.name}: {exc}")

    if export_depth:
        sd.export_depth_images(str(depth_dir))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract ScanNet .sens scenes.")
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--scene-limit", type=int, default=10)
    parser.add_argument("--scannet-repo", type=Path, default=Path("/root/autodl-tmp/ScanNet"))
    parser.add_argument("--patched-dir", type=Path, default=Path("/tmp/scannet_sensreader_py3"))
    parser.add_argument("--export-depth", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenes = read_scenes(args.scene_list, args.scene_limit)
    sensreader_dir = ensure_scannet_repo(args.scannet_repo)
    if sensreader_dir.name == "sensreader_py3":
        patched_dir = sensreader_dir
    else:
        patched_dir = patch_sensreader(sensreader_dir, args.patched_dir)

    sys.path.insert(0, str(patched_dir))
    from SensorData import SensorData  # type: ignore

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for scene in scenes:
        sens_file = args.raw_dir / scene / f"{scene}.sens"
        if not sens_file.exists():
            print(f"[extract] missing {sens_file}")
            continue
        try:
            extract_scene(SensorData, sens_file, args.out_dir / scene, args.export_depth)
        except Exception as exc:
            print(f"[extract] ERROR {scene}: {exc}")
            shutil.rmtree(args.out_dir / scene, ignore_errors=True)


if __name__ == "__main__":
    main()
