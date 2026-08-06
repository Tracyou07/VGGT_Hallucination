"""Validate an external refiner dataset against the frozen ScanNet split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pre_experiments.camera_refiner_data_construction.dataset import (  # noqa: E402
    validate_dataset_manifest,
)
from pre_experiments.local_global_consistency.split import (  # noqa: E402
    load_split_manifest,
)


def _scene_list(path: Path) -> list[str]:
    scenes = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not scenes or len(scenes) != len(set(scenes)):
        raise ValueError("scene list must contain unique scene IDs")
    return scenes


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=ROOT / "configs" / "scannet50_local_global_split.json",
    )
    parser.add_argument(
        "--scene-list",
        type=Path,
        default=ROOT / "configs" / "fastvggt_scannet50.txt",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    split = load_split_manifest(
        args.split_manifest.resolve(),
        _scene_list(args.scene_list.resolve()),
    )
    report = validate_dataset_manifest(
        args.manifest.resolve(),
        args.dataset_root.resolve(),
        protected_holdout_scenes=split["holdout_scenes"],
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
