"""Export authenticated numeric adaptive-alpha calibration or holdout runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Sequence

from pre_experiments.camera_hidden_state_attribution.artifacts import (
    canonical_digest,
)


COMMON_FILES = (
    "run_metadata.json",
    "complete.json",
    "frozen_selector.json",
    "summary.json",
)
CALIBRATION_FILES = (
    *COMMON_FILES,
    "scene_features.csv",
    "loocv_per_scene.csv",
    "oracle_alpha_per_scene.csv",
)
HOLDOUT_FILES = (
    *COMMON_FILES,
    "frozen_replacement.json",
    "per_scene.csv",
    "per_frame.csv",
)


def _json_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def export_adaptive_alpha(source: Path, destination_root: Path) -> Path:
    source = source.resolve()
    destination_root = destination_root.resolve()
    metadata = _json_object(source / "run_metadata.json")
    complete = _json_object(source / "complete.json")
    partition = str(metadata.get("partition", ""))
    allowed = (
        CALIBRATION_FILES if partition == "calibration" else HOLDOUT_FILES
    )
    if partition not in {"calibration", "holdout"}:
        raise ValueError("adaptive-alpha export partition is invalid")
    unexpected = [
        path.name
        for path in source.iterdir()
        if path.is_file() and path.name not in allowed
    ]
    if unexpected:
        raise ValueError(f"unexpected root artifact: {sorted(unexpected)[0]}")
    for name in allowed:
        path = source / name
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"required numeric artifact missing: {path}")
        if path.suffix == ".json":
            _json_object(path)
    selector = _json_object(source / "frozen_selector.json")
    selector_digest = selector.pop("selector_digest", None)
    if (
        metadata.get("run_id") != source.name
        or metadata.get("study_name") != "camera_hidden_adaptive_alpha"
        or complete.get("run_id") != source.name
        or complete.get("partition") != partition
        or complete.get("protocol_complete") is not True
        or complete.get("analysis_complete") is not True
        or not isinstance(selector_digest, str)
        or selector_digest != canonical_digest(selector)
        or complete.get("selector_digest") != selector_digest
    ):
        raise ValueError("adaptive-alpha run or selector digest mismatch")
    if partition == "holdout":
        frozen = _json_object(source / "frozen_replacement.json")
        frozen_digest = frozen.pop("frozen_digest", None)
        if (
            not isinstance(frozen_digest, str)
            or frozen_digest != canonical_digest(frozen)
            or complete.get("frozen_replacement_digest") != frozen_digest
        ):
            raise ValueError("adaptive-alpha replacement digest mismatch")
    destination = destination_root / source.name
    if destination.exists():
        raise FileExistsError(f"published run already exists: {destination}")
    destination.mkdir(parents=True)
    for name in allowed:
        shutil.copy2(source / name, destination / name)
    return destination


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination-root", type=Path, required=True)
    args = parser.parse_args(argv)
    print(
        f"published={export_adaptive_alpha(args.source, args.destination_root)}"
    )


if __name__ == "__main__":
    main()
