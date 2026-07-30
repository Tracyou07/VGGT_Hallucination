"""Export only authenticated numeric Camera hidden causal-preference results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Sequence

from pre_experiments.camera_hidden_state_attribution.artifacts import (
    canonical_digest,
)


ALLOWED_FILES = (
    "run_metadata.json",
    "complete.json",
    "per_position.csv",
    "direct_checks.csv",
    "frozen_causal_normalization.json",
    "summary.json",
)


def _json_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def export_causal_preference(
    source: Path,
    destination_root: Path,
) -> Path:
    source = source.resolve()
    destination_root = destination_root.resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"run does not exist: {source}")
    unexpected = [
        path.name
        for path in source.iterdir()
        if path.is_file() and path.name not in ALLOWED_FILES
    ]
    if unexpected:
        raise ValueError(f"unexpected root artifact: {sorted(unexpected)[0]}")
    for name in ALLOWED_FILES:
        path = source / name
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(
                f"required numeric artifact is missing: {path}"
            )

    metadata = _json_object(source / "run_metadata.json")
    complete = _json_object(source / "complete.json")
    frozen = _json_object(
        source / "frozen_causal_normalization.json"
    )
    summary = _json_object(source / "summary.json")
    frozen_unsigned = dict(frozen)
    frozen_digest = frozen_unsigned.pop("frozen_digest", None)
    if (
        metadata.get("run_id") != source.name
        or metadata.get("study_name")
        != "camera_hidden_causal_preference"
        or metadata.get("partition") != "holdout"
        or metadata.get("protocol_complete") is not True
        or complete.get("run_id") != source.name
        or complete.get("partition") != "holdout"
        or complete.get("analysis_complete") is not True
        or complete.get("protocol_complete") is not True
        or summary.get("partition") != "holdout"
        or not isinstance(frozen_digest, str)
        or frozen_digest != canonical_digest(frozen_unsigned)
        or complete.get("frozen_digest") != frozen_digest
    ):
        raise ValueError("source is not a complete formal causal-preference run")

    destination = destination_root / source.name
    if destination.exists():
        raise FileExistsError(
            f"published run already exists: {destination}"
        )
    destination.mkdir(parents=True)
    for name in ALLOWED_FILES:
        shutil.copy2(source / name, destination / name)
    return destination


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination-root", type=Path, required=True)
    args = parser.parse_args(argv)
    destination = export_causal_preference(
        args.source,
        args.destination_root,
    )
    print(f"published={destination}")


if __name__ == "__main__":
    main()
