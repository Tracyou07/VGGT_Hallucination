"""Publish authenticated ScanNet-50 calibration and holdout numeric evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Sequence

from pre_experiments.local_global_consistency.split import load_split_manifest
from pre_experiments.local_global_consistency.thresholds import (
    load_frozen_thresholds,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DESTINATION = (
    REPO_ROOT / "results" / "local_global_consistency" / "scannet50"
)
DEFAULT_MAX_BYTES = 50 * 1024 * 1024
CALIBRATION_FILES = (
    "run_metadata.json",
    "window_manifest.json",
    "complete.json",
    "frozen_reliability_thresholds.json",
    "calibration_prediction_scores_per_frame.csv",
    "calibration_gt_validation_per_frame.csv",
    "calibration_summary.csv",
    "calibration_summary.json",
)
HOLDOUT_FILES = (
    "run_metadata.json",
    "window_manifest.json",
    "complete.json",
    "holdout_complete.json",
    "holdout_prediction_scores_per_frame.csv",
    "holdout_gt_validation_per_frame.csv",
    "holdout_per_scene_summary.csv",
    "holdout_aggregate_summary.csv",
    "holdout_aggregate_summary.json",
)


def _json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON file: {path}") from error


def _json_object(path: Path) -> dict[str, object]:
    value = _json(path)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_scannet50() -> list[str]:
    path = REPO_ROOT / "configs" / "fastvggt_scannet50.txt"
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _validate_run(
    source: Path,
    *,
    partition: str,
    allowed_files: Sequence[str],
    split_digest: str,
    source_run_id: str,
    max_file_bytes: int,
) -> tuple[dict[str, object], dict[str, object], list[Path]]:
    if not source.is_dir():
        raise FileNotFoundError(f"{partition} run does not exist: {source}")
    allowed = set(allowed_files)
    unexpected = [
        path.name
        for path in source.iterdir()
        if path.is_file() and path.name not in allowed
    ]
    if unexpected:
        raise ValueError(
            f"unexpected root artifact in {partition} run: {sorted(unexpected)[0]}"
        )
    candidates = []
    for filename in allowed_files:
        path = source / filename
        if not path.is_file() or path.is_symlink() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"required numeric result is missing: {path}")
        if path.stat().st_size > max_file_bytes:
            raise ValueError(f"numeric result exceeds size limit: {path}")
        if path.suffix == ".json":
            _json(path)
        candidates.append(path)

    metadata = _json_object(source / "run_metadata.json")
    complete = _json_object(source / "complete.json")
    run_id = metadata.get("run_id")
    if not isinstance(run_id, str) or run_id != source.name:
        raise ValueError(f"{partition} run ID must match its source directory")
    if (
        metadata.get("study_name") != "local_global_consistency"
        or metadata.get("partition") != partition
        or metadata.get("protocol_complete") is not True
    ):
        raise ValueError(f"source is not a formal {partition} run")
    if (
        complete.get("run_id") != run_id
        or complete.get("partition") != partition
        or complete.get("analysis_complete") is not True
    ):
        raise ValueError(f"{partition} analysis is not complete")
    for field, expected in (
        ("split_digest", split_digest),
        ("source_run_id", source_run_id),
    ):
        if metadata.get(field) != expected or complete.get(field) != expected:
            raise ValueError(f"{partition} {field} provenance mismatch")
    return metadata, complete, candidates


def export_numeric_results(
    calibration_source: Path,
    holdout_source: Path,
    split_manifest: Path,
    destination_root: Path = DEFAULT_DESTINATION,
    max_file_bytes: int = DEFAULT_MAX_BYTES,
) -> Path:
    """Copy only authenticated scalar evidence into one repository directory."""
    calibration_source = calibration_source.expanduser().resolve()
    holdout_source = holdout_source.expanduser().resolve()
    split_manifest = split_manifest.expanduser().resolve()
    destination_root = destination_root.expanduser().resolve()
    if max_file_bytes < 1:
        raise ValueError("max_file_bytes must be positive")
    if not split_manifest.is_file() or split_manifest.is_symlink():
        raise FileNotFoundError(f"split manifest is missing: {split_manifest}")
    if split_manifest.stat().st_size > max_file_bytes:
        raise ValueError("split manifest exceeds size limit")
    split = load_split_manifest(split_manifest, _expected_scannet50())
    split_digest = str(split["split_digest"])
    source_run_id = str(split["source_run_id"])

    calibration_metadata, calibration_complete, calibration_files = _validate_run(
        calibration_source,
        partition="calibration",
        allowed_files=CALIBRATION_FILES,
        split_digest=split_digest,
        source_run_id=source_run_id,
        max_file_bytes=max_file_bytes,
    )
    threshold_path = calibration_source / "frozen_reliability_thresholds.json"
    thresholds = load_frozen_thresholds(
        threshold_path,
        expected_split_digest=split_digest,
        expected_source_run_id=source_run_id,
    )
    threshold_digest = thresholds["threshold_digest"]
    if calibration_complete.get("threshold_digest") != threshold_digest:
        raise ValueError("calibration completion threshold digest mismatch")
    if calibration_complete.get("scenes") != split["calibration_scenes"]:
        raise ValueError("calibration completion scenes do not match the split")

    holdout_metadata, holdout_complete, holdout_files = _validate_run(
        holdout_source,
        partition="holdout",
        allowed_files=HOLDOUT_FILES,
        split_digest=split_digest,
        source_run_id=source_run_id,
        max_file_bytes=max_file_bytes,
    )
    holdout_marker = _json_object(holdout_source / "holdout_complete.json")
    holdout_summary = _json_object(
        holdout_source / "holdout_aggregate_summary.json"
    )
    for payload in (holdout_complete, holdout_marker, holdout_summary):
        if payload.get("threshold_digest") != threshold_digest:
            raise ValueError("holdout threshold digest is missing or inconsistent")
        if payload.get("scenes") != split["holdout_scenes"]:
            raise ValueError("holdout summary scenes do not match the split")
    if set(thresholds["calibration_scenes"]) != set(split["calibration_scenes"]):
        raise ValueError("threshold calibration scenes do not match the split")
    calibration_run_id = str(calibration_metadata["run_id"])
    holdout_run_id = str(holdout_metadata["run_id"])
    destination = destination_root / f"{calibration_run_id}__{holdout_run_id}"
    if destination.exists():
        raise FileExistsError(f"published result already exists: {destination}")
    destination_root.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(
            prefix=f".{calibration_run_id}__{holdout_run_id}.",
            dir=destination_root,
        )
    )
    try:
        manifest_files = []
        for subdirectory, candidates in (
            ("calibration", calibration_files),
            ("holdout", holdout_files),
        ):
            target_dir = stage / subdirectory
            target_dir.mkdir()
            for source_path in candidates:
                target = target_dir / source_path.name
                shutil.copy2(source_path, target)
                manifest_files.append(
                    {
                        "path": target.relative_to(stage).as_posix(),
                        "bytes": target.stat().st_size,
                        "sha256": _sha256(target),
                    }
                )
        split_target = stage / "scannet50_local_global_split.json"
        shutil.copy2(split_manifest, split_target)
        manifest_files.append(
            {
                "path": split_target.name,
                "bytes": split_target.stat().st_size,
                "sha256": _sha256(split_target),
            }
        )
        manifest = {
            "schema_version": 2,
            "calibration_run_id": calibration_run_id,
            "holdout_run_id": holdout_run_id,
            "source_run_id": source_run_id,
            "split_digest": split_digest,
            "threshold_digest": threshold_digest,
            "source_git_commits": {
                "calibration": calibration_metadata.get("git_commit"),
                "holdout": holdout_metadata.get("git_commit"),
            },
            "file_count": len(manifest_files),
            "total_bytes": sum(item["bytes"] for item in manifest_files),
            "files": manifest_files,
        }
        (stage / "publish_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(stage, destination)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return destination


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-run", type=Path, required=True)
    parser.add_argument("--holdout-run", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--destination-root", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--max-file-mb", type=float, default=50.0)
    args = parser.parse_args(argv)
    destination = export_numeric_results(
        args.calibration_run,
        args.holdout_run,
        args.split_manifest,
        args.destination_root,
        max_file_bytes=int(args.max_file_mb * 1024 * 1024),
    )
    print(f"published={destination}")


if __name__ == "__main__":
    main()
