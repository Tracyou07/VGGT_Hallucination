"""Export a completed Round 2A run using a strict scalar-only contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DESTINATION = REPO_ROOT / "results" / "local_global_consistency"
DEFAULT_MAX_BYTES = 50 * 1024 * 1024
FILES = (
    "run_metadata.json",
    "window_manifest.json",
    "complete.json",
    "local_observations.csv",
    "local_overlap_scores.csv",
    "prediction_scores_per_frame.csv",
    "gt_validation_per_frame.csv",
    "local_global_summary.csv",
    "local_global_summary.json",
    "reliability_thresholds.json",
)
HIGH_DIM_SUFFIXES = {".npy", ".npz", ".pt", ".pth", ".safetensors", ".bin"}


def _json_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON object: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_numeric_results(
    source: Path,
    destination_root: Path = DEFAULT_DESTINATION,
    max_file_bytes: int = DEFAULT_MAX_BYTES,
) -> Path:
    source = source.expanduser().resolve()
    destination_root = destination_root.expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"source run does not exist: {source}")
    if max_file_bytes < 1:
        raise ValueError("max_file_bytes must be positive")
    unexpected = [
        path.relative_to(source).as_posix()
        for path in source.iterdir()
        if path.is_file() and path.suffix.lower() in HIGH_DIM_SUFFIXES
    ]
    if unexpected:
        raise ValueError(f"unexpected high-dimensional root artifact: {unexpected[0]}")

    metadata = _json_object(source / "run_metadata.json")
    complete = _json_object(source / "complete.json")
    run_id = metadata.get("run_id")
    if not isinstance(run_id, str) or run_id != source.name:
        raise ValueError("run_metadata.json run_id must match source directory")
    if metadata.get("study_name") != "local_global_consistency":
        raise ValueError("source is not a local-global consistency run")
    if complete.get("run_id") != run_id or complete.get("analysis_complete") is not True:
        raise ValueError("Round 2 analysis is not complete")

    candidates = []
    for filename in FILES:
        path = source / filename
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"required scalar result is missing or empty: {path}")
        if path.stat().st_size > max_file_bytes:
            raise ValueError(f"scalar result exceeds size limit: {path}")
        if path.suffix == ".json":
            json.loads(path.read_text(encoding="utf-8"))
        candidates.append(path)

    destination = destination_root / run_id
    if destination.exists():
        raise FileExistsError(f"published run already exists: {destination}")
    destination_root.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{run_id}.", dir=destination_root))
    try:
        manifest_files = []
        for path in candidates:
            target = stage / path.name
            shutil.copy2(path, target)
            manifest_files.append(
                {
                    "path": path.name,
                    "bytes": target.stat().st_size,
                    "sha256": _sha256(target),
                }
            )
        manifest = {
            "schema_version": 1,
            "run_id": run_id,
            "source_git_commit": metadata.get("git_commit"),
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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination-root", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--max-file-mb", type=float, default=50.0)
    args = parser.parse_args(argv)
    destination = export_numeric_results(
        args.source,
        args.destination_root,
        max_file_bytes=int(args.max_file_mb * 1024 * 1024),
    )
    print(f"published={destination}")


if __name__ == "__main__":
    main()
