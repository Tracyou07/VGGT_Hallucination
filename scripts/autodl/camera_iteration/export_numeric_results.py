"""Export a completed Camera Iteration run as compact Git-safe artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import zipfile


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DESTINATION_ROOT = REPO_ROOT / "results" / "camera_iteration"
DEFAULT_MAX_FILE_BYTES = 50 * 1024 * 1024
ROOT_FILES = ("run_metadata.json", "summary.json", "summary.csv")
SELECTION_FILES = (
    "iteration_metrics.json",
    "iteration_metrics.csv",
    "selected_frame_ids.json",
    "complete.json",
    "camera_trace.npz",
)
TRACE_MEMBERS = {
    "frame_ids.npy",
    "pose_enc_by_iteration.npy",
    "raw_pose_enc_by_iteration.npy",
    "pose_delta_by_iteration.npy",
    "delta_norm.npy",
}
CAMERA_TOKEN_MEMBERS = {
    "normalized_camera_tokens.npy",
    "pose_tokens_modulated.npy",
}
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _read_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON object: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _validate_file(path: Path, max_file_bytes: int) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"required result file is missing: {path}")
    size = path.stat().st_size
    if size <= 0:
        raise ValueError(f"required result file is empty: {path}")
    if size > max_file_bytes:
        raise ValueError(
            f"result file exceeds size limit ({size} > {max_file_bytes} bytes): {path}"
        )


def _validate_trace(path: Path) -> None:
    try:
        with zipfile.ZipFile(path) as archive:
            members = set(archive.namelist())
            bad_tokens = members.intersection(CAMERA_TOKEN_MEMBERS)
            if bad_tokens:
                raise ValueError(
                    f"Camera Token arrays are not publishable: {sorted(bad_tokens)}"
                )
            if members != TRACE_MEMBERS:
                raise ValueError(
                    "camera_trace.npz must contain only compact pose trace arrays; "
                    f"found {sorted(members)}"
                )
    except zipfile.BadZipFile as error:
        raise ValueError(f"invalid camera trace NPZ: {path}") from error


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _selection_dirs(source: Path) -> list[Path]:
    return sorted(
        path
        for scene_dir in source.iterdir()
        if scene_dir.is_dir()
        for path in scene_dir.iterdir()
        if path.is_dir() and path.name.startswith("frames_")
    )


def export_numeric_results(
    source: Path,
    destination_root: Path = DEFAULT_DESTINATION_ROOT,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
) -> Path:
    """Validate and export one completed run into an immutable destination."""
    source = source.expanduser().resolve()
    destination_root = destination_root.expanduser().resolve()
    if max_file_bytes < 1:
        raise ValueError("max_file_bytes must be positive")
    if not source.is_dir():
        raise FileNotFoundError(f"source run directory does not exist: {source}")

    metadata_path = source / "run_metadata.json"
    _validate_file(metadata_path, max_file_bytes)
    metadata = _read_object(metadata_path)
    run_id = metadata.get("run_id")
    if not isinstance(run_id, str) or not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError("run_metadata.json has an invalid run_id")

    destination = destination_root / run_id
    if destination.exists():
        raise FileExistsError(f"published run already exists: {destination}")

    candidates: list[tuple[Path, Path]] = []
    for filename in ROOT_FILES:
        path = source / filename
        _validate_file(path, max_file_bytes)
        candidates.append((path, Path(filename)))

    selections = _selection_dirs(source)
    if not selections:
        raise ValueError(f"source run has no scene/frame selections: {source}")
    for selection in selections:
        relative_dir = selection.relative_to(source)
        complete_path = selection / "complete.json"
        _validate_file(complete_path, max_file_bytes)
        completion = _read_object(complete_path)
        if completion.get("run_id") != run_id:
            raise ValueError(f"selection complete.json run_id mismatch: {complete_path}")
        for filename in SELECTION_FILES:
            path = selection / filename
            _validate_file(path, max_file_bytes)
            if filename == "camera_trace.npz":
                _validate_trace(path)
            candidates.append((path, relative_dir / filename))

    destination_root.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{run_id}.", dir=destination_root))
    try:
        manifest_files = []
        for source_path, relative_path in sorted(
            candidates, key=lambda pair: pair[1].as_posix()
        ):
            target = stage / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, target)
            manifest_files.append(
                {
                    "path": relative_path.as_posix(),
                    "bytes": target.stat().st_size,
                    "sha256": _sha256(target),
                }
            )

        manifest = {
            "schema_version": 1,
            "run_id": run_id,
            "source_git_commit": metadata.get("git_commit"),
            "file_count": len(manifest_files),
            "total_bytes": sum(entry["bytes"] for entry in manifest_files),
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--destination-root",
        type=Path,
        default=DEFAULT_DESTINATION_ROOT,
    )
    parser.add_argument("--max-file-mb", type=float, default=50.0)
    args = parser.parse_args(argv)
    if args.max_file_mb <= 0:
        parser.error("--max-file-mb must be positive")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    destination = export_numeric_results(
        args.source,
        args.destination_root,
        max_file_bytes=int(args.max_file_mb * 1024 * 1024),
    )
    manifest = _read_object(destination / "publish_manifest.json")
    print(f"published={destination}")
    print(f"files={manifest['file_count']} bytes={manifest['total_bytes']}")


if __name__ == "__main__":
    main()
