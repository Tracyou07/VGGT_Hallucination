"""Resumable, RGB-only CO3Dv2 subset construction for AutoDL."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import gzip
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
from typing import Iterable, Iterator, Mapping, Sequence, TextIO
import zipfile
import zlib

DEFAULT_BASE_URL = "https://dl.fbaipublicfiles.com/co3dv2_231130"
DEFAULT_OUTPUT_ROOT = Path("/root/autodl-tmp/datasets/co3dv2_2050")
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CATEGORY_FILE = REPO_ROOT / "configs" / "co3d_train41.txt"
STATE_SCHEMA_VERSION = 1
MANIFEST_SCHEMA_VERSION = 1
_CATEGORY_PATTERN = re.compile(r"[a-z0-9]+")
_ARCHIVE_PATTERN = re.compile(r"_(\d{3})\.zip$")
_REQUIRED_METADATA = ("frame_annotations.jgz", "sequence_annotations.jgz")
_OPTIONAL_METADATA = ("set_lists.json",)


@dataclass(frozen=True)
class SequenceCandidate:
    """One sequence with valid RGB paths and camera-pose annotations."""

    sequence_name: str
    quality_score: float
    image_paths: tuple[str, ...]

    @property
    def valid_frame_count(self) -> int:
        return len(self.image_paths)


@dataclass(frozen=True)
class ArchiveImage:
    """A ZIP member paired with its canonical CO3D destination path."""

    source_member: str
    relative_path: str
    uncompressed_size: int


class ArchiveNotFoundError(RuntimeError):
    """Raised when a numbered category archive does not exist upstream."""


def _atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def archive_url(base_url: str, category: str, archive_index: int) -> str:
    """Build an official numbered CO3Dv2 category archive URL."""
    _validate_category(category)
    if archive_index < 0 or archive_index > 999:
        raise ValueError("archive_index must be between 0 and 999")
    return f"{base_url.rstrip('/')}/{category}_{archive_index:03d}.zip"


def build_curl_command(url: str, partial_path: Path, curl_bin: str = "curl") -> list[str]:
    """Return a visible, retrying curl command that resumes a partial ZIP."""
    return [
        curl_bin,
        "--fail",
        "--location",
        "--show-error",
        "--progress-bar",
        "--retry",
        "20",
        "--retry-delay",
        "5",
        "--retry-all-errors",
        "--connect-timeout",
        "30",
        "--speed-time",
        "180",
        "--speed-limit",
        "1024",
        "--continue-at",
        "-",
        "--output",
        str(partial_path),
        url,
    ]


def load_categories(path: Path) -> tuple[str, ...]:
    """Load an ordered category list while rejecting duplicates."""
    categories = tuple(
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    if not categories:
        raise ValueError(f"category file is empty: {path}")
    for category in categories:
        _validate_category(category)
    if len(set(categories)) != len(categories):
        raise ValueError(f"category file contains duplicates: {path}")
    return categories


def load_eligible_sequences(
    category_dir: Path,
    *,
    category: str,
    min_frames: int,
    min_quality: float,
) -> dict[str, SequenceCandidate]:
    """Read CO3D annotations and retain sequences suitable for pose training."""
    if min_frames <= 0:
        raise ValueError("min_frames must be positive")
    if not math.isfinite(min_quality):
        raise ValueError("min_quality must be finite")
    qualities: dict[str, float] = {}
    for row in _iter_jgz_rows(
        category_dir / "sequence_annotations.jgz",
        keys=("sequences", "sequence_annotations"),
    ):
        name = row.get("sequence_name")
        score = row.get("viewpoint_quality_score")
        if not isinstance(name, str):
            continue
        try:
            numeric_score = float(score)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric_score) and numeric_score >= min_quality:
            qualities[name] = numeric_score

    paths_by_sequence: dict[str, set[str]] = defaultdict(set)
    for row in _iter_jgz_rows(
        category_dir / "frame_annotations.jgz",
        keys=("frames", "frame_annotations"),
    ):
        sequence_name = row.get("sequence_name")
        image = row.get("image")
        if not isinstance(sequence_name, str) or not isinstance(image, Mapping):
            continue
        if sequence_name not in qualities:
            continue
        if not _has_valid_pose(row.get("viewpoint")):
            continue
        raw_path = image.get("path")
        if not isinstance(raw_path, str):
            continue
        try:
            canonical = _canonical_image_path(raw_path, category, sequence_name)
        except ValueError:
            continue
        paths_by_sequence[sequence_name].add(canonical)

    candidates: dict[str, SequenceCandidate] = {}
    for sequence_name in sorted(paths_by_sequence):
        quality = qualities.get(sequence_name)
        image_paths = tuple(sorted(paths_by_sequence[sequence_name]))
        if quality is None or quality < min_quality or len(image_paths) < min_frames:
            continue
        candidates[sequence_name] = SequenceCandidate(
            sequence_name=sequence_name,
            quality_score=quality,
            image_paths=image_paths,
        )
    return candidates


def inspect_data_archive(
    archive_path: Path,
    category: str,
    candidates: Mapping[str, SequenceCandidate],
) -> dict[str, tuple[ArchiveImage, ...]]:
    """Find complete eligible RGB sequences in one local category archive."""
    expected: dict[str, str] = {}
    for sequence_name, candidate in candidates.items():
        for image_path in candidate.image_paths:
            expected[image_path] = sequence_name

    found: dict[str, dict[str, ArchiveImage]] = defaultdict(dict)
    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            if info.is_dir() or info.file_size <= 0:
                continue
            try:
                canonical = _canonical_archive_image(info.filename, category)
            except ValueError:
                continue
            sequence_name = expected.get(canonical)
            if sequence_name is None:
                continue
            found[sequence_name][canonical] = ArchiveImage(
                source_member=info.filename,
                relative_path=canonical,
                uncompressed_size=info.file_size,
            )

    complete: dict[str, tuple[ArchiveImage, ...]] = {}
    for sequence_name, records in found.items():
        if len(records) != candidates[sequence_name].valid_frame_count:
            continue
        complete[sequence_name] = tuple(records[path] for path in sorted(records))
    return complete


def select_archive_sequences(
    *,
    category: str,
    available: Mapping[str, Sequence[ArchiveImage]],
    needed: int,
    seed: int,
    excluded: set[str],
) -> tuple[str, ...]:
    """Select a stable subset from the current archive."""
    if needed < 0:
        raise ValueError("needed must be non-negative")
    names = (name for name in available if name not in excluded)
    ordered = sorted(names, key=lambda name: _selection_key(seed, category, name))
    return tuple(ordered[:needed])


def extract_archive_images(
    archive_path: Path,
    output_root: Path,
    selected: Mapping[str, Sequence[ArchiveImage]],
) -> dict[str, int]:
    """Extract selected image members without trusting ZIP member paths."""
    output_root.mkdir(parents=True, exist_ok=True)
    root = output_root.resolve()
    counts: dict[str, int] = {}
    with zipfile.ZipFile(archive_path) as archive:
        for sequence_name, records in selected.items():
            count = 0
            for record in records:
                _safe_parts(record.source_member)
                relative_parts = _safe_parts(record.relative_path)
                destination = output_root.joinpath(*relative_parts)
                resolved_destination = destination.resolve()
                if not resolved_destination.is_relative_to(root):
                    raise ValueError(f"unsafe extraction path: {record.relative_path}")
                info = archive.getinfo(record.source_member)
                if info.file_size != record.uncompressed_size or info.file_size <= 0:
                    raise ValueError(f"archive member changed: {record.source_member}")
                destination.parent.mkdir(parents=True, exist_ok=True)
                if _file_matches_zip_info(destination, info):
                    count += 1
                    continue
                temporary = destination.with_suffix(destination.suffix + ".part")
                with archive.open(info) as source, temporary.open("wb") as target:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
                if not _file_matches_zip_info(temporary, info):
                    temporary.unlink(missing_ok=True)
                    raise RuntimeError(f"extracted image failed CRC validation: {record.relative_path}")
                temporary.replace(destination)
                count += 1
            counts[sequence_name] = count
    return counts


def build_dataset_manifest(
    *,
    categories: Sequence[str],
    category_states: Mapping[str, Mapping[str, object]],
    sequences_per_category: int,
    min_frames: int,
    min_quality: float,
    seed: int,
    source_base_url: str,
) -> dict[str, object]:
    """Build a strict final manifest and authenticate its sequence selection."""
    sequences: list[dict[str, object]] = []
    for category in categories:
        state = category_states.get(category)
        if state is None:
            raise ValueError(f"missing state for category {category}")
        selected = state.get("selected")
        if not state.get("completed") or not isinstance(selected, list):
            raise ValueError(f"category {category} is incomplete")
        if len(selected) != sequences_per_category:
            raise ValueError(
                f"category {category} must contain exactly {sequences_per_category} sequences"
            )
        names: set[str] = set()
        for raw_entry in selected:
            if not isinstance(raw_entry, Mapping):
                raise ValueError(f"category {category} has an invalid sequence entry")
            name = raw_entry.get("sequence_name")
            if not isinstance(name, str) or name in names:
                raise ValueError(f"category {category} has duplicate or invalid sequence names")
            names.add(name)
            valid_count = int(raw_entry.get("valid_frame_count", 0))
            extracted_count = int(raw_entry.get("extracted_frame_count", 0))
            quality = float(raw_entry.get("quality_score", float("nan")))
            if valid_count < min_frames or extracted_count != valid_count:
                raise ValueError(f"category {category}/{name} has incomplete RGB frames")
            if not math.isfinite(quality) or quality < min_quality:
                raise ValueError(f"category {category}/{name} fails the quality threshold")
            sequences.append(
                {
                    "category": category,
                    "sequence_name": name,
                    "relative_image_dir": f"{category}/{name}/images",
                    "quality_score": quality,
                    "valid_frame_count": valid_count,
                    "source_archive": str(raw_entry.get("source_archive", "")),
                }
            )

    selection_payload = {
        "categories": list(categories),
        "sequences_per_category": sequences_per_category,
        "min_frames": min_frames,
        "min_quality": min_quality,
        "seed": seed,
        "sequences": sequences,
    }
    selection_sha256 = hashlib.sha256(
        json.dumps(selection_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "dataset": "CO3Dv2",
        "source_base_url": source_base_url.rstrip("/"),
        "license_url": "https://github.com/facebookresearch/co3d/blob/main/LICENSE",
        "layout": "<root>/<category>/<sequence>/images plus category annotations",
        "category_count": len(categories),
        "sequences_per_category": sequences_per_category,
        "sequence_count": len(sequences),
        "min_valid_rgb_pose_frames": min_frames,
        "min_viewpoint_quality_score": min_quality,
        "selection_seed": seed,
        "selection_sha256": selection_sha256,
        "sequences": sequences,
    }


def _validate_category(category: str) -> None:
    if _CATEGORY_PATTERN.fullmatch(category) is None:
        raise ValueError(f"invalid CO3D category: {category!r}")


class _JsonTextStream:
    """Incrementally decode JSON values while keeping a bounded text buffer."""

    def __init__(self, handle: TextIO, *, chunk_size: int = 64 * 1024) -> None:
        self.handle = handle
        self.chunk_size = chunk_size
        self.buffer = ""
        self.position = 0
        self.eof = False
        self.decoder = json.JSONDecoder()

    def _fill(self) -> bool:
        if self.position:
            self.buffer = self.buffer[self.position :]
            self.position = 0
        chunk = self.handle.read(self.chunk_size)
        if not chunk:
            self.eof = True
            return False
        self.buffer += chunk
        return True

    def peek(self) -> str:
        while True:
            while self.position < len(self.buffer) and self.buffer[self.position].isspace():
                self.position += 1
            if self.position < len(self.buffer):
                return self.buffer[self.position]
            if self.eof or not self._fill():
                return ""

    def take(self, expected: str) -> None:
        actual = self.peek()
        if actual != expected:
            raise ValueError(f"expected {expected!r} in JSON stream, found {actual!r}")
        self.position += 1

    def value(self) -> object:
        while True:
            if not self.peek():
                raise ValueError("unexpected end of JSON stream")
            try:
                value, end = self.decoder.raw_decode(self.buffer, self.position)
            except json.JSONDecodeError as error:
                if self.eof or not self._fill():
                    raise ValueError("invalid JSON annotation payload") from error
                continue
            self.position = end
            return value


def _iter_stream_array(stream: _JsonTextStream) -> Iterator[Mapping[str, object]]:
    stream.take("[")
    if stream.peek() == "]":
        stream.take("]")
        return
    while True:
        row = stream.value()
        if not isinstance(row, Mapping):
            raise ValueError("CO3D annotation array must contain objects")
        yield row
        delimiter = stream.peek()
        if delimiter == ",":
            stream.take(",")
            continue
        if delimiter == "]":
            stream.take("]")
            return
        raise ValueError(f"invalid JSON array delimiter: {delimiter!r}")


def _skip_stream_value(stream: _JsonTextStream) -> None:
    marker = stream.peek()
    if marker == "[":
        stream.take("[")
        if stream.peek() == "]":
            stream.take("]")
            return
        while True:
            _skip_stream_value(stream)
            delimiter = stream.peek()
            if delimiter == ",":
                stream.take(",")
                continue
            stream.take("]")
            return
    if marker == "{":
        stream.take("{")
        if stream.peek() == "}":
            stream.take("}")
            return
        while True:
            if not isinstance(stream.value(), str):
                raise ValueError("JSON object key must be a string")
            stream.take(":")
            _skip_stream_value(stream)
            delimiter = stream.peek()
            if delimiter == ",":
                stream.take(",")
                continue
            stream.take("}")
            return
    stream.value()


def _iter_jgz_rows(path: Path, *, keys: Sequence[str]) -> Iterator[Mapping[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(f"missing CO3D annotation: {path}")
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        stream = _JsonTextStream(handle)
        marker = stream.peek()
        if marker == "[":
            yield from _iter_stream_array(stream)
            return
        if marker != "{":
            raise ValueError("CO3D annotation payload must contain an array of objects")

        stream.take("{")
        if stream.peek() == "}":
            raise ValueError("CO3D annotation object does not contain a row array")
        while True:
            key = stream.value()
            if not isinstance(key, str):
                raise ValueError("JSON object key must be a string")
            stream.take(":")
            if key in keys:
                if stream.peek() != "[":
                    raise ValueError(f"CO3D annotation field {key!r} must be an array")
                yield from _iter_stream_array(stream)
                return
            _skip_stream_value(stream)
            delimiter = stream.peek()
            if delimiter == ",":
                stream.take(",")
                continue
            if delimiter == "}":
                break
            raise ValueError(f"invalid JSON object delimiter: {delimiter!r}")
        raise ValueError("CO3D annotation object does not contain a row array")


def _finite_vector(value: object, length: int) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        return False
    try:
        return all(math.isfinite(float(item)) for item in value)
    except (TypeError, ValueError):
        return False


def _has_valid_pose(viewpoint: object) -> bool:
    if not isinstance(viewpoint, Mapping):
        return False
    rotation = viewpoint.get("R")
    translation = viewpoint.get("T")
    if not isinstance(rotation, (list, tuple)) or len(rotation) != 3:
        return False
    return all(_finite_vector(row, 3) for row in rotation) and _finite_vector(
        translation, 3
    )


def _safe_parts(raw_path: str) -> tuple[str, ...]:
    normalized = raw_path.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or not path.parts or any(part in ("", "..") for part in path.parts):
        raise ValueError(f"unsafe archive path: {raw_path}")
    return path.parts


def _canonical_image_path(raw_path: str, category: str, sequence_name: str) -> str:
    parts = _safe_parts(raw_path)
    for index, part in enumerate(parts):
        suffix = parts[index:]
        if (
            part == category
            and len(suffix) >= 4
            and suffix[1] == sequence_name
            and suffix[2] == "images"
        ):
            return "/".join(suffix)
    raise ValueError(f"not a canonical image path for {category}/{sequence_name}: {raw_path}")


def _canonical_archive_image(member_name: str, category: str) -> str:
    parts = _safe_parts(member_name)
    for index, part in enumerate(parts):
        suffix = parts[index:]
        if part == category and len(suffix) >= 4 and suffix[2] == "images":
            return "/".join(suffix)
    raise ValueError(f"not an RGB image member for category {category}: {member_name}")


def _selection_key(seed: int, category: str, sequence_name: str) -> str:
    return hashlib.sha256(f"{seed}\0{category}\0{sequence_name}".encode("utf-8")).hexdigest()


def _crc32(path: Path) -> int:
    checksum = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            checksum = zlib.crc32(chunk, checksum)
    return checksum & 0xFFFFFFFF


def _file_matches_zip_info(path: Path, info: zipfile.ZipInfo) -> bool:
    return path.is_file() and path.stat().st_size == info.file_size and _crc32(path) == info.CRC


def _zip_is_readable(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with zipfile.ZipFile(path) as archive:
            archive.infolist()
    except (OSError, zipfile.BadZipFile):
        return False
    return True


def _probe_http_size(url: str, curl_bin: str) -> int | None:
    result = subprocess.run(
        [
            curl_bin,
            "--head",
            "--location",
            "--silent",
            "--show-error",
            "--connect-timeout",
            "30",
            url,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    matches = re.findall(r"(?im)^content-length:\s*(\d+)\s*$", result.stdout)
    if not matches:
        return None
    size = int(matches[-1])
    return size if size > 0 else None


def _remove_file_prefix(path: Path, prefix_size: int) -> None:
    if prefix_size <= 0:
        raise ValueError("prefix_size must be positive")

    fallocate = shutil.which("fallocate")
    if fallocate is not None:
        result = subprocess.run(
            [
                fallocate,
                "--collapse-range",
                "--offset",
                "0",
                "--length",
                str(prefix_size),
                str(path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return

    temporary = path.with_suffix(path.suffix + ".repair")
    temporary.unlink(missing_ok=True)
    with path.open("rb") as source, temporary.open("wb") as target:
        source.seek(prefix_size)
        shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
    temporary.replace(path)


def _repair_oversized_archive(path: Path, *, expected_size: int) -> bool:
    """Remove a stale resume prefix when a full ZIP was appended to it."""
    if not path.is_file() or expected_size <= 0:
        return False
    actual_size = path.stat().st_size
    if actual_size <= expected_size:
        return False
    prefix_size = actual_size - expected_size
    with path.open("rb") as handle:
        handle.seek(prefix_size)
        signature = handle.read(4)
    if signature not in (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08"):
        return False

    print(
        f"[repair] stripping {prefix_size} stale prefix bytes from {path}",
        flush=True,
    )
    _remove_file_prefix(path, prefix_size)
    return path.stat().st_size == expected_size and _zip_is_readable(path)


def _probe_http_status(url: str, curl_bin: str) -> int | None:
    result = subprocess.run(
        [
            curl_bin,
            "--head",
            "--location",
            "--silent",
            "--show-error",
            "--connect-timeout",
            "30",
            "--output",
            "/dev/null",
            "--write-out",
            "%{http_code}",
            url,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not result.stdout.strip().isdigit():
        return None
    return int(result.stdout.strip())


def download_archive(url: str, destination: Path, *, curl_bin: str) -> Path:
    """Download one archive with retries while retaining partial progress."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    expected_size = _probe_http_size(url, curl_bin)
    if expected_size is not None:
        _repair_oversized_archive(destination, expected_size=expected_size)
        _repair_oversized_archive(partial, expected_size=expected_size)
    if _zip_is_readable(destination):
        return destination
    if _zip_is_readable(partial):
        partial.replace(destination)
        return destination
    if destination.exists():
        if not partial.exists() or destination.stat().st_size > partial.stat().st_size:
            partial.unlink(missing_ok=True)
            destination.replace(partial)
        else:
            destination.unlink()

    print(f"[download] {url}", flush=True)
    result = subprocess.run(build_curl_command(url, partial, curl_bin), check=False)
    if expected_size is not None:
        _repair_oversized_archive(partial, expected_size=expected_size)
    if result.returncode != 0:
        status = _probe_http_status(url, curl_bin)
        if status == 404:
            raise ArchiveNotFoundError(f"archive does not exist: {url}")
        raise RuntimeError(
            f"curl failed with exit code {result.returncode}; rerun to resume {partial}"
        )
    if not _zip_is_readable(partial):
        raise RuntimeError(f"downloaded file is not a readable ZIP: {partial}")
    partial.replace(destination)
    return destination


def _copy_metadata(archive_path: Path, category_dir: Path, category: str) -> None:
    wanted = set(_REQUIRED_METADATA + _OPTIONAL_METADATA)
    with zipfile.ZipFile(archive_path) as archive:
        choices: dict[str, list[zipfile.ZipInfo]] = defaultdict(list)
        for info in archive.infolist():
            if info.is_dir():
                continue
            try:
                parts = _safe_parts(info.filename)
            except ValueError:
                continue
            basename = parts[-1]
            if basename in wanted and (category in parts or len(parts) == 1):
                choices[basename].append(info)
        for basename in _REQUIRED_METADATA:
            if not choices[basename]:
                raise RuntimeError(f"{archive_path} is missing {basename}")
        category_dir.mkdir(parents=True, exist_ok=True)
        for basename in wanted:
            if not choices[basename]:
                continue
            info = sorted(choices[basename], key=lambda item: len(_safe_parts(item.filename)))[0]
            destination = category_dir / basename
            temporary = destination.with_suffix(destination.suffix + ".part")
            with archive.open(info) as source, temporary.open("wb") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
            temporary.replace(destination)


def _ensure_metadata(
    *,
    output_root: Path,
    category: str,
    base_url: str,
    curl_bin: str,
    keep_archives: bool,
) -> None:
    category_dir = output_root / category
    if all((category_dir / name).is_file() for name in _REQUIRED_METADATA):
        return
    archive_path = output_root / ".archives" / category / f"{category}_000.zip"
    download_archive(archive_url(base_url, category, 0), archive_path, curl_bin=curl_bin)
    _copy_metadata(archive_path, category_dir, category)
    if not keep_archives:
        archive_path.unlink(missing_ok=True)


def _new_category_state(
    category: str,
    *,
    sequences_per_category: int,
    min_frames: int,
    min_quality: float,
    seed: int,
) -> dict[str, object]:
    return {
        "schema_version": STATE_SCHEMA_VERSION,
        "category": category,
        "sequences_per_category": sequences_per_category,
        "min_frames": min_frames,
        "min_quality": min_quality,
        "seed": seed,
        "next_archive_index": 1,
        "completed": False,
        "selected": [],
    }


def _load_category_state(
    path: Path,
    category: str,
    *,
    sequences_per_category: int,
    min_frames: int,
    min_quality: float,
    seed: int,
) -> dict[str, object]:
    expected = _new_category_state(
        category,
        sequences_per_category=sequences_per_category,
        min_frames=min_frames,
        min_quality=min_quality,
        seed=seed,
    )
    if not path.is_file():
        return expected
    state = json.loads(path.read_text(encoding="utf-8"))
    for key in (
        "schema_version",
        "category",
        "sequences_per_category",
        "min_frames",
        "min_quality",
        "seed",
    ):
        if state.get(key) != expected[key]:
            raise ValueError(
                f"resume policy mismatch for {category}: {key}={state.get(key)!r}, "
                f"expected {expected[key]!r}; use a different OUTPUT_ROOT"
            )
    if not isinstance(state.get("selected"), list):
        raise ValueError(f"invalid selected list in {path}")
    return state


def _archive_index(name: object) -> int:
    match = _ARCHIVE_PATTERN.search(str(name))
    if match is None:
        return 1
    return int(match.group(1))


def _reconcile_state(
    state: dict[str, object],
    *,
    output_root: Path,
    candidates: Mapping[str, SequenceCandidate],
) -> None:
    selected = state["selected"]
    assert isinstance(selected, list)
    retained: list[dict[str, object]] = []
    rewind_indices: list[int] = []
    seen: set[str] = set()
    for raw_entry in selected:
        if not isinstance(raw_entry, dict):
            raise ValueError(f"invalid sequence state for {state['category']}")
        sequence_name = raw_entry.get("sequence_name")
        candidate = candidates.get(str(sequence_name))
        if candidate is None or sequence_name in seen:
            rewind_indices.append(_archive_index(raw_entry.get("source_archive")))
            continue
        present = sum(
            (output_root.joinpath(*PurePosixPath(path).parts)).is_file()
            for path in candidate.image_paths
        )
        if present != candidate.valid_frame_count:
            rewind_indices.append(_archive_index(raw_entry.get("source_archive")))
            continue
        raw_entry["quality_score"] = candidate.quality_score
        raw_entry["valid_frame_count"] = candidate.valid_frame_count
        raw_entry["extracted_frame_count"] = present
        retained.append(raw_entry)
        seen.add(str(sequence_name))
    state["selected"] = retained
    if rewind_indices:
        state["next_archive_index"] = min(
            int(state.get("next_archive_index", 1)), min(rewind_indices)
        )
    state["completed"] = False


def _process_category(
    *,
    output_root: Path,
    category: str,
    sequences_per_category: int,
    min_frames: int,
    min_quality: float,
    seed: int,
    base_url: str,
    curl_bin: str,
    keep_archives: bool,
    max_data_archives: int,
) -> dict[str, object]:
    print(f"[category] {category}", flush=True)
    _ensure_metadata(
        output_root=output_root,
        category=category,
        base_url=base_url,
        curl_bin=curl_bin,
        keep_archives=keep_archives,
    )
    candidates = load_eligible_sequences(
        output_root / category,
        category=category,
        min_frames=min_frames,
        min_quality=min_quality,
    )
    if len(candidates) < sequences_per_category:
        raise RuntimeError(
            f"{category} has only {len(candidates)} eligible annotated sequences; "
            f"need {sequences_per_category}"
        )

    state_path = output_root / ".download_state" / f"{category}.json"
    state = _load_category_state(
        state_path,
        category,
        sequences_per_category=sequences_per_category,
        min_frames=min_frames,
        min_quality=min_quality,
        seed=seed,
    )
    _reconcile_state(state, output_root=output_root, candidates=candidates)
    selected = state["selected"]
    assert isinstance(selected, list)
    if len(selected) == sequences_per_category:
        state["completed"] = True
        _atomic_write_json(state_path, state)
        print(f"[resume] {category}: already complete", flush=True)
        return state

    archive_index = max(1, int(state.get("next_archive_index", 1)))
    archives_processed = 0
    while len(selected) < sequences_per_category:
        if max_data_archives and archives_processed >= max_data_archives:
            raise RuntimeError(
                f"{category} still needs {sequences_per_category - len(selected)} sequences "
                f"after MAX_DATA_ARCHIVES={max_data_archives}; rerun without that limit"
            )
        archive_name = f"{category}_{archive_index:03d}.zip"
        archive_path = output_root / ".archives" / category / archive_name
        try:
            download_archive(
                archive_url(base_url, category, archive_index),
                archive_path,
                curl_bin=curl_bin,
            )
        except ArchiveNotFoundError as error:
            raise RuntimeError(
                f"exhausted {category} archives with {len(selected)}/"
                f"{sequences_per_category} selected sequences"
            ) from error

        available = inspect_data_archive(archive_path, category, candidates)
        selected_names = {str(entry["sequence_name"]) for entry in selected}
        chosen = select_archive_sequences(
            category=category,
            available=available,
            needed=sequences_per_category - len(selected),
            seed=seed,
            excluded=selected_names,
        )
        if chosen:
            counts = extract_archive_images(
                archive_path,
                output_root,
                {name: available[name] for name in chosen},
            )
            for name in chosen:
                candidate = candidates[name]
                if counts[name] != candidate.valid_frame_count:
                    raise RuntimeError(f"incomplete extraction for {category}/{name}")
                selected.append(
                    {
                        "sequence_name": name,
                        "quality_score": candidate.quality_score,
                        "valid_frame_count": candidate.valid_frame_count,
                        "extracted_frame_count": counts[name],
                        "source_archive": archive_name,
                    }
                )
            print(
                f"[select] {category}: +{len(chosen)} => "
                f"{len(selected)}/{sequences_per_category}",
                flush=True,
            )
        else:
            print(f"[select] {category}: no new eligible sequences in {archive_name}", flush=True)

        archive_index += 1
        archives_processed += 1
        state["next_archive_index"] = archive_index
        state["completed"] = len(selected) == sequences_per_category
        _atomic_write_json(state_path, state)
        if not keep_archives:
            archive_path.unlink(missing_ok=True)

    return state


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download a resumable CO3Dv2 RGB subset from official category archives. "
            "The default protocol selects 41 categories x 50 sequences on AutoDL."
        )
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--category-file", type=Path, default=DEFAULT_CATEGORY_FILE)
    parser.add_argument("--sequences-per-category", type=int, default=50)
    parser.add_argument("--min-frames", type=int, default=50)
    parser.add_argument("--min-quality", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--curl-bin", default="curl")
    parser.add_argument(
        "--category-limit",
        type=int,
        default=0,
        help="Process only the first N configured categories; 0 means all.",
    )
    parser.add_argument(
        "--max-data-archives",
        type=int,
        default=0,
        help="Per-category safety cap for this invocation; 0 means unlimited.",
    )
    parser.add_argument("--keep-archives", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.sequences_per_category <= 0:
        raise ValueError("sequences-per-category must be positive")
    if args.min_frames <= 0:
        raise ValueError("min-frames must be positive")
    if args.category_limit < 0 or args.max_data_archives < 0:
        raise ValueError("category-limit and max-data-archives must be non-negative")
    if shutil.which(args.curl_bin) is None:
        raise RuntimeError(f"curl executable not found: {args.curl_bin}")

    categories = load_categories(args.category_file)
    if args.category_limit:
        categories = categories[: args.category_limit]
    args.output_root.mkdir(parents=True, exist_ok=True)
    print(
        f"[plan] root={args.output_root} categories={len(categories)} "
        f"quota={args.sequences_per_category} total={len(categories) * args.sequences_per_category}",
        flush=True,
    )

    states: dict[str, Mapping[str, object]] = {}
    for category in categories:
        states[category] = _process_category(
            output_root=args.output_root,
            category=category,
            sequences_per_category=args.sequences_per_category,
            min_frames=args.min_frames,
            min_quality=args.min_quality,
            seed=args.seed,
            base_url=args.base_url,
            curl_bin=args.curl_bin,
            keep_archives=args.keep_archives,
            max_data_archives=args.max_data_archives,
        )

    manifest = build_dataset_manifest(
        categories=categories,
        category_states=states,
        sequences_per_category=args.sequences_per_category,
        min_frames=args.min_frames,
        min_quality=args.min_quality,
        seed=args.seed,
        source_base_url=args.base_url,
    )
    manifest_path = args.output_root / "download_manifest.json"
    _atomic_write_json(manifest_path, manifest)
    print(
        f"[done] sequences={manifest['sequence_count']} manifest={manifest_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
