"""Plan, seal, and validate immutable fixed-eight ScanNet observations."""

from __future__ import annotations

import ctypes
import errno
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
from typing import Any, Callable, Mapping
import zlib

import numpy as np
from PIL import Image

from .contracts import canonical_json_bytes, canonical_json_sha256, sha256_file, validate_schema
from .sens_index import SensIndex, SensIndexError, index_sens


PLAN_SCHEMA = "camera_solution_space_01.observation_plan.v1"
MANIFEST_SCHEMA = "camera_solution_space_01.observation_manifest.v1"
COMPLETE_SCHEMA = "camera_solution_space_01.observation_complete.v1"
SELECTION_VERSION = "fixed8_stride15_v1"
FRAME_COUNT = 8
FRAME_STRIDE = 15
AT_FDCWD = -100
RENAME_NOREPLACE = 1
PLAN_KEYS = frozenset(
    {
        "schema",
        "source",
        "scene_id",
        "split",
        "selection_version",
        "frame_ids",
        "frames",
        "header_fingerprint",
        "selection",
        "plan_id",
    }
)
SOURCE_KEYS = frozenset({"path", "size", "sha256"})
FRAME_KEYS = frozenset({"frame_id", "timestamp_color_us", "timestamp_depth_us"})
SELECTION_KEYS = frozenset({"candidate_count", "eligible_count", "chosen_start"})


class ObservationError(ValueError):
    """Raised when a fixed observation cannot be planned, sealed, or validated."""


def source_fingerprint(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise ObservationError(f"source is not a regular file: {source}")
    return {
        "path": str(source.resolve()),
        "size": source.stat().st_size,
        "sha256": sha256_file(source),
    }


def _header_document(index: SensIndex) -> dict[str, Any]:
    return {
        "version": index.version,
        "sensor_name": index.sensor_name,
        "intrinsic_color": list(index.intrinsic_color),
        "extrinsic_color": list(index.extrinsic_color),
        "intrinsic_depth": list(index.intrinsic_depth),
        "extrinsic_depth": list(index.extrinsic_depth),
        "color_compression": index.color_compression,
        "depth_compression": index.depth_compression,
        "color_width": index.color_width,
        "color_height": index.color_height,
        "depth_width": index.depth_width,
        "depth_height": index.depth_height,
        "depth_shift": index.depth_shift,
        "frame_count": len(index.frames),
    }


def _header_fingerprint(index: SensIndex) -> str:
    return canonical_json_sha256(_header_document(index))


def _eligibility_value(
    eligibility: Mapping[int, bool] | Callable[[int, tuple[int, ...]], bool],
    start: int,
    frame_ids: tuple[int, ...],
) -> bool:
    if isinstance(eligibility, Mapping):
        value = eligibility.get(start, False)
    elif callable(eligibility):
        value = eligibility(start, frame_ids)
    else:
        raise ObservationError("eligibility must be an explicit mapping or callback")
    if not isinstance(value, bool):
        raise ObservationError(f"eligibility for start {start} must be bool")
    return value


def _plan_without_id(plan: Mapping[str, Any]) -> dict[str, Any]:
    document = dict(plan)
    document.pop("plan_id", None)
    return document


def _require_exact_keys(value: Any, expected: frozenset[str], label: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise ObservationError(f"{label} must be a native JSON object")
    actual = frozenset(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ObservationError(f"{label} keys mismatch: missing={missing}, extra={extra}")
    return value


def _require_nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ObservationError(f"{label} must be a non-empty string")
    return value


def _require_uint(value: Any, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ObservationError(f"{label} must be a nonnegative integer")
    return value


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ObservationError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _validate_plan(
    plan: Mapping[str, Any], frame_count: int | None = None
) -> dict[str, Any]:
    checked = _require_exact_keys(plan, PLAN_KEYS, "plan top-level")
    try:
        validate_schema(checked, PLAN_SCHEMA)
    except ValueError as error:
        raise ObservationError(str(error)) from error

    source = _require_exact_keys(checked["source"], SOURCE_KEYS, "source")
    _require_nonempty_string(source["path"], "source path")
    _require_uint(source["size"], "source size")
    _require_sha256(source["sha256"], "source sha256")
    _require_nonempty_string(checked["scene_id"], "scene_id")
    _require_nonempty_string(checked["split"], "split")
    if checked["selection_version"] != SELECTION_VERSION:
        raise ObservationError("unexpected selection_version")
    _require_sha256(checked["header_fingerprint"], "header_fingerprint")

    frame_ids = checked["frame_ids"]
    if type(frame_ids) is not list or len(frame_ids) != FRAME_COUNT:
        raise ObservationError("plan frame_ids must contain exactly eight integers")
    if any(type(frame_id) is not int or frame_id < 0 for frame_id in frame_ids):
        raise ObservationError("plan frame_ids must be nonnegative integers, excluding bool")
    if frame_ids != [
        frame_ids[0] + FRAME_STRIDE * offset for offset in range(FRAME_COUNT)
    ]:
        raise ObservationError("plan frame_ids must be one fixed eight-frame stride-15 sequence")
    if frame_count is not None:
        if type(frame_count) is not int or frame_count < 0:
            raise ObservationError("source frame count must be a nonnegative integer")
        for frame_id in frame_ids:
            if frame_id >= frame_count:
                raise ObservationError(
                    f"plan frame ID {frame_id} is outside source frame range 0..{frame_count - 1}"
                )

    frames = checked["frames"]
    if type(frames) is not list or len(frames) != FRAME_COUNT:
        raise ObservationError("plan frames must contain eight timestamp records")
    for frame_id, record in zip(frame_ids, frames):
        record = _require_exact_keys(record, FRAME_KEYS, "frame record")
        if type(record["frame_id"]) is not int or record["frame_id"] != frame_id:
            raise ObservationError("plan frame records do not match frame_ids")
        _require_uint(record["timestamp_color_us"], "frame timestamp_color_us")
        _require_uint(record["timestamp_depth_us"], "frame timestamp_depth_us")

    selection = _require_exact_keys(checked["selection"], SELECTION_KEYS, "selection")
    candidate_count = _require_uint(selection["candidate_count"], "selection candidate_count")
    eligible_count = _require_uint(selection["eligible_count"], "selection eligible_count")
    chosen_start = _require_uint(selection["chosen_start"], "selection chosen_start")
    if candidate_count == 0 or chosen_start >= candidate_count:
        raise ObservationError("selection chosen_start must be within candidate_count")
    if eligible_count == 0 or eligible_count > candidate_count:
        raise ObservationError("selection eligible_count must be within 1..candidate_count")
    if chosen_start != frame_ids[0]:
        raise ObservationError("selection chosen_start must equal the first frame ID")
    if frame_count is not None:
        expected_candidates = max(
            0, frame_count - ((FRAME_COUNT - 1) * FRAME_STRIDE)
        )
        if candidate_count != expected_candidates:
            raise ObservationError(
                "selection candidate_count does not match the indexed source"
            )

    plan_id = _require_sha256(checked["plan_id"], "plan_id")
    try:
        canonical_id = canonical_json_sha256(_plan_without_id(checked))
    except ValueError as error:
        raise ObservationError(f"plan is not canonical JSON: {error}") from error
    if plan_id != canonical_id:
        raise ObservationError("plan_id does not match canonical plan hash")
    return checked


def plan_observation(
    source_path: str | Path,
    index: SensIndex,
    scene_id: str,
    split: str,
    eligibility: Mapping[int, bool] | Callable[[int, tuple[int, ...]], bool],
) -> dict[str, Any]:
    """Choose the lowest explicitly eligible fixed eight-frame window."""
    source = Path(source_path)
    if source.resolve() != index.path.resolve():
        raise ObservationError("source_path must match the indexed SENS path")
    if not isinstance(scene_id, str) or not scene_id or not isinstance(split, str) or not split:
        raise ObservationError("scene_id and split must be non-empty strings")
    selected_start = None
    candidate_count = max(0, len(index.frames) - ((FRAME_COUNT - 1) * FRAME_STRIDE))
    eligible_count = 0
    for start in range(candidate_count):
        frame_ids = tuple(start + FRAME_STRIDE * offset for offset in range(FRAME_COUNT))
        if _eligibility_value(eligibility, start, frame_ids):
            eligible_count += 1
            if selected_start is None:
                selected_start = start
    if selected_start is None:
        raise ObservationError("no explicitly eligible fixed8 window")
    frame_ids = [selected_start + FRAME_STRIDE * offset for offset in range(FRAME_COUNT)]
    frames = [
        {
            "frame_id": frame_id,
            "timestamp_color_us": index.frames[frame_id].timestamp_color_us,
            "timestamp_depth_us": index.frames[frame_id].timestamp_depth_us,
        }
        for frame_id in frame_ids
    ]
    plan: dict[str, Any] = {
        "schema": PLAN_SCHEMA,
        "source": source_fingerprint(source),
        "scene_id": scene_id,
        "split": split,
        "selection_version": SELECTION_VERSION,
        "frame_ids": frame_ids,
        "frames": frames,
        "header_fingerprint": _header_fingerprint(index),
        "selection": {
            "candidate_count": candidate_count,
            "eligible_count": eligible_count,
            "chosen_start": selected_start,
        },
    }
    plan["plan_id"] = canonical_json_sha256(plan)
    return plan


def _read_payload(stream, offset: int, size: int, label: str, audit: list[dict[str, Any]]) -> bytes:
    stream.seek(offset)
    data = stream.read(size)
    if len(data) != size:
        raise ObservationError(f"truncated planned {label} payload at offset {offset}")
    audit.append({"offset": offset, "size": size, "kind": label})
    return data


def _decode_rgb(payload: bytes, index: SensIndex, frame_id: int) -> np.ndarray:
    try:
        with Image.open(io.BytesIO(payload)) as image:
            if image.format != "JPEG":
                raise ObservationError(
                    f"frame {frame_id}: detected {image.format!r} payload, expected JPEG"
                )
            image.load()
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    except ObservationError:
        raise
    except Exception as error:
        raise ObservationError(f"frame {frame_id}: invalid JPEG color payload") from error
    if rgb.shape != (index.color_height, index.color_width, 3):
        raise ObservationError(
            f"frame {frame_id}: JPEG dimensions {rgb.shape[1]}x{rgb.shape[0]} do not match "
            f"{index.color_width}x{index.color_height}"
        )
    return rgb


def _decode_depth(payload: bytes, index: SensIndex, frame_id: int) -> np.ndarray:
    try:
        raw = zlib.decompress(payload)
    except zlib.error as error:
        raise ObservationError(f"frame {frame_id}: invalid zlib uint16 depth payload") from error
    expected_size = index.depth_width * index.depth_height * 2
    if len(raw) != expected_size:
        raise ObservationError(
            f"frame {frame_id}: decoded depth has {len(raw)} bytes, expected {expected_size}"
        )
    return np.frombuffer(raw, dtype="<u2").reshape(index.depth_height, index.depth_width).copy()


def _write_json(path: Path, document: Any) -> None:
    path.write_bytes(canonical_json_bytes(document))


def _safe_relative(value: Any) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ObservationError("manifest path must be a non-empty portable relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts or path.as_posix() != value:
        raise ObservationError(f"manifest path escapes observation root: {value!r}")
    return path


def _artifact_files(root: Path) -> list[dict[str, Any]]:
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ObservationError(f"observation artifacts must not contain symlinks: {path}")
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            files.append({"path": relative, "size": path.stat().st_size, "sha256": sha256_file(path)})
    return files


def _merkle_hash(files: list[dict[str, Any]]) -> str:
    return canonical_json_sha256(
        {"schema": "camera_solution_space_01.artifact_merkle.v1", "files": files}
    )

def _rename_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish a path while refusing every pre-existing directory entry."""
    try:
        libc = ctypes.CDLL(None, use_errno=True)
    except OSError as error:
        raise ObservationError("atomic no-replace publication is unavailable") from error
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise ObservationError("atomic no-replace publication is unavailable")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    ctypes.set_errno(0)
    result = renameat2(
        AT_FDCWD,
        os.fsencode(source),
        AT_FDCWD,
        os.fsencode(destination),
        RENAME_NOREPLACE,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in (errno.EEXIST, errno.ENOTEMPTY):
        raise FileExistsError(error_number, os.strerror(error_number), destination)
    unavailable_errors = {errno.ENOSYS, errno.EINVAL, errno.EOPNOTSUPP}
    if error_number in unavailable_errors:
        raise ObservationError(
            f"atomic no-replace publication is unavailable: {os.strerror(error_number)}"
        )
    raise OSError(error_number, os.strerror(error_number), destination)


def seal_observation(
    plan: Mapping[str, Any], source_path: str | Path, output_parent: str | Path
) -> Path:
    """Decode only planned RGB-D payloads into an atomically sealed observation."""
    checked_plan = _validate_plan(plan)
    source = Path(source_path)
    if source_fingerprint(source) != checked_plan["source"]:
        raise ObservationError("source fingerprint changed since planning")
    try:
        index = index_sens(source)
    except SensIndexError as error:
        raise ObservationError(f"cannot index source during sealing: {error}") from error
    checked_plan = _validate_plan(checked_plan, frame_count=len(index.frames))
    if _header_fingerprint(index) != checked_plan["header_fingerprint"]:
        raise ObservationError("source header fingerprint changed since planning")
    if index.color_compression != "jpeg" or index.depth_compression != "zlib_ushort":
        raise ObservationError(
            f"unsupported active compression: color={index.color_compression}, depth={index.depth_compression}"
        )
    for planned in checked_plan["frames"]:
        frame = index.frames[planned["frame_id"]]
        if (
            frame.timestamp_color_us != planned["timestamp_color_us"]
            or frame.timestamp_depth_us != planned["timestamp_depth_us"]
        ):
            raise ObservationError("source frame timestamps changed since planning")

    parent = Path(output_parent)
    parent.mkdir(parents=True, exist_ok=True)
    target = parent / checked_plan["plan_id"]
    if os.path.lexists(target):
        try:
            if validate_observation(target, source) == checked_plan["plan_id"]:
                return target
        except ObservationError as error:
            raise ObservationError(f"existing output is invalid and will not be overwritten: {error}") from error
        raise ObservationError("existing output does not match the requested plan")
    temporary = Path(tempfile.mkdtemp(prefix=f".{checked_plan['plan_id']}.tmp-", dir=parent))
    try:
        (temporary / "rgb").mkdir()
        (temporary / "depth").mkdir()
        _write_json(temporary / "plan.json", checked_plan)
        _write_json(temporary / "intrinsics.json", _header_document(index))
        pose_records = []
        read_audit: list[dict[str, Any]] = []
        with source.open("rb") as stream:
            for position, frame_id in enumerate(checked_plan["frame_ids"]):
                frame = index.frames[frame_id]
                color = _read_payload(stream, frame.color_data_offset, frame.color_size, "color", read_audit)
                depth = _read_payload(stream, frame.depth_data_offset, frame.depth_size, "depth", read_audit)
                np.save(temporary / "rgb" / f"{position:06d}.npy", _decode_rgb(color, index, frame_id), allow_pickle=False)
                np.save(
                    temporary / "depth" / f"{position:06d}.npy",
                    _decode_depth(depth, index, frame_id),
                    allow_pickle=False,
                )
                pose_records.append(
                    {
                        "frame_id": frame_id,
                        "timestamp_color_us": frame.timestamp_color_us,
                        "timestamp_depth_us": frame.timestamp_depth_us,
                        "camera_to_world": list(frame.camera_to_world),
                    }
                )
        _write_json(temporary / "pose_audit.json", {"schema": "camera_solution_space_01.pose_audit.v1", "poses": pose_records})
        _write_json(temporary / "read_audit.json", read_audit)
        files = _artifact_files(temporary)
        ordered_model_input = [
            {
                "frame_id": frame_id,
                "rgb": f"rgb/{position:06d}.npy",
                "depth": f"depth/{position:06d}.npy",
                "intrinsics": "intrinsics.json",
                "pose_audit": "pose_audit.json",
            }
            for position, frame_id in enumerate(checked_plan["frame_ids"])
        ]
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "observation_id": checked_plan["plan_id"],
            "plan_id": checked_plan["plan_id"],
            "source": checked_plan["source"],
            "files": files,
            "ordered_model_input": ordered_model_input,
            "artifact_merkle_hash": _merkle_hash(files),
        }
        _write_json(temporary / "manifest.json", manifest)
        _write_json(
            temporary / "complete.json",
            {
                "schema": COMPLETE_SCHEMA,
                "observation_id": checked_plan["plan_id"],
                "plan_id": checked_plan["plan_id"],
                "manifest_sha256": sha256_file(temporary / "manifest.json"),
            },
        )
        try:
            _rename_noreplace(temporary, target)
        except FileExistsError:
            try:
                if validate_observation(target, source) == checked_plan["plan_id"]:
                    return target
            except ObservationError as error:
                raise ObservationError(
                    f"existing output is invalid and will not be overwritten: {error}"
                ) from error
            raise ObservationError("existing output does not match the requested plan")
        temporary = None
    finally:
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary)
    return target


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ObservationError(f"invalid {label}") from error
    if not isinstance(value, dict):
        raise ObservationError(f"{label} must be a JSON object")
    return value


def validate_observation(observation_root: str | Path, source_path: str | Path) -> str:
    """Deeply validate a sealed observation and return its immutable observation ID."""
    root = Path(observation_root)
    if not root.is_dir() or root.is_symlink():
        raise ObservationError("observation root must be a real directory")
    complete = _load_json(root / "complete.json", "complete marker")
    try:
        validate_schema(complete, COMPLETE_SCHEMA)
    except ValueError as error:
        raise ObservationError(str(error)) from error
    manifest = _load_json(root / "manifest.json", "manifest")
    try:
        validate_schema(manifest, MANIFEST_SCHEMA)
    except ValueError as error:
        raise ObservationError(str(error)) from error
    if complete.get("manifest_sha256") != sha256_file(root / "manifest.json"):
        raise ObservationError("manifest hash does not match complete marker")
    observation_id = manifest.get("observation_id")
    if not isinstance(observation_id, str) or observation_id != manifest.get("plan_id"):
        raise ObservationError("manifest observation_id and plan_id must match")
    if complete.get("observation_id") != observation_id or complete.get("plan_id") != observation_id:
        raise ObservationError("complete marker does not match manifest")
    if root.name != observation_id:
        raise ObservationError("observation root name does not match observation_id")
    if source_fingerprint(source_path) != manifest.get("source"):
        raise ObservationError("source fingerprint changed since sealing")
    files = manifest.get("files")
    if not isinstance(files, list):
        raise ObservationError("manifest files must be a list")
    expected_paths = set()
    for entry in files:
        if not isinstance(entry, dict):
            raise ObservationError("manifest file entry must be an object")
        relative = _safe_relative(entry.get("path"))
        relative_text = relative.as_posix()
        if relative_text in expected_paths:
            raise ObservationError("manifest files contain duplicate paths")
        expected_paths.add(relative_text)
        path = root.joinpath(*relative.parts)
        if not path.is_file() or path.is_symlink():
            raise ObservationError(f"missing manifest file: {relative_text}")
        if path.stat().st_size != entry.get("size"):
            raise ObservationError(f"size mismatch for {relative_text}")
        if sha256_file(path) != entry.get("sha256"):
            raise ObservationError(f"hash mismatch for {relative_text}")
    if manifest.get("artifact_merkle_hash") != _merkle_hash(files):
        raise ObservationError("artifact Merkle hash mismatch")
    actual_paths = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    expected_with_protocol = expected_paths | {"manifest.json", "complete.json"}
    extra = actual_paths - expected_with_protocol
    missing = expected_with_protocol - actual_paths
    if extra:
        raise ObservationError(f"extra observation files: {sorted(extra)}")
    if missing:
        raise ObservationError(f"missing observation files: {sorted(missing)}")
    plan_path = root / "plan.json"
    if "plan.json" not in expected_paths:
        raise ObservationError("manifest must include plan.json")
    plan = _validate_plan(_load_json(plan_path, "plan"))
    if plan["plan_id"] != observation_id or plan["source"] != manifest["source"]:
        raise ObservationError("plan does not match manifest")
    ordered = manifest.get("ordered_model_input")
    expected_ordered = [
        {
            "frame_id": frame_id,
            "rgb": f"rgb/{position:06d}.npy",
            "depth": f"depth/{position:06d}.npy",
            "intrinsics": "intrinsics.json",
            "pose_audit": "pose_audit.json",
        }
        for position, frame_id in enumerate(plan["frame_ids"])
    ]
    if type(ordered) is not list or ordered != expected_ordered:
        raise ObservationError(
            "ordered model input must exactly match all eight planned position mappings"
        )
    return observation_id
