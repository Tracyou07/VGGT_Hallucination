"""Read-only, bounded random-access indexing for ScanNet SENS v4 files."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import struct
from typing import BinaryIO


SENS_VERSION = 4
MAX_STRING_BYTES = 1024 * 1024
MAX_PAYLOAD_BYTES = 1 << 34
MAX_DIMENSION = 100_000
COLOR_COMPRESSION = {0: "raw", 1: "png", 2: "jpeg"}
DEPTH_COMPRESSION = {0: "raw_ushort", 1: "zlib_ushort", 2: "occi_ushort"}


class SensIndexError(ValueError):
    """Base class for malformed or unreadable SENS inputs."""


class SensFormatError(SensIndexError):
    """Raised when a SENS field is syntactically present but invalid."""


class SensTruncationError(SensFormatError):
    """Raised when declared SENS content exceeds the file boundary."""


@dataclass(frozen=True)
class SensFrame:
    frame_index: int
    record_offset: int
    camera_to_world: tuple[float, ...]
    timestamp_color_us: int
    timestamp_depth_us: int
    color_size: int
    color_data_offset: int
    depth_size: int
    depth_data_offset: int
    next_record_offset: int


@dataclass(frozen=True)
class SensIndex:
    path: Path
    file_size: int
    version: int
    sensor_name: str
    intrinsic_color: tuple[float, ...]
    extrinsic_color: tuple[float, ...]
    intrinsic_depth: tuple[float, ...]
    extrinsic_depth: tuple[float, ...]
    color_compression: str
    depth_compression: str
    color_width: int
    color_height: int
    depth_width: int
    depth_height: int
    depth_shift: float
    frames: tuple[SensFrame, ...]


def _read_exact(stream: BinaryIO, size: int, field: str) -> bytes:
    data = stream.read(size)
    if len(data) != size:
        raise SensTruncationError(f"truncated {field}: expected {size} bytes, got {len(data)}")
    return data


def _read_struct(stream: BinaryIO, format_string: str, field: str):
    return struct.unpack(format_string, _read_exact(stream, struct.calcsize(format_string), field))


def _read_matrix(stream: BinaryIO, field: str) -> tuple[float, ...]:
    matrix = tuple(_read_struct(stream, "<16f", field))
    if not all(math.isfinite(value) for value in matrix):
        raise SensFormatError(f"{field}: matrix values must be finite")
    return matrix


def _validate_dimension(value: int, field: str) -> None:
    if value <= 0 or value > MAX_DIMENSION:
        raise SensFormatError(f"{field}: expected 1..{MAX_DIMENSION}, got {value}")


def _validate_size(size: int, field: str, maximum: int) -> None:
    if size > maximum:
        raise SensFormatError(f"{field}: declared size {size} exceeds limit {maximum}")


def _validate_payload_range(file_size: int, start: int, size: int, field: str) -> None:
    _validate_size(size, field, MAX_PAYLOAD_BYTES)
    end = start + size
    if end > file_size:
        raise SensTruncationError(f"{field}: declared end {end} exceeds file size {file_size}")


def index_sens(path: str | Path) -> SensIndex:
    """Parse a SENS v4 header and frame table without decoding payloads."""
    source = Path(path)
    file_size = source.stat().st_size
    with source.open("rb") as stream:
        version = _read_struct(stream, "<I", "version")[0]
        if version != SENS_VERSION:
            raise SensFormatError(f"version: expected {SENS_VERSION}, got {version}")

        sensor_name_size = _read_struct(stream, "<Q", "sensor_name length")[0]
        _validate_size(sensor_name_size, "sensor_name length", MAX_STRING_BYTES)
        try:
            sensor_name = _read_exact(stream, sensor_name_size, "sensor_name").decode("utf-8")
        except UnicodeDecodeError as error:
            raise SensFormatError("sensor_name: expected UTF-8") from error

        intrinsic_color = _read_matrix(stream, "intrinsic_color")
        extrinsic_color = _read_matrix(stream, "extrinsic_color")
        intrinsic_depth = _read_matrix(stream, "intrinsic_depth")
        extrinsic_depth = _read_matrix(stream, "extrinsic_depth")

        color_code = _read_struct(stream, "<i", "color_compression")[0]
        depth_code = _read_struct(stream, "<i", "depth_compression")[0]
        if color_code not in COLOR_COMPRESSION:
            raise SensFormatError(f"color_compression: unsupported code {color_code}")
        if depth_code not in DEPTH_COMPRESSION:
            raise SensFormatError(f"depth_compression: unsupported code {depth_code}")

        color_width, color_height, depth_width, depth_height = _read_struct(stream, "<4I", "dimensions")
        _validate_dimension(color_width, "color_width")
        _validate_dimension(color_height, "color_height")
        _validate_dimension(depth_width, "depth_width")
        _validate_dimension(depth_height, "depth_height")
        depth_shift = _read_struct(stream, "<f", "depth_shift")[0]
        if not math.isfinite(depth_shift) or depth_shift <= 0:
            raise SensFormatError(f"depth_shift: expected finite positive value, got {depth_shift}")
        frame_count = _read_struct(stream, "<Q", "frame_count")[0]

        frames = []
        for frame_index in range(frame_count):
            record_offset = stream.tell()
            context = f"frame {frame_index}"
            camera_to_world = _read_matrix(stream, f"{context} camera_to_world")
            timestamp_color_us = _read_struct(stream, "<Q", f"{context} timestamp_color_us")[0]
            timestamp_depth_us = _read_struct(stream, "<Q", f"{context} timestamp_depth_us")[0]
            color_size = _read_struct(stream, "<Q", f"{context} color_size")[0]
            depth_size = _read_struct(stream, "<Q", f"{context} depth_size")[0]
            color_data_offset = stream.tell()
            depth_data_offset = color_data_offset + color_size
            next_record_offset = depth_data_offset + depth_size
            _validate_payload_range(file_size, color_data_offset, color_size, f"{context} color payload")
            _validate_payload_range(file_size, depth_data_offset, depth_size, f"{context} depth payload")
            stream.seek(next_record_offset)
            frames.append(
                SensFrame(
                    frame_index=frame_index,
                    record_offset=record_offset,
                    camera_to_world=camera_to_world,
                    timestamp_color_us=timestamp_color_us,
                    timestamp_depth_us=timestamp_depth_us,
                    color_size=color_size,
                    color_data_offset=color_data_offset,
                    depth_size=depth_size,
                    depth_data_offset=depth_data_offset,
                    next_record_offset=next_record_offset,
                )
            )
        if stream.tell() != file_size:
            raise SensFormatError(
                f"trailing undeclared bytes: frame table ended at {stream.tell()}, file size is {file_size}"
            )

    return SensIndex(
        path=source,
        file_size=file_size,
        version=version,
        sensor_name=sensor_name,
        intrinsic_color=intrinsic_color,
        extrinsic_color=extrinsic_color,
        intrinsic_depth=intrinsic_depth,
        extrinsic_depth=extrinsic_depth,
        color_compression=COLOR_COMPRESSION[color_code],
        depth_compression=DEPTH_COMPRESSION[depth_code],
        color_width=color_width,
        color_height=color_height,
        depth_width=depth_width,
        depth_height=depth_height,
        depth_shift=depth_shift,
        frames=tuple(frames),
    )
