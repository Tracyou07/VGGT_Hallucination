"""Memory-conscious reader for ScanNet v4 .sens color frames and poses."""

from __future__ import annotations

import os
from pathlib import Path
import struct

import cv2
import numpy as np


COMPRESSION_TYPE_COLOR = {-1: "unknown", 0: "raw", 1: "png", 2: "jpeg"}


def _read_exact(handle, size: int) -> bytes:
    data = handle.read(size)
    if len(data) != size:
        raise EOFError(f"Unexpected end of .sens file: wanted {size} bytes")
    return data


def _unpack(handle, format_string: str):
    size = struct.calcsize(format_string)
    return struct.unpack(format_string, _read_exact(handle, size))


class RGBDFrame:
    """Frame metadata with offsets into the source file instead of payload copies."""

    def load(self, handle) -> None:
        self.camera_to_world = np.asarray(
            _unpack(handle, "<16f"),
            dtype=np.float32,
        ).reshape(4, 4)
        self.timestamp_color = _unpack(handle, "<Q")[0]
        self.timestamp_depth = _unpack(handle, "<Q")[0]
        self.color_size_bytes = _unpack(handle, "<Q")[0]
        self.depth_size_bytes = _unpack(handle, "<Q")[0]
        self.color_offset = handle.tell()
        handle.seek(self.color_size_bytes, os.SEEK_CUR)
        self.depth_offset = handle.tell()
        handle.seek(self.depth_size_bytes, os.SEEK_CUR)


class SensorData:
    """Read ScanNet metadata eagerly and compressed color bytes on export."""

    version = 4

    def __init__(self, filename: str) -> None:
        self.filename = Path(filename)
        self.load()

    def load(self) -> None:
        with self.filename.open("rb") as handle:
            version = _unpack(handle, "<I")[0]
            if version != self.version:
                raise ValueError(f"Unsupported SensorData version: {version}")
            name_length = _unpack(handle, "<Q")[0]
            self.sensor_name = _read_exact(handle, name_length).decode(
                "utf-8",
                errors="replace",
            )
            self.intrinsic_color = self._read_matrix(handle)
            self.extrinsic_color = self._read_matrix(handle)
            self.intrinsic_depth = self._read_matrix(handle)
            self.extrinsic_depth = self._read_matrix(handle)
            compression_code = _unpack(handle, "<i")[0]
            try:
                self.color_compression_type = COMPRESSION_TYPE_COLOR[compression_code]
            except KeyError as error:
                raise ValueError(
                    f"Unsupported color compression code: {compression_code}"
                ) from error
            self.depth_compression_code = _unpack(handle, "<i")[0]
            self.color_width = _unpack(handle, "<I")[0]
            self.color_height = _unpack(handle, "<I")[0]
            self.depth_width = _unpack(handle, "<I")[0]
            self.depth_height = _unpack(handle, "<I")[0]
            self.depth_shift = _unpack(handle, "<f")[0]
            frame_count = _unpack(handle, "<Q")[0]
            self.frames: list[RGBDFrame] = []
            for _ in range(frame_count):
                frame = RGBDFrame()
                frame.load(handle)
                self.frames.append(frame)

    @staticmethod
    def _read_matrix(handle) -> np.ndarray:
        return np.asarray(_unpack(handle, "<16f"), dtype=np.float32).reshape(4, 4)

    def _read_color(self, handle, frame: RGBDFrame) -> np.ndarray:
        if self.color_compression_type not in {"jpeg", "png"}:
            raise ValueError(
                f"Unsupported color compression: {self.color_compression_type}"
            )
        handle.seek(frame.color_offset)
        payload = _read_exact(handle, frame.color_size_bytes)
        image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode a ScanNet color frame")
        return image

    @staticmethod
    def _validate_frame_skip(frame_skip: int) -> None:
        if type(frame_skip) is not int or frame_skip < 1:
            raise ValueError("frame_skip must be a positive integer")

    def export_color_images(
        self,
        output_path: str,
        image_size: tuple[int, int] | None = None,
        frame_skip: int = 1,
    ) -> None:
        self._validate_frame_skip(frame_skip)
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        selected = range(0, len(self.frames), frame_skip)
        print(f"exporting {len(selected)} color frames to {output_dir}")
        with self.filename.open("rb") as handle:
            for frame_index in selected:
                color = self._read_color(handle, self.frames[frame_index])
                if image_size is not None:
                    color = cv2.resize(
                        color,
                        (image_size[1], image_size[0]),
                        interpolation=cv2.INTER_AREA,
                    )
                output_file = output_dir / f"{frame_index}.jpg"
                if not cv2.imwrite(str(output_file), color):
                    raise OSError(f"Failed to write color frame: {output_file}")

    def export_poses(self, output_path: str, frame_skip: int = 1) -> None:
        self._validate_frame_skip(frame_skip)
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        selected = range(0, len(self.frames), frame_skip)
        print(f"exporting {len(selected)} camera poses to {output_dir}")
        for frame_index in selected:
            np.savetxt(
                output_dir / f"{frame_index}.txt",
                self.frames[frame_index].camera_to_world,
                fmt="%.8f",
            )
