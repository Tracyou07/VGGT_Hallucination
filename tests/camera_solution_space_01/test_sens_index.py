import math
from pathlib import Path
import struct
import tempfile
import unittest

from pre_experiments.camera_solution_space_01.sens_index import (
    SensFormatError,
    SensTruncationError,
    index_sens,
)


def _matrix(value=1.0):
    return [value if index % 5 == 0 else 0.0 for index in range(16)]


def _fixture_bytes(
    *,
    version=4,
    color_compression=2,
    depth_compression=1,
    color_width=2,
    color_height=1,
    depth_width=2,
    depth_height=1,
    matrix_value=1.0,
    color_size=None,
    frame_count=2,
    suffix=b"",
):
    color_payloads = [b"\xff\xd8\xff", b"\xff\xd8\x00\xff"]
    depth_payloads = [b"\x78\x9c\x03\x00", b"\x78\x9c\x63\x60\x00"]
    payload = bytearray()
    payload.extend(struct.pack("<I", version))
    payload.extend(struct.pack("<Q", 6))
    payload.extend(b"sensor")
    for _ in range(4):
        payload.extend(struct.pack("<16f", *_matrix(matrix_value)))
    payload.extend(struct.pack("<ii", color_compression, depth_compression))
    payload.extend(struct.pack("<IIII", color_width, color_height, depth_width, depth_height))
    payload.extend(struct.pack("<f", 1000.0))
    payload.extend(struct.pack("<Q", frame_count))
    for frame_index, (color, depth) in enumerate(zip(color_payloads[:frame_count], depth_payloads[:frame_count])):
        payload.extend(struct.pack("<16f", *_matrix(float(frame_index + 1))))
        payload.extend(struct.pack("<QQ", 100 + frame_index, 200 + frame_index))
        payload.extend(struct.pack("<Q", len(color) if color_size is None else color_size))
        payload.extend(struct.pack("<Q", len(depth)))
        payload.extend(color)
        payload.extend(depth)
    return bytes(payload) + suffix


class SensIndexTests(unittest.TestCase):
    def _index(self, payload):
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        path = Path(temporary_directory.name) / "fixture.sens"
        path.write_bytes(payload)
        return index_sens(path)

    def test_indexes_two_frames_with_exact_offsets_without_decoding_payloads(self):
        index = self._index(_fixture_bytes())
        self.assertEqual(index.sensor_name, "sensor")
        self.assertEqual(index.version, 4)
        self.assertEqual(index.color_compression, "jpeg")
        self.assertEqual(index.depth_compression, "zlib_ushort")
        self.assertEqual((index.color_width, index.color_height), (2, 1))
        self.assertEqual((index.depth_width, index.depth_height), (2, 1))
        self.assertEqual(index.depth_shift, 1000.0)
        self.assertEqual(len(index.frames), 2)
        self.assertEqual(index.frames[0].timestamp_color_us, 100)
        self.assertEqual(index.frames[0].timestamp_depth_us, 200)
        self.assertEqual(index.frames[0].color_data_offset, 406)
        self.assertEqual(index.frames[0].depth_data_offset, 409)
        self.assertEqual(index.frames[0].next_record_offset, 413)
        self.assertEqual(index.frames[1].record_offset, 413)
        self.assertEqual(index.frames[1].color_data_offset, 509)
        self.assertEqual(index.frames[1].depth_data_offset, 513)
        self.assertEqual(index.frames[1].next_record_offset, 518)
        self.assertFalse(hasattr(index.frames[0], "color_payload"))
        self.assertFalse(hasattr(index.frames[0], "depth_payload"))

    def test_rejects_truncated_header(self):
        with self.assertRaisesRegex(SensTruncationError, "version"):
            self._index(b"\x04")

    def test_rejects_invalid_version_compression_dimensions_and_matrix(self):
        cases = [
            ("version", _fixture_bytes(version=3)),
            ("color_compression", _fixture_bytes(color_compression=99)),
            ("color_width", _fixture_bytes(color_width=0)),
            ("intrinsic_color", _fixture_bytes(matrix_value=math.nan)),
        ]
        for field, payload in cases:
            with self.subTest(field=field):
                with self.assertRaisesRegex(SensFormatError, field):
                    self._index(payload)

    def test_rejects_payload_size_past_eof(self):
        with self.assertRaisesRegex(SensTruncationError, "frame 0 color payload"):
            self._index(_fixture_bytes(color_size=100, frame_count=1))

    def test_rejects_trailing_undeclared_bytes_with_exact_eof_rule(self):
        with self.assertRaisesRegex(SensFormatError, "trailing"):
            self._index(_fixture_bytes(suffix=b"unexpected"))

    def test_rejects_truncated_frame_with_frame_context(self):
        with self.assertRaisesRegex(SensTruncationError, "frame 1"):
            self._index(_fixture_bytes()[:-2])


if __name__ == "__main__":
    unittest.main()
