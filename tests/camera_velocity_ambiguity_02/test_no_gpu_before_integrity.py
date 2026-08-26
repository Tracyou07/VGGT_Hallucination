from __future__ import annotations

from datetime import timedelta
from pathlib import Path
import tempfile
import unittest

from pre_experiments.camera_velocity_ambiguity_02.contracts import ProtocolViolation
from pre_experiments.camera_velocity_ambiguity_02.input_gate import gate_gpu_launch
from tests.camera_velocity_ambiguity_02.test_input_gate import (
    REMOTE_ROOT,
    VERIFIED_AT,
    _marker,
    _scene_digest,
    _scenes,
    _write,
)


class NoGpuBeforeIntegrityTest(unittest.TestCase):
    def test_invalid_marker_never_calls_gpu_loader(self) -> None:
        calls: list[object] = []

        def gpu_loader(verified: object) -> None:
            calls.append(verified)
            raise AssertionError("GPU loader must not run")

        with tempfile.TemporaryDirectory() as directory:
            marker_path = Path(directory) / "invalid.json"
            marker = _marker()
            marker["asset_count"] = 99
            _write(marker_path, marker)

            with self.assertRaises(ProtocolViolation):
                gate_gpu_launch(
                    gpu_loader,
                    marker_path,
                    expected_remote_root=REMOTE_ROOT,
                    expected_scene_list_sha256=_scene_digest(),
                    expected_scenes=_scenes(),
                    now=VERIFIED_AT,
                    max_age=timedelta(days=7),
                )

        self.assertEqual(calls, [])

    def test_valid_marker_calls_loader_with_verified_contract(self) -> None:
        calls: list[object] = []

        def gpu_loader(verified: object) -> str:
            calls.append(verified)
            return "loaded"

        with tempfile.TemporaryDirectory() as directory:
            marker_path = Path(directory) / "valid.json"
            _write(marker_path, _marker())
            result = gate_gpu_launch(
                gpu_loader,
                marker_path,
                expected_remote_root=REMOTE_ROOT,
                expected_scene_list_sha256=_scene_digest(),
                expected_scenes=_scenes(),
                now=VERIFIED_AT,
                max_age=timedelta(days=7),
            )

        self.assertEqual(result, "loaded")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].asset_count, 100)


if __name__ == "__main__":
    unittest.main()
