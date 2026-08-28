from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from pre_experiments.long_short_camera_head.pipeline import (
    DEFAULT_RESULT_ROOT,
    verify_completed_run,
)


class PipelineContractTests(unittest.TestCase):
    def test_default_result_root_is_under_vggt(self) -> None:
        self.assertEqual(
            DEFAULT_RESULT_ROOT,
            Path("/data/yjh/output/vggt/long_short_camera_head"),
        )

    def test_completion_verification_fails_closed_on_partial_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "incomplete"):
                verify_completed_run(Path(directory))


if __name__ == "__main__":
    unittest.main()
