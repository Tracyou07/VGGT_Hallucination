from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from pre_experiments.local_global_consistency.context_source import (
    load_context_frame_ids,
)


class _GuardedArchive:
    files = [
        "frame_ids",
        "normalized_camera_tokens",
        "pred_c2w_raw",
        "gt_c2w_raw",
    ]

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def __getitem__(self, name):
        if name != "frame_ids":
            raise AssertionError(f"prediction member was read: {name}")
        return np.arange(500, dtype=np.int64)


class ContextFrameIdTest(unittest.TestCase):
    def test_load_context_frame_ids_reads_no_prediction_member(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "context_diagnostics.npz"
            path.touch()
            with mock.patch(
                "pre_experiments.local_global_consistency.context_source.np.load",
                return_value=_GuardedArchive(),
            ):
                frame_ids = load_context_frame_ids(path)

        np.testing.assert_array_equal(frame_ids, np.arange(500))

    def test_load_context_frame_ids_rejects_non_integer_or_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "context_diagnostics.npz"
            path.touch()
            for kind, values in (
                ("non_integer", np.array([0.0, 1.5])),
                ("duplicate", np.array([0, 0], dtype=np.int64)),
            ):
                with self.subTest(kind=kind):
                    with mock.patch(
                        "pre_experiments.local_global_consistency.context_source.np.load"
                    ) as loader:
                        archive = loader.return_value.__enter__.return_value
                        archive.files = ["frame_ids"]
                        archive.__getitem__.return_value = values
                        with self.assertRaises(ValueError):
                            load_context_frame_ids(path)


if __name__ == "__main__":
    unittest.main()
