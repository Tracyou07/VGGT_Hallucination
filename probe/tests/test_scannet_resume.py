import json
import tempfile
import unittest
from pathlib import Path

from experiments.scannet_hallucination.resume import load_completed_metrics


class ScanNetResumeTest(unittest.TestCase):
    def test_load_completed_metrics_adds_run_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            metrics_path = Path(tmp) / "metrics.json"
            metrics_path.write_text(
                json.dumps({"scene": "scene0000_00", "frame_count_actual": 500.0}),
                encoding="utf-8",
            )

            row = load_completed_metrics(metrics_path, requested_count=500, sampling="prefix")

        self.assertIsNotNone(row)
        self.assertEqual(row["scene"], "scene0000_00")
        self.assertEqual(row["frame_count_requested"], 500.0)
        self.assertEqual(row["sampling"], "prefix")

    def test_load_completed_metrics_ignores_invalid_or_missing_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            missing = load_completed_metrics(
                tmp_path / "missing.json", requested_count=100, sampling="prefix"
            )
            invalid_path = tmp_path / "metrics.json"
            invalid_path.write_text("{", encoding="utf-8")
            invalid = load_completed_metrics(invalid_path, requested_count=100, sampling="prefix")
            partial_path = tmp_path / "partial.json"
            partial_path.write_text("{}", encoding="utf-8")
            partial = load_completed_metrics(partial_path, requested_count=100, sampling="prefix")

        self.assertIsNone(missing)
        self.assertIsNone(invalid)
        self.assertIsNone(partial)


if __name__ == "__main__":
    unittest.main()
