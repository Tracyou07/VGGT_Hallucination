from __future__ import annotations

from pathlib import Path
import json
import tempfile
import unittest

from pre_experiments.conditional_hierarchical_vrfm.report import write_stage_a_report


class ReportTests(unittest.TestCase):
    def test_report_is_recomputed_and_existing_tampering_fails(self) -> None:
        payload = {
            "schema": "conditional_hierarchical_vrfm.stage_a_report.v1",
            "git_commit": "a" * 40,
            "classification": "LATENT_LIFT_FAILED",
            "failed_gates": ["positive_mean"],
            "scene_metrics": [{"scene": "scene0000_00", "mean_full_scene_utility": -0.1}],
            "provenance": {"checkpoint_sha256": "b" * 64},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            json_path, markdown_path = write_stage_a_report(root, payload)
            self.assertEqual(json.loads(json_path.read_text()), payload)
            self.assertIn("LATENT_LIFT_FAILED", markdown_path.read_text())
            json_path.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "existing report"):
                write_stage_a_report(root, payload)


if __name__ == "__main__":
    unittest.main()
