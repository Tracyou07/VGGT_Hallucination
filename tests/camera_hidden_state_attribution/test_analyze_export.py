import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from pre_experiments.camera_hidden_state_attribution.analyze import (
    intervention_metrics,
    unit_mask,
    write_numeric_summary,
)
from scripts.autodl.camera_hidden_state_attribution.export_numeric_results import (
    export_numeric_results,
)


class AnalyzeExportTest(unittest.TestCase):
    def test_unit_mask_and_component_metrics(self):
        frozen = {
            "selected": {
                "translation": [
                    {"iteration": 0, "unit": 2},
                    {"iteration": 3, "unit": 1},
                ]
            }
        }
        mask = unit_mask(frozen, "translation", "selected", 4, 5)
        self.assertEqual(mask.shape, (4, 5))
        self.assertTrue(mask[0, 2])
        self.assertTrue(mask[3, 1])
        self.assertEqual(int(mask.sum()), 2)

        baseline = np.tile(np.eye(4), (3, 1, 1))
        changed = baseline.copy()
        changed[:, 0, 3] = [1.0, 2.0, 3.0]
        baseline_pose = np.zeros((3, 9))
        changed_pose = baseline_pose.copy()
        changed_pose[:, 7:] = 0.5
        metrics = intervention_metrics(
            baseline, changed, baseline_pose, changed_pose
        )
        self.assertAlmostEqual(metrics["camera_center_displacement_mean"], 2.0)
        self.assertAlmostEqual(metrics["rotation_change_deg_mean"], 0.0)
        self.assertAlmostEqual(metrics["fov_change_mean"], np.sqrt(0.5))

    def test_numeric_summary_and_export_reject_raw_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "run"
            run.mkdir()
            frozen = {
                "frozen_digest": "digest",
                "selected": {"translation": [], "rotation": [], "fov": []},
                "controls": {"translation": [], "rotation": [], "fov": []},
                "scores": {"translation": [], "rotation": [], "fov": []},
            }
            rows = [
                {
                    "scene": "a",
                    "group": "translation",
                    "set": "selected",
                    "camera_center_displacement_mean": 1.0,
                    "rotation_change_deg_mean": 0.1,
                    "fov_change_mean": 0.01,
                    "aligned_translation_error_mean": 0.8,
                    "aligned_translation_error_delta": -0.2,
                },
                {
                    "scene": "b",
                    "group": "translation",
                    "set": "selected",
                    "camera_center_displacement_mean": 3.0,
                    "rotation_change_deg_mean": 0.3,
                    "fov_change_mean": 0.03,
                    "aligned_translation_error_mean": 1.8,
                    "aligned_translation_error_delta": -0.1,
                },
            ]
            write_numeric_summary(run, frozen, rows, partition="holdout")
            complete = {
                "run_id": "run",
                "partition": "holdout",
                "analysis_complete": True,
                "protocol_complete": True,
                "frozen_digest": "digest",
            }
            (run / "complete.json").write_text(json.dumps(complete))
            (run / "run_metadata.json").write_text(
                json.dumps(
                    {
                        "run_id": "run",
                        "study_name": "camera_hidden_state_attribution",
                        "partition": "holdout",
                    }
                )
            )
            destination = export_numeric_results(run, root / "published")
            self.assertTrue((destination / "summary.json").is_file())
            (run / "raw_trace.npz").write_bytes(b"not allowed")
            with self.assertRaisesRegex(ValueError, "unexpected"):
                export_numeric_results(run, root / "published2")


if __name__ == "__main__":
    unittest.main()
