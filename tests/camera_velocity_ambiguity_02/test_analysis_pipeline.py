from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from pre_experiments.camera_velocity_ambiguity_02.analyze import (
    analyze_pair_records,
    publish_scene_records,
)
from pre_experiments.camera_velocity_ambiguity_02.events import EventClass, EventPolicy


class AnalysisPipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.prediction = {
            "sample_id": "scene0000_00/window_000__window_001",
            "scene": "scene0000_00",
            "pair_id": "scene0000_00/window_000__window_001",
            "route": "primary",
            "alignment_valid": True,
            "direction_evaluable": True,
            "flattened_cosine": -0.8,
            "normalized_separation": 0.2,
        }
        self.privileged = {
            "sample_id": self.prediction["sample_id"],
            "left_endpoint_valid": True,
            "right_endpoint_valid": True,
            "global_rms": 2.0,
            "left_rms": 1.0,
            "right_rms": 1.1,
        }
        self.rgbd = {
            "sample_id": self.prediction["sample_id"],
            "rgbd_valid": True,
            "interior_barrier": 0.3,
            "temporal_support": True,
        }

    def test_joins_prediction_privileged_and_rgbd_only_by_sample_id(self) -> None:
        decision = analyze_pair_records(
            self.prediction,
            self.privileged,
            self.rgbd,
            EventPolicy(-0.2, 0.05, 0.1),
        )
        self.assertEqual(decision["event_class"], EventClass.MULTIMODAL_VELOCITY_SUPPORTED.value)
        self.assertEqual(decision["sample_id"], self.prediction["sample_id"])

        bad = dict(self.rgbd, sample_id="different")
        with self.assertRaisesRegex(ValueError, "sample ID"):
            analyze_pair_records(
                self.prediction,
                self.privileged,
                bad,
                EventPolicy(-0.2, 0.05, 0.1),
            )

    def test_publishes_separate_prediction_and_privileged_sidecars(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = publish_scene_records(
                Path(directory),
                scene="scene0000_00",
                prediction_rows=[self.prediction],
                privileged_rows=[self.privileged],
                rgbd_rows=[self.rgbd],
                decision_rows=[{"sample_id": self.prediction["sample_id"], "event_class": "X"}],
            )
            self.assertEqual(manifest["counts"]["prediction_only"], 1)
            prediction_text = (Path(directory) / "prediction_only.jsonl").read_text()
            privileged_text = (Path(directory) / "privileged_labels.jsonl").read_text()
            self.assertNotIn("endpoint", prediction_text)
            self.assertIn("left_endpoint_valid", privileged_text)
            loaded = json.loads((Path(directory) / "records_manifest.json").read_text())
            self.assertEqual(loaded["manifest_digest"], manifest["manifest_digest"])


if __name__ == "__main__":
    unittest.main()
