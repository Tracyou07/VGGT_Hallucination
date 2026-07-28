import json
from pathlib import Path
import tempfile
import unittest

from pre_experiments.local_global_consistency.thresholds import (
    fit_frozen_thresholds,
    load_frozen_thresholds,
)


def _provenance() -> dict[str, object]:
    return {
        "calibration_scenes": [f"scene{index:04d}_00" for index in range(10)],
        "source_run_id": "source-run",
        "calibration_run_id": "calibration-run",
        "split_digest": "a" * 64,
        "code_commit": "b" * 40,
    }


def _score_rows() -> list[dict[str, object]]:
    rows = []
    for scene_index, scene in enumerate(_provenance()["calibration_scenes"]):
        for sample_index in range(4):
            value = float(scene_index * 4 + sample_index + 1)
            rows.append(
                {
                    "scene": scene,
                    "frame_id": sample_index,
                    "local_local_token_cosine": value,
                    "local_local_pose_translation": value * 2.0,
                    "local_local_pose_rotation_deg": value * 3.0,
                }
            )
    rows.append(
        {
            "scene": _provenance()["calibration_scenes"][0],
            "frame_id": 99,
            "local_local_token_cosine": None,
            "local_local_pose_translation": None,
            "local_local_pose_rotation_deg": None,
        }
    )
    return rows


class FrozenThresholdTest(unittest.TestCase):
    def test_fit_records_prediction_only_values_counts_and_provenance(self):
        payload = fit_frozen_thresholds(_score_rows(), _provenance())

        self.assertEqual(
            set(payload["thresholds"]),
            {
                "token_cosine_p95",
                "pose_translation_p95",
                "pose_rotation_deg_p95",
            },
        )
        self.assertEqual(
            payload["sample_counts"],
            {
                "token_cosine_p95": 40,
                "pose_translation_p95": 40,
                "pose_rotation_deg_p95": 40,
            },
        )
        self.assertEqual(payload["calibration_scenes"], _provenance()["calibration_scenes"])
        self.assertEqual(payload["source_run_id"], "source-run")
        self.assertEqual(payload["calibration_run_id"], "calibration-run")
        self.assertEqual(len(payload["threshold_digest"]), 64)

    def test_fit_rejects_scene_leakage_and_gt_named_score_fields(self):
        rows = _score_rows()
        rows.append(
            {
                "scene": "holdout-scene",
                "frame_id": 1,
                "local_local_token_cosine": 1.0,
                "local_local_pose_translation": 1.0,
                "local_local_pose_rotation_deg": 1.0,
            }
        )
        with self.assertRaisesRegex(ValueError, "scene set"):
            fit_frozen_thresholds(rows, _provenance())

        rows = _score_rows()
        rows[0]["gt_error"] = 123.0
        with self.assertRaisesRegex(ValueError, "GT-derived"):
            fit_frozen_thresholds(rows, _provenance())

    def test_fit_rejects_missing_prediction_metric(self):
        rows = _score_rows()
        del rows[0]["local_local_pose_translation"]

        with self.assertRaisesRegex(ValueError, "missing"):
            fit_frozen_thresholds(rows, _provenance())

    def test_load_rejects_tampering_or_provenance_mismatch(self):
        payload = fit_frozen_thresholds(_score_rows(), _provenance())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "thresholds.json"
            path.write_text(json.dumps(payload), encoding="utf-8")

            loaded = load_frozen_thresholds(
                path,
                expected_split_digest="a" * 64,
                expected_source_run_id="source-run",
            )
            self.assertEqual(loaded, payload)

            payload["thresholds"]["token_cosine_p95"] = 999.0
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "digest"):
                load_frozen_thresholds(
                    path,
                    expected_split_digest="a" * 64,
                    expected_source_run_id="source-run",
                )

            path.write_text(json.dumps(loaded), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "split"):
                load_frozen_thresholds(
                    path,
                    expected_split_digest="c" * 64,
                    expected_source_run_id="source-run",
                )


if __name__ == "__main__":
    unittest.main()
