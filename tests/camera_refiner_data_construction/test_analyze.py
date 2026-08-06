import tempfile
import unittest
from pathlib import Path

import numpy as np

from pre_experiments.camera_refiner_data_construction.analyze import (
    freeze_candidate_policy,
    summarize_scene_shards,
    validate_frozen_policy,
    write_numeric_summary,
)


def _scene(
    name: str,
    candidate_a: float,
    candidate_b: float,
) -> dict[str, object]:
    baseline = np.ones(4)
    return {
        "scene": name,
        "candidate_names": np.array(["baseline", "candidate_a", "candidate_b"]),
        "candidate_alpha": np.array([0.0, 0.02, 0.05]),
        "candidate_beta": np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.2, 0.3, 0.5]]
        ),
        "translation_error_aligned": np.stack(
            [baseline, baseline + candidate_a, baseline + candidate_b]
        ),
        "rotation_error_deg_aligned": np.stack(
            [baseline, baseline + 0.01, baseline - 0.02]
        ),
        "fov_change_mean": np.array([0.0, 0.001, 0.002]),
        "hidden_displacement_rms": np.array([0.0, 0.01, 0.02]),
    }


class MultiscaleAnalyzeTest(unittest.TestCase):
    def test_freezes_scene_robust_candidate_not_single_scene_oracle(self):
        summary = summarize_scene_shards(
            [
                _scene("scene_a", candidate_a=-0.2, candidate_b=-1.0),
                _scene("scene_b", candidate_a=-0.1, candidate_b=0.2),
            ],
            partition="calibration",
            bootstrap_samples=1000,
            seed=33,
        )

        rows = {row["candidate"]: row for row in summary["candidate_rows"]}
        self.assertAlmostEqual(rows["candidate_a"]["translation_delta_mean"], -0.15)
        self.assertAlmostEqual(
            rows["candidate_a"]["translation_delta_ci95_high"], -0.1
        )
        self.assertLess(rows["candidate_a"]["leave_one_out_max_mean"], 0.0)
        self.assertGreater(rows["candidate_b"]["leave_one_out_max_mean"], 0.0)

        frozen = freeze_candidate_policy(
            summary,
            split_digest="split",
            calibration_scenes=["scene_a", "scene_b"],
            source_run_id="source",
            scale_run_ids={100: "run100", 200: "run200", 300: "run300"},
            max_rotation_delta_deg=0.05,
            max_fov_change=0.01,
            min_improved_scene_fraction=0.5,
        )

        self.assertEqual(frozen["selected_candidate"], "candidate_a")
        self.assertEqual(frozen["selected_beta"], [1.0, 0.0, 0.0])
        self.assertEqual(frozen["selected_alpha"], 0.02)
        self.assertEqual(
            validate_frozen_policy(
                frozen,
                split_digest="split",
                calibration_scenes=["scene_a", "scene_b"],
                source_run_id="source",
                scale_run_ids={100: "run100", 200: "run200", 300: "run300"},
            )["selected_candidate"],
            "candidate_a",
        )

    def test_holdout_summary_refuses_extra_candidates_or_refit(self):
        calibration = summarize_scene_shards(
            [
                _scene("scene_a", -0.2, -1.0),
                _scene("scene_b", -0.1, 0.2),
            ],
            partition="calibration",
            bootstrap_samples=100,
        )
        frozen = freeze_candidate_policy(
            calibration,
            split_digest="split",
            calibration_scenes=["scene_a", "scene_b"],
            source_run_id="source",
            scale_run_ids={100: "run100", 200: "run200", 300: "run300"},
        )
        holdout = _scene("scene_holdout", -0.05, -0.2)
        with self.assertRaisesRegex(ValueError, "exactly baseline and frozen"):
            summarize_scene_shards(
                [holdout],
                partition="holdout",
                frozen_policy=frozen,
                bootstrap_samples=100,
            )

        for name in (
            "candidate_names",
            "candidate_alpha",
            "candidate_beta",
            "translation_error_aligned",
            "rotation_error_deg_aligned",
            "fov_change_mean",
            "hidden_displacement_rms",
        ):
            holdout[name] = np.asarray(holdout[name])[:2]
        summary = summarize_scene_shards(
            [holdout],
            partition="holdout",
            frozen_policy=frozen,
            bootstrap_samples=100,
        )
        self.assertEqual(summary["candidate_rows"][0]["candidate"], "candidate_a")

    def test_numeric_writer_contains_only_scalar_summary(self):
        summary = summarize_scene_shards(
            [_scene("scene_a", -0.2, -1.0), _scene("scene_b", -0.1, 0.2)],
            partition="calibration",
            bootstrap_samples=100,
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_numeric_summary(root, summary)
            csv_text = (root / "candidate_summary.csv").read_text(encoding="utf-8")
            json_text = (root / "candidate_summary.json").read_text(encoding="utf-8")

        self.assertIn("translation_delta_mean", csv_text)
        self.assertNotIn("gt_c2w_raw", csv_text)
        self.assertNotIn("translation_error_aligned\"", json_text)


if __name__ == "__main__":
    unittest.main()
