import json
import unittest

from pre_experiments.local_global_consistency.aggregate import (
    bootstrap_holdout,
    summarize_holdout_scenes,
)


def _rows():
    scores = []
    validation = []
    for scene_index in range(4):
        scene = f"scene{scene_index:04d}_00"
        for frame_id in range(5):
            value = float(scene_index + frame_id + 1)
            reliable = frame_id < 4
            scores.append(
                {
                    "scene": scene,
                    "frame_id": frame_id,
                    "global_local_token_cosine": value,
                    "global_local_pose_translation": value * 2.0,
                    "global_local_pose_rotation_deg": value * 3.0,
                    "local_local_token_cosine": value / 10.0,
                    "local_local_pose_translation": value / 5.0,
                    "local_local_pose_rotation_deg": value / 2.0,
                    "token_local_reliable": reliable,
                    "pose_local_reliable": reliable,
                }
            )
            translation_growth = value - 2.0
            rotation_growth = value * 2.0 - 5.0
            validation.append(
                {
                    "scene": scene,
                    "frame_id": frame_id,
                    "translation_error_growth_global_minus_local": translation_growth,
                    "rotation_error_growth_global_minus_local_deg": rotation_growth,
                }
            )
    return scores, validation


class HoldoutAggregateTest(unittest.TestCase):
    def test_scene_summary_contains_growth_coverage_and_correlations(self):
        scores, validation = _rows()

        rows = summarize_holdout_scenes(scores, validation)

        self.assertEqual(len(rows), 4)
        first = rows[0]
        self.assertEqual(first["frame_count"], 5)
        self.assertAlmostEqual(first["translation_growth_mean"], 1.0)
        self.assertAlmostEqual(first["translation_growth_median"], 1.0)
        self.assertAlmostEqual(first["translation_growth_positive_fraction"], 0.6)
        self.assertAlmostEqual(first["token_reliable_coverage"], 0.8)
        self.assertAlmostEqual(first["pose_reliable_coverage"], 0.8)
        self.assertAlmostEqual(
            first[
                "global_local_token_cosine_vs_translation_growth_pearson"
            ],
            1.0,
        )
        self.assertAlmostEqual(
            first[
                "global_local_pose_rotation_deg_vs_rotation_growth_spearman"
            ],
            1.0,
        )

    def test_bootstrap_is_scene_level_finite_and_deterministic(self):
        scores, validation = _rows()
        scene_rows = summarize_holdout_scenes(scores, validation)

        first = bootstrap_holdout(scene_rows, samples=500, seed=33)
        second = bootstrap_holdout(scene_rows, samples=500, seed=33)

        self.assertEqual(
            json.dumps(first, sort_keys=True),
            json.dumps(second, sort_keys=True),
        )
        metrics = {row["metric"]: row for row in first}
        for name in (
            "translation_growth_mean",
            "translation_growth_median",
            "translation_growth_positive_fraction",
            "rotation_growth_mean",
            "rotation_growth_median",
            "rotation_growth_positive_fraction",
            "token_reliable_coverage",
            "pose_reliable_coverage",
        ):
            row = metrics[name]
            self.assertLessEqual(row["ci95_low"], row["estimate"])
            self.assertLessEqual(row["estimate"], row["ci95_high"])
            self.assertEqual(row["bootstrap_unit"], "scene")


if __name__ == "__main__":
    unittest.main()
