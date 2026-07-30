import unittest

from pre_experiments.camera_hidden_state_attribution.adaptive_alpha import (
    FEATURE_FIELDS,
    evaluate_leave_one_out,
)
from pre_experiments.camera_hidden_state_attribution.adaptive_replacement import (
    compare_adaptive_to_fixed,
    summarize_adaptive_rows,
)


class AdaptiveReplacementTest(unittest.TestCase):
    def test_leave_one_out_reports_adaptive_fixed_and_oracle_deltas(self):
        features = []
        replacement_rows = []
        labels = {}
        for index in range(4):
            scene = f"scene_{index}"
            features.append(
                {
                    "scene": scene,
                    FEATURE_FIELDS[0]: float(index),
                    FEATURE_FIELDS[1]: float(index),
                    FEATURE_FIELDS[2]: float(3 - index),
                }
            )
            oracle = (0.01, 0.01, 0.05, 0.05)[index]
            labels[scene] = oracle
            for alpha in (0.01, 0.02, 0.05):
                replacement_rows.append(
                    {
                        "scene": scene,
                        "condition_family": "selected",
                        "alpha": alpha,
                        "aligned_translation_error_delta": (
                            abs(alpha - oracle)
                        ),
                    }
                )

        report = evaluate_leave_one_out(
            features,
            labels,
            replacement_rows,
            split_digest="split",
            score_run_id="scores",
            replacement_run_id="replacement",
        )

        self.assertEqual(report["scene_count"], 4)
        self.assertEqual(len(report["rows"]), 4)
        self.assertTrue(
            all(row["training_scene_count"] == 3 for row in report["rows"])
        )
        self.assertTrue(
            all(row["oracle_delta"] == 0.0 for row in report["rows"])
        )
        self.assertAlmostEqual(
            report["fixed_alpha_delta_mean"],
            0.02,
        )

    def test_adaptive_summary_pairs_scene_specific_alphas_and_controls(self):
        rows = []
        for scene, alpha, selected, controls in (
            ("a", 0.01, -0.2, (-0.1, 0.0)),
            ("b", 0.05, -0.4, (0.1, -0.1)),
        ):
            rows.append(
                {
                    "scene": scene,
                    "condition": "baseline",
                    "condition_family": "baseline",
                    "alpha": 0.0,
                    "aligned_translation_error_delta": 0.0,
                }
            )
            rows.append(
                {
                    "scene": scene,
                    "condition": f"selected_a{alpha}",
                    "condition_family": "selected",
                    "alpha": alpha,
                    "aligned_translation_error_delta": selected,
                }
            )
            for index, delta in enumerate(controls):
                rows.append(
                    {
                        "scene": scene,
                        "condition": f"control_{index:02d}_a{alpha}",
                        "condition_family": "control",
                        "alpha": alpha,
                        "aligned_translation_error_delta": delta,
                    }
                )

        summary = summarize_adaptive_rows(rows)

        self.assertEqual(
            summary["alpha_scene_counts"],
            {"0.01": 1, "0.05": 1},
        )
        self.assertEqual(summary["evaluated_control_repeats"], 2)
        self.assertAlmostEqual(
            summary["selected_delta"]["estimate"],
            -0.3,
        )
        self.assertAlmostEqual(
            summary["control_mean_delta"]["estimate"],
            -0.025,
        )
        self.assertAlmostEqual(
            summary["selected_minus_control"]["estimate"],
            -0.275,
        )

        fixed_rows = [
            {
                "scene": "a",
                "condition_family": "selected",
                "alpha": 0.02,
                "aligned_translation_error_delta": -0.1,
            },
            {
                "scene": "b",
                "condition_family": "selected",
                "alpha": 0.02,
                "aligned_translation_error_delta": -0.3,
            },
        ]
        comparison = compare_adaptive_to_fixed(rows, fixed_rows)
        self.assertAlmostEqual(
            comparison["adaptive_minus_fixed"]["estimate"],
            -0.1,
        )
        self.assertEqual(
            comparison["adaptive_beat_fixed_scene_fraction"],
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
