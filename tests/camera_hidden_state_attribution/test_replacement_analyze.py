import unittest

from pre_experiments.camera_hidden_state_attribution.replacement_analyze import (
    select_calibration_alpha,
    summarize_replacement_rows,
)


class HiddenReplacementAnalyzeTest(unittest.TestCase):
    def test_summary_uses_scene_paired_selected_and_control_deltas(self):
        rows = []
        for scene, selected_deltas, control_deltas in (
            ("a", {0.1: -0.4, 1.0: 0.5}, {0.1: -0.1, 1.0: 0.0}),
            ("b", {0.1: -0.2, 1.0: 0.3}, {0.1: 0.0, 1.0: 0.1}),
        ):
            rows.append(
                {
                    "scene": scene,
                    "condition": "baseline",
                    "condition_family": "baseline",
                    "alpha": 0.0,
                    "aligned_translation_error_delta": 0.0,
                    "aligned_rotation_error_deg_delta": 0.0,
                }
            )
            for alpha in (0.1, 1.0):
                rows.append(
                    {
                        "scene": scene,
                        "condition": f"selected_{alpha}",
                        "condition_family": "selected",
                        "alpha": alpha,
                        "aligned_translation_error_delta": selected_deltas[
                            alpha
                        ],
                        "aligned_rotation_error_deg_delta": 0.0,
                    }
                )
                rows.append(
                    {
                        "scene": scene,
                        "condition": f"control_{alpha}",
                        "condition_family": "control",
                        "alpha": alpha,
                        "aligned_translation_error_delta": control_deltas[
                            alpha
                        ],
                        "aligned_rotation_error_deg_delta": 0.0,
                    }
                )

        summary = summarize_replacement_rows(rows)

        primary = summary["alpha_tests"][0]
        self.assertEqual(primary["alpha"], 0.1)
        self.assertAlmostEqual(
            primary["selected_delta"]["estimate"],
            -0.3,
        )
        self.assertAlmostEqual(
            primary["control_mean_delta"]["estimate"],
            -0.05,
        )
        self.assertAlmostEqual(
            primary["selected_minus_control"]["estimate"],
            -0.25,
        )
        self.assertEqual(primary["selected_improved_scene_fraction"], 1.0)
        self.assertEqual(primary["selected_beat_control_scene_fraction"], 1.0)
        self.assertEqual(summary["evaluated_control_repeats"], 1)
        selected = select_calibration_alpha(summary)
        self.assertEqual(selected["selected_alpha"], 0.1)
        self.assertAlmostEqual(
            selected["calibration_selected_delta"],
            -0.3,
        )

    def test_summary_averages_multiple_controls_within_each_scene(self):
        rows = []
        for scene, selected, controls in (
            ("a", -0.4, (-0.2, 0.0)),
            ("b", -0.2, (0.0, 0.2)),
        ):
            rows.extend(
                [
                    {
                        "scene": scene,
                        "condition": "baseline",
                        "condition_family": "baseline",
                        "alpha": 0.0,
                        "aligned_translation_error_delta": 0.0,
                    },
                    {
                        "scene": scene,
                        "condition": "selected_a0p02",
                        "condition_family": "selected",
                        "alpha": 0.02,
                        "aligned_translation_error_delta": selected,
                    },
                    *[
                        {
                            "scene": scene,
                            "condition": f"control_{index:02d}_a0p02",
                            "condition_family": "control",
                            "alpha": 0.02,
                            "aligned_translation_error_delta": delta,
                        }
                        for index, delta in enumerate(controls)
                    ],
                ]
            )

        summary = summarize_replacement_rows(rows)

        primary = summary["alpha_tests"][0]
        self.assertEqual(summary["evaluated_control_repeats"], 2)
        self.assertAlmostEqual(
            primary["control_mean_delta"]["estimate"],
            0.0,
        )
        self.assertAlmostEqual(
            primary["selected_minus_control"]["estimate"],
            -0.3,
        )


if __name__ == "__main__":
    unittest.main()
