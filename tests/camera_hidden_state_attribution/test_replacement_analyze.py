import unittest

from pre_experiments.camera_hidden_state_attribution.replacement_analyze import (
    summarize_replacement_rows,
)


class HiddenReplacementAnalyzeTest(unittest.TestCase):
    def test_summary_uses_scene_paired_selected_and_control_deltas(self):
        rows = []
        for scene, selected_delta, controls in (
            ("a", -0.4, (-0.1, 0.0)),
            ("b", -0.2, (0.1, -0.1)),
        ):
            rows.append(
                {
                    "scene": scene,
                    "condition": "baseline",
                    "aligned_translation_error_delta": 0.0,
                    "aligned_rotation_error_deg_delta": 0.0,
                }
            )
            rows.append(
                {
                    "scene": scene,
                    "condition": "selected",
                    "aligned_translation_error_delta": selected_delta,
                    "aligned_rotation_error_deg_delta": 0.0,
                }
            )
            for index, value in enumerate(controls):
                rows.append(
                    {
                        "scene": scene,
                        "condition": f"control_{index:02d}",
                        "aligned_translation_error_delta": value,
                        "aligned_rotation_error_deg_delta": 0.0,
                    }
                )

        summary = summarize_replacement_rows(rows)

        primary = summary["primary_translation_test"]
        self.assertAlmostEqual(
            primary["selected_delta"]["estimate"],
            -0.3,
        )
        self.assertAlmostEqual(
            primary["control_mean_delta"]["estimate"],
            -0.025,
        )
        self.assertAlmostEqual(
            primary["selected_minus_control"]["estimate"],
            -0.275,
        )
        self.assertEqual(primary["selected_improved_scene_fraction"], 1.0)
        self.assertEqual(primary["selected_beat_control_scene_fraction"], 1.0)


if __name__ == "__main__":
    unittest.main()
