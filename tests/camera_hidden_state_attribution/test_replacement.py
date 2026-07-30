import unittest

import numpy as np

from pre_experiments.camera_hidden_state_attribution.replacement import (
    assemble_short_hidden,
    freeze_replacement_manifest,
    replacement_mask,
)


class HiddenReplacementProtocolTest(unittest.TestCase):
    def test_freezes_calibration_intersection_and_matched_controls(self):
        old_rows = []
        old_order = ((0, 0), (0, 1), (1, 0), (0, 2), (1, 1), (1, 2))
        for rank, (iteration, unit) in enumerate(old_order, start=1):
            old_rows.append(
                {
                    "group": "translation",
                    "iteration": str(iteration),
                    "unit": str(unit),
                    "partition_rank": str(rank),
                }
            )
        causal_scores = {
            (0, 0): 4.0,
            (0, 1): 9.0,
            (0, 2): 2.0,
            (0, 3): 1.0,
            (1, 0): 8.0,
            (1, 1): 7.0,
            (1, 2): 3.0,
            (1, 3): 0.5,
        }
        causal_rows = [
            {
                "iteration": str(iteration),
                "unit": str(unit),
                "translation_effect_mean": str(score),
            }
            for (iteration, unit), score in causal_scores.items()
        ]

        frozen = freeze_replacement_manifest(
            old_rows,
            causal_rows,
            split_digest="split",
            calibration_scenes=["scene_a", "scene_b"],
            source_top_k=3,
            control_repeats=2,
            seed=7,
        )

        selected = {
            (item["iteration"], item["unit"])
            for item in frozen["selected"]
        }
        self.assertEqual(selected, {(0, 1), (1, 0)})
        self.assertEqual(frozen["selected_count"], 2)
        self.assertEqual(len(frozen["control_sets"]), 2)
        source_union = {(0, 0), (0, 1), (1, 0), (1, 1)}
        for control in frozen["control_sets"]:
            positions = {
                (item["iteration"], item["unit"])
                for item in control["positions"]
            }
            self.assertEqual(
                sorted(iteration for iteration, _ in positions),
                [0, 1],
            )
            self.assertTrue(positions.isdisjoint(source_union))
        self.assertIsInstance(frozen["frozen_digest"], str)

    def test_assembles_hidden_from_most_interior_window(self):
        first = np.zeros((2, 4, 3), dtype=np.float32)
        second = np.zeros((2, 4, 3), dtype=np.float32)
        for local_index in range(4):
            first[:, local_index] = 10 + local_index
            second[:, local_index] = 20 + local_index

        assembled = assemble_short_hidden(
            np.arange(5),
            [
                {
                    "window_index": 0,
                    "frame_ids": np.arange(4),
                    "hidden": first,
                },
                {
                    "window_index": 1,
                    "frame_ids": np.arange(1, 5),
                    "hidden": second,
                },
            ],
        )

        np.testing.assert_array_equal(
            assembled["selected_window_index"],
            np.array([0, 0, 0, 1, 1]),
        )
        np.testing.assert_allclose(
            assembled["hidden"][0, :, 0],
            np.array([10, 11, 12, 22, 23]),
        )

    def test_replacement_mask_rejects_out_of_range_position(self):
        frozen = {
            "selected": [{"iteration": 0, "unit": 2}],
            "control_sets": [
                {
                    "name": "control_00",
                    "positions": [{"iteration": 1, "unit": 4}],
                }
            ],
        }
        selected = replacement_mask(
            frozen,
            "selected",
            iterations=2,
            hidden_dim=5,
        )
        self.assertTrue(selected[0, 2])
        with self.assertRaisesRegex(ValueError, "out of range"):
            replacement_mask(
                frozen,
                "control_00",
                iterations=1,
                hidden_dim=4,
            )


if __name__ == "__main__":
    unittest.main()
