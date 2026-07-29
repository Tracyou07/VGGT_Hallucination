import unittest

import numpy as np

from pre_experiments.camera_hidden_state_attribution.attribution import (
    contribution_drift,
    freeze_unit_sets,
    group_weight_norms,
)


class AttributionTest(unittest.TestCase):
    def test_group_weight_norms_and_drift_are_exact(self):
        weight = np.zeros((9, 2), dtype=np.float64)
        weight[:3, 0] = [3.0, 4.0, 0.0]
        weight[3:7, 1] = [0.0, 0.0, 0.0, 2.0]
        weight[7:, 1] = [3.0, 4.0]
        norms = group_weight_norms(weight)
        np.testing.assert_allclose(norms["translation"], [5.0, 0.0])
        np.testing.assert_allclose(norms["rotation"], [0.0, 2.0])
        np.testing.assert_allclose(norms["fov"], [0.0, 5.0])

        global_hidden = np.zeros((1, 2, 2), dtype=np.float64)
        local_hidden = np.array([[[2.0, 1.0], [4.0, 3.0]]])
        drift = contribution_drift(global_hidden, local_hidden, weight)
        np.testing.assert_allclose(drift["translation"], [[15.0, 0.0]])
        np.testing.assert_allclose(drift["rotation"], [[0.0, 4.0]])
        np.testing.assert_allclose(drift["fov"], [[0.0, 10.0]])

    def test_freeze_is_scene_equal_deterministic_and_iteration_matched(self):
        specificity = {
            "translation": np.array([1.0, 0.5, 0.25, 0.1, 0.1, 0.1]),
            "rotation": np.array([0.2, 1.0, 0.5, 0.1, 0.1, 0.1]),
            "fov": np.array([0.5, 0.25, 1.0, 0.1, 0.1, 0.1]),
        }
        scenes = []
        for multiplier in (1.0, 3.0):
            scenes.append(
                {
                    "scene": f"scene{int(multiplier)}",
                    "drift": {
                        "translation": multiplier
                        * np.array(
                            [[3.0, 2.0, 1.0, 0.0, 0.0, 0.0],
                             [1.0, 4.0, 2.0, 0.0, 0.0, 0.0]]
                        ),
                        "rotation": multiplier
                        * np.array(
                            [[1.0, 3.0, 2.0, 0.0, 0.0, 0.0],
                             [4.0, 1.0, 2.0, 0.0, 0.0, 0.0]]
                        ),
                        "fov": multiplier
                        * np.array(
                            [[2.0, 1.0, 3.0, 0.0, 0.0, 0.0],
                             [1.0, 2.0, 4.0, 0.0, 0.0, 0.0]]
                        ),
                    },
                    "specificity": specificity,
                }
            )

        first = freeze_unit_sets(scenes, top_k=2, seed=33)
        second = freeze_unit_sets(scenes, top_k=2, seed=33)
        self.assertEqual(first, second)
        self.assertEqual(
            first["selected"]["translation"],
            [{"iteration": 0, "unit": 0}, {"iteration": 1, "unit": 1}],
        )
        for group in ("translation", "rotation", "fov"):
            selected_counts = np.bincount(
                [item["iteration"] for item in first["selected"][group]],
                minlength=2,
            )
            control_counts = np.bincount(
                [item["iteration"] for item in first["controls"][group]],
                minlength=2,
            )
            np.testing.assert_array_equal(control_counts, selected_counts)


if __name__ == "__main__":
    unittest.main()
