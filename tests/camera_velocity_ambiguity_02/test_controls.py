from __future__ import annotations

import unittest

import numpy as np

from pre_experiments.camera_velocity_ambiguity_02.controls import (
    CONTROL_NAMES,
    build_negative_controls,
)


class NegativeControlTest(unittest.TestCase):
    def test_builds_all_frozen_controls_through_signed_residual_metrics(self) -> None:
        left = np.tile(np.asarray([[1.0, 0.0, 0.0]]), (6, 1))
        right = np.tile(np.asarray([[-1.0, 0.0, 0.0]]), (6, 1))
        wrong = np.arange(18, dtype=np.float64).reshape(6, 3) / 10.0
        first = build_negative_controls(
            left, right, wrong_window_residual=wrong, scene_scale=2.0, seed=33
        )
        second = build_negative_controls(
            left, right, wrong_window_residual=wrong, scene_scale=2.0, seed=33
        )

        self.assertEqual(set(first), CONTROL_NAMES)
        self.assertAlmostEqual(first["self"].metrics.normalized_rms_separation, 0.0)
        self.assertFalse(first["gauge_copy"].metrics.direction_evaluable)
        self.assertLess(first["epsilon"].metrics.normalized_rms_separation, 1e-4)
        self.assertAlmostEqual(first["sign_inversion"].metrics.flattened_cosine, -1.0)
        self.assertFalse(first["degenerate_alignment"].alignment_valid)
        np.testing.assert_array_equal(
            first["random_wrong_window"].metrics.right_residual,
            second["random_wrong_window"].metrics.right_residual,
        )

    def test_rejects_shape_or_nonfinite_control_inputs(self) -> None:
        left = np.ones((3, 3))
        with self.assertRaises(ValueError):
            build_negative_controls(
                left, np.ones((2, 3)), wrong_window_residual=left, scene_scale=1.0
            )
        left[0, 0] = np.nan
        with self.assertRaises(ValueError):
            build_negative_controls(
                left, np.ones((3, 3)), wrong_window_residual=np.ones((3, 3)), scene_scale=1.0
            )


if __name__ == "__main__":
    unittest.main()
