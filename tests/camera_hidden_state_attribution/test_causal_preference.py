import unittest

import numpy as np

from pre_experiments.camera_hidden_state_attribution.causal_preference import (
    activation_rms,
    apply_causal_normalization,
    central_output_jacobians,
    fit_causal_normalization,
    project_hidden_effects,
)


class CausalPreferenceTest(unittest.TestCase):
    def test_activation_rms_uses_absolute_floor_for_inactive_units(self):
        hidden = np.asarray([[[3.0, 0.0], [4.0, 0.0]]])
        actual = activation_rms(
            hidden,
            floor_ratio=0.0,
            absolute_floor=0.25,
        )
        np.testing.assert_allclose(
            actual,
            [[np.sqrt(12.5), 0.25]],
        )

    def test_central_jacobians_and_projection_match_known_derivatives(self):
        iterations = 1
        basis_dimensions = 2
        frames = 2
        step = 0.25
        center_jacobian = np.zeros(
            (iterations, basis_dimensions, frames, 3)
        )
        center_jacobian[0, 0, :, 0] = 1.0
        center_jacobian[0, 1, :, 1] = 2.0

        rotation_jacobian = np.zeros(
            (iterations, basis_dimensions, frames, 3, 3)
        )
        rotation_jacobian[0, 0, :, 0, 1] = -1.0
        rotation_jacobian[0, 0, :, 1, 0] = 1.0

        fov_jacobian = np.zeros(
            (iterations, basis_dimensions, frames, 2)
        )
        fov_jacobian[0, 1, :, :] = [3.0, 4.0]

        baseline = {
            "camera_center": np.zeros((frames, 3)),
            "rotation": np.tile(np.eye(3), (frames, 1, 1)),
            "fov": np.zeros((frames, 2)),
        }
        jacobians = {
            "camera_center": center_jacobian,
            "rotation": rotation_jacobian,
            "fov": fov_jacobian,
        }
        positive = {
            name: baseline[name][None, None] + step * values
            for name, values in jacobians.items()
        }
        negative = {
            name: baseline[name][None, None] - step * values
            for name, values in jacobians.items()
        }

        recovered = central_output_jacobians(
            positive,
            negative,
            basis_step=step,
        )
        for name in jacobians:
            np.testing.assert_allclose(recovered[name], jacobians[name])

        effects = project_hidden_effects(
            recovered,
            baseline_rotations=baseline["rotation"],
            output_weight=np.eye(2),
            activation_scales=np.asarray([[2.0, 0.5]]),
            unit_chunk_size=1,
        )
        np.testing.assert_allclose(effects["translation"], [[2.0, 1.0]])
        np.testing.assert_allclose(
            effects["rotation"],
            [[np.degrees(2.0), 0.0]],
        )
        np.testing.assert_allclose(effects["fov"], [[0.0, 2.5]])

    def test_normalization_scales_are_frozen_before_preferences(self):
        calibration = {
            "translation": np.asarray([[1.0, 3.0]]),
            "rotation": np.asarray([[2.0, 6.0]]),
            "fov": np.asarray([[4.0, 12.0]]),
        }
        scales = fit_causal_normalization(
            calibration,
            quantile=0.5,
            minimum_scale=0.1,
        )
        self.assertEqual(
            scales,
            {"translation": 2.0, "rotation": 4.0, "fov": 8.0},
        )

        holdout = {
            "translation": np.asarray([[4.0, 1.0]]),
            "rotation": np.asarray([[2.0, 6.0]]),
            "fov": np.asarray([[1.0, 3.0]]),
        }
        normalized = apply_causal_normalization(
            holdout,
            {"translation": 1.0, "rotation": 1.0, "fov": 1.0},
        )
        np.testing.assert_allclose(
            normalized["preferences"]["translation"],
            [[4.0 / 7.0, 0.1]],
        )
        np.testing.assert_allclose(
            normalized["preferences"]["rotation"],
            [[2.0 / 7.0, 0.6]],
        )
        np.testing.assert_allclose(
            normalized["preferences"]["fov"],
            [[1.0 / 7.0, 0.3]],
        )
        np.testing.assert_array_equal(
            normalized["preferred_group"],
            [["translation", "rotation"]],
        )

    def test_invalid_shapes_and_nonfinite_values_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "hidden"):
            activation_rms(np.zeros((2, 3)))
        with self.assertRaisesRegex(ValueError, "finite"):
            activation_rms(np.asarray([[[np.nan]]]))
        with self.assertRaisesRegex(ValueError, "basis_step"):
            central_output_jacobians(
                {
                    "camera_center": np.zeros((1, 1, 1, 3)),
                    "rotation": np.zeros((1, 1, 1, 3, 3)),
                    "fov": np.zeros((1, 1, 1, 2)),
                },
                {
                    "camera_center": np.zeros((1, 1, 1, 3)),
                    "rotation": np.zeros((1, 1, 1, 3, 3)),
                    "fov": np.zeros((1, 1, 1, 2)),
                },
                basis_step=0.0,
            )


if __name__ == "__main__":
    unittest.main()
