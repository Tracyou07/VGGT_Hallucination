from __future__ import annotations

import unittest

import numpy as np

from pre_experiments.camera_velocity_ambiguity_02.data import (
    observation_calibration,
    select_sensor_frame_ids,
)


class SelectedSensorDataTest(unittest.TestCase):
    def test_selects_exact_fastvggt_ids_from_finite_sensor_poses(self) -> None:
        finite = np.ones(12, dtype=bool)
        finite[[3, 7]] = False
        selected = select_sensor_frame_ids(finite, input_frames=5)
        self.assertEqual(selected, (0, 1, 4, 6, 9))

    def test_observation_calibration_requires_registered_streams(self) -> None:
        color = np.array(
            [[100.0, 0.0, 49.5, 0.0], [0.0, 100.0, 39.5, 0.0], [0, 0, 1, 0], [0, 0, 0, 1]]
        )
        depth = np.array(
            [[50.0, 0.0, 24.75, 0.0], [0.0, 50.0, 19.75, 0.0], [0, 0, 1, 0], [0, 0, 0, 1]]
        )
        intrinsics = observation_calibration(
            color_intrinsic=color,
            depth_intrinsic=depth,
            color_extrinsic=np.eye(4),
            depth_extrinsic=np.eye(4),
            color_hw=(80, 100),
            depth_hw=(40, 50),
            observation_hw=(20, 25),
        )
        np.testing.assert_allclose(
            intrinsics,
            [[25.0, 0.0, 12.375], [0.0, 25.0, 9.875], [0.0, 0.0, 1.0]],
        )

        wrong = np.eye(4)
        wrong[0, 3] = 0.1
        with self.assertRaisesRegex(ValueError, "registered"):
            observation_calibration(
                color_intrinsic=color,
                depth_intrinsic=depth,
                color_extrinsic=np.eye(4),
                depth_extrinsic=wrong,
                color_hw=(80, 100),
                depth_hw=(40, 50),
                observation_hw=(20, 25),
            )


if __name__ == "__main__":
    unittest.main()
