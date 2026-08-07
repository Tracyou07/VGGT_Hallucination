import unittest

import numpy as np

from pre_experiments.camera_refiner_training.geometry import (
    SceneGauge,
    align_centers_to_reference,
    apply_center_corrections,
    fuse_window_corrections,
)


def poses_from_centers(centers: np.ndarray) -> np.ndarray:
    poses = np.tile(np.eye(4), (len(centers), 1, 1))
    poses[:, :3, 3] = centers
    return poses


class SceneGaugeTest(unittest.TestCase):
    def test_canonical_round_trip_uses_prediction_only_scene_gauge(self):
        centers = np.stack(
            [np.linspace(2.0, 8.0, 100), np.linspace(-1.0, 1.0, 100), np.zeros(100)],
            axis=1,
        )
        poses = poses_from_centers(centers)
        gauge = SceneGauge.from_c2w(poses)

        canonical = gauge.canonicalize(centers)

        np.testing.assert_allclose(gauge.restore(canonical), centers, atol=1e-10)
        np.testing.assert_allclose(canonical[0], np.zeros(3), atol=1e-10)
        self.assertGreater(gauge.scale, 0.0)

    def test_prediction_alignment_recovers_similarity_transform(self):
        reference = np.stack(
            [np.linspace(0.0, 3.0, 100), np.sin(np.linspace(0, 2, 100)), np.zeros(100)],
            axis=1,
        )
        moving = reference * 2.5 + np.asarray([7.0, -3.0, 1.0])

        aligned, residual = align_centers_to_reference(moving, reference)

        np.testing.assert_allclose(aligned, reference, atol=1e-9)
        np.testing.assert_allclose(residual, np.zeros(100), atol=1e-9)

    def test_fusion_tapers_boundaries_and_preserves_global_rotations(self):
        corrections = np.zeros((2, 4, 3), dtype=np.float64)
        corrections[0, :, 0] = 1.0
        corrections[1, :, 0] = 3.0
        confidence = np.ones((2, 4), dtype=np.float64)

        fused, fused_confidence = fuse_window_corrections(
            corrections,
            confidence,
            starts=np.asarray([0, 2]),
            total_frames=6,
        )
        global_poses = poses_from_centers(np.zeros((6, 3)))
        theta = 0.3
        rotation = np.asarray(
            [[np.cos(theta), -np.sin(theta), 0.0], [np.sin(theta), np.cos(theta), 0.0], [0.0, 0.0, 1.0]]
        )
        global_poses[:, :3, :3] = rotation

        corrected = apply_center_corrections(global_poses, fused)

        self.assertTrue(np.all((fused_confidence >= 0.0) & (fused_confidence <= 1.0)))
        self.assertGreater(fused[2, 0], 1.0)
        self.assertLess(fused[2, 0], 3.0)
        np.testing.assert_array_equal(corrected[:, :3, :3], global_poses[:, :3, :3])
        np.testing.assert_allclose(corrected[:, :3, 3], fused)


if __name__ == "__main__":
    unittest.main()
