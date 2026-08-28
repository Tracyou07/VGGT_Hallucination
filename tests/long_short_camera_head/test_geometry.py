from __future__ import annotations

import unittest

import torch

from pre_experiments.long_short_camera_head.geometry import (
    apply_sim3_torch,
    relative_translation_loss,
    rotation_matrix_loss,
)


class DifferentiableGeometryTests(unittest.TestCase):
    def test_apply_sim3_maps_centers_and_keeps_gradient(self) -> None:
        poses = torch.eye(4).repeat(3, 1, 1)
        poses[:, 0, 3] = torch.tensor([0.0, 1.0, 2.0])
        poses.requires_grad_()

        aligned = apply_sim3_torch(
            poses,
            scale=torch.tensor(2.0),
            rotation=torch.eye(3),
            translation=torch.tensor([1.0, 2.0, 3.0]),
        )

        torch.testing.assert_close(
            aligned[:, :3, 3],
            torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 3.0], [5.0, 2.0, 3.0]]),
        )
        aligned[:, :3, 3].sum().backward()
        self.assertIsNotNone(poses.grad)
        self.assertGreater(float(poses.grad.abs().sum()), 0.0)

    def test_rotation_loss_is_zero_only_for_matching_rotations(self) -> None:
        target = torch.eye(3).repeat(2, 1, 1)
        predicted = target.clone()
        self.assertEqual(float(rotation_matrix_loss(predicted, target)), 0.0)

        predicted[1, 0, 0] = -1.0
        self.assertGreater(float(rotation_matrix_loss(predicted, target)), 0.0)

    def test_relative_translation_loss_checks_requested_lags(self) -> None:
        target = torch.zeros(1, 6, 3)
        predicted = target.clone()
        predicted[:, 3:, 0] = 1.0

        value = relative_translation_loss(predicted, target, lags=(1, 5))

        self.assertGreater(float(value), 0.0)
        with self.assertRaisesRegex(ValueError, "lag"):
            relative_translation_loss(predicted, target, lags=(6,))


if __name__ == "__main__":
    unittest.main()

