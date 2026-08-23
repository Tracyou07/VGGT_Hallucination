"""Tests for eight-frame gauge-fixed camera trajectories."""

import math
import unittest

import numpy as np
import torch

from pre_experiments.camera_solution_space_01.se3 import compose, inverse, se3_exp
from pre_experiments.camera_solution_space_01.trajectory import (
    DEDUP_DISTANCE,
    FAR_DISTANCE,
    gauge_fix_w2c,
    pack_trajectory,
    product_geodesic,
    trajectory_distance,
    unpack_trajectory,
    validate_trajectory,
)


def quaternion_to_matrix(quaternion):
    """Independent literal quaternion conversion used only for sign equivalence."""
    w, x, y, z = quaternion / np.linalg.norm(quaternion)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


class TrajectoryTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(4801)
        twists = rng.normal(scale=0.15, size=(8, 6)).astype(np.float64)
        twists[0] = 0.0
        self.trajectory = se3_exp(twists)

    def test_gauge_fix_is_exact_right_global_gauge_invariant(self):
        base = se3_exp(np.array([0.3, -0.1, 0.2, 0.2, -0.1, 0.15], dtype=np.float64))
        world_to_camera = compose(self.trajectory, base)
        gauge_copy = compose(
            world_to_camera,
            se3_exp(np.array([-0.4, 0.1, 0.3, -0.2, 0.25, 0.1], dtype=np.float64)),
        )
        expected = compose(world_to_camera, inverse(world_to_camera[0]))
        fixed = gauge_fix_w2c(world_to_camera)
        np.testing.assert_allclose(fixed, expected, atol=2e-15)
        np.testing.assert_allclose(gauge_fix_w2c(gauge_copy), fixed, atol=3e-15)
        np.testing.assert_allclose(fixed[0], np.eye(4), atol=2e-15)

    def test_gauge_fix_requires_exactly_eight_w2c_transforms(self):
        for bad in (self.trajectory[:7], np.concatenate([self.trajectory, self.trajectory[:1]])):
            with self.subTest(shape=bad.shape):
                with self.assertRaisesRegex(ValueError, "eight|8"):
                    gauge_fix_w2c(bad)

    def test_pack_unpack_round_trip_has_exactly_42_values_and_fixed_anchor(self):
        packed = pack_trajectory(self.trajectory)
        self.assertEqual(packed.shape, (42,))
        recovered = unpack_trajectory(packed)
        np.testing.assert_array_equal(recovered[0], np.eye(4, dtype=np.float64))
        np.testing.assert_allclose(recovered, self.trajectory, atol=2e-15)

    def test_pack_rejects_nonidentity_anchor_and_unpack_rejects_dimension(self):
        bad_anchor = self.trajectory.copy()
        bad_anchor[0] = se3_exp(np.array([0.01, 0, 0, 0, 0, 0], dtype=np.float64))
        with self.assertRaisesRegex(ValueError, "anchor|first"):
            pack_trajectory(bad_anchor)
        for bad in (np.zeros(41, dtype=np.float64), np.zeros(43, dtype=np.float64), np.zeros((7, 6), dtype=np.float64)):
            with self.subTest(shape=bad.shape):
                with self.assertRaisesRegex(ValueError, "42"):
                    unpack_trajectory(bad)

    def test_distance_has_exact_translation_and_rotation_scales(self):
        identity = np.broadcast_to(np.eye(4, dtype=np.float64), (8, 4, 4)).copy()
        translated = identity.copy()
        translated[1:] = se3_exp(
            np.tile(np.array([0.10, 0, 0, 0, 0, 0], dtype=np.float64), (7, 1))
        )
        rotated = identity.copy()
        rotated[1:] = se3_exp(
            np.tile(np.array([0, 0, 0, 0, 0, math.radians(5)], dtype=np.float64), (7, 1))
        )
        self.assertAlmostEqual(trajectory_distance(identity, translated), 1.0, places=13)
        self.assertAlmostEqual(trajectory_distance(identity, rotated), 1.0, places=13)

        one_frame = identity.copy()
        one_frame[4] = se3_exp(np.array([0.10, 0, 0, 0, 0, 0], dtype=np.float64))
        self.assertAlmostEqual(trajectory_distance(identity, one_frame), 1 / math.sqrt(7), places=13)
        self.assertEqual(FAR_DISTANCE, 1.0)
        self.assertEqual(DEDUP_DISTANCE, 0.10)

    def test_quaternion_sign_equivalent_matrices_have_zero_distance(self):
        quaternion = np.array([0.35, -0.2, 0.4, 0.81], dtype=np.float64)
        first = self.trajectory.copy()
        second = self.trajectory.copy()
        first[3, :3, :3] = quaternion_to_matrix(quaternion)
        second[3, :3, :3] = quaternion_to_matrix(-quaternion)
        self.assertEqual(trajectory_distance(first, second), 0.0)

    def test_product_geodesic_endpoints_midpoint_and_anchor(self):
        other_twists = np.array(
            [[0.0] * 6]
            + [[0.03 * i, -0.01 * i, 0.02, 0.01, -0.015 * i, 0.02] for i in range(1, 8)],
            dtype=np.float64,
        )
        other = se3_exp(other_twists)
        np.testing.assert_array_equal(product_geodesic(self.trajectory, other, 0.0), self.trajectory)
        np.testing.assert_array_equal(product_geodesic(self.trajectory, other, 1.0), other)
        midpoint = product_geodesic(self.trajectory, other, 0.5)
        validate_trajectory(midpoint)
        np.testing.assert_array_equal(midpoint[0], np.eye(4, dtype=np.float64))
        self.assertAlmostEqual(
            trajectory_distance(self.trajectory, midpoint),
            0.5 * trajectory_distance(self.trajectory, other),
            places=11,
        )
        with self.assertRaisesRegex(ValueError, "\[0, 1\]"):
            product_geodesic(self.trajectory, other, -0.01)

    def test_validation_rejects_nonfinite_bad_homogeneous_and_bad_so3(self):
        with self.assertRaisesRegex((TypeError, ValueError), "float64"):
            validate_trajectory(self.trajectory.astype(np.float32))
        nonfinite = self.trajectory.copy()
        nonfinite[2, 0, 3] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            validate_trajectory(nonfinite)
        bad_row = self.trajectory.copy()
        bad_row[5, 3, 1] = 1e-3
        with self.assertRaisesRegex(ValueError, "homogeneous"):
            validate_trajectory(bad_row)
        bad_rotation = self.trajectory.copy()
        bad_rotation[6, 0, 0] += 0.02
        with self.assertRaisesRegex(ValueError, "orthonormal"):
            validate_trajectory(bad_rotation)

    def test_torch_pack_unpack_distance_and_geodesic_preserve_backend(self):
        tensor = torch.tensor(self.trajectory, dtype=torch.float64)
        packed = pack_trajectory(tensor)
        self.assertIsInstance(packed, torch.Tensor)
        self.assertEqual(packed.dtype, torch.float64)
        self.assertEqual(packed.device, tensor.device)
        recovered = unpack_trajectory(packed)
        self.assertEqual(recovered.device, tensor.device)
        torch.testing.assert_close(recovered, tensor, atol=2e-15, rtol=2e-15)

        distance = trajectory_distance(tensor, tensor)
        self.assertIsInstance(distance, torch.Tensor)
        self.assertEqual(distance.dtype, torch.float64)
        self.assertEqual(distance.device, tensor.device)
        self.assertEqual(distance.item(), 0.0)
        midpoint = product_geodesic(tensor, tensor, 0.5)
        self.assertEqual(midpoint.device, tensor.device)

    def test_product_geodesic_preserves_torch_scalar_parameter_gradient(self):
        first = torch.eye(4, dtype=torch.float64).repeat(8, 1, 1)
        second = first.clone()
        second[1:] = se3_exp(
            torch.tensor([[0.2, 0.0, 0.0, 0.0, 0.0, 0.0]] * 7, dtype=torch.float64)
        )
        parameter = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
        midpoint = product_geodesic(first, second, parameter)
        gradient = torch.autograd.grad(midpoint[1, 0, 3], parameter)[0]
        self.assertTrue(torch.isfinite(gradient).item())
        self.assertAlmostEqual(gradient.item(), 0.2, places=14)

    def test_equal_torch_distance_retains_every_trainable_input_graph(self):
        base = torch.eye(4, dtype=torch.float64).repeat(8, 1, 1)

        first_fixed = base.clone()
        second_trainable = base.clone().requires_grad_()
        distance = trajectory_distance(first_fixed, second_trainable)
        second_gradient = torch.autograd.grad(distance, second_trainable)[0]
        self.assertTrue(torch.isfinite(second_gradient).all().item())
        self.assertEqual(torch.count_nonzero(second_gradient).item(), 0)

        first_trainable = base.clone().requires_grad_()
        second_fixed = base.clone()
        distance = trajectory_distance(first_trainable, second_fixed)
        first_gradient = torch.autograd.grad(distance, first_trainable)[0]
        self.assertTrue(torch.isfinite(first_gradient).all().item())
        self.assertEqual(torch.count_nonzero(first_gradient).item(), 0)

        first_trainable = base.clone().requires_grad_()
        second_trainable = base.clone().requires_grad_()
        distance = trajectory_distance(first_trainable, second_trainable)
        gradients = torch.autograd.grad(distance, (first_trainable, second_trainable))
        for gradient in gradients:
            self.assertTrue(torch.isfinite(gradient).all().item())
            self.assertEqual(torch.count_nonzero(gradient).item(), 0)


if __name__ == "__main__":
    unittest.main()
