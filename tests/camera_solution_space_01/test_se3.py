"""Contract tests for backend-preserving SE(3) geometry."""

import math
import unittest

import numpy as np
import torch

from pre_experiments.camera_solution_space_01.se3 import (
    compose,
    geodesic_interpolate,
    inverse,
    se3_exp,
    se3_log,
    so3_exp,
    so3_log,
)


class Se3Tests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(20260823)

    def test_so3_exp_log_zero_tiny_random_and_near_pi(self):
        vectors = [
            np.zeros(3, dtype=np.float64),
            np.array([1e-12, -2e-12, 3e-12], dtype=np.float64),
            np.array([0.31, -0.27, 0.19], dtype=np.float64),
            np.array([1.0, -2.0, 3.0], dtype=np.float64)
            / math.sqrt(14.0)
            * (math.pi - 1e-8),
        ]
        for omega in vectors:
            with self.subTest(omega=omega):
                rotation = so3_exp(omega)
                recovered_rotation = so3_exp(so3_log(rotation))
                np.testing.assert_allclose(recovered_rotation, rotation, atol=2e-9, rtol=2e-9)

    def test_so3_batched_random_round_trip(self):
        axes = self.rng.normal(size=(16, 3))
        axes /= np.linalg.norm(axes, axis=-1, keepdims=True)
        angles = self.rng.uniform(0.0, math.pi - 1e-4, size=(16, 1))
        omega = (axes * angles).astype(np.float64)
        recovered = so3_log(so3_exp(omega))
        np.testing.assert_allclose(recovered, omega, atol=2e-10, rtol=2e-10)

    def test_se3_exp_log_zero_tiny_random_and_near_pi(self):
        twists = [
            np.zeros(6, dtype=np.float64),
            np.array([1e-11, -2e-11, 3e-11, 1e-12, 2e-12, -1e-12]),
            np.array([0.4, -0.2, 0.1, 0.31, -0.27, 0.19]),
            np.concatenate(
                [
                    np.array([0.2, -0.1, 0.3]),
                    np.array([1.0, 4.0, -2.0]) / math.sqrt(21.0) * (math.pi - 1e-8),
                ]
            ),
        ]
        for twist in twists:
            twist = twist.astype(np.float64)
            with self.subTest(twist=twist):
                transform = se3_exp(twist)
                np.testing.assert_allclose(se3_exp(se3_log(transform)), transform, atol=3e-9, rtol=3e-9)

    def test_twist_convention_is_translation_first_and_uses_left_jacobian(self):
        pure_translation = np.array([1.5, -2.0, 0.25, 0.0, 0.0, 0.0], dtype=np.float64)
        transform = se3_exp(pure_translation)
        np.testing.assert_allclose(transform[:3, 3], pure_translation[:3], atol=0.0, rtol=0.0)
        np.testing.assert_allclose(se3_log(transform), pure_translation, atol=0.0, rtol=0.0)

        quarter_turn = np.array([1.0, 0.0, 0.0, 0.0, 0.0, math.pi / 2], dtype=np.float64)
        transformed = se3_exp(quarter_turn)
        np.testing.assert_allclose(transformed[:3, 3], [2 / math.pi, 2 / math.pi, 0.0], atol=2e-15)

    def test_inverse_and_compose_support_batches(self):
        twists = self.rng.normal(scale=0.2, size=(5, 6)).astype(np.float64)
        transforms = se3_exp(twists)
        identities = compose(transforms, inverse(transforms))
        expected = np.broadcast_to(np.eye(4, dtype=np.float64), (5, 4, 4))
        np.testing.assert_allclose(identities, expected, atol=2e-15)

    def test_rejects_wrong_dtype_shape_nonfinite_and_invalid_rotations(self):
        with self.assertRaisesRegex((TypeError, ValueError), "float64"):
            so3_exp(np.zeros(3, dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "shape"):
            so3_exp(np.zeros(4, dtype=np.float64))
        with self.assertRaisesRegex(ValueError, "finite"):
            so3_exp(np.array([0.0, np.nan, 0.0], dtype=np.float64))
        with self.assertRaisesRegex(ValueError, "shape"):
            se3_log(np.eye(3, dtype=np.float64))

        non_homogeneous = np.eye(4, dtype=np.float64)
        non_homogeneous[3, 0] = 0.01
        with self.assertRaisesRegex(ValueError, "homogeneous"):
            inverse(non_homogeneous)

        non_orthonormal = np.eye(4, dtype=np.float64)
        non_orthonormal[0, 0] = 1.01
        with self.assertRaisesRegex(ValueError, "orthonormal"):
            se3_log(non_orthonormal)

        reflection = np.eye(4, dtype=np.float64)
        reflection[0, 0] = -1.0
        with self.assertRaisesRegex(ValueError, "determinant"):
            se3_log(reflection)

        nonfinite = np.eye(4, dtype=np.float64)
        nonfinite[1, 3] = np.inf
        with self.assertRaisesRegex(ValueError, "finite"):
            inverse(nonfinite)

    def test_numpy_torch_parity_and_torch_preserves_dtype_and_device(self):
        twists = self.rng.normal(scale=0.3, size=(4, 6)).astype(np.float64)
        numpy_result = se3_exp(twists)
        tensor = torch.tensor(twists, dtype=torch.float64)
        torch_result = se3_exp(tensor)
        self.assertIsInstance(torch_result, torch.Tensor)
        self.assertEqual(torch_result.dtype, torch.float64)
        self.assertEqual(torch_result.device, tensor.device)
        np.testing.assert_allclose(torch_result.detach().cpu().numpy(), numpy_result, atol=2e-15)
        np.testing.assert_allclose(se3_log(torch_result).detach().cpu().numpy(), twists, atol=3e-15)

        if torch.cuda.is_available():
            cuda_tensor = tensor.cuda()
            self.assertEqual(se3_exp(cuda_tensor).device, cuda_tensor.device)

    def test_torch_gradients_are_finite_away_from_singularities(self):
        twist = torch.tensor(
            [0.3, -0.2, 0.1, 0.25, -0.35, 0.15], dtype=torch.float64, requires_grad=True
        )
        transform = se3_exp(twist)
        objective = transform[0, 1] + 0.7 * transform[2, 3] - 0.2 * transform[1, 0]
        gradient = torch.autograd.grad(objective, twist)[0]
        self.assertTrue(torch.isfinite(gradient).all().item())
        self.assertGreater(torch.linalg.vector_norm(gradient).item(), 0.0)

        recovered = se3_log(se3_exp(twist))
        jacobian_loss = (recovered * torch.arange(1, 7, dtype=torch.float64)).sum()
        log_gradient = torch.autograd.grad(jacobian_loss, twist)[0]
        self.assertTrue(torch.isfinite(log_gradient).all().item())

    def test_torch_zero_and_identity_geometry_have_finite_gradients(self):
        omega = torch.zeros(3, dtype=torch.float64, requires_grad=True)
        rotation_gradient = torch.autograd.grad(so3_exp(omega)[2, 1], omega)[0]
        self.assertTrue(torch.isfinite(rotation_gradient).all().item())
        torch.testing.assert_close(rotation_gradient, torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64))

        twist = torch.zeros(6, dtype=torch.float64, requires_grad=True)
        transform = se3_exp(twist)
        exp_gradient = torch.autograd.grad(transform[0, 3] + transform[2, 1], twist)[0]
        self.assertTrue(torch.isfinite(exp_gradient).all().item())

        identity = torch.eye(4, dtype=torch.float64, requires_grad=True)
        weights = torch.arange(1, 7, dtype=torch.float64)
        log_gradient = torch.autograd.grad((se3_log(identity) * weights).sum(), identity)[0]
        self.assertTrue(torch.isfinite(log_gradient).all().item())

    def test_equal_transform_geodesic_has_finite_torch_gradient(self):
        twist = torch.zeros(6, dtype=torch.float64, requires_grad=True)
        transform = se3_exp(twist)
        midpoint = geodesic_interpolate(transform, transform, 0.5)
        gradient = torch.autograd.grad(midpoint[0, 3] + midpoint[2, 1], twist)[0]
        self.assertTrue(torch.isfinite(gradient).all().item())

    def test_small_angle_left_jacobian_is_accurate_and_continuous(self):
        angles = (1.0000001e-8, 0.999e-4, 1.001e-4)
        for angle in angles:
            with self.subTest(angle=angle):
                twist = np.array([0.0, 1.0, 0.0, angle, 0.0, 0.0], dtype=np.float64)
                transform = se3_exp(twist)
                expected = np.array(
                    [0.0, math.sin(angle) / angle, 2.0 * math.sin(angle / 2.0) ** 2 / angle],
                    dtype=np.float64,
                )
                np.testing.assert_allclose(transform[:3, 3], expected, atol=2e-15, rtol=2e-15)
                np.testing.assert_allclose(se3_log(transform), twist, atol=2e-15, rtol=2e-15)
                torch_result = se3_exp(torch.tensor(twist, dtype=torch.float64))
                np.testing.assert_allclose(torch_result.numpy(), transform, atol=2e-15, rtol=2e-15)
        self.assertAlmostEqual(
            se3_exp(np.array([0.0, 1.0, 0.0, 1.0000001e-8, 0.0, 0.0]))[2, 3],
            5.0000005e-9,
            places=16,
        )

    def test_torch_scalar_t_remains_differentiable(self):
        start = torch.eye(4, dtype=torch.float64)
        end = se3_exp(torch.tensor([0.2, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64))
        for value, expected in ((0.0, start), (0.5, None), (1.0, end)):
            with self.subTest(value=value):
                parameter = torch.tensor(value, dtype=torch.float64, requires_grad=True)
                interpolated = geodesic_interpolate(start, end, parameter)
                if expected is not None:
                    torch.testing.assert_close(interpolated, expected, atol=0.0, rtol=0.0)
                gradient = torch.autograd.grad(interpolated[0, 3], parameter)[0]
                self.assertTrue(torch.isfinite(gradient).item())
                self.assertAlmostEqual(gradient.item(), 0.2, places=14)

    def test_geodesic_validates_pairs_before_endpoints_and_broadcasts_consistently(self):
        numpy_identity = np.eye(4, dtype=np.float64)
        torch_identity = torch.eye(4, dtype=torch.float64)
        with self.assertRaisesRegex(TypeError, "backend|NumPy|torch"):
            geodesic_interpolate(torch_identity, numpy_identity, 0.0)
        with self.assertRaisesRegex(TypeError, "float64"):
            geodesic_interpolate(
                torch_identity, torch_identity, torch.tensor(0.0, dtype=torch.float32)
            )

        start = torch_identity.expand(2, 1, 4, 4)
        end = torch_identity.expand(1, 3, 4, 4)
        for parameter in (0.0, 0.5, 1.0):
            with self.subTest(parameter=parameter):
                result = geodesic_interpolate(start, end, parameter)
                self.assertEqual(result.shape, (2, 3, 4, 4))
                torch.testing.assert_close(result, torch.eye(4, dtype=torch.float64).expand(2, 3, 4, 4))

    def test_empty_rotation_batches_are_supported(self):
        numpy_result = so3_log(np.empty((0, 3, 3), dtype=np.float64))
        torch_input = torch.empty((0, 3, 3), dtype=torch.float64, requires_grad=True)
        torch_result = so3_log(torch_input)
        self.assertEqual(numpy_result.shape, (0, 3))
        self.assertEqual(torch_result.shape, (0, 3))
        self.assertEqual(torch_result.dtype, torch.float64)
        self.assertEqual(torch_result.device, torch_input.device)

    def test_geodesic_endpoints_are_exact_and_midpoint_is_valid(self):
        start = se3_exp(np.array([0.1, -0.2, 0.3, 0.2, 0.1, -0.15], dtype=np.float64))
        end = se3_exp(np.array([-0.3, 0.1, 0.2, -0.1, 0.3, 0.25], dtype=np.float64))
        np.testing.assert_array_equal(geodesic_interpolate(start, end, 0.0), start)
        np.testing.assert_array_equal(geodesic_interpolate(start, end, 1.0), end)
        midpoint = geodesic_interpolate(start, end, 0.5)
        np.testing.assert_allclose(midpoint[3], [0.0, 0.0, 0.0, 1.0], atol=0.0)
        np.testing.assert_allclose(midpoint[:3, :3].T @ midpoint[:3, :3], np.eye(3), atol=2e-15)
        with self.assertRaisesRegex(ValueError, "\[0, 1\]"):
            geodesic_interpolate(start, end, 1.1)


if __name__ == "__main__":
    unittest.main()
