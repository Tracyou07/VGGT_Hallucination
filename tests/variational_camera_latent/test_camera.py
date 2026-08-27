from __future__ import annotations

import unittest

import torch

from pre_experiments.variational_camera_latent.camera import (
    decode_camera_tokens,
    run_latent_preflight,
)


class RecordingCameraHead:
    def __init__(self, *, finite: bool = True) -> None:
        self.calls = 0
        self.finite = finite

    def decode_pose_tokens(
        self, tokens: torch.Tensor, *, num_iterations: int
    ) -> list[torch.Tensor]:
        self.calls += 1
        output = tokens[..., :9].clone()
        if not self.finite:
            output[..., 0] = torch.nan
        return [output for _ in range(num_iterations)]


class CameraCompatibilityTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(3)
        self.long = torch.randn(1, 50, 2048)
        self.left = torch.randn(1, 50, 2048)
        self.right = torch.randn(1, 50, 2048)

    def test_preflight_checks_both_paths_at_fixed_alphas(self) -> None:
        head = RecordingCameraHead()

        report = run_latent_preflight(head, self.long, self.left, self.right)

        self.assertEqual(report["alphas"], [0.0, 0.25, 0.5, 0.75, 1.0])
        self.assertTrue(report["all_finite"])
        self.assertEqual(head.calls, 13)
        self.assertEqual(report["decoded_shape"], [1, 50, 9])

    def test_decode_rejects_nonfinite_pose(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-finite"):
            decode_camera_tokens(RecordingCameraHead(finite=False), self.long)

    def test_decode_rejects_malformed_token_shape_before_head(self) -> None:
        with self.assertRaisesRegex(ValueError, "2048"):
            decode_camera_tokens(RecordingCameraHead(), torch.zeros(1, 50, 32))


if __name__ == "__main__":
    unittest.main()
