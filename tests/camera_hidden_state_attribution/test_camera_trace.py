import unittest

import torch

from vggt.heads.camera_head import CameraHead
from vggt.layers.mlp import Mlp


class HiddenTraceTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.head = CameraHead(
            dim_in=32,
            trunk_depth=1,
            num_heads=4,
            mlp_ratio=2,
        ).eval()
        self.tokens = [torch.randn(1, 5, 3, 32)]

    def test_mlp_feature_head_split_matches_forward(self):
        mlp = Mlp(8, hidden_features=4, out_features=3).eval()
        values = torch.randn(2, 8)
        with torch.no_grad():
            expected = mlp(values)
            hidden = mlp.forward_features(values)
            actual = mlp.forward_head(hidden)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(hidden.shape, (2, 4))

    def test_hidden_trace_preserves_predictions_and_shapes(self):
        with torch.no_grad():
            expected = self.head(self.tokens, num_iterations=2)
            actual, trace = self.head(
                self.tokens,
                num_iterations=2,
                return_trace=True,
                trace_pose_tokens=True,
            )
        for expected_pose, actual_pose in zip(expected, actual):
            torch.testing.assert_close(actual_pose, expected_pose)
        self.assertEqual(len(trace["trunk_output_list"]), 2)
        self.assertEqual(trace["trunk_output_list"][0].shape, (1, 5, 32))
        self.assertEqual(len(trace["pose_branch_hidden_list"]), 2)
        self.assertEqual(trace["pose_branch_hidden_list"][0].shape, (1, 5, 16))
        self.assertNotIn("pose_tokens_modulated_list", trace)

    def test_ablation_mask_is_validated_and_false_mask_is_identity(self):
        false_mask = torch.zeros(2, 16, dtype=torch.bool)
        true_mask = false_mask.clone()
        true_mask[0, 0] = True
        with torch.no_grad():
            expected = self.head(self.tokens, num_iterations=2)
            identity = self.head(
                self.tokens,
                num_iterations=2,
                hidden_ablation_mask=false_mask,
            )
            ablated = self.head(
                self.tokens,
                num_iterations=2,
                hidden_ablation_mask=true_mask,
            )
        torch.testing.assert_close(identity[-1], expected[-1])
        self.assertFalse(torch.equal(ablated[-1], expected[-1]))
        with self.assertRaisesRegex(ValueError, "hidden_ablation_mask"):
            self.head(
                self.tokens,
                num_iterations=2,
                hidden_ablation_mask=torch.zeros(1, 16, dtype=torch.bool),
            )

    def test_zero_additive_perturbations_preserve_predictions(self):
        normalized = self.head.token_norm(self.tokens[-1][:, :, 0])
        with torch.no_grad():
            expected = self.head.decode_pose_tokens(
                normalized,
                num_iterations=2,
            )
            actual = self.head.decode_pose_tokens(
                normalized,
                num_iterations=2,
                hidden_additive_perturbation=torch.zeros(2, 1, 16),
                pose_delta_additive_perturbation=torch.zeros(2, 1, 9),
            )
        for expected_pose, actual_pose in zip(expected, actual):
            torch.testing.assert_close(actual_pose, expected_pose)

    def test_additive_perturbations_are_isolated_by_batch_sample(self):
        normalized = self.head.token_norm(self.tokens[-1][:, :, 0])
        normalized = normalized.expand(2, -1, -1).clone()
        hidden_perturbation = torch.zeros(2, 2, 16)
        hidden_perturbation[0, 0, 3] = 0.25
        pose_delta_perturbation = torch.zeros(2, 2, 9)
        pose_delta_perturbation[1, 0, 0] = 0.1
        with torch.no_grad():
            expected = self.head.decode_pose_tokens(
                normalized,
                num_iterations=2,
            )
            actual = self.head.decode_pose_tokens(
                normalized,
                num_iterations=2,
                hidden_additive_perturbation=hidden_perturbation,
                pose_delta_additive_perturbation=pose_delta_perturbation,
            )
        self.assertFalse(torch.equal(actual[-1][0], expected[-1][0]))
        torch.testing.assert_close(actual[-1][1], expected[-1][1])

    def test_additive_perturbations_reject_invalid_tensors(self):
        normalized = self.head.token_norm(self.tokens[-1][:, :, 0])
        invalid_cases = (
            (
                "hidden_additive_perturbation",
                {"hidden_additive_perturbation": torch.zeros(1, 1, 16)},
            ),
            (
                "hidden_additive_perturbation",
                {
                    "hidden_additive_perturbation": torch.zeros(
                        2, 1, 16, dtype=torch.bool
                    )
                },
            ),
            (
                "hidden_additive_perturbation",
                {
                    "hidden_additive_perturbation": torch.full(
                        (2, 1, 16), float("nan")
                    )
                },
            ),
            (
                "pose_delta_additive_perturbation",
                {"pose_delta_additive_perturbation": torch.zeros(2, 1, 8)},
            ),
        )
        for message, kwargs in invalid_cases:
            with self.subTest(message=message, kwargs=tuple(kwargs)):
                with self.assertRaisesRegex(ValueError, message):
                    self.head.decode_pose_tokens(
                        normalized,
                        num_iterations=2,
                        **kwargs,
                    )


if __name__ == "__main__":
    unittest.main()
