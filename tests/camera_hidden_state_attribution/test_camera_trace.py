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


if __name__ == "__main__":
    unittest.main()
