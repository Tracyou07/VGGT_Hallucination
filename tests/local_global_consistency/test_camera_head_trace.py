import unittest

import torch

from vggt.heads.camera_head import CameraHead


class CameraHeadTraceTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.head = CameraHead(
            dim_in=32,
            trunk_depth=1,
            num_heads=4,
            mlp_ratio=2,
        ).eval()
        self.tokens = [torch.randn(2, 3, 5, 32)]

    def test_default_return_is_unchanged_when_trace_is_enabled(self):
        with torch.no_grad():
            baseline = self.head(self.tokens, num_iterations=3)
            traced, trace = self.head(
                self.tokens,
                num_iterations=3,
                return_trace=True,
            )

        self.assertIsInstance(baseline, list)
        self.assertEqual(len(baseline), 3)
        for expected, actual in zip(baseline, traced):
            torch.testing.assert_close(actual, expected)
        self.assertEqual(trace["delta_norm"].shape, (3, 2, 3))
        self.assertEqual(trace["normalized_camera_tokens"].shape, (2, 3, 32))
        self.assertEqual(trace["pose_tokens_modulated_list"], [])

    def test_full_trace_matches_pose_delta_norms(self):
        with torch.no_grad():
            poses, trace = self.head(
                self.tokens,
                num_iterations=2,
                return_trace=True,
                trace_pose_tokens=True,
            )

        self.assertEqual(len(poses), 2)
        self.assertEqual(len(trace["raw_pose_enc_list"]), 2)
        self.assertEqual(len(trace["pose_delta_list"]), 2)
        self.assertEqual(len(trace["pose_tokens_modulated_list"]), 2)
        expected = torch.stack(
            [delta.float().norm(dim=-1) for delta in trace["pose_delta_list"]]
        )
        torch.testing.assert_close(trace["delta_norm"], expected)

    def test_invalid_trace_options_raise(self):
        with self.assertRaisesRegex(ValueError, "num_iterations"):
            self.head(self.tokens, num_iterations=0)
        with self.assertRaisesRegex(ValueError, "return_trace"):
            self.head(self.tokens, trace_pose_tokens=True)

    def test_decode_pose_tokens_validates_rank(self):
        with self.assertRaisesRegex(ValueError, r"\[B, S, C\]"):
            self.head.decode_pose_tokens(torch.randn(3, 32))


if __name__ == "__main__":
    unittest.main()
