import unittest

import numpy as np
import torch

from pre_experiments.camera_head_amplification.metrics import (
    build_stage_per_frame_rows,
    build_stage_summary_rows,
    match_frame_indices,
    validate_replay_baseline,
)
from pre_experiments.camera_head_amplification.replay import replay_camera_head
from vggt.heads.camera_head import CameraHead


class FrameMatchingTest(unittest.TestCase):
    def test_matches_long_indices_in_short_frame_order(self):
        indices = match_frame_indices(
            np.array([30, 10, 50]),
            np.array([10, 20, 30, 40, 50]),
        )

        np.testing.assert_array_equal(indices, [2, 0, 4])

    def test_rejects_missing_or_duplicate_frame_ids(self):
        with self.assertRaisesRegex(ValueError, "missing"):
            match_frame_indices(np.array([10, 30]), np.array([10, 20]))
        with self.assertRaisesRegex(ValueError, "unique"):
            match_frame_indices(np.array([10]), np.array([10, 10]))


class CameraHeadReplayTest(unittest.TestCase):
    def test_captures_each_iteration_and_trunk_stage(self):
        torch.manual_seed(17)
        head = CameraHead(
            dim_in=32,
            trunk_depth=2,
            num_heads=4,
            mlp_ratio=2,
        ).eval()
        tokens = torch.randn(1, 5, 32)

        replay = replay_camera_head(head, tokens, num_iterations=3)

        self.assertEqual(replay.activated_pose.shape, (3, 1, 5, 9))
        self.assertEqual(replay.raw_pose.shape, (3, 1, 5, 9))
        self.assertEqual(replay.pose_delta.shape, (3, 1, 5, 9))
        self.assertEqual(
            list(replay.representations),
            ["adaln_input", "block_1", "block_2", "trunk_norm"],
        )
        for value in replay.representations.values():
            self.assertEqual(value.shape, (3, 1, 5, 32))

    def test_replay_matches_direct_camera_head_decode(self):
        torch.manual_seed(23)
        head = CameraHead(
            dim_in=32,
            trunk_depth=1,
            num_heads=4,
            mlp_ratio=2,
        ).eval()
        tokens = torch.randn(1, 4, 32)

        with torch.no_grad():
            direct = torch.stack(head.decode_pose_tokens(tokens, num_iterations=2))
        replay = replay_camera_head(head, tokens, num_iterations=2)

        torch.testing.assert_close(replay.activated_pose, direct)


class AmplificationMetricTest(unittest.TestCase):
    def test_per_frame_rows_identify_the_comparison(self):
        tokens = np.zeros((2, 2), dtype=np.float32)
        stages = {"block_1": np.zeros((1, 2, 2), dtype=np.float32)}

        rows = build_stage_per_frame_rows(
            scene="scene",
            comparison="token_perturbation",
            frame_ids=np.array([10, 20]),
            baseline_tokens=tokens,
            perturbed_tokens=tokens,
            baseline_stages=stages,
            perturbed_stages=stages,
        )

        self.assertEqual({row["comparison"] for row in rows}, {"token_perturbation"})

    def test_reports_input_relative_rms_drift(self):
        baseline_tokens = np.zeros((2, 2), dtype=np.float32)
        perturbed_tokens = np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)
        baseline_stages = {
            "block_1": np.zeros((1, 2, 2), dtype=np.float32),
        }
        perturbed_stages = {
            "block_1": np.array(
                [[[6.0, 8.0], [0.0, 0.0]]], dtype=np.float32
            ),
        }

        rows = build_stage_summary_rows(
            scene="scene",
            comparison="fixed_length",
            baseline_tokens=baseline_tokens,
            perturbed_tokens=perturbed_tokens,
            baseline_stages=baseline_stages,
            perturbed_stages=perturbed_stages,
        )

        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["input_token_rms_drift"], 5.0 / np.sqrt(2))
        self.assertAlmostEqual(rows[0]["stage_rms_drift"], 10.0 / np.sqrt(2))
        self.assertAlmostEqual(rows[0]["amplification_ratio"], 2.0)

    def test_rejects_zero_input_drift(self):
        tokens = np.ones((2, 3), dtype=np.float32)
        stages = {"block_1": np.ones((1, 2, 3), dtype=np.float32)}
        with self.assertRaisesRegex(ValueError, "non-zero"):
            build_stage_summary_rows(
                scene="scene",
                comparison="fixed_length",
                baseline_tokens=tokens,
                perturbed_tokens=tokens,
                baseline_stages=stages,
                perturbed_stages=stages,
            )

    def test_allows_zero_input_for_head_context_effect(self):
        tokens = np.ones((2, 3), dtype=np.float32)
        baseline = {"block_1": np.zeros((1, 2, 3), dtype=np.float32)}
        perturbed = {"block_1": np.ones((1, 2, 3), dtype=np.float32)}

        rows = build_stage_summary_rows(
            scene="scene",
            comparison="head_context_effect",
            baseline_tokens=tokens,
            perturbed_tokens=tokens,
            baseline_stages=baseline,
            perturbed_stages=perturbed,
            allow_zero_input=True,
        )

        self.assertEqual(rows[0]["input_token_rms_drift"], 0.0)
        self.assertIsNone(rows[0]["amplification_ratio"])

    def test_baseline_validation_fails_closed(self):
        expected = np.zeros((4, 9), dtype=np.float32)
        actual = expected.copy()
        actual[0, 0] = 0.1

        with self.assertRaisesRegex(ValueError, "replay baseline mismatch"):
            validate_replay_baseline(actual, expected, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
