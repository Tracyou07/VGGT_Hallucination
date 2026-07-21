import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from pre_experiments.camera_context.artifacts import build_context_diagnostics
from pre_experiments.camera_context.analyze import write_context_analysis
from pre_experiments.camera_context.metrics import compare_contexts
from pre_experiments.camera_iteration.pose_metrics import align_pose_sequence


def rotation_z(angle):
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return np.array(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def make_trajectory():
    poses = np.tile(np.eye(4, dtype=np.float64), (3, 1, 1))
    poses[1, :3, 3] = [1.0, 0.0, 0.0]
    poses[2, :3, 3] = [1.0, 2.0, 0.0]
    poses[1, :3, :3] = rotation_z(0.2)
    poses[2, :3, :3] = rotation_z(0.4)
    return poses


class PerFrameAlignmentTest(unittest.TestCase):
    def test_alignment_returns_per_frame_errors_without_changing_raw_gt(self):
        gt_c2w = make_trajectory()
        original_gt = gt_c2w.copy()
        scale = 2.0
        rotation = rotation_z(0.3)
        translation = np.array([3.0, -1.0, 0.5])
        pred_c2w = gt_c2w.copy()
        pred_c2w[:, :3, 3] = (
            (gt_c2w[:, :3, 3] - translation) @ rotation
        ) / scale
        pred_c2w[:, :3, :3] = np.einsum(
            "ij,sjk->sik", rotation.T, gt_c2w[:, :3, :3]
        )

        aligned = align_pose_sequence(np.linalg.inv(pred_c2w), gt_c2w)

        np.testing.assert_allclose(gt_c2w, original_gt)
        np.testing.assert_allclose(aligned["aligned_c2w"], gt_c2w, atol=1e-10)
        np.testing.assert_allclose(aligned["translation_error_aligned"], 0.0, atol=1e-10)
        np.testing.assert_allclose(aligned["rotation_error_deg_aligned"], 0.0, atol=1e-6)
        self.assertAlmostEqual(aligned["sim3_scale"], scale, places=10)


class ContextComparisonTest(unittest.TestCase):
    def test_comparison_matches_shared_frame_ids_and_reports_drift(self):
        identity = np.tile(np.eye(4, dtype=np.float64), (3, 1, 1))
        short = {
            "frame_ids": np.array([10, 30]),
            "normalized_camera_tokens": np.array([[1.0, 0.0], [0.0, 1.0]]),
            "pred_c2w_aligned": identity[[0, 2]],
            "translation_error_aligned": np.array([0.1, 0.2]),
            "rotation_error_deg_aligned": np.array([1.0, 2.0]),
            "delta_norm": np.array([0.01, 0.02]),
        }
        long = {
            "frame_ids": np.array([10, 20, 30]),
            "normalized_camera_tokens": np.array(
                [[1.0, 0.0], [1.0, 1.0], [1.0, 0.0]]
            ),
            "pred_c2w_aligned": identity.copy(),
            "translation_error_aligned": np.array([0.3, 0.0, 0.5]),
            "rotation_error_deg_aligned": np.array([3.0, 0.0, 5.0]),
            "delta_norm": np.array([0.04, 0.0, 0.08]),
        }

        rows, summary = compare_contexts(short, long, short_frames=2, long_frames=3)

        self.assertEqual([row["frame_id"] for row in rows], [10, 30])
        self.assertAlmostEqual(rows[0]["token_cosine_distance"], 0.0)
        self.assertAlmostEqual(rows[1]["token_cosine_distance"], 1.0)
        self.assertAlmostEqual(rows[1]["translation_error_change"], 0.3)
        self.assertEqual(summary["shared_frame_count"], 2)
        self.assertGreater(summary["token_pairwise_affinity_drift"], 0.0)
        self.assertGreater(summary["token_translation_error_change_pearson"], 0.9)

    def test_comparison_rejects_contexts_without_shared_frames(self):
        empty_pose = np.empty((1, 4, 4), dtype=np.float64)
        left = {
            "frame_ids": np.array([1]),
            "normalized_camera_tokens": np.ones((1, 2)),
            "pred_c2w_aligned": empty_pose,
            "translation_error_aligned": np.zeros(1),
            "rotation_error_deg_aligned": np.zeros(1),
            "delta_norm": np.zeros(1),
        }
        right = {**left, "frame_ids": np.array([2])}

        with self.assertRaisesRegex(ValueError, "shared frame"):
            compare_contexts(left, right, short_frames=1, long_frames=2)


class ContextArtifactTest(unittest.TestCase):
    def test_artifact_keeps_raw_gt_and_stores_aligned_prediction(self):
        gt_c2w = make_trajectory()
        pred_w2c = np.linalg.inv(gt_c2w)
        tokens = np.arange(12, dtype=np.float32).reshape(3, 4) + 1.0

        artifact = build_context_diagnostics(
            frame_ids=np.array([10, 20, 30]),
            normalized_camera_tokens=tokens,
            pred_w2c=pred_w2c,
            gt_c2w_raw=gt_c2w,
            delta_norm=np.array([0.3, 0.2, 0.1]),
        )

        np.testing.assert_array_equal(artifact["frame_ids"], [10, 20, 30])
        np.testing.assert_allclose(artifact["gt_c2w_raw"], gt_c2w)
        np.testing.assert_allclose(artifact["pred_c2w_raw"], gt_c2w, atol=1e-12)
        np.testing.assert_allclose(artifact["pred_c2w_aligned"], gt_c2w, atol=1e-12)
        np.testing.assert_allclose(
            artifact["translation_error_aligned"], 0.0, atol=1e-12
        )
        np.testing.assert_allclose(artifact["delta_norm"], [0.3, 0.2, 0.1])

    def test_artifact_rejects_mismatched_sequence_lengths(self):
        gt_c2w = make_trajectory()
        with self.assertRaisesRegex(ValueError, "sequence length"):
            build_context_diagnostics(
                frame_ids=np.array([10, 20]),
                normalized_camera_tokens=np.ones((3, 4)),
                pred_w2c=np.linalg.inv(gt_c2w),
                gt_c2w_raw=gt_c2w,
                delta_norm=np.ones(3),
            )


class ContextRunAnalysisTest(unittest.TestCase):
    def test_analysis_writes_matched_frame_and_summary_tables(self):
        poses = make_trajectory()
        short = build_context_diagnostics(
            frame_ids=np.array([10, 30]),
            normalized_camera_tokens=np.array([[1.0, 0.0], [0.0, 1.0]]),
            pred_w2c=np.linalg.inv(poses[[0, 2]]),
            gt_c2w_raw=poses[[0, 2]],
            delta_norm=np.array([0.1, 0.2]),
        )
        long = build_context_diagnostics(
            frame_ids=np.array([10, 20, 30]),
            normalized_camera_tokens=np.array(
                [[1.0, 0.0], [1.0, 1.0], [1.0, 0.0]]
            ),
            pred_w2c=np.linalg.inv(poses),
            gt_c2w_raw=poses,
            delta_norm=np.array([0.1, 0.2, 0.3]),
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            short_dir = run_dir / "scene0000_00" / "frames_2"
            long_dir = run_dir / "scene0000_00" / "frames_3"
            short_dir.mkdir(parents=True)
            long_dir.mkdir(parents=True)
            np.savez_compressed(short_dir / "context_diagnostics.npz", **short)
            np.savez_compressed(long_dir / "context_diagnostics.npz", **long)

            frame_rows, summaries = write_context_analysis(run_dir)

            self.assertEqual(len(frame_rows), 2)
            self.assertEqual(len(summaries), 1)
            self.assertEqual(frame_rows[0]["scene"], "scene0000_00")
            self.assertTrue((run_dir / "context_per_frame.csv").is_file())
            self.assertTrue((run_dir / "context_summary.csv").is_file())
            self.assertTrue((run_dir / "context_summary.json").is_file())


if __name__ == "__main__":
    unittest.main()
