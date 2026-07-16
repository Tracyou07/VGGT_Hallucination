import math
import unittest

import numpy as np
import torch

from pre_experiments.camera_iteration.metrics import (
    build_iteration_rows,
    validate_iterations,
)
from pre_experiments.camera_iteration.pose_metrics import evaluate_pose
from vggt.utils.pose_enc import extri_intri_to_pose_encoding


def rotation_z(angle):
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return np.array(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def make_c2w(center, yaw):
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = rotation_z(yaw)
    pose[:3, 3] = center
    return pose


def ground_truth_trajectory():
    centers = [
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([1.0, 2.0, 0.0]),
        np.array([2.0, 2.0, 1.0]),
    ]
    return np.stack(
        [make_c2w(center, yaw) for center, yaw in zip(centers, [0.0, 0.1, 0.3, 0.5])]
    )


class PoseMetricsTest(unittest.TestCase):
    def test_known_sim3_is_aligned_before_pose_metrics(self):
        gt_c2w = ground_truth_trajectory()
        scale = 2.5
        rotation = rotation_z(0.4)
        translation = np.array([3.0, -2.0, 1.0])
        pred_c2w = []
        for gt_pose in gt_c2w:
            pred_pose = np.eye(4, dtype=np.float64)
            pred_pose[:3, 3] = rotation.T @ ((gt_pose[:3, 3] - translation) / scale)
            pred_pose[:3, :3] = rotation.T @ gt_pose[:3, :3]
            pred_c2w.append(pred_pose)
        pred_w2c = np.linalg.inv(np.stack(pred_c2w))

        metrics = evaluate_pose(pred_w2c, gt_c2w)

        self.assertLess(metrics["pose_ate_rmse_aligned"], 1e-10)
        self.assertLess(metrics["pose_are_mean_deg_aligned"], 1e-6)
        self.assertLess(metrics["pose_rpe_rot_mean_deg"], 1e-6)
        self.assertLess(metrics["pose_rpe_trans_mean_aligned"], 1e-10)
        self.assertAlmostEqual(metrics["pose_sim3_scale"], scale, places=10)

    def test_rpe_translation_uses_relative_transform_not_only_step_length(self):
        gt_c2w = ground_truth_trajectory()
        pred_c2w = gt_c2w.copy()
        pred_c2w[0, :3, :3] = rotation_z(math.pi / 2.0)
        pred_c2w[1, :3, :3] = rotation_z(-math.pi / 2.0)

        metrics = evaluate_pose(np.linalg.inv(pred_c2w), gt_c2w)

        self.assertGreater(metrics["pose_rpe_trans_mean_aligned"], 0.5)


class IterationMetricsTest(unittest.TestCase):
    def test_exact_pose_encodings_produce_rows_for_requested_iterations(self):
        gt_c2w = ground_truth_trajectory()
        pred_w2c = np.linalg.inv(gt_c2w)
        extrinsics = torch.from_numpy(pred_w2c[:, :3, :4]).unsqueeze(0).float()
        intrinsics = torch.eye(3).repeat(1, len(gt_c2w), 1, 1)
        intrinsics[..., 0, 0] = 100.0
        intrinsics[..., 1, 1] = 100.0
        intrinsics[..., 0, 2] = 50.0
        intrinsics[..., 1, 2] = 50.0
        pose_enc = extri_intri_to_pose_encoding(extrinsics, intrinsics, (100, 100))
        pose_enc_list = [pose_enc.clone() for _ in range(4)]
        delta_norm = torch.stack(
            [torch.full((1, len(gt_c2w)), float(index)) for index in range(1, 5)]
        )

        rows = build_iteration_rows(
            scene="scene0000_00",
            frame_count=len(gt_c2w),
            requested_iterations=[1, 4],
            pose_enc_list=pose_enc_list,
            delta_norm=delta_norm,
            gt_c2w=gt_c2w,
            image_hw=(100, 100),
        )

        self.assertEqual([row["iteration"] for row in rows], [1, 4])
        self.assertLess(rows[0]["pose_ate_rmse_aligned"], 1e-5)
        self.assertAlmostEqual(rows[0]["delta_norm_mean"], 1.0)
        self.assertAlmostEqual(rows[1]["delta_norm_mean"], 4.0)
        self.assertAlmostEqual(rows[1]["delta_norm_p95"], 4.0)
        self.assertAlmostEqual(rows[1]["delta_norm_max"], 4.0)

    def test_iteration_indices_must_be_unique_increasing_and_available(self):
        self.assertEqual(validate_iterations([1, 2, 4], available=4), [1, 2, 4])
        for invalid in ([0, 4], [1, 1], [2, 1], [1, 5]):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    validate_iterations(list(invalid), available=4)

    def test_delta_norm_rejects_multiple_batches(self):
        gt_c2w = np.tile(np.eye(4, dtype=np.float64), (2, 1, 1))
        gt_c2w[1, 0, 3] = 1.0
        pose_enc_list = [torch.zeros(1, 2, 9)]

        with self.assertRaisesRegex(ValueError, "batch dimension 1"):
            build_iteration_rows(
                scene="scene0000_00",
                frame_count=2,
                requested_iterations=[1],
                pose_enc_list=pose_enc_list,
                delta_norm=torch.zeros(1, 2, 2),
                gt_c2w=gt_c2w,
                image_hw=(100, 100),
            )


if __name__ == "__main__":
    unittest.main()
