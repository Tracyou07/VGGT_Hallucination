from __future__ import annotations

import unittest

import torch

from pre_experiments.long_short_camera_head.losses import (
    LossWeights,
    TrainingLabels,
    camera_head_losses,
)


def _identity_pose_encoding(frames: int) -> torch.Tensor:
    pose = torch.zeros(1, frames, 9)
    pose[..., 6] = 1.0
    return pose


class CameraHeadLossTests(unittest.TestCase):
    def setUp(self) -> None:
        frames = 30
        gt = torch.eye(4).repeat(1, frames, 1, 1)
        gt[:, :, 0, 3] = torch.linspace(0.0, 1.0, frames)
        teacher = gt.clone()
        self.labels = TrainingLabels(
            gt_c2w=gt,
            oracle_scale=torch.tensor(1.0),
            oracle_rotation=torch.eye(3),
            oracle_translation=torch.zeros(3),
            gt_scene_scale=torch.tensor(1.0),
            teacher_c2w_gt_gauge=teacher,
            teacher_weight=torch.ones(1, frames),
        )
        self.baseline = _identity_pose_encoding(frames)

    def test_teacher_term_is_exactly_disabled_for_gt_only(self) -> None:
        student = self.baseline.clone().requires_grad_()

        losses = camera_head_losses(
            student,
            self.baseline,
            self.labels,
            teacher_coefficient=0.0,
            weights=LossWeights(),
        )

        self.assertEqual(float(losses["teacher"].detach()), 0.0)
        self.assertGreater(float(losses["gt_translation"].detach()), 0.0)
        losses["total"].backward()
        self.assertTrue(torch.isfinite(student.grad).all())

    def test_teacher_loss_uses_only_positive_weight_frames(self) -> None:
        student = self.baseline.clone().requires_grad_()
        masked_teacher = self.labels.teacher_c2w_gt_gauge.clone()
        masked_teacher[:, 10:] = torch.nan
        labels = TrainingLabels(
            gt_c2w=self.labels.gt_c2w,
            oracle_scale=self.labels.oracle_scale,
            oracle_rotation=self.labels.oracle_rotation,
            oracle_translation=self.labels.oracle_translation,
            gt_scene_scale=self.labels.gt_scene_scale,
            teacher_c2w_gt_gauge=masked_teacher,
            teacher_weight=torch.cat((torch.ones(1, 10), torch.zeros(1, 20)), dim=1),
        )

        losses = camera_head_losses(
            student,
            self.baseline,
            labels,
            teacher_coefficient=1.0,
            weights=LossWeights(),
        )

        self.assertTrue(torch.isfinite(losses["teacher"]))
        self.assertGreater(float(losses["teacher"].detach()), 0.0)

    def test_rejects_nonfinite_student_before_loss(self) -> None:
        student = self.baseline.clone()
        student[..., 0] = torch.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            camera_head_losses(
                student,
                self.baseline,
                self.labels,
                teacher_coefficient=1.0,
                weights=LossWeights(),
            )


if __name__ == "__main__":
    unittest.main()
