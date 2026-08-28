from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor
import torch.nn.functional as F

from pre_experiments.variational_camera_latent.camera import pose_encoding_to_c2w

from .geometry import apply_sim3_torch, relative_translation_loss, rotation_matrix_loss


@dataclass(frozen=True)
class LossWeights:
    gt_translation: float = 1.0
    relative_translation: float = 0.5
    rotation: float = 0.1
    anchor: float = 0.01
    teacher: float = 0.5

    def __post_init__(self) -> None:
        values = (
            self.gt_translation,
            self.relative_translation,
            self.rotation,
            self.anchor,
            self.teacher,
        )
        if not all(math.isfinite(value) and value >= 0.0 for value in values):
            raise ValueError("loss weights must be finite and nonnegative")


@dataclass(frozen=True)
class TrainingLabels:
    gt_c2w: Tensor
    oracle_scale: Tensor
    oracle_rotation: Tensor
    oracle_translation: Tensor
    gt_scene_scale: Tensor
    teacher_c2w_gt_gauge: Tensor
    teacher_weight: Tensor


def _validate_inputs(
    student_pose: Tensor,
    baseline_pose: Tensor,
    labels: TrainingLabels,
    teacher_coefficient: float,
) -> None:
    if (
        student_pose.shape != baseline_pose.shape
        or student_pose.ndim != 3
        or student_pose.shape[0] != 1
        or student_pose.shape[-1] != 9
    ):
        raise ValueError("student and baseline poses must have shape [1,frames,9]")
    frames = student_pose.shape[1]
    if labels.gt_c2w.shape != (1, frames, 4, 4):
        raise ValueError("GT poses must have shape [1,frames,4,4]")
    if labels.teacher_c2w_gt_gauge.shape != (1, frames, 4, 4):
        raise ValueError("teacher poses must have shape [1,frames,4,4]")
    if labels.teacher_weight.shape != (1, frames):
        raise ValueError("teacher weights must have shape [1,frames]")
    if not math.isfinite(teacher_coefficient) or teacher_coefficient < 0.0:
        raise ValueError("teacher coefficient must be finite and nonnegative")
    always_finite = (
        student_pose,
        baseline_pose,
        labels.gt_c2w,
        labels.oracle_scale,
        labels.oracle_rotation,
        labels.oracle_translation,
        labels.gt_scene_scale,
        labels.teacher_weight,
    )
    if not all(torch.isfinite(value).all() for value in always_finite):
        raise ValueError("Camera Head loss inputs must be finite")
    if labels.gt_scene_scale.numel() != 1 or float(labels.gt_scene_scale) <= 0.0:
        raise ValueError("GT scene scale must be positive")
    if torch.any(labels.teacher_weight < 0.0):
        raise ValueError("teacher weights must be nonnegative")
    valid = labels.teacher_weight > 0.0
    teacher_finite = torch.isfinite(labels.teacher_c2w_gt_gauge).all(dim=-1).all(dim=-1)
    if not torch.equal(teacher_finite, valid):
        raise ValueError("teacher pose finiteness must match its positive-weight mask")


def camera_head_losses(
    student_pose: Tensor,
    baseline_pose: Tensor,
    labels: TrainingLabels,
    *,
    teacher_coefficient: float,
    weights: LossWeights = LossWeights(),
    lags: tuple[int, ...] = (1, 5, 10, 25),
) -> dict[str, Tensor]:
    _validate_inputs(student_pose, baseline_pose, labels, teacher_coefficient)
    student_c2w = pose_encoding_to_c2w(student_pose.float())
    aligned = apply_sim3_torch(
        student_c2w,
        scale=labels.oracle_scale.float(),
        rotation=labels.oracle_rotation.float(),
        translation=labels.oracle_translation.float(),
    )
    scale = labels.gt_scene_scale.float().reshape(())
    student_centers = aligned[..., :3, 3] / scale
    gt_centers = labels.gt_c2w.float()[..., :3, 3] / scale
    gt_translation = F.smooth_l1_loss(student_centers, gt_centers)
    relative = relative_translation_loss(student_centers, gt_centers, lags=lags)
    rotation = rotation_matrix_loss(
        aligned[..., :3, :3], labels.gt_c2w.float()[..., :3, :3]
    )
    anchor = F.smooth_l1_loss(student_pose.float(), baseline_pose.float())

    if teacher_coefficient == 0.0:
        teacher = student_pose.sum() * 0.0
    else:
        valid = labels.teacher_weight > 0.0
        student_valid = student_centers[valid]
        teacher_valid = labels.teacher_c2w_gt_gauge.float()[..., :3, 3][valid] / scale
        per_coordinate = F.smooth_l1_loss(
            student_valid, teacher_valid, reduction="none"
        ).mean(dim=-1)
        selected_weights = labels.teacher_weight.float()[valid]
        raw_teacher = torch.sum(per_coordinate * selected_weights) / torch.sum(
            selected_weights
        ).clamp_min(torch.finfo(selected_weights.dtype).eps)
        teacher = raw_teacher * float(teacher_coefficient)

    total = (
        weights.gt_translation * gt_translation
        + weights.relative_translation * relative
        + weights.rotation * rotation
        + weights.anchor * anchor
        + weights.teacher * teacher
    )
    return {
        "total": total,
        "gt_translation": gt_translation,
        "relative_translation": relative,
        "rotation": rotation,
        "anchor": anchor,
        "teacher": teacher,
    }

