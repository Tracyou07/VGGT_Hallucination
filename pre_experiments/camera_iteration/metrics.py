"""Per-iteration Camera Head metric row construction."""

from __future__ import annotations

import numpy as np
import torch

from pre_experiments.camera_iteration.pose_metrics import evaluate_pose, to_homogeneous
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


MetricRow = dict[str, float | int | str]


def validate_iterations(iterations: list[int], available: int) -> list[int]:
    """Validate one-based, strictly increasing Camera Head iteration indices."""
    if type(available) is not int or available < 1:
        raise ValueError("available must be a positive integer")
    if not iterations:
        raise ValueError("iterations must not be empty")
    if any(type(iteration) is not int for iteration in iterations):
        raise ValueError("iterations must contain integers")
    if any(iteration < 1 or iteration > available for iteration in iterations):
        raise ValueError(f"iterations must be between 1 and {available}")
    if any(current >= following for current, following in zip(iterations, iterations[1:])):
        raise ValueError("iterations must be unique and strictly increasing")
    return list(iterations)


def build_iteration_rows(
    scene: str,
    frame_count: int,
    requested_iterations: list[int],
    pose_enc_list: list[torch.Tensor],
    delta_norm: torch.Tensor,
    gt_c2w: np.ndarray,
    image_hw: tuple[int, int],
) -> list[MetricRow]:
    """Convert selected Camera Head iterations into aligned pose metric rows."""
    iterations = validate_iterations(requested_iterations, len(pose_enc_list))
    if delta_norm.ndim != 3 or delta_norm.shape[0] != len(pose_enc_list):
        raise ValueError("delta_norm must have shape [K, B, S]")
    if tuple(delta_norm.shape[1:]) != (1, frame_count):
        raise ValueError("delta_norm must have batch dimension 1 and match frame_count")
    if frame_count != len(gt_c2w):
        raise ValueError("frame_count must match gt_c2w")

    rows: list[MetricRow] = []
    for iteration in iterations:
        pose_encoding = pose_enc_list[iteration - 1]
        if pose_encoding.ndim != 3 or pose_encoding.shape[0] != 1:
            raise ValueError("pose encodings must have shape [1, S, 9]")
        if pose_encoding.shape[1] != frame_count:
            raise ValueError("pose encoding sequence length must match frame_count")

        extrinsic, _ = pose_encoding_to_extri_intri(
            pose_encoding,
            image_hw,
            build_intrinsics=False,
        )
        predicted_w2c = to_homogeneous(
            extrinsic[0].detach().float().cpu().numpy()
        )
        iteration_delta = (
            delta_norm[iteration - 1].detach().float().cpu().numpy().reshape(-1)
        )
        if len(iteration_delta) == 0 or not np.isfinite(iteration_delta).all():
            raise ValueError("delta_norm must contain finite values")

        row: MetricRow = {
            "scene": scene,
            "frame_count_actual": frame_count,
            "iteration": iteration,
            "delta_norm_mean": float(iteration_delta.mean()),
            "delta_norm_p95": float(np.quantile(iteration_delta, 0.95)),
            "delta_norm_max": float(iteration_delta.max()),
        }
        row.update(evaluate_pose(predicted_w2c, gt_c2w))
        rows.append(row)
    return rows
