from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F


def apply_sim3_torch(
    c2w: Tensor,
    *,
    scale: Tensor,
    rotation: Tensor,
    translation: Tensor,
) -> Tensor:
    """Apply a fixed orientation-preserving Sim(3) without breaking autograd."""
    if c2w.ndim < 3 or c2w.shape[-2:] != (4, 4):
        raise ValueError("c2w must end with shape [4,4]")
    if scale.numel() != 1 or rotation.shape != (3, 3) or translation.shape != (3,):
        raise ValueError("Sim(3) members have invalid shapes")
    if not all(torch.isfinite(value).all() for value in (c2w, scale, rotation, translation)):
        raise ValueError("Sim(3) application requires finite tensors")
    dtype = c2w.dtype
    device = c2w.device
    rotation = rotation.to(device=device, dtype=dtype)
    translation = translation.to(device=device, dtype=dtype)
    scale = scale.to(device=device, dtype=dtype).reshape(())
    aligned_rotation = torch.matmul(rotation, c2w[..., :3, :3])
    aligned_center = scale * torch.matmul(c2w[..., :3, 3], rotation.transpose(0, 1))
    aligned_center = aligned_center + translation
    upper = torch.cat((aligned_rotation, aligned_center.unsqueeze(-1)), dim=-1)
    bottom = torch.zeros((*c2w.shape[:-2], 1, 4), dtype=dtype, device=device)
    bottom[..., 0, 3] = 1.0
    return torch.cat((upper, bottom), dim=-2)


def rotation_matrix_loss(predicted: Tensor, target: Tensor) -> Tensor:
    if predicted.shape != target.shape or predicted.shape[-2:] != (3, 3):
        raise ValueError("rotation stacks must have matching [...,3,3] shapes")
    if not torch.isfinite(predicted).all() or not torch.isfinite(target).all():
        raise ValueError("rotation stacks must be finite")
    return torch.mean((predicted - target) ** 2)


def relative_translation_loss(
    predicted: Tensor,
    target: Tensor,
    *,
    lags: tuple[int, ...] = (1, 5, 10, 25),
) -> Tensor:
    if predicted.shape != target.shape or predicted.ndim != 3 or predicted.shape[-1] != 3:
        raise ValueError("translations must have matching [batch,frames,3] shapes")
    frames = predicted.shape[1]
    if not lags or any(isinstance(lag, bool) or lag < 1 or lag >= frames for lag in lags):
        raise ValueError("each lag must lie between one and frames minus one")
    losses = []
    for lag in lags:
        pred_delta = predicted[:, lag:] - predicted[:, :-lag]
        target_delta = target[:, lag:] - target[:, :-lag]
        losses.append(F.smooth_l1_loss(pred_delta, target_delta))
    return torch.stack(losses).mean()

