from __future__ import annotations

import torch
from torch import Tensor


def listwise_quality_loss(
    scores: Tensor,
    utilities: Tensor,
    *,
    tau: float = 0.05,
    return_target: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Cross entropy against a soft, quality-weighted distribution per group."""
    if (
        tau <= 0.0
        or not torch.isfinite(torch.as_tensor(tau))
        or scores.ndim != 2
        or scores.shape != utilities.shape
        or not torch.is_floating_point(scores)
        or not torch.is_floating_point(utilities)
        or not torch.isfinite(scores).all()
        or not torch.isfinite(utilities).all()
    ):
        raise ValueError(
            "scores/utilities must be matching finite floating groups and tau must be positive"
        )
    target = torch.softmax(utilities / float(tau), dim=-1)
    loss = -(target * torch.log_softmax(scores, dim=-1)).sum(dim=-1).mean()
    if not torch.isfinite(loss):
        raise ValueError("listwise quality loss is non-finite")
    return (loss, target) if return_target else loss

