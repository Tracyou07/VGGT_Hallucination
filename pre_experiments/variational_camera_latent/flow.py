from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .model import RecognitionPosterior, VRFMModel


@dataclass(frozen=True)
class TrainingBatch:
    context: Tensor
    x0: Tensor
    x1: Tensor
    span_starts: Tensor
    endpoint_side: Tensor
    weights: Tensor


@dataclass(frozen=True)
class LossOutput:
    total: Tensor
    velocity_mse: Tensor
    kl: Tensor
    beta: float


def make_training_pairs(
    long_tokens: Tensor,
    left_tokens: Tensor,
    right_tokens: Tensor,
    *,
    context: Tensor,
    span_starts: Tensor,
) -> TrainingBatch:
    """Turn every overlap into equal-weight left and right teacher pairs."""
    if long_tokens.shape != left_tokens.shape or long_tokens.shape != right_tokens.shape:
        raise ValueError("long, left, and right token tensors must have identical shapes")
    pairs = long_tokens.shape[0]
    if context.ndim != 3 or context.shape[0] != pairs or context.shape[-1] != 2048:
        raise ValueError("context must have one [frames, 2048] sequence per overlap")
    if span_starts.shape != (pairs,):
        raise ValueError("span_starts must have one entry per overlap")
    endpoints = torch.stack((left_tokens, right_tokens), dim=1).flatten(0, 1)
    device = long_tokens.device
    return TrainingBatch(
        context=context.repeat_interleave(2, dim=0),
        x0=long_tokens.repeat_interleave(2, dim=0),
        x1=endpoints,
        span_starts=span_starts.repeat_interleave(2),
        endpoint_side=torch.tensor([0, 1], device=device).repeat(pairs),
        weights=torch.ones(2 * pairs, device=device, dtype=long_tokens.dtype),
    )


def vrfm_loss(
    model: VRFMModel,
    posterior: RecognitionPosterior,
    batch: TrainingBatch,
    *,
    progress: float,
    beta_max: float = 1e-4,
) -> LossOutput:
    if not 0.0 <= progress <= 1.0:
        raise ValueError("progress must lie in [0, 1]")
    if beta_max < 0.0:
        raise ValueError("beta_max must be non-negative")
    count = batch.x0.shape[0]
    t = torch.rand(count, device=batch.x0.device, dtype=batch.x0.dtype)
    x_t = torch.lerp(batch.x0, batch.x1, t[:, None, None])
    target = batch.x1 - batch.x0
    mu, log_var = posterior(batch.context, batch.x0, batch.x1, batch.span_starts)
    z = mu + torch.exp(0.5 * log_var) * torch.randn_like(mu)
    prediction = model(x_t, t, z, batch.context, batch.span_starts)
    per_example = torch.mean((prediction - target) ** 2, dim=(1, 2))
    mse = torch.sum(per_example * batch.weights) / torch.sum(batch.weights)
    kl = -0.5 * torch.mean(1.0 + log_var - mu.square() - log_var.exp())
    beta = beta_max * min(max(progress / 0.2, 0.0), 1.0)
    return LossOutput(total=mse + beta * kl, velocity_mse=mse, kl=kl, beta=beta)


def heun_sample(
    model: nn.Module,
    x0: Tensor,
    context: Tensor,
    span: Tensor,
    z: Tensor,
    *,
    steps: int = 16,
) -> Tensor:
    """Integrate one trajectory while holding a single segment z fixed."""
    if steps < 1:
        raise ValueError("steps must be positive")
    state = x0
    dt = 1.0 / steps
    for index in range(steps):
        t0 = torch.full(
            (x0.shape[0],), index * dt, device=x0.device, dtype=x0.dtype
        )
        first = model(state, t0, z, context, span)
        proposal = state + dt * first
        second = model(proposal, t0 + dt, z, context, span)
        state = state + 0.5 * dt * (first + second)
    return state


def deterministic_loss(model: nn.Module, batch: TrainingBatch) -> Tensor:
    count = batch.x0.shape[0]
    t = torch.rand(count, device=batch.x0.device, dtype=batch.x0.dtype)
    x_t = torch.lerp(batch.x0, batch.x1, t[:, None, None])
    prediction = model(x_t, t, batch.context, batch.span_starts)
    per_example = torch.mean((prediction - (batch.x1 - batch.x0)) ** 2, dim=(1, 2))
    return torch.sum(per_example * batch.weights) / torch.sum(batch.weights)
