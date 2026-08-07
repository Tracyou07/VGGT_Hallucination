"""Forward diffusion and deterministic DDIM sampling for residual trajectories."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol

import torch


@dataclass(frozen=True)
class DiffusionSchedule:
    alpha_cumprod: torch.Tensor

    def __post_init__(self) -> None:
        if self.alpha_cumprod.ndim != 1 or len(self.alpha_cumprod) < 2:
            raise ValueError("alpha_cumprod must contain at least two steps")
        if not torch.isfinite(self.alpha_cumprod).all():
            raise ValueError("alpha_cumprod must be finite")

    @property
    def steps(self) -> int:
        return int(len(self.alpha_cumprod))

    @classmethod
    def cosine(cls, steps: int, offset: float = 0.008) -> "DiffusionSchedule":
        if steps < 2:
            raise ValueError("steps must be at least two")
        positions = torch.linspace(0, steps, steps + 1, dtype=torch.float64) / steps
        cumulative = torch.cos((positions + offset) / (1.0 + offset) * math.pi / 2) ** 2
        cumulative = cumulative / cumulative[0]
        betas = 1.0 - cumulative[1:] / cumulative[:-1]
        betas = betas.clamp(0.0, 0.999)
        return cls(torch.cumprod(1.0 - betas, dim=0).float())


def _coefficients(
    schedule: DiffusionSchedule,
    timestep: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    if timestep.ndim != 1 or len(timestep) != len(reference):
        raise ValueError("timestep must have shape [B]")
    if torch.any(timestep < 0) or torch.any(timestep >= schedule.steps):
        raise ValueError("timestep is outside diffusion schedule")
    values = schedule.alpha_cumprod.to(device=reference.device, dtype=reference.dtype)
    return values[timestep.long()].view(len(reference), *([1] * (reference.ndim - 1)))


def q_sample(
    clean: torch.Tensor,
    timestep: torch.Tensor,
    noise: torch.Tensor,
    schedule: DiffusionSchedule,
) -> torch.Tensor:
    if clean.shape != noise.shape or clean.ndim != 3:
        raise ValueError("clean and noise must have matching [B, S, D] shape")
    alpha = _coefficients(schedule, timestep, clean)
    return alpha.sqrt() * clean + (1.0 - alpha).sqrt() * noise


class Denoiser(Protocol):
    def __call__(
        self,
        noisy_residual: torch.Tensor,
        condition: torch.Tensor,
        timestep: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


@torch.no_grad()
def ddim_sample(
    model: Denoiser,
    condition: torch.Tensor,
    schedule: DiffusionSchedule,
    *,
    sample_steps: int = 10,
    generator: torch.Generator | None = None,
    initial_noise: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if condition.ndim != 3:
        raise ValueError("condition must have shape [B, S, D]")
    if sample_steps < 1 or sample_steps > schedule.steps:
        raise ValueError("sample_steps must be within the diffusion schedule")
    shape = (*condition.shape[:2], 3)
    if initial_noise is None:
        current = torch.randn(
            shape,
            device=condition.device,
            dtype=condition.dtype,
            generator=generator,
        )
    else:
        current = initial_noise.to(device=condition.device, dtype=condition.dtype)
        if current.shape != shape:
            raise ValueError("initial_noise shape does not match condition")
    indices = torch.linspace(
        schedule.steps - 1,
        0,
        sample_steps,
        dtype=torch.float64,
    ).round().long().unique_consecutive()
    confidence = torch.zeros((*condition.shape[:2], 1), device=condition.device, dtype=condition.dtype)
    alpha_values = schedule.alpha_cumprod.to(condition.device, condition.dtype)
    for index, timestep_value in enumerate(indices):
        timestep = torch.full(
            (len(condition),),
            int(timestep_value),
            device=condition.device,
            dtype=torch.long,
        )
        clean, confidence = model(current, condition, timestep)
        if index == len(indices) - 1:
            current = clean
            break
        previous = int(indices[index + 1])
        alpha = alpha_values[int(timestep_value)]
        alpha_previous = alpha_values[previous]
        predicted_noise = (current - alpha.sqrt() * clean) / (1.0 - alpha).sqrt().clamp_min(1e-8)
        current = alpha_previous.sqrt() * clean + (1.0 - alpha_previous).sqrt() * predicted_noise
    return current, confidence
