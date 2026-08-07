"""Compact one-dimensional DiT for camera-center residuals."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    condition_dim: int
    hidden_size: int = 256
    depth: int = 6
    num_heads: int = 8
    max_frames: int = 100
    mlp_ratio: float = 4.0

    def __post_init__(self) -> None:
        if self.condition_dim < 1 or self.hidden_size < 4 or self.depth < 1:
            raise ValueError("condition_dim, hidden_size, and depth must be positive")
        if self.num_heads < 1 or self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.max_frames < 2 or self.mlp_ratio <= 0:
            raise ValueError("max_frames and mlp_ratio are invalid")

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def timestep_embedding(timestep: torch.Tensor, dimension: int) -> torch.Tensor:
    if timestep.ndim != 1:
        raise ValueError("timestep must have shape [B]")
    half = dimension // 2
    frequencies = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, device=timestep.device, dtype=torch.float32)
        / max(half, 1)
    )
    angles = timestep.float()[:, None] * frequencies[None]
    embedding = torch.cat([torch.cos(angles), torch.sin(angles)], dim=1)
    if dimension % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=1)
    return embedding


def _modulate(value: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return value * (1.0 + scale[:, None]) + shift[:, None]


class DiTBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float) -> None:
        super().__init__()
        self.norm_attention = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.attention = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.norm_mlp = nn.LayerNorm(hidden_size, elementwise_affine=False)
        inner = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, inner),
            nn.GELU(approximate="tanh"),
            nn.Linear(inner, hidden_size),
        )
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size),
        )
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, value: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        shift_attention, scale_attention, gate_attention, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation(timestep).chunk(6, dim=1)
        )
        attention_input = _modulate(
            self.norm_attention(value), shift_attention, scale_attention
        )
        attended, _ = self.attention(
            attention_input,
            attention_input,
            attention_input,
            need_weights=False,
        )
        value = value + gate_attention[:, None] * attended
        value = value + gate_mlp[:, None] * self.mlp(
            _modulate(self.norm_mlp(value), shift_mlp, scale_mlp)
        )
        return value


class ResidualDiT(nn.Module):
    """Denoise `[B, S, 3]` residuals conditioned by per-frame VGGT evidence."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.input_projection = nn.Linear(config.condition_dim + 3, config.hidden_size)
        self.position = nn.Parameter(torch.zeros(1, config.max_frames, config.hidden_size))
        self.time_mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.SiLU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )
        self.blocks = nn.ModuleList(
            [
                DiTBlock(config.hidden_size, config.num_heads, config.mlp_ratio)
                for _ in range(config.depth)
            ]
        )
        self.final_norm = nn.LayerNorm(config.hidden_size, elementwise_affine=False)
        self.final_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(config.hidden_size, 2 * config.hidden_size),
        )
        self.residual_head = nn.Linear(config.hidden_size, 3)
        self.confidence_head = nn.Linear(config.hidden_size, 1)
        nn.init.normal_(self.position, std=0.02)
        nn.init.zeros_(self.final_modulation[-1].weight)
        nn.init.zeros_(self.final_modulation[-1].bias)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        nn.init.zeros_(self.confidence_head.weight)
        nn.init.zeros_(self.confidence_head.bias)

    def forward(
        self,
        noisy_residual: torch.Tensor,
        condition: torch.Tensor,
        timestep: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if noisy_residual.ndim != 3 or noisy_residual.shape[2] != 3:
            raise ValueError("noisy_residual must have shape [B, S, 3]")
        if condition.ndim != 3 or condition.shape[:2] != noisy_residual.shape[:2]:
            raise ValueError("condition batch and frame dimensions must match residual")
        if condition.shape[2] != self.config.condition_dim:
            raise ValueError("condition feature dimension does not match model config")
        if timestep.ndim != 1 or len(timestep) != len(noisy_residual):
            raise ValueError("timestep must have shape [B]")
        if noisy_residual.shape[1] > self.config.max_frames:
            raise ValueError("frame count exceeds model max_frames")
        value = self.input_projection(torch.cat([noisy_residual, condition], dim=-1))
        value = value + self.position[:, : value.shape[1]]
        time = self.time_mlp(timestep_embedding(timestep, self.config.hidden_size))
        for block in self.blocks:
            value = block(value, time)
        shift, scale = self.final_modulation(time).chunk(2, dim=1)
        value = _modulate(self.final_norm(value), shift, scale)
        return self.residual_head(value), torch.sigmoid(self.confidence_head(value))
