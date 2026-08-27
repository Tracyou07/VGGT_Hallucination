from __future__ import annotations

import torch
from torch import Tensor, nn


class _ContextBlock(nn.Module):
    def __init__(self, d_model: int, heads: int) -> None:
        super().__init__()
        self.self_norm = nn.LayerNorm(d_model)
        self.self_attention = nn.MultiheadAttention(d_model, heads, batch_first=True)
        self.cross_norm = nn.LayerNorm(d_model)
        self.cross_attention = nn.MultiheadAttention(d_model, heads, batch_first=True)
        self.feed_norm = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, state: Tensor, context: Tensor) -> Tensor:
        normalized = self.self_norm(state)
        state = state + self.self_attention(
            normalized, normalized, normalized, need_weights=False
        )[0]
        normalized = self.cross_norm(state)
        state = state + self.cross_attention(
            normalized, context, context, need_weights=False
        )[0]
        return state + self.feed_forward(self.feed_norm(state))


class _ContextualVelocity(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        layers: int,
        heads: int,
        z_dim: int | None,
    ) -> None:
        super().__init__()
        if d_model < 1 or layers < 1 or heads < 1 or d_model % heads:
            raise ValueError("d_model must be positive and divisible by heads; layers must be positive")
        self.z_dim = z_dim
        self.input_adapter = nn.Linear(2048, d_model)
        self.context_adapter = nn.Linear(2048, d_model)
        self.time_embedding = nn.Sequential(nn.Linear(1, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.span_embedding = nn.Embedding(8, d_model)
        self.z_embedding = None if z_dim is None else nn.Linear(z_dim, d_model)
        self.blocks = nn.ModuleList([_ContextBlock(d_model, heads) for _ in range(layers)])
        self.output_norm = nn.LayerNorm(d_model)
        self.output_adapter = nn.Linear(d_model, 2048)

    def _forward(
        self,
        x_t: Tensor,
        t: Tensor,
        global_tokens: Tensor,
        span_starts: Tensor,
        z: Tensor | None,
    ) -> Tensor:
        if x_t.ndim != 3 or x_t.shape[-1] != 2048:
            raise ValueError("x_t must have shape [batch, frames, 2048]")
        batch = x_t.shape[0]
        if t.shape != (batch,):
            raise ValueError("t must have shape [batch]")
        if global_tokens.ndim != 3 or global_tokens.shape[0] != batch or global_tokens.shape[-1] != 2048:
            raise ValueError("global_tokens must have shape [batch, context_frames, 2048]")
        if span_starts.shape != (batch,) or torch.any(span_starts % 50 != 0):
            raise ValueError("span_starts must contain one 50-frame-aligned start per batch item")
        span_indices = span_starts.to(dtype=torch.long) // 50
        if torch.any(span_indices < 0) or torch.any(span_indices >= 8):
            raise ValueError("span_starts must lie between 0 and 350")
        state = self.input_adapter(x_t)
        state = state + self.time_embedding(t[:, None])[:, None, :]
        state = state + self.span_embedding(span_indices)[:, None, :]
        if self.z_embedding is not None:
            if z is None or z.shape != (batch, self.z_dim):
                raise ValueError(f"z must have shape [batch, {self.z_dim}]")
            state = state + self.z_embedding(z)[:, None, :]
        context = self.context_adapter(global_tokens)
        for block in self.blocks:
            state = block(state, context)
        return self.output_adapter(self.output_norm(state))


class VRFMModel(_ContextualVelocity):
    def __init__(
        self,
        *,
        d_model: int = 256,
        z_dim: int = 16,
        layers: int = 4,
        heads: int = 8,
    ) -> None:
        if z_dim < 1:
            raise ValueError("z_dim must be positive")
        super().__init__(d_model=d_model, layers=layers, heads=heads, z_dim=z_dim)

    def forward(
        self,
        x_t: Tensor,
        t: Tensor,
        z: Tensor,
        global_tokens: Tensor,
        span_starts: Tensor,
    ) -> Tensor:
        return self._forward(x_t, t, global_tokens, span_starts, z)


class DeterministicRFMModel(_ContextualVelocity):
    def __init__(
        self,
        *,
        d_model: int = 256,
        layers: int = 4,
        heads: int = 8,
    ) -> None:
        super().__init__(d_model=d_model, layers=layers, heads=heads, z_dim=None)

    def forward(
        self,
        x_t: Tensor,
        t: Tensor,
        global_tokens: Tensor,
        span_starts: Tensor,
    ) -> Tensor:
        return self._forward(x_t, t, global_tokens, span_starts, None)


class RecognitionPosterior(nn.Module):
    """Training-only q(z | long source, local teacher, global context)."""

    def __init__(self, *, d_model: int = 256, z_dim: int = 16) -> None:
        super().__init__()
        if d_model < 1 or z_dim < 1:
            raise ValueError("d_model and z_dim must be positive")
        self.delta_adapter = nn.Linear(2048, d_model)
        self.context_adapter = nn.Linear(2048, d_model)
        self.span_embedding = nn.Embedding(8, d_model)
        self.output = nn.Sequential(
            nn.Linear(3 * d_model, 2 * d_model),
            nn.SiLU(),
            nn.Linear(2 * d_model, 2 * z_dim),
        )

    def forward(
        self,
        global_tokens: Tensor,
        x0: Tensor,
        x1: Tensor,
        span_starts: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if x0.shape != x1.shape or x0.ndim != 3 or x0.shape[-1] != 2048:
            raise ValueError("x0 and x1 must have matching [batch, frames, 2048] shapes")
        batch = x0.shape[0]
        if global_tokens.ndim != 3 or global_tokens.shape[0] != batch or global_tokens.shape[-1] != 2048:
            raise ValueError("global_tokens must have shape [batch, context_frames, 2048]")
        if span_starts.shape != (batch,):
            raise ValueError("span_starts must have shape [batch]")
        span_indices = span_starts.to(dtype=torch.long) // 50
        delta = self.delta_adapter(x1 - x0).mean(dim=1)
        context = self.context_adapter(global_tokens).mean(dim=1)
        stats = self.output(
            torch.cat((delta, context, self.span_embedding(span_indices)), dim=-1)
        )
        return stats.chunk(2, dim=-1)
