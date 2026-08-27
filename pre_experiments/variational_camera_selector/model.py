from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def summarize_sequence(projected: Tensor) -> Tensor:
    """Mean/std summaries of values and adjacent temporal differences."""
    if projected.ndim < 3 or projected.shape[-2] < 2:
        raise ValueError("projected sequence must have at least two temporal tokens")
    if not torch.is_floating_point(projected) or not torch.isfinite(projected).all():
        raise ValueError("projected sequence must contain finite floating-point values")
    difference = projected[..., 1:, :] - projected[..., :-1, :]
    return torch.cat(
        (
            projected.mean(dim=-2),
            projected.std(dim=-2, unbiased=False),
            difference.mean(dim=-2),
            difference.std(dim=-2, unbiased=False),
        ),
        dim=-1,
    )


class CandidateRanker(nn.Module):
    """Score latent correction choices using prediction-only Camera-token features."""

    def __init__(
        self,
        *,
        d_model: int = 128,
        z_dim: int = 16,
        include_global_context: bool = True,
        input_dim: int = 2048,
        span_count: int = 8,
    ) -> None:
        super().__init__()
        if d_model < 1 or z_dim < 1 or input_dim < 1 or span_count != 8:
            raise ValueError("ranker dimensions must be positive and span_count must be eight")
        self.d_model = int(d_model)
        self.z_dim = int(z_dim)
        self.input_dim = int(input_dim)
        self.span_count = int(span_count)
        self.include_global_context = bool(include_global_context)

        self.token_projector = nn.Linear(self.input_dim, self.d_model)
        self.span_embedding = nn.Embedding(self.span_count, self.d_model)
        self.alpha_embedding = nn.Sequential(
            nn.Linear(3, self.d_model),
            nn.SiLU(),
        )
        self.z_embedding = nn.Sequential(
            nn.Linear(self.z_dim, self.d_model),
            nn.SiLU(),
        )
        # Three 4*d summaries plus span/alpha/z embeddings and residual RMS.
        feature_dim = 15 * self.d_model + 1
        self.feature_norm = nn.LayerNorm(feature_dim)
        self.score_head = nn.Sequential(
            nn.Linear(feature_dim, 2 * self.d_model),
            nn.SiLU(),
            nn.Linear(2 * self.d_model, 1),
        )

    def _validate_inputs(
        self,
        global_tokens: Tensor,
        x0: Tensor,
        delta_tokens: Tensor,
        alphas: Tensor,
        span_starts: Tensor,
        z: Tensor,
    ) -> tuple[int, int]:
        if global_tokens.ndim != 3 or global_tokens.shape[-1] != self.input_dim:
            raise ValueError("global_tokens must have shape [batch, frames, input_dim]")
        if x0.ndim != 3 or x0.shape[-1] != self.input_dim:
            raise ValueError("x0 must have shape [batch, frames, input_dim]")
        if delta_tokens.ndim != 4 or delta_tokens.shape[-1] != self.input_dim:
            raise ValueError(
                "delta_tokens must have shape [batch, choices, frames, input_dim]"
            )
        batch, choices = delta_tokens.shape[:2]
        if global_tokens.shape[0] != batch or x0.shape[0] != batch:
            raise ValueError("global, x0, and candidate batch dimensions must match")
        if x0.shape[1] != delta_tokens.shape[2]:
            raise ValueError("x0 and candidate temporal dimensions must match")
        if alphas.shape != (batch, choices):
            raise ValueError("alphas must have shape [batch, choices]")
        if z.shape != (batch, choices, self.z_dim):
            raise ValueError("z must have shape [batch, choices, z_dim]")
        if span_starts.shape not in {(batch,), (batch, choices)}:
            raise ValueError("span_starts must have shape [batch] or [batch, choices]")
        for name, value in (
            ("x0", x0),
            ("delta_tokens", delta_tokens),
            ("alphas", alphas),
            ("z", z),
        ):
            if not torch.is_floating_point(value) or not torch.isfinite(value).all():
                raise ValueError(f"{name} must contain finite floating-point values")
        if self.include_global_context and (
            not torch.is_floating_point(global_tokens)
            or not torch.isfinite(global_tokens).all()
        ):
            raise ValueError("global_tokens must contain finite floating-point values")
        if not torch.is_floating_point(alphas) or torch.any((alphas < 0) | (alphas > 1)):
            raise ValueError("alphas must lie in [0, 1]")
        return batch, choices

    def _span_indices(
        self, span_starts: Tensor, *, batch: int, choices: int
    ) -> Tensor:
        if torch.is_floating_point(span_starts):
            raise ValueError("span_starts must be integer-valued")
        values = span_starts.to(dtype=torch.long)
        if values.numel() and int(values.max()) >= self.span_count:
            if torch.any(values % 50 != 0):
                raise ValueError("span starts must be multiples of 50")
            values = values // 50
        if torch.any((values < 0) | (values >= self.span_count)):
            raise ValueError("span indices must lie in [0, 7]")
        if values.ndim == 1:
            values = values[:, None].expand(batch, choices)
        return values

    def forward(
        self,
        global_tokens: Tensor,
        x0: Tensor,
        delta_tokens: Tensor,
        alphas: Tensor,
        span_starts: Tensor,
        z: Tensor,
    ) -> Tensor:
        batch, choices = self._validate_inputs(
            global_tokens, x0, delta_tokens, alphas, span_starts, z
        )
        x0_summary = summarize_sequence(self.token_projector(x0))
        candidate_projected = self.token_projector(
            delta_tokens.reshape(batch * choices, delta_tokens.shape[2], self.input_dim)
        )
        candidate_summary = summarize_sequence(candidate_projected).reshape(
            batch, choices, 4 * self.d_model
        )
        if self.include_global_context:
            global_summary = summarize_sequence(self.token_projector(global_tokens))
        else:
            # Keep the head capacity exactly matched while skipping G computation entirely.
            global_summary = x0_summary.new_zeros((batch, 4 * self.d_model))
        global_features = global_summary[:, None].expand(-1, choices, -1)
        x0_features = x0_summary[:, None].expand(-1, choices, -1)
        span_features = self.span_embedding(
            self._span_indices(span_starts, batch=batch, choices=choices)
        )
        alpha_features_raw = torch.stack(
            (
                alphas,
                alphas.square(),
                torch.log1p(100.0 * alphas) / math.log(101.0),
            ),
            dim=-1,
        )
        alpha_features = self.alpha_embedding(alpha_features_raw)
        z_features = self.z_embedding(z)
        residual_rms = torch.sqrt(
            delta_tokens.float().square().mean(dim=(-2, -1)).clamp_min(0.0)
        ).to(delta_tokens.dtype)[..., None]
        features = torch.cat(
            (
                global_features,
                x0_features,
                candidate_summary,
                span_features,
                alpha_features,
                z_features,
                residual_rms,
            ),
            dim=-1,
        )
        scores = self.score_head(self.feature_norm(features)).squeeze(-1)
        if scores.shape != (batch, choices) or not torch.isfinite(scores).all():
            raise ValueError("ranker produced invalid scores")
        return scores
