"""Supervised and cross-window objectives for translation residual refinement."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional


@dataclass(frozen=True)
class LossWeights:
    denoising: float = 1.0
    center: float = 1.0
    relative_motion: float = 0.5
    overlap: float = 0.25
    gate: float = 0.001

    def __post_init__(self) -> None:
        if any(value < 0 for value in self.__dict__.values()):
            raise ValueError("loss weights must be non-negative")


def _overlap_loss(
    applied: torch.Tensor,
    scene_ids: tuple[str, ...],
    starts: torch.Tensor,
) -> torch.Tensor:
    terms = []
    window_length = applied.shape[1]
    for left in range(len(applied)):
        for right in range(left + 1, len(applied)):
            if scene_ids[left] != scene_ids[right]:
                continue
            left_start = int(starts[left])
            right_start = int(starts[right])
            overlap_start = max(left_start, right_start)
            overlap_stop = min(left_start + window_length, right_start + window_length)
            if overlap_stop <= overlap_start:
                continue
            left_slice = slice(overlap_start - left_start, overlap_stop - left_start)
            right_slice = slice(overlap_start - right_start, overlap_stop - right_start)
            terms.append(functional.mse_loss(applied[left, left_slice], applied[right, right_slice]))
    if not terms:
        return applied.new_zeros(())
    return torch.stack(terms).mean()


def training_losses(
    predicted_clean: torch.Tensor,
    confidence: torch.Tensor,
    target_residual: torch.Tensor,
    global_centers: torch.Tensor,
    *,
    scene_ids: tuple[str, ...],
    starts: torch.Tensor,
    weights: LossWeights = LossWeights(),
    lags: tuple[int, ...] = (1, 5, 10, 25),
) -> dict[str, torch.Tensor]:
    """Compute losses in a shared prediction-derived scene gauge."""
    if predicted_clean.ndim != 3 or predicted_clean.shape[2] != 3:
        raise ValueError("predicted_clean must have shape [B, S, 3]")
    if target_residual.shape != predicted_clean.shape or global_centers.shape != predicted_clean.shape:
        raise ValueError("target_residual and global_centers must match prediction")
    if confidence.shape != (*predicted_clean.shape[:2], 1):
        raise ValueError("confidence must have shape [B, S, 1]")
    if len(scene_ids) != len(predicted_clean) or starts.shape != (len(predicted_clean),):
        raise ValueError("scene_ids and starts must match batch size")
    if not lags or any(lag < 1 or lag >= predicted_clean.shape[1] for lag in lags):
        raise ValueError("lags must be positive and shorter than the window")

    gated = predicted_clean * confidence
    corrected = global_centers + gated
    target_centers = global_centers + target_residual
    denoising = functional.mse_loss(predicted_clean, target_residual)
    center = functional.mse_loss(corrected, target_centers)
    relative_terms = []
    for lag in lags:
        predicted_motion = corrected[:, lag:] - corrected[:, :-lag]
        target_motion = target_centers[:, lag:] - target_centers[:, :-lag]
        relative_terms.append(functional.mse_loss(predicted_motion, target_motion))
    relative_motion = torch.stack(relative_terms).mean()
    overlap = _overlap_loss(gated, scene_ids, starts)
    gate = torch.linalg.vector_norm(gated, dim=-1).mean()
    total = (
        weights.denoising * denoising
        + weights.center * center
        + weights.relative_motion * relative_motion
        + weights.overlap * overlap
        + weights.gate * gate
    )
    return {
        "total": total,
        "denoising": denoising,
        "center": center,
        "relative_motion": relative_motion,
        "overlap": overlap,
        "gate": gate,
    }
