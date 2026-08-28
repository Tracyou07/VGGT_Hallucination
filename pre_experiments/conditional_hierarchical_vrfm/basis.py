from __future__ import annotations

import torch
from torch import Tensor


def temporal_dct_basis(
    frame_count: int = 500,
    rank: int = 32,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Return the fixed orthonormal DCT-II basis with shape [frames, rank]."""
    if frame_count < 2 or rank < 1 or rank > frame_count:
        raise ValueError("rank must lie in [1, frame_count]")
    n = torch.arange(frame_count, device=device, dtype=torch.float64)[:, None]
    k = torch.arange(rank, device=device, dtype=torch.float64)[None, :]
    basis = torch.cos(torch.pi * (n + 0.5) * k / frame_count)
    basis[:, 0] *= frame_count ** -0.5
    if rank > 1:
        basis[:, 1:] *= (2.0 / frame_count) ** 0.5
    return basis.to(dtype=dtype)


def _validate_coefficients(coefficients: Tensor) -> None:
    if coefficients.ndim != 3 or coefficients.shape[1:] != (32, 2048):
        raise ValueError("coefficients must have shape [batch,32,2048]")


def expand_residual(coefficients: Tensor, basis: Tensor) -> Tensor:
    """Expand [batch, 32, 2048] DCT coefficients to [batch, 500, 2048]."""
    _validate_coefficients(coefficients)
    if basis.shape != (500, 32):
        raise ValueError("basis must have shape [500,32]")
    return torch.einsum("fr,brc->bfc", basis, coefficients)


def split_hierarchical_coefficients(
    coefficients: Tensor, global_rank: int = 4
) -> tuple[Tensor, Tensor]:
    """Split low-frequency global terms from remaining local residual terms."""
    _validate_coefficients(coefficients)
    if isinstance(global_rank, bool) or not 0 <= global_rank <= 32:
        raise ValueError("global_rank must lie in [0, 32]")
    return coefficients[:, :global_rank], coefficients[:, global_rank:]
