from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


def pose_encoding_to_c2w(raw_pose_encoding: Tensor) -> Tensor:
    """Convert raw Camera Head pose encodings to homogeneous camera-to-world matrices."""
    if (
        not isinstance(raw_pose_encoding, Tensor)
        or raw_pose_encoding.ndim != 3
        or raw_pose_encoding.shape[-1] != 9
        or not torch.isfinite(raw_pose_encoding).all()
    ):
        raise ValueError("raw pose encoding must be finite with shape [batch, frames, 9]")
    w2c_3x4, _ = pose_encoding_to_extri_intri(
        raw_pose_encoding, build_intrinsics=False
    )
    bottom = torch.zeros(
        (*w2c_3x4.shape[:2], 1, 4),
        dtype=w2c_3x4.dtype,
        device=w2c_3x4.device,
    )
    bottom[..., 0, 3] = 1.0
    w2c = torch.cat((w2c_3x4, bottom), dim=-2)
    try:
        c2w = torch.linalg.inv(w2c)
    except RuntimeError as error:
        raise ValueError("Camera Head produced non-invertible extrinsics") from error
    if not torch.isfinite(c2w).all():
        raise ValueError("Camera Head produced non-finite camera matrices")
    return c2w


def decode_camera_tokens(
    camera_head: nn.Module | Any,
    tokens: Tensor,
    *,
    iterations: int = 4,
) -> Tensor:
    """Decode normalized Camera tokens through the frozen VGGT Camera Head."""
    if not isinstance(tokens, Tensor) or tokens.ndim != 3 or tokens.shape[-1] != 2048:
        raise ValueError("Camera tokens must have shape [batch, frames, 2048]")
    if iterations < 1:
        raise ValueError("iterations must be positive")
    if not torch.isfinite(tokens).all():
        raise ValueError("Camera tokens contain non-finite values")
    with torch.no_grad():
        trace = camera_head.decode_pose_tokens(tokens, num_iterations=iterations)
    if not isinstance(trace, (list, tuple)) or not trace:
        raise ValueError("Camera Head returned a malformed decode trace")
    raw = trace[-1]
    if (
        not isinstance(raw, Tensor)
        or raw.shape != (*tokens.shape[:2], 9)
        or not torch.isfinite(raw).all()
    ):
        raise ValueError("Camera Head produced non-finite or malformed pose encoding")
    return raw


def run_latent_preflight(
    camera_head: nn.Module | Any,
    long_tokens: Tensor,
    left_tokens: Tensor,
    right_tokens: Tensor,
) -> dict[str, object]:
    """Check endpoints and both straight latent paths without judging quality."""
    if left_tokens.shape != long_tokens.shape or right_tokens.shape != long_tokens.shape:
        raise ValueError("long, left, and right Camera tokens must have identical shapes")
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    decoded_long = decode_camera_tokens(camera_head, long_tokens)
    decoded_left = decode_camera_tokens(camera_head, left_tokens)
    decoded_right = decode_camera_tokens(camera_head, right_tokens)
    path_norms: dict[str, list[float]] = {"left": [], "right": []}
    for name, endpoint in (("left", left_tokens), ("right", right_tokens)):
        for alpha in alphas:
            mixed = torch.lerp(long_tokens, endpoint, alpha)
            decoded = decode_camera_tokens(camera_head, mixed)
            path_norms[name].append(float(torch.linalg.vector_norm(decoded).cpu()))
    return {
        "alphas": alphas,
        "all_finite": all(
            torch.isfinite(value).all().item()
            for value in (decoded_long, decoded_left, decoded_right)
        ),
        "decoded_shape": list(decoded_long.shape),
        "endpoint_pose_mse": {
            "left": float(torch.mean((decoded_long - decoded_left) ** 2).cpu()),
            "right": float(torch.mean((decoded_long - decoded_right) ** 2).cpu()),
        },
        "path_pose_norms": path_norms,
    }
