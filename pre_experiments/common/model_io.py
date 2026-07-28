"""Local VGGT checkpoint and device handling for the camera study."""

from __future__ import annotations

from pathlib import Path

import torch

from vggt.models.vggt import VGGT

try:
    from safetensors.torch import load_file as load_safetensors
except ImportError:  # pragma: no cover - reported only when that checkpoint type is used.
    load_safetensors = None


def find_checkpoint(checkpoint_dir: Path) -> Path:
    """Find a supported local VGGT checkpoint without network fallback."""
    for name in ("model.safetensors", "model.pt"):
        candidate = checkpoint_dir / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"No model.safetensors or model.pt found in CKPT_DIR={checkpoint_dir}"
    )


def resolve_device(value: str) -> torch.device:
    """Resolve auto/cpu/cuda while failing clearly for unavailable CUDA."""
    if value not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda")
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(value)


def load_local_model(checkpoint_dir: Path) -> VGGT:
    """Construct the pose-only VGGT path and load a local state dictionary."""
    checkpoint_path = find_checkpoint(checkpoint_dir)
    model = VGGT(enable_track=False, enable_point=False, enable_depth=False)
    if checkpoint_path.suffix == ".safetensors":
        if load_safetensors is None:
            raise RuntimeError("safetensors is required to load model.safetensors")
        state_dict = load_safetensors(str(checkpoint_path), device="cpu")
    else:
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    return model
