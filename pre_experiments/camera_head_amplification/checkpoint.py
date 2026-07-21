"""Selective local checkpoint loading for the Camera Head only."""

from __future__ import annotations

from pathlib import Path

import torch

from pre_experiments.camera_iteration.model_io import find_checkpoint
from vggt.heads.camera_head import CameraHead


def _camera_state_from_torch(path: Path) -> dict[str, torch.Tensor]:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError(f"checkpoint must contain a state dictionary: {path}")
    return {
        key.removeprefix("camera_head."): value
        for key, value in state.items()
        if isinstance(key, str) and key.startswith("camera_head.")
    }


def _camera_state_from_safetensors(path: Path) -> dict[str, torch.Tensor]:
    try:
        from safetensors import safe_open
    except ImportError as error:  # pragma: no cover - environment setup owns this.
        raise RuntimeError("safetensors is required to load model.safetensors") from error

    state: dict[str, torch.Tensor] = {}
    with safe_open(str(path), framework="pt", device="cpu") as checkpoint:
        for key in checkpoint.keys():
            if key.startswith("camera_head."):
                state[key.removeprefix("camera_head.")] = checkpoint.get_tensor(key)
    return state


def load_local_camera_head(checkpoint_dir: Path) -> CameraHead:
    """Load only Camera Head tensors from an official local VGGT checkpoint."""
    checkpoint_path = find_checkpoint(checkpoint_dir)
    if checkpoint_path.suffix == ".safetensors":
        state = _camera_state_from_safetensors(checkpoint_path)
    else:
        state = _camera_state_from_torch(checkpoint_path)
    if not state:
        raise ValueError(f"checkpoint has no camera_head parameters: {checkpoint_path}")
    camera_head = CameraHead()
    camera_head.load_state_dict(state, strict=True)
    return camera_head
