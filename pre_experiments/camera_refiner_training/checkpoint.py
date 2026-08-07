"""Checkpoint format with strict run-configuration identity."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from pre_experiments.camera_refiner_training.model import ModelConfig


@dataclass(frozen=True)
class CheckpointState:
    epoch: int
    model_config: ModelConfig
    condition_mean: torch.Tensor
    condition_std: torch.Tensor
    run_config: dict[str, Any]
    run_digest: str


def run_config_digest(run_config: dict[str, Any]) -> str:
    encoded = json.dumps(
        run_config, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _torch_load(path: Path, map_location: str | torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # PyTorch before weights_only was introduced.
        return torch.load(path, map_location=map_location)


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    model_config: ModelConfig,
    condition_mean: torch.Tensor,
    condition_std: torch.Tensor,
    run_config: dict[str, Any],
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    payload = {
        "schema_version": 1,
        "epoch": int(epoch),
        "model_config": model_config.to_dict(),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "condition_mean": condition_mean.detach().cpu(),
        "condition_std": condition_std.detach().cpu(),
        "run_config": run_config,
        "run_digest": run_config_digest(run_config),
    }
    torch.save(payload, temporary)
    temporary.replace(destination)


def load_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    expected_run_digest: str | None = None,
    map_location: str | torch.device = "cpu",
) -> CheckpointState:
    payload = _torch_load(Path(path), map_location)
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported checkpoint schema")
    digest = str(payload["run_digest"])
    if digest != run_config_digest(payload["run_config"]):
        raise ValueError("checkpoint configuration digest is corrupt")
    if expected_run_digest is not None and digest != expected_run_digest:
        raise ValueError("checkpoint configuration does not match requested run")
    model.load_state_dict(payload["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    return CheckpointState(
        epoch=int(payload["epoch"]),
        model_config=ModelConfig(**payload["model_config"]),
        condition_mean=payload["condition_mean"],
        condition_std=payload["condition_std"],
        run_config=dict(payload["run_config"]),
        run_digest=digest,
    )


def read_checkpoint_payload(
    path: Path, map_location: str | torch.device = "cpu"
) -> dict[str, Any]:
    """Read metadata before constructing the checkpoint's model."""
    return _torch_load(Path(path), map_location)
