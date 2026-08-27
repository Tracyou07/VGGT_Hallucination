from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random
from typing import Sequence

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .flow import TrainingBatch, deterministic_loss, vrfm_loss
from .model import DeterministicRFMModel, RecognitionPosterior, VRFMModel
from .source import load_source_shard


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class OverlapDataset(Dataset):
    """Prediction-only left/right overlap pairs; no privileged input exists."""

    def __init__(self, source_paths: Sequence[Path]) -> None:
        if not source_paths:
            raise ValueError("at least one source shard is required")
        self._sources = [load_source_shard(Path(path)) for path in source_paths]

    def __len__(self) -> int:
        return 16 * len(self._sources)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        source_index, within = divmod(index, 16)
        overlap, side = divmod(within, 2)
        arrays = self._sources[source_index]
        endpoint_name = "overlap_left_tokens" if side == 0 else "overlap_right_tokens"
        return {
            "context": torch.from_numpy(arrays["global_camera_tokens"]),
            "x0": torch.from_numpy(arrays["overlap_long_tokens"][overlap]),
            "x1": torch.from_numpy(arrays[endpoint_name][overlap]),
            "span_starts": torch.tensor(arrays["span_starts"][overlap], dtype=torch.long),
            "endpoint_side": torch.tensor(side, dtype=torch.long),
            "weight": torch.tensor(1.0, dtype=torch.float32),
        }


@dataclass(frozen=True)
class TrainConfig:
    source_paths: tuple[Path, ...]
    run_root: Path
    max_steps: int
    batch_size: int = 8
    learning_rate: float = 1e-4
    seed: int = 20260827
    device: str = "cuda"
    d_model: int = 256
    z_dim: int = 16
    layers: int = 4
    heads: int = 8
    beta_max: float = 1e-4
    checkpoint_interval: int = 100
    git_commit: str = "unknown"


@dataclass(frozen=True)
class TrainingResult:
    start_step: int
    completed_step: int
    checkpoint_path: Path
    metrics_path: Path


def _immutable_config_payload(config: TrainConfig) -> dict[str, object]:
    return {
        "source_paths": [str(Path(path).resolve()) for path in config.source_paths],
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "seed": config.seed,
        "device": config.device,
        "d_model": config.d_model,
        "z_dim": config.z_dim,
        "layers": config.layers,
        "heads": config.heads,
        "beta_max": config.beta_max,
        "git_commit": config.git_commit,
    }


def _digest_json(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _stack_batch(examples: list[dict[str, Tensor]], device: torch.device) -> TrainingBatch:
    def stack(name: str) -> Tensor:
        return torch.stack([example[name] for example in examples]).to(device)

    return TrainingBatch(
        context=stack("context"),
        x0=stack("x0"),
        x1=stack("x1"),
        span_starts=stack("span_starts"),
        endpoint_side=stack("endpoint_side"),
        weights=stack("weight"),
    )


def _checkpoint_payload(
    *,
    step: int,
    config_digest: str,
    source_digest: str,
    vrfm: VRFMModel,
    posterior: RecognitionPosterior,
    deterministic: DeterministicRFMModel,
    vrfm_optimizer: torch.optim.Optimizer,
    deterministic_optimizer: torch.optim.Optimizer,
) -> dict[str, object]:
    return {
        "schema": "variational_camera_latent.training_checkpoint.v1",
        "step": step,
        "config_digest": config_digest,
        "source_digest": source_digest,
        "vrfm": vrfm.state_dict(),
        "posterior": posterior.state_dict(),
        "deterministic": deterministic.state_dict(),
        "vrfm_optimizer": vrfm_optimizer.state_dict(),
        "deterministic_optimizer": deterministic_optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
    }


def _save_checkpoint(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _trim_metrics(path: Path, completed_step: int) -> None:
    if not path.is_file():
        return
    kept: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError("training metrics JSONL is malformed") from error
        if int(payload["step"]) < completed_step:
            kept.append(json.dumps(payload, sort_keys=True))
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(line + "\n" for line in kept), encoding="utf-8")
    temporary.replace(path)


def train_models(config: TrainConfig) -> TrainingResult:
    """Train matched VRFM/RFM models and resume from the exact next step."""
    if config.max_steps < 1 or config.batch_size < 1 or config.checkpoint_interval < 1:
        raise ValueError("max_steps, batch_size, and checkpoint_interval must be positive")
    if config.learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA training requested but CUDA is unavailable")

    dataset = OverlapDataset(config.source_paths)
    config_digest = _digest_json(_immutable_config_payload(config))
    source_digest = _digest_json(
        [{"path": str(Path(path).resolve()), "sha256": _sha256_file(path)} for path in config.source_paths]
    )
    run_root = Path(config.run_root)
    checkpoint_path = run_root / "checkpoints" / "latest.pt"
    metrics_path = run_root / "metrics" / "training.jsonl"
    state_path = run_root / "training_state.json"

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    vrfm = VRFMModel(
        d_model=config.d_model,
        z_dim=config.z_dim,
        layers=config.layers,
        heads=config.heads,
    ).to(device)
    posterior = RecognitionPosterior(d_model=config.d_model, z_dim=config.z_dim).to(device)
    deterministic = DeterministicRFMModel(
        d_model=config.d_model, layers=config.layers, heads=config.heads
    ).to(device)
    vrfm_optimizer = torch.optim.AdamW(
        list(vrfm.parameters()) + list(posterior.parameters()), lr=config.learning_rate
    )
    deterministic_optimizer = torch.optim.AdamW(
        deterministic.parameters(), lr=config.learning_rate
    )
    start_step = 0
    if checkpoint_path.is_file():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if checkpoint.get("config_digest") != config_digest or checkpoint.get("source_digest") != source_digest:
            raise ValueError("checkpoint config or source digest mismatch")
        start_step = int(checkpoint["step"])
        vrfm.load_state_dict(checkpoint["vrfm"])
        posterior.load_state_dict(checkpoint["posterior"])
        deterministic.load_state_dict(checkpoint["deterministic"])
        vrfm_optimizer.load_state_dict(checkpoint["vrfm_optimizer"])
        deterministic_optimizer.load_state_dict(checkpoint["deterministic_optimizer"])
        torch.set_rng_state(checkpoint["torch_rng_state"])
        np.random.set_state(checkpoint["numpy_rng_state"])
        random.setstate(checkpoint["python_rng_state"])
    if start_step > config.max_steps:
        raise ValueError("checkpoint step exceeds requested max_steps")
    _trim_metrics(metrics_path, start_step)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    vrfm.train()
    posterior.train()
    deterministic.train()
    for step in range(start_step, config.max_steps):
        indices = [
            (step * config.batch_size + offset) % len(dataset)
            for offset in range(config.batch_size)
        ]
        batch = _stack_batch([dataset[index] for index in indices], device)
        progress = (step + 1) / config.max_steps

        vrfm_optimizer.zero_grad(set_to_none=True)
        variational = vrfm_loss(
            vrfm, posterior, batch, progress=progress, beta_max=config.beta_max
        )
        variational.total.backward()
        vrfm_optimizer.step()

        deterministic_optimizer.zero_grad(set_to_none=True)
        baseline = deterministic_loss(deterministic, batch)
        baseline.backward()
        deterministic_optimizer.step()

        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "step": step,
                        "vrfm_total": float(variational.total.detach().cpu()),
                        "velocity_mse": float(variational.velocity_mse.detach().cpu()),
                        "kl": float(variational.kl.detach().cpu()),
                        "beta": variational.beta,
                        "deterministic_mse": float(baseline.detach().cpu()),
                    },
                    sort_keys=True,
                )
                + "\n"
            )

        completed = step + 1
        if completed % config.checkpoint_interval == 0 or completed == config.max_steps:
            _save_checkpoint(
                checkpoint_path,
                _checkpoint_payload(
                    step=completed,
                    config_digest=config_digest,
                    source_digest=source_digest,
                    vrfm=vrfm,
                    posterior=posterior,
                    deterministic=deterministic,
                    vrfm_optimizer=vrfm_optimizer,
                    deterministic_optimizer=deterministic_optimizer,
                ),
            )
            _atomic_json(
                state_path,
                {
                    "schema": "variational_camera_latent.training_state.v1",
                    "completed_step": completed,
                    "max_steps": config.max_steps,
                    "config_digest": config_digest,
                    "source_digest": source_digest,
                    "checkpoint_sha256": _sha256_file(checkpoint_path),
                    "git_commit": config.git_commit,
                },
            )
    return TrainingResult(
        start_step=start_step,
        completed_step=config.max_steps,
        checkpoint_path=checkpoint_path,
        metrics_path=metrics_path,
    )
