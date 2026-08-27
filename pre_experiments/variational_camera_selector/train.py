from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor

from .dataset import CandidateGroup, SelectorTrainingDataset
from .loss import listwise_quality_loss
from .model import CandidateRanker


FROZEN_TRAIN_SCENES = (
    "scene0000_00",
    "scene0013_02",
    "scene0029_01",
    "scene0691_00",
    "scene0084_01",
    "scene0121_01",
    "scene0207_01",
    "scene0280_00",
)
FROZEN_VALIDATION_SCENES = ("scene0325_01", "scene0675_00")
_CHECKPOINT_SCHEMA = "variational_camera_selector.training_checkpoint.v1"
_TRAINING_STATE_SCHEMA = "variational_camera_selector.training_state.v1"


@dataclass(frozen=True)
class SelectorTrainConfig:
    prediction_manifest: Path
    privileged_manifest: Path
    run_root: Path
    max_steps: int
    batch_size: int = 1
    learning_rate: float = 1e-4
    tau: float = 0.05
    seed: int = 20260827
    d_model: int = 128
    device: str = "cuda"
    checkpoint_interval: int = 50
    git_commit: str = "unknown"
    train_scenes: tuple[str, ...] = FROZEN_TRAIN_SCENES


@dataclass(frozen=True)
class SelectorTrainingResult:
    start_step: int
    completed_step: int
    checkpoint_path: Path
    metrics_path: Path
    training_state_path: Path
    config_digest: str
    input_digest: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _atomic_json(path: Path, payload: object) -> None:
    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _atomic_torch_save(path: Path, payload: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _validate_config(config: SelectorTrainConfig) -> None:
    if config.max_steps < 1 or config.batch_size < 1 or config.checkpoint_interval < 1:
        raise ValueError("training steps, batch size, and checkpoint interval must be positive")
    if (
        not np.isfinite(config.learning_rate)
        or config.learning_rate <= 0.0
        or not np.isfinite(config.tau)
        or config.tau <= 0.0
        or config.d_model < 1
    ):
        raise ValueError("learning rate, tau, and model width must be positive and finite")
    if not config.train_scenes or not set(config.train_scenes).issubset(FROZEN_TRAIN_SCENES):
        raise ValueError("train_scenes must be a non-empty ordered subset of the frozen train split")
    if len(set(config.train_scenes)) != len(config.train_scenes):
        raise ValueError("train_scenes must be unique")
    if set(config.train_scenes) & set(FROZEN_VALIDATION_SCENES):
        raise ValueError("validation scenes may not enter selector training")
    if config.git_commit != "unknown" and (
        len(config.git_commit) != 40
        or any(character not in "0123456789abcdef" for character in config.git_commit)
    ):
        raise ValueError("git_commit must be unknown or a lowercase 40-character SHA")


def _immutable_config_payload(config: SelectorTrainConfig) -> dict[str, object]:
    return {
        "prediction_manifest": str(Path(config.prediction_manifest).resolve()),
        "privileged_manifest": str(Path(config.privileged_manifest).resolve()),
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "tau": config.tau,
        "seed": config.seed,
        "d_model": config.d_model,
        "device": config.device,
        "checkpoint_interval": config.checkpoint_interval,
        "git_commit": config.git_commit,
        "train_scenes": list(config.train_scenes),
    }


def _input_payload(config: SelectorTrainConfig) -> dict[str, object]:
    return {
        "prediction_manifest_sha256": _sha256_file(config.prediction_manifest),
        "privileged_manifest_sha256": _sha256_file(config.privileged_manifest),
        "train_scenes": list(config.train_scenes),
    }


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _capture_rng_state() -> dict[str, object]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng_state(state: dict[str, object]) -> None:
    try:
        random.setstate(state["python"])  # type: ignore[arg-type]
        np.random.set_state(state["numpy"])  # type: ignore[arg-type]
        torch.set_rng_state(state["torch"])  # type: ignore[arg-type]
        cuda_state = state["cuda"]
        if torch.cuda.is_available() and isinstance(cuda_state, list):
            torch.cuda.set_rng_state_all(cuda_state)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("checkpoint RNG state is invalid") from error


def _collate(
    groups: Sequence[CandidateGroup], device: torch.device
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    if not groups or any(group.utilities is None for group in groups):
        raise ValueError("training batches require at least one fully joined candidate group")
    try:
        global_tokens = torch.from_numpy(
            np.stack([group.global_tokens for group in groups]).astype(np.float32)
        ).to(device)
        x0 = torch.from_numpy(
            np.stack([group.x0 for group in groups]).astype(np.float32)
        ).to(device)
        delta = torch.from_numpy(
            np.stack([group.delta_tokens for group in groups]).astype(np.float32)
        ).to(device)
        alphas = torch.from_numpy(
            np.stack([group.alphas for group in groups]).astype(np.float32)
        ).to(device)
        spans = torch.as_tensor(
            [group.span_start for group in groups], dtype=torch.long, device=device
        )
        z = torch.from_numpy(np.stack([group.z for group in groups]).astype(np.float32)).to(
            device
        )
        utilities = torch.from_numpy(
            np.stack([group.utilities for group in groups]).astype(np.float32)  # type: ignore[arg-type]
        ).to(device)
    except (ValueError, TypeError) as error:
        raise ValueError("candidate groups cannot be collated into matching tensors") from error
    return global_tokens, x0, delta, alphas, spans, z, utilities


def _finite_gradients(model: torch.nn.Module) -> bool:
    return all(
        parameter.grad is None or torch.isfinite(parameter.grad).all().item()
        for parameter in model.parameters()
    )


def _metrics_text(rows: Sequence[dict[str, object]]) -> str:
    return "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)


def _checkpoint_payload(
    *,
    completed_step: int,
    full_model: CandidateRanker,
    residual_model: CandidateRanker,
    full_optimizer: torch.optim.Optimizer,
    residual_optimizer: torch.optim.Optimizer,
    config_digest: str,
    input_digest: str,
    model_config: dict[str, object],
    metrics_rows: Sequence[dict[str, object]],
) -> dict[str, object]:
    return {
        "schema": _CHECKPOINT_SCHEMA,
        "completed_step": completed_step,
        "full_context_model": full_model.state_dict(),
        "residual_only_model": residual_model.state_dict(),
        "full_context_optimizer": full_optimizer.state_dict(),
        "residual_only_optimizer": residual_optimizer.state_dict(),
        "rng_state": _capture_rng_state(),
        "config_digest": config_digest,
        "input_digest": input_digest,
        "model_config": model_config,
        "metrics_rows": list(metrics_rows),
    }


def _save_training_state(
    path: Path,
    *,
    completed_step: int,
    config: SelectorTrainConfig,
    config_digest: str,
    input_digest: str,
    checkpoint_path: Path,
) -> None:
    _atomic_json(
        path,
        {
            "schema": _TRAINING_STATE_SCHEMA,
            "completed_step": completed_step,
            "requested_max_steps": config.max_steps,
            "config_digest": config_digest,
            "input_digest": input_digest,
            "latest_checkpoint": str(checkpoint_path),
            "git_commit": config.git_commit,
            "train_scenes": list(config.train_scenes),
        },
    )


def train_selectors(config: SelectorTrainConfig) -> SelectorTrainingResult:
    """Train matched full-context and residual-only listwise selectors."""
    _validate_config(config)
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA selector training was requested but CUDA is unavailable")
    _seed_all(config.seed)
    dataset = SelectorTrainingDataset(
        config.prediction_manifest, config.privileged_manifest, roles=("train",)
    )
    if tuple(dataset.scenes) != tuple(config.train_scenes):
        raise ValueError("training dataset scenes do not match the frozen requested split")
    if set(dataset.scenes) & set(FROZEN_VALIDATION_SCENES):
        raise ValueError("validation scenes were constructed by the training dataset")
    if len(dataset) < 1:
        raise ValueError("selector training dataset is empty")
    first_group = dataset[0]
    if first_group.utilities is None or first_group.z.ndim != 2:
        raise ValueError("selector training dataset returned an invalid first group")
    z_dim = int(first_group.z.shape[1])
    model_config: dict[str, object] = {
        "d_model": config.d_model,
        "z_dim": z_dim,
        "input_dim": int(first_group.global_tokens.shape[-1]),
        "span_count": 8,
    }
    full_model = CandidateRanker(
        **model_config, include_global_context=True  # type: ignore[arg-type]
    ).to(device)
    residual_model = CandidateRanker(
        **model_config, include_global_context=False  # type: ignore[arg-type]
    ).to(device)
    full_optimizer = torch.optim.AdamW(
        full_model.parameters(), lr=config.learning_rate
    )
    residual_optimizer = torch.optim.AdamW(
        residual_model.parameters(), lr=config.learning_rate
    )

    config_digest = _sha256_json(_immutable_config_payload(config))
    input_digest = _sha256_json(_input_payload(config))
    training_root = Path(config.run_root) / "training"
    checkpoint_root = training_root / "checkpoints"
    checkpoint_path = checkpoint_root / "latest.pt"
    metrics_path = training_root / "metrics.jsonl"
    training_state_path = training_root / "training_state.json"
    start_step = 0
    metrics_rows: list[dict[str, object]] = []

    if checkpoint_path.is_file():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if (
            not isinstance(checkpoint, dict)
            or checkpoint.get("schema") != _CHECKPOINT_SCHEMA
            or checkpoint.get("config_digest") != config_digest
            or checkpoint.get("input_digest") != input_digest
            or checkpoint.get("model_config") != model_config
        ):
            raise ValueError("checkpoint config or input digest does not match")
        try:
            start_step = int(checkpoint["completed_step"])
            full_model.load_state_dict(checkpoint["full_context_model"])
            residual_model.load_state_dict(checkpoint["residual_only_model"])
            full_optimizer.load_state_dict(checkpoint["full_context_optimizer"])
            residual_optimizer.load_state_dict(checkpoint["residual_only_optimizer"])
            rows = checkpoint["metrics_rows"]
            if not isinstance(rows, list) or len(rows) != start_step:
                raise ValueError("checkpoint metrics do not match its completed step")
            metrics_rows = rows
            _restore_rng_state(checkpoint["rng_state"])
        except (KeyError, TypeError, RuntimeError, ValueError) as error:
            if isinstance(error, ValueError) and "config or input digest" in str(error):
                raise
            raise ValueError("checkpoint training state is invalid") from error
        if config.max_steps < start_step:
            raise ValueError("max_steps may not move behind the completed checkpoint")
        _atomic_text(metrics_path, _metrics_text(metrics_rows))
    elif metrics_path.exists() or training_state_path.exists():
        raise ValueError("training outputs exist without a valid resumable checkpoint")

    order = np.random.default_rng(config.seed).permutation(len(dataset)).tolist()
    completed_step = start_step
    for zero_based_step in range(start_step, config.max_steps):
        batch_indices = [
            order[(zero_based_step * config.batch_size + offset) % len(order)]
            for offset in range(config.batch_size)
        ]
        groups = [first_group if index == 0 else dataset[index] for index in batch_indices]
        global_tokens, x0, delta, alphas, spans, z, utilities = _collate(groups, device)

        full_optimizer.zero_grad(set_to_none=True)
        full_scores = full_model(global_tokens, x0, delta, alphas, spans, z)
        full_loss = listwise_quality_loss(full_scores, utilities, tau=config.tau)
        assert isinstance(full_loss, Tensor)
        full_loss.backward()
        if not _finite_gradients(full_model):
            raise ValueError("full-context selector produced non-finite gradients")
        full_grad_norm = torch.nn.utils.clip_grad_norm_(full_model.parameters(), 10.0)
        if not torch.isfinite(full_grad_norm):
            raise ValueError("full-context selector gradient norm is non-finite")
        full_optimizer.step()

        residual_optimizer.zero_grad(set_to_none=True)
        residual_scores = residual_model(global_tokens, x0, delta, alphas, spans, z)
        residual_loss = listwise_quality_loss(residual_scores, utilities, tau=config.tau)
        assert isinstance(residual_loss, Tensor)
        residual_loss.backward()
        if not _finite_gradients(residual_model):
            raise ValueError("residual-only selector produced non-finite gradients")
        residual_grad_norm = torch.nn.utils.clip_grad_norm_(residual_model.parameters(), 10.0)
        if not torch.isfinite(residual_grad_norm):
            raise ValueError("residual-only selector gradient norm is non-finite")
        residual_optimizer.step()

        completed_step = zero_based_step + 1
        metrics_rows.append(
            {
                "step": completed_step,
                "sample_ids": [group.sample_id for group in groups],
                "full_context_loss": float(full_loss.detach().cpu()),
                "residual_only_loss": float(residual_loss.detach().cpu()),
                "full_context_grad_norm": float(full_grad_norm.detach().cpu()),
                "residual_only_grad_norm": float(residual_grad_norm.detach().cpu()),
            }
        )
        _atomic_text(metrics_path, _metrics_text(metrics_rows))
        should_checkpoint = (
            completed_step % config.checkpoint_interval == 0
            or completed_step == config.max_steps
        )
        if should_checkpoint:
            payload = _checkpoint_payload(
                completed_step=completed_step,
                full_model=full_model,
                residual_model=residual_model,
                full_optimizer=full_optimizer,
                residual_optimizer=residual_optimizer,
                config_digest=config_digest,
                input_digest=input_digest,
                model_config=model_config,
                metrics_rows=metrics_rows,
            )
            periodic_path = checkpoint_root / f"step_{completed_step:08d}.pt"
            _atomic_torch_save(periodic_path, payload)
            _atomic_torch_save(checkpoint_path, payload)
            _save_training_state(
                training_state_path,
                completed_step=completed_step,
                config=config,
                config_digest=config_digest,
                input_digest=input_digest,
                checkpoint_path=checkpoint_path,
            )

    return SelectorTrainingResult(
        start_step=start_step,
        completed_step=completed_step,
        checkpoint_path=checkpoint_path,
        metrics_path=metrics_path,
        training_state_path=training_state_path,
        config_digest=config_digest,
        input_digest=input_digest,
    )
