from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
from torch import Tensor
from torch import nn
from vggt.heads.camera_head import CameraHead

from pre_experiments.common.model_io import find_checkpoint
from pre_experiments.variational_camera_latent.camera import pose_encoding_to_c2w

from .geometry import apply_sim3_torch
from .data import load_long_context
from .labels import load_privileged_labels
from .losses import LossWeights, TrainingLabels, camera_head_losses

try:
    from safetensors import safe_open
except ImportError:  # pragma: no cover - exercised only on hosts without safetensors.
    safe_open = None


CHECKPOINT_SCHEMA = "long_short_camera_head.training_checkpoint.v1"


@dataclass(frozen=True)
class ResumeState:
    step: int
    best_step: int
    best_validation_rms: float


@dataclass(frozen=True)
class TrainingExample:
    scene: str
    tokens: Tensor
    baseline_pose: Tensor
    gt_c2w: Tensor
    oracle_scale: Tensor
    oracle_rotation: Tensor
    oracle_translation: Tensor
    gt_scene_scale: Tensor
    teacher_c2w: Tensor
    teacher_weight: Tensor

    def to(self, device: torch.device) -> "TrainingExample":
        return TrainingExample(
            scene=self.scene,
            tokens=self.tokens.to(device),
            baseline_pose=self.baseline_pose.to(device),
            gt_c2w=self.gt_c2w.to(device),
            oracle_scale=self.oracle_scale.to(device),
            oracle_rotation=self.oracle_rotation.to(device),
            oracle_translation=self.oracle_translation.to(device),
            gt_scene_scale=self.gt_scene_scale.to(device),
            teacher_c2w=self.teacher_c2w.to(device),
            teacher_weight=self.teacher_weight.to(device),
        )

    def labels(self) -> TrainingLabels:
        return TrainingLabels(
            gt_c2w=self.gt_c2w,
            oracle_scale=self.oracle_scale,
            oracle_rotation=self.oracle_rotation,
            oracle_translation=self.oracle_translation,
            gt_scene_scale=self.gt_scene_scale,
            teacher_c2w_gt_gauge=self.teacher_c2w,
            teacher_weight=self.teacher_weight,
        )


@dataclass(frozen=True)
class TrainingResult:
    start_step: int
    completed_step: int
    initial_training_loss: float
    final_training_loss: float
    best_validation_rms: float
    best_step: int
    best_checkpoint: Path
    latest_checkpoint: Path
    metrics_path: Path
    validation_metrics_path: Path


@dataclass(frozen=True)
class TrainConfig:
    checkpoint_dir: Path
    run_root: Path
    variant: str
    train_pairs: tuple[tuple[Path, Path], ...]
    validation_pairs: tuple[tuple[Path, Path], ...]
    max_steps: int = 400
    learning_rate: float = 2e-6
    weight_decay: float = 1e-4
    checkpoint_interval: int = 25
    patience: int = 100
    seed: int = 20260828
    device: torch.device = torch.device("cuda")
    weights: LossWeights = LossWeights()


def load_training_example(
    long_context_path: Path,
    privileged_path: Path,
) -> TrainingExample:
    """Join prediction-only input and privileged labels by immutable identity."""
    long_context = load_long_context(long_context_path)
    privileged = load_privileged_labels(privileged_path)
    long_scene = str(long_context["scene"])
    privileged_scene = str(privileged["scene"])
    if long_scene != privileged_scene:
        raise ValueError("long context and privileged labels have different scenes")
    if not np.array_equal(long_context["frame_ids"], privileged["frame_ids"]):
        raise ValueError("long context and privileged labels have different frame IDs")
    if str(long_context["source_sha256"]) != str(privileged["source_sha256"]):
        raise ValueError("long context and privileged labels have different source identity")
    return TrainingExample(
        scene=long_scene,
        tokens=torch.from_numpy(long_context["camera_tokens"]).unsqueeze(0),
        baseline_pose=torch.from_numpy(privileged["baseline_pose_encoding"]).unsqueeze(0),
        gt_c2w=torch.from_numpy(privileged["gt_c2w"]).unsqueeze(0),
        oracle_scale=torch.from_numpy(privileged["oracle_scale"]),
        oracle_rotation=torch.from_numpy(privileged["oracle_rotation"]),
        oracle_translation=torch.from_numpy(privileged["oracle_translation"]),
        gt_scene_scale=torch.from_numpy(privileged["gt_scene_scale"]),
        teacher_c2w=torch.from_numpy(privileged["teacher_c2w_gt_gauge"]).unsqueeze(0),
        teacher_weight=torch.from_numpy(privileged["teacher_weight"]).unsqueeze(0),
    )


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest_json(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_base_camera_head(checkpoint_dir: Path) -> tuple[CameraHead, str]:
    """Load only native Camera Head weights from a local VGGT checkpoint."""
    checkpoint = find_checkpoint(Path(checkpoint_dir))
    prefix = "camera_head."
    if checkpoint.suffix == ".safetensors":
        if safe_open is None:
            raise RuntimeError("safetensors is required for the local VGGT checkpoint")
        state: dict[str, Tensor] = {}
        with safe_open(str(checkpoint), framework="pt", device="cpu") as archive:
            for name in archive.keys():
                if name.startswith(prefix):
                    state[name[len(prefix) :]] = archive.get_tensor(name)
    else:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict):
            raise ValueError("local VGGT checkpoint must contain a state dictionary")
        state = {
            name[len(prefix) :]: value
            for name, value in payload.items()
            if isinstance(name, str) and name.startswith(prefix)
        }
    if not state:
        raise ValueError("local VGGT checkpoint contains no Camera Head weights")
    head = CameraHead()
    head.load_state_dict(state, strict=True)
    return head, _hash_file(checkpoint)


def load_camera_head_checkpoint(
    path: Path,
    checkpoint_dir: Path,
    device: torch.device,
) -> nn.Module:
    """Reconstruct the native Camera Head and apply a fine-tuned state."""
    head, _ = load_base_camera_head(checkpoint_dir)
    try:
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, ValueError) as error:
        raise ValueError(f"invalid Camera Head checkpoint: {path}") from error
    if not isinstance(payload, dict) or payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("Camera Head checkpoint schema mismatch")
    head.load_state_dict(payload["model"], strict=True)
    return head.to(device).eval()


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def train_camera_head(config: TrainConfig) -> TrainingResult:
    """Run one matched Camera Head variant from strict long/privileged pairs."""
    if not config.train_pairs or not config.validation_pairs:
        raise ValueError("training and validation pairs are required")
    train_examples = tuple(load_training_example(*pair) for pair in config.train_pairs)
    validation_examples = tuple(
        load_training_example(*pair) for pair in config.validation_pairs
    )
    train_scenes = tuple(example.scene for example in train_examples)
    validation_scenes = tuple(example.scene for example in validation_examples)
    if len(set(train_scenes)) != len(train_scenes) or len(set(validation_scenes)) != len(
        validation_scenes
    ):
        raise ValueError("scene lists must not contain duplicates")
    if set(train_scenes) & set(validation_scenes):
        raise ValueError("training and validation scenes must be disjoint")

    model, base_checkpoint_sha256 = load_base_camera_head(config.checkpoint_dir)
    for _, privileged_path in (*config.train_pairs, *config.validation_pairs):
        labels = load_privileged_labels(privileged_path)
        if str(labels["checkpoint_sha256"]) != base_checkpoint_sha256:
            raise ValueError("privileged labels use a different base checkpoint")
    trainable_names = configure_trainable_scope(model)
    weights_payload = {
        name: float(getattr(config.weights, name))
        for name in ("gt_translation", "relative_translation", "rotation", "anchor", "teacher")
    }
    config_payload = {
        "schema": "long_short_camera_head.train_config.v1",
        "variant": config.variant,
        "max_steps": config.max_steps,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "checkpoint_interval": config.checkpoint_interval,
        "patience": config.patience,
        "seed": config.seed,
        "precision": "bf16_autocast" if config.device.type == "cuda" else "float32",
        "weights": weights_payload,
        "train_scenes": list(train_scenes),
        "validation_scenes": list(validation_scenes),
        "base_checkpoint_sha256": base_checkpoint_sha256,
    }
    config_digest = _digest_json(config_payload)
    file_rows = [
        {
            "kind": kind,
            "scene": example.scene,
            "long_sha256": _hash_file(pair[0]),
            "privileged_sha256": _hash_file(pair[1]),
        }
        for kind, pairs, examples in (
            ("train", config.train_pairs, train_examples),
            ("validation", config.validation_pairs, validation_examples),
        )
        for pair, example in zip(pairs, examples)
    ]
    data_digest = _digest_json(file_rows)
    provenance = {
        **config_payload,
        "config_digest": config_digest,
        "data_digest": data_digest,
        "device": str(config.device),
        "trainable_parameter_names": list(trainable_names),
        "trainable_parameter_count": int(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        ),
        "files": file_rows,
    }
    provenance_path = Path(config.run_root) / "training_provenance.json"
    if provenance_path.is_file():
        try:
            existing = json.loads(provenance_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("existing training provenance is invalid") from error
        if existing != provenance:
            raise ValueError("existing training provenance does not match this run")
    else:
        _write_json_atomic(provenance_path, provenance)
    return run_training_loop(
        model=model,
        train_examples=train_examples,
        validation_examples=validation_examples,
        run_root=config.run_root,
        variant=config.variant,
        max_steps=config.max_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        checkpoint_interval=config.checkpoint_interval,
        patience=config.patience,
        seed=config.seed,
        device=config.device,
        config_digest=config_digest,
        data_digest=data_digest,
        weights=config.weights,
    )


def configure_trainable_scope(camera_head: nn.Module) -> tuple[str, ...]:
    """Freeze the head except its final native transformer block and decoder."""
    if not hasattr(camera_head, "trunk") or len(camera_head.trunk) < 1:
        raise ValueError("Camera Head must expose a non-empty trunk")
    for parameter in camera_head.parameters():
        parameter.requires_grad_(False)
    modules = (
        camera_head.trunk[-1],
        camera_head.trunk_norm,
        camera_head.embed_pose,
        camera_head.poseLN_modulation,
        camera_head.pose_branch,
    )
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    camera_head.empty_pose_tokens.requires_grad_(True)
    return tuple(
        sorted(name for name, parameter in camera_head.named_parameters() if parameter.requires_grad)
    )


def save_training_checkpoint(
    path: Path,
    *,
    step: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config_digest: str,
    data_digest: str,
    best_validation_rms: float,
    best_step: int | None = None,
) -> None:
    if step < 0 or len(config_digest) != 64 or len(data_digest) != 64:
        raise ValueError("checkpoint step or digest is invalid")
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "step": int(step),
        "best_step": int(step if best_step is None else best_step),
        "best_validation_rms": float(best_validation_rms),
        "config_digest": config_digest,
        "data_digest": data_digest,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_training_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config_digest: str,
    data_digest: str,
    device: torch.device,
) -> ResumeState:
    try:
        payload = torch.load(Path(path), map_location=device, weights_only=False)
    except (OSError, RuntimeError, ValueError) as error:
        raise ValueError(f"invalid training checkpoint: {path}") from error
    if not isinstance(payload, dict) or payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("training checkpoint schema mismatch")
    if payload.get("config_digest") != config_digest or payload.get("data_digest") != data_digest:
        raise ValueError("training checkpoint digest mismatch")
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    torch.set_rng_state(payload["torch_rng_state"])
    return ResumeState(
        step=int(payload["step"]),
        best_step=int(payload.get("best_step", payload["step"])),
        best_validation_rms=float(payload["best_validation_rms"]),
    )


def _decode_final(
    camera_head: nn.Module,
    tokens: Tensor,
    *,
    autocast_enabled: bool | None = None,
) -> Tensor:
    enabled = tokens.device.type == "cuda" if autocast_enabled is None else autocast_enabled
    device_type = "cuda" if enabled else tokens.device.type
    with torch.autocast(
        device_type=device_type,
        dtype=torch.bfloat16,
        enabled=enabled,
    ):
        trace = camera_head.decode_pose_tokens(tokens, num_iterations=4)
    if not isinstance(trace, (list, tuple)) or len(trace) != 4:
        raise ValueError("Camera Head must return exactly four decode iterations")
    output = trace[-1]
    del trace
    if output.shape != (*tokens.shape[:2], 9) or not torch.isfinite(output).all():
        raise ValueError("Camera Head produced malformed or non-finite pose encoding")
    return output


def _teacher_coefficient(variant: str) -> float:
    if variant == "gt_only":
        return 0.0
    if variant == "long_short":
        return 1.0
    raise ValueError("variant must be gt_only or long_short")


@torch.no_grad()
def _validation_rms(model: nn.Module, examples: tuple[TrainingExample, ...]) -> float:
    model.eval()
    values: list[float] = []
    for example in examples:
        pose = _decode_final(model, example.tokens)
        c2w = pose_encoding_to_c2w(pose.float())
        aligned = apply_sim3_torch(
            c2w,
            scale=example.oracle_scale.float(),
            rotation=example.oracle_rotation.float(),
            translation=example.oracle_translation.float(),
        )
        differences = aligned[..., :3, 3] - example.gt_c2w.float()[..., :3, 3]
        values.append(float(torch.sqrt(torch.mean(torch.sum(differences * differences, dim=-1)))))
    return float(np.mean(values))


def _example_loss(
    model: nn.Module,
    example: TrainingExample,
    *,
    teacher_coefficient: float,
    weights: LossWeights,
) -> dict[str, Tensor]:
    pose = _decode_final(model, example.tokens)
    return camera_head_losses(
        pose,
        example.baseline_pose,
        example.labels(),
        teacher_coefficient=teacher_coefficient,
        weights=weights,
    )


def run_training_loop(
    *,
    model: nn.Module,
    train_examples: tuple[TrainingExample, ...],
    validation_examples: tuple[TrainingExample, ...],
    run_root: Path,
    variant: str,
    max_steps: int,
    learning_rate: float,
    weight_decay: float,
    checkpoint_interval: int,
    patience: int,
    seed: int,
    device: torch.device,
    config_digest: str,
    data_digest: str,
    weights: LossWeights = LossWeights(),
) -> TrainingResult:
    if not train_examples or not validation_examples:
        raise ValueError("training and validation examples are required")
    if (
        max_steps < 1
        or checkpoint_interval < 1
        or patience < checkpoint_interval
        or learning_rate <= 0.0
        or weight_decay < 0.0
    ):
        raise ValueError("training schedule is invalid")
    if len(config_digest) != 64 or len(data_digest) != 64:
        raise ValueError("training digests must be SHA-256 strings")
    teacher_coefficient = _teacher_coefficient(variant)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = model.to(device)
    train_examples = tuple(example.to(device) for example in train_examples)
    validation_examples = tuple(example.to(device) for example in validation_examples)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("training model has no trainable parameters")
    optimizer = torch.optim.AdamW(
        parameters,
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    run_root = Path(run_root)
    latest_checkpoint = run_root / "checkpoints" / "latest.pt"
    best_checkpoint = run_root / "checkpoints" / "best.pt"
    metrics_path = run_root / "metrics" / "training.jsonl"
    validation_metrics_path = run_root / "metrics" / "validation.jsonl"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    start_step = 0
    best_validation = float("inf")
    best_step = 0
    if latest_checkpoint.is_file():
        resumed = load_training_checkpoint(
            latest_checkpoint,
            model=model,
            optimizer=optimizer,
            config_digest=config_digest,
            data_digest=data_digest,
            device=device,
        )
        start_step = resumed.step
        best_validation = resumed.best_validation_rms
        best_step = resumed.best_step
    if start_step > max_steps:
        raise ValueError("checkpoint step exceeds requested maximum")

    model.eval()
    with torch.no_grad():
        initial_loss = float(
            _example_loss(
                model,
                train_examples[0],
                teacher_coefficient=teacher_coefficient,
                weights=weights,
            )["total"]
        )
    final_loss = initial_loss
    completed_step = start_step
    model.train()
    for step in range(start_step, max_steps):
        example = train_examples[step % len(train_examples)]
        optimizer.zero_grad(set_to_none=True)
        losses = _example_loss(
            model,
            example,
            teacher_coefficient=teacher_coefficient,
            weights=weights,
        )
        if not torch.isfinite(losses["total"]):
            raise ValueError("training loss became non-finite")
        losses["total"].backward()
        if any(
            parameter.grad is not None and not torch.isfinite(parameter.grad).all()
            for parameter in parameters
        ):
            raise ValueError("training gradient became non-finite")
        torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
        optimizer.step()
        completed_step = step + 1
        final_loss = float(losses["total"].detach())
        row = {
            "step": step,
            "scene": example.scene,
            "variant": variant,
            **{name: float(value.detach()) for name, value in losses.items()},
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        if completed_step % checkpoint_interval == 0 or completed_step == max_steps:
            validation = _validation_rms(model, validation_examples)
            if not math.isfinite(validation):
                raise ValueError("validation RMS became non-finite")
            improved = validation < best_validation
            if improved:
                best_validation = validation
                best_step = completed_step
            save_training_checkpoint(
                latest_checkpoint,
                step=completed_step,
                model=model,
                optimizer=optimizer,
                config_digest=config_digest,
                data_digest=data_digest,
                best_validation_rms=best_validation,
                best_step=best_step,
            )
            if improved:
                save_training_checkpoint(
                    best_checkpoint,
                    step=completed_step,
                    model=model,
                    optimizer=optimizer,
                    config_digest=config_digest,
                    data_digest=data_digest,
                    best_validation_rms=best_validation,
                    best_step=best_step,
                )
            with validation_metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "step": completed_step,
                            "variant": variant,
                            "mean_frozen_oracle_rms": validation,
                            "improved": improved,
                            "best_step": best_step,
                            "best_validation_rms": best_validation,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
            if completed_step - best_step >= patience:
                break
            model.train()
    if not best_checkpoint.is_file():
        raise ValueError("training produced no best checkpoint")
    model.eval()
    with torch.no_grad():
        final_loss = float(
            _example_loss(
                model,
                train_examples[0],
                teacher_coefficient=teacher_coefficient,
                weights=weights,
            )["total"]
        )
    if not math.isfinite(final_loss):
        raise ValueError("final training loss became non-finite")
    return TrainingResult(
        start_step=start_step,
        completed_step=completed_step,
        initial_training_loss=initial_loss,
        final_training_loss=final_loss,
        best_validation_rms=best_validation,
        best_step=best_step,
        best_checkpoint=best_checkpoint,
        latest_checkpoint=latest_checkpoint,
        metrics_path=metrics_path,
        validation_metrics_path=validation_metrics_path,
    )
