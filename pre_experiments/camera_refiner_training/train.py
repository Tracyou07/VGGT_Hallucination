"""Train the translation-only camera residual refiner."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random

import numpy as np

import torch

from pre_experiments.camera_refiner_training.checkpoint import (
    load_checkpoint,
    run_config_digest,
    save_checkpoint,
)
from pre_experiments.camera_refiner_training.data import (
    DatasetManifest,
    SceneWindows,
    build_scene_windows,
    load_dataset_manifest,
    load_translation_units,
)
from pre_experiments.camera_refiner_training.diffusion import (
    DiffusionSchedule,
    q_sample,
)
from pre_experiments.camera_refiner_training.losses import LossWeights, training_losses
from pre_experiments.camera_refiner_training.model import ModelConfig, ResidualDiT
from pre_experiments.camera_refiner_training.visualize import write_history_plot
from pre_experiments.camera_refiner_training.io import atomic_write_json


def fit_condition_stats(scenes: list[SceneWindows]) -> tuple[torch.Tensor, torch.Tensor]:
    if not scenes:
        raise ValueError("at least one training scene is required")
    values = torch.from_numpy(
        np.concatenate([scene.condition.reshape(-1, scene.condition.shape[-1]) for scene in scenes])
    ).float()
    return values.mean(dim=0), values.std(dim=0, unbiased=False).clamp_min(1e-6)


def make_scene_batch(
    scene: SceneWindows,
    condition_mean: torch.Tensor,
    condition_std: torch.Tensor,
    device: torch.device,
) -> dict[str, object]:
    condition = torch.from_numpy(np.asarray(scene.condition)).float()
    condition = (condition - condition_mean) / condition_std
    window_count = len(condition)
    return {
        "condition": condition.to(device),
        "target_residual": torch.from_numpy(scene.target_residual).float().to(device),
        "global_centers": torch.from_numpy(scene.global_centers).float().to(device),
        "scene_ids": tuple(scene.scene for _ in range(window_count)),
        "starts": torch.from_numpy(scene.starts).long().to(device),
    }


def _load_scenes(
    manifest: DatasetManifest,
    local_run_dir: Path,
    units: object,
    window_length: int,
    stride: int,
) -> list[SceneWindows]:
    return [
        build_scene_windows(
            entry.shard,
            Path(local_run_dir) / entry.scene,
            units,
            window_length=window_length,
            stride=stride,
        )
        for entry in manifest.entries
    ]


@torch.no_grad()
def validation_loss(
    model: torch.nn.Module,
    scenes: list[SceneWindows],
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
    *,
    schedule: DiffusionSchedule,
    model_kind: str,
    seed: int,
) -> float:
    model.eval()
    values = []
    generator = torch.Generator(device=device.type).manual_seed(seed)
    for scene in scenes:
        batch = make_scene_batch(scene, mean, std, device)
        target = batch["target_residual"]
        if model_kind == "diffusion":
            timestep = torch.full(
                (len(target),), schedule.steps // 2, dtype=torch.long, device=device
            )
            noise = torch.randn(
                target.shape,
                dtype=target.dtype,
                device=device,
                generator=generator,
            )
            model_input = q_sample(target, timestep, noise, schedule)
        elif model_kind == "deterministic":
            timestep = torch.zeros(len(target), dtype=torch.long, device=device)
            model_input = torch.zeros_like(target)
        else:
            raise ValueError("model_kind must be diffusion or deterministic")
        prediction, confidence = model(model_input, batch["condition"], timestep)
        values.append(float(torch.mean((prediction * confidence - target) ** 2).cpu()))
    return float(np.mean(values))


def train_batch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, object],
    schedule: DiffusionSchedule,
    *,
    model_kind: str,
    generator: torch.Generator,
    lags: tuple[int, ...] = (1, 5, 10, 25),
    weights: LossWeights = LossWeights(),
) -> dict[str, torch.Tensor]:
    target = batch["target_residual"]
    condition = batch["condition"]
    if not isinstance(target, torch.Tensor) or not isinstance(condition, torch.Tensor):
        raise TypeError("batch tensors are required")
    if model_kind == "diffusion":
        timestep = torch.randint(
            schedule.steps,
            (len(target),),
            generator=generator,
            device=target.device,
        )
        noise = torch.randn(
            target.shape,
            dtype=target.dtype,
            device=target.device,
            generator=generator,
        )
        model_input = q_sample(target, timestep, noise, schedule)
    elif model_kind == "deterministic":
        timestep = torch.zeros(len(target), dtype=torch.long, device=target.device)
        model_input = torch.zeros_like(target)
    else:
        raise ValueError("model_kind must be diffusion or deterministic")
    predicted, confidence = model(model_input, condition, timestep)
    losses = training_losses(
        predicted,
        confidence,
        target,
        batch["global_centers"],
        scene_ids=batch["scene_ids"],
        starts=batch["starts"],
        weights=weights,
        lags=lags,
    )
    optimizer.zero_grad(set_to_none=True)
    losses["total"].backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return {name: value.detach() for name, value in losses.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--local-run-dir", type=Path, required=True)
    parser.add_argument("--frozen-units", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model-kind", choices=("diffusion", "deterministic"), default="diffusion")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--unit-count", type=int, default=41)
    parser.add_argument("--iteration", type=int, default=0)
    parser.add_argument("--window-length", type=int, default=100)
    parser.add_argument("--stride", type=int, default=50)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--diffusion-steps", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.learning_rate <= 0:
        parser.error("epochs and learning-rate must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    units = load_translation_units(
        args.frozen_units, count=args.unit_count, iteration=args.iteration
    )
    train_manifest = load_dataset_manifest(
        args.dataset_manifest, args.dataset_root, roles={"train"}
    )
    validation_manifest = load_dataset_manifest(
        args.dataset_manifest, args.dataset_root, roles={"validation"}
    )
    train_scenes = _load_scenes(
        train_manifest, args.local_run_dir, units, args.window_length, args.stride
    )
    validation_scenes = _load_scenes(
        validation_manifest, args.local_run_dir, units, args.window_length, args.stride
    )
    condition_mean, condition_std = fit_condition_stats(train_scenes)
    model_config = ModelConfig(
        condition_dim=int(train_scenes[0].condition.shape[-1]),
        hidden_size=args.hidden_size,
        depth=args.depth,
        num_heads=args.num_heads,
        max_frames=args.window_length,
    )
    run_config = {
        "dataset_digest": train_manifest.digest,
        "unit_digest": units.digest,
        "unit_count": args.unit_count,
        "iteration": args.iteration,
        "model_kind": args.model_kind,
        "model": model_config.to_dict(),
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "window_length": args.window_length,
        "stride": args.stride,
        "diffusion_steps": args.diffusion_steps,
        "train_scenes": [scene.scene for scene in train_scenes],
        "validation_scenes": [scene.scene for scene in validation_scenes],
    }
    model = ResidualDiT(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    schedule = DiffusionSchedule.cosine(args.diffusion_steps)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    last_path = args.out_dir / "last.pt"
    start_epoch = 0
    history: list[dict[str, float | int]] = []
    if args.resume:
        if not last_path.is_file():
            parser.error(f"resume checkpoint does not exist: {last_path}")
        state = load_checkpoint(
            last_path,
            model=model,
            optimizer=optimizer,
            expected_run_digest=run_config_digest(run_config),
            map_location=device,
        )
        start_epoch = state.epoch
        history_path = args.out_dir / "history.json"
        if history_path.is_file():
            history = json.loads(history_path.read_text(encoding="utf-8"))
    generator = torch.Generator(device=device.type).manual_seed(args.seed)
    best = min((float(row["validation_loss"]) for row in history), default=float("inf"))
    for epoch in range(start_epoch, args.epochs):
        model.train()
        order = torch.randperm(len(train_scenes), generator=torch.Generator().manual_seed(args.seed + epoch))
        totals = []
        for index in order.tolist():
            batch = make_scene_batch(
                train_scenes[index], condition_mean, condition_std, device
            )
            losses = train_batch(
                model,
                optimizer,
                batch,
                schedule,
                model_kind=args.model_kind,
                generator=generator,
            )
            totals.append(float(losses["total"].cpu()))
        validation = validation_loss(
            model,
            validation_scenes,
            condition_mean,
            condition_std,
            device,
            schedule=schedule,
            model_kind=args.model_kind,
            seed=args.seed,
        )
        row = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(totals)),
            "validation_loss": validation,
        }
        history.append(row)
        save_checkpoint(
            last_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch + 1,
            model_config=model_config,
            condition_mean=condition_mean,
            condition_std=condition_std,
            run_config=run_config,
        )
        if validation < best:
            best = validation
            save_checkpoint(
                args.out_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                model_config=model_config,
                condition_mean=condition_mean,
                condition_std=condition_std,
                run_config=run_config,
            )
        atomic_write_json(args.out_dir / "history.json", history)
        atomic_write_json(
            args.out_dir / "run.json",
            {"run_config": run_config, "run_digest": run_config_digest(run_config)},
        )
        write_history_plot(args.out_dir / "history.png", history)
        print(json.dumps(row, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
