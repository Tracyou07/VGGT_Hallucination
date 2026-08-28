"""Differentiable, fail-closed lifting of short teachers into Camera-token residuals."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import pickle
from typing import Any, Mapping

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from pre_experiments.camera_velocity_ambiguity_02.frozen_oracle import FrozenOracle
from pre_experiments.conditional_hierarchical_vrfm.basis import (
    expand_residual,
    temporal_dct_basis,
)
from pre_experiments.long_short_camera_head.geometry import apply_sim3_torch
from pre_experiments.variational_camera_latent.camera import pose_encoding_to_c2w


_CHECKPOINT_SCHEMA = "conditional_hierarchical_vrfm.lift.v1"
_DIGEST_LENGTH = 64


@dataclass(frozen=True)
class LiftConfig:
    rank: int = 32
    global_rank: int = 4
    max_steps: int = 250
    learning_rate: float = 5e-3
    teacher_translation: float = 1.0
    relative_translation: float = 0.5
    rotation: float = 0.1
    uncovered_anchor: float = 0.2
    smoothness: float = 0.05
    residual_norm: float = 1e-4
    gradient_clip: float = 1.0


@dataclass(frozen=True)
class LiftResult:
    coefficients: Tensor
    decoded_c2w_raw: Tensor
    initial_loss: float
    final_loss: float
    completed_steps: int
    finite: bool
    loss_trace: tuple[float, ...] = field(default_factory=tuple)


def _validate_config(config: LiftConfig) -> None:
    if config.rank != 32 or not 0 <= config.global_rank <= config.rank:
        raise ValueError("lift requires fixed rank=32 and global_rank in [0, 32]")
    if isinstance(config.max_steps, bool) or config.max_steps < 1:
        raise ValueError("max_steps must be positive")
    values = (
        config.learning_rate, config.teacher_translation, config.relative_translation,
        config.rotation, config.uncovered_anchor, config.smoothness,
        config.residual_norm, config.gradient_clip,
    )
    if not all(isinstance(value, (float, int)) and float(value) >= 0.0 for value in values):
        raise ValueError("lift configuration weights must be nonnegative")
    if config.learning_rate <= 0.0 or config.gradient_clip <= 0.0:
        raise ValueError("learning_rate and gradient_clip must be positive")


def _config_digest(config: LiftConfig) -> str:
    """Bind optimizer semantics; max_steps is an execution limit and may increase on resume."""
    payload = asdict(config)
    del payload["max_steps"]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _valid_digest(value: str, name: str) -> str:
    if not isinstance(value, str) or len(value) != _DIGEST_LENGTH or any(ch not in "0123456789abcdef" for ch in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _as_batched_pose(value: Tensor, name: str) -> Tensor:
    if not isinstance(value, Tensor):
        raise ValueError(f"{name} must be a tensor")
    if value.ndim == 3:
        value = value.unsqueeze(0)
    if value.ndim != 4 or value.shape[0] != 1 or value.shape[1:] != (500, 4, 4):
        raise ValueError(f"{name} must have shape [1,500,4,4] or [500,4,4]")
    return value


def _validate_homogeneous(c2w: Tensor, name: str) -> None:
    if not torch.isfinite(c2w).all():
        raise ValueError(f"{name} contains non-finite poses")
    expected = torch.tensor((0.0, 0.0, 0.0, 1.0), dtype=c2w.dtype, device=c2w.device)
    if not torch.allclose(c2w[..., 3, :], expected, atol=1e-5, rtol=0.0):
        raise ValueError(f"{name} contains non-homogeneous poses")


def _decode_trace(camera_head: nn.Module, tokens: Tensor) -> Tensor:
    if not isinstance(camera_head, nn.Module):
        raise ValueError("camera_head must be an nn.Module")
    if tokens.shape != (1, 500, 2048) or not torch.isfinite(tokens).all():
        raise ValueError("Camera tokens must be finite with shape [1,500,2048]")
    amp = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if tokens.device.type == "cuda"
        else nullcontext()
    )
    with amp:
        trace = camera_head.decode_pose_tokens(tokens, num_iterations=4)
    if not isinstance(trace, (tuple, list)) or not trace:
        raise ValueError("Camera Head returned a malformed decode trace")
    raw = trace[-1]
    if not isinstance(raw, Tensor) or raw.shape != (1, 500, 9) or not torch.isfinite(raw).all():
        raise ValueError("Camera Head produced non-finite or malformed pose encoding")
    # The explicit float conversion keeps all loss accumulation in float32 while retaining autograd.
    decoded = pose_encoding_to_c2w(raw.float())
    _validate_homogeneous(decoded, "decoded Camera Head output")
    return decoded


def decode_coefficients(camera_head: nn.Module, long_tokens: Tensor, coefficients: Tensor) -> Tensor:
    """Expand one native residual and decode it directly through the Camera Head graph."""
    if not isinstance(long_tokens, Tensor) or long_tokens.shape != (1, 500, 2048):
        raise ValueError("long_tokens must have shape [1,500,2048]")
    if not isinstance(coefficients, Tensor) or coefficients.shape != (1, 32, 2048):
        raise ValueError("coefficients must have shape [1,32,2048]")
    if not torch.isfinite(long_tokens).all() or not torch.isfinite(coefficients).all():
        raise ValueError("long_tokens and coefficients must be finite")
    basis = temporal_dct_basis(device=long_tokens.device, dtype=torch.float32)
    residual = expand_residual(coefficients.float(), basis)
    return _decode_trace(camera_head, long_tokens.float() + residual)


def _oracle_tensors(oracle: FrozenOracle, device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
    if not isinstance(oracle, FrozenOracle):
        raise ValueError("oracle must be a FrozenOracle")
    values = (oracle.scale, *sum((tuple(row) for row in oracle.rotation), ()), *oracle.translation)
    if not all(torch.isfinite(torch.tensor(value)).item() for value in values):
        raise ValueError("oracle must be finite")
    return (
        torch.tensor(oracle.scale, dtype=torch.float32, device=device),
        torch.tensor(oracle.rotation, dtype=torch.float32, device=device),
        torch.tensor(oracle.translation, dtype=torch.float32, device=device),
    )


def _zero_like_loss(reference: Tensor) -> Tensor:
    return reference.sum() * 0.0


def latent_lift_loss(
    *,
    corrected_c2w_raw: Tensor,
    baseline_c2w_raw: Tensor,
    teacher_c2w_gt_gauge: Tensor,
    coverage_weight: Tensor,
    oracle: FrozenOracle,
    residual: Tensor,
    config: LiftConfig,
) -> dict[str, Tensor]:
    """Compute the fixed-gauge loss, never arithmetically touching uncovered NaN teachers."""
    _validate_config(config)
    corrected = _as_batched_pose(corrected_c2w_raw, "corrected_c2w_raw").float()
    baseline = _as_batched_pose(baseline_c2w_raw, "baseline_c2w_raw").to(
        device=corrected.device, dtype=torch.float32
    )
    teacher = _as_batched_pose(teacher_c2w_gt_gauge, "teacher_c2w_gt_gauge").to(
        device=corrected.device, dtype=torch.float32
    )
    _validate_homogeneous(corrected, "corrected_c2w_raw")
    _validate_homogeneous(baseline, "baseline_c2w_raw")
    if not isinstance(coverage_weight, Tensor):
        raise ValueError("coverage_weight must be a tensor")
    coverage = coverage_weight.to(device=corrected.device, dtype=torch.float32).reshape(-1)
    if coverage.shape != (500,) or not torch.isfinite(coverage).all() or torch.any(coverage < 0.0):
        raise ValueError("coverage_weight must be finite, nonnegative, and shape [500]")
    if residual.shape != (1, 500, 2048) or not torch.isfinite(residual).all():
        raise ValueError("residual must be finite with shape [1,500,2048]")

    scale, rotation, translation = _oracle_tensors(oracle, corrected.device)
    corrected_gt = apply_sim3_torch(corrected, scale=scale, rotation=rotation, translation=translation)
    baseline_gt = apply_sim3_torch(baseline, scale=scale, rotation=rotation, translation=translation)
    teacher_finite = torch.isfinite(teacher).all(dim=(-1, -2))[0]
    covered = (coverage > 0.0) & teacher_finite
    uncovered = ~covered
    center = corrected_gt[..., :3, 3]
    baseline_center = baseline_gt[..., :3, 3]

    if torch.any(covered):
        predicted_centers = center[0, covered]
        teacher_centers = teacher[0, covered, :3, 3]
        teacher_center_loss = F.smooth_l1_loss(predicted_centers, teacher_centers)
        covered_rotation_loss = torch.mean(
            (corrected_gt[0, covered, :3, :3] - teacher[0, covered, :3, :3]).square()
        )
    else:
        teacher_center_loss = _zero_like_loss(center)
        covered_rotation_loss = _zero_like_loss(center)

    relative_terms: list[Tensor] = []
    for lag in (1, 5, 10, 25):
        pair = covered[lag:] & covered[:-lag]
        if torch.any(pair):
            predicted_delta = center[0, lag:][pair] - center[0, :-lag][pair]
            teacher_delta = teacher[0, lag:, :3, 3][pair] - teacher[0, :-lag, :3, 3][pair]
            relative_terms.append(F.smooth_l1_loss(predicted_delta, teacher_delta))
    relative_motion_loss = torch.stack(relative_terms).mean() if relative_terms else _zero_like_loss(center)
    uncovered_center_anchor = (
        F.smooth_l1_loss(center[0, uncovered], baseline_center[0, uncovered])
        if torch.any(uncovered) else _zero_like_loss(center)
    )
    second_difference_loss = torch.mean((center[:, 2:] - 2.0 * center[:, 1:-1] + center[:, :-2]).square())
    residual_norm_loss = residual.square().mean()
    total = (
        config.teacher_translation * teacher_center_loss
        + config.relative_translation * relative_motion_loss
        + config.rotation * covered_rotation_loss
        + config.uncovered_anchor * uncovered_center_anchor
        + config.smoothness * second_difference_loss
        + config.residual_norm * residual_norm_loss
    )
    return {
        "teacher_center_loss": teacher_center_loss,
        "relative_motion_loss": relative_motion_loss,
        "covered_rotation_loss": covered_rotation_loss,
        "uncovered_center_anchor": uncovered_center_anchor,
        "second_difference_loss": second_difference_loss,
        "residual_norm_loss": residual_norm_loss,
        "total": total,
    }


def _checkpoint_payload_valid(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, Mapping) or payload.get("schema") != _CHECKPOINT_SCHEMA:
        raise ValueError("invalid lift checkpoint schema")
    required = {
        "schema", "coefficients", "optimizer", "variant_index", "next_step", "config_digest",
        "source_sha256", "teacher_sha256", "rng_state", "loss_trace", "initial_loss",
    }
    if set(payload) != required:
        raise ValueError("invalid lift checkpoint structure")
    coefficients = payload["coefficients"]
    if not isinstance(coefficients, Tensor) or coefficients.shape != (1, 32, 2048) or coefficients.dtype != torch.float32:
        raise ValueError("invalid lift checkpoint coefficients")
    if not torch.isfinite(coefficients).all() or not isinstance(payload["optimizer"], Mapping):
        raise ValueError("invalid lift checkpoint state")
    if any(isinstance(payload[name], bool) or not isinstance(payload[name], int) or payload[name] < 0 for name in ("variant_index", "next_step")):
        raise ValueError("invalid lift checkpoint step")
    for name in ("config_digest", "source_sha256", "teacher_sha256"):
        _valid_digest(payload[name], name)
    if not isinstance(payload["rng_state"], Tensor) or payload["rng_state"].dtype != torch.uint8:
        raise ValueError("invalid lift checkpoint RNG state")
    trace = payload["loss_trace"]
    if not isinstance(trace, (list, tuple)) or not all(isinstance(value, (float, int)) and torch.isfinite(torch.tensor(value)).item() for value in trace):
        raise ValueError("invalid lift checkpoint loss trace")
    if not isinstance(payload["initial_loss"], (float, int)) or not torch.isfinite(torch.tensor(payload["initial_loss"])).item():
        raise ValueError("invalid lift checkpoint initial loss")
    return dict(payload)


def save_lift_checkpoint(
    path: Path,
    *,
    coefficients: Tensor,
    optimizer: torch.optim.Optimizer,
    variant_index: int,
    next_step: int,
    config_digest: str,
    source_sha256: str,
    teacher_sha256: str,
    rng_state: Tensor,
    loss_trace: tuple[float, ...] = (),
    initial_loss: float = 0.0,
) -> None:
    """Atomically save trusted internal optimizer state, then validate it on load."""
    _valid_digest(config_digest, "config_digest")
    _valid_digest(source_sha256, "source_sha256")
    _valid_digest(teacher_sha256, "teacher_sha256")
    payload = {
        "schema": _CHECKPOINT_SCHEMA,
        "coefficients": coefficients.detach().to(device="cpu", dtype=torch.float32).clone(),
        "optimizer": optimizer.state_dict(),
        "variant_index": variant_index,
        "next_step": next_step,
        "config_digest": config_digest,
        "source_sha256": source_sha256,
        "teacher_sha256": teacher_sha256,
        "rng_state": rng_state.detach().to(device="cpu", dtype=torch.uint8).clone(),
        "loss_trace": [float(value) for value in loss_trace],
        "initial_loss": float(initial_loss),
    }
    _checkpoint_payload_valid(payload)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_lift_checkpoint(path: Path) -> dict[str, Any]:
    """Load only an internal trusted checkpoint and validate its complete structure."""
    try:
        try:
            payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        except TypeError:  # PyTorch versions before weights_only accepted this trusted load.
            payload = torch.load(Path(path), map_location="cpu")
    except (OSError, RuntimeError, EOFError, ValueError, pickle.UnpicklingError) as error:
        raise ValueError("invalid lift checkpoint") from error
    try:
        return _checkpoint_payload_valid(payload)
    except ValueError as error:
        raise ValueError("invalid lift checkpoint") from error


def _snapshot_head(camera_head: nn.Module) -> tuple[list[tuple[Tensor, Tensor]], list[tuple[nn.Module, bool]], list[tuple[Tensor, bool]]]:
    if not isinstance(camera_head, nn.Module):
        raise ValueError("camera_head must be an nn.Module")
    tensors = [(tensor, tensor.detach().clone()) for tensor in [*camera_head.parameters(), *camera_head.buffers()]]
    modes = [(module, module.training) for module in camera_head.modules()]
    requires_grad = [(parameter, parameter.requires_grad) for parameter in camera_head.parameters()]
    return tensors, modes, requires_grad


def _restore_head(snapshot: tuple[list[tuple[Tensor, Tensor]], list[tuple[nn.Module, bool]], list[tuple[Tensor, bool]]]) -> bool:
    tensors, modes, requires_grad = snapshot
    parameter_count = len(requires_grad)
    changed_parameter = any(not torch.equal(tensor, value) for tensor, value in tensors[:parameter_count])
    with torch.no_grad():
        for tensor, value in tensors:
            tensor.copy_(value)
    for module, training in modes:
        module.training = training
    for parameter, required in requires_grad:
        parameter.requires_grad_(required)
    return changed_parameter


def _residual(coefficients: Tensor, device: torch.device) -> Tensor:
    return expand_residual(coefficients, temporal_dct_basis(device=device, dtype=torch.float32))


def optimize_latent_target(
    camera_head: nn.Module,
    long_tokens: Tensor,
    teacher_c2w_gt_gauge: Tensor,
    oracle: FrozenOracle,
    config: LiftConfig = LiftConfig(),
    *,
    coverage_weight: Tensor | None = None,
    variant_index: int = 0,
    checkpoint_path: Path | None = None,
    resume: bool = False,
    source_sha256: str = "0" * 64,
    teacher_sha256: str = "0" * 64,
) -> LiftResult:
    """Optimize a single variant; callers process four variants sequentially."""
    _validate_config(config)
    _valid_digest(source_sha256, "source_sha256")
    _valid_digest(teacher_sha256, "teacher_sha256")
    if isinstance(variant_index, bool) or not isinstance(variant_index, int) or not 0 <= variant_index < 4:
        raise ValueError("variant_index must lie in [0, 3]")
    if not isinstance(long_tokens, Tensor) or long_tokens.shape != (1, 500, 2048):
        raise ValueError("long_tokens must have shape [1,500,2048]")
    if coverage_weight is None:
        coverage_weight = torch.isfinite(_as_batched_pose(teacher_c2w_gt_gauge, "teacher_c2w_gt_gauge")).all(dim=(-1, -2))[0].float()
    device = long_tokens.device
    config_digest = _config_digest(config)
    coefficients = nn.Parameter(torch.zeros((1, 32, 2048), dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW([coefficients], lr=config.learning_rate, weight_decay=0.0)
    next_step = 0
    loss_trace: list[float] = []
    initial_loss: float | None = None
    if resume:
        if checkpoint_path is None:
            raise ValueError("resume requires checkpoint_path")
        payload = load_lift_checkpoint(checkpoint_path)
        if payload["config_digest"] != config_digest:
            raise ValueError("checkpoint config digest does not match")
        if payload["source_sha256"] != source_sha256:
            raise ValueError("checkpoint source digest does not match")
        if payload["teacher_sha256"] != teacher_sha256:
            raise ValueError("checkpoint teacher digest does not match")
        if payload["variant_index"] != variant_index:
            raise ValueError("checkpoint variant does not match")
        next_step = payload["next_step"]
        if next_step > config.max_steps:
            raise ValueError("checkpoint next step exceeds max_steps")
        with torch.no_grad():
            coefficients.copy_(payload["coefficients"].to(device=device))
        optimizer.load_state_dict(payload["optimizer"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, Tensor):
                    state[key] = value.to(device=device)
        torch.set_rng_state(payload["rng_state"])
        loss_trace = [float(value) for value in payload["loss_trace"]]
        initial_loss = float(payload["initial_loss"])

    snapshot = _snapshot_head(camera_head)
    try:
        camera_head.eval()
        for parameter in camera_head.parameters():
            parameter.requires_grad_(False)
        with torch.no_grad():
            baseline = decode_coefficients(camera_head, long_tokens, torch.zeros_like(coefficients))
        if _restore_head(snapshot):
            raise ValueError("Camera Head parameters changed during frozen decode")
        snapshot = _snapshot_head(camera_head)
        camera_head.eval()
        for parameter in camera_head.parameters():
            parameter.requires_grad_(False)
        if initial_loss is None:
            with torch.no_grad():
                baseline_residual = _residual(torch.zeros_like(coefficients), device)
                initial_loss = float(latent_lift_loss(
                    corrected_c2w_raw=baseline, baseline_c2w_raw=baseline,
                    teacher_c2w_gt_gauge=teacher_c2w_gt_gauge, coverage_weight=coverage_weight,
                    oracle=oracle, residual=baseline_residual, config=config,
                )["total"].cpu())

        best_coefficients = coefficients.detach().clone()
        best_loss = float("inf")
        best_decoded: Tensor | None = None
        for step in range(next_step, config.max_steps):
            optimizer.zero_grad(set_to_none=True)
            decoded = decode_coefficients(camera_head, long_tokens, coefficients)
            evaluated_coefficients = coefficients.detach().clone()
            residual = _residual(coefficients, device)
            losses = latent_lift_loss(
                corrected_c2w_raw=decoded, baseline_c2w_raw=baseline,
                teacher_c2w_gt_gauge=teacher_c2w_gt_gauge, coverage_weight=coverage_weight,
                oracle=oracle, residual=residual, config=config,
            )
            total = losses["total"].float()
            if not torch.isfinite(total):
                raise ValueError("non-finite latent lift loss")
            total.backward()
            if coefficients.grad is None or not torch.isfinite(coefficients.grad).all():
                raise ValueError("non-finite latent lift gradient")
            if _restore_head(snapshot):
                raise ValueError("Camera Head parameters changed during frozen decode")
            snapshot = _snapshot_head(camera_head)
            camera_head.eval()
            for parameter in camera_head.parameters():
                parameter.requires_grad_(False)
            torch.nn.utils.clip_grad_norm_([coefficients], config.gradient_clip)
            optimizer.step()
            current_loss = float(total.detach().cpu())
            loss_trace.append(current_loss)
            if current_loss < best_loss:
                best_loss = current_loss
                best_coefficients = evaluated_coefficients
                best_decoded = decoded.detach().clone()
            if checkpoint_path is not None:
                save_lift_checkpoint(
                    checkpoint_path, coefficients=coefficients, optimizer=optimizer,
                    variant_index=variant_index, next_step=step + 1, config_digest=config_digest,
                    source_sha256=source_sha256, teacher_sha256=teacher_sha256,
                    rng_state=torch.get_rng_state(), loss_trace=tuple(loss_trace), initial_loss=initial_loss,
                )
        if best_decoded is None or not torch.isfinite(torch.tensor(best_loss)) or best_loss >= initial_loss:
            raise ValueError("latent lift did not reduce the initial loss")
        with torch.no_grad():
            coefficients.copy_(best_coefficients)
            final_decoded = decode_coefficients(camera_head, long_tokens, coefficients).detach().clone()
        if _restore_head(snapshot):
            raise ValueError("Camera Head parameters changed during frozen decode")
        _validate_homogeneous(final_decoded, "best decoded Camera Head output")
        return LiftResult(
            coefficients=best_coefficients.detach().clone(), decoded_c2w_raw=final_decoded,
            initial_loss=initial_loss, final_loss=best_loss, completed_steps=config.max_steps,
            finite=True, loss_trace=tuple(loss_trace),
        )
    except (RuntimeError, ValueError):
        _restore_head(snapshot)
        raise
    finally:
        _restore_head(snapshot)
