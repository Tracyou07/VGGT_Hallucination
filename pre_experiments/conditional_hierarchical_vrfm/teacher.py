"""Offline privileged short-window teachers with one immutable evaluation gauge."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from typing import Sequence

import numpy as np
import torch
from torch import nn

from pre_experiments.camera_velocity_ambiguity_02.frozen_oracle import (
    FrozenOracle,
    apply_frozen_oracle,
    evaluate_with_frozen_oracle,
    fit_frozen_oracle,
)
from pre_experiments.camera_velocity_ambiguity_02.geometry import (
    align_local_to_global,
    global_scene_scale,
)
from pre_experiments.long_short_camera_head.data import load_prepared_gt
from pre_experiments.long_short_camera_head.labels import fuse_teacher_trajectories
from pre_experiments.variational_camera_latent.camera import (
    decode_camera_tokens,
    pose_encoding_to_c2w,
)
from pre_experiments.variational_camera_latent.source import load_source_shard


@dataclass(frozen=True)
class TeacherVariantSet:
    """Four privileged fused teachers in the one frozen baseline-to-GT gauge."""

    scene: str
    frame_ids: np.ndarray
    aligned_short_c2w: np.ndarray
    window_weights: np.ndarray
    window_masks: np.ndarray
    fused_c2w: np.ndarray
    coverage_weights: np.ndarray
    oracle: FrozenOracle
    checkpoint_sha256: str
    variant_utilities: np.ndarray


_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _window_coverage(mask: np.ndarray) -> np.ndarray:
    coverage = np.zeros(500, dtype=bool)
    for index, selected in enumerate(mask):
        if selected:
            coverage[index * 50 : index * 50 + 100] = True
    return coverage


def build_variant_window_masks(scene: str, window_weights: np.ndarray) -> np.ndarray:
    """Return the all-positive mask plus three stable, distinct 75% bootstraps."""
    weights = np.asarray(window_weights, dtype=np.float64)
    if not isinstance(scene, str) or not scene:
        raise ValueError("scene must be non-empty")
    if weights.shape != (9,) or not np.isfinite(weights).all() or np.any(weights < 0.0):
        raise ValueError("window weights must be a finite nonnegative vector with shape [9]")
    positive = weights > 0.0
    if np.count_nonzero(positive) < 3:
        raise ValueError("at least three positive teacher windows are required for four variants")

    masks = [positive.copy()]
    seen = {positive.tobytes()}
    target_coverage = _window_coverage(positive)
    all_candidates: list[np.ndarray] = []
    positive_indices = np.flatnonzero(positive)
    for bits in range(1, 1 << len(positive_indices)):
        candidate = np.zeros(9, dtype=bool)
        candidate[positive_indices] = np.asarray(
            [(bits >> offset) & 1 for offset in range(len(positive_indices))], dtype=bool
        )
        all_candidates.append(candidate)
    for index in range(1, 4):
        seed = int.from_bytes(
            hashlib.sha256(f"{scene}:teacher_variant:{index}".encode("utf-8")).digest()[:8],
            byteorder="big",
            signed=False,
        )
        generator = np.random.default_rng(seed)
        candidate: np.ndarray | None = None
        for _ in range(128):
            proposal = positive & (generator.random(9) < 0.75)
            available = [value for value in all_candidates if value.tobytes() not in seen]
            preserving = [
                value for value in available
                if np.array_equal(_window_coverage(value), target_coverage)
            ]
            pool = preserving or available
            if not pool:
                break
            distances = np.asarray([np.count_nonzero(value != proposal) for value in pool])
            nearest = np.flatnonzero(distances == distances.min())
            candidate = pool[int(nearest[generator.integers(len(nearest))])].copy()
            break
        if candidate is None:
            raise ValueError("could not construct a unique deterministic teacher mask")
        masks.append(candidate)
        seen.add(candidate.tobytes())
    return np.stack(masks)


def _scene_from_source(source: dict[str, np.ndarray]) -> str:
    sample_ids = source["sample_ids"]
    scene = str(sample_ids[0]).split(":", 1)[0]
    if not scene or any(not str(value).startswith(scene + ":") for value in sample_ids):
        raise ValueError("source sample IDs must bind exactly one scene")
    return scene


def _decode(camera_head: nn.Module, tokens: np.ndarray, device: torch.device) -> np.ndarray:
    token_tensor = torch.from_numpy(np.asarray(tokens, dtype=np.float32)).to(device)
    raw = decode_camera_tokens(camera_head, token_tensor)
    c2w = pose_encoding_to_c2w(raw)
    result = c2w.detach().to(dtype=torch.float64, device="cpu").numpy()
    if not np.isfinite(result).all():
        raise ValueError("Camera Head produced non-finite camera poses")
    return result


@contextmanager
def _frozen_camera_head(camera_head: nn.Module, device: torch.device):
    """Decode in eval mode while restoring every module flag, parameter, and buffer."""
    if not isinstance(camera_head, nn.Module):
        raise ValueError("camera_head must be an nn.Module")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("requested camera-head device is unavailable")
    tensors = list(camera_head.parameters()) + list(camera_head.buffers())
    for tensor in tensors:
        if tensor.device.type != device.type or (
            device.index is not None and tensor.device.index != device.index
        ):
            raise ValueError("camera head tensors must be on the requested device")
    snapshots = [(tensor, tensor.detach().clone()) for tensor in tensors]
    modes = [(module, module.training) for module in camera_head.modules()]
    try:
        camera_head.eval()
        yield
    finally:
        with torch.no_grad():
            for tensor, value in snapshots:
                tensor.copy_(value)
        for module, training in modes:
            module.training = training


def _covered_variant_utility(
    baseline_aligned: np.ndarray,
    fused: np.ndarray,
    coverage: np.ndarray,
    raw_gt: np.ndarray,
) -> float:
    covered = np.asarray(coverage) > 0.0
    if covered.shape != (500,) or not np.any(covered):
        return 0.0
    if not np.isfinite(fused[covered]).all():
        raise ValueError("covered fused teacher poses must be finite")
    baseline_error = baseline_aligned[covered, :3, 3] - raw_gt[covered, :3, 3]
    teacher_error = fused[covered, :3, 3] - raw_gt[covered, :3, 3]
    baseline_rms = float(np.sqrt(np.mean(np.sum(baseline_error * baseline_error, axis=1))))
    teacher_rms = float(np.sqrt(np.mean(np.sum(teacher_error * teacher_error, axis=1))))
    return float((baseline_rms - teacher_rms) / max(baseline_rms, 1e-12))


def build_teacher_variants(
    source_path: Path,
    prepared_scene: Path,
    camera_head: nn.Module,
    *,
    checkpoint_sha256: str,
    device: torch.device,
    variant_count: int = 4,
) -> TeacherVariantSet:
    """Build offline-only GT-weighted short teachers without changing source bytes."""
    if variant_count != 4:
        raise ValueError("exactly four teacher variants are required")
    if not isinstance(checkpoint_sha256, str) or _SHA256_RE.fullmatch(checkpoint_sha256) is None:
        raise ValueError("checkpoint_sha256 must be a canonical lowercase SHA-256 digest")
    source = load_source_shard(Path(source_path))
    if "global_pred_c2w" not in source:
        raise ValueError("source shard lacks authenticated baseline camera poses")
    scene = _scene_from_source(source)
    frame_ids = source["global_frame_ids"].astype(np.int64, copy=True)
    with _frozen_camera_head(camera_head, device):
        decoded_baseline = _decode(camera_head, source["global_camera_tokens"][None], device)[0]
        short_c2w = _decode(camera_head, source["short_camera_tokens"], device)
    authenticated_baseline = source["global_pred_c2w"].astype(np.float64, copy=False)
    if not np.allclose(decoded_baseline, authenticated_baseline, atol=2e-4, rtol=2e-4):
        raise ValueError("frozen Camera Head does not reproduce authenticated baseline")
    baseline = decoded_baseline.copy()
    raw_gt = load_prepared_gt(Path(prepared_scene), frame_ids)

    oracle = fit_frozen_oracle(scene, frame_ids, baseline, raw_gt)
    scene_scale = global_scene_scale(baseline)
    aligned_short = np.full((9, 100, 4, 4), np.nan, dtype=np.float64)
    weights = np.zeros(9, dtype=np.float64)
    for index, start in enumerate(range(0, 401, 50)):
        stop = start + 100
        baseline_rms = evaluate_with_frozen_oracle(
            oracle, baseline[start:stop], raw_gt[start:stop]
        ).rms_translation_error
        alignment = align_local_to_global(
            baseline[start:stop], short_c2w[index], scene_scale=scene_scale
        )
        if not alignment.valid or alignment.aligned_c2w is None:
            continue
        aligned = apply_frozen_oracle(oracle, alignment.aligned_c2w)
        aligned_short[index] = aligned
        teacher_rms = evaluate_with_frozen_oracle(
            oracle, alignment.aligned_c2w, raw_gt[start:stop]
        ).rms_translation_error
        weights[index] = float(
            np.clip(
                (baseline_rms - teacher_rms) / max(baseline_rms, 1e-12),
                0.0,
                1.0,
            )
        )
    masks = build_variant_window_masks(scene, weights)
    fused: list[np.ndarray] = []
    coverage: list[np.ndarray] = []
    for mask in masks:
        windows = [
            (start, aligned_short[index], float(weights[index]))
            for index, start in enumerate(range(0, 401, 50))
            if mask[index] and weights[index] > 0.0
        ]
        trajectory, frame_weights = fuse_teacher_trajectories(frame_count=500, windows=windows)
        fused.append(trajectory)
        coverage.append(frame_weights)
    fused_array = np.stack(fused)
    coverage_array = np.stack(coverage)
    baseline_aligned = apply_frozen_oracle(oracle, baseline)
    utilities = np.asarray(
        [
            _covered_variant_utility(baseline_aligned, fused_array[index], coverage_array[index], raw_gt)
            for index in range(variant_count)
        ],
        dtype=np.float64,
    )
    return TeacherVariantSet(
        scene=scene,
        frame_ids=frame_ids,
        aligned_short_c2w=aligned_short,
        window_weights=weights,
        window_masks=masks,
        fused_c2w=fused_array,
        coverage_weights=coverage_array,
        oracle=oracle,
        checkpoint_sha256=checkpoint_sha256,
        variant_utilities=utilities,
    )


def summarize_teacher_upper_bound(teachers: Sequence[TeacherVariantSet]) -> dict[str, object]:
    """Summarize variant-zero coverage and GT-derived window utility offline only."""
    if not teachers:
        raise ValueError("at least one teacher set is required")
    coverage = np.asarray(
        [np.mean(teacher.coverage_weights[0] > 0.0) for teacher in teachers], dtype=np.float64
    )
    utility = np.asarray([teacher.variant_utilities[0] for teacher in teachers], dtype=np.float64)
    if not np.isfinite(utility).all():
        raise ValueError("teacher variant utilities must be finite")
    return {
        "scene_count": len(teachers),
        "positive_scene_count": int(np.count_nonzero(utility > 0.0)),
        "mean_coverage": float(np.mean(coverage)),
        "mean_utility": float(np.mean(utility)),
    }
