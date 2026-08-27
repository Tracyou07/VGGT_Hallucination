from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from pre_experiments.camera_velocity_ambiguity_02.frozen_oracle import (
    evaluate_with_frozen_oracle,
    fit_frozen_oracle,
)

from .camera import decode_camera_tokens, pose_encoding_to_c2w
from .privileged import _load_prepared_gt
from .source import load_source_shard


DEFAULT_ALPHAS = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)
_SIDES = np.asarray(("left", "right"), dtype="U8")
_PREDICTION_MEMBERS = {
    "alphas",
    "sides",
    "source_sample_ids",
    "overlap_frame_ids",
    "span_starts",
    "decode_context_frames",
    "decoded_camera_raw",
    "decoded_camera_c2w",
    "endpoint_delta_rms",
    "checkpoint_sha256",
    "source_shard_sha256",
}
_PRIVILEGED_MEMBERS = {
    "alphas",
    "sides",
    "sample_ids",
    "gt_frame_ids",
    "gt_c2w",
    "baseline_rms",
    "candidate_rms",
    "relative_improvement",
    "best_side_index",
    "best_alpha_index",
    "best_alpha",
    "best_candidate_rms",
    "best_relative_improvement",
    "oracle_scale",
    "oracle_rotation",
    "oracle_translation",
    "oracle_digest",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_digest(value: str, name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _normalize_alphas(values: Sequence[float] | np.ndarray) -> np.ndarray:
    alphas = np.asarray(values, dtype=np.float64)
    if (
        alphas.ndim != 1
        or len(alphas) < 2
        or not np.isfinite(alphas).all()
        or alphas[0] != 0.0
        or alphas[-1] != 1.0
        or np.any(alphas[1:] <= alphas[:-1])
        or np.any((alphas < 0.0) | (alphas > 1.0))
    ):
        raise ValueError("alphas must be strictly increasing from exactly 0 to 1")
    return alphas


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _load_npz(path: Path, label: str) -> dict[str, np.ndarray]:
    try:
        with np.load(Path(path), allow_pickle=False) as archive:
            return {name: np.asarray(archive[name]).copy() for name in archive.files}
    except (OSError, ValueError, KeyError) as error:
        raise ValueError(f"invalid {label}: {path}") from error


def _validate_prediction(arrays: dict[str, np.ndarray]) -> None:
    if set(arrays) != _PREDICTION_MEMBERS:
        raise ValueError("alpha-scan prediction members do not match the schema")
    if any(value.dtype.hasobject for value in arrays.values()):
        raise ValueError("alpha-scan prediction may not contain object arrays")
    alphas = _normalize_alphas(arrays["alphas"])
    count = len(alphas)
    expected = {
        "sides": (2,),
        "source_sample_ids": (8,),
        "overlap_frame_ids": (8, 50),
        "span_starts": (8,),
        "decode_context_frames": (),
        "decoded_camera_raw": (8, 2, count, 50, 9),
        "decoded_camera_c2w": (8, 2, count, 50, 4, 4),
        "endpoint_delta_rms": (8, 2),
        "checkpoint_sha256": (),
        "source_shard_sha256": (),
    }
    for name, shape in expected.items():
        if arrays[name].shape != shape:
            raise ValueError(f"alpha-scan prediction member {name} has invalid shape")
    if not np.array_equal(arrays["sides"], _SIDES):
        raise ValueError("alpha-scan sides must be left and right")
    if arrays["source_sample_ids"].dtype.kind != "U":
        raise ValueError("alpha-scan sample IDs must be Unicode")
    if (
        not np.issubdtype(arrays["decode_context_frames"].dtype, np.integer)
        or int(arrays["decode_context_frames"]) != 500
    ):
        raise ValueError("alpha-scan candidates must be decoded in the full 500-frame context")
    if arrays["checkpoint_sha256"].dtype.kind != "U" or arrays[
        "source_shard_sha256"
    ].dtype.kind != "U":
        raise ValueError("alpha-scan digests must be Unicode")
    _validate_digest(str(arrays["checkpoint_sha256"]), "checkpoint_sha256")
    _validate_digest(str(arrays["source_shard_sha256"]), "source_shard_sha256")
    for name in ("decoded_camera_raw", "decoded_camera_c2w", "endpoint_delta_rms"):
        if not np.issubdtype(arrays[name].dtype, np.floating) or not np.isfinite(
            arrays[name]
        ).all():
            raise ValueError(f"alpha-scan prediction member {name} must be finite")
    if not np.allclose(
        arrays["decoded_camera_c2w"][..., 3, :],
        [0.0, 0.0, 0.0, 1.0],
        atol=1e-8,
        rtol=0,
    ):
        raise ValueError("alpha-scan decoded cameras must be homogeneous")


def load_alpha_scan_candidates(path: Path) -> dict[str, np.ndarray]:
    arrays = _load_npz(path, "alpha-scan prediction")
    _validate_prediction(arrays)
    return arrays


def generate_alpha_scan_candidates(
    source_path: Path,
    destination: Path,
    *,
    camera_head: Any,
    checkpoint_sha256: str,
    device: str = "cuda",
    alphas: Sequence[float] = DEFAULT_ALPHAS,
) -> Path:
    """Decode straight latent paths without accepting GT or quality labels."""
    _validate_digest(checkpoint_sha256, "checkpoint_sha256")
    alpha_values = _normalize_alphas(alphas)
    source = load_source_shard(source_path)
    if "overlap_long_c2w" not in source:
        raise ValueError("alpha scan requires source long-window camera predictions")
    device_value = torch.device(device)
    full_tokens = torch.from_numpy(source["global_camera_tokens"]).to(device_value)
    long_tokens = torch.from_numpy(source["overlap_long_tokens"]).to(device_value)
    endpoints = (
        torch.from_numpy(source["overlap_left_tokens"]).to(device_value),
        torch.from_numpy(source["overlap_right_tokens"]).to(device_value),
    )
    raw = np.empty((8, 2, len(alpha_values), 50, 9), dtype=np.float32)
    c2w = np.empty((8, 2, len(alpha_values), 50, 4, 4), dtype=np.float64)
    with torch.inference_mode():
        for side_index, endpoint in enumerate(endpoints):
            for alpha_index, alpha in enumerate(alpha_values):
                mixed = torch.lerp(long_tokens, endpoint, float(alpha))
                sequences = full_tokens[None].expand(8, -1, -1).clone()
                for overlap, span_start in enumerate(source["span_starts"]):
                    overlap_start = int(span_start) + 50
                    sequences[overlap, overlap_start : overlap_start + 50] = mixed[
                        overlap
                    ]
                decoded_full = decode_camera_tokens(camera_head, sequences)
                decoded = torch.stack(
                    [
                        decoded_full[
                            overlap,
                            int(span_start) + 50 : int(span_start) + 100,
                        ]
                        for overlap, span_start in enumerate(source["span_starts"])
                    ]
                )
                raw[:, side_index, alpha_index] = decoded.float().cpu().numpy()
                c2w[:, side_index, alpha_index] = (
                    pose_encoding_to_c2w(decoded).double().cpu().numpy()
                )
    c2w[:, :, 0] = source["overlap_long_c2w"][:, None]
    delta = np.stack(
        (
            source["overlap_left_tokens"] - source["overlap_long_tokens"],
            source["overlap_right_tokens"] - source["overlap_long_tokens"],
        ),
        axis=1,
    ).astype(np.float64)
    arrays = {
        "alphas": alpha_values,
        "sides": _SIDES.copy(),
        "source_sample_ids": source["sample_ids"].copy(),
        "overlap_frame_ids": source["overlap_frame_ids"].copy(),
        "span_starts": source["span_starts"].copy(),
        "decode_context_frames": np.asarray(500, dtype=np.int64),
        "decoded_camera_raw": raw,
        "decoded_camera_c2w": c2w,
        "endpoint_delta_rms": np.sqrt(np.mean(delta * delta, axis=(2, 3))),
        "checkpoint_sha256": np.asarray(checkpoint_sha256, dtype="U64"),
        "source_shard_sha256": np.asarray(_sha256_file(source_path), dtype="U64"),
    }
    _validate_prediction(arrays)
    _atomic_npz(destination, arrays)
    return Path(destination)


def _validate_privileged(arrays: dict[str, np.ndarray]) -> None:
    if set(arrays) != _PRIVILEGED_MEMBERS:
        raise ValueError("alpha-scan privileged members do not match the schema")
    if any(value.dtype.hasobject for value in arrays.values()):
        raise ValueError("alpha-scan privileged sidecar may not contain object arrays")
    alphas = _normalize_alphas(arrays["alphas"])
    count = len(alphas)
    expected = {
        "sides": (2,),
        "sample_ids": (8,),
        "gt_frame_ids": (8, 50),
        "gt_c2w": (8, 50, 4, 4),
        "baseline_rms": (8,),
        "candidate_rms": (8, 2, count),
        "relative_improvement": (8, 2, count),
        "best_side_index": (8,),
        "best_alpha_index": (8,),
        "best_alpha": (8,),
        "best_candidate_rms": (8,),
        "best_relative_improvement": (8,),
        "oracle_scale": (),
        "oracle_rotation": (3, 3),
        "oracle_translation": (3,),
        "oracle_digest": (),
    }
    for name, shape in expected.items():
        if arrays[name].shape != shape:
            raise ValueError(f"alpha-scan privileged member {name} has invalid shape")
    if not np.array_equal(arrays["sides"], _SIDES):
        raise ValueError("alpha-scan privileged sides must be left and right")
    if arrays["sample_ids"].dtype.kind != "U" or arrays["oracle_digest"].dtype.kind != "U":
        raise ValueError("alpha-scan privileged identities must be Unicode")
    for name, value in arrays.items():
        if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
            raise ValueError(f"alpha-scan privileged member {name} contains non-finite values")


def load_alpha_scan_privileged(path: Path) -> dict[str, np.ndarray]:
    arrays = _load_npz(path, "alpha-scan privileged sidecar")
    _validate_privileged(arrays)
    return arrays


def write_alpha_scan_privileged_sidecar(
    source_path: Path,
    candidate_path: Path,
    prepared_scene_root: Path,
    destination: Path,
) -> Path:
    source = load_source_shard(source_path)
    if "global_pred_c2w" not in source or "overlap_long_c2w" not in source:
        raise ValueError("alpha scan requires source prediction poses")
    candidate = load_alpha_scan_candidates(candidate_path)
    if not np.array_equal(candidate["source_sample_ids"], source["sample_ids"]):
        raise ValueError("alpha-scan candidate and source sample IDs do not match")
    if str(candidate["source_shard_sha256"]) != _sha256_file(source_path):
        raise ValueError("alpha-scan candidate source digest does not match")
    gt_ids, gt_c2w = _load_prepared_gt(prepared_scene_root)
    if not np.array_equal(gt_ids, source["global_frame_ids"]):
        raise ValueError("prepared GT and source frame IDs do not match")
    scene = str(source["sample_ids"][0]).split(":", 1)[0]
    oracle = fit_frozen_oracle(
        scene,
        source["global_frame_ids"],
        source["global_pred_c2w"],
        gt_c2w,
    )
    alphas = candidate["alphas"]
    gt_overlap = np.empty((8, 50, 4, 4), dtype=np.float64)
    baseline_rms = np.empty(8, dtype=np.float64)
    candidate_rms = np.empty((8, 2, len(alphas)), dtype=np.float64)
    for overlap, frame_ids in enumerate(source["overlap_frame_ids"]):
        indices = np.searchsorted(gt_ids, frame_ids)
        if not np.array_equal(gt_ids[indices], frame_ids):
            raise ValueError("overlap frame IDs are absent from prepared GT")
        gt_overlap[overlap] = gt_c2w[indices]
        baseline_rms[overlap] = evaluate_with_frozen_oracle(
            oracle,
            source["overlap_long_c2w"][overlap],
            gt_overlap[overlap],
        ).rms_translation_error
        for side in range(2):
            candidate_rms[overlap, side, 0] = baseline_rms[overlap]
            for alpha_index in range(1, len(alphas)):
                candidate_rms[overlap, side, alpha_index] = evaluate_with_frozen_oracle(
                    oracle,
                    candidate["decoded_camera_c2w"][overlap, side, alpha_index],
                    gt_overlap[overlap],
                ).rms_translation_error
    denominator = np.maximum(baseline_rms[:, None, None], np.finfo(np.float64).eps)
    relative = (baseline_rms[:, None, None] - candidate_rms) / denominator
    relative[:, :, 0] = 0.0
    flattened = candidate_rms.reshape(8, -1)
    best_flat = np.argmin(flattened, axis=1).astype(np.int64)
    best_side = best_flat // len(alphas)
    best_alpha_index = best_flat % len(alphas)
    rows = np.arange(8)
    arrays = {
        "alphas": alphas.copy(),
        "sides": candidate["sides"].copy(),
        "sample_ids": source["sample_ids"].copy(),
        "gt_frame_ids": source["overlap_frame_ids"].copy(),
        "gt_c2w": gt_overlap,
        "baseline_rms": baseline_rms,
        "candidate_rms": candidate_rms,
        "relative_improvement": relative,
        "best_side_index": best_side,
        "best_alpha_index": best_alpha_index,
        "best_alpha": alphas[best_alpha_index],
        "best_candidate_rms": candidate_rms[rows, best_side, best_alpha_index],
        "best_relative_improvement": relative[rows, best_side, best_alpha_index],
        "oracle_scale": np.asarray(oracle.scale, dtype=np.float64),
        "oracle_rotation": np.asarray(oracle.rotation, dtype=np.float64),
        "oracle_translation": np.asarray(oracle.translation, dtype=np.float64),
        "oracle_digest": np.asarray(oracle.transform_digest, dtype="U64"),
    }
    _validate_privileged(arrays)
    _atomic_npz(destination, arrays)
    return Path(destination)


def summarize_alpha_statistics(
    alphas: Sequence[float] | np.ndarray,
    relative_improvement: np.ndarray,
    *,
    min_improvement: float = 0.01,
) -> dict[str, object]:
    alpha_values = _normalize_alphas(alphas)
    relative = np.asarray(relative_improvement, dtype=np.float64)
    if (
        relative.ndim != 3
        or relative.shape[1:] != (2, len(alpha_values))
        or len(relative) < 1
        or not np.isfinite(relative).all()
    ):
        raise ValueError("relative improvement must have shape [overlaps, 2, alphas]")
    if not np.isfinite(min_improvement) or min_improvement <= 0.0:
        raise ValueError("min_improvement must be finite and positive")
    nonzero = relative[:, :, 1:]
    flat = nonzero.reshape(len(relative), -1)
    best_flat = np.argmax(flat, axis=1)
    best_side = best_flat // (len(alpha_values) - 1)
    best_alpha_index = best_flat % (len(alpha_values) - 1) + 1
    rows = np.arange(len(relative))
    best_improvement = relative[rows, best_side, best_alpha_index]
    best_alpha = alpha_values[best_alpha_index]
    useful = best_improvement > min_improvement
    same_side_full = relative[rows, best_side, -1]
    small_step = best_alpha < 1.0
    rescued = useful & small_step & (same_side_full <= 0.0)
    useful_count = int(np.count_nonzero(useful))
    rescued_count = int(np.count_nonzero(rescued))
    if useful_count > len(relative) / 2:
        diagnosis = (
            "DIRECTION_USEFUL_STEP_TOO_LARGE"
            if rescued_count > useful_count / 2
            else "DIRECTION_USEFUL"
        )
    elif useful_count > 0:
        diagnosis = "SPARSE_OR_MIXED_DIRECTION"
    else:
        diagnosis = "DIRECTION_NOT_USEFUL"
    alpha_counts = {
        f"{value:g}": int(np.count_nonzero(useful & (best_alpha == value)))
        for value in alpha_values[1:]
    }
    return {
        "diagnosis": diagnosis,
        "overlap_count": int(len(relative)),
        "min_improvement": float(min_improvement),
        "useful_direction_count": useful_count,
        "useful_direction_fraction": float(useful_count / len(relative)),
        "small_step_best_count": int(np.count_nonzero(useful & small_step)),
        "small_step_rescue_count": rescued_count,
        "full_step_useful_count": int(
            np.count_nonzero(np.max(relative[:, :, -1], axis=1) > min_improvement)
        ),
        "median_best_nonzero_relative_improvement": float(np.median(best_improvement)),
        "median_safe_best_relative_improvement": float(
            np.median(np.maximum(best_improvement, 0.0))
        ),
        "best_alpha_counts": alpha_counts,
        "best_side_counts": {
            "left": int(np.count_nonzero(useful & (best_side == 0))),
            "right": int(np.count_nonzero(useful & (best_side == 1))),
        },
    }


def write_alpha_scan_report(
    sidecar_paths: Sequence[Path],
    destination: Path,
    *,
    min_improvement: float = 0.01,
) -> dict[str, object]:
    if not sidecar_paths:
        raise ValueError("at least one alpha-scan sidecar is required")
    loaded = [load_alpha_scan_privileged(path) for path in sidecar_paths]
    alphas = loaded[0]["alphas"]
    if any(not np.array_equal(arrays["alphas"], alphas) for arrays in loaded[1:]):
        raise ValueError("alpha-scan sidecars use different alpha grids")
    relative = np.concatenate([arrays["relative_improvement"] for arrays in loaded], axis=0)
    report = {
        "schema": "variational_camera_latent.alpha_scan_report.v1",
        "scene_count": len(loaded),
        "alphas": alphas.tolist(),
        **summarize_alpha_statistics(
            alphas,
            relative,
            min_improvement=min_improvement,
        ),
    }
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return report
