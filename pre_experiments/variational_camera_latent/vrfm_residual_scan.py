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

from .alpha_scan import DEFAULT_ALPHAS
from .camera import decode_camera_tokens, pose_encoding_to_c2w
from .candidates import load_candidate_shard
from .privileged import _load_prepared_gt
from .source import load_source_shard


_PREDICTION_MEMBERS = {
    "alphas",
    "source_sample_ids",
    "overlap_frame_ids",
    "span_starts",
    "z",
    "latent_cluster_ids",
    "sample_seeds",
    "decode_context_frames",
    "camera_iterations",
    "decode_protocol",
    "decoded_camera_raw",
    "decoded_camera_c2w",
    "residual_rms",
    "vrfm_checkpoint_sha256",
    "camera_head_checkpoint_sha256",
    "source_shard_sha256",
    "candidate_shard_sha256",
    "producer_git_commit",
}
_PRIVILEGED_MEMBERS = {
    "alphas",
    "source_sample_ids",
    "sample_seeds",
    "gt_frame_ids",
    "gt_c2w",
    "baseline_rms",
    "candidate_rms",
    "relative_improvement",
    "accept_correction",
    "best_sample_index",
    "best_alpha_index",
    "best_alpha",
    "best_sample_seed",
    "best_latent_cluster_id",
    "best_candidate_rms",
    "best_relative_improvement",
    "best_nonzero_sample_index",
    "best_nonzero_alpha_index",
    "best_nonzero_alpha",
    "best_nonzero_sample_seed",
    "best_nonzero_latent_cluster_id",
    "best_nonzero_candidate_rms",
    "best_nonzero_relative_improvement",
    "prediction_sha256",
    "prepared_gt_sha256",
    "oracle_scale",
    "oracle_rotation",
    "oracle_translation",
    "oracle_digest",
}
_DECODE_PROTOCOL = "vrfm_residual_alpha_full_g500.v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_digest(value: str, name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _validate_git_commit(value: str) -> None:
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("producer_git_commit must be a lowercase 40-character Git SHA")


def prepared_gt_sha256(prepared_scene_root: Path) -> str:
    """Hash the exact frame IDs and poses used by a privileged sidecar."""
    frame_ids, poses = _load_prepared_gt(prepared_scene_root)
    digest = hashlib.sha256()
    for array in (frame_ids, poses):
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


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
        raise ValueError("VRFM residual prediction members do not match the schema")
    if any(value.dtype.hasobject for value in arrays.values()):
        raise ValueError("VRFM residual prediction may not contain object arrays")
    alphas = _normalize_alphas(arrays["alphas"])
    z = arrays["z"]
    if z.ndim != 3 or z.shape[0] != 8 or z.shape[1] < 1 or z.shape[2] < 1:
        raise ValueError("VRFM residual z must have shape [8, samples, z_dim]")
    samples = z.shape[1]
    expected = {
        "source_sample_ids": (8,),
        "overlap_frame_ids": (8, 50),
        "span_starts": (8,),
        "latent_cluster_ids": (8, samples),
        "sample_seeds": (8, samples),
        "decode_context_frames": (),
        "camera_iterations": (),
        "decode_protocol": (),
        "decoded_camera_raw": (8, samples, len(alphas), 50, 9),
        "decoded_camera_c2w": (8, samples, len(alphas), 50, 4, 4),
        "residual_rms": (8, samples),
        "vrfm_checkpoint_sha256": (),
        "camera_head_checkpoint_sha256": (),
        "source_shard_sha256": (),
        "candidate_shard_sha256": (),
        "producer_git_commit": (),
    }
    for name, shape in expected.items():
        if arrays[name].shape != shape:
            raise ValueError(f"VRFM residual prediction member {name} has invalid shape")
    if arrays["source_sample_ids"].dtype.kind != "U":
        raise ValueError("VRFM residual sample IDs must be Unicode")
    if (
        not np.issubdtype(arrays["decode_context_frames"].dtype, np.integer)
        or int(arrays["decode_context_frames"]) != 500
    ):
        raise ValueError("VRFM residual candidates must be decoded in 500-frame context")
    if (
        not np.issubdtype(arrays["camera_iterations"].dtype, np.integer)
        or int(arrays["camera_iterations"]) != 4
    ):
        raise ValueError("VRFM residual candidates must use four Camera Head iterations")
    if arrays["decode_protocol"].dtype.kind != "U" or str(
        arrays["decode_protocol"]
    ) != _DECODE_PROTOCOL:
        raise ValueError("VRFM residual decode protocol does not match")
    for name in (
        "vrfm_checkpoint_sha256",
        "camera_head_checkpoint_sha256",
        "source_shard_sha256",
        "candidate_shard_sha256",
    ):
        if arrays[name].dtype.kind != "U":
            raise ValueError(f"VRFM residual {name} must be Unicode")
        _validate_digest(str(arrays[name]), name)
    if arrays["producer_git_commit"].dtype.kind != "U":
        raise ValueError("VRFM residual producer Git commit must be Unicode")
    _validate_git_commit(str(arrays["producer_git_commit"]))
    for name in ("z", "decoded_camera_raw", "decoded_camera_c2w", "residual_rms"):
        if not np.issubdtype(arrays[name].dtype, np.floating) or not np.isfinite(
            arrays[name]
        ).all():
            raise ValueError(f"VRFM residual prediction member {name} must be finite")
    if not np.allclose(
        arrays["decoded_camera_c2w"][..., 3, :],
        [0.0, 0.0, 0.0, 1.0],
        atol=1e-6,
        rtol=0,
    ):
        raise ValueError("VRFM residual decoded cameras must be homogeneous")


def load_vrfm_residual_alpha_scan(path: Path) -> dict[str, np.ndarray]:
    arrays = _load_npz(path, "VRFM residual alpha-scan prediction")
    _validate_prediction(arrays)
    return arrays


def generate_vrfm_residual_alpha_scan(
    source_path: Path,
    candidate_path: Path,
    destination: Path,
    *,
    camera_head: Any,
    camera_head_checkpoint_sha256: str,
    producer_git_commit: str,
    device: str = "cuda",
    alphas: Sequence[float] = DEFAULT_ALPHAS,
    batch_size: int = 8,
) -> Path:
    """Scale existing VRFM residuals and decode them only in full G500 context."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    _validate_digest(
        camera_head_checkpoint_sha256, "camera_head_checkpoint_sha256"
    )
    _validate_git_commit(producer_git_commit)
    alpha_values = _normalize_alphas(alphas)
    source = load_source_shard(source_path)
    candidate = load_candidate_shard(candidate_path)
    if "overlap_long_c2w" not in source:
        raise ValueError("VRFM residual scan requires source long-window camera predictions")
    if not np.array_equal(candidate["source_sample_ids"], source["sample_ids"]):
        raise ValueError("candidate and source sample IDs do not match")
    if not np.array_equal(candidate["span_starts"], source["span_starts"]):
        raise ValueError("candidate and source span starts do not match")
    if not np.array_equal(candidate["source_long_tokens"], source["overlap_long_tokens"]):
        raise ValueError("candidate and source long-window tokens do not match")

    samples = candidate["z"].shape[1]
    device_value = torch.device(device)
    full_tokens = torch.from_numpy(source["global_camera_tokens"]).to(device_value)
    raw = np.empty((8, samples, len(alpha_values), 50, 9), dtype=np.float32)
    c2w = np.empty((8, samples, len(alpha_values), 50, 4, 4), dtype=np.float64)
    residual = (
        candidate["corrected_camera_tokens"]
        - candidate["source_long_tokens"][:, None]
    ).astype(np.float32)

    with torch.inference_mode():
        baseline_raw_full = decode_camera_tokens(camera_head, full_tokens[None])[0]
        for overlap, span_start in enumerate(source["span_starts"]):
            overlap_start = int(span_start) + 50
            baseline_raw = baseline_raw_full[overlap_start : overlap_start + 50]
            raw[overlap, :, 0] = baseline_raw.float().cpu().numpy()[None]
            c2w[overlap, :, 0] = source["overlap_long_c2w"][overlap][None]
            long_tokens = torch.from_numpy(
                candidate["source_long_tokens"][overlap]
            ).to(device_value)
            for alpha_index, alpha in enumerate(alpha_values[1:], start=1):
                for first in range(0, samples, batch_size):
                    last = min(first + batch_size, samples)
                    residual_batch = torch.from_numpy(
                        residual[overlap, first:last]
                    ).to(device_value)
                    mixed = long_tokens[None] + float(alpha) * residual_batch
                    sequences = full_tokens[None].expand(last - first, -1, -1).clone()
                    sequences[:, overlap_start : overlap_start + 50] = mixed
                    decoded_full = decode_camera_tokens(camera_head, sequences)
                    decoded = decoded_full[:, overlap_start : overlap_start + 50]
                    raw[overlap, first:last, alpha_index] = (
                        decoded.float().cpu().numpy()
                    )
                    c2w[overlap, first:last, alpha_index] = (
                        pose_encoding_to_c2w(decoded).double().cpu().numpy()
                    )

    arrays = {
        "alphas": alpha_values,
        "source_sample_ids": source["sample_ids"].copy(),
        "overlap_frame_ids": source["overlap_frame_ids"].copy(),
        "span_starts": source["span_starts"].copy(),
        "z": candidate["z"].copy(),
        "latent_cluster_ids": candidate["latent_cluster_ids"].copy(),
        "sample_seeds": candidate["sample_seeds"].copy(),
        "decode_context_frames": np.asarray(500, dtype=np.int64),
        "camera_iterations": np.asarray(4, dtype=np.int64),
        "decode_protocol": np.asarray(_DECODE_PROTOCOL, dtype="U64"),
        "decoded_camera_raw": raw,
        "decoded_camera_c2w": c2w,
        "residual_rms": np.sqrt(
            np.mean(residual.astype(np.float64) ** 2, axis=(2, 3))
        ),
        "vrfm_checkpoint_sha256": candidate["checkpoint_sha256"].copy(),
        "camera_head_checkpoint_sha256": np.asarray(
            camera_head_checkpoint_sha256, dtype="U64"
        ),
        "source_shard_sha256": np.asarray(
            _sha256_file(source_path), dtype="U64"
        ),
        "candidate_shard_sha256": np.asarray(
            _sha256_file(candidate_path), dtype="U64"
        ),
        "producer_git_commit": np.asarray(producer_git_commit, dtype="U40"),
    }
    _validate_prediction(arrays)
    _atomic_npz(destination, arrays)
    return Path(destination)


def _validate_privileged(arrays: dict[str, np.ndarray]) -> None:
    if set(arrays) != _PRIVILEGED_MEMBERS:
        raise ValueError("VRFM residual privileged members do not match the schema")
    if any(value.dtype.hasobject for value in arrays.values()):
        raise ValueError("VRFM residual privileged sidecar may not contain object arrays")
    alphas = _normalize_alphas(arrays["alphas"])
    candidate_rms = arrays["candidate_rms"]
    if candidate_rms.ndim != 3 or candidate_rms.shape[0] != 8:
        raise ValueError("VRFM residual candidate RMS must have shape [8, samples, alphas]")
    samples = candidate_rms.shape[1]
    if candidate_rms.shape[2] != len(alphas) or samples < 1:
        raise ValueError("VRFM residual candidate RMS dimensions do not match metadata")
    expected = {
        "source_sample_ids": (8,),
        "sample_seeds": (8, samples),
        "gt_frame_ids": (8, 50),
        "gt_c2w": (8, 50, 4, 4),
        "baseline_rms": (8,),
        "relative_improvement": (8, samples, len(alphas)),
        "accept_correction": (8,),
        "best_sample_index": (8,),
        "best_alpha_index": (8,),
        "best_alpha": (8,),
        "best_sample_seed": (8,),
        "best_latent_cluster_id": (8,),
        "best_candidate_rms": (8,),
        "best_relative_improvement": (8,),
        "best_nonzero_sample_index": (8,),
        "best_nonzero_alpha_index": (8,),
        "best_nonzero_alpha": (8,),
        "best_nonzero_sample_seed": (8,),
        "best_nonzero_latent_cluster_id": (8,),
        "best_nonzero_candidate_rms": (8,),
        "best_nonzero_relative_improvement": (8,),
        "prediction_sha256": (),
        "prepared_gt_sha256": (),
        "oracle_scale": (),
        "oracle_rotation": (3, 3),
        "oracle_translation": (3,),
        "oracle_digest": (),
    }
    for name, shape in expected.items():
        if arrays[name].shape != shape:
            raise ValueError(f"VRFM residual privileged member {name} has invalid shape")
    if arrays["source_sample_ids"].dtype.kind != "U":
        raise ValueError("VRFM residual privileged sample IDs must be Unicode")
    if not np.issubdtype(arrays["accept_correction"].dtype, np.bool_):
        raise ValueError("VRFM residual accept_correction must be Boolean")
    accepted = arrays["accept_correction"]
    if (
        np.any(arrays["best_sample_index"][accepted] < 0)
        or np.any(arrays["best_alpha_index"][accepted] <= 0)
        or np.any(arrays["best_sample_index"][~accepted] != -1)
        or np.any(arrays["best_alpha_index"][~accepted] != 0)
        or np.any(arrays["best_sample_seed"][~accepted] != -1)
        or np.any(arrays["best_latent_cluster_id"][~accepted] != -1)
    ):
        raise ValueError("VRFM residual no-op and accepted labels are inconsistent")
    for name in ("prediction_sha256", "prepared_gt_sha256", "oracle_digest"):
        if arrays[name].dtype.kind != "U":
            raise ValueError(f"VRFM residual privileged {name} must be Unicode")
        _validate_digest(str(arrays[name]), name)
    for name, value in arrays.items():
        if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
            raise ValueError(f"VRFM residual privileged member {name} contains non-finite values")


def load_vrfm_residual_privileged(path: Path) -> dict[str, np.ndarray]:
    arrays = _load_npz(path, "VRFM residual privileged sidecar")
    _validate_privileged(arrays)
    return arrays


def write_vrfm_residual_privileged_sidecar(
    source_path: Path,
    prediction_path: Path,
    prepared_scene_root: Path,
    destination: Path,
) -> Path:
    """Evaluate a prediction-only residual scan in a physically separate GT sidecar."""
    source = load_source_shard(source_path)
    prediction = load_vrfm_residual_alpha_scan(prediction_path)
    if "global_pred_c2w" not in source or "overlap_long_c2w" not in source:
        raise ValueError("VRFM residual scan requires source prediction poses")
    if not np.array_equal(prediction["source_sample_ids"], source["sample_ids"]):
        raise ValueError("VRFM residual prediction and source sample IDs do not match")
    if str(prediction["source_shard_sha256"]) != _sha256_file(source_path):
        raise ValueError("VRFM residual prediction source digest does not match")
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
    samples = prediction["z"].shape[1]
    alphas = prediction["alphas"]
    gt_overlap = np.empty((8, 50, 4, 4), dtype=np.float64)
    baseline_rms = np.empty(8, dtype=np.float64)
    candidate_rms = np.empty((8, samples, len(alphas)), dtype=np.float64)
    for overlap, frame_ids in enumerate(source["overlap_frame_ids"]):
        indices = np.searchsorted(gt_ids, frame_ids)
        if np.any(indices >= len(gt_ids)) or not np.array_equal(gt_ids[indices], frame_ids):
            raise ValueError("overlap frame IDs are absent from prepared GT")
        gt_overlap[overlap] = gt_c2w[indices]
        baseline_rms[overlap] = evaluate_with_frozen_oracle(
            oracle,
            source["overlap_long_c2w"][overlap],
            gt_overlap[overlap],
        ).rms_translation_error
        candidate_rms[overlap, :, 0] = baseline_rms[overlap]
        for sample in range(samples):
            for alpha_index in range(1, len(alphas)):
                candidate_rms[overlap, sample, alpha_index] = (
                    evaluate_with_frozen_oracle(
                        oracle,
                        prediction["decoded_camera_c2w"][
                            overlap, sample, alpha_index
                        ],
                        gt_overlap[overlap],
                    ).rms_translation_error
                )

    denominator = np.maximum(
        baseline_rms[:, None, None], np.finfo(np.float64).eps
    )
    relative = (baseline_rms[:, None, None] - candidate_rms) / denominator
    relative[:, :, 0] = 0.0
    rows = np.arange(8)
    nonzero_flattened = candidate_rms[:, :, 1:].reshape(8, -1)
    best_nonzero_flat = np.argmin(nonzero_flattened, axis=1).astype(np.int64)
    best_nonzero_sample = best_nonzero_flat // (len(alphas) - 1)
    best_nonzero_alpha_index = best_nonzero_flat % (len(alphas) - 1) + 1
    best_nonzero_rms = candidate_rms[
        rows, best_nonzero_sample, best_nonzero_alpha_index
    ]
    best_nonzero_relative = relative[
        rows, best_nonzero_sample, best_nonzero_alpha_index
    ]
    accept_correction = best_nonzero_rms < baseline_rms
    best_sample = np.where(accept_correction, best_nonzero_sample, -1).astype(
        np.int64
    )
    best_alpha_index = np.where(
        accept_correction, best_nonzero_alpha_index, 0
    ).astype(np.int64)
    best_sample_seed = np.where(
        accept_correction,
        prediction["sample_seeds"][rows, best_nonzero_sample],
        -1,
    ).astype(np.int64)
    best_cluster = np.where(
        accept_correction,
        prediction["latent_cluster_ids"][rows, best_nonzero_sample],
        -1,
    ).astype(np.int64)
    best_candidate_rms = np.where(
        accept_correction, best_nonzero_rms, baseline_rms
    )
    best_relative = np.where(
        accept_correction, best_nonzero_relative, 0.0
    )
    arrays = {
        "alphas": alphas.copy(),
        "source_sample_ids": source["sample_ids"].copy(),
        "sample_seeds": prediction["sample_seeds"].copy(),
        "gt_frame_ids": source["overlap_frame_ids"].copy(),
        "gt_c2w": gt_overlap,
        "baseline_rms": baseline_rms,
        "candidate_rms": candidate_rms,
        "relative_improvement": relative,
        "accept_correction": accept_correction,
        "best_sample_index": best_sample,
        "best_alpha_index": best_alpha_index,
        "best_alpha": alphas[best_alpha_index],
        "best_sample_seed": best_sample_seed,
        "best_latent_cluster_id": best_cluster,
        "best_candidate_rms": best_candidate_rms,
        "best_relative_improvement": best_relative,
        "best_nonzero_sample_index": best_nonzero_sample,
        "best_nonzero_alpha_index": best_nonzero_alpha_index,
        "best_nonzero_alpha": alphas[best_nonzero_alpha_index],
        "best_nonzero_sample_seed": prediction["sample_seeds"][
            rows, best_nonzero_sample
        ],
        "best_nonzero_latent_cluster_id": prediction["latent_cluster_ids"][
            rows, best_nonzero_sample
        ],
        "best_nonzero_candidate_rms": best_nonzero_rms,
        "best_nonzero_relative_improvement": best_nonzero_relative,
        "prediction_sha256": np.asarray(
            _sha256_file(prediction_path), dtype="U64"
        ),
        "prepared_gt_sha256": np.asarray(
            prepared_gt_sha256(prepared_scene_root), dtype="U64"
        ),
        "oracle_scale": np.asarray(oracle.scale, dtype=np.float64),
        "oracle_rotation": np.asarray(oracle.rotation, dtype=np.float64),
        "oracle_translation": np.asarray(oracle.translation, dtype=np.float64),
        "oracle_digest": np.asarray(oracle.transform_digest, dtype="U64"),
    }
    _validate_privileged(arrays)
    _atomic_npz(destination, arrays)
    return Path(destination)


def summarize_vrfm_residual_statistics(
    alphas: Sequence[float] | np.ndarray,
    relative_improvement: np.ndarray,
    *,
    min_improvement: float = 0.01,
) -> dict[str, object]:
    alpha_values = _normalize_alphas(alphas)
    relative = np.asarray(relative_improvement, dtype=np.float64)
    if (
        relative.ndim != 3
        or relative.shape[0] < 1
        or relative.shape[1] < 1
        or relative.shape[2] != len(alpha_values)
        or not np.isfinite(relative).all()
    ):
        raise ValueError(
            "relative improvement must have shape [overlaps, samples, alphas]"
        )
    if not np.isfinite(min_improvement) or min_improvement <= 0.0:
        raise ValueError("min_improvement must be finite and positive")

    overlaps, samples, _ = relative.shape
    nonzero = relative[:, :, 1:]
    direction_best = np.max(nonzero, axis=2)
    flat = nonzero.reshape(overlaps, -1)
    best_flat = np.argmax(flat, axis=1)
    best_sample = best_flat // (len(alpha_values) - 1)
    best_alpha_index = best_flat % (len(alpha_values) - 1) + 1
    rows = np.arange(overlaps)
    best_improvement = relative[rows, best_sample, best_alpha_index]
    best_alpha = alpha_values[best_alpha_index]
    useful_overlap = best_improvement > min_improvement
    full_step_best = np.max(relative[:, :, -1], axis=1)
    small_step_best = useful_overlap & (best_alpha < 1.0)
    small_step_rescue = small_step_best & (full_step_best <= 0.0)
    useful_count = int(np.count_nonzero(useful_overlap))
    rescue_count = int(np.count_nonzero(small_step_rescue))
    if useful_count > overlaps / 2:
        diagnosis = (
            "VRFM_ORACLE_CANDIDATE_SET_CONTAINS_USEFUL_SMALL_STEP"
            if rescue_count > useful_count / 2
            else "VRFM_ORACLE_CANDIDATE_SET_CONTAINS_USEFUL_CORRECTIONS"
        )
    elif useful_count > 0:
        diagnosis = "VRFM_ORACLE_CANDIDATE_SET_SPARSE_OR_MIXED"
    else:
        diagnosis = "VRFM_ORACLE_CANDIDATE_SET_HAS_NO_USEFUL_CORRECTION"

    per_alpha: dict[str, object] = {}
    for alpha_index, alpha in enumerate(alpha_values):
        values = relative[:, :, alpha_index]
        best_values = np.max(values, axis=1)
        per_alpha[f"{alpha:g}"] = {
            "median_best_of_samples_relative_improvement": float(
                np.median(best_values)
            ),
            "positive_overlap_count": int(np.count_nonzero(best_values > 0.0)),
            "useful_overlap_count": int(
                np.count_nonzero(best_values > min_improvement)
            ),
            "positive_candidate_fraction": float(np.mean(values > 0.0)),
            "useful_candidate_fraction": float(
                np.mean(values > min_improvement)
            ),
        }

    return {
        "diagnosis": diagnosis,
        "oracle_upper_bound": True,
        "overlap_count": int(overlaps),
        "samples_per_overlap": int(samples),
        "direction_count": int(overlaps * samples),
        "min_improvement": float(min_improvement),
        "useful_overlap_count": useful_count,
        "useful_overlap_fraction": float(useful_count / overlaps),
        "positive_direction_count": int(np.count_nonzero(direction_best > 0.0)),
        "useful_direction_count": int(
            np.count_nonzero(direction_best > min_improvement)
        ),
        "useful_direction_fraction": float(
            np.mean(direction_best > min_improvement)
        ),
        "small_step_best_overlap_count": int(np.count_nonzero(small_step_best)),
        "small_step_rescue_overlap_count": rescue_count,
        "full_step_useful_overlap_count": int(
            np.count_nonzero(full_step_best > min_improvement)
        ),
        "no_op_best_overlap_count": int(np.count_nonzero(best_improvement <= 0.0)),
        "median_best_nonzero_relative_improvement": float(
            np.median(best_improvement)
        ),
        "median_safe_best_relative_improvement": float(
            np.median(np.maximum(best_improvement, 0.0))
        ),
        "best_alpha_counts": {
            f"{alpha:g}": int(
                np.count_nonzero(useful_overlap & (best_alpha == alpha))
            )
            for alpha in alpha_values[1:]
        },
        "per_alpha": per_alpha,
    }


def write_vrfm_residual_report(
    sidecar_paths: Sequence[Path],
    destination: Path,
    *,
    min_improvement: float = 0.01,
) -> dict[str, object]:
    if not sidecar_paths:
        raise ValueError("at least one VRFM residual sidecar is required")
    loaded = [load_vrfm_residual_privileged(path) for path in sidecar_paths]
    alphas = loaded[0]["alphas"]
    samples = loaded[0]["candidate_rms"].shape[1]
    if any(not np.array_equal(arrays["alphas"], alphas) for arrays in loaded[1:]):
        raise ValueError("VRFM residual sidecars use different alpha grids")
    if any(arrays["candidate_rms"].shape[1] != samples for arrays in loaded[1:]):
        raise ValueError("VRFM residual sidecars use different sample counts")
    relative = np.concatenate(
        [arrays["relative_improvement"] for arrays in loaded], axis=0
    )
    per_scene = []
    for arrays in loaded:
        scene = str(arrays["source_sample_ids"][0]).split(":", 1)[0]
        per_scene.append(
            {
                "scene": scene,
                **summarize_vrfm_residual_statistics(
                    alphas,
                    arrays["relative_improvement"],
                    min_improvement=min_improvement,
                ),
            }
        )
    report = {
        "schema": "variational_camera_latent.vrfm_residual_alpha_scan_full_context_report.v1",
        "scene_count": len(loaded),
        "alphas": alphas.tolist(),
        "selection_uses_privileged_labels": True,
        "prediction_data_contains_privileged_labels": False,
        "claim_scope": "GT-oracle existence test over the frozen VRFM candidate set; not a deployable selector",
        "primary_metric": "frozen_scene_sim3_rms_translation_error",
        "rotation_not_used_for_selection": True,
        "decode_context_frames": 500,
        "supersedes_isolated_50_frame_quality_metrics": True,
        "per_scene": per_scene,
        **summarize_vrfm_residual_statistics(
            alphas,
            relative,
            min_improvement=min_improvement,
        ),
    }
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(destination)
    return report
