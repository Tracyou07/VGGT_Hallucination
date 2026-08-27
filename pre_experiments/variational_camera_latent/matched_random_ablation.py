"""Matched equal-norm random controls for the VRFM residual alpha scan.

Prediction artifacts produced here are deliberately prediction-only.  Ground
truth poses and all evaluation labels are written by the physically separate
privileged-sidecar API near the end of this module.
"""

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
from .candidates import load_candidate_shard
from .privileged import _load_prepared_gt
from .source import load_source_shard
from .vrfm_residual_scan import (
    _atomic_npz,
    _load_npz,
    _normalize_alphas,
    _sha256_file,
    _validate_digest,
    _validate_git_commit,
    load_vrfm_residual_alpha_scan,
    load_vrfm_residual_privileged,
    prepared_gt_sha256,
    summarize_vrfm_residual_statistics,
)


_FEATURE_DIMENSION = 2048
_OVERLAPS = 8
_OVERLAP_FRAMES = 50
_CONTEXT_FRAMES = 500
_CAMERA_ITERATIONS = 4
_RNG_ALGORITHM = "numpy.PCG64"
_TRANSFORM_PROTOCOL = "shared_signed_hadamard_permutation_features_2048.v1"
_DECODE_PROTOCOL = "matched_random_ablation_full_g500.v1"

_PREDICTION_MEMBERS = {
    "alphas",
    "source_sample_ids",
    "overlap_frame_ids",
    "span_starts",
    "z",
    "latent_cluster_ids",
    "sample_seeds",
    "random_direction_seeds",
    "random_direction_sha256",
    "transform_identity_sha256",
    "transform_sha256",
    "transform_protocol",
    "base_seed",
    "rng_algorithm",
    "decode_context_frames",
    "camera_iterations",
    "decode_protocol",
    "decoded_camera_raw",
    "decoded_camera_c2w",
    "vrfm_residual_rms",
    "random_residual_rms",
    "cosine_to_vrfm",
    "vrfm_checkpoint_sha256",
    "camera_head_checkpoint_sha256",
    "source_shard_sha256",
    "candidate_shard_sha256",
    "paired_vrfm_prediction_sha256",
    "paired_vrfm_producer_git_commit",
    "producer_git_commit",
}

_PRIVILEGED_MEMBERS = {
    "alphas",
    "source_sample_ids",
    "sample_seeds",
    "gt_frame_ids",
    "gt_c2w",
    "baseline_rms",
    "vrfm_candidate_rms",
    "random_candidate_rms",
    "vrfm_relative_improvement",
    "random_relative_improvement",
    "paired_relative_advantage",
    "random_prediction_sha256",
    "vrfm_prediction_sha256",
    "vrfm_privileged_sha256",
    "prepared_gt_sha256",
    "oracle_scale",
    "oracle_rotation",
    "oracle_translation",
    "oracle_digest",
}

_FORBIDDEN_PREDICTION_MEMBER_FRAGMENTS = (
    "gt",
    "privileged",
    "error",
    "quality",
    "depth",
)


def _validate_base_seed(value: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError("base_seed must be an integer")
    normalized = int(value)
    if normalized < 0 or normalized >= 2**63:
        raise ValueError("base_seed must be in [0, 2**63)")
    return normalized


def _transform_seed_material(base_seed: int, identity_sha256: str) -> bytes:
    payload = f"{_TRANSFORM_PROTOCOL}|{base_seed}|{identity_sha256}"
    return hashlib.sha256(payload.encode("ascii")).digest()


def _build_shared_transform(
    *,
    base_seed: int,
    identity_sha256: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, str]:
    """Build one implicit orthogonal transform shared by every residual row."""
    seed_material = _transform_seed_material(base_seed, identity_sha256)
    generator = np.random.Generator(
        np.random.PCG64(int.from_bytes(seed_material, byteorder="big"))
    )
    pre_signs = (
        generator.integers(0, 2, size=_FEATURE_DIMENSION, dtype=np.int8) * 2 - 1
    )
    permutation = generator.permutation(_FEATURE_DIMENSION).astype(np.int64)
    post_signs = (
        generator.integers(0, 2, size=_FEATURE_DIMENSION, dtype=np.int8) * 2 - 1
    )
    recorded_seed = int.from_bytes(seed_material[:8], byteorder="big") & (2**63 - 1)

    digest = hashlib.sha256()
    digest.update(_TRANSFORM_PROTOCOL.encode("ascii"))
    digest.update(_RNG_ALGORITHM.encode("ascii"))
    digest.update(np.asarray(base_seed, dtype=np.int64).tobytes())
    digest.update(identity_sha256.encode("ascii"))
    digest.update(pre_signs.tobytes())
    digest.update(permutation.tobytes())
    digest.update(post_signs.tobytes())
    return pre_signs, permutation, post_signs, recorded_seed, digest.hexdigest()


def _fwht_in_place(values: np.ndarray) -> None:
    width = values.shape[1]
    stride = 1
    while stride < width:
        blocks = values.reshape(-1, width // (2 * stride), 2, stride)
        left = blocks[:, :, 0, :].copy()
        right = blocks[:, :, 1, :].copy()
        blocks[:, :, 0, :] = left + right
        blocks[:, :, 1, :] = left - right
        stride *= 2


def _apply_shared_transform(
    residuals: np.ndarray,
    pre_signs: np.ndarray,
    permutation: np.ndarray,
    post_signs: np.ndarray,
    *,
    row_batch_size: int = 512,
) -> np.ndarray:
    flat = np.asarray(residuals, dtype=np.float32).reshape(-1, _FEATURE_DIMENSION)
    transformed = np.empty(flat.shape, dtype=np.float32)
    normalization = np.sqrt(float(_FEATURE_DIMENSION))
    for first in range(0, len(flat), row_batch_size):
        last = min(first + row_batch_size, len(flat))
        working = flat[first:last].astype(np.float64, copy=True)
        working *= pre_signs[None]
        _fwht_in_place(working)
        working /= normalization
        working = working[:, permutation]
        working *= post_signs[None]
        transformed[first:last] = working.astype(np.float32)
    return transformed.reshape(residuals.shape)


def _direction_digest(
    direction: np.ndarray,
    *,
    source_sample_id: str,
    sample_seed: int,
    candidate_sha256: str,
    transform_sha256: str,
) -> str:
    contiguous = np.ascontiguousarray(direction, dtype=np.float32)
    digest = hashlib.sha256()
    digest.update(source_sample_id.encode("utf-8"))
    digest.update(np.asarray(sample_seed, dtype=np.int64).tobytes())
    digest.update(candidate_sha256.encode("ascii"))
    digest.update(transform_sha256.encode("ascii"))
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def make_matched_random_directions(
    residuals: np.ndarray,
    *,
    source_sample_ids: np.ndarray,
    sample_seeds: np.ndarray,
    candidate_sha256: str,
    base_seed: int,
    transform_identity_sha256: str | None = None,
) -> dict[str, np.ndarray]:
    """Apply one deterministic shared orthogonal Q to all VRFM residual rows.

    The transform acts only on the 2048-dimensional feature axis.  Consequently
    it preserves every row norm and the complete row Gram matrix, including all
    temporal and cross-sample inner products.  A shared manifest digest may be
    supplied as ``transform_identity_sha256`` to reuse the exact same Q across
    scene-level calls.
    """
    values = np.asarray(residuals)
    if (
        values.ndim != 4
        or values.shape[0] != _OVERLAPS
        or values.shape[2:] != (_OVERLAP_FRAMES, _FEATURE_DIMENSION)
        or values.shape[1] < 1
        or values.dtype != np.float32
        or not np.isfinite(values).all()
    ):
        raise ValueError(
            "residuals must be finite float32 with shape [8, samples, 50, 2048]"
        )
    samples = values.shape[1]
    identities = np.asarray(source_sample_ids)
    seeds = np.asarray(sample_seeds)
    if identities.shape != (_OVERLAPS,) or identities.dtype.kind != "U":
        raise ValueError("source_sample_ids must be a Unicode vector with shape [8]")
    if len(np.unique(identities)) != _OVERLAPS:
        raise ValueError("source_sample_ids must be unique")
    if seeds.shape != (_OVERLAPS, samples) or not np.issubdtype(
        seeds.dtype, np.integer
    ):
        raise ValueError("sample_seeds must be an integer array [8, samples]")
    _validate_digest(candidate_sha256, "candidate_sha256")
    identity_digest = (
        candidate_sha256
        if transform_identity_sha256 is None
        else transform_identity_sha256
    )
    _validate_digest(identity_digest, "transform_identity_sha256")
    normalized_seed = _validate_base_seed(base_seed)

    pre_signs, permutation, post_signs, recorded_seed, transform_digest = (
        _build_shared_transform(
            base_seed=normalized_seed,
            identity_sha256=identity_digest,
        )
    )
    random_residuals = _apply_shared_transform(
        values.astype(np.float32, copy=False),
        pre_signs,
        permutation,
        post_signs,
    )

    vrfm_rms = np.empty((_OVERLAPS, samples), dtype=np.float64)
    random_rms = np.empty((_OVERLAPS, samples), dtype=np.float64)
    cosine = np.empty((_OVERLAPS, samples), dtype=np.float64)
    direction_digests = np.empty((_OVERLAPS, samples), dtype="U64")
    for overlap in range(_OVERLAPS):
        for sample in range(samples):
            # Keep diagnostics bounded to one direction instead of materializing
            # two full float64 copies of the scene-level residual tensor.
            vrfm64 = values[overlap, sample].astype(np.float64)
            random64 = random_residuals[overlap, sample].astype(np.float64)
            vrfm_squared_norm = float(np.sum(vrfm64 * vrfm64))
            random_squared_norm = float(np.sum(random64 * random64))
            vrfm_rms[overlap, sample] = np.sqrt(
                vrfm_squared_norm / vrfm64.size
            )
            random_rms[overlap, sample] = np.sqrt(
                random_squared_norm / random64.size
            )
            denominator = np.sqrt(vrfm_squared_norm * random_squared_norm)
            cosine[overlap, sample] = (
                float(np.sum(vrfm64 * random64)) / denominator
                if denominator > 0.0
                else 0.0
            )
            direction_digests[overlap, sample] = _direction_digest(
                random_residuals[overlap, sample],
                source_sample_id=str(identities[overlap]),
                sample_seed=int(seeds[overlap, sample]),
                candidate_sha256=candidate_sha256,
                transform_sha256=transform_digest,
            )

    shared_seeds = np.full((_OVERLAPS, samples), recorded_seed, dtype=np.int64)
    return {
        "random_residuals": random_residuals,
        "random_direction_seeds": shared_seeds,
        "random_direction_sha256": direction_digests,
        "vrfm_residual_rms": vrfm_rms,
        "random_residual_rms": random_rms,
        "cosine_to_vrfm": cosine,
        "transform_identity_sha256": np.asarray(identity_digest, dtype="U64"),
        "transform_sha256": np.asarray(transform_digest, dtype="U64"),
        # Public alias retained for the low-level direction-generation API.
        "random_transform_sha256": np.asarray(transform_digest, dtype="U64"),
        "transform_protocol": np.asarray(_TRANSFORM_PROTOCOL, dtype="U96"),
        "base_seed": np.asarray(normalized_seed, dtype=np.int64),
        "rng_algorithm": np.asarray(_RNG_ALGORITHM, dtype="U32"),
    }


def _validate_prediction(arrays: dict[str, np.ndarray]) -> None:
    if set(arrays) != _PREDICTION_MEMBERS:
        raise ValueError("matched-random prediction members do not match the schema")
    if any(value.dtype.hasobject for value in arrays.values()):
        raise ValueError("matched-random prediction may not contain object arrays")
    if any(
        forbidden in name.lower()
        for name in arrays
        for forbidden in _FORBIDDEN_PREDICTION_MEMBER_FRAGMENTS
    ):
        raise ValueError("matched-random prediction contains a privileged member name")

    alphas = _normalize_alphas(arrays["alphas"])
    z = arrays["z"]
    if z.ndim != 3 or z.shape[0] != _OVERLAPS or z.shape[1] < 1 or z.shape[2] < 1:
        raise ValueError("matched-random z must have shape [8, samples, z_dim]")
    samples = z.shape[1]
    expected = {
        "source_sample_ids": (_OVERLAPS,),
        "overlap_frame_ids": (_OVERLAPS, _OVERLAP_FRAMES),
        "span_starts": (_OVERLAPS,),
        "latent_cluster_ids": (_OVERLAPS, samples),
        "sample_seeds": (_OVERLAPS, samples),
        "random_direction_seeds": (_OVERLAPS, samples),
        "random_direction_sha256": (_OVERLAPS, samples),
        "transform_identity_sha256": (),
        "transform_sha256": (),
        "transform_protocol": (),
        "base_seed": (),
        "rng_algorithm": (),
        "decode_context_frames": (),
        "camera_iterations": (),
        "decode_protocol": (),
        "decoded_camera_raw": (
            _OVERLAPS,
            samples,
            len(alphas),
            _OVERLAP_FRAMES,
            9,
        ),
        "decoded_camera_c2w": (
            _OVERLAPS,
            samples,
            len(alphas),
            _OVERLAP_FRAMES,
            4,
            4,
        ),
        "vrfm_residual_rms": (_OVERLAPS, samples),
        "random_residual_rms": (_OVERLAPS, samples),
        "cosine_to_vrfm": (_OVERLAPS, samples),
        "vrfm_checkpoint_sha256": (),
        "camera_head_checkpoint_sha256": (),
        "source_shard_sha256": (),
        "candidate_shard_sha256": (),
        "paired_vrfm_prediction_sha256": (),
        "paired_vrfm_producer_git_commit": (),
        "producer_git_commit": (),
    }
    for name, shape in expected.items():
        if arrays[name].shape != shape:
            raise ValueError(f"matched-random prediction member {name} has invalid shape")
    if arrays["source_sample_ids"].dtype.kind != "U":
        raise ValueError("matched-random sample IDs must be Unicode")
    if arrays["random_direction_sha256"].dtype.kind != "U":
        raise ValueError("matched-random direction digests must be Unicode")
    for digest in arrays["random_direction_sha256"].flat:
        _validate_digest(str(digest), "random_direction_sha256")
    for name in (
        "transform_identity_sha256",
        "transform_sha256",
        "vrfm_checkpoint_sha256",
        "camera_head_checkpoint_sha256",
        "source_shard_sha256",
        "candidate_shard_sha256",
        "paired_vrfm_prediction_sha256",
    ):
        if arrays[name].dtype.kind != "U":
            raise ValueError(f"matched-random {name} must be Unicode")
        _validate_digest(str(arrays[name]), name)
    for name in ("paired_vrfm_producer_git_commit", "producer_git_commit"):
        if arrays[name].dtype.kind != "U":
            raise ValueError(f"matched-random {name} must be Unicode")
        _validate_git_commit(str(arrays[name]))
    if arrays["transform_protocol"].dtype.kind != "U" or str(
        arrays["transform_protocol"]
    ) != _TRANSFORM_PROTOCOL:
        raise ValueError("matched-random transform protocol does not match")
    if arrays["rng_algorithm"].dtype.kind != "U" or str(
        arrays["rng_algorithm"]
    ) != _RNG_ALGORITHM:
        raise ValueError("matched-random RNG algorithm does not match")
    if arrays["decode_protocol"].dtype.kind != "U" or str(
        arrays["decode_protocol"]
    ) != _DECODE_PROTOCOL:
        raise ValueError("matched-random decode protocol does not match")
    if (
        not np.issubdtype(arrays["decode_context_frames"].dtype, np.integer)
        or int(arrays["decode_context_frames"]) != _CONTEXT_FRAMES
    ):
        raise ValueError("matched-random candidates must be decoded in 500-frame context")
    if (
        not np.issubdtype(arrays["camera_iterations"].dtype, np.integer)
        or int(arrays["camera_iterations"]) != _CAMERA_ITERATIONS
    ):
        raise ValueError("matched-random candidates must use four Camera Head iterations")
    if not np.issubdtype(arrays["base_seed"].dtype, np.integer):
        raise ValueError("matched-random base_seed must be integral")
    normalized_seed = _validate_base_seed(int(arrays["base_seed"]))
    for name in (
        "span_starts",
        "latent_cluster_ids",
        "sample_seeds",
        "random_direction_seeds",
    ):
        if not np.issubdtype(arrays[name].dtype, np.integer):
            raise ValueError(f"matched-random {name} must be integral")
    (
        _,
        _,
        _,
        expected_recorded_seed,
        expected_transform_digest,
    ) = _build_shared_transform(
        base_seed=normalized_seed,
        identity_sha256=str(arrays["transform_identity_sha256"]),
    )
    if str(arrays["transform_sha256"]) != expected_transform_digest:
        raise ValueError("matched-random transform digest is inconsistent")
    if not np.all(
        arrays["random_direction_seeds"] == expected_recorded_seed
    ):
        raise ValueError("matched-random transform seed is inconsistent")
    for name in (
        "alphas",
        "z",
        "decoded_camera_raw",
        "decoded_camera_c2w",
        "vrfm_residual_rms",
        "random_residual_rms",
        "cosine_to_vrfm",
    ):
        if not np.issubdtype(arrays[name].dtype, np.floating) or not np.isfinite(
            arrays[name]
        ).all():
            raise ValueError(f"matched-random prediction member {name} must be finite")
    if np.any(np.abs(arrays["cosine_to_vrfm"]) > 1.0 + 1e-6):
        raise ValueError("matched-random residual cosine is outside [-1, 1]")
    if not np.allclose(
        arrays["random_residual_rms"],
        arrays["vrfm_residual_rms"],
        rtol=1e-6,
        atol=1e-10,
    ):
        raise ValueError("matched-random residual norms do not match VRFM residual norms")
    if not np.allclose(
        arrays["decoded_camera_c2w"][..., 3, :],
        [0.0, 0.0, 0.0, 1.0],
        atol=1e-6,
        rtol=0,
    ):
        raise ValueError("matched-random decoded cameras must be homogeneous")


def load_matched_random_ablation(path: Path) -> dict[str, np.ndarray]:
    arrays = _load_npz(path, "matched-random prediction")
    _validate_prediction(arrays)
    return arrays


def generate_matched_random_ablation(
    source_path: Path,
    candidate_path: Path,
    vrfm_prediction_path: Path,
    destination: Path,
    *,
    camera_head: Any,
    camera_head_checkpoint_sha256: str,
    producer_git_commit: str,
    base_seed: int,
    transform_identity_sha256: str | None = None,
    device: str = "cuda",
    batch_size: int = 8,
) -> Path:
    """Decode matched-random directions under the original full G500 context."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    normalized_seed = _validate_base_seed(base_seed)
    _validate_digest(
        camera_head_checkpoint_sha256, "camera_head_checkpoint_sha256"
    )
    _validate_git_commit(producer_git_commit)

    source = load_source_shard(source_path)
    candidate = load_candidate_shard(candidate_path)
    vrfm_prediction = load_vrfm_residual_alpha_scan(vrfm_prediction_path)
    if "overlap_long_c2w" not in source:
        raise ValueError("matched-random ablation requires source long-window poses")
    if not np.array_equal(candidate["source_sample_ids"], source["sample_ids"]):
        raise ValueError("candidate and source sample IDs do not match")
    if not np.array_equal(candidate["span_starts"], source["span_starts"]):
        raise ValueError("candidate and source span starts do not match")
    if not np.array_equal(
        candidate["source_long_tokens"], source["overlap_long_tokens"]
    ):
        raise ValueError("candidate and source long-window tokens do not match")

    source_digest = _sha256_file(source_path)
    candidate_digest = _sha256_file(candidate_path)
    if str(vrfm_prediction["source_shard_sha256"]) != source_digest:
        raise ValueError("VRFM prediction source digest does not match")
    if str(vrfm_prediction["candidate_shard_sha256"]) != candidate_digest:
        raise ValueError("VRFM prediction candidate digest does not match")
    if str(vrfm_prediction["camera_head_checkpoint_sha256"]) != (
        camera_head_checkpoint_sha256
    ):
        raise ValueError("VRFM and matched-random Camera Head checkpoints differ")
    paired_metadata = (
        ("source_sample_ids", source["sample_ids"]),
        ("overlap_frame_ids", source["overlap_frame_ids"]),
        ("span_starts", source["span_starts"]),
        ("z", candidate["z"]),
        ("latent_cluster_ids", candidate["latent_cluster_ids"]),
        ("sample_seeds", candidate["sample_seeds"]),
    )
    for name, expected in paired_metadata:
        if not np.array_equal(vrfm_prediction[name], expected):
            raise ValueError(f"VRFM paired metadata {name} does not match")
    if str(vrfm_prediction["vrfm_checkpoint_sha256"]) != str(
        candidate["checkpoint_sha256"]
    ):
        raise ValueError("VRFM prediction checkpoint does not match candidate shard")

    residuals = (
        candidate["corrected_camera_tokens"]
        - candidate["source_long_tokens"][:, None]
    ).astype(np.float32)
    directions = make_matched_random_directions(
        residuals,
        source_sample_ids=source["sample_ids"],
        sample_seeds=candidate["sample_seeds"],
        candidate_sha256=candidate_digest,
        transform_identity_sha256=transform_identity_sha256,
        base_seed=normalized_seed,
    )
    random_residuals = directions["random_residuals"]
    alpha_values = vrfm_prediction["alphas"].copy()
    samples = candidate["z"].shape[1]
    raw = np.empty(
        (_OVERLAPS, samples, len(alpha_values), _OVERLAP_FRAMES, 9),
        dtype=np.float32,
    )
    c2w = np.empty(
        (_OVERLAPS, samples, len(alpha_values), _OVERLAP_FRAMES, 4, 4),
        dtype=np.float64,
    )
    device_value = torch.device(device)
    full_tokens = torch.from_numpy(source["global_camera_tokens"]).to(device_value)

    with torch.inference_mode():
        baseline_raw_full = decode_camera_tokens(camera_head, full_tokens[None])[0]
        for overlap, span_start in enumerate(source["span_starts"]):
            overlap_start = int(span_start) + _OVERLAP_FRAMES
            baseline_raw = baseline_raw_full[
                overlap_start : overlap_start + _OVERLAP_FRAMES
            ]
            raw[overlap, :, 0] = baseline_raw.float().cpu().numpy()[None]
            c2w[overlap, :, 0] = source["overlap_long_c2w"][overlap][None]
            long_tokens = torch.from_numpy(
                candidate["source_long_tokens"][overlap]
            ).to(device_value)
            for alpha_index, alpha in enumerate(alpha_values[1:], start=1):
                for first in range(0, samples, batch_size):
                    last = min(first + batch_size, samples)
                    random_batch = torch.from_numpy(
                        random_residuals[overlap, first:last]
                    ).to(device_value)
                    mixed = long_tokens[None] + float(alpha) * random_batch
                    sequences = full_tokens[None].expand(last - first, -1, -1).clone()
                    sequences[
                        :, overlap_start : overlap_start + _OVERLAP_FRAMES
                    ] = mixed
                    decoded_full = decode_camera_tokens(camera_head, sequences)
                    decoded = decoded_full[
                        :, overlap_start : overlap_start + _OVERLAP_FRAMES
                    ]
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
        "random_direction_seeds": directions["random_direction_seeds"].copy(),
        "random_direction_sha256": directions["random_direction_sha256"].copy(),
        "transform_identity_sha256": directions[
            "transform_identity_sha256"
        ].copy(),
        "transform_sha256": directions["transform_sha256"].copy(),
        "transform_protocol": directions["transform_protocol"].copy(),
        "base_seed": directions["base_seed"].copy(),
        "rng_algorithm": directions["rng_algorithm"].copy(),
        "decode_context_frames": np.asarray(_CONTEXT_FRAMES, dtype=np.int64),
        "camera_iterations": np.asarray(_CAMERA_ITERATIONS, dtype=np.int64),
        "decode_protocol": np.asarray(_DECODE_PROTOCOL, dtype="U64"),
        "decoded_camera_raw": raw,
        "decoded_camera_c2w": c2w,
        "vrfm_residual_rms": directions["vrfm_residual_rms"].copy(),
        "random_residual_rms": directions["random_residual_rms"].copy(),
        "cosine_to_vrfm": directions["cosine_to_vrfm"].copy(),
        "vrfm_checkpoint_sha256": candidate["checkpoint_sha256"].copy(),
        "camera_head_checkpoint_sha256": np.asarray(
            camera_head_checkpoint_sha256, dtype="U64"
        ),
        "source_shard_sha256": np.asarray(source_digest, dtype="U64"),
        "candidate_shard_sha256": np.asarray(candidate_digest, dtype="U64"),
        "paired_vrfm_prediction_sha256": np.asarray(
            _sha256_file(vrfm_prediction_path), dtype="U64"
        ),
        "paired_vrfm_producer_git_commit": vrfm_prediction[
            "producer_git_commit"
        ].copy(),
        "producer_git_commit": np.asarray(producer_git_commit, dtype="U40"),
    }
    _validate_prediction(arrays)
    _atomic_npz(destination, arrays)
    return Path(destination)


def _validate_privileged(arrays: dict[str, np.ndarray]) -> None:
    if set(arrays) != _PRIVILEGED_MEMBERS:
        raise ValueError("matched-random privileged members do not match the schema")
    if any(value.dtype.hasobject for value in arrays.values()):
        raise ValueError("matched-random privileged sidecar may not contain object arrays")
    alphas = _normalize_alphas(arrays["alphas"])
    random_rms = arrays["random_candidate_rms"]
    if random_rms.ndim != 3 or random_rms.shape[0] != _OVERLAPS:
        raise ValueError("random candidate RMS must have shape [8, samples, alphas]")
    samples = random_rms.shape[1]
    if samples < 1 or random_rms.shape[2] != len(alphas):
        raise ValueError("random candidate RMS dimensions do not match metadata")
    expected = {
        "source_sample_ids": (_OVERLAPS,),
        "sample_seeds": (_OVERLAPS, samples),
        "gt_frame_ids": (_OVERLAPS, _OVERLAP_FRAMES),
        "gt_c2w": (_OVERLAPS, _OVERLAP_FRAMES, 4, 4),
        "baseline_rms": (_OVERLAPS,),
        "vrfm_candidate_rms": (_OVERLAPS, samples, len(alphas)),
        "vrfm_relative_improvement": (_OVERLAPS, samples, len(alphas)),
        "random_relative_improvement": (_OVERLAPS, samples, len(alphas)),
        "paired_relative_advantage": (_OVERLAPS, samples, len(alphas)),
        "random_prediction_sha256": (),
        "vrfm_prediction_sha256": (),
        "vrfm_privileged_sha256": (),
        "prepared_gt_sha256": (),
        "oracle_scale": (),
        "oracle_rotation": (3, 3),
        "oracle_translation": (3,),
        "oracle_digest": (),
    }
    for name, shape in expected.items():
        if arrays[name].shape != shape:
            raise ValueError(f"matched-random privileged member {name} has invalid shape")
    if arrays["source_sample_ids"].dtype.kind != "U":
        raise ValueError("matched-random privileged sample IDs must be Unicode")
    for name in ("sample_seeds", "gt_frame_ids"):
        if not np.issubdtype(arrays[name].dtype, np.integer):
            raise ValueError(f"matched-random privileged {name} must be integral")
    for name in (
        "random_prediction_sha256",
        "vrfm_prediction_sha256",
        "vrfm_privileged_sha256",
        "prepared_gt_sha256",
        "oracle_digest",
    ):
        if arrays[name].dtype.kind != "U":
            raise ValueError(f"matched-random privileged {name} must be Unicode")
        _validate_digest(str(arrays[name]), name)
    for name, value in arrays.items():
        if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
            raise ValueError(
                f"matched-random privileged member {name} contains non-finite values"
            )
    if not np.allclose(
        arrays["gt_c2w"][:, :, 3, :],
        [0.0, 0.0, 0.0, 1.0],
        atol=1e-10,
        rtol=0,
    ):
        raise ValueError("matched-random privileged GT poses must be homogeneous")
    expected_baseline = np.repeat(
        arrays["baseline_rms"][:, None], samples, axis=1
    )
    for name in ("vrfm_candidate_rms", "random_candidate_rms"):
        if np.any(arrays[name] < 0.0):
            raise ValueError(f"matched-random privileged {name} must be non-negative")
        if not np.allclose(
            arrays[name][:, :, 0], expected_baseline, atol=1e-12, rtol=0
        ):
            raise ValueError(f"matched-random {name} alpha zero must equal the baseline")
    if np.any(arrays["baseline_rms"] < 0.0):
        raise ValueError("matched-random privileged baseline RMS must be non-negative")
    denominator = np.maximum(
        arrays["baseline_rms"][:, None, None], np.finfo(np.float64).eps
    )
    expected_vrfm_relative = (
        arrays["baseline_rms"][:, None, None] - arrays["vrfm_candidate_rms"]
    ) / denominator
    expected_random_relative = (
        arrays["baseline_rms"][:, None, None] - arrays["random_candidate_rms"]
    ) / denominator
    expected_vrfm_relative[:, :, 0] = 0.0
    expected_random_relative[:, :, 0] = 0.0
    for name, expected_relative in (
        ("vrfm_relative_improvement", expected_vrfm_relative),
        ("random_relative_improvement", expected_random_relative),
    ):
        if not np.allclose(
            arrays[name], expected_relative, atol=1e-12, rtol=0
        ):
            raise ValueError(f"matched-random privileged {name} is inconsistent")
    if not np.allclose(
        arrays["paired_relative_advantage"],
        arrays["vrfm_relative_improvement"]
        - arrays["random_relative_improvement"],
        atol=1e-12,
        rtol=0,
    ):
        raise ValueError("paired relative advantage is inconsistent")


def load_matched_random_privileged(path: Path) -> dict[str, np.ndarray]:
    arrays = _load_npz(path, "matched-random privileged sidecar")
    _validate_privileged(arrays)
    return arrays


def write_matched_random_privileged_sidecar(
    source_path: Path,
    random_prediction_path: Path,
    vrfm_prediction_path: Path,
    vrfm_privileged_path: Path,
    prepared_scene_root: Path,
    destination: Path,
) -> Path:
    """Compare paired VRFM/random cells using the already-frozen VRFM oracle."""
    source = load_source_shard(source_path)
    random_prediction = load_matched_random_ablation(random_prediction_path)
    vrfm_prediction = load_vrfm_residual_alpha_scan(vrfm_prediction_path)
    upstream = load_vrfm_residual_privileged(vrfm_privileged_path)
    if "global_pred_c2w" not in source or "overlap_long_c2w" not in source:
        raise ValueError("matched-random sidecar requires source prediction poses")

    source_digest = _sha256_file(source_path)
    vrfm_prediction_digest = _sha256_file(vrfm_prediction_path)
    if str(random_prediction["source_shard_sha256"]) != source_digest:
        raise ValueError("matched-random prediction source digest does not match")
    if str(vrfm_prediction["source_shard_sha256"]) != source_digest:
        raise ValueError("VRFM prediction source digest does not match")
    if str(random_prediction["paired_vrfm_prediction_sha256"]) != (
        vrfm_prediction_digest
    ):
        raise ValueError("matched-random prediction is not paired to this VRFM scan")
    if str(upstream["prediction_sha256"]) != vrfm_prediction_digest:
        raise ValueError("VRFM privileged sidecar is not paired to this VRFM scan")
    for name, expected in (
        ("source_sample_ids", source["sample_ids"]),
        ("sample_seeds", vrfm_prediction["sample_seeds"]),
        ("alphas", vrfm_prediction["alphas"]),
    ):
        if not np.array_equal(random_prediction[name], expected):
            raise ValueError(f"matched-random paired metadata {name} does not match")
        if not np.array_equal(upstream[name], expected):
            raise ValueError(f"VRFM privileged metadata {name} does not match")

    prepared_digest = prepared_gt_sha256(prepared_scene_root)
    if str(upstream["prepared_gt_sha256"]) != prepared_digest:
        raise ValueError("VRFM privileged prepared-GT digest does not match")
    gt_ids, raw_gt_c2w = _load_prepared_gt(prepared_scene_root)
    if not np.array_equal(gt_ids, source["global_frame_ids"]):
        raise ValueError("prepared GT and source frame IDs do not match")
    scene = str(source["sample_ids"][0]).split(":", 1)[0]
    oracle = fit_frozen_oracle(
        scene,
        source["global_frame_ids"],
        source["global_pred_c2w"],
        raw_gt_c2w,
    )
    if oracle.transform_digest != str(upstream["oracle_digest"]):
        raise ValueError("VRFM privileged sidecar uses a different frozen oracle")
    if not np.array_equal(
        np.asarray(oracle.rotation, dtype=np.float64), upstream["oracle_rotation"]
    ) or not np.array_equal(
        np.asarray(oracle.translation, dtype=np.float64), upstream["oracle_translation"]
    ) or float(upstream["oracle_scale"]) != oracle.scale:
        raise ValueError("VRFM privileged oracle parameters do not match")

    samples = random_prediction["z"].shape[1]
    alphas = random_prediction["alphas"]
    gt_overlap = np.empty((_OVERLAPS, _OVERLAP_FRAMES, 4, 4), dtype=np.float64)
    random_candidate_rms = np.empty(
        (_OVERLAPS, samples, len(alphas)), dtype=np.float64
    )
    vrfm_candidate_check = np.empty_like(random_candidate_rms)
    baseline_rms = upstream["baseline_rms"].copy()
    for overlap, frame_ids in enumerate(source["overlap_frame_ids"]):
        indices = np.searchsorted(gt_ids, frame_ids)
        if np.any(indices >= len(gt_ids)) or not np.array_equal(
            gt_ids[indices], frame_ids
        ):
            raise ValueError("overlap frame IDs are absent from prepared GT")
        gt_overlap[overlap] = raw_gt_c2w[indices]
        baseline_check = evaluate_with_frozen_oracle(
            oracle,
            source["overlap_long_c2w"][overlap],
            gt_overlap[overlap],
        ).rms_translation_error
        if not np.isclose(
            baseline_check, baseline_rms[overlap], atol=1e-12, rtol=0
        ):
            raise ValueError("VRFM privileged baseline does not match the frozen oracle")
        random_candidate_rms[overlap, :, 0] = baseline_rms[overlap]
        vrfm_candidate_check[overlap, :, 0] = baseline_rms[overlap]
        for sample in range(samples):
            for alpha_index in range(1, len(alphas)):
                vrfm_candidate_check[overlap, sample, alpha_index] = (
                    evaluate_with_frozen_oracle(
                        oracle,
                        vrfm_prediction["decoded_camera_c2w"][
                            overlap, sample, alpha_index
                        ],
                        gt_overlap[overlap],
                    ).rms_translation_error
                )
                random_candidate_rms[overlap, sample, alpha_index] = (
                    evaluate_with_frozen_oracle(
                        oracle,
                        random_prediction["decoded_camera_c2w"][
                            overlap, sample, alpha_index
                        ],
                        gt_overlap[overlap],
                    ).rms_translation_error
                )
    if not np.array_equal(upstream["gt_frame_ids"], source["overlap_frame_ids"]):
        raise ValueError("VRFM privileged frame IDs do not match")
    if not np.array_equal(upstream["gt_c2w"], gt_overlap):
        raise ValueError("VRFM privileged GT poses do not match")
    if not np.allclose(
        upstream["candidate_rms"], vrfm_candidate_check, atol=1e-12, rtol=0
    ):
        raise ValueError("VRFM privileged candidate RMS does not match its prediction")

    denominator = np.maximum(
        baseline_rms[:, None, None], np.finfo(np.float64).eps
    )
    random_relative = (
        baseline_rms[:, None, None] - random_candidate_rms
    ) / denominator
    random_relative[:, :, 0] = 0.0
    vrfm_relative = upstream["relative_improvement"].copy()
    vrfm_relative_check = (
        baseline_rms[:, None, None] - vrfm_candidate_check
    ) / denominator
    vrfm_relative_check[:, :, 0] = 0.0
    if not np.allclose(
        vrfm_relative, vrfm_relative_check, atol=1e-12, rtol=0
    ):
        raise ValueError("VRFM privileged relative improvement is inconsistent")
    paired_advantage = vrfm_relative - random_relative
    arrays = {
        "alphas": alphas.copy(),
        "source_sample_ids": source["sample_ids"].copy(),
        "sample_seeds": random_prediction["sample_seeds"].copy(),
        "gt_frame_ids": source["overlap_frame_ids"].copy(),
        "gt_c2w": gt_overlap,
        "baseline_rms": baseline_rms,
        "vrfm_candidate_rms": upstream["candidate_rms"].copy(),
        "random_candidate_rms": random_candidate_rms,
        "vrfm_relative_improvement": vrfm_relative,
        "random_relative_improvement": random_relative,
        "paired_relative_advantage": paired_advantage,
        "random_prediction_sha256": np.asarray(
            _sha256_file(random_prediction_path), dtype="U64"
        ),
        "vrfm_prediction_sha256": np.asarray(
            vrfm_prediction_digest, dtype="U64"
        ),
        "vrfm_privileged_sha256": np.asarray(
            _sha256_file(vrfm_privileged_path), dtype="U64"
        ),
        "prepared_gt_sha256": np.asarray(prepared_digest, dtype="U64"),
        "oracle_scale": upstream["oracle_scale"].copy(),
        "oracle_rotation": upstream["oracle_rotation"].copy(),
        "oracle_translation": upstream["oracle_translation"].copy(),
        "oracle_digest": upstream["oracle_digest"].copy(),
    }
    _validate_privileged(arrays)
    _atomic_npz(destination, arrays)
    return Path(destination)


def summarize_matched_random_statistics(
    alphas: Sequence[float] | np.ndarray,
    vrfm_relative_improvement: np.ndarray,
    random_relative_improvement: np.ndarray,
    *,
    min_improvement: float = 0.01,
) -> dict[str, object]:
    """Summarize strictly paired sample/alpha cells and overlap-level optima."""
    alpha_values = _normalize_alphas(alphas)
    vrfm = np.asarray(vrfm_relative_improvement, dtype=np.float64)
    random = np.asarray(random_relative_improvement, dtype=np.float64)
    if (
        vrfm.shape != random.shape
        or vrfm.ndim != 3
        or vrfm.shape[0] < 1
        or vrfm.shape[1] < 1
        or vrfm.shape[2] != len(alpha_values)
        or not np.isfinite(vrfm).all()
        or not np.isfinite(random).all()
    ):
        raise ValueError(
            "paired relative improvements must have shape [overlaps, samples, alphas]"
        )
    if not np.isfinite(min_improvement) or min_improvement <= 0.0:
        raise ValueError("min_improvement must be finite and positive")

    paired_advantage = vrfm[:, :, 1:] - random[:, :, 1:]
    vrfm_wins = paired_advantage >= min_improvement
    random_wins = paired_advantage <= -min_improvement
    ties = ~(vrfm_wins | random_wins)
    vrfm_best = np.maximum(0.0, np.max(vrfm[:, :, 1:], axis=(1, 2)))
    random_best = np.maximum(0.0, np.max(random[:, :, 1:], axis=(1, 2)))
    oracle_advantage = vrfm_best - random_best
    vrfm_oracle_wins = oracle_advantage >= min_improvement
    random_oracle_wins = oracle_advantage <= -min_improvement
    oracle_ties = ~(vrfm_oracle_wins | random_oracle_wins)

    vrfm_oracle_count = int(np.count_nonzero(vrfm_oracle_wins))
    random_oracle_count = int(np.count_nonzero(random_oracle_wins))
    if vrfm_oracle_count > random_oracle_count:
        diagnosis = "VRFM_DIRECTIONS_OUTPERFORM_MATCHED_RANDOM"
    elif random_oracle_count > vrfm_oracle_count:
        diagnosis = "MATCHED_RANDOM_OUTPERFORMS_VRFM_DIRECTIONS"
    else:
        diagnosis = "VRFM_AND_MATCHED_RANDOM_TIED"

    return {
        "paired_nonzero_cell_count": int(paired_advantage.size),
        "vrfm_paired_win_count": int(np.count_nonzero(vrfm_wins)),
        "random_paired_win_count": int(np.count_nonzero(random_wins)),
        "paired_tie_count": int(np.count_nonzero(ties)),
        "median_paired_relative_advantage": float(np.median(paired_advantage)),
        "vrfm_oracle_best_win_overlap_count": vrfm_oracle_count,
        "random_oracle_best_win_overlap_count": random_oracle_count,
        "oracle_best_tie_overlap_count": int(np.count_nonzero(oracle_ties)),
        "median_oracle_best_relative_advantage": float(
            np.median(oracle_advantage)
        ),
        "min_improvement": float(min_improvement),
        "diagnosis": diagnosis,
    }


def write_matched_random_report(
    sidecar_paths: Sequence[Path],
    destination: Path,
    *,
    min_improvement: float = 0.01,
) -> dict[str, object]:
    """Write the paired one-null pilot report without overstating attribution."""
    if not sidecar_paths:
        raise ValueError("at least one matched-random privileged sidecar is required")
    loaded = [load_matched_random_privileged(path) for path in sidecar_paths]
    alphas = loaded[0]["alphas"]
    samples = loaded[0]["random_candidate_rms"].shape[1]
    if any(not np.array_equal(arrays["alphas"], alphas) for arrays in loaded[1:]):
        raise ValueError("matched-random sidecars use different alpha grids")
    if any(
        arrays["random_candidate_rms"].shape[1] != samples
        for arrays in loaded[1:]
    ):
        raise ValueError("matched-random sidecars use different sample counts")

    per_scene: list[dict[str, object]] = []
    scene_effects: list[float] = []
    scene_names: list[str] = []
    for arrays in loaded:
        scene = str(arrays["source_sample_ids"][0]).split(":", 1)[0]
        scene_names.append(scene)
        vrfm = arrays["vrfm_relative_improvement"]
        random = arrays["random_relative_improvement"]
        vrfm_best = np.maximum(0.0, np.max(vrfm[:, :, 1:], axis=(1, 2)))
        random_best = np.maximum(0.0, np.max(random[:, :, 1:], axis=(1, 2)))
        overlap_advantage = vrfm_best - random_best
        scene_effect = float(np.median(overlap_advantage))
        scene_effects.append(scene_effect)
        per_scene.append(
            {
                "scene": scene,
                "scene_median_overlap_oracle_best_relative_advantage": scene_effect,
                "paired_comparison": summarize_matched_random_statistics(
                    alphas,
                    vrfm,
                    random,
                    min_improvement=min_improvement,
                ),
                "vrfm_summary": summarize_vrfm_residual_statistics(
                    alphas,
                    vrfm,
                    min_improvement=min_improvement,
                ),
                "matched_random_summary": summarize_vrfm_residual_statistics(
                    alphas,
                    random,
                    min_improvement=min_improvement,
                ),
            }
        )
    if len(set(scene_names)) != len(scene_names):
        raise ValueError("matched-random report scene identities must be unique")

    vrfm_all = np.concatenate(
        [arrays["vrfm_relative_improvement"] for arrays in loaded], axis=0
    )
    random_all = np.concatenate(
        [arrays["random_relative_improvement"] for arrays in loaded], axis=0
    )
    scene_effect_array = np.asarray(scene_effects, dtype=np.float64)
    vrfm_scene_wins = scene_effect_array >= min_improvement
    random_scene_wins = scene_effect_array <= -min_improvement
    scene_ties = ~(vrfm_scene_wins | random_scene_wins)
    vrfm_scene_win_count = int(np.count_nonzero(vrfm_scene_wins))
    random_scene_win_count = int(np.count_nonzero(random_scene_wins))
    if vrfm_scene_win_count > random_scene_win_count:
        diagnosis = "VRFM_DIRECTIONS_OUTPERFORM_MATCHED_RANDOM"
    elif random_scene_win_count > vrfm_scene_win_count:
        diagnosis = "MATCHED_RANDOM_OUTPERFORMS_VRFM_DIRECTIONS"
    else:
        diagnosis = "VRFM_AND_MATCHED_RANDOM_TIED"
    report: dict[str, object] = {
        "schema": "variational_camera_latent.matched_random_vs_vrfm_pilot_report.v1",
        "scene_count": len(loaded),
        "min_improvement": float(min_improvement),
        "inference_unit": "scene",
        "structured_null_replicate_count": 1,
        "formal_training_attribution": False,
        "formal_scene_level_p_value_available": False,
        "calibration_set_only": True,
        "oracle_upper_bound": True,
        "selection_uses_privileged_labels": True,
        "prediction_data_contains_privileged_labels": False,
        "primary_metric": "frozen_scene_sim3_rms_translation_error",
        "claim_scope": (
            "Directional-structure pilot on the calibration set using one shared "
            "orthogonal structured-null transform; not formal learned-signal "
            "attribution and not a deployable selector"
        ),
        "control_preserves_feature_row_gram_geometry": True,
        "same_control_transform_across_scenes": True,
        "matched_oracle_budget": {
            "directions_per_overlap": int(samples),
            "nonzero_alpha_count": int(len(alphas) - 1),
            "no_op_count_per_overlap": 1,
            "alphas": alphas.tolist(),
        },
        "required_followups_for_formal_attribution": [
            "at_least_20_independent_structured_null_replicates",
            "norm_matched_untrained_model_control",
            "in_basis_residual_derangement_control",
        ],
        "scene_level_statistic": (
            "mean_of_scene_median_overlap_oracle_best_relative_advantage"
        ),
        "scene_level_inference_limitation": (
            "One shared structured-null transform and one calibration set do not "
            "identify a formal null distribution; scene-level values are descriptive"
        ),
        "scene_level_observed_mean_relative_advantage": float(
            np.mean(scene_effect_array)
        ),
        "scene_level_observed_median_relative_advantage": float(
            np.median(scene_effect_array)
        ),
        "scene_level_vrfm_win_count": vrfm_scene_win_count,
        "scene_level_random_win_count": random_scene_win_count,
        "scene_level_tie_count": int(np.count_nonzero(scene_ties)),
        "diagnosis_basis": "scene_level",
        "diagnosis_rule": (
            "Compare counts of scenes whose median overlap oracle-best advantage "
            "exceeds the signed minimum-improvement threshold"
        ),
        "diagnosis": diagnosis,
        "per_scene": per_scene,
        "paired_comparison": summarize_matched_random_statistics(
            alphas,
            vrfm_all,
            random_all,
            min_improvement=min_improvement,
        ),
        "vrfm_summary": summarize_vrfm_residual_statistics(
            alphas,
            vrfm_all,
            min_improvement=min_improvement,
        ),
        "matched_random_summary": summarize_vrfm_residual_statistics(
            alphas,
            random_all,
            min_improvement=min_improvement,
        ),
    }
    destination = Path(destination)
    if destination.is_file():
        try:
            existing = json.loads(destination.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("existing matched-random report is invalid") from error
        if existing != report:
            raise ValueError("existing matched-random report differs")
        return report
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(destination)
    return report
