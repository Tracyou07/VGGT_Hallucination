from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .camera import decode_camera_tokens
from .clustering import two_means
from .contracts import CandidateShardRecord
from .flow import heun_sample
from .model import DeterministicRFMModel, VRFMModel
from .source import load_source_shard


_REQUIRED = {
    "z",
    "corrected_camera_tokens",
    "latent_cluster_ids",
    "latent_cluster_centers",
    "source_long_tokens",
    "source_sample_ids",
    "span_starts",
    "sample_seeds",
    "checkpoint_sha256",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_candidate_arrays(arrays: dict[str, np.ndarray]) -> None:
    missing = _REQUIRED - set(arrays)
    extra = set(arrays) - _REQUIRED - {"decoded_camera_raw"}
    if missing or extra:
        raise ValueError(f"candidate members mismatch; missing={sorted(missing)}, extra={sorted(extra)}")
    if any(np.asarray(value).dtype.hasobject for value in arrays.values()):
        raise ValueError("candidate shard may not contain object arrays")
    z = arrays["z"]
    corrected = arrays["corrected_camera_tokens"]
    if z.ndim != 3 or z.shape[0] != 8 or z.shape[1] < 1:
        raise ValueError("z must have shape [8, samples, z_dim]")
    if corrected.shape != (8, z.shape[1], 50, 2048):
        raise ValueError("corrected_camera_tokens has an invalid shape")
    if arrays["latent_cluster_ids"].shape != z.shape[:2]:
        raise ValueError("latent_cluster_ids must have shape [8, samples]")
    if arrays["latent_cluster_centers"].shape != (8, 2, 50, 2048):
        raise ValueError("latent_cluster_centers has an invalid shape")
    if arrays["source_long_tokens"].shape != (8, 50, 2048):
        raise ValueError("source_long_tokens has an invalid shape")
    if arrays["source_sample_ids"].shape != (8,) or arrays["source_sample_ids"].dtype.kind != "U":
        raise ValueError("source_sample_ids must be a Unicode vector [8]")
    if arrays["span_starts"].shape != (8,) or arrays["sample_seeds"].shape != z.shape[:2]:
        raise ValueError("candidate span or seed metadata has an invalid shape")
    if arrays["checkpoint_sha256"].shape != () or arrays["checkpoint_sha256"].dtype.kind != "U":
        raise ValueError("checkpoint_sha256 must be a Unicode scalar")
    if "decoded_camera_raw" in arrays and arrays["decoded_camera_raw"].shape != (8, z.shape[1], 50, 9):
        raise ValueError("decoded_camera_raw has an invalid shape")
    for name, value in arrays.items():
        array = np.asarray(value)
        if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
            raise ValueError(f"candidate member {name} contains non-finite values")


def _save_candidate_shard(path: Path, arrays: dict[str, np.ndarray]) -> None:
    _validate_candidate_arrays(arrays)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def load_candidate_shard(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(Path(path), allow_pickle=False) as archive:
            arrays = {name: np.asarray(archive[name]).copy() for name in archive.files}
    except (OSError, ValueError, KeyError) as error:
        raise ValueError(f"invalid candidate shard: {path}") from error
    _validate_candidate_arrays(arrays)
    return arrays


def _load_model(checkpoint_path: Path, device: torch.device) -> VRFMModel:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    try:
        config = checkpoint["model_config"]
        model = VRFMModel(
            d_model=int(config["d_model"]),
            z_dim=int(config["z_dim"]),
            layers=int(config["layers"]),
            heads=int(config["heads"]),
        ).to(device)
        model.load_state_dict(checkpoint["vrfm"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("checkpoint lacks a valid VRFM model configuration") from error
    return model.eval()


def _load_deterministic_model(
    checkpoint_path: Path, device: torch.device
) -> DeterministicRFMModel:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    try:
        config = checkpoint["model_config"]
        model = DeterministicRFMModel(
            d_model=int(config["d_model"]),
            layers=int(config["layers"]),
            heads=int(config["heads"]),
        ).to(device)
        model.load_state_dict(checkpoint["deterministic"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("checkpoint lacks a valid deterministic model configuration") from error
    return model.eval()


def _heun_deterministic(
    model: DeterministicRFMModel,
    x0: torch.Tensor,
    context: torch.Tensor,
    span: torch.Tensor,
    *,
    steps: int,
) -> torch.Tensor:
    state = x0
    dt = 1.0 / steps
    for index in range(steps):
        t0 = torch.full((x0.shape[0],), index * dt, device=x0.device, dtype=x0.dtype)
        first = model(state, t0, context, span)
        proposal = state + dt * first
        second = model(proposal, t0 + dt, context, span)
        state = state + 0.5 * dt * (first + second)
    return state


def generate_scene_candidates(
    source_path: Path,
    checkpoint_path: Path,
    destination: Path,
    *,
    samples: int = 32,
    steps: int = 16,
    seed: int = 20260827,
    device: str = "cuda",
    camera_head: Any | None = None,
) -> CandidateShardRecord:
    """Sample raw prediction-only trajectories; no GT argument is accepted."""
    if samples < 2 or steps < 1:
        raise ValueError("samples must be at least two and steps must be positive")
    torch_device = torch.device(device)
    source = load_source_shard(source_path)
    model = _load_model(Path(checkpoint_path), torch_device)
    z_dim = int(model.z_dim)
    corrected = np.empty((8, samples, 50, 2048), dtype=np.float32)
    z_values = np.empty((8, samples, z_dim), dtype=np.float32)
    sample_seeds = np.empty((8, samples), dtype=np.int64)
    cluster_ids = np.empty((8, samples), dtype=np.int64)
    centers = np.empty((8, 2, 50, 2048), dtype=np.float32)
    decoded = None if camera_head is None else np.empty((8, samples, 50, 9), dtype=np.float32)

    with torch.no_grad():
        for overlap in range(8):
            x0 = torch.from_numpy(source["overlap_long_tokens"][overlap]).to(torch_device)
            context = torch.from_numpy(source["global_camera_tokens"]).to(torch_device)
            x0_batch = x0[None].expand(samples, -1, -1)
            context_batch = context[None].expand(samples, -1, -1)
            span = torch.full(
                (samples,), int(source["span_starts"][overlap]), device=torch_device, dtype=torch.long
            )
            rows: list[torch.Tensor] = []
            for sample in range(samples):
                sample_seed = int(seed + overlap * 100_000 + sample)
                sample_seeds[overlap, sample] = sample_seed
                generator = torch.Generator(device=torch_device).manual_seed(sample_seed)
                rows.append(torch.randn((1, z_dim), generator=generator, device=torch_device))
            z = torch.cat(rows, dim=0)
            output = heun_sample(model, x0_batch, context_batch, span, z, steps=steps)
            output_array = output.float().cpu().numpy()
            corrected[overlap] = output_array
            z_values[overlap] = z.float().cpu().numpy()
            features = (output_array - source["overlap_long_tokens"][overlap][None]).reshape(samples, -1)
            clustered = two_means(features)
            cluster_ids[overlap] = clustered.labels
            centers[overlap] = clustered.centers.reshape(2, 50, 2048)
            if decoded is not None:
                decoded[overlap] = decode_camera_tokens(camera_head, output).float().cpu().numpy()

    arrays: dict[str, np.ndarray] = {
        "z": z_values,
        "corrected_camera_tokens": corrected,
        "latent_cluster_ids": cluster_ids,
        "latent_cluster_centers": centers,
        "source_long_tokens": source["overlap_long_tokens"].astype(np.float32),
        "source_sample_ids": source["sample_ids"].copy(),
        "span_starts": source["span_starts"].copy(),
        "sample_seeds": sample_seeds,
        "checkpoint_sha256": np.asarray(_sha256_file(Path(checkpoint_path)), dtype="U64"),
    }
    if decoded is not None:
        arrays["decoded_camera_raw"] = decoded
    destination = Path(destination)
    _save_candidate_shard(destination, arrays)
    scene = str(source["sample_ids"][0]).split(":", 1)[0]
    return CandidateShardRecord(scene, destination, 8, samples, _sha256_file(destination))


def generate_deterministic_candidates(
    source_path: Path,
    checkpoint_path: Path,
    destination: Path,
    *,
    steps: int = 16,
    device: str = "cuda",
    camera_head: Any | None = None,
) -> CandidateShardRecord:
    """Export one prediction-only trajectory per overlap without artificial z."""
    if steps < 1:
        raise ValueError("steps must be positive")
    torch_device = torch.device(device)
    source = load_source_shard(source_path)
    model = _load_deterministic_model(Path(checkpoint_path), torch_device)
    x0 = torch.from_numpy(source["overlap_long_tokens"]).to(torch_device)
    context = torch.from_numpy(source["global_camera_tokens"]).to(torch_device)
    context = context[None].expand(8, -1, -1)
    span = torch.from_numpy(source["span_starts"]).to(torch_device)
    with torch.no_grad():
        output = _heun_deterministic(model, x0, context, span, steps=steps)
        output_array = output.float().cpu().numpy()
        decoded = (
            None
            if camera_head is None
            else decode_camera_tokens(camera_head, output).float().cpu().numpy()
        )
    arrays: dict[str, np.ndarray] = {
        "corrected_camera_tokens": output_array,
        "source_long_tokens": source["overlap_long_tokens"].astype(np.float32),
        "source_sample_ids": source["sample_ids"].copy(),
        "span_starts": source["span_starts"].copy(),
        "checkpoint_sha256": np.asarray(_sha256_file(Path(checkpoint_path)), dtype="U64"),
    }
    if decoded is not None:
        arrays["decoded_camera_raw"] = decoded
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(destination)
    scene = str(source["sample_ids"][0]).split(":", 1)[0]
    return CandidateShardRecord(scene, destination, 8, 1, _sha256_file(destination))


def analyze_candidate_shard(path: Path) -> dict[str, object]:
    """Replay latent clustering using only the published candidate shard."""
    arrays = load_candidate_shard(path)
    analyses: list[dict[str, object]] = []
    for overlap in range(8):
        features = (
            arrays["corrected_camera_tokens"][overlap]
            - arrays["source_long_tokens"][overlap][None]
        ).reshape(arrays["z"].shape[1], -1)
        result = two_means(features)
        analyses.append(
            {
                "overlap": overlap,
                "labels": result.labels.tolist(),
                "counts": [int(np.sum(result.labels == index)) for index in (0, 1)],
                "one_to_two_sse_ratio": result.one_to_two_sse_ratio,
                "center_distance": float(np.linalg.norm(result.centers[0] - result.centers[1])),
            }
        )
    return {"schema": "variational_camera_latent.candidate_analysis.v1", "overlaps": analyses}
