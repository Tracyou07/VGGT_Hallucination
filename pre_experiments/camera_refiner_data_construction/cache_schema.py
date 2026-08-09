"""Authenticated shard contract consumed by the full-hidden latent refiner."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


SCHEMA_VERSION = 2
STUDY_TYPE = "full_hidden_sequence_refiner"
REQUIRED_MEMBERS = {
    "scene_names",
    "clip_ids",
    "long_hidden",
    "camera_tokens",
    "baseline_raw_pose",
    "baseline_pose",
    "short_pose",
    "diagnostics",
    "frame_ids",
    "starts",
    "gt_c2w_raw",
    "short_pose_observations",
    "short_frame_indices",
    "short_observation_count",
    "selected_short_window",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(destination)


def _validate_pose(name: str, value: np.ndarray, prefix: tuple[int, ...]) -> None:
    if value.shape != (*prefix, 9) or not np.issubdtype(value.dtype, np.floating):
        raise ValueError(f"{name} must have shape {(*prefix, 9)}")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")
    if np.any(np.linalg.norm(value[..., 3:7], axis=-1) <= 1e-8):
        raise ValueError(f"{name} contains a zero quaternion")
    if np.any(value[..., 7:] < 0):
        raise ValueError(f"{name} contains negative FOV values")


def validate_sequence_shard(arrays: Mapping[str, np.ndarray]) -> None:
    if "short_hidden" in arrays:
        raise ValueError("short_hidden is forbidden as a training label")
    missing = REQUIRED_MEMBERS - set(arrays)
    if missing:
        raise ValueError(f"sequence shard is missing members: {sorted(missing)}")
    if any(np.asarray(value).dtype.hasobject for value in arrays.values()):
        raise ValueError("sequence shards may not contain object arrays")

    hidden = np.asarray(arrays["long_hidden"])
    if hidden.ndim != 3 or hidden.shape[2] != 1024 or not np.issubdtype(
        hidden.dtype, np.floating
    ):
        raise ValueError("long_hidden must have shape [N, S, 1024]")
    sample_count, frame_count = hidden.shape[:2]
    if sample_count < 1 or frame_count < 2 or not np.isfinite(hidden).all():
        raise ValueError("long_hidden must contain finite non-empty windows")
    prefix = (sample_count, frame_count)

    tokens = np.asarray(arrays["camera_tokens"])
    if tokens.shape != (*prefix, 2048) or not np.issubdtype(tokens.dtype, np.floating):
        raise ValueError("camera_tokens must have shape [N, S, 2048]")
    if not np.isfinite(tokens).all():
        raise ValueError("camera_tokens must contain only finite values")

    raw_pose = np.asarray(arrays["baseline_raw_pose"])
    baseline = np.asarray(arrays["baseline_pose"])
    short_pose = np.asarray(arrays["short_pose"])
    _validate_pose("baseline_pose", baseline, prefix)
    _validate_pose("short_pose", short_pose, prefix)
    if raw_pose.shape != (*prefix, 9) or not np.isfinite(raw_pose).all():
        raise ValueError("baseline_raw_pose must have shape [N, S, 9]")
    expected_baseline = raw_pose.copy()
    expected_baseline[..., 7:] = np.maximum(expected_baseline[..., 7:], 0)
    if not np.allclose(baseline, expected_baseline, rtol=1e-5, atol=1e-6):
        raise ValueError("baseline_pose must be the activated baseline_raw_pose")

    diagnostics = np.asarray(arrays["diagnostics"])
    if diagnostics.ndim != 3 or diagnostics.shape[:2] != prefix or diagnostics.shape[2] < 1:
        raise ValueError("diagnostics must have shape [N, S, I]")
    if not np.isfinite(diagnostics).all():
        raise ValueError("diagnostics must contain only finite values")

    frame_ids = np.asarray(arrays["frame_ids"])
    starts = np.asarray(arrays["starts"])
    if frame_ids.shape != prefix or not np.issubdtype(frame_ids.dtype, np.integer):
        raise ValueError("frame_ids must have shape [N, S]")
    if np.any(np.diff(frame_ids, axis=1) <= 0):
        raise ValueError("frame IDs must be strictly increasing")
    if starts.shape != (sample_count,) or np.any(starts < 0):
        raise ValueError("starts must contain one non-negative value per sample")

    scene_names = np.asarray(arrays["scene_names"]).astype(str)
    clip_ids = np.asarray(arrays["clip_ids"]).astype(str)
    if scene_names.shape != (sample_count,) or any(not value for value in scene_names):
        raise ValueError("scene_names must identify every sample")
    if (
        clip_ids.shape != (sample_count,)
        or any(not value for value in clip_ids)
        or len(set(clip_ids.tolist())) != sample_count
    ):
        raise ValueError("clip_ids must be non-empty and unique within a shard")

    gt_c2w = np.asarray(arrays["gt_c2w_raw"])
    if gt_c2w.shape != (*prefix, 4, 4) or not np.isfinite(gt_c2w).all():
        raise ValueError("gt_c2w_raw must have shape [N, S, 4, 4]")
    if not np.allclose(gt_c2w[..., 3, :], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise ValueError("gt_c2w_raw has invalid homogeneous rows")

    observations = np.asarray(arrays["short_pose_observations"])
    indices = np.asarray(arrays["short_frame_indices"])
    if observations.ndim != 4 or observations.shape[0] != sample_count:
        raise ValueError("short_pose_observations must have shape [N, K, L, 9]")
    short_prefix = observations.shape[:-1]
    _validate_pose("short_pose_observations", observations, short_prefix)
    if indices.shape != short_prefix or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("short_frame_indices must have shape [N, K, L]")
    if (
        np.any(indices < 0)
        or np.any(indices >= frame_count)
        or np.any(np.diff(indices, axis=2) <= 0)
    ):
        raise ValueError("short_frame_indices are invalid")
    observation_count = np.asarray(arrays["short_observation_count"])
    selected = np.asarray(arrays["selected_short_window"])
    if observation_count.shape != prefix or np.any(observation_count < 1):
        raise ValueError("short_observation_count must cover every frame")
    if selected.shape != prefix or np.any(selected < 0) or np.any(selected >= observations.shape[1]):
        raise ValueError("selected_short_window is invalid")


def save_sequence_shard(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    normalized = {name: np.asarray(value) for name, value in arrays.items()}
    validate_sequence_shard(normalized)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **normalized)
    temporary.replace(destination)


def load_sequence_shard(path: Path) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    validate_sequence_shard(arrays)
    return arrays


def _relative_file(path: Path, root: Path, description: str) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError(f"{description} must be inside dataset_root") from error


def write_sequence_manifest(
    path: Path,
    *,
    dataset_root: Path,
    projection_path: Path,
    shard_records: Sequence[Mapping[str, object]],
    camera_iterations: int,
    source_manifest_digest: str,
) -> dict[str, object]:
    root = Path(dataset_root).resolve()
    if camera_iterations < 1:
        raise ValueError("camera_iterations must be positive")
    if len(source_manifest_digest) != 64:
        raise ValueError("source_manifest_digest must be a SHA-256 value")
    projection = Path(projection_path).resolve()
    if not projection.is_file():
        raise FileNotFoundError(f"missing pose projection: {projection}")
    projection_value = np.load(projection, allow_pickle=False)
    if projection_value.shape != (9, 1024) or not np.isfinite(projection_value).all():
        raise ValueError("pose projection must have finite shape [9, 1024]")
    if not shard_records:
        raise ValueError("sequence manifest requires at least one shard")

    records = []
    identities = set()
    total_samples = 0
    for source_record in shard_records:
        shard_path = Path(source_record["path"]).resolve()
        role = str(source_record.get("role", ""))
        scene = str(source_record.get("scene", ""))
        sample_count = int(source_record.get("sample_count", 0))
        if role not in {"train", "validation"} or not scene or sample_count < 1:
            raise ValueError("shard role, scene, and sample_count are invalid")
        if scene in identities:
            raise ValueError(f"duplicate sequence shard: {scene}")
        identities.add(scene)
        arrays = load_sequence_shard(shard_path)
        if len(arrays["long_hidden"]) != sample_count:
            raise ValueError(f"sample count mismatch for {scene}")
        records.append(
            {
                "path": _relative_file(shard_path, root, "sequence shard"),
                "role": role,
                "scene": scene,
                "sample_count": sample_count,
                "sha256": sha256_file(shard_path),
            }
        )
        total_samples += sample_count
    records.sort(key=lambda item: (str(item["role"]), str(item["scene"])))
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "study_type": STUDY_TYPE,
        "source": "co3d",
        "pose_encoding": "absT_quaR_FoV",
        "camera_iterations": camera_iterations,
        "source_manifest_digest": source_manifest_digest,
        "sample_count": total_samples,
        "pose_projection": {
            "path": _relative_file(projection, root, "pose projection"),
            "sha256": sha256_file(projection),
        },
        "shards": records,
    }
    _atomic_json(Path(path), payload)
    return payload
