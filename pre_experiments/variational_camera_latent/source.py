from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from pre_experiments.camera_velocity_ambiguity_02.artifacts import (
    PREDICTION_SCHEMA,
    PredictionIdentity,
    load_completed_prediction,
)

from .contracts import SourceShardRecord
from .schema import SOURCE_SCHEMA, validate_source_shard


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity_from_completion(path: Path) -> PredictionIdentity:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid authenticated prediction sidecar: {path}") from error
    if not isinstance(payload, dict) or payload.get("schema") != PREDICTION_SCHEMA:
        raise ValueError(f"invalid authenticated prediction schema: {path}")
    fields = (
        "run_id",
        "scene",
        "artifact_kind",
        "window_index",
        "frame_digest",
        "checkpoint_sha256",
        "git_commit",
        "preprocess",
        "camera_iterations",
        "protocol_digest",
    )
    try:
        return PredictionIdentity(**{name: payload[name] for name in fields})
    except (KeyError, TypeError) as error:
        raise ValueError(f"prediction sidecar has malformed identity: {path}") from error


def _load_completed(directory: Path) -> tuple[dict[str, np.ndarray], PredictionIdentity]:
    directory = Path(directory)
    artifact = directory / "prediction.npz"
    completion = directory / "complete.json"
    identity = _identity_from_completion(completion)
    arrays = load_completed_prediction(artifact, completion, identity)
    return arrays, identity


def save_source_shard(path: Path, arrays: dict[str, np.ndarray]) -> None:
    normalized = {name: np.asarray(value) for name, value in arrays.items()}
    validate_source_shard(normalized)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **normalized)
    temporary.replace(path)


def load_source_shard(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(Path(path), allow_pickle=False) as archive:
            arrays = {name: np.asarray(archive[name]).copy() for name in archive.files}
    except (OSError, ValueError, KeyError) as error:
        raise ValueError(f"invalid source shard: {path}; object arrays are forbidden") from error
    validate_source_shard(arrays)
    return arrays


def build_scene_source_shard(
    prediction_root: Path,
    destination: Path,
    *,
    role: str,
) -> SourceShardRecord:
    """Build eight aligned prediction-only overlap examples for one scene."""
    if role not in {"train", "validation", "smoke"}:
        raise ValueError("role must be train, validation, or smoke")
    root = Path(prediction_root)
    global_arrays, global_identity = _load_completed(root / "global")
    local_directories = sorted((root / "local").glob("window_*"))
    if len(local_directories) != 9:
        raise ValueError("exactly nine authenticated short-window predictions are required")

    locals_: list[dict[str, np.ndarray]] = []
    for index, directory in enumerate(local_directories):
        arrays, identity = _load_completed(directory)
        if identity.scene != global_identity.scene or identity.run_id != global_identity.run_id:
            raise ValueError("global and local predictions must have the same scene and run identity")
        if identity.artifact_kind != "local" or identity.window_index != index:
            raise ValueError("local prediction window identity or ordering is invalid")
        if (
            identity.checkpoint_sha256 != global_identity.checkpoint_sha256
            or identity.protocol_digest != global_identity.protocol_digest
            or identity.git_commit != global_identity.git_commit
        ):
            raise ValueError("global and local prediction provenance must match")
        locals_.append(arrays)

    global_ids = global_arrays["frame_ids"].astype(np.int64, copy=False)
    global_tokens = global_arrays["normalized_camera_tokens"].astype(np.float32, copy=False)
    short_ids = np.stack([arrays["frame_ids"] for arrays in locals_]).astype(np.int64)
    short_tokens = np.stack(
        [arrays["normalized_camera_tokens"] for arrays in locals_]
    ).astype(np.float32)
    span_starts = np.arange(0, 400, 50, dtype=np.int64)
    overlap_ids = np.stack(
        [global_ids[start + 50 : start + 100] for start in span_starts]
    )
    overlap_long = np.stack(
        [global_tokens[start + 50 : start + 100] for start in span_starts]
    )
    overlap_left = short_tokens[:-1, 50:].copy()
    overlap_right = short_tokens[1:, :50].copy()
    sample_ids = np.asarray(
        [f"{global_identity.scene}:overlap_{index:03d}" for index in range(8)],
        dtype="U64",
    )
    arrays = {
        "global_frame_ids": global_ids,
        "global_camera_tokens": global_tokens,
        "short_frame_ids": short_ids,
        "short_camera_tokens": short_tokens,
        "overlap_frame_ids": overlap_ids,
        "overlap_long_tokens": overlap_long,
        "overlap_left_tokens": overlap_left,
        "overlap_right_tokens": overlap_right,
        "span_starts": span_starts,
        "sample_ids": sample_ids,
        "global_pred_c2w": global_arrays["pred_c2w_raw"].astype(np.float64),
        "overlap_long_c2w": np.stack(
            [global_arrays["pred_c2w_raw"][start + 50 : start + 100] for start in span_starts]
        ).astype(np.float64),
    }
    save_source_shard(destination, arrays)
    return SourceShardRecord(
        scene=global_identity.scene,
        role=role,
        path=Path(destination),
        overlap_count=8,
        sha256=_sha256_file(Path(destination)),
    )


def write_source_manifest(
    path: Path,
    *,
    dataset_root: Path,
    records: Sequence[SourceShardRecord],
    source_run_digest: str,
) -> dict[str, object]:
    if len(source_run_digest) != 64:
        raise ValueError("source_run_digest must be a SHA-256 digest")
    payload: dict[str, object] = {
        "schema": SOURCE_SCHEMA,
        "dataset_root": str(Path(dataset_root)),
        "source_run_digest": source_run_digest,
        "records": [
            {**asdict(record), "path": str(record.path)} for record in records
        ],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    return payload
