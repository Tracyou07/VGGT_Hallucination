from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
from typing import Mapping, Sequence

import numpy as np

from pre_experiments.variational_camera_latent.candidates import load_candidate_shard
from pre_experiments.variational_camera_latent.source import load_source_shard

from .contracts import LongContextRecord


LONG_CONTEXT_SCHEMA = "variational_camera_selector.long_context.v1"
PREDICTION_BINDING_SCHEMA = "variational_camera_selector.prediction_binding_manifest.v1"
LONG_CONTEXT_MEMBERS = {
    "global_frame_ids",
    "global_camera_tokens",
    "overlap_frame_ids",
    "overlap_long_tokens",
    "span_starts",
    "source_sample_ids",
    "scene",
    "role",
    "source_shard_sha256",
    "candidate_shard_sha256",
    "producer_git_commit",
}
_FORBIDDEN_NAME_PARTS = (
    "short",
    "teacher",
    "gt",
    "ground_truth",
    "privileged",
    "depth",
    "quality",
    "error",
)
_ROLES = {"train", "validation", "smoke"}
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _scalar_text(value: np.ndarray, label: str) -> str:
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind != "U":
        raise ValueError(f"{label} must be a Unicode scalar")
    return str(array)


def _validate_digest(value: str, label: str) -> None:
    if _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")


def _validate_commit(value: str) -> None:
    if _COMMIT_RE.fullmatch(value) is None:
        raise ValueError("producer_git_commit must be a lowercase 40-character git commit")


def validate_long_context_shard(arrays: Mapping[str, np.ndarray]) -> None:
    names = set(arrays)
    forbidden = sorted(
        name for name in names if any(part in name.lower() for part in _FORBIDDEN_NAME_PARTS)
    )
    if forbidden:
        raise ValueError(f"long-context shard contains forbidden members: {forbidden}")
    if names != LONG_CONTEXT_MEMBERS:
        raise ValueError(
            "long-context shard members mismatch; "
            f"missing={sorted(LONG_CONTEXT_MEMBERS - names)}, "
            f"extra={sorted(names - LONG_CONTEXT_MEMBERS)}"
        )

    normalized = {name: np.asarray(value) for name, value in arrays.items()}
    if any(value.dtype.hasobject for value in normalized.values()):
        raise ValueError("long-context shard may not contain object arrays")

    global_ids = normalized["global_frame_ids"]
    overlap_ids = normalized["overlap_frame_ids"]
    global_tokens = normalized["global_camera_tokens"]
    overlap_tokens = normalized["overlap_long_tokens"]
    span_starts = normalized["span_starts"]
    sample_ids = normalized["source_sample_ids"]

    if global_ids.shape != (500,) or not np.issubdtype(global_ids.dtype, np.integer):
        raise ValueError("global_frame_ids must be an integer vector [500]")
    if overlap_ids.shape != (8, 50) or not np.issubdtype(overlap_ids.dtype, np.integer):
        raise ValueError("overlap_frame_ids must be an integer matrix [8, 50]")
    if np.any(global_ids[1:] <= global_ids[:-1]):
        raise ValueError("global_frame_ids must be strictly increasing")
    if any(np.any(row[1:] <= row[:-1]) for row in overlap_ids):
        raise ValueError("overlap frame IDs must be strictly increasing")

    for name, expected_shape in (
        ("global_camera_tokens", (500, 2048)),
        ("overlap_long_tokens", (8, 50, 2048)),
    ):
        value = normalized[name]
        if value.shape != expected_shape or value.dtype != np.float32:
            raise ValueError(f"{name} must contain float32 values with shape {expected_shape}")
        if not np.isfinite(value).all():
            raise ValueError(f"{name} must contain finite values")

    expected_starts = np.arange(0, 400, 50, dtype=np.int64)
    if span_starts.shape != (8,) or not np.array_equal(span_starts, expected_starts):
        raise ValueError("span_starts must be [0, 50, ..., 350]")
    if sample_ids.shape != (8,) or sample_ids.dtype.kind != "U":
        raise ValueError("source_sample_ids must be a Unicode vector [8]")
    if len(set(sample_ids.tolist())) != 8 or any(not value for value in sample_ids.tolist()):
        raise ValueError("source sample IDs must be non-empty and unique")

    scene = _scalar_text(normalized["scene"], "scene")
    if not scene or any(not value.startswith(scene + ":") for value in sample_ids.tolist()):
        raise ValueError("source sample IDs must belong to the declared scene")
    role = _scalar_text(normalized["role"], "role")
    if role not in _ROLES:
        raise ValueError("role must be train, validation, or smoke")
    _validate_digest(_scalar_text(normalized["source_shard_sha256"], "source digest"), "source digest")
    _validate_digest(
        _scalar_text(normalized["candidate_shard_sha256"], "candidate digest"),
        "candidate digest",
    )
    _validate_commit(_scalar_text(normalized["producer_git_commit"], "producer commit"))

    for index, start in enumerate(range(50, 401, 50)):
        if not np.array_equal(overlap_ids[index], global_ids[start : start + 50]):
            raise ValueError(f"overlap {index} frame IDs do not align with global context")
        if not np.array_equal(overlap_tokens[index], global_tokens[start : start + 50]):
            raise ValueError(f"overlap {index} tokens do not align with global context")


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _atomic_json(path: Path, payload: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_long_context_shard(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(Path(path), allow_pickle=False) as archive:
            arrays = {name: np.asarray(archive[name]).copy() for name in archive.files}
    except (OSError, ValueError, KeyError) as error:
        raise ValueError(f"invalid long-context shard: {path}; object arrays are forbidden") from error
    validate_long_context_shard(arrays)
    return arrays


def build_long_context_shard(
    source_path: Path,
    candidate_path: Path,
    destination: Path,
    *,
    role: str,
    producer_git_commit: str,
) -> LongContextRecord:
    if role not in _ROLES:
        raise ValueError("role must be train, validation, or smoke")
    _validate_commit(producer_git_commit)
    source_path = Path(source_path)
    candidate_path = Path(candidate_path)
    source = load_source_shard(source_path)
    candidate = load_candidate_shard(candidate_path)
    if not np.array_equal(candidate["source_sample_ids"], source["sample_ids"]):
        raise ValueError("candidate and source sample IDs do not match")
    if not np.array_equal(candidate["span_starts"], source["span_starts"]):
        raise ValueError("candidate and source spans do not match")
    if not np.array_equal(candidate["source_long_tokens"], source["overlap_long_tokens"]):
        raise ValueError("candidate source-long tokens do not match the source shard")

    scene_names = {str(value).split(":", 1)[0] for value in source["sample_ids"].tolist()}
    if len(scene_names) != 1:
        raise ValueError("source sample IDs must identify exactly one scene")
    scene = next(iter(scene_names))
    source_sha256 = _sha256_file(source_path)
    candidate_sha256 = _sha256_file(candidate_path)
    arrays = {
        "global_frame_ids": source["global_frame_ids"].astype(np.int64, copy=True),
        "global_camera_tokens": source["global_camera_tokens"].astype(np.float32, copy=True),
        "overlap_frame_ids": source["overlap_frame_ids"].astype(np.int64, copy=True),
        "overlap_long_tokens": source["overlap_long_tokens"].astype(np.float32, copy=True),
        "span_starts": source["span_starts"].astype(np.int64, copy=True),
        "source_sample_ids": source["sample_ids"].copy(),
        "scene": np.asarray(scene, dtype="U64"),
        "role": np.asarray(role, dtype="U16"),
        "source_shard_sha256": np.asarray(source_sha256, dtype="U64"),
        "candidate_shard_sha256": np.asarray(candidate_sha256, dtype="U64"),
        "producer_git_commit": np.asarray(producer_git_commit, dtype="U40"),
    }
    validate_long_context_shard(arrays)
    destination = Path(destination)
    _atomic_npz(destination, arrays)
    return LongContextRecord(
        scene=scene,
        role=role,
        path=destination,
        sha256=_sha256_file(destination),
        source_sha256=source_sha256,
        candidate_sha256=candidate_sha256,
    )


def write_prediction_binding_manifest(
    path: Path,
    *,
    records: Sequence[LongContextRecord],
    upstream_run_root: Path,
    upstream_completion_sha256: str,
    producer_git_commit: str,
) -> dict[str, object]:
    _validate_digest(upstream_completion_sha256, "upstream completion digest")
    _validate_commit(producer_git_commit)
    scenes = [record.scene for record in records]
    if len(set(scenes)) != len(scenes):
        raise ValueError("prediction binding manifest contains a duplicate scene")
    for record in records:
        if record.role not in _ROLES:
            raise ValueError("record role must be train, validation, or smoke")
        for label, digest in (
            ("record digest", record.sha256),
            ("source digest", record.source_sha256),
            ("candidate digest", record.candidate_sha256),
        ):
            _validate_digest(digest, label)

    payload: dict[str, object] = {
        "schema": PREDICTION_BINDING_SCHEMA,
        "scene_count": len(records),
        "roles": {role: sum(record.role == role for record in records) for role in sorted(_ROLES)},
        "upstream_run_root": str(Path(upstream_run_root)),
        "upstream_completion_sha256": upstream_completion_sha256,
        "producer_git_commit": producer_git_commit,
        "records": [
            {**asdict(record), "path": str(record.path)} for record in records
        ],
    }
    _atomic_json(Path(path), payload)
    return payload
