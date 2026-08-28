from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from pre_experiments.variational_camera_latent.schema import SOURCE_SCHEMA
from pre_experiments.variational_camera_latent.source import load_source_shard


LONG_CONTEXT_MEMBERS = {
    "scene",
    "frame_ids",
    "camera_tokens",
    "baseline_c2w",
    "source_sha256",
}


@dataclass(frozen=True)
class SceneRecord:
    scene: str
    role: str
    path: Path
    sha256: str


@dataclass(frozen=True)
class LongContextRecord:
    scene: str
    role: str
    path: Path
    sha256: str
    source_sha256: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def load_source_records(source_run: Path) -> tuple[SceneRecord, ...]:
    """Rebind migrated manifest rows below an explicit source run and verify bytes."""
    source_run = Path(source_run).resolve()
    manifest = _read_json(source_run / "manifests" / "source_manifest.json")
    if manifest.get("schema") != SOURCE_SCHEMA:
        raise ValueError("source manifest schema mismatch")
    rows = manifest.get("records")
    if not isinstance(rows, list) or not rows:
        raise ValueError("source manifest must contain records")
    records: list[SceneRecord] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("source manifest record must be an object")
        try:
            scene = str(row["scene"])
            role = str(row["role"])
            expected_digest = str(row["sha256"])
        except KeyError as error:
            raise ValueError("source manifest record is incomplete") from error
        if scene in seen or role not in {"train", "validation", "smoke"}:
            raise ValueError("source manifest scenes must be unique with a valid role")
        if len(expected_digest) != 64:
            raise ValueError("source manifest digest is malformed")
        rebound = source_run / "prediction_only" / "source" / f"{scene}.npz"
        if not rebound.is_file() or sha256_file(rebound) != expected_digest:
            raise ValueError(f"source shard digest mismatch: {scene}")
        load_source_shard(rebound)
        records.append(SceneRecord(scene, role, rebound, expected_digest))
        seen.add(scene)
    return tuple(records)


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _validate_long_context(arrays: dict[str, np.ndarray]) -> None:
    if set(arrays) != LONG_CONTEXT_MEMBERS:
        raise ValueError("long-context members do not match the strict schema")
    if any(value.dtype.hasobject for value in arrays.values()):
        raise ValueError("long-context arrays may not use object dtype")
    if arrays["scene"].shape != () or arrays["scene"].dtype.kind != "U":
        raise ValueError("long-context scene must be a Unicode scalar")
    if arrays["frame_ids"].shape != (500,) or not np.issubdtype(
        arrays["frame_ids"].dtype, np.integer
    ):
        raise ValueError("long-context frame IDs must have shape [500]")
    if np.any(arrays["frame_ids"][1:] <= arrays["frame_ids"][:-1]):
        raise ValueError("long-context frame IDs must be strictly increasing")
    tokens = arrays["camera_tokens"]
    if tokens.shape != (500, 2048) or not np.issubdtype(tokens.dtype, np.floating):
        raise ValueError("long-context Camera tokens must have shape [500,2048]")
    poses = arrays["baseline_c2w"]
    if poses.shape != (500, 4, 4):
        raise ValueError("long-context baseline poses must have shape [500,4,4]")
    if not np.isfinite(tokens).all() or not np.isfinite(poses).all():
        raise ValueError("long-context tensors must be finite")
    if not np.allclose(poses[:, 3, :], [0.0, 0.0, 0.0, 1.0]):
        raise ValueError("long-context baseline poses must be homogeneous")
    digest = arrays["source_sha256"]
    if digest.shape != () or digest.dtype.kind != "U" or len(str(digest)) != 64:
        raise ValueError("long-context source digest is malformed")


def publish_long_context(record: SceneRecord, destination: Path) -> LongContextRecord:
    source = load_source_shard(record.path)
    arrays = {
        "scene": np.asarray(record.scene, dtype="U32"),
        "frame_ids": source["global_frame_ids"].astype(np.int64, copy=True),
        "camera_tokens": source["global_camera_tokens"].astype(np.float32, copy=True),
        "baseline_c2w": source["global_pred_c2w"].astype(np.float64, copy=True),
        "source_sha256": np.asarray(record.sha256, dtype="U64"),
    }
    _validate_long_context(arrays)
    _atomic_npz(Path(destination), arrays)
    return LongContextRecord(
        scene=record.scene,
        role=record.role,
        path=Path(destination),
        sha256=sha256_file(Path(destination)),
        source_sha256=record.sha256,
    )


def load_long_context(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(Path(path), allow_pickle=False) as archive:
            arrays = {name: np.asarray(archive[name]).copy() for name in archive.files}
    except (OSError, ValueError, KeyError) as error:
        raise ValueError(f"invalid long-context shard: {path}") from error
    _validate_long_context(arrays)
    return arrays


def load_prepared_gt(prepared_scene: Path, frame_ids: np.ndarray) -> np.ndarray:
    prepared_scene = Path(prepared_scene)
    ids = np.asarray(frame_ids)
    try:
        poses = np.stack(
            [
                np.load(prepared_scene / "pose" / f"{int(frame_id)}.npy", allow_pickle=False)
                for frame_id in ids
            ]
        ).astype(np.float64)
    except (OSError, ValueError) as error:
        raise ValueError(f"prepared scene has incomplete GT poses: {prepared_scene}") from error
    if ids.shape != (500,) or poses.shape != (500, 4, 4):
        raise ValueError("prepared GT must match exactly 500 requested frames")
    if not np.isfinite(poses).all() or not np.allclose(
        poses[:, 3, :], [0.0, 0.0, 0.0, 1.0]
    ):
        raise ValueError("prepared GT poses must be finite homogeneous matrices")
    return poses

