from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from pre_experiments.camera_velocity_ambiguity_02.frozen_oracle import (
    evaluate_with_frozen_oracle,
    fit_frozen_oracle,
)

from .candidates import load_candidate_shard
from .contracts import PrivilegedShardRecord
from .source import load_source_shard


_REQUIRED = {
    "sample_ids",
    "gt_frame_ids",
    "gt_c2w",
    "baseline_rms",
    "candidate_rms",
    "relative_improvement",
    "best_candidate_index",
    "best_candidate_rms",
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


def _load_prepared_gt(root: Path) -> tuple[np.ndarray, np.ndarray]:
    root = Path(root)
    direct_ids = root / "frame_ids.npy"
    direct_poses = root / "raw_gt_c2w.npy"
    if direct_ids.is_file() and direct_poses.is_file():
        ids = np.load(direct_ids, allow_pickle=False)
        poses = np.load(direct_poses, allow_pickle=False)
    else:
        try:
            manifest = json.loads((root / "prepared.json").read_text(encoding="utf-8"))
            ids = np.asarray(manifest["frame_ids"], dtype=np.int64)
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
            raise ValueError("prepared scene has no valid frame identity") from error
        try:
            poses = np.stack(
                [np.load(root / "pose" / f"{int(frame_id)}.npy", allow_pickle=False) for frame_id in ids]
            )
        except (OSError, ValueError) as error:
            raise ValueError("prepared scene has incomplete GT pose files") from error
    ids = np.asarray(ids)
    poses = np.asarray(poses, dtype=np.float64)
    if ids.shape != (500,) or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError("prepared GT frame IDs must have shape [500]")
    if poses.shape != (500, 4, 4) or not np.isfinite(poses).all():
        raise ValueError("prepared GT poses must have finite shape [500, 4, 4]")
    if not np.allclose(poses[:, 3, :], [0.0, 0.0, 0.0, 1.0]):
        raise ValueError("prepared GT poses must be homogeneous")
    return ids.astype(np.int64), poses


def _validate(arrays: dict[str, np.ndarray]) -> None:
    if set(arrays) != _REQUIRED:
        raise ValueError("privileged sidecar members do not match the schema")
    if any(value.dtype.hasobject for value in arrays.values()):
        raise ValueError("privileged sidecar may not contain object arrays")
    samples = arrays["candidate_rms"].shape[1] if arrays["candidate_rms"].ndim == 2 else 0
    expected = {
        "sample_ids": (8,),
        "gt_frame_ids": (8, 50),
        "gt_c2w": (8, 50, 4, 4),
        "baseline_rms": (8,),
        "candidate_rms": (8, samples),
        "relative_improvement": (8, samples),
        "best_candidate_index": (8,),
        "best_candidate_rms": (8,),
        "oracle_scale": (),
        "oracle_rotation": (3, 3),
        "oracle_translation": (3,),
        "oracle_digest": (),
    }
    for name, shape in expected.items():
        if arrays[name].shape != shape:
            raise ValueError(f"privileged member {name} has invalid shape")
    if arrays["sample_ids"].dtype.kind != "U" or arrays["oracle_digest"].dtype.kind != "U":
        raise ValueError("privileged identities must be Unicode arrays")
    for name, value in arrays.items():
        if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
            raise ValueError(f"privileged member {name} contains non-finite values")


def load_privileged_sidecar(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(Path(path), allow_pickle=False) as archive:
            arrays = {name: np.asarray(archive[name]).copy() for name in archive.files}
    except (OSError, ValueError, KeyError) as error:
        raise ValueError(f"invalid privileged sidecar: {path}") from error
    _validate(arrays)
    return arrays


def write_privileged_scene_sidecar(
    source_path: Path,
    candidate_path: Path,
    prepared_scene_root: Path,
    destination: Path,
) -> PrivilegedShardRecord:
    """Join candidate and GT only by sample/frame ID and publish outside prediction data."""
    source = load_source_shard(source_path)
    if "global_pred_c2w" not in source or "overlap_long_c2w" not in source:
        raise ValueError("source shard lacks prediction poses required by the frozen oracle")
    candidate = load_candidate_shard(candidate_path)
    if "decoded_camera_c2w" not in candidate:
        raise ValueError("candidate shard lacks decoded camera matrices")
    if not np.array_equal(candidate["source_sample_ids"], source["sample_ids"]):
        raise ValueError("candidate and source sample IDs do not match")
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
    samples = candidate["decoded_camera_c2w"].shape[1]
    gt_overlap = np.empty((8, 50, 4, 4), dtype=np.float64)
    baseline_rms = np.empty(8, dtype=np.float64)
    candidate_rms = np.empty((8, samples), dtype=np.float64)
    for overlap, frame_ids in enumerate(source["overlap_frame_ids"]):
        indices = np.searchsorted(gt_ids, frame_ids)
        if not np.array_equal(gt_ids[indices], frame_ids):
            raise ValueError("overlap frame IDs are absent from prepared GT")
        gt_overlap[overlap] = gt_c2w[indices]
        baseline_rms[overlap] = evaluate_with_frozen_oracle(
            oracle, source["overlap_long_c2w"][overlap], gt_overlap[overlap]
        ).rms_translation_error
        for sample in range(samples):
            candidate_rms[overlap, sample] = evaluate_with_frozen_oracle(
                oracle,
                candidate["decoded_camera_c2w"][overlap, sample],
                gt_overlap[overlap],
            ).rms_translation_error
    denominator = np.maximum(baseline_rms[:, None], np.finfo(np.float64).eps)
    relative = (baseline_rms[:, None] - candidate_rms) / denominator
    best_indices = np.argmin(candidate_rms, axis=1).astype(np.int64)
    best_rms = candidate_rms[np.arange(8), best_indices]
    arrays = {
        "sample_ids": source["sample_ids"].copy(),
        "gt_frame_ids": source["overlap_frame_ids"].copy(),
        "gt_c2w": gt_overlap,
        "baseline_rms": baseline_rms,
        "candidate_rms": candidate_rms,
        "relative_improvement": relative,
        "best_candidate_index": best_indices,
        "best_candidate_rms": best_rms,
        "oracle_scale": np.asarray(oracle.scale, dtype=np.float64),
        "oracle_rotation": np.asarray(oracle.rotation, dtype=np.float64),
        "oracle_translation": np.asarray(oracle.translation, dtype=np.float64),
        "oracle_digest": np.asarray(oracle.transform_digest, dtype="U64"),
    }
    _validate(arrays)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(destination)
    return PrivilegedShardRecord(scene, destination, 8, samples, _sha256_file(destination))
