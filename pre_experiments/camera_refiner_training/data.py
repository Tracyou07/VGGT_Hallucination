"""Strict adapter from external refiner shards to 100-frame training windows."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from pre_experiments.camera_refiner_training.geometry import (
    SceneGauge,
    align_centers_to_reference,
)
from pre_experiments.camera_refiner_training.windowing import (
    assemble_windows_in_reference_gauge,
    build_sliding_windows,
)


@dataclass(frozen=True)
class UnitSelection:
    indices: tuple[int, ...]
    iteration: int
    digest: str


@dataclass(frozen=True)
class DatasetEntry:
    scene: str
    role: str
    shard: Path


@dataclass(frozen=True)
class DatasetManifest:
    entries: tuple[DatasetEntry, ...]
    digest: str


@dataclass
class SceneWindows:
    scene: str
    condition: np.ndarray
    target_residual: np.ndarray
    global_centers: np.ndarray
    frame_ids: np.ndarray
    starts: np.ndarray
    alignment_residual: np.ndarray
    global_c2w: np.ndarray
    gt_c2w_raw: np.ndarray
    gauge: SceneGauge


def _digest_json(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_translation_units(
    path: Path,
    *,
    count: int = 41,
    iteration: int = 0,
) -> UnitSelection:
    if count < 1 or iteration < 0:
        raise ValueError("count must be positive and iteration non-negative")
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    candidates = payload.get("translation_units")
    if candidates is None:
        candidates = payload.get("selected_units")
    if candidates is None:
        scores = payload.get("scores", {})
        candidates = scores.get("translation") if isinstance(scores, dict) else None
    if not isinstance(candidates, list):
        raise ValueError("unit manifest has no translation unit list")
    indices = []
    for item in candidates:
        if isinstance(item, int):
            item_iteration, unit = iteration, item
        elif isinstance(item, dict):
            item_iteration = int(item.get("iteration", -1))
            unit = int(item.get("unit", -1))
        else:
            raise ValueError("translation unit entries must be integers or objects")
        if item_iteration == iteration:
            if unit < 0 or unit in indices:
                raise ValueError("translation unit indices must be unique and non-negative")
            indices.append(unit)
        if len(indices) == count:
            break
    if len(indices) != count:
        raise ValueError(f"requested {count} iteration-{iteration} units, found {len(indices)}")
    return UnitSelection(tuple(indices), iteration, _digest_json(payload))


def load_dataset_manifest(
    manifest_path: Path,
    dataset_root: Path,
    *,
    roles: set[str] | None = None,
) -> DatasetManifest:
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    strict = "schema_version" in payload
    if strict:
        canonical = dict(payload)
        declared_digest = canonical.pop("dataset_digest", None)
        if declared_digest != _digest_json(canonical):
            raise ValueError("dataset manifest digest is invalid")
    entries = payload.get("shards")
    if not isinstance(entries, list) or not entries:
        raise ValueError("dataset manifest must contain non-empty shards")
    root = Path(dataset_root).resolve()
    selected = []
    seen = set()
    for item in entries:
        if not isinstance(item, dict):
            raise ValueError("dataset shard entries must be objects")
        scene = str(item.get("scene", ""))
        role = str(item.get("role", ""))
        if not scene or not role or scene in seen:
            raise ValueError("dataset scenes and roles must be non-empty and unique")
        seen.add(scene)
        if roles is not None and role not in roles:
            continue
        shard = (root / str(item.get("path", ""))).resolve()
        try:
            shard.relative_to(root)
        except ValueError as error:
            raise ValueError(f"dataset shard escapes root: {scene}") from error
        if not shard.is_file():
            raise FileNotFoundError(f"missing dataset shard: {shard}")
        if strict:
            expected_hash = item.get("sha256")
            if not isinstance(expected_hash, str) or _digest_file(shard) != expected_hash:
                raise ValueError(f"dataset shard checksum mismatch: {scene}")
        selected.append(DatasetEntry(scene, role, shard))
    if not selected:
        raise ValueError("no dataset scenes match requested roles")
    digest = str(payload.get("dataset_digest") or _digest_json(payload))
    return DatasetManifest(tuple(selected), digest)


def _load_local_windows(scene_dir: Path) -> list[dict[str, np.ndarray]]:
    records = []
    for path in sorted(Path(scene_dir).glob("window_*/window_diagnostics.npz")):
        with np.load(path, allow_pickle=False) as archive:
            if not {"frame_ids", "pred_c2w_raw"}.issubset(archive.files):
                raise ValueError(f"local window is incomplete: {path}")
            records.append(
                {
                    "frame_ids": np.asarray(archive["frame_ids"], dtype=np.int64),
                    "pred_c2w_raw": np.asarray(archive["pred_c2w_raw"], dtype=np.float64),
                }
            )
    if not records:
        raise FileNotFoundError(f"no local windows found under {scene_dir}")
    return records


def _window_slices(frame_count: int, length: int, stride: int) -> list[slice]:
    frames = build_sliding_windows(
        np.arange(frame_count, dtype=np.int64),
        length=length,
        stride=stride,
    )
    return [slice(window.start, window.stop) for window in frames]


def build_scene_windows(
    shard_path: Path,
    local_scene_dir: Path,
    units: UnitSelection,
    *,
    window_length: int = 100,
    stride: int = 50,
) -> SceneWindows:
    """Join one scene shard with local poses; GT affects only target residuals."""
    with np.load(shard_path, allow_pickle=False) as archive:
        required = {
            "scene_name", "frame_ids", "scales", "global_hidden", "local_hidden",
            "selected_boundary_distance", "local_observation_count", "pred_c2w_raw",
            "gt_c2w_raw",
        }
        if not required.issubset(archive.files):
            raise ValueError(f"scene shard is missing members: {sorted(required - set(archive.files))}")
        scene = str(np.asarray(archive["scene_name"]).item())
        frame_ids = np.asarray(archive["frame_ids"], dtype=np.int64)
        scales = np.asarray(archive["scales"], dtype=np.int64)
        global_hidden = np.asarray(archive["global_hidden"], dtype=np.float32)
        local_hidden = np.asarray(archive["local_hidden"], dtype=np.float32)
        boundary = np.asarray(archive["selected_boundary_distance"], dtype=np.float32)
        observations = np.asarray(archive["local_observation_count"], dtype=np.float32)
        predicted = np.asarray(archive["pred_c2w_raw"], dtype=np.float64)
        gt_c2w = np.asarray(archive["gt_c2w_raw"], dtype=np.float64)
    if len(frame_ids) < window_length or len(np.unique(frame_ids)) != len(frame_ids):
        raise ValueError("scene frame IDs are invalid")
    matches = np.flatnonzero(scales == window_length)
    if len(matches) != 1:
        raise ValueError(f"scene shard must contain scale {window_length} exactly once")
    scale_index = int(matches[0])
    if global_hidden.ndim != 3 or units.iteration >= global_hidden.shape[0]:
        raise ValueError("global hidden tensor does not contain selected iteration")
    if local_hidden.ndim != 4 or local_hidden.shape[:3] != (
        len(scales), global_hidden.shape[0], len(frame_ids)
    ):
        raise ValueError("local hidden tensor shape is invalid")
    if any(unit >= global_hidden.shape[2] for unit in units.indices):
        raise ValueError("selected unit exceeds hidden dimension")
    if predicted.ndim != 4 or predicted.shape[1:] != (len(frame_ids), 4, 4):
        raise ValueError("pred_c2w_raw must have shape [candidate, frame, 4, 4]")
    global_c2w = predicted[0]
    if gt_c2w.shape != global_c2w.shape:
        raise ValueError("raw GT and global pose shape mismatch")

    local_records = _load_local_windows(local_scene_dir)
    assembled = assemble_windows_in_reference_gauge(
        local_records,
        reference_frame_ids=frame_ids,
        reference_c2w=global_c2w,
    )
    local_c2w = np.asarray(assembled["assembled_c2w"], dtype=np.float64)
    global_centers_raw = global_c2w[:, :3, 3]
    local_centers_raw = local_c2w[:, :3, 3]
    gauge = SceneGauge.from_c2w(global_c2w)
    global_centers = gauge.canonicalize(global_centers_raw)
    local_centers = gauge.canonicalize(local_centers_raw)
    target_raw, _ = align_centers_to_reference(gt_c2w[:, :3, 3], global_centers_raw)
    target_centers = gauge.canonicalize(target_raw)
    target_residual = target_centers - global_centers
    alignment_residual = np.linalg.norm(local_centers - global_centers, axis=1)

    unit_index = np.asarray(units.indices, dtype=np.int64)
    global_units = global_hidden[units.iteration, :, unit_index].T
    local_units = local_hidden[scale_index, units.iteration, :, unit_index].T
    position = np.linspace(0.0, 1.0, len(frame_ids), dtype=np.float32)[:, None]
    boundary_values = boundary[scale_index, :, None] / max(window_length - 1, 1)
    residual_values = alignment_residual.astype(np.float32)[:, None]
    valid = (observations[scale_index, :, None] > 0).astype(np.float32)
    features = np.concatenate(
        [
            global_centers.astype(np.float32),
            local_centers.astype(np.float32),
            (local_centers - global_centers).astype(np.float32),
            global_units.astype(np.float32),
            local_units.astype(np.float32),
            (local_units - global_units).astype(np.float32),
            position,
            boundary_values.astype(np.float32),
            residual_values,
            valid,
        ],
        axis=1,
    )
    slices = _window_slices(len(frame_ids), window_length, stride)
    return SceneWindows(
        scene=scene,
        condition=np.stack([features[value] for value in slices]),
        target_residual=np.stack([target_residual[value] for value in slices]).astype(np.float32),
        global_centers=np.stack([global_centers[value] for value in slices]).astype(np.float32),
        frame_ids=np.stack([frame_ids[value] for value in slices]),
        starts=np.asarray([value.start for value in slices], dtype=np.int64),
        alignment_residual=np.stack([alignment_residual[value] for value in slices]).astype(np.float32),
        global_c2w=global_c2w,
        gt_c2w_raw=gt_c2w,
        gauge=gauge,
    )
