"""Prediction-free access to Camera Context source frame identities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np

from pre_experiments.common.scannet import load_scene_frames, uniform_frame_ids
from pre_experiments.local_global_consistency.artifacts import load_global_context


EXPECTED_CONTEXT_PROTOCOL = {
    "frame_counts": [500],
    "iterations": [4],
    "sampling": "nested_uniform",
    "preprocess_mode": "pad",
    "save_context_diagnostics": True,
}


def load_context_frame_ids(path: Path) -> np.ndarray:
    """Read only frame IDs from a context artifact.

    Split construction must remain independent of every VGGT prediction array.
    """
    if not path.is_file():
        raise FileNotFoundError(f"context diagnostics artifact is missing: {path}")
    with np.load(path, allow_pickle=False) as archive:
        if "frame_ids" not in archive.files:
            raise ValueError(f"context diagnostics has no frame_ids member: {path}")
        raw_ids = np.asarray(archive["frame_ids"])
    if raw_ids.ndim != 1 or len(raw_ids) < 2:
        raise ValueError("context frame_ids must be a one-dimensional sequence")
    if not np.issubdtype(raw_ids.dtype, np.integer):
        raise ValueError("context frame_ids must have an integer dtype")
    frame_ids = raw_ids.astype(np.int64, copy=True)
    if np.any(np.diff(frame_ids) <= 0):
        raise ValueError("context frame_ids must be strictly increasing and unique")
    return frame_ids


def _json_object(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON object: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def validate_context_source_metadata(
    source: Path,
    scenes: list[str],
) -> dict[str, object]:
    """Validate source identity and protocol without opening any NPZ member."""
    if len(scenes) != 50 or len(set(scenes)) != 50:
        raise ValueError("context source contract requires 50 unique scenes")
    metadata = _json_object(source / "run_metadata.json")
    invocation = metadata.get("invocation")
    if not isinstance(invocation, dict):
        raise ValueError("source metadata must declare an invocation object")
    if invocation.get("scenes") != scenes:
        raise ValueError("source metadata scenes must exactly match the split scene order")
    for field, expected in EXPECTED_CONTEXT_PROTOCOL.items():
        if invocation.get(field) != expected:
            raise ValueError(
                f"source metadata {field} must equal {expected!r}, "
                f"found {invocation.get(field)!r}"
            )
    source_run_id = metadata.get("run_id")
    if not isinstance(source_run_id, str) or not source_run_id:
        raise ValueError("source metadata must declare a non-empty run_id")
    discovered = {
        path.parents[1].name
        for path in source.glob("*/frames_500/context_diagnostics.npz")
    }
    if discovered != set(scenes):
        missing = sorted(set(scenes).difference(discovered))
        extra = sorted(discovered.difference(scenes))
        raise ValueError(
            f"context artifact scene set mismatch; missing={missing}, extra={extra}"
        )
    return metadata


def validate_context_source(
    source: Path,
    split: Mapping[str, object],
    data_dir: Path,
) -> dict[str, object]:
    """Validate the complete ScanNet-50 global source before GPU inference."""
    scenes = split.get("scene_order")
    if (
        not isinstance(scenes, list)
        or len(scenes) != 50
        or len(set(scenes)) != 50
        or not all(isinstance(scene, str) for scene in scenes)
    ):
        raise ValueError("split must declare exactly 50 unique ordered scenes")
    metadata = validate_context_source_metadata(source, scenes)
    source_run_id = metadata.get("run_id")
    if split.get("source_run_id") != source_run_id:
        raise ValueError("split source_run_id does not match source metadata")

    frame_ids_by_scene: dict[str, list[int]] = {}
    for scene in scenes:
        artifact_path = source / scene / "frames_500" / "context_diagnostics.npz"
        frame_ids = load_context_frame_ids(artifact_path)
        _, poses_by_id, valid_ids = load_scene_frames(data_dir, scene)
        expected_ids = np.asarray(uniform_frame_ids(valid_ids, 500), dtype=np.int64)
        if not np.array_equal(frame_ids, expected_ids):
            raise ValueError(f"context frame IDs do not match processed ScanNet: {scene}")
        global_artifact = load_global_context(artifact_path)
        if not np.array_equal(global_artifact["frame_ids"], expected_ids):
            raise ValueError(f"global artifact frame IDs are inconsistent: {scene}")
        raw_gt = np.stack([poses_by_id[int(frame_id)] for frame_id in frame_ids])
        if not np.allclose(
            global_artifact["gt_c2w_raw"], raw_gt, atol=1e-10, rtol=0
        ):
            raise ValueError(f"raw GT mismatch with processed ScanNet: {scene}")
        frame_ids_by_scene[scene] = frame_ids.tolist()

    return {
        "metadata": metadata,
        "source_run_id": source_run_id,
        "scenes": scenes.copy(),
        "frame_ids_by_scene": frame_ids_by_scene,
    }
