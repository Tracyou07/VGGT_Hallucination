"""Load and validate the authenticated CVA02 split-v2 protocol."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from pre_experiments.camera_velocity_ambiguity_02.contracts import (
    FrozenProtocol,
    ProtocolCounts,
    ProtocolViolation,
    canonical_json_digest,
)
from pre_experiments.local_global_consistency.split import load_split_manifest
from pre_experiments.local_global_consistency.windows import build_sliding_windows


PROTOCOL_NAME = "camera_velocity_ambiguity_02"
SCHEMA_VERSION = 2
DEVELOPMENT_NAME = "development_evaluation"
PARENT_SPLIT_PATH = "configs/scannet50_local_global_split.json"
FRAME_SELECTION = {
    "name": "fastvggt_scannet_floor_stride_v1",
    "sampling": "floor_stride",
    "preserve_first_frame": True,
    "default_count": 500,
    "exceptions": {"scene0150_00": 430},
}
WINDOWING = {"length": 100, "stride": 50}
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def _read_object(path: Path, label: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProtocolViolation(f"invalid {label}: {path}") from error
    if not isinstance(payload, dict):
        raise ProtocolViolation(f"{label} must be a JSON object")
    return payload


def _read_scene_list(path: Path) -> tuple[str, ...]:
    try:
        scenes = tuple(
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    except OSError as error:
        raise ProtocolViolation(f"invalid scene list: {path}") from error
    if len(scenes) != 50 or len(set(scenes)) != 50:
        raise ProtocolViolation("scene list must contain exactly 50 unique scenes")
    return scenes


def _require_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ProtocolViolation(f"{label} must be an object")
    return value


def _derive_counts(
    scenes: tuple[str, ...],
    calibration: tuple[str, ...],
    *,
    default_frame_count: int,
    exceptions: Mapping[str, int],
    window_length: int,
    window_stride: int,
) -> ProtocolCounts:
    calibration_set = set(calibration)
    local_windows = 0
    adjacent_pairs = 0
    primary_pairs = 0
    secondary_pairs = 0
    calibration_primary_pairs = 0
    development_primary_pairs = 0

    for scene in scenes:
        frame_count = int(exceptions.get(scene, default_frame_count))
        windows = build_sliding_windows(
            np.arange(frame_count, dtype=np.int64),
            length=window_length,
            stride=window_stride,
        )
        local_windows += len(windows)
        adjacent_pairs += max(0, len(windows) - 1)
        for left, right in zip(windows, windows[1:]):
            overlap = left.stop - right.start
            if overlap == window_length - window_stride:
                primary_pairs += 1
                if scene in calibration_set:
                    calibration_primary_pairs += 1
                else:
                    development_primary_pairs += 1
            else:
                secondary_pairs += 1

    return ProtocolCounts(
        scenes=len(scenes),
        global_runs=len(scenes),
        local_windows=local_windows,
        adjacent_pairs=adjacent_pairs,
        primary_pairs=primary_pairs,
        secondary_pairs=secondary_pairs,
        calibration_primary_pairs=calibration_primary_pairs,
        development_primary_pairs=development_primary_pairs,
    )


def load_protocol_v2(
    path: Path,
    *,
    parent_split_path: Path,
    scene_list_path: Path,
) -> FrozenProtocol:
    """Load split v2 only when every declared and derived identity matches."""
    payload = _read_object(Path(path), "CVA02 protocol")
    digest = payload.get("config_digest")
    unsigned = {key: value for key, value in payload.items() if key != "config_digest"}
    if not isinstance(digest, str) or digest != canonical_json_digest(unsigned):
        raise ProtocolViolation("config digest mismatch")

    scenes = _read_scene_list(Path(scene_list_path))
    parent = load_split_manifest(Path(parent_split_path), scenes)
    parent_spec = _require_mapping(payload.get("parent_split"), "parent_split")
    if parent_spec.get("path") != PARENT_SPLIT_PATH:
        raise ProtocolViolation("parent split path mismatch")
    if parent_spec.get("digest") != parent.get("split_digest"):
        raise ProtocolViolation("parent split digest mismatch")

    if payload.get("name") != PROTOCOL_NAME or payload.get("schema_version") != 2:
        raise ProtocolViolation("unsupported CVA02 protocol identity")
    if payload.get("scene_order") != list(scenes):
        raise ProtocolViolation("scene order differs from the official list")

    cohorts = _require_mapping(payload.get("cohorts"), "cohorts")
    calibration = tuple(cohorts.get("calibration", ()))
    development = tuple(cohorts.get(DEVELOPMENT_NAME, ()))
    if calibration != tuple(parent["calibration_scenes"]):
        raise ProtocolViolation("calibration membership differs from parent split")
    if development != tuple(parent["holdout_scenes"]):
        raise ProtocolViolation("development membership differs from parent split")
    if set(calibration).intersection(development) or set(calibration).union(development) != set(scenes):
        raise ProtocolViolation("cohorts do not exactly partition ScanNet-50")

    frame_selection = _require_mapping(payload.get("frame_selection"), "frame_selection")
    if frame_selection != FRAME_SELECTION:
        raise ProtocolViolation("frame selection protocol mismatch")
    windowing = _require_mapping(payload.get("windowing"), "windowing")
    if windowing != WINDOWING:
        raise ProtocolViolation("windowing protocol mismatch")
    if payload.get("alphas") != list(ALPHAS):
        raise ProtocolViolation("alpha grid mismatch")

    exceptions = {
        str(scene): int(count)
        for scene, count in FRAME_SELECTION["exceptions"].items()
    }
    counts = _derive_counts(
        scenes,
        calibration,
        default_frame_count=int(FRAME_SELECTION["default_count"]),
        exceptions=exceptions,
        window_length=int(WINDOWING["length"]),
        window_stride=int(WINDOWING["stride"]),
    )
    if payload.get("expected_counts") != asdict(counts):
        raise ProtocolViolation("declared protocol counts differ from mechanical counts")

    return FrozenProtocol(
        name=PROTOCOL_NAME,
        schema_version=SCHEMA_VERSION,
        config_digest=digest,
        parent_split_digest=str(parent["split_digest"]),
        scene_order=scenes,
        calibration_scenes=calibration,
        development_scenes=development,
        development_name=DEVELOPMENT_NAME,
        default_frame_count=int(FRAME_SELECTION["default_count"]),
        frame_count_exceptions=tuple(sorted(exceptions.items())),
        window_length=int(WINDOWING["length"]),
        window_stride=int(WINDOWING["stride"]),
        alphas=ALPHAS,
        counts=counts,
    )
