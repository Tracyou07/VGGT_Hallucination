"""Immutable protocol contracts shared by CVA02 components."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from typing import Mapping


class ProtocolViolation(ValueError):
    """Raised when authenticated inputs violate the frozen CVA02 protocol."""


class EvidenceSource(str, Enum):
    """Provenance classes enforced by the CVA02 metric firewall."""

    PREDICTION_ONLY = "PREDICTION_ONLY"
    PRIVILEGED_GT = "PRIVILEGED_GT"
    OBSERVATION_RGBD = "OBSERVATION_RGBD"
    PRESENTATION_ONLY = "PRESENTATION_ONLY"


def canonical_json_digest(payload: Mapping[str, object]) -> str:
    """Return the SHA-256 of compact, key-sorted canonical JSON."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ProtocolCounts:
    """Mechanically derived scene, window, and adjacent-pair counts."""

    scenes: int
    global_runs: int
    local_windows: int
    adjacent_pairs: int
    primary_pairs: int
    secondary_pairs: int
    calibration_primary_pairs: int
    development_primary_pairs: int


@dataclass(frozen=True)
class FrozenProtocol:
    """Authenticated CVA02 protocol with immutable tuple-backed membership."""

    name: str
    schema_version: int
    config_digest: str
    parent_split_digest: str
    scene_order: tuple[str, ...]
    calibration_scenes: tuple[str, ...]
    development_scenes: tuple[str, ...]
    development_name: str
    default_frame_count: int
    frame_count_exceptions: tuple[tuple[str, int], ...]
    window_length: int
    window_stride: int
    alphas: tuple[float, ...]
    counts: ProtocolCounts

    def frame_count(self, scene: str) -> int:
        """Return the frozen selected-frame count for one protocol scene."""
        if scene not in self.scene_order:
            raise ProtocolViolation(f"scene is absent from the frozen protocol: {scene}")
        for exception_scene, count in self.frame_count_exceptions:
            if scene == exception_scene:
                return count
        return self.default_frame_count
