from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SourceShardRecord:
    """Published prediction-only source shard."""

    scene: str
    role: str
    path: Path
    overlap_count: int
    sha256: str


@dataclass(frozen=True)
class CandidateShardRecord:
    """Published raw VRFM samples for one scene."""

    scene: str
    path: Path
    overlap_count: int
    sample_count: int
    sha256: str
