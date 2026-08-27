from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LongContextRecord:
    """Published long-context-only shard bound to one candidate shard."""

    scene: str
    role: str
    path: Path
    sha256: str
    source_sha256: str
    candidate_sha256: str

