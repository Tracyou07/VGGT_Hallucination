"""Resume helpers for ScanNet hallucination evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def selection_out_dir(out_dir: Path, scene: str, frame_count: int) -> Path:
    return out_dir / scene / f"frames_{frame_count}"


def load_completed_metrics(
    metrics_path: Path,
    requested_count: int,
    sampling: str,
) -> dict[str, Any] | None:
    if not metrics_path.exists():
        return None
    try:
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    if "scene" not in data or "frame_count_actual" not in data:
        return None
    row = dict(data)
    row["frame_count_requested"] = float(requested_count)
    row["sampling"] = sampling
    return row
