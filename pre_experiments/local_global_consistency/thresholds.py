"""Frozen prediction-only reliability threshold artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Mapping, Sequence

import numpy as np


SCHEMA_VERSION = 1
THRESHOLD_FIELDS = {
    "token_cosine_p95": "local_local_token_cosine",
    "pose_translation_p95": "local_local_pose_translation",
    "pose_rotation_deg_p95": "local_local_pose_rotation_deg",
}
_HEX_40 = re.compile(r"[0-9a-f]{40}")
_HEX_64 = re.compile(r"[0-9a-f]{64}")


def _digest(payload: Mapping[str, object]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _required_string(
    payload: Mapping[str, object],
    name: str,
    pattern: re.Pattern[str] | None = None,
) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    if pattern is not None and pattern.fullmatch(value) is None:
        raise ValueError(f"{name} has an invalid digest or commit format")
    return value


def _calibration_scenes(provenance: Mapping[str, object]) -> list[str]:
    scenes = provenance.get("calibration_scenes")
    if (
        not isinstance(scenes, list)
        or len(scenes) != 10
        or len(set(scenes)) != 10
        or not all(isinstance(scene, str) and scene for scene in scenes)
    ):
        raise ValueError("calibration_scenes must contain ten unique scene IDs")
    return scenes


def fit_frozen_thresholds(
    score_rows: Sequence[Mapping[str, object]],
    provenance: Mapping[str, object],
) -> dict[str, object]:
    """Fit three P95 thresholds from complete calibration prediction rows."""
    scenes = _calibration_scenes(provenance)
    source_run_id = _required_string(provenance, "source_run_id")
    calibration_run_id = _required_string(provenance, "calibration_run_id")
    split_digest = _required_string(provenance, "split_digest", _HEX_64)
    code_commit = _required_string(provenance, "code_commit", _HEX_40)
    if not score_rows:
        raise ValueError("threshold fitting requires prediction score rows")

    forbidden = {
        key
        for row in score_rows
        for key in row
        if "gt" in key.lower() or "error" in key.lower()
    }
    if forbidden:
        raise ValueError(
            f"threshold fitting received GT-derived score fields: {sorted(forbidden)}"
        )
    actual_scenes = {str(row.get("scene")) for row in score_rows}
    if actual_scenes != set(scenes):
        raise ValueError(
            "threshold score scene set must exactly match calibration_scenes"
        )

    thresholds: dict[str, float] = {}
    sample_counts: dict[str, int] = {}
    for output_name, field in THRESHOLD_FIELDS.items():
        if any(field not in row for row in score_rows):
            raise ValueError(f"threshold score row is missing {field}")
        values = [
            float(row[field])
            for row in score_rows
            if row.get(field) is not None
        ]
        if not values or not np.isfinite(values).all():
            raise ValueError(f"no finite calibration values available for {field}")
        thresholds[output_name] = float(np.percentile(values, 95))
        sample_counts[output_name] = len(values)

    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "metric_fields": dict(THRESHOLD_FIELDS),
        "thresholds": thresholds,
        "sample_counts": sample_counts,
        "calibration_scenes": scenes.copy(),
        "source_run_id": source_run_id,
        "calibration_run_id": calibration_run_id,
        "split_digest": split_digest,
        "code_commit": code_commit,
    }
    payload["threshold_digest"] = _digest(payload)
    return payload


def load_frozen_thresholds(
    path: Path,
    expected_split_digest: str,
    expected_source_run_id: str,
) -> dict[str, object]:
    """Load and authenticate a calibration threshold artifact."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid frozen threshold artifact: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError("frozen threshold artifact must be a JSON object")
    threshold_digest = payload.get("threshold_digest")
    unsigned = {key: value for key, value in payload.items() if key != "threshold_digest"}
    if (
        not isinstance(threshold_digest, str)
        or _HEX_64.fullmatch(threshold_digest) is None
        or threshold_digest != _digest(unsigned)
    ):
        raise ValueError("frozen threshold digest mismatch")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported frozen threshold schema")
    if payload.get("metric_fields") != THRESHOLD_FIELDS:
        raise ValueError("frozen threshold metric mapping is invalid")
    if payload.get("split_digest") != expected_split_digest:
        raise ValueError("frozen threshold split digest mismatch")
    if payload.get("source_run_id") != expected_source_run_id:
        raise ValueError("frozen threshold source run ID mismatch")
    _calibration_scenes(payload)
    _required_string(payload, "calibration_run_id")
    _required_string(payload, "code_commit", _HEX_40)
    thresholds = payload.get("thresholds")
    sample_counts = payload.get("sample_counts")
    if (
        not isinstance(thresholds, dict)
        or set(thresholds) != set(THRESHOLD_FIELDS)
        or not all(
            isinstance(value, (int, float)) and np.isfinite(value)
            for value in thresholds.values()
        )
    ):
        raise ValueError("frozen threshold values are invalid")
    if (
        not isinstance(sample_counts, dict)
        or set(sample_counts) != set(THRESHOLD_FIELDS)
        or not all(isinstance(value, int) and value > 0 for value in sample_counts.values())
    ):
        raise ValueError("frozen threshold sample counts are invalid")
    return payload
