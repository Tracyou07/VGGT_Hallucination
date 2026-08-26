"""Scalar record firewall and publication for the CVA02 analysis."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

from pre_experiments.camera_velocity_ambiguity_02.contracts import (
    EvidenceSource,
    canonical_json_digest,
)
from pre_experiments.camera_velocity_ambiguity_02.events import (
    EventPolicy,
    MetricEvidence,
    classify_event,
)


PREDICTION_FIELDS = {
    "sample_id",
    "scene",
    "pair_id",
    "route",
    "alignment_valid",
    "direction_evaluable",
    "flattened_cosine",
    "normalized_separation",
}
PRIVILEGED_FIELDS = {
    "sample_id",
    "left_endpoint_valid",
    "right_endpoint_valid",
    "global_rms",
    "left_rms",
    "right_rms",
}
RGBD_FIELDS = {
    "sample_id",
    "rgbd_valid",
    "interior_barrier",
    "temporal_support",
}


def _exact(record: Mapping[str, object], fields: set[str], label: str) -> None:
    if set(record) != fields:
        raise ValueError(f"{label} record schema mismatch")


def analyze_pair_records(
    prediction: Mapping[str, object],
    privileged: Mapping[str, object],
    rgbd: Mapping[str, object],
    policy: EventPolicy,
) -> dict[str, object]:
    """Join exact, separated evidence layers and produce one decision row."""
    _exact(prediction, PREDICTION_FIELDS, "prediction-only")
    _exact(privileged, PRIVILEGED_FIELDS, "privileged")
    _exact(rgbd, RGBD_FIELDS, "RGB-D")
    sample_ids = {prediction["sample_id"], privileged["sample_id"], rgbd["sample_id"]}
    if len(sample_ids) != 1:
        raise ValueError("evidence layers have different sample IDs")
    evidence = {
        "alignment_valid": MetricEvidence(prediction["alignment_valid"], EvidenceSource.PREDICTION_ONLY),
        "direction_evaluable": MetricEvidence(prediction["direction_evaluable"], EvidenceSource.PREDICTION_ONLY),
        "flattened_cosine": MetricEvidence(prediction["flattened_cosine"], EvidenceSource.PREDICTION_ONLY),
        "normalized_separation": MetricEvidence(prediction["normalized_separation"], EvidenceSource.PREDICTION_ONLY),
        "left_endpoint_valid": MetricEvidence(privileged["left_endpoint_valid"], EvidenceSource.PRIVILEGED_GT),
        "right_endpoint_valid": MetricEvidence(privileged["right_endpoint_valid"], EvidenceSource.PRIVILEGED_GT),
        "rgbd_valid": MetricEvidence(rgbd["rgbd_valid"], EvidenceSource.OBSERVATION_RGBD),
        "interior_barrier": MetricEvidence(rgbd["interior_barrier"], EvidenceSource.OBSERVATION_RGBD),
        "temporal_support": MetricEvidence(rgbd["temporal_support"], EvidenceSource.OBSERVATION_RGBD),
    }
    result = classify_event(evidence, policy)
    return {
        "sample_id": prediction["sample_id"],
        "scene": prediction["scene"],
        "pair_id": prediction["pair_id"],
        "route": prediction["route"],
        "event_class": result.event_class.value,
        "reason": result.reason,
    }


def _jsonl_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    return "".join(
        json.dumps(dict(row), sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    ).encode("utf-8")


def _publish(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)
    return hashlib.sha256(payload).hexdigest()


def publish_scene_records(
    output_dir: Path,
    *,
    scene: str,
    prediction_rows: Sequence[Mapping[str, object]],
    privileged_rows: Sequence[Mapping[str, object]],
    rgbd_rows: Sequence[Mapping[str, object]],
    decision_rows: Sequence[Mapping[str, object]],
    control_rows: Sequence[Mapping[str, object]] = (),
) -> dict[str, object]:
    """Publish evidence layers separately and bind them with a small manifest."""
    root = Path(output_dir)
    layers = {
        "prediction_only": ("prediction_only.jsonl", prediction_rows),
        "privileged_labels": ("privileged_labels.jsonl", privileged_rows),
        "rgbd_observation": ("rgbd_observation.jsonl", rgbd_rows),
        "decisions": ("decisions.jsonl", decision_rows),
        "controls": ("controls.jsonl", control_rows),
    }
    hashes: dict[str, str] = {}
    counts: dict[str, int] = {}
    for name, (filename, rows) in layers.items():
        hashes[name] = _publish(root / filename, _jsonl_bytes(rows))
        counts[name] = len(rows)
    unsigned = {
        "schema": "camera_velocity_ambiguity_02.scene_records.v1",
        "scene": scene,
        "counts": counts,
        "sha256": hashes,
    }
    manifest = {**unsigned, "manifest_digest": canonical_json_digest(unsigned)}
    _publish(
        root / "records_manifest.json",
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return manifest
