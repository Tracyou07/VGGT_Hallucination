"""Fail-closed evidence firewall and CVA02 event classification."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping

import numpy as np

from pre_experiments.camera_velocity_ambiguity_02.contracts import EvidenceSource


class EventClass(str, Enum):
    NOT_SUPPORTED = "NOT_SUPPORTED"
    SELECTOR_PROBLEM = "SELECTOR_PROBLEM"
    CONTINUOUS_REDUNDANCY = "CONTINUOUS_REDUNDANCY"
    MULTIMODAL_VELOCITY_SUPPORTED = "MULTIMODAL_VELOCITY_SUPPORTED"
    UNIDENTIFIABLE_WITH_TRANSLATION_ONLY = "UNIDENTIFIABLE_WITH_TRANSLATION_ONLY"


@dataclass(frozen=True)
class EventPolicy:
    direction_cosine_max: float
    normalized_separation_min: float
    barrier_margin: float

    def __post_init__(self) -> None:
        values = (
            self.direction_cosine_max,
            self.normalized_separation_min,
            self.barrier_margin,
        )
        if not np.isfinite(values).all():
            raise ValueError("event policy thresholds must be finite")
        if not -1.0 <= self.direction_cosine_max <= 1.0:
            raise ValueError("direction cosine threshold must be in [-1,1]")
        if self.normalized_separation_min < 0 or self.barrier_margin < 0:
            raise ValueError("separation and barrier thresholds must be non-negative")


@dataclass(frozen=True)
class MetricEvidence:
    value: object
    source: EvidenceSource


@dataclass(frozen=True)
class EventDecision:
    event_class: EventClass
    reason: str


_SOURCES = {
    "alignment_valid": EvidenceSource.PREDICTION_ONLY,
    "direction_evaluable": EvidenceSource.PREDICTION_ONLY,
    "flattened_cosine": EvidenceSource.PREDICTION_ONLY,
    "normalized_separation": EvidenceSource.PREDICTION_ONLY,
    "left_endpoint_valid": EvidenceSource.PRIVILEGED_GT,
    "right_endpoint_valid": EvidenceSource.PRIVILEGED_GT,
    "rgbd_valid": EvidenceSource.OBSERVATION_RGBD,
    "interior_barrier": EvidenceSource.OBSERVATION_RGBD,
    "temporal_support": EvidenceSource.OBSERVATION_RGBD,
}


def _boolean(metrics: Mapping[str, MetricEvidence], name: str) -> bool:
    value = metrics[name].value
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be boolean")
    return bool(value)


def _number(metrics: Mapping[str, MetricEvidence], name: str) -> float:
    value = metrics[name].value
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def classify_event(
    metrics: Mapping[str, MetricEvidence], policy: EventPolicy
) -> EventDecision:
    """Classify only from exact fields with the frozen source policy."""
    if set(metrics) != set(_SOURCES):
        raise ValueError("event evidence fields do not match the frozen metric firewall")
    for name, expected in _SOURCES.items():
        evidence = metrics[name]
        if not isinstance(evidence, MetricEvidence) or evidence.source != expected:
            raise ValueError(f"event evidence source mismatch: {name}")
        if evidence.source == EvidenceSource.PRESENTATION_ONLY:
            raise ValueError("presentation-only evidence cannot enter event decisions")

    alignment = _boolean(metrics, "alignment_valid")
    direction = _boolean(metrics, "direction_evaluable")
    cosine = _number(metrics, "flattened_cosine")
    separation = _number(metrics, "normalized_separation")
    left_valid = _boolean(metrics, "left_endpoint_valid")
    right_valid = _boolean(metrics, "right_endpoint_valid")
    rgbd_valid = _boolean(metrics, "rgbd_valid")
    barrier = _number(metrics, "interior_barrier")
    temporal = _boolean(metrics, "temporal_support")

    if (
        not alignment
        or not direction
        or cosine > policy.direction_cosine_max
        or separation < policy.normalized_separation_min
    ):
        return EventDecision(EventClass.NOT_SUPPORTED, "directions_not_separated")
    if left_valid != right_valid:
        return EventDecision(EventClass.SELECTOR_PROBLEM, "only_one_endpoint_valid")
    if not left_valid and not right_valid:
        return EventDecision(EventClass.NOT_SUPPORTED, "neither_endpoint_valid")
    if not rgbd_valid or not temporal:
        return EventDecision(
            EventClass.UNIDENTIFIABLE_WITH_TRANSLATION_ONLY,
            "independent_observation_barrier_unavailable",
        )
    if barrier > policy.barrier_margin:
        return EventDecision(
            EventClass.MULTIMODAL_VELOCITY_SUPPORTED,
            "two_valid_endpoints_with_interior_rgbd_barrier",
        )
    return EventDecision(
        EventClass.CONTINUOUS_REDUNDANCY,
        "two_valid_endpoints_connected_without_barrier",
    )
