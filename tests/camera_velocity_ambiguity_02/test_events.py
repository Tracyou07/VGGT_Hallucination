from __future__ import annotations

import unittest

from pre_experiments.camera_velocity_ambiguity_02.contracts import EvidenceSource
from pre_experiments.camera_velocity_ambiguity_02.events import (
    EventClass,
    EventPolicy,
    MetricEvidence,
    classify_event,
)


POLICY = EventPolicy(
    direction_cosine_max=0.25,
    normalized_separation_min=0.1,
    barrier_margin=0.2,
)


def _evidence(**changes: object) -> dict[str, MetricEvidence]:
    values: dict[str, tuple[object, EvidenceSource]] = {
        "alignment_valid": (True, EvidenceSource.PREDICTION_ONLY),
        "direction_evaluable": (True, EvidenceSource.PREDICTION_ONLY),
        "flattened_cosine": (-0.5, EvidenceSource.PREDICTION_ONLY),
        "normalized_separation": (0.5, EvidenceSource.PREDICTION_ONLY),
        "left_endpoint_valid": (True, EvidenceSource.PRIVILEGED_GT),
        "right_endpoint_valid": (True, EvidenceSource.PRIVILEGED_GT),
        "rgbd_valid": (True, EvidenceSource.OBSERVATION_RGBD),
        "interior_barrier": (0.5, EvidenceSource.OBSERVATION_RGBD),
        "temporal_support": (True, EvidenceSource.OBSERVATION_RGBD),
    }
    for name, value in changes.items():
        source = values[name][1]
        values[name] = (value, source)
    return {name: MetricEvidence(value, source) for name, (value, source) in values.items()}


class EventClassifierTest(unittest.TestCase):
    def test_classifies_all_four_scientific_cases(self) -> None:
        not_supported = classify_event(
            _evidence(flattened_cosine=0.9), POLICY
        )
        selector = classify_event(
            _evidence(right_endpoint_valid=False), POLICY
        )
        continuous = classify_event(
            _evidence(interior_barrier=0.05), POLICY
        )
        multimodal = classify_event(_evidence(), POLICY)

        self.assertEqual(not_supported.event_class, EventClass.NOT_SUPPORTED)
        self.assertEqual(selector.event_class, EventClass.SELECTOR_PROBLEM)
        self.assertEqual(continuous.event_class, EventClass.CONTINUOUS_REDUNDANCY)
        self.assertEqual(multimodal.event_class, EventClass.MULTIMODAL_VELOCITY_SUPPORTED)

    def test_missing_or_invalid_rgbd_barrier_is_explicitly_unidentifiable(self) -> None:
        for changes in (
            {"rgbd_valid": False},
            {"temporal_support": False},
        ):
            with self.subTest(changes=changes):
                result = classify_event(_evidence(**changes), POLICY)
                self.assertEqual(
                    result.event_class,
                    EventClass.UNIDENTIFIABLE_WITH_TRANSLATION_ONLY,
                )

    def test_metric_firewall_rejects_unknown_wrong_source_or_presentation_fields(self) -> None:
        unknown = _evidence()
        unknown["mystery"] = MetricEvidence(1.0, EvidenceSource.PREDICTION_ONLY)
        with self.assertRaises(ValueError):
            classify_event(unknown, POLICY)

        wrong_source = _evidence()
        wrong_source["interior_barrier"] = MetricEvidence(
            0.5, EvidenceSource.PRIVILEGED_GT
        )
        with self.assertRaises(ValueError):
            classify_event(wrong_source, POLICY)

        presentation = _evidence()
        presentation["flattened_cosine"] = MetricEvidence(
            -0.5, EvidenceSource.PRESENTATION_ONLY
        )
        with self.assertRaises(ValueError):
            classify_event(presentation, POLICY)


if __name__ == "__main__":
    unittest.main()
