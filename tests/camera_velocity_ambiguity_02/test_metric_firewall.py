from __future__ import annotations

import unittest

from pre_experiments.camera_velocity_ambiguity_02.analyze import analyze_pair_records
from pre_experiments.camera_velocity_ambiguity_02.events import EventPolicy


class MetricFirewallTest(unittest.TestCase):
    def test_rejects_unknown_plot_and_cross_layer_fields(self) -> None:
        prediction = {
            "sample_id": "s/p",
            "scene": "scene0000_00",
            "pair_id": "s/p",
            "route": "primary",
            "alignment_valid": True,
            "direction_evaluable": True,
            "flattened_cosine": -0.5,
            "normalized_separation": 0.1,
        }
        privileged = {
            "sample_id": "s/p",
            "left_endpoint_valid": True,
            "right_endpoint_valid": True,
            "global_rms": 2.0,
            "left_rms": 1.0,
            "right_rms": 1.0,
        }
        rgbd = {
            "sample_id": "s/p",
            "rgbd_valid": True,
            "interior_barrier": 0.2,
            "temporal_support": True,
        }
        policy = EventPolicy(0.0, 0.01, 0.1)
        for layer, field in (
            (prediction, "fastvggt_ate"),
            (prediction, "plot_score"),
            (prediction, "left_endpoint_valid"),
            (privileged, "interior_barrier"),
            (rgbd, "gt_ate"),
        ):
            tampered = dict(layer, **{field: 0.0})
            args = [prediction, privileged, rgbd, policy]
            args[[prediction, privileged, rgbd].index(layer)] = tampered
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "schema"):
                analyze_pair_records(*args)


if __name__ == "__main__":
    unittest.main()
