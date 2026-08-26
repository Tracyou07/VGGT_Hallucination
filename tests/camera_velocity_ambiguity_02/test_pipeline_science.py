from __future__ import annotations

import unittest

from pre_experiments.camera_velocity_ambiguity_02.pipeline import (
    endpoint_validities,
    provisional_smoke_policy,
)


class PipelineScienceTest(unittest.TestCase):
    def test_endpoint_validity_is_a_privileged_strict_improvement_label(self) -> None:
        self.assertEqual(endpoint_validities(2.0, 1.9, 2.1), (True, False))
        self.assertEqual(endpoint_validities(2.0, 2.0, 2.0), (False, False))
        with self.assertRaises(ValueError):
            endpoint_validities(0.0, 1.0, 1.0)

    def test_smoke_policy_is_fixed_and_not_data_fitted(self) -> None:
        policy = provisional_smoke_policy()
        self.assertEqual(policy.direction_cosine_max, 0.0)
        self.assertEqual(policy.normalized_separation_min, 1e-4)
        self.assertEqual(policy.barrier_margin, 1e-5)


if __name__ == "__main__":
    unittest.main()
