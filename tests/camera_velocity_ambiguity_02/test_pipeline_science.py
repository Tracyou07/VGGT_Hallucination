from __future__ import annotations

import unittest
from pathlib import Path
import json
import tempfile

from pre_experiments.camera_velocity_ambiguity_02.pipeline import (
    resolve_prediction_commit,
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

    def test_existing_run_keeps_prediction_commit_across_analysis_bugfix(self) -> None:
        old = "1" * 40
        current = "2" * 40
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.json"
            self.assertEqual(resolve_prediction_commit(path, current), current)
            path.write_text(json.dumps({"git_commit": old}), encoding="utf-8")
            self.assertEqual(resolve_prediction_commit(path, current), old)
            path.write_text(json.dumps({"prediction_git_commit": old}), encoding="utf-8")
            self.assertEqual(resolve_prediction_commit(path, current), old)


if __name__ == "__main__":
    unittest.main()
