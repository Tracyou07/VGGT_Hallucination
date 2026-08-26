from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from pre_experiments.camera_velocity_ambiguity_02.state import (
    StudyPhase,
    StudyState,
    apply_development_policy,
    fit_and_freeze_policy,
    load_frozen_policy,
)
from pre_experiments.camera_velocity_ambiguity_02.statistics import (
    aggregate_scene_prevalence,
    paired_scene_bootstrap,
)


CALIBRATION = tuple(f"scene{index:04d}_00" for index in range(10))
DEVELOPMENT = tuple(f"scene{index:04d}_00" for index in range(10, 50))


def _calibration_rows() -> list[dict[str, object]]:
    rows = []
    for scene in CALIBRATION:
        for index in range(8):
            rows.append(
                {
                    "scene": scene,
                    "pair_id": f"{scene}/{index}",
                    "route": "primary",
                    "direction_evaluable": True,
                    "flattened_cosine": -0.8 + index * 0.1,
                    "normalized_separation": 0.1 + index * 0.02,
                    "control_barrier": 0.01 * index,
                }
            )
    return rows


class StateAndStatisticsTest(unittest.TestCase):
    def test_state_machine_is_monotonic_and_rejects_development_before_freeze(self) -> None:
        state = StudyState.initial()
        self.assertEqual(state.phase, StudyPhase.INPUTS_VERIFIED)
        with self.assertRaises(ValueError):
            state.transition(StudyPhase.DEVELOPMENT_COMPLETE)
        state = state.transition(StudyPhase.CALIBRATION_COMPLETE)
        state = state.transition(StudyPhase.POLICY_FROZEN)
        state = state.transition(StudyPhase.DEVELOPMENT_COMPLETE)
        self.assertEqual(state.phase, StudyPhase.DEVELOPMENT_COMPLETE)

    def test_policy_requires_exact_calibration_and_is_hash_bound_and_nonoverwritable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "policy.json"
            policy = fit_and_freeze_policy(
                path,
                _calibration_rows(),
                calibration_scenes=CALIBRATION,
                protocol_digest="1" * 64,
                input_digest="2" * 64,
                git_commit="3" * 40,
            )
            self.assertEqual(len(policy.policy_digest), 64)
            loaded = load_frozen_policy(
                path,
                protocol_digest="1" * 64,
                input_digest="2" * 64,
                git_commit="3" * 40,
            )
            self.assertEqual(loaded, policy)
            with self.assertRaises(FileExistsError):
                fit_and_freeze_policy(
                    path,
                    _calibration_rows(),
                    calibration_scenes=CALIBRATION,
                    protocol_digest="1" * 64,
                    input_digest="2" * 64,
                    git_commit="3" * 40,
                )

            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["barrier_margin"] += 1.0
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_frozen_policy(
                    path,
                    protocol_digest="1" * 64,
                    input_digest="2" * 64,
                    git_commit="3" * 40,
                )

        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError):
                fit_and_freeze_policy(
                    Path(temporary) / "policy.json",
                    _calibration_rows()[:-1],
                    calibration_scenes=CALIBRATION,
                    protocol_digest="1" * 64,
                    input_digest="2" * 64,
                    git_commit="3" * 40,
                )

    def test_development_requires_exact_40_scenes_primary_rows_and_no_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            policy = fit_and_freeze_policy(
                Path(temporary) / "policy.json",
                _calibration_rows(),
                calibration_scenes=CALIBRATION,
                protocol_digest="1" * 64,
                input_digest="2" * 64,
                git_commit="3" * 40,
            )
            rows = [
                {"scene": scene, "route": "primary", "value": 0.0}
                for scene in DEVELOPMENT
            ]
            self.assertEqual(len(apply_development_policy(policy, rows, DEVELOPMENT)), 40)
            with self.assertRaises(ValueError):
                apply_development_policy(
                    policy, rows, DEVELOPMENT, threshold_overrides={"barrier_margin": 0.0}
                )

    def test_primary_scene_prevalence_and_seeded_paired_bootstrap_are_deterministic(self) -> None:
        rows = [
            {"scene": "a", "route": "primary", "positive": True},
            {"scene": "a", "route": "primary", "positive": False},
            {"scene": "a", "route": "secondary", "positive": True},
            {"scene": "b", "route": "primary", "positive": True},
        ]
        prevalence = aggregate_scene_prevalence(rows, positive_field="positive")
        self.assertEqual(prevalence, {"a": 0.5, "b": 1.0})
        left = {"a": 0.5, "b": 1.0, "c": 0.25}
        right = {"a": 0.25, "b": 0.5, "c": 0.0}
        first = paired_scene_bootstrap(left, right, seed=33, samples=10_000)
        second = paired_scene_bootstrap(left, right, seed=33, samples=10_000)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first.mean_difference, np.mean([0.25, 0.5, 0.25]))
        self.assertEqual(first.samples, 10_000)


if __name__ == "__main__":
    unittest.main()
