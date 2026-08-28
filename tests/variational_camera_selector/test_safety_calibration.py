from __future__ import annotations

import unittest

import numpy as np

from pre_experiments.variational_camera_selector.safety_calibration import (
    GateAcceptance,
    GateSceneObservation,
    evaluate_gate_policy,
    fit_frozen_gate,
    default_policy_candidates,
    observation_from_arrays,
    search_gate_policy,
)
from pre_experiments.variational_camera_selector.safety_gate import (
    GateFeatures,
    GatePolicy,
)


def _scene(scene: str, *, all_negative: bool = False) -> GateSceneObservation:
    proposed_indices = np.arange(1, 9, dtype=np.int64)
    proposed_alphas = np.asarray([0.02, 0.2] * 4, dtype=np.float64)
    features = GateFeatures(
        proposed_indices=proposed_indices,
        proposed_alphas=proposed_alphas,
        advantage_z=np.full(8, 2.0, dtype=np.float64),
        prominence_z=np.full(8, 1.0, dtype=np.float64),
        residual_support_z=np.full(8, 1.0, dtype=np.float64),
        top_agreement=np.ones(8, dtype=np.bool_),
    )
    utility = (
        np.full(8, -0.2, dtype=np.float64)
        if all_negative
        else np.asarray([0.10, -1.0] * 4, dtype=np.float64)
    )
    return GateSceneObservation(scene=scene, features=features, proposed_utility=utility)


def _policies() -> tuple[GatePolicy, GatePolicy]:
    common = dict(
        deployable=True,
        min_advantage_z=0.0,
        min_prominence_z=0.0,
        min_residual_support_z=0.0,
        require_top_agreement=False,
    )
    safe = GatePolicy(max_alpha=0.05, **common)
    unsafe = GatePolicy(max_alpha=0.2, **common)
    return safe, unsafe


class SafetyGateCalibrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.acceptance = GateAcceptance(
            min_coverage=0.25,
            min_positive_precision=0.70,
            max_catastrophic_rate=0.05,
            catastrophic_utility=-0.05,
            min_mean_utility=0.0,
            min_worst_scene_mean=-0.01,
        )

    def test_policy_metrics_count_rejected_proposals_as_noop(self) -> None:
        # Catches reports that drop rejected overlaps and inflate the mean utility.
        safe, _ = _policies()
        metrics = evaluate_gate_policy([_scene("scene_a")], safe)

        self.assertEqual(metrics.selected_count, 4)
        self.assertAlmostEqual(metrics.coverage, 0.5)
        self.assertAlmostEqual(metrics.positive_precision, 1.0)
        self.assertAlmostEqual(metrics.mean_utility, 0.05)
        self.assertAlmostEqual(metrics.worst_scene_mean, 0.05)

    def test_search_rejects_aggressive_policy_and_selects_safe_policy(self) -> None:
        # Catches optimizing mean ranker score while ignoring real downside risk.
        safe, unsafe = _policies()
        result = search_gate_policy(
            [_scene("scene_a"), _scene("scene_b")],
            acceptance=self.acceptance,
            candidates=(unsafe, safe),
        )

        self.assertTrue(result.policy.deployable)
        self.assertEqual(result.policy, safe)
        self.assertEqual(result.metrics.catastrophic_count, 0)

    def test_search_returns_fail_closed_when_no_candidate_is_safe(self) -> None:
        # Catches forced deployment when all observed corrections are harmful.
        safe, unsafe = _policies()
        result = search_gate_policy(
            [_scene("scene_a", all_negative=True), _scene("scene_b", all_negative=True)],
            acceptance=self.acceptance,
            candidates=(safe, unsafe),
        )

        self.assertFalse(result.policy.deployable)
        self.assertEqual(result.metrics.selected_count, 0)
        self.assertEqual(result.metrics.mean_utility, 0.0)

    def test_frozen_gate_requires_scene_level_crossfit_acceptance(self) -> None:
        # Catches freezing a rule that only looks good after fitting and evaluating on all rows.
        safe, unsafe = _policies()
        result = fit_frozen_gate(
            [_scene(f"scene_{index}") for index in range(4)],
            acceptance=self.acceptance,
            candidates=(unsafe, safe),
        )

        self.assertTrue(result.policy.deployable)
        self.assertEqual(result.policy, safe)
        self.assertEqual(result.crossfit_metrics.scene_count, 4)
        self.assertGreater(result.crossfit_metrics.mean_utility, 0.0)

    def test_default_search_grid_never_allows_aggressive_alpha(self) -> None:
        # Catches reintroducing the alpha=.5/1 failure mode through the policy grid.
        candidates = default_policy_candidates()
        self.assertGreater(len(candidates), 1)
        self.assertEqual(len(candidates), len(set(candidates)))
        self.assertTrue(all(policy.deployable for policy in candidates))
        self.assertLessEqual(max(policy.max_alpha for policy in candidates), 0.2)

    def test_observation_join_rejects_mismatched_sample_ids(self) -> None:
        # Catches attaching one scene's GT outcomes to another scene's ranker scores.
        score = _score_arrays("scene_a")
        evaluation = _evaluation_arrays("scene_a", score)
        observation = observation_from_arrays(score, evaluation)
        self.assertEqual(observation.scene, "scene_a")
        np.testing.assert_array_equal(observation.proposed_utility, np.arange(8) / 100.0)

        evaluation["source_sample_ids"] = np.asarray(
            [f"wrong:{index}" for index in range(8)], dtype="U32"
        )
        with self.assertRaisesRegex(ValueError, "sample"):
            observation_from_arrays(score, evaluation)


def _score_arrays(scene: str) -> dict[str, np.ndarray]:
    scores = np.full((8, 225), -2.0, dtype=np.float64)
    residual = np.full((8, 225), -2.0, dtype=np.float64)
    alphas = np.full((8, 225), 0.02, dtype=np.float64)
    alphas[:, 0] = 0.0
    selected = np.arange(1, 9, dtype=np.int64)
    for row, index in enumerate(selected):
        scores[row, 0] = 0.0
        scores[row, index] = 2.0
        residual[row, 0] = 0.0
        residual[row, index] = 1.0
    return {
        "scene": np.asarray(scene, dtype="U32"),
        "source_sample_ids": np.asarray(
            [f"{scene}:{index}" for index in range(8)], dtype="U64"
        ),
        "full_context_scores": scores,
        "residual_only_scores": residual,
        "alphas": alphas,
        "full_context_selected_indices": selected,
    }


def _evaluation_arrays(
    scene: str, score: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    return {
        "scene": np.asarray(scene, dtype="U32"),
        "source_sample_ids": score["source_sample_ids"].copy(),
        "full_context_selected_indices": score["full_context_selected_indices"].copy(),
        "full_context_utility": np.arange(8, dtype=np.float64) / 100.0,
    }


if __name__ == "__main__":
    unittest.main()
