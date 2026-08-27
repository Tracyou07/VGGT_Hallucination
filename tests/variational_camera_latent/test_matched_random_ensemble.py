from __future__ import annotations

import json
import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np


try:
    from pre_experiments.variational_camera_latent import matched_random_ensemble
except ImportError:
    matched_random_ensemble = None


class MatchedRandomEnsembleTests(unittest.TestCase):
    def _module(self):
        if matched_random_ensemble is None:
            self.fail("20-Q matched-random ensemble is not implemented")
        return matched_random_ensemble

    @staticmethod
    def _plan_kwargs() -> dict[str, object]:
        return {
            "master_seed": 2026082701,
            "transform_identity_sha256": "a" * 64,
            "reference_prediction_manifest_sha256": "b" * 64,
            "reference_transform_sha256": "c" * 64,
            "producer_git_commit": "d" * 40,
            "camera_head_checkpoint_sha256": "e" * 64,
            "source_manifest_sha256": "f" * 64,
            "candidate_manifest_sha256": "1" * 64,
            "vrfm_prediction_manifest_sha256": "2" * 64,
            "scenes": [f"scene{index:04d}_00" for index in range(10)],
        }

    def _score_bindings(self):
        plan = self._module().build_matched_random_ensemble_plan(
            **self._plan_kwargs()
        )
        expected = plan["replicates"]
        actual = [
            {
                "replicate_index": row["replicate_index"],
                "replicate_seed": row["replicate_seed"],
                "transform_sha256": row["expected_transform_sha256"],
            }
            for row in expected
        ]
        return expected, actual

    def test_plan_freezes_twenty_new_unique_seeds_and_transforms(self) -> None:
        module = self._module()
        first = module.build_matched_random_ensemble_plan(**self._plan_kwargs())
        second = module.build_matched_random_ensemble_plan(**self._plan_kwargs())

        self.assertEqual(first, second)
        self.assertEqual(first["replicate_count"], 20)
        self.assertEqual(first["formal_replicate_indices"], list(range(20)))
        self.assertIs(first["observed_pilot_included_in_formal_null"], False)
        self.assertEqual(
            [row["replicate_id"] for row in first["replicates"]],
            [f"formal_null_{index:03d}" for index in range(20)],
        )
        seeds = [row["replicate_seed"] for row in first["replicates"]]
        transforms = [row["expected_transform_sha256"] for row in first["replicates"]]
        self.assertEqual(len(set(seeds)), 20)
        self.assertEqual(len(set(transforms)), 20)
        self.assertNotIn("c" * 64, transforms)
        self.assertTrue(all(0 <= seed < 2**63 for seed in seeds))
        self.assertEqual(
            first["replicates"][0]["replicate_seed"], 1466835178304884443
        )
        self.assertEqual(
            first["replicates"][0]["seed_material_sha256"],
            "945b3ed2b2e8cedbbe0fd732e825258ee61b165cafe35972bee68d21cf5223cb",
        )
        self.assertEqual(
            first["replicates"][0]["expected_transform_sha256"],
            "cc78f63be94dbec5c32a2203d53aa19977aeea4e233e04788263602cca0d3fd9",
        )
        self.assertEqual(
            first["replicates"][-1]["replicate_seed"], 3857096706355983964
        )
        self.assertEqual(
            first["replicates"][-1]["seed_material_sha256"],
            "35872904c6ad765cc227cee57da16586feacf16dff006d8495113ee41d0bb061",
        )
        self.assertEqual(
            first["replicates"][-1]["expected_transform_sha256"],
            "ef38eff98e58b33438059506fbaf11bd7d8abd1e4d6a19144c3d53168a85c135",
        )

        serialized = json.dumps(first, sort_keys=True).lower()
        for forbidden in ("privileged", "prepared_gt", "error_label", "quality_label"):
            self.assertNotIn(forbidden, serialized)

    def test_plan_paths_are_disjoint_and_budget_is_fixed_per_replicate(self) -> None:
        module = self._module()
        plan = module.build_matched_random_ensemble_plan(**self._plan_kwargs())
        roots = [row["prediction_root"] for row in plan["replicates"]]
        manifests = [row["prediction_manifest_path"] for row in plan["replicates"]]
        completions = [row["prediction_completion_path"] for row in plan["replicates"]]

        self.assertEqual(len(set(roots)), 20)
        self.assertEqual(len(set(manifests)), 20)
        self.assertEqual(len(set(completions)), 20)
        self.assertEqual(plan["matched_oracle_budget_per_replicate"]["directions"], 2560)
        self.assertEqual(plan["matched_oracle_budget_per_replicate"]["grid_cells"], 20480)
        self.assertEqual(plan["matched_oracle_budget_per_replicate"]["unique_poses"], 18000)
        self.assertEqual(plan["random_family_budget"]["directions"], 51200)
        self.assertEqual(plan["random_family_budget"]["grid_cells"], 409600)
        self.assertEqual(plan["random_family_budget"]["unique_poses"], 358480)
        self.assertEqual(plan["matched_oracle_budget_per_replicate"]["no_op_count_per_overlap"], 1)
        self.assertEqual(plan["prediction_contract"]["decode_context_frames"], 500)
        self.assertEqual(plan["prediction_contract"]["camera_iterations"], 4)
        self.assertIs(plan["prediction_contract"]["same_transform_across_all_scenes"], True)

    def test_plan_resume_is_exact_and_refuses_overwrite(self) -> None:
        module = self._module()
        plan = module.build_matched_random_ensemble_plan(**self._plan_kwargs())
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "plan.json"
            first = module.write_matched_random_ensemble_plan(destination, plan)
            second = module.write_matched_random_ensemble_plan(destination, plan)
            self.assertEqual(first, second)
            original = destination.read_bytes()

            changed = {**plan, "master_seed": int(plan["master_seed"]) + 1}
            with self.assertRaisesRegex(ValueError, "20-Q plan"):
                module.write_matched_random_ensemble_plan(destination, changed)
            self.assertEqual(destination.read_bytes(), original)

    def test_plan_validation_recomputes_seed_transform_and_order(self) -> None:
        module = self._module()
        plan = module.build_matched_random_ensemble_plan(**self._plan_kwargs())
        for mutate in (
            lambda value: value["replicates"].pop(),
            lambda value: value["replicates"][0].update(
                replicate_seed=value["replicates"][1]["replicate_seed"]
            ),
            lambda value: value["replicates"].reverse(),
            lambda value: value["replicates"][0].update(
                expected_transform_sha256="0" * 64
            ),
        ):
            with self.subTest(mutation=mutate):
                changed = copy.deepcopy(plan)
                mutate(changed)
                with self.assertRaisesRegex(ValueError, "20-Q plan"):
                    module.validate_matched_random_ensemble_plan(changed)

    def test_summary_uses_scene_then_transform_as_the_units(self) -> None:
        module = self._module()
        identity = np.full((10, 8), 0.20, dtype=np.float64)
        random = np.stack(
            [np.full((10, 8), index / 100.0, dtype=np.float64) for index in range(20)]
        )
        # A single spectacular overlap must not dominate the scene median.
        random[0, 0, 0] = 1.0
        expected, actual = self._score_bindings()

        summary = module.summarize_matched_random_ensemble(
            identity,
            random,
            replicate_indices=list(range(20)),
            expected_replicates=expected,
            actual_replicate_bindings=actual,
            observed_pilot_score=0.91,
        )

        self.assertAlmostEqual(summary["identity_score"], 0.20)
        self.assertAlmostEqual(summary["replicates"][0]["score"], 0.0)
        self.assertEqual(summary["identity_rank_descending_best_tie"], 1)
        self.assertEqual(summary["identity_rank_descending_worst_tie"], 1)
        self.assertAlmostEqual(summary["p_identity_unusually_good"], 1 / 21)
        self.assertAlmostEqual(summary["p_identity_unusually_bad"], 1.0)
        self.assertAlmostEqual(summary["p_identity_two_sided"], 2 / 21)
        self.assertEqual(summary["formal_null_replicate_count"], 20)
        self.assertEqual(summary["randomization_unit"], "structured_null_transform")
        self.assertEqual(summary["observed_pilot_score"], 0.91)
        self.assertIs(summary["observed_pilot_included_in_p_values"], False)

    def test_ties_count_conservatively_in_both_one_sided_p_values(self) -> None:
        module = self._module()
        identity = np.full((10, 8), 0.10, dtype=np.float64)
        values = [0.05] * 9 + [0.10] * 2 + [0.15] * 9
        random = np.stack(
            [np.full((10, 8), value, dtype=np.float64) for value in values]
        )
        expected, actual = self._score_bindings()

        summary = module.summarize_matched_random_ensemble(
            identity,
            random,
            replicate_indices=list(range(20)),
            expected_replicates=expected,
            actual_replicate_bindings=actual,
        )

        self.assertAlmostEqual(summary["p_identity_unusually_good"], 12 / 21)
        self.assertAlmostEqual(summary["p_identity_unusually_bad"], 12 / 21)
        self.assertEqual(summary["identity_rank_descending_best_tie"], 10)
        self.assertEqual(summary["identity_rank_descending_worst_tie"], 12)
        self.assertEqual(summary["null_less_count"], 9)
        self.assertEqual(summary["null_tie_count"], 2)
        self.assertEqual(summary["null_greater_count"], 9)
        self.assertEqual(summary["p_identity_two_sided"], 1.0)

    def test_global_score_is_mean_of_scene_medians_not_pooled_median(self) -> None:
        module = self._module()
        identity = np.ones((10, 8), dtype=np.float64)
        identity[0] = 0.0
        identity[0, -1] = 1.0
        random = np.zeros((20, 10, 8), dtype=np.float64)
        expected, actual = self._score_bindings()

        summary = module.summarize_matched_random_ensemble(
            identity,
            random,
            replicate_indices=list(range(20)),
            expected_replicates=expected,
            actual_replicate_bindings=actual,
        )

        self.assertEqual(summary["identity_scene_scores"], [0.0] + [1.0] * 9)
        self.assertAlmostEqual(summary["identity_score"], 0.9)
        self.assertAlmostEqual(float(np.median(identity)), 1.0)

    def test_summary_requires_exactly_twenty_unique_formal_replicates(self) -> None:
        module = self._module()
        identity = np.zeros((10, 8), dtype=np.float64)
        random = np.zeros((20, 10, 8), dtype=np.float64)
        expected, actual = self._score_bindings()
        with self.assertRaisesRegex(ValueError, "exactly 20"):
            module.summarize_matched_random_ensemble(
                identity,
                random[:19],
                replicate_indices=list(range(19)),
                expected_replicates=expected,
                actual_replicate_bindings=actual,
            )
        with self.assertRaisesRegex(ValueError, "unique replicate"):
            module.summarize_matched_random_ensemble(
                identity,
                random,
                replicate_indices=[0] * 20,
                expected_replicates=expected,
                actual_replicate_bindings=actual,
            )
        bad_indices = list(range(20))
        bad_indices[0] = False
        with self.assertRaisesRegex(ValueError, "indices must be integers"):
            module.summarize_matched_random_ensemble(
                identity,
                random,
                replicate_indices=bad_indices,
                expected_replicates=expected,
                actual_replicate_bindings=actual,
            )

    def test_summary_rejects_out_of_range_benefit_and_pilot_binding(self) -> None:
        module = self._module()
        identity = np.zeros((10, 8), dtype=np.float64)
        random = np.zeros((20, 10, 8), dtype=np.float64)
        expected, actual = self._score_bindings()
        random[0, 0, 0] = 1.01
        with self.assertRaisesRegex(ValueError, r"in \[0, 1\]"):
            module.summarize_matched_random_ensemble(
                identity,
                random,
                replicate_indices=list(range(20)),
                expected_replicates=expected,
                actual_replicate_bindings=actual,
            )

        random[0, 0, 0] = 0.0
        actual[0] = {**actual[0], "transform_sha256": "c" * 64}
        with self.assertRaisesRegex(ValueError, "frozen 20-Q plan"):
            module.summarize_matched_random_ensemble(
                identity,
                random,
                replicate_indices=list(range(20)),
                expected_replicates=expected,
                actual_replicate_bindings=actual,
            )


if __name__ == "__main__":
    unittest.main()
