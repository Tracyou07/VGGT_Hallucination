from __future__ import annotations

import unittest
from pathlib import Path
import tempfile

import numpy as np

from pre_experiments.variational_camera_selector.safety_gate import (
    GatePolicy,
    apply_gate_policy,
    extract_gate_features,
    load_gate_policy,
    write_gate_policy,
)


def _score_fixture() -> dict[str, np.ndarray]:
    full = np.full((8, 225), -2.0, dtype=np.float64)
    residual = np.full((8, 225), -2.0, dtype=np.float64)
    alphas = np.full((8, 225), 0.02, dtype=np.float64)
    alphas[:, 0] = 0.0
    for row in range(8):
        proposed = row + 1
        full[row, 0] = 0.0
        full[row, proposed] = 2.0
        full[row, proposed + 16] = 1.0
        residual[row, 0] = 0.0
        residual[row, proposed] = 1.0
        residual[row, proposed + 32] = 0.5
    return {
        "full_context_scores": full,
        "residual_only_scores": residual,
        "alphas": alphas,
    }


class SafetyGatePredictionTests(unittest.TestCase):
    def test_feature_normalization_is_invariant_to_score_scale_and_offset(self) -> None:
        # Catches a gate that thresholds raw checkpoint-dependent score magnitudes.
        arrays = _score_fixture()
        reference = extract_gate_features(arrays)
        transformed = {
            **arrays,
            "full_context_scores": arrays["full_context_scores"] * 7.0 + 13.0,
            "residual_only_scores": arrays["residual_only_scores"] * 3.0 - 9.0,
        }
        observed = extract_gate_features(transformed)

        np.testing.assert_array_equal(observed.proposed_indices, reference.proposed_indices)
        np.testing.assert_allclose(observed.advantage_z, reference.advantage_z)
        np.testing.assert_allclose(observed.prominence_z, reference.prominence_z)
        np.testing.assert_allclose(observed.residual_support_z, reference.residual_support_z)

    def test_gate_chooses_only_the_ranker_top_candidate_or_noop(self) -> None:
        # Catches accidental re-ranking or replacement with a GT-informed candidate.
        arrays = _score_fixture()
        features = extract_gate_features(arrays)
        policy = GatePolicy(
            deployable=True,
            max_alpha=0.05,
            min_advantage_z=0.1,
            min_prominence_z=0.1,
            min_residual_support_z=0.1,
            require_top_agreement=False,
        )
        decision = apply_gate_policy(features, policy)

        np.testing.assert_array_equal(decision.selected_indices, features.proposed_indices)
        self.assertTrue(decision.gate_pass.all())

        arrays["alphas"][3, features.proposed_indices[3]] = 0.5
        rejected = apply_gate_policy(extract_gate_features(arrays), policy)
        self.assertEqual(int(rejected.selected_indices[3]), 0)
        self.assertFalse(bool(rejected.gate_pass[3]))

    def test_alpha_cap_includes_the_matching_float32_grid_value(self) -> None:
        # The production shards store alphas as float32 (e.g. .05 is slightly above .05).
        arrays = _score_fixture()
        arrays["alphas"] = arrays["alphas"].astype(np.float32)
        features = extract_gate_features(arrays)
        arrays["alphas"][np.arange(8), features.proposed_indices] = np.float32(0.05)
        features = extract_gate_features(arrays)
        policy = GatePolicy(
            deployable=True,
            max_alpha=0.05,
            min_advantage_z=0.0,
            min_prominence_z=0.0,
            min_residual_support_z=0.0,
            require_top_agreement=False,
        )

        decision = apply_gate_policy(features, policy)

        self.assertTrue(decision.gate_pass.all())

    def test_required_ranker_agreement_rejects_disagreement(self) -> None:
        # Catches an agreement flag that is recorded but not enforced.
        arrays = _score_fixture()
        arrays["residual_only_scores"][2, 100] = 4.0
        features = extract_gate_features(arrays)
        policy = GatePolicy(
            deployable=True,
            max_alpha=0.05,
            min_advantage_z=0.0,
            min_prominence_z=0.0,
            min_residual_support_z=-10.0,
            require_top_agreement=True,
        )
        decision = apply_gate_policy(features, policy)

        self.assertEqual(int(decision.selected_indices[2]), 0)
        self.assertFalse(bool(decision.gate_pass[2]))

    def test_non_deployable_policy_is_fail_closed(self) -> None:
        # Catches fallback code that still emits a correction when calibration failed.
        features = extract_gate_features(_score_fixture())
        policy = GatePolicy.fail_closed()
        decision = apply_gate_policy(features, policy)

        np.testing.assert_array_equal(decision.selected_indices, np.zeros(8, dtype=np.int64))
        self.assertFalse(decision.gate_pass.any())

    def test_gate_rejects_a_policy_above_the_frozen_alpha_cap(self) -> None:
        features = extract_gate_features(_score_fixture())
        policy = GatePolicy(
            deployable=True,
            max_alpha=0.5,
            min_advantage_z=0.0,
            min_prominence_z=0.0,
            min_residual_support_z=0.0,
            require_top_agreement=False,
        )

        with self.assertRaisesRegex(ValueError, "0.2"):
            apply_gate_policy(features, policy)

    def test_malformed_or_nonfinite_scores_are_rejected(self) -> None:
        # Catches silent NaN propagation into a seemingly valid gate decision.
        arrays = _score_fixture()
        arrays["full_context_scores"][0, 1] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            extract_gate_features(arrays)

    def test_frozen_policy_round_trip_contains_no_label_fields(self) -> None:
        # Catches publishing calibration labels inside the inference policy artifact.
        policy = GatePolicy(
            deployable=True,
            max_alpha=0.05,
            min_advantage_z=1.0,
            min_prominence_z=0.1,
            min_residual_support_z=0.0,
            require_top_agreement=False,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.json"
            write_gate_policy(
                path,
                policy,
                training_scenes=tuple(f"scene_{index}" for index in range(8)),
                fit_manifest_sha256="a" * 64,
            )
            observed = load_gate_policy(path)
            text = path.read_text(encoding="utf-8").lower()

        self.assertEqual(observed, policy)
        self.assertFalse(
            any(
                fragment in text
                for fragment in ("utility", "quality", "error", "depth", "privileged", "ground_truth")
            )
        )


if __name__ == "__main__":
    unittest.main()
