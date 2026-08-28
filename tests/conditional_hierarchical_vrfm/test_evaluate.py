from __future__ import annotations

import copy
import unittest
from unittest import mock

import numpy as np

from pre_experiments.conditional_hierarchical_vrfm import evaluate
from pre_experiments.conditional_hierarchical_vrfm.evaluate import (
    classify_stage_a,
    evaluate_latent_targets,
)
from pre_experiments.conditional_hierarchical_vrfm.basis import canonical_basis_sha256


def _poses(count: int, x: float) -> np.ndarray:
    value = np.broadcast_to(np.eye(4, dtype=np.float64), (count, 4, 4)).copy()
    value[:, 0, 3] = x
    return value


class LatentEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        frame_ids = np.arange(500, dtype=np.int64)
        baseline = _poses(500, 0.0)
        gt = _poses(500, 1.0)
        coverage = np.zeros((4, 500), dtype=np.float64)
        coverage[:, 100:400] = 1.0
        fused = np.full((4, 500, 4, 4), np.nan, dtype=np.float64)
        fused[:, 100:400] = _poses(300, 0.9)
        decoded = np.broadcast_to(baseline, (4, 500, 4, 4)).copy()
        decoded[:, 100:400, 0, 3] = 0.75
        self.long_context = {
            "scene": np.asarray("scene0000_00", dtype="U32"),
            "frame_ids": frame_ids,
            "camera_tokens": np.zeros((500, 2048), dtype=np.float32),
            "baseline_c2w": baseline,
            "source_sha256": np.asarray("a" * 64, dtype="U64"),
        }
        self.labels = {
            "scene": np.asarray("scene0000_00", dtype="U32"),
            "frame_ids": frame_ids,
            "gt_c2w": gt,
            "gt_scene_scale": np.asarray(2.0),
            "baseline_c2w_raw": baseline,
            "oracle_scene": np.asarray("scene0000_00", dtype="U32"),
            "oracle_frame_digest": np.asarray("b" * 64, dtype="U64"),
            "oracle_fit_count": np.asarray(500),
            "oracle_scale": np.asarray(1.0),
            "oracle_rotation": np.eye(3),
            "oracle_translation": np.zeros(3),
            "oracle_rank": np.asarray(3),
            "oracle_condition": np.asarray(1.0),
            "oracle_digest": np.asarray("c" * 64, dtype="U64"),
            "window_weights": np.ones(9),
            "window_masks": np.ones((4, 9), dtype=np.uint8),
            "coverage_weights": coverage,
            "fused_c2w": fused,
            "variant_utilities": np.full(4, 0.9),
            "source_sha256": np.asarray("a" * 64, dtype="U64"),
            "formal_label_sha256": np.asarray("d" * 64, dtype="U64"),
            "checkpoint_sha256": np.asarray("e" * 64, dtype="U64"),
            "git_commit": np.asarray("f" * 40, dtype="U40"),
            "artifact_sha256": "1" * 64,
        }
        self.targets = {
            "scene": np.asarray("scene0000_00", dtype="U32"),
            "frame_ids": frame_ids,
            "teacher_variant_ids": np.arange(4),
            "teacher_window_masks": np.ones((4, 9), dtype=np.uint8),
            "coverage_masks": (coverage > 0).astype(np.uint8),
            "residual_coefficients": np.zeros((4, 32, 2048), dtype=np.float32),
            "decoded_c2w_raw": decoded,
            "optimization_steps": np.full(4, 250),
            "initial_losses": np.ones(4),
            "final_losses": np.zeros(4),
            "basis_sha256": np.asarray(canonical_basis_sha256(), dtype="U64"),
            "source_sha256": np.asarray("a" * 64, dtype="U64"),
            "teacher_sha256": np.asarray("1" * 64, dtype="U64"),
            "checkpoint_sha256": np.asarray("e" * 64, dtype="U64"),
            "git_commit": np.asarray("f" * 40, dtype="U40"),
        }

    def test_metrics_use_one_baseline_frozen_oracle_for_every_variant(self) -> None:
        with mock.patch.object(evaluate, "fit_frozen_oracle", side_effect=AssertionError):
            metrics = evaluate_latent_targets(self.long_context, self.targets, self.labels)
        self.assertEqual(metrics["variant_count"], 4)
        self.assertAlmostEqual(metrics["mean_covered_utility"], 0.75)
        self.assertAlmostEqual(metrics["teacher_retention"], 0.75 / 0.9)

    def test_saved_scene_scale_controls_uncovered_drift(self) -> None:
        self.targets["decoded_c2w_raw"][:, :100, 0, 3] = 0.2
        metrics = evaluate_latent_targets(self.long_context, self.targets, self.labels)
        expected = np.sqrt((100 * 0.2**2) / 200) / 2.0
        self.assertAlmostEqual(metrics["uncovered_drift_fraction"], expected)

    def test_cross_artifact_semantics_are_not_trusted(self) -> None:
        self.targets["teacher_sha256"] = np.asarray("9" * 64, dtype="U64")
        with self.assertRaisesRegex(ValueError, "teacher digest"):
            evaluate_latent_targets(self.long_context, self.targets, self.labels)

    def test_evaluation_rejects_non_so3_saved_pose(self) -> None:
        self.targets["decoded_c2w_raw"][0, 10, :3, :3] *= 2.0
        with self.assertRaisesRegex(ValueError, "SO\(3\)"):
            evaluate_latent_targets(self.long_context, self.targets, self.labels)

    def test_nonpositive_teacher_reference_is_rejected(self) -> None:
        self.labels["fused_c2w"][:, 100:400] = _poses(300, 0.0)
        with self.assertRaisesRegex(ValueError, "teacher-reference"):
            evaluate_latent_targets(self.long_context, self.targets, self.labels)


class StageAGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.expected = tuple(f"scene{index:04d}_00" for index in range(10))

    def passing_scenes(self) -> list[dict[str, object]]:
        return [
            {
                "scene": scene, "variant_count": 4, "all_finite": True,
                "teacher_retention": 0.8, "mean_full_scene_utility": 0.02,
                "mean_rotation_delta_deg": 0.01, "uncovered_drift_fraction": 0.001,
            }
            for scene in self.expected
        ]

    def test_stage_a_requires_every_frozen_gate(self) -> None:
        result = classify_stage_a(
            self.passing_scenes(), expected_scenes=self.expected,
            prediction_contract_clean=True,
        )
        self.assertEqual(result["classification"], "LATENT_TARGETS_READY")
        harmed = self.passing_scenes()
        harmed[0]["mean_full_scene_utility"] = -0.0101
        result = classify_stage_a(
            harmed, expected_scenes=self.expected, prediction_contract_clean=True,
        )
        self.assertEqual(result["classification"], "LATENT_LIFT_FAILED")
        self.assertIn("per_scene_harm", result["failed_gates"])

    def test_gate_rejects_nan_hidden_behind_all_finite_and_duplicate_scene(self) -> None:
        rows = self.passing_scenes()
        rows[0]["teacher_retention"] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            classify_stage_a(rows, expected_scenes=self.expected, prediction_contract_clean=True)
        rows = self.passing_scenes()
        rows[1]["scene"] = rows[0]["scene"]
        with self.assertRaisesRegex(ValueError, "exact expected"):
            classify_stage_a(rows, expected_scenes=self.expected, prediction_contract_clean=True)

    def test_complete_scientific_failure_remains_classified(self) -> None:
        rows = self.passing_scenes()
        result = classify_stage_a(rows, expected_scenes=self.expected, prediction_contract_clean=False)
        self.assertEqual(result["classification"], "LATENT_LIFT_FAILED")
        self.assertEqual(result["failed_gates"], ["leakage_audit"])


if __name__ == "__main__":
    unittest.main()
