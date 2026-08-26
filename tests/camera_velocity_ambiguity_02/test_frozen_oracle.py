from __future__ import annotations

from dataclasses import FrozenInstanceError
import unittest
from unittest import mock

import numpy as np

from pre_experiments.camera_velocity_ambiguity_02.artifacts import frame_digest
from pre_experiments.camera_velocity_ambiguity_02.frozen_oracle import (
    OracleLimits,
    apply_frozen_oracle,
    evaluate_with_frozen_oracle,
    fit_frozen_oracle,
)


def _trajectory(count: int) -> np.ndarray:
    t = np.linspace(0.0, 8.0, count)
    centers = np.stack((t, np.sin(t), np.cos(t * 0.7)), axis=1)
    poses = np.repeat(np.eye(4, dtype=np.float64)[None], count, axis=0)
    poses[:, :3, 3] = centers
    return poses


class FrozenOracleTest(unittest.TestCase):
    def test_recovers_and_binds_exact_500_and_430_frame_scene_transforms(self) -> None:
        for scene, count in (("scene0000_00", 500), ("scene0150_00", 430)):
            with self.subTest(scene=scene):
                prediction = _trajectory(count)
                ground_truth = prediction.copy()
                ground_truth[:, :3, 3] = prediction[:, :3, 3] * 2.5 + [4.0, -3.0, 1.0]
                ids = np.arange(count, dtype=np.int64) * 3
                oracle = fit_frozen_oracle(scene, ids, prediction, ground_truth)

                self.assertEqual(oracle.scene, scene)
                self.assertEqual(oracle.fit_count, count)
                self.assertEqual(oracle.frame_digest, frame_digest(ids))
                self.assertAlmostEqual(oracle.scale, 2.5, places=10)
                self.assertEqual(len(oracle.transform_digest), 64)
                aligned = apply_frozen_oracle(oracle, prediction)
                np.testing.assert_allclose(aligned[:, :3, 3], ground_truth[:, :3, 3], atol=1e-10)
                with self.assertRaises(FrozenInstanceError):
                    oracle.scale = 1.0  # type: ignore[misc]

    def test_rejects_subset_nonfinite_zero_variance_condition_and_scale(self) -> None:
        prediction = _trajectory(500)
        ground_truth = prediction.copy()
        with self.assertRaisesRegex(ValueError, "exactly 500"):
            fit_frozen_oracle(
                "scene0000_00", np.arange(499), prediction[:499], ground_truth[:499]
            )

        nonfinite = ground_truth.copy()
        nonfinite[0, 0, 3] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            fit_frozen_oracle("scene0000_00", np.arange(500), prediction, nonfinite)

        constant = prediction.copy()
        constant[:, :3, 3] = 0.0
        with self.assertRaisesRegex(ValueError, "rank"):
            fit_frozen_oracle("scene0000_00", np.arange(500), constant, ground_truth)

        poor = prediction.copy()
        poor[:, 1:3, 3] *= 1e-12
        with self.assertRaisesRegex(ValueError, "condition"):
            fit_frozen_oracle(
                "scene0000_00",
                np.arange(500),
                poor,
                ground_truth,
                limits=OracleLimits(max_condition=100.0),
            )

        huge = prediction.copy()
        huge[:, :3, 3] *= 100.0
        with self.assertRaisesRegex(ValueError, "scale"):
            fit_frozen_oracle(
                "scene0000_00",
                np.arange(500),
                prediction,
                huge,
                limits=OracleLimits(max_scale=10.0),
            )

    def test_gt_input_is_bytewise_unchanged(self) -> None:
        prediction = _trajectory(500)
        ground_truth = prediction.copy()
        before = ground_truth.tobytes()
        fit_frozen_oracle("scene0000_00", np.arange(500), prediction, ground_truth)
        self.assertEqual(ground_truth.tobytes(), before)

    def test_candidate_evaluation_reuses_frozen_transform_without_refitting(self) -> None:
        prediction = _trajectory(500)
        ground_truth = prediction.copy()
        ground_truth[:, :3, 3] = prediction[:, :3, 3] * 1.7 + [2.0, 3.0, -4.0]
        oracle = fit_frozen_oracle(
            "scene0000_00", np.arange(500), prediction, ground_truth
        )
        subset = prediction[200:250]
        subset_gt = ground_truth[200:250]

        with mock.patch(
            "pre_experiments.camera_velocity_ambiguity_02.frozen_oracle.fit_frozen_oracle",
            side_effect=AssertionError("candidate-specific refit is forbidden"),
        ):
            for _ in range(7):
                result = evaluate_with_frozen_oracle(oracle, subset, subset_gt)
                self.assertAlmostEqual(result.rms_translation_error, 0.0, places=10)
                self.assertEqual(result.fit_transform_digest, oracle.transform_digest)


if __name__ == "__main__":
    unittest.main()
