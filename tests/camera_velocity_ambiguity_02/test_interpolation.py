from __future__ import annotations

import inspect
import unittest

import numpy as np

from pre_experiments.camera_velocity_ambiguity_02.contracts import ProtocolViolation
from pre_experiments.camera_velocity_ambiguity_02.frozen_oracle import fit_frozen_oracle
from pre_experiments.camera_velocity_ambiguity_02.interpolation import (
    TranslationCurve,
    assert_translation_curve_convex,
    build_translation_candidates,
    evaluate_translation_candidates,
)


def _poses(centers: np.ndarray) -> np.ndarray:
    poses = np.repeat(np.eye(4, dtype=np.float64)[None], len(centers), axis=0)
    poses[:, :3, 3] = centers
    return poses


def _full_trajectory() -> np.ndarray:
    t = np.linspace(0.0, 8.0, 500)
    return _poses(np.stack((t, np.sin(t), np.cos(t * 0.7)), axis=1))


class TranslationInterpolationTest(unittest.TestCase):
    def test_endpoints_midpoint_and_global_rotation_fov_are_preserved(self) -> None:
        global_pose = _poses(np.asarray([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]]))
        global_pose[:, :3, :3] = np.asarray(
            [[[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]] * 2
        )
        fov = np.asarray([[40.0, 45.0], [41.0, 46.0]])
        left = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        right = -left
        candidates = build_translation_candidates(
            global_pose,
            left,
            right,
            alphas=(0.0, 0.5, 1.0),
            global_fov=fov,
        )

        np.testing.assert_allclose(
            candidates[0].c2w[:, :3, 3], global_pose[:, :3, 3] + left
        )
        np.testing.assert_allclose(
            candidates[1].c2w[:, :3, 3], global_pose[:, :3, 3]
        )
        np.testing.assert_allclose(
            candidates[2].c2w[:, :3, 3], global_pose[:, :3, 3] + right
        )
        for candidate in candidates:
            np.testing.assert_array_equal(candidate.c2w[:, :3, :3], global_pose[:, :3, :3])
            np.testing.assert_array_equal(candidate.fov, fov)
            self.assertFalse(np.shares_memory(candidate.fov, fov))
        self.assertNotIn("pose_encoding", inspect.signature(build_translation_candidates).parameters)

    def test_frozen_oracle_metrics_include_ate_rte_and_same_transform_identity(self) -> None:
        prediction = _full_trajectory()
        ground_truth = prediction.copy()
        ground_truth[:, :3, 3] = prediction[:, :3, 3] * 2.0 + [3.0, -2.0, 1.0]
        oracle = fit_frozen_oracle(
            "scene0000_00", np.arange(500), prediction, ground_truth
        )
        global_shared = prediction[100:103]
        raw_gt_shared = ground_truth[100:103]
        left = np.zeros((3, 3), dtype=np.float64)
        right = np.asarray([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]])
        candidates = build_translation_candidates(
            global_shared, left, right, alphas=(0.0, 0.5, 1.0)
        )
        curve = evaluate_translation_candidates(oracle, candidates, raw_gt_shared)

        self.assertEqual(curve.transform_digest, oracle.transform_digest)
        self.assertEqual(curve.per_frame_l2.shape, (3, 3))
        self.assertAlmostEqual(curve.mean_l2[0], 0.0, places=10)
        self.assertAlmostEqual(curve.rms_l2[0], 0.0, places=10)
        self.assertAlmostEqual(curve.rte_rms[0], 0.0, places=10)
        self.assertGreater(curve.rte_rms[-1], 0.0)
        assert_translation_curve_convex(curve)

    def test_convexity_guard_covers_per_frame_mean_and_rms_and_fails_closed(self) -> None:
        alphas = np.asarray([0.0, 0.5, 1.0])
        valid = TranslationCurve(
            alphas=alphas,
            per_frame_l2=np.asarray([[1.0, 2.0], [0.0, 1.0], [1.0, 2.0]]),
            mean_l2=np.asarray([1.5, 0.5, 1.5]),
            rms_l2=np.asarray([np.sqrt(2.5), np.sqrt(0.5), np.sqrt(2.5)]),
            rte_rms=np.zeros(3),
            transform_digest="a" * 64,
        )
        assert_translation_curve_convex(valid)

        for field in ("per_frame_l2", "mean_l2", "rms_l2"):
            values = {
                "alphas": valid.alphas.copy(),
                "per_frame_l2": valid.per_frame_l2.copy(),
                "mean_l2": valid.mean_l2.copy(),
                "rms_l2": valid.rms_l2.copy(),
                "rte_rms": valid.rte_rms.copy(),
                "transform_digest": valid.transform_digest,
            }
            if field == "per_frame_l2":
                values[field][1, 0] = 5.0
            else:
                values[field][1] = 5.0
            with self.subTest(field=field), self.assertRaises(ProtocolViolation):
                assert_translation_curve_convex(TranslationCurve(**values))


if __name__ == "__main__":
    unittest.main()
