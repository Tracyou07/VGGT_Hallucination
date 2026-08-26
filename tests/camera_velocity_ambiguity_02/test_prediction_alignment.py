from __future__ import annotations

import inspect
import unittest

import numpy as np

from pre_experiments.camera_velocity_ambiguity_02.geometry import (
    AlignmentLimits,
    align_local_to_global,
    build_pair_geometry,
    compute_residual_direction_metrics,
    global_scene_scale,
)
from pre_experiments.camera_velocity_ambiguity_02.units import build_overlap_units
from pre_experiments.local_global_consistency.windows import build_sliding_windows


def _poses(centers: np.ndarray) -> np.ndarray:
    values = np.asarray(centers, dtype=np.float64)
    poses = np.repeat(np.eye(4, dtype=np.float64)[None], len(values), axis=0)
    poses[:, :3, 3] = values
    return poses


POINTS = np.asarray(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.2],
        [0.0, 2.0, 0.4],
        [1.5, 1.0, 1.0],
        [2.0, -1.0, 0.7],
        [-0.5, 1.5, 1.8],
    ],
    dtype=np.float64,
)


class PredictionOnlyAlignmentTest(unittest.TestCase):
    def test_recovers_known_sim3_and_pure_gauge_copy_has_zero_residual(self) -> None:
        reference = _poses(POINTS)
        moving = _poses((POINTS - np.asarray([4.0, -3.0, 2.0])) / 2.5)
        scale = global_scene_scale(reference)
        result = align_local_to_global(reference, moving, scene_scale=scale)

        self.assertTrue(result.valid, result.exclusion_reason)
        self.assertEqual(result.fit_count, len(POINTS))
        self.assertEqual(result.rank, 3)
        self.assertAlmostEqual(result.scale, 2.5, places=10)
        self.assertAlmostEqual(result.rotation_determinant, 1.0, places=10)
        self.assertAlmostEqual(result.rms, 0.0, places=10)
        np.testing.assert_allclose(result.aligned_c2w[:, :3, 3], POINTS, atol=1e-10)

    def test_alignment_gates_rank_condition_scale_and_normalized_rms(self) -> None:
        limits = AlignmentLimits(
            min_rank=2,
            max_condition=100.0,
            min_scale=0.1,
            max_scale=10.0,
            max_normalized_rms=0.01,
        )
        collinear = np.asarray([[index, 0.0, 0.0] for index in range(6)])
        rank = align_local_to_global(
            _poses(collinear), _poses(collinear), scene_scale=1.0, limits=limits
        )
        self.assertFalse(rank.valid)
        self.assertEqual(rank.exclusion_reason, "rank_below_minimum")

        poorly_conditioned = POINTS.copy()
        poorly_conditioned[:, 1:] *= 1e-6
        condition = align_local_to_global(
            _poses(poorly_conditioned),
            _poses(poorly_conditioned),
            scene_scale=1.0,
            limits=limits,
        )
        self.assertFalse(condition.valid)
        self.assertEqual(condition.exclusion_reason, "condition_above_maximum")

        scale = align_local_to_global(
            _poses(POINTS * 100.0), _poses(POINTS), scene_scale=1.0, limits=limits
        )
        self.assertFalse(scale.valid)
        self.assertEqual(scale.exclusion_reason, "scale_out_of_range")

        warped = POINTS.copy()
        warped[-1] += [0.5, -0.7, 0.9]
        rms = align_local_to_global(
            _poses(POINTS), _poses(warped), scene_scale=global_scene_scale(_poses(POINTS)), limits=limits
        )
        self.assertFalse(rms.valid)
        self.assertEqual(rms.exclusion_reason, "normalized_rms_above_maximum")

    def test_signed_residual_metrics_distinguish_opposite_same_and_zero_directions(self) -> None:
        global_centers = np.zeros((2, 3), dtype=np.float64)
        left = np.asarray([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        opposite = -left
        result = compute_residual_direction_metrics(
            global_centers, left, opposite, scene_scale=1.0
        )
        np.testing.assert_array_equal(result.left_residual, left)
        np.testing.assert_array_equal(result.right_residual, opposite)
        self.assertTrue(result.direction_evaluable)
        self.assertAlmostEqual(result.flattened_cosine, -1.0)
        self.assertAlmostEqual(result.normalized_rms_separation, 2.0)
        self.assertAlmostEqual(result.per_frame_direction_agreement, 0.0)

        same = compute_residual_direction_metrics(
            global_centers, left, left, scene_scale=1.0
        )
        self.assertAlmostEqual(same.flattened_cosine, 1.0)
        self.assertAlmostEqual(same.normalized_rms_separation, 0.0)
        self.assertAlmostEqual(same.per_frame_direction_agreement, 1.0)

        zero = compute_residual_direction_metrics(
            global_centers, np.zeros_like(left), opposite, scene_scale=1.0
        )
        self.assertFalse(zero.direction_evaluable)
        self.assertIsNone(zero.flattened_cosine)
        self.assertIsNone(zero.per_frame_direction_agreement)

    def test_pair_alignment_uses_all_local_frames_not_only_shared_frames(self) -> None:
        global_pose = _poses(POINTS)
        windows = build_sliding_windows(np.arange(6), length=4, stride=2)
        unit = build_overlap_units("scene0000_00", windows, primary_overlap=2)[0]
        left = _poses((POINTS[:4] - [2.0, 1.0, -1.0]) / 1.5)
        right = _poses((POINTS[2:] - [-1.0, 3.0, 2.0]) / 0.75)

        result = build_pair_geometry(
            unit,
            global_c2w=global_pose,
            left_local_c2w=left,
            right_local_c2w=right,
            scene_scale=global_scene_scale(global_pose),
        )

        self.assertTrue(result.left_alignment.valid)
        self.assertTrue(result.right_alignment.valid)
        self.assertEqual(result.left_alignment.fit_count, 4)
        self.assertEqual(result.right_alignment.fit_count, 4)
        self.assertEqual(result.shared_frame_ids, (2, 3))
        np.testing.assert_allclose(result.metrics.left_residual, 0.0, atol=1e-10)
        np.testing.assert_allclose(result.metrics.right_residual, 0.0, atol=1e-10)
        self.assertFalse(result.metrics.direction_evaluable)

    def test_public_scientific_path_has_no_gt_parameter(self) -> None:
        for function in (
            align_local_to_global,
            build_pair_geometry,
            compute_residual_direction_metrics,
        ):
            for parameter in inspect.signature(function).parameters:
                self.assertFalse(parameter == "gt" or parameter.startswith("gt_"))
        with self.assertRaises(TypeError):
            build_pair_geometry(  # type: ignore[call-arg]
                None,
                global_c2w=np.empty((0, 4, 4)),
                left_local_c2w=np.empty((0, 4, 4)),
                right_local_c2w=np.empty((0, 4, 4)),
                scene_scale=1.0,
                gt_c2w=object(),
            )


if __name__ == "__main__":
    unittest.main()
