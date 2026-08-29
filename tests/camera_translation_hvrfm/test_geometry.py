from __future__ import annotations

import math
import unittest
import warnings

import numpy as np

from pre_experiments.camera_translation_hvrfm.geometry import (
    apply_translation_endpoint,
    baseline_fill_teacher_centers,
    build_translation_endpoint,
    prediction_scale,
)


FRAME_COUNT = 500
ENDPOINT_COUNT = 4


def _camera_system() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    frame_ids = np.arange(1000, 1000 + FRAME_COUNT, dtype=np.int64)
    centers = np.zeros((FRAME_COUNT, 3), dtype=np.float64)
    centers[:, 0] = np.arange(FRAME_COUNT, dtype=np.float64) - 249.5

    c2w = np.broadcast_to(np.eye(4, dtype=np.float64), (FRAME_COUNT, 4, 4)).copy()
    c2w[:, :3, 3] = centers

    pose = np.zeros((FRAME_COUNT, 9), dtype=np.float32)
    pose[:, :3] = (-centers).astype(np.float32)
    pose[:, 6] = 1.0
    pose[:, 7] = np.linspace(0.7, 1.1, FRAME_COUNT, dtype=np.float32)
    pose[:, 8] = np.linspace(0.8, 1.2, FRAME_COUNT, dtype=np.float32)

    coverage = np.ones((ENDPOINT_COUNT, FRAME_COUNT), dtype=np.uint8)
    coverage[0, ::7] = 0
    coverage[1, 100:160] = 0
    coverage[2, 300:] = 0
    coverage[3, :25] = 0

    offsets = np.asarray(
        [[1.0, -2.0, 0.5], [-0.25, 0.75, 1.5], [2.0, 0.0, -1.0], [0.5, 1.0, 2.0]],
        dtype=np.float64,
    )
    teachers = centers[None, :, :] + offsets[:, None, :]
    teachers[coverage == 0] = np.nan
    return frame_ids, c2w, pose, teachers, coverage


class PredictionScaleTests(unittest.TestCase):
    def test_uses_center_rms_from_baseline_c2w(self) -> None:
        _, c2w, _, _, _ = _camera_system()
        expected = math.sqrt((FRAME_COUNT * FRAME_COUNT - 1) / 12.0)
        self.assertEqual(prediction_scale(c2w), expected)

    def test_rejects_degenerate_malformed_nonfinite_and_improper_poses(self) -> None:
        _, valid, _, _, _ = _camera_system()
        cases: dict[str, np.ndarray] = {
            "wrong_dtype": valid.astype(np.float32),
            "wrong_shape": valid[:-1],
            "nonfinite": valid.copy(),
            "bad_homogeneous": valid.copy(),
            "reflection": valid.copy(),
            "shear": valid.copy(),
            "zero_scale": np.broadcast_to(
                np.eye(4, dtype=np.float64), (FRAME_COUNT, 4, 4)
            ).copy(),
        }
        cases["nonfinite"][0, 0, 0] = np.nan
        cases["bad_homogeneous"][0, 3, 0] = 1e-8
        cases["reflection"][0, :3, :3] = np.diag([-1.0, 1.0, 1.0])
        cases["shear"][0, 0, 1] = 3e-6
        for name, value in cases.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                prediction_scale(value)

    def test_rejects_near_degenerate_prediction_scale(self) -> None:
        _, c2w, _, _, _ = _camera_system()
        c2w[:, :3, 3] *= 1e-15
        with self.assertRaises(ValueError):
            prediction_scale(c2w)


class TranslationEndpointTests(unittest.TestCase):
    def test_fills_only_uncovered_teacher_centers_without_nan_arithmetic(self) -> None:
        ids, c2w, _, teachers, coverage = _camera_system()
        filled = baseline_fill_teacher_centers(
            long_frame_ids=ids,
            teacher_frame_ids=ids.copy(),
            baseline_c2w=c2w,
            teacher_centers=teachers,
            coverage_mask=coverage,
        )
        self.assertEqual(filled.dtype, np.dtype(np.float64))
        self.assertEqual(filled.shape, (ENDPOINT_COUNT, FRAME_COUNT, 3))
        self.assertTrue(np.isfinite(filled).all())
        baseline_centers = c2w[:, :3, 3]
        np.testing.assert_array_equal(
            filled[coverage == 0],
            np.broadcast_to(baseline_centers, filled.shape)[coverage == 0],
        )
        np.testing.assert_array_equal(filled[coverage != 0], teachers[coverage != 0])

    def test_builds_float32_endpoint_with_bitwise_zero_uncovered_entries(self) -> None:
        ids, c2w, pose, teachers, coverage = _camera_system()
        endpoint, filled, scale = build_translation_endpoint(
            long_frame_ids=ids,
            teacher_frame_ids=ids.copy(),
            baseline_c2w=c2w,
            baseline_pose_encoding=pose,
            teacher_centers=teachers,
            coverage_mask=coverage,
        )
        self.assertEqual(endpoint.shape, (ENDPOINT_COUNT, FRAME_COUNT, 3))
        self.assertEqual(endpoint.dtype, np.dtype(np.float32))
        self.assertEqual(filled.dtype, np.dtype(np.float64))
        self.assertEqual(scale, prediction_scale(c2w))
        expected_offsets = filled - c2w[None, :, :3, 3]
        expected = (-expected_offsets / scale).astype(np.float32)
        np.testing.assert_array_equal(endpoint[coverage != 0], expected[coverage != 0])
        self.assertTrue(np.all(endpoint.view(np.uint32)[coverage == 0] == 0))

    def test_rejects_bad_frames_masks_teachers_and_pose_replay(self) -> None:
        ids, c2w, pose, teachers, coverage = _camera_system()
        valid = {
            "long_frame_ids": ids,
            "teacher_frame_ids": ids.copy(),
            "baseline_c2w": c2w,
            "baseline_pose_encoding": pose,
            "teacher_centers": teachers,
            "coverage_mask": coverage,
        }
        cases: list[tuple[str, str, np.ndarray]] = []
        bad_teacher_ids = ids.copy()
        bad_teacher_ids[-1] += 1
        cases.append(("frame_mismatch", "teacher_frame_ids", bad_teacher_ids))
        cases.append(("frame_dtype", "long_frame_ids", ids.astype(np.int32)))
        cases.append(("mask_dtype", "coverage_mask", coverage.astype(np.int64)))
        bad_mask = coverage.copy()
        bad_mask[0, 0] = 2
        cases.append(("mask_nonbinary", "coverage_mask", bad_mask))
        partial_nan = teachers.copy()
        partial_nan[0, 0] = [np.nan, 0.0, np.nan]
        cases.append(("partial_nan", "teacher_centers", partial_nan))
        uncovered_inf = teachers.copy()
        uncovered_inf[0, 0] = [np.inf, np.inf, np.inf]
        cases.append(("uncovered_inf", "teacher_centers", uncovered_inf))
        covered_nan = teachers.copy()
        covered_nan[0, 1] = np.nan
        cases.append(("covered_nan", "teacher_centers", covered_nan))
        cases.append(("teacher_dtype", "teacher_centers", teachers.astype(np.float32)))
        wrong_translation = pose.copy()
        wrong_translation[0, 0] += 0.25
        cases.append(("pose_translation", "baseline_pose_encoding", wrong_translation))
        wrong_quaternion = pose.copy()
        wrong_quaternion[0, 3:7] = [0.0, 0.0, 1.0, 0.0]
        cases.append(("pose_rotation", "baseline_pose_encoding", wrong_quaternion))
        cases.append(("pose_dtype", "baseline_pose_encoding", pose.astype(np.float64)))
        reflected = c2w.copy()
        reflected[0, :3, :3] = np.diag([-1.0, 1.0, 1.0])
        cases.append(("reflection", "baseline_c2w", reflected))
        for name, key, value in cases:
            arguments = dict(valid)
            arguments[key] = value
            with self.subTest(name=name), self.assertRaises(ValueError):
                build_translation_endpoint(**arguments)

    def test_pose_replay_tolerance_is_independent_of_absolute_world_gauge(self) -> None:
        ids, c2w, pose, teachers, coverage = _camera_system()
        gauge_offset = np.float64(1_000_000.0)
        c2w[:, 0, 3] += gauge_offset
        teachers[..., 0] += gauge_offset
        pose[:, :3] = (-c2w[:, :3, 3]).astype(np.float32)
        build_translation_endpoint(
            long_frame_ids=ids,
            teacher_frame_ids=ids,
            baseline_c2w=c2w,
            baseline_pose_encoding=pose,
            teacher_centers=teachers,
            coverage_mask=coverage,
        )
        wrong_pose = pose.copy()
        wrong_pose[0, 0] += np.float32(1.0)
        with self.assertRaises(ValueError):
            build_translation_endpoint(
                long_frame_ids=ids,
                teacher_frame_ids=ids,
                baseline_c2w=c2w,
                baseline_pose_encoding=wrong_pose,
                teacher_centers=teachers,
                coverage_mask=coverage,
            )

    def test_rejects_float32_endpoint_overflow_without_runtime_warning(self) -> None:
        ids, c2w, pose, teachers, coverage = _camera_system()
        teachers[0, 1, 0] = np.finfo(np.float64).max
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with self.assertRaises(ValueError):
                build_translation_endpoint(
                    long_frame_ids=ids,
                    teacher_frame_ids=ids,
                    baseline_c2w=c2w,
                    baseline_pose_encoding=pose,
                    teacher_centers=teachers,
                    coverage_mask=coverage,
                )
        self.assertEqual(caught, [])

    def test_apply_changes_only_translation_and_round_trips_teacher_centers(self) -> None:
        ids, c2w, pose, teachers, coverage = _camera_system()
        endpoint, filled, scale = build_translation_endpoint(
            long_frame_ids=ids,
            teacher_frame_ids=ids,
            baseline_c2w=c2w,
            baseline_pose_encoding=pose,
            teacher_centers=teachers,
            coverage_mask=coverage,
        )
        corrected = apply_translation_endpoint(pose, endpoint, scale=scale)
        self.assertEqual(corrected.shape, (ENDPOINT_COUNT, FRAME_COUNT, 9))
        self.assertEqual(corrected.dtype, np.dtype(np.float32))
        expected_tail = np.broadcast_to(pose[None, :, 3:9], corrected[..., 3:9].shape)
        self.assertEqual(corrected[..., 3:9].tobytes(), expected_tail.tobytes())

        rotations_w2c = np.swapaxes(c2w[:, :3, :3], -1, -2)
        recovered = -np.einsum(
            "tij,kti->ktj", rotations_w2c, corrected[..., :3].astype(np.float64)
        )
        np.testing.assert_allclose(recovered, filled, atol=scale * 2e-7, rtol=2e-7)

    def test_zero_endpoint_is_a_bitwise_tiled_baseline_noop(self) -> None:
        _, _, pose, _, _ = _camera_system()
        pose[0, 0] = np.float32(-0.0)
        pose[1, 4] = np.float32(-0.0)
        zeros = np.zeros((ENDPOINT_COUNT, FRAME_COUNT, 3), dtype=np.float32)
        corrected = apply_translation_endpoint(pose, zeros, scale=17.0)
        expected = np.broadcast_to(pose[None], corrected.shape).copy()
        self.assertEqual(corrected.tobytes(), expected.tobytes())

    def test_mixed_endpoint_preserves_every_zero_component_bitwise(self) -> None:
        _, _, pose, _, _ = _camera_system()
        pose[:, :3] = np.float32(-0.0)
        pose[::2, :3] = np.float32(0.0)
        endpoint = np.zeros((ENDPOINT_COUNT, FRAME_COUNT, 3), dtype=np.float32)
        endpoint[0, 0, 0] = 0.5
        endpoint[0, 1, 1] = np.float32(-0.0)
        endpoint[1, 2, 2] = np.float32(-0.0)
        corrected = apply_translation_endpoint(pose, endpoint, scale=2.0)
        expected = np.broadcast_to(pose[None], corrected.shape).copy()
        expected[0, 0, 0] = 1.0
        self.assertEqual(corrected.tobytes(), expected.tobytes())

    def test_apply_rejects_wrong_shape_dtype_nonfinite_and_scale(self) -> None:
        _, _, pose, _, _ = _camera_system()
        endpoint = np.zeros((ENDPOINT_COUNT, FRAME_COUNT, 3), dtype=np.float32)
        cases = (
            (pose.astype(np.float64), endpoint, 1.0),
            (pose[:-1], endpoint, 1.0),
            (pose, endpoint.astype(np.float64), 1.0),
            (pose, endpoint[:-1], 1.0),
            (pose, endpoint, 0.0),
            (pose, endpoint, np.nan),
        )
        for baseline, value, scale in cases:
            with self.subTest(scale=scale, shape=value.shape), self.assertRaises(ValueError):
                apply_translation_endpoint(baseline, value, scale=scale)
        endpoint[0, 0, 0] = np.inf
        with self.assertRaises(ValueError):
            apply_translation_endpoint(pose, endpoint, scale=1.0)

    def test_normalized_endpoint_is_invariant_to_full_camera_system_sim3(self) -> None:
        ids, c2w, pose, teachers, coverage = _camera_system()
        original, _, original_scale = build_translation_endpoint(
            long_frame_ids=ids,
            teacher_frame_ids=ids,
            baseline_c2w=c2w,
            baseline_pose_encoding=pose,
            teacher_centers=teachers,
            coverage_mask=coverage,
        )

        angle = 0.37
        cosine, sine = math.cos(angle), math.sin(angle)
        rotation = np.asarray(
            [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        factor = 3.25
        translation = np.asarray([4.0, -7.0, 2.5], dtype=np.float64)
        transformed_c2w = c2w.copy()
        transformed_c2w[:, :3, :3] = np.einsum(
            "ij,tjk->tik", rotation, c2w[:, :3, :3]
        )
        transformed_c2w[:, :3, 3] = (
            factor * (c2w[:, :3, 3] @ rotation.T) + translation
        )
        transformed_teachers = teachers.copy()
        transformed_teachers[coverage != 0] = (
            factor * (teachers[coverage != 0] @ rotation.T) + translation
        )

        transformed_pose = pose.copy()
        old_t = pose[:, :3].astype(np.float64)
        transformed_pose[:, :3] = (
            factor * old_t - (rotation.T @ translation)[None]
        ).astype(np.float32)
        transformed_pose[:, 3:7] = np.asarray(
            [0.0, 0.0, math.sin(-angle / 2.0), math.cos(-angle / 2.0)],
            dtype=np.float32,
        )
        transformed, _, transformed_scale = build_translation_endpoint(
            long_frame_ids=ids,
            teacher_frame_ids=ids,
            baseline_c2w=transformed_c2w,
            baseline_pose_encoding=transformed_pose,
            teacher_centers=transformed_teachers,
            coverage_mask=coverage,
        )
        self.assertAlmostEqual(transformed_scale, factor * original_scale, places=10)
        np.testing.assert_allclose(transformed, original, atol=2e-6, rtol=2e-6)


if __name__ == "__main__":
    unittest.main()
