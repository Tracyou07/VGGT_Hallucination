from __future__ import annotations

import unittest

import numpy as np

from pre_experiments.camera_velocity_ambiguity_02.interpolation import TranslationCandidate
from pre_experiments.camera_velocity_ambiguity_02.rgbd_gate import (
    FrozenObservationScale,
    RGBDConfig,
    build_rgbd_observations,
    compute_rgbd_energy,
    evaluate_rgbd_path,
    freeze_observation_scale,
)


def _poses(second_translation: tuple[float, float, float]) -> np.ndarray:
    poses = np.repeat(np.eye(4, dtype=np.float64)[None], 2, axis=0)
    poses[1, :3, 3] = second_translation
    return poses


def _payload(*, depth_value: float = 1.0) -> dict[str, np.ndarray]:
    height = width = 8
    stripe = (np.arange(width) % 2).astype(np.float64)
    image = np.tile(stripe[None, :, None], (height, 1, 3))
    rgb = np.stack((image, image))
    depth = np.full((2, height, width), depth_value, dtype=np.float64)
    intrinsics = np.repeat(np.eye(3, dtype=np.float64)[None], 2, axis=0)
    return {
        "frame_ids": np.asarray([10, 20], dtype=np.int64),
        "rgb": rgb,
        "depth": depth,
        "intrinsics": intrinsics,
    }


CONFIG = RGBDConfig(
    pixel_stride=1,
    pixel_margin=2,
    min_correspondences=8,
    photometric_weight=1.0,
    depth_weight=1.0,
    free_space_weight=1.0,
    occlusion_weight=1.0,
    coverage_weight=1.0,
    flat_energy_tolerance=1e-8,
    scale_candidates=(0.5, 1.0, 2.0),
)


class RGBDObservationGateTest(unittest.TestCase):
    def test_planar_scene_correct_pose_is_low_and_wrong_translation_is_higher(self) -> None:
        observations = build_rgbd_observations(_payload())
        correct = compute_rgbd_energy(observations, _poses((0.0, 0.0, 0.0)), CONFIG)
        wrong = compute_rgbd_energy(observations, _poses((1.0, 0.0, 0.0)), CONFIG)

        self.assertEqual(correct.direction_count, 2)
        self.assertGreaterEqual(correct.correspondence_count, 16)
        self.assertAlmostEqual(correct.total, 0.0, places=12)
        self.assertGreater(wrong.total, correct.total)
        self.assertGreater(wrong.photometric, 0.0)

    def test_reports_free_space_occlusion_and_coverage_penalties(self) -> None:
        observations = build_rgbd_observations(_payload())
        z_shift = compute_rgbd_energy(observations, _poses((0.0, 0.0, 0.2)), CONFIG)
        outside = compute_rgbd_energy(observations, _poses((20.0, 0.0, 0.0)), CONFIG)

        self.assertGreater(z_shift.free_space, 0.0)
        self.assertGreater(z_shift.occlusion, 0.0)
        self.assertGreater(outside.coverage, 0.0)
        self.assertLess(outside.correspondence_count, z_shift.correspondence_count)

    def test_insufficient_correspondence_is_invalid(self) -> None:
        payload = _payload()
        payload["depth"][:] = 0.0
        result = compute_rgbd_energy(
            build_rgbd_observations(payload), _poses((0.0, 0.0, 0.0)), CONFIG
        )
        self.assertFalse(result.valid)
        self.assertEqual(result.reason, "insufficient_correspondence")

    def test_freezes_one_interior_scene_scale_and_flags_boundary_solution(self) -> None:
        observations = build_rgbd_observations(_payload())
        fitted = freeze_observation_scale(
            observations, _poses((2.0, 0.0, 0.0)), CONFIG
        )
        self.assertTrue(fitted.valid, fitted.reason)
        self.assertEqual(fitted.scale, 1.0)
        self.assertEqual(len(fitted.digest), 64)

        boundary = freeze_observation_scale(
            observations, _poses((0.0, 0.0, 0.0)), CONFIG
        )
        self.assertFalse(boundary.valid)
        self.assertEqual(boundary.reason, "scale_at_search_boundary")

    def test_two_low_endpoints_with_high_interior_produces_observation_barrier(self) -> None:
        observations = build_rgbd_observations(_payload())
        candidates = (
            TranslationCandidate(0.0, _poses((0.0, 0.0, 0.0)), None),
            TranslationCandidate(0.5, _poses((1.0, 0.0, 0.0)), None),
            TranslationCandidate(1.0, _poses((2.0, 0.0, 0.0)), None),
        )
        scale = FrozenObservationScale(
            scale=1.0,
            valid=True,
            reason=None,
            candidate_scales=(0.5, 1.0, 2.0),
            candidate_energies=(1.0, 0.0, 1.0),
            digest="a" * 64,
        )
        result = evaluate_rgbd_path(observations, candidates, scale, CONFIG)

        self.assertTrue(result.valid, result.reason)
        self.assertEqual(result.scale_digest, scale.digest)
        self.assertAlmostEqual(result.energies[0], 0.0, places=12)
        self.assertAlmostEqual(result.energies[-1], 0.0, places=12)
        self.assertGreater(result.energies[1], result.energies[0])
        self.assertGreater(result.interior_barrier, 0.0)

        flat = evaluate_rgbd_path(
            observations,
            tuple(
                TranslationCandidate(alpha, _poses((0.0, 0.0, 0.0)), None)
                for alpha in (0.0, 0.5, 1.0)
            ),
            scale,
            CONFIG,
        )
        self.assertFalse(flat.valid)
        self.assertEqual(flat.reason, "flat_energy_curve")

    def test_observation_payload_rejects_gt_and_presentation_fields(self) -> None:
        for name in ("gt_c2w", "ground_truth_pose", "fastvggt_plot_metric"):
            payload = _payload()
            payload[name] = np.zeros(1)
            with self.subTest(name=name), self.assertRaises(ValueError):
                build_rgbd_observations(payload)


if __name__ == "__main__":
    unittest.main()
