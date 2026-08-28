from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
from torch import nn

from pre_experiments.conditional_hierarchical_vrfm.teacher import (
    TeacherVariantSet,
    build_teacher_variants,
    build_variant_window_masks,
    summarize_teacher_upper_bound,
)
from pre_experiments.camera_velocity_ambiguity_02.frozen_oracle import FrozenOracle
from pre_experiments.variational_camera_latent.source import save_source_shard


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _poses_from_raw(raw: torch.Tensor) -> torch.Tensor:
    """A controlled Camera Head boundary with c2w translations in raw[..., 6:]."""
    poses = torch.eye(4, dtype=torch.float64, device=raw.device).repeat(
        raw.shape[0], raw.shape[1], 1, 1
    )
    poses[..., :3, 3] = raw[..., 6:].to(torch.float64)
    return poses


class _TokenCameraHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))

    def decode_pose_tokens(self, tokens: torch.Tensor, *, num_iterations: int) -> list[torch.Tensor]:
        self.last_iterations = num_iterations
        return [tokens[..., :9]]


class _StatefulTokenCameraHead(_TokenCameraHead):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("running_value", torch.zeros(1))
        self.saw_eval = False

    def decode_pose_tokens(self, tokens: torch.Tensor, *, num_iterations: int) -> list[torch.Tensor]:
        self.saw_eval = not self.training
        self.running_value.add_(1.0)
        self.anchor.data.add_(1.0)
        return super().decode_pose_tokens(tokens, num_iterations=num_iterations)


class TeacherVariantTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.source_path = root / "source.npz"
        self.prepared_scene = root / "scene0000_00"
        self.fake_camera_head = _TokenCameraHead()
        frame_ids = np.arange(500, dtype=np.int64)
        t = np.linspace(0.0, 8.0, 500)
        gt_centers = np.stack((t, np.sin(t), np.cos(0.7 * t)), axis=1)
        baseline_centers = gt_centers + np.stack(
            (0.2 * np.sin(3.0 * t), 0.1 * np.cos(2.0 * t), 0.05 * np.sin(5.0 * t)),
            axis=1,
        )
        global_tokens = self._tokens(baseline_centers)
        short_tokens = np.stack(
            [self._tokens(gt_centers[start : start + 100]) for start in range(0, 401, 50)]
        )
        global_c2w = self._c2w(baseline_centers)
        arrays = {
            "global_frame_ids": frame_ids,
            "global_camera_tokens": global_tokens,
            "short_frame_ids": np.stack([frame_ids[start : start + 100] for start in range(0, 401, 50)]),
            "short_camera_tokens": short_tokens,
            "overlap_frame_ids": np.stack([frame_ids[start : start + 50] for start in range(50, 401, 50)]),
            "overlap_long_tokens": np.stack([global_tokens[start : start + 50] for start in range(50, 401, 50)]),
            "overlap_left_tokens": np.stack([short_tokens[index, 50:] for index in range(8)]),
            "overlap_right_tokens": np.stack([short_tokens[index + 1, :50] for index in range(8)]),
            "span_starts": np.arange(0, 400, 50, dtype=np.int64),
            "sample_ids": np.asarray([f"scene0000_00:overlap_{index:03d}" for index in range(8)], dtype="U64"),
            "global_pred_c2w": global_c2w,
            "overlap_long_c2w": np.stack([global_c2w[start : start + 50] for start in range(50, 401, 50)]),
        }
        save_source_shard(self.source_path, arrays)
        pose_directory = self.prepared_scene / "pose"
        pose_directory.mkdir(parents=True)
        for frame_id, pose in zip(frame_ids, self._c2w(gt_centers)):
            np.save(pose_directory / f"{frame_id}.npy", pose, allow_pickle=False)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def _tokens(centers: np.ndarray) -> np.ndarray:
        tokens = np.zeros((len(centers), 2048), dtype=np.float32)
        tokens[:, 0] = 1.0
        tokens[:, 4] = 1.0
        tokens[:, 6:9] = centers
        return tokens

    @staticmethod
    def _c2w(centers: np.ndarray) -> np.ndarray:
        poses = np.repeat(np.eye(4, dtype=np.float64)[None], len(centers), axis=0)
        poses[:, :3, 3] = centers
        return poses

    def test_variant_zero_uses_all_positive_windows_and_other_masks_are_stable(self) -> None:
        weights = np.array([0.2, 0.0, 0.1, 0.4, 0.3, 0.0, 0.2, 0.1, 0.5])
        first = build_variant_window_masks("scene0000_00", weights)
        second = build_variant_window_masks("scene0000_00", weights)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first[0], weights > 0)
        self.assertEqual(len({row.tobytes() for row in first}), 4)
        self.assertTrue(np.all(~first[:, weights == 0]))

    def test_masks_preserve_variant_zero_frame_coverage_when_distinct_subsets_exist(self) -> None:
        cases = {
            "all_positive": np.ones(9),
            "sparse_positive": np.array([0.5, 0.2, 0.3, 0.0, 0.4, 0.1, 0.3, 0.2, 0.6]),
        }
        for name, weights in cases.items():
            with self.subTest(name=name):
                masks = build_variant_window_masks("scene0000_00", weights)
                coverage = np.zeros((4, 500), dtype=bool)
                for variant, mask in enumerate(masks):
                    for index, selected in enumerate(mask):
                        if selected:
                            coverage[variant, index * 50 : index * 50 + 100] = True
                for variant in range(1, 4):
                    np.testing.assert_array_equal(coverage[variant], coverage[0])

    def test_teacher_builder_never_mutates_authenticated_prediction_source(self) -> None:
        before = sha256_file(self.source_path)
        with mock.patch(
            "pre_experiments.conditional_hierarchical_vrfm.teacher.pose_encoding_to_c2w",
            side_effect=_poses_from_raw,
        ):
            teachers = build_teacher_variants(
                self.source_path,
                self.prepared_scene,
                self.fake_camera_head,
                checkpoint_sha256="a" * 64,
                device=torch.device("cpu"),
            )
        self.assertEqual(sha256_file(self.source_path), before)
        self.assertEqual(teachers.scene, "scene0000_00")
        self.assertEqual(teachers.frame_ids.dtype, np.int64)
        self.assertEqual(teachers.aligned_short_c2w.shape, (9, 100, 4, 4))
        self.assertEqual(teachers.window_weights.shape, (9,))
        self.assertEqual(teachers.window_masks.shape, (4, 9))
        self.assertEqual(teachers.fused_c2w.shape, (4, 500, 4, 4))
        self.assertEqual(teachers.coverage_weights.shape, (4, 500))
        covered = teachers.coverage_weights > 0.0
        self.assertTrue(np.isfinite(teachers.fused_c2w[covered]).all())
        self.assertTrue(np.isnan(teachers.fused_c2w[~covered]).all())
        self.assertTrue(np.allclose(teachers.fused_c2w[covered, 3, :], [0.0, 0.0, 0.0, 1.0]))
        self.assertTrue(np.any(teachers.window_weights > 0.0))
        self.assertGreater(teachers.variant_utilities[0], 0.0)

    def test_builder_restores_frozen_head_state_and_binds_canonical_digest(self) -> None:
        head = _StatefulTokenCameraHead()
        head.train()
        parameter_before = head.anchor.detach().clone()
        buffer_before = head.running_value.detach().clone()
        with mock.patch(
            "pre_experiments.conditional_hierarchical_vrfm.teacher.pose_encoding_to_c2w",
            side_effect=_poses_from_raw,
        ):
            teachers = build_teacher_variants(
                self.source_path, self.prepared_scene, head,
                checkpoint_sha256="a" * 64, device=torch.device("cpu"),
            )
        self.assertTrue(head.training)
        self.assertTrue(head.saw_eval)
        torch.testing.assert_close(head.anchor, parameter_before)
        torch.testing.assert_close(head.running_value, buffer_before)
        self.assertEqual(teachers.checkpoint_sha256, "a" * 64)

    def test_builder_rejects_noncanonical_digest_and_head_device_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "canonical"):
            build_teacher_variants(
                self.source_path, self.prepared_scene, self.fake_camera_head,
                checkpoint_sha256="A" * 64, device=torch.device("cpu"),
            )
        with self.assertRaisesRegex(ValueError, "device"):
            build_teacher_variants(
                self.source_path, self.prepared_scene, self.fake_camera_head,
                checkpoint_sha256="a" * 64, device=torch.device("meta"),
            )

    def test_upper_bound_summary_replays_the_formal_ten_scene_labels_exactly(self) -> None:
        oracle = FrozenOracle(
            scene="scene0000_00", frame_digest="a" * 64, fit_count=500,
            scale=1.0, rotation=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            translation=(0.0, 0.0, 0.0), rank=3, condition=1.0, transform_digest="b" * 64,
        )
        coverage = np.zeros((4, 500), dtype=np.float64)
        coverage[:, :445] = 1.0
        poses = np.full((4, 500, 4, 4), np.nan, dtype=np.float64)
        poses[:, :445] = np.eye(4, dtype=np.float64)
        baseline = np.repeat(np.eye(4, dtype=np.float64)[None], 500, axis=0)
        baseline[:445, 0, 3] = 1.0
        poses[:, :445, 0, 3] = 1.0 - 0.1293578271441714
        gt = np.repeat(np.eye(4, dtype=np.float64)[None], 500, axis=0)
        covered = coverage[0] > 0.0
        baseline_error = baseline[covered, :3, 3] - gt[covered, :3, 3]
        teacher_error = poses[0, covered, :3, 3] - gt[covered, :3, 3]
        baseline_rms = float(np.sqrt(np.mean(np.sum(baseline_error * baseline_error, axis=1))))
        teacher_rms = float(np.sqrt(np.mean(np.sum(teacher_error * teacher_error, axis=1))))
        replay_utility = (baseline_rms - teacher_rms) / max(baseline_rms, 1e-12)
        teachers = [
            TeacherVariantSet(
                scene=f"scene{index:04d}_00", frame_ids=np.arange(500, dtype=np.int64),
                aligned_short_c2w=np.zeros((9, 100, 4, 4), dtype=np.float64),
                window_weights=np.zeros(9, dtype=np.float64),
                window_masks=np.ones((4, 9), dtype=bool), fused_c2w=poses,
                coverage_weights=coverage, oracle=oracle,
                checkpoint_sha256="a" * 64,
                variant_utilities=np.full(4, replay_utility, dtype=np.float64),
            )
            for index in range(10)
        ]
        summary = summarize_teacher_upper_bound(teachers)
        self.assertEqual(summary["positive_scene_count"], 10)
        self.assertAlmostEqual(float(summary["mean_coverage"]), 0.89, places=10)
        self.assertAlmostEqual(float(summary["mean_utility"]), 0.1293578271441714, places=10)


if __name__ == "__main__":
    unittest.main()
