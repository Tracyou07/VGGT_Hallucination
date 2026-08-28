from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from pre_experiments.long_short_camera_head.labels import (
    build_privileged_labels,
    construct_privileged_arrays,
    fuse_teacher_trajectories,
    load_privileged_labels,
    positive_teacher_weight,
    save_privileged_labels,
)
from pre_experiments.variational_camera_latent.source import save_source_shard
from tests.long_short_camera_head.test_data import _source_arrays


class TokenPoseHead:
    def decode_pose_tokens(
        self, tokens: torch.Tensor, *, num_iterations: int
    ) -> list[torch.Tensor]:
        pose = tokens[..., :9]
        return [pose for _ in range(num_iterations)]


class TeacherLabelTests(unittest.TestCase):
    def test_constructed_labels_freeze_one_scene_alignment_and_quality_weights(self) -> None:
        time = np.linspace(0.0, 1.0, 500)
        gt = np.repeat(np.eye(4, dtype=np.float64)[None], 500, axis=0)
        gt[:, 0, 3] = time
        gt[:, 1, 3] = time * time
        gt[:, 2, 3] = 0.1 * np.sin(6.0 * time)
        baseline = gt.copy()
        baseline[:, 1, 3] += 0.05 * np.sin(31.0 * time)
        short = np.stack(
            [gt[start : start + 100] for start in range(0, 401, 50)]
        )

        arrays = construct_privileged_arrays(
            scene="scene0000_00",
            frame_ids=np.arange(500, dtype=np.int64),
            source_sha256="b" * 64,
            checkpoint_sha256="c" * 64,
            baseline_pose_encoding=np.zeros((500, 9), dtype=np.float32),
            baseline_c2w=baseline,
            short_c2w=short,
            gt_c2w=gt,
        )

        self.assertEqual(set(arrays), {
            "scene", "frame_ids", "gt_c2w", "oracle_scale", "oracle_rotation",
            "oracle_translation", "oracle_digest", "gt_scene_scale",
            "baseline_pose_encoding", "teacher_c2w_gt_gauge", "teacher_weight",
            "window_teacher_weight", "window_baseline_rms", "window_teacher_rms",
            "source_sha256", "checkpoint_sha256",
        })
        self.assertTrue(np.isfinite(arrays["window_teacher_weight"]).all())
        self.assertGreater(np.count_nonzero(arrays["teacher_weight"]), 0)
        self.assertEqual(len(str(arrays["oracle_digest"])), 64)

    def test_builder_decodes_frozen_tokens_and_writes_separate_privileged_shard(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arrays = _source_arrays()
            time = np.linspace(0.0, 1.0, 500)
            gt = np.repeat(np.eye(4, dtype=np.float64)[None], 500, axis=0)
            gt[:, 0, 3] = time
            gt[:, 1, 3] = time * time
            gt[:, 2, 3] = 0.1 * np.sin(6.0 * time)
            baseline = gt.copy()
            baseline[:, 1, 3] += 0.05 * np.sin(31.0 * time)

            def pose_tokens(c2w: np.ndarray) -> np.ndarray:
                encoded = np.zeros((len(c2w), 2048), dtype=np.float32)
                encoded[:, :3] = -c2w[:, :3, 3]
                encoded[:, 6] = 1.0
                return encoded

            arrays["global_camera_tokens"] = pose_tokens(baseline)
            arrays["short_camera_tokens"] = np.stack(
                [pose_tokens(gt[start : start + 100]) for start in range(0, 401, 50)]
            )
            arrays["overlap_long_tokens"] = np.stack(
                [arrays["global_camera_tokens"][start : start + 50] for start in range(50, 401, 50)]
            )
            arrays["overlap_left_tokens"] = arrays["short_camera_tokens"][:-1, 50:].copy()
            arrays["overlap_right_tokens"] = arrays["short_camera_tokens"][1:, :50].copy()
            arrays["global_pred_c2w"] = baseline
            arrays["overlap_long_c2w"] = np.stack(
                [baseline[start : start + 50] for start in range(50, 401, 50)]
            )
            source = root / "prediction_only" / "source.npz"
            save_source_shard(source, arrays)
            prepared = root / "prepared"
            (prepared / "pose").mkdir(parents=True)
            for frame_id, pose in enumerate(gt):
                np.save(prepared / "pose" / f"{frame_id}.npy", pose, allow_pickle=False)
            destination = root / "privileged_labels" / "scene0000_00.npz"

            record = build_privileged_labels(
                source,
                prepared,
                TokenPoseHead(),
                destination,
                checkpoint_sha256="c" * 64,
                device=torch.device("cpu"),
            )

            loaded = load_privileged_labels(record.path)
            self.assertGreater(np.count_nonzero(loaded["teacher_weight"]), 0)
            self.assertEqual(record.scene, "scene0000_00")
            self.assertTrue(record.sha256)

    def test_only_improving_teacher_gets_positive_weight(self) -> None:
        self.assertAlmostEqual(positive_teacher_weight(1.0, 0.5), 0.5)
        self.assertEqual(positive_teacher_weight(1.0, 1.0), 0.0)
        self.assertEqual(positive_teacher_weight(1.0, 1.2), 0.0)
        with self.assertRaisesRegex(ValueError, "finite"):
            positive_teacher_weight(1.0, float("nan"))

    def test_fusion_uses_literal_positive_weights_and_masks_uncovered_frames(self) -> None:
        identity = np.repeat(np.eye(4, dtype=np.float64)[None], 4, axis=0)
        first = identity.copy()
        first[:, 0, 3] = 2.0
        second = identity.copy()
        second[:, 0, 3] = 8.0

        fused, weights = fuse_teacher_trajectories(
            frame_count=6,
            windows=((0, first, 0.75), (2, second, 0.25)),
        )

        np.testing.assert_allclose(fused[2:4, 0, 3], 3.5)
        np.testing.assert_allclose(weights, [0.75, 0.75, 1.0, 1.0, 0.25, 0.25])
        empty, empty_weights = fuse_teacher_trajectories(frame_count=3, windows=())
        self.assertTrue(np.isnan(empty).all())
        np.testing.assert_array_equal(empty_weights, np.zeros(3))

    def test_privileged_round_trip_is_strict(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.npz"
            arrays = {
                "scene": np.asarray("scene0000_00", dtype="U32"),
                "frame_ids": np.arange(500, dtype=np.int64),
                "gt_c2w": np.repeat(np.eye(4)[None], 500, axis=0),
                "oracle_scale": np.asarray(1.0),
                "oracle_rotation": np.eye(3),
                "oracle_translation": np.zeros(3),
                "oracle_digest": np.asarray("a" * 64, dtype="U64"),
                "gt_scene_scale": np.asarray(1.0),
                "baseline_pose_encoding": np.zeros((500, 9), dtype=np.float32),
                "teacher_c2w_gt_gauge": np.repeat(np.eye(4)[None], 500, axis=0),
                "teacher_weight": np.ones(500),
                "window_teacher_weight": np.ones(9),
                "window_baseline_rms": np.ones(9),
                "window_teacher_rms": np.zeros(9),
                "source_sha256": np.asarray("b" * 64, dtype="U64"),
                "checkpoint_sha256": np.asarray("c" * 64, dtype="U64"),
            }
            save_privileged_labels(path, arrays)
            loaded = load_privileged_labels(path)
            self.assertEqual(set(loaded), set(arrays))

            arrays["forbidden_extra"] = np.asarray(1)
            with self.assertRaisesRegex(ValueError, "members"):
                save_privileged_labels(path, arrays)


if __name__ == "__main__":
    unittest.main()
