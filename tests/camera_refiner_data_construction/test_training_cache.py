from __future__ import annotations

import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from pre_experiments.camera_refiner_data_construction.cache_schema import (
    load_sequence_shard,
    save_sequence_shard,
    write_sequence_manifest,
)
from pre_experiments.camera_refiner_data_construction.geometry import (
    align_pose_to_reference,
    c2w_to_pose_encoding,
    pose_encoding_to_c2w,
    select_consensus_short_pose,
)


def _trajectory(count: int, *, scale: float = 1.0, offset: float = 0.0) -> np.ndarray:
    poses = np.tile(np.eye(4, dtype=np.float64), (count, 1, 1))
    poses[:, 0, 3] = scale * np.arange(count, dtype=np.float64) + offset
    poses[:, 1, 3] = 0.1 * scale * np.arange(count, dtype=np.float64) ** 2
    return poses


class TrainingGeometryTest(unittest.TestCase):
    def test_aligns_short_pose_encoding_into_long_prediction_gauge(self) -> None:
        reference_c2w = _trajectory(4)
        moving_c2w = _trajectory(4, scale=2.5, offset=7.0)
        fov = np.full((4, 2), 0.9, dtype=np.float32)
        reference = c2w_to_pose_encoding(reference_c2w, fov)
        moving = c2w_to_pose_encoding(moving_c2w, fov)

        result = align_pose_to_reference(reference, moving)
        aligned_c2w = pose_encoding_to_c2w(result.aligned_pose)

        np.testing.assert_allclose(
            aligned_c2w[:, :3, 3], reference_c2w[:, :3, 3], atol=1e-5
        )
        self.assertLess(float(result.translation_residual.max()), 1e-5)
        np.testing.assert_allclose(result.aligned_pose[:, 7:], fov)

    def test_consensus_uses_center_most_observation_and_retains_counts(self) -> None:
        first = np.zeros((4, 9), dtype=np.float32)
        second = np.ones((4, 9), dtype=np.float32)
        first[:, 3] = 1.0
        second[:, 3] = 1.0
        first[:, 7:] = 0.5
        second[:, 7:] = 0.5

        result = select_consensus_short_pose(
            frame_count=6,
            frame_indices=(np.arange(4), np.arange(2, 6)),
            aligned_poses=(first, second),
        )

        np.testing.assert_array_equal(result.observation_count, [1, 1, 2, 2, 1, 1])
        np.testing.assert_array_equal(result.selected_window, [0, 0, 0, 1, 1, 1])
        np.testing.assert_array_equal(result.pose[2], first[2])
        np.testing.assert_array_equal(result.pose[3], second[1])


class SequenceShardContractTest(unittest.TestCase):
    def _arrays(self) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(17)
        count, frames, short_count, short_length = 2, 6, 2, 4
        raw = rng.standard_normal((count, frames, 9), dtype=np.float32)
        raw[..., 3] += 2.0
        activated = raw.copy()
        activated[..., 7:] = np.maximum(activated[..., 7:], 0)
        short = activated.copy()
        short_observations = np.zeros(
            (count, short_count, short_length, 9), dtype=np.float32
        )
        short_observations[..., 3] = 1.0
        short_observations[..., 7:] = 0.5
        return {
            "scene_names": np.asarray(["apple/seq_a", "apple/seq_a"]),
            "clip_ids": np.asarray(["clip_a", "clip_b"]),
            "long_hidden": rng.standard_normal((count, frames, 1024)).astype(np.float16),
            "camera_tokens": rng.standard_normal((count, frames, 2048)).astype(np.float16),
            "baseline_raw_pose": raw,
            "baseline_pose": activated,
            "short_pose": short,
            "diagnostics": rng.standard_normal((count, frames, 4), dtype=np.float32),
            "frame_ids": np.tile(np.arange(frames, dtype=np.int64), (count, 1)),
            "starts": np.asarray([0, 2], dtype=np.int64),
            "gt_c2w_raw": np.tile(np.eye(4), (count, frames, 1, 1)),
            "short_pose_observations": short_observations,
            "short_frame_indices": np.tile(
                np.asarray([[0, 1, 2, 3], [2, 3, 4, 5]], dtype=np.int64),
                (count, 1, 1),
            ),
            "short_observation_count": np.tile(
                np.asarray([1, 1, 2, 2, 1, 1], dtype=np.int64), (count, 1)
            ),
            "selected_short_window": np.tile(
                np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64), (count, 1)
            ),
        }

    def test_round_trip_matches_latent_refiner_schema_without_short_hidden(self) -> None:
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "sequence.npz"
            save_sequence_shard(path, self._arrays())
            restored = load_sequence_shard(path)

        self.assertEqual(restored["long_hidden"].shape, (2, 6, 1024))
        self.assertEqual(restored["camera_tokens"].shape, (2, 6, 2048))
        self.assertNotIn("short_hidden", restored)
        self.assertEqual(restored["short_pose_observations"].shape, (2, 2, 4, 9))

    def test_manifest_authenticates_projection_and_shards(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            shard = root / "shards" / "apple" / "sequence_a.npz"
            save_sequence_shard(shard, self._arrays())
            projection = root / "pose_projection.npy"
            np.save(projection, np.ones((9, 1024), dtype=np.float32))
            projection_digest = hashlib.sha256(projection.read_bytes()).hexdigest()
            manifest = write_sequence_manifest(
                root / "manifest.json",
                dataset_root=root,
                projection_path=projection,
                shard_records=(
                    {
                        "path": shard,
                        "role": "train",
                        "scene": "apple/sequence_a",
                        "sample_count": 2,
                    },
                ),
                camera_iterations=4,
                source_manifest_digest="b" * 64,
            )

        self.assertEqual(manifest["schema_version"], 2)
        self.assertEqual(manifest["study_type"], "full_hidden_sequence_refiner")
        self.assertEqual(
            manifest["pose_projection"]["sha256"],
            projection_digest,
        )
        self.assertEqual(manifest["shards"][0]["role"], "train")


if __name__ == "__main__":
    unittest.main()
