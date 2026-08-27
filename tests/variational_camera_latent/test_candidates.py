from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch

from pre_experiments.variational_camera_latent.candidates import (
    analyze_candidate_shard,
    generate_deterministic_candidates,
    generate_scene_candidates,
    load_candidate_shard,
)
from pre_experiments.variational_camera_latent.clustering import two_means
from pre_experiments.variational_camera_latent.source import save_source_shard
from pre_experiments.variational_camera_latent.train import TrainConfig, train_models


class CandidateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "source.npz"
        global_ids = np.arange(500, dtype=np.int64)
        short_ids = np.stack([global_ids[start : start + 100] for start in range(0, 401, 50)])
        global_tokens = np.zeros((500, 2048), dtype=np.float32)
        short_tokens = np.zeros((9, 100, 2048), dtype=np.float32)
        save_source_shard(
            self.source,
            {
                "global_frame_ids": global_ids,
                "global_camera_tokens": global_tokens,
                "short_frame_ids": short_ids,
                "short_camera_tokens": short_tokens,
                "overlap_frame_ids": np.stack(
                    [global_ids[start : start + 50] for start in range(50, 401, 50)]
                ),
                "overlap_long_tokens": np.stack(
                    [global_tokens[start : start + 50] for start in range(50, 401, 50)]
                ),
                "overlap_left_tokens": short_tokens[:-1, 50:],
                "overlap_right_tokens": short_tokens[1:, :50],
                "span_starts": np.arange(0, 400, 50, dtype=np.int64),
                "sample_ids": np.asarray(
                    [f"scene0000_00:overlap_{index:03d}" for index in range(8)], dtype="U64"
                ),
            },
        )
        trained = train_models(
            TrainConfig(
                source_paths=(self.source,),
                run_root=self.root / "run",
                max_steps=1,
                batch_size=1,
                device="cpu",
                d_model=8,
                z_dim=2,
                layers=1,
                heads=2,
                checkpoint_interval=1,
                git_commit="e" * 40,
            )
        )
        self.checkpoint = trained.checkpoint_path
        self.candidates = self.root / "candidates.npz"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_candidate_shard_keeps_raw_samples(self) -> None:
        record = generate_scene_candidates(
            self.source,
            self.checkpoint,
            self.candidates,
            samples=3,
            steps=2,
            seed=17,
            device="cpu",
        )

        arrays = load_candidate_shard(record.path)
        self.assertEqual(arrays["z"].shape, (8, 3, 2))
        self.assertEqual(arrays["corrected_camera_tokens"].shape, (8, 3, 50, 2048))
        self.assertEqual(arrays["latent_cluster_ids"].shape, (8, 3))
        self.assertEqual(arrays["source_sample_ids"].shape, (8,))
        self.assertEqual(record.sample_count, 3)

    def test_clustering_can_be_replayed_without_model(self) -> None:
        generate_scene_candidates(
            self.source,
            self.checkpoint,
            self.candidates,
            samples=3,
            steps=1,
            seed=19,
            device="cpu",
        )

        first = analyze_candidate_shard(self.candidates)
        second = analyze_candidate_shard(self.candidates)

        self.assertEqual(first, second)

    def test_deterministic_baseline_exports_one_candidate_without_fake_z(self) -> None:
        path = self.root / "deterministic.npz"

        class IdentityHead:
            def decode_pose_tokens(self, tokens, *, num_iterations):
                raw = torch.zeros((*tokens.shape[:2], 9), device=tokens.device)
                raw[..., 3] = 1.0
                return [raw] * num_iterations

        record = generate_deterministic_candidates(
            self.source,
            self.checkpoint,
            path,
            steps=2,
            device="cpu",
            camera_head=IdentityHead(),
        )
        with np.load(record.path, allow_pickle=False) as archive:
            self.assertNotIn("z", archive.files)
            self.assertEqual(archive["corrected_camera_tokens"].shape, (8, 50, 2048))
            self.assertEqual(archive["decoded_camera_raw"].shape, (8, 50, 9))
            self.assertEqual(archive["decoded_camera_c2w"].shape, (8, 50, 4, 4))

    def test_two_means_separates_literal_two_cloud_fixture(self) -> None:
        features = np.asarray([[0.0], [0.1], [9.9], [10.0]], dtype=np.float32)

        result = two_means(features)

        self.assertEqual(result.labels.tolist(), [0, 0, 1, 1])
        self.assertGreater(result.one_to_two_sse_ratio, 100.0)


if __name__ == "__main__":
    unittest.main()
