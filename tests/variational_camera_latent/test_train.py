from __future__ import annotations

import inspect
import tempfile
from pathlib import Path
import unittest

import numpy as np

from pre_experiments.variational_camera_latent.source import save_source_shard
from pre_experiments.variational_camera_latent.train import (
    OverlapDataset,
    TrainConfig,
    train_models,
)


class TrainingTests(unittest.TestCase):
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
                    [f"scene0000_00:overlap_{index:03d}" for index in range(8)],
                    dtype="U64",
                ),
            },
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _config(self, max_steps: int) -> TrainConfig:
        return TrainConfig(
            source_paths=(self.source,),
            run_root=self.root / "run",
            max_steps=max_steps,
            batch_size=1,
            learning_rate=1e-3,
            seed=13,
            device="cpu",
            d_model=8,
            z_dim=2,
            layers=1,
            heads=2,
            checkpoint_interval=1,
            git_commit="d" * 40,
        )

    def test_training_dataset_never_accepts_privileged_root(self) -> None:
        self.assertNotIn("privileged", inspect.signature(OverlapDataset).parameters)
        dataset = OverlapDataset((self.source,))
        self.assertEqual(len(dataset), 16)
        self.assertEqual(dataset[0]["endpoint_side"].item(), 0)
        self.assertEqual(dataset[1]["endpoint_side"].item(), 1)

    def test_resume_restores_exact_next_step(self) -> None:
        first = train_models(self._config(max_steps=2))
        resumed = train_models(self._config(max_steps=3))

        self.assertEqual(first.start_step, 0)
        self.assertEqual(first.completed_step, 2)
        self.assertEqual(resumed.start_step, 2)
        self.assertEqual(resumed.completed_step, 3)
        self.assertTrue(resumed.checkpoint_path.is_file())


if __name__ == "__main__":
    unittest.main()
