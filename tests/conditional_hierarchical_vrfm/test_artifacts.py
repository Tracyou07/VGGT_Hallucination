from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from pre_experiments.conditional_hierarchical_vrfm.artifacts import (
    load_latent_targets,
    save_latent_targets,
)


class LatentTargetArtifactTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary_directory.name) / "targets.npz"

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    @staticmethod
    def valid_arrays() -> dict[str, np.ndarray]:
        poses = np.repeat(np.eye(4, dtype=np.float64)[None, None], 4 * 500, axis=0)
        return {
            "scene": np.asarray("scene0000_00", dtype="U32"),
            "frame_ids": np.arange(500, dtype=np.int64),
            "teacher_variant_ids": np.arange(4, dtype=np.int64),
            "teacher_window_masks": np.ones((4, 9), dtype=np.uint8),
            "coverage_masks": np.ones((4, 500), dtype=np.uint8),
            "residual_coefficients": np.zeros((4, 32, 2048), dtype=np.float32),
            "decoded_c2w_raw": poses.reshape(4, 500, 4, 4),
            "optimization_steps": np.ones(4, dtype=np.int64),
            "initial_losses": np.ones(4, dtype=np.float64),
            "final_losses": np.zeros(4, dtype=np.float64),
            "basis_sha256": np.asarray("a" * 64, dtype="U64"),
            "source_sha256": np.asarray("b" * 64, dtype="U64"),
            "teacher_sha256": np.asarray("c" * 64, dtype="U64"),
            "checkpoint_sha256": np.asarray("d" * 64, dtype="U64"),
            "git_commit": np.asarray("e" * 40, dtype="U40"),
        }

    def test_latent_target_round_trip_uses_exact_schema(self) -> None:
        digest = save_latent_targets(self.path, self.valid_arrays())
        self.assertEqual(len(digest), 64)
        loaded = load_latent_targets(self.path)
        self.assertEqual(loaded["residual_coefficients"].shape, (4, 32, 2048))

    def test_latent_target_rejects_missing_binding_and_nonfinite_coefficients(self) -> None:
        arrays = self.valid_arrays()
        del arrays["teacher_sha256"]
        with self.assertRaisesRegex(ValueError, "exact schema"):
            save_latent_targets(self.path, arrays)
        arrays = self.valid_arrays()
        arrays["residual_coefficients"][0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            save_latent_targets(self.path, arrays)


if __name__ == "__main__":
    unittest.main()
