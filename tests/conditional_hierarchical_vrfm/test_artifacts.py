from __future__ import annotations

from io import BytesIO
from pathlib import Path
import tempfile
import unittest
import warnings
import zipfile

import numpy as np
from pre_experiments.camera_velocity_ambiguity_02.contracts import canonical_json_digest

from pre_experiments.conditional_hierarchical_vrfm.artifacts import (
    load_teacher_artifact,
    load_latent_targets,
    save_teacher_artifact,
    reuse_or_save_teacher_artifact,
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

    def test_latent_target_rejects_non_floating_coefficients_and_losses(self) -> None:
        for name in ("residual_coefficients", "initial_losses", "final_losses"):
            with self.subTest(name=name):
                arrays = self.valid_arrays()
                arrays[name] = np.full(arrays[name].shape, "0", dtype="U1")
                with self.assertRaisesRegex(ValueError, "floating"):
                    save_latent_targets(self.path, arrays)

    def test_load_rejects_duplicate_archive_members(self) -> None:
        arrays = self.valid_arrays()
        encoded: dict[str, bytes] = {}
        for name, value in arrays.items():
            handle = BytesIO()
            np.save(handle, value, allow_pickle=False)
            encoded[name] = handle.getvalue()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with zipfile.ZipFile(self.path, "w") as archive:
                for name, value in encoded.items():
                    archive.writestr(f"{name}.npy", value)
                archive.writestr("scene.npy", encoded["scene"])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            load_latent_targets(self.path)

    @staticmethod
    def valid_teacher_arrays() -> dict[str, np.ndarray]:
        poses = np.broadcast_to(np.eye(4, dtype=np.float64), (4, 500, 4, 4)).copy()
        gt = poses[0].copy()
        oracle_digest = canonical_json_digest({
            "scene": "scene0000_00", "frame_digest": "a" * 64, "fit_count": 500,
            "scale": 1.0,
            "rotation": ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            "translation": (0.0, 0.0, 0.0),
        })
        return {
            "scene": np.asarray("scene0000_00", dtype="U32"),
            "frame_ids": np.arange(500, dtype=np.int64),
            "gt_c2w": gt,
            "gt_scene_scale": np.asarray(1.0, dtype=np.float64),
            "baseline_c2w_raw": gt.copy(),
            "oracle_scene": np.asarray("scene0000_00", dtype="U32"),
            "oracle_frame_digest": np.asarray("a" * 64, dtype="U64"),
            "oracle_fit_count": np.asarray(500, dtype=np.int64),
            "oracle_scale": np.asarray(1.0, dtype=np.float64),
            "oracle_rotation": np.eye(3, dtype=np.float64),
            "oracle_translation": np.zeros(3, dtype=np.float64),
            "oracle_rank": np.asarray(3, dtype=np.int64),
            "oracle_condition": np.asarray(1.0, dtype=np.float64),
            "oracle_digest": np.asarray(oracle_digest, dtype="U64"),
            "window_weights": np.ones(9, dtype=np.float64),
            "window_masks": np.ones((4, 9), dtype=np.uint8),
            "coverage_weights": np.ones((4, 500), dtype=np.float64),
            "fused_c2w": poses,
            "variant_utilities": np.ones(4, dtype=np.float64),
            "source_sha256": np.asarray("c" * 64, dtype="U64"),
            "formal_label_sha256": np.asarray("d" * 64, dtype="U64"),
            "checkpoint_sha256": np.asarray("e" * 64, dtype="U64"),
            "git_commit": np.asarray("f" * 40, dtype="U40"),
        }

    def test_teacher_artifact_binds_formal_label_and_all_four_variants(self) -> None:
        path = self.path.with_name("teacher.npz")
        arrays = self.valid_teacher_arrays()
        digest = reuse_or_save_teacher_artifact(path, arrays)
        self.assertEqual(reuse_or_save_teacher_artifact(path, arrays), digest)
        loaded = load_teacher_artifact(path)
        self.assertEqual(len(digest), 64)
        self.assertEqual(loaded["fused_c2w"].shape, (4, 500, 4, 4))
        self.assertEqual(str(loaded["formal_label_sha256"]), "d" * 64)
        mismatched = self.valid_teacher_arrays()
        mismatched["source_sha256"] = np.asarray("0" * 64, dtype="U64")
        with self.assertRaisesRegex(ValueError, "existing teacher artifact"):
            reuse_or_save_teacher_artifact(path, mismatched)

    def test_teacher_artifact_rejects_non_so3_oracle(self) -> None:
        arrays = self.valid_teacher_arrays()
        arrays["oracle_rotation"] *= 2.0
        with self.assertRaisesRegex(ValueError, "SO\(3\)"):
            save_teacher_artifact(self.path.with_name("teacher.npz"), arrays)

    def test_teacher_artifact_accepts_raw_so3_roundoff_but_rejects_distortion(self) -> None:
        arrays = self.valid_teacher_arrays()
        roundoff_rotation = np.diag([1.0 + 7e-7, 1.0 + 3e-7, 1.0 + 3e-7])
        arrays["gt_c2w"][0, :3, :3] = roundoff_rotation
        path = self.path.with_name("teacher_roundoff.npz")
        try:
            save_teacher_artifact(path, arrays)
        except ValueError as error:
            self.fail(f"raw authenticated SO(3) roundoff must be accepted: {error}")
        loaded = load_teacher_artifact(path)
        np.testing.assert_array_equal(loaded["gt_c2w"], arrays["gt_c2w"])

        strict_oracle = self.valid_teacher_arrays()
        strict_oracle["oracle_rotation"] = roundoff_rotation
        strict_oracle["oracle_digest"] = np.asarray(canonical_json_digest({
            "scene": "scene0000_00", "frame_digest": "a" * 64, "fit_count": 500,
            "scale": 1.0,
            "rotation": tuple(tuple(float(value) for value in row) for row in roundoff_rotation),
            "translation": (0.0, 0.0, 0.0),
        }), dtype="U64")
        with self.assertRaisesRegex(ValueError, "SO\(3\)"):
            save_teacher_artifact(
                self.path.with_name("teacher_strict_oracle.npz"), strict_oracle
            )

        just_over_threshold = np.eye(3, dtype=np.float64)
        just_over_threshold[0, 1] = 2.1e-6
        larger_shear = np.eye(3, dtype=np.float64)
        larger_shear[0, 1] = 1e-5
        invalid_rotations = {
            "just_over_threshold": just_over_threshold,
            "one_e_minus_five_shear": larger_shear,
            "reflection": np.diag([-1.0, 1.0, 1.0]),
            "nan": np.full((3, 3), np.nan, dtype=np.float64),
        }
        for name, rotation in invalid_rotations.items():
            with self.subTest(name=name):
                distorted = self.valid_teacher_arrays()
                distorted["gt_c2w"][0, :3, :3] = rotation
                with self.assertRaises(ValueError):
                    save_teacher_artifact(
                        self.path.with_name(f"teacher_{name}.npz"), distorted
                    )

    def test_teacher_artifact_recomputes_oracle_digest(self) -> None:
        arrays = self.valid_teacher_arrays()
        arrays["oracle_digest"] = np.asarray("b" * 64, dtype="U64")
        with self.assertRaisesRegex(ValueError, "oracle digest"):
            save_teacher_artifact(self.path.with_name("teacher.npz"), arrays)

    def test_target_teacher_digest_must_match_teacher_artifact(self) -> None:
        teacher_path = self.path.with_name("teacher.npz")
        teacher_digest = save_teacher_artifact(teacher_path, self.valid_teacher_arrays())
        arrays = self.valid_arrays()
        arrays["teacher_sha256"] = np.asarray(teacher_digest, dtype="U64")
        save_latent_targets(self.path, arrays, teacher_artifact=teacher_path)
        arrays["teacher_sha256"] = np.asarray("0" * 64, dtype="U64")
        with self.assertRaisesRegex(ValueError, "teacher artifact"):
            save_latent_targets(self.path, arrays, teacher_artifact=teacher_path)


if __name__ == "__main__":
    unittest.main()
