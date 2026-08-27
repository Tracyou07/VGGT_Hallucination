from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from pre_experiments.variational_camera_latent.alpha_scan import DEFAULT_ALPHAS
from pre_experiments.variational_camera_latent.candidates import _save_candidate_shard
from pre_experiments.variational_camera_latent.source import save_source_shard
from pre_experiments.variational_camera_selector.dataset import (
    PredictionCandidateDataset,
    SelectorTrainingDataset,
)
from pre_experiments.variational_camera_selector.schema import build_long_context_shard


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)


class SelectorDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.scene = "scene0000_00"
        self.role = "train"
        self.source_path = self.root / "source.npz"
        self.candidate_path = self.root / "candidate.npz"
        self.long_path = self.root / "long.npz"
        self.residual_path = self.root / "residual_prediction.npz"
        self.sidecar_path = self.root / "residual_privileged.npz"
        self.prediction_manifest = self.root / "candidate_binding_manifest.json"
        self.privileged_manifest = self.root / "privileged_binding_manifest.json"
        self.frame_ids = np.arange(500, dtype=np.int64)
        self.overlap_ids = np.stack(
            [self.frame_ids[start : start + 50] for start in range(50, 401, 50)]
        )
        self.sample_ids = np.asarray(
            [f"{self.scene}:overlap_{index:03d}" for index in range(8)], dtype="U64"
        )
        self.alphas = np.asarray(DEFAULT_ALPHAS, dtype=np.float64)
        self.sample_seeds = np.arange(256, dtype=np.int64).reshape(8, 32)
        self.z = np.repeat(
            np.arange(32, dtype=np.float32)[None, :, None], 8, axis=0
        )
        self.z = np.concatenate((self.z, -self.z), axis=2)
        self._write_source_candidate_and_long()
        self._write_residual_prediction()
        self._write_sidecar()
        self._write_manifests()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_source_candidate_and_long(self) -> None:
        global_tokens = np.zeros((500, 2048), dtype=np.float32)
        overlap_tokens = np.zeros((8, 50, 2048), dtype=np.float32)
        short_ids = np.stack(
            [self.frame_ids[start : start + 100] for start in range(0, 401, 50)]
        )
        short_tokens = np.zeros((9, 100, 2048), dtype=np.float32)
        save_source_shard(
            self.source_path,
            {
                "global_frame_ids": self.frame_ids,
                "global_camera_tokens": global_tokens,
                "short_frame_ids": short_ids,
                "short_camera_tokens": short_tokens,
                "overlap_frame_ids": self.overlap_ids,
                "overlap_long_tokens": overlap_tokens,
                "overlap_left_tokens": short_tokens[:-1, 50:],
                "overlap_right_tokens": short_tokens[1:, :50],
                "span_starts": np.arange(0, 400, 50, dtype=np.int64),
                "sample_ids": self.sample_ids,
            },
        )
        direction_values = np.arange(1, 33, dtype=np.float16)
        corrected = np.broadcast_to(
            direction_values[None, :, None, None], (8, 32, 50, 2048)
        ).copy()
        _save_candidate_shard(
            self.candidate_path,
            {
                "z": self.z,
                "corrected_camera_tokens": corrected,
                "latent_cluster_ids": np.zeros((8, 32), dtype=np.int64),
                "latent_cluster_centers": np.zeros((8, 2, 50, 2048), dtype=np.float16),
                "source_long_tokens": overlap_tokens,
                "source_sample_ids": self.sample_ids,
                "span_starts": np.arange(0, 400, 50, dtype=np.int64),
                "sample_seeds": self.sample_seeds,
                "checkpoint_sha256": np.asarray("a" * 64, dtype="U64"),
            },
        )
        build_long_context_shard(
            self.source_path,
            self.candidate_path,
            self.long_path,
            role=self.role,
            producer_git_commit="b" * 40,
        )

    def _write_residual_prediction(self) -> None:
        c2w = np.zeros((8, 32, 8, 50, 4, 4), dtype=np.float16)
        c2w[..., 3, 3] = 1.0
        _write_npz(
            self.residual_path,
            {
                "alphas": self.alphas,
                "source_sample_ids": self.sample_ids,
                "overlap_frame_ids": self.overlap_ids,
                "span_starts": np.arange(0, 400, 50, dtype=np.int64),
                "z": self.z,
                "latent_cluster_ids": np.zeros((8, 32), dtype=np.int64),
                "sample_seeds": self.sample_seeds,
                "decode_context_frames": np.asarray(500, dtype=np.int64),
                "camera_iterations": np.asarray(4, dtype=np.int64),
                "decode_protocol": np.asarray(
                    "vrfm_residual_alpha_full_g500.v1", dtype="U64"
                ),
                "decoded_camera_raw": np.zeros((8, 32, 8, 50, 9), dtype=np.float16),
                "decoded_camera_c2w": c2w,
                "residual_rms": np.ones((8, 32), dtype=np.float32),
                "vrfm_checkpoint_sha256": np.asarray("a" * 64, dtype="U64"),
                "camera_head_checkpoint_sha256": np.asarray("c" * 64, dtype="U64"),
                "source_shard_sha256": np.asarray(_sha256(self.source_path), dtype="U64"),
                "candidate_shard_sha256": np.asarray(
                    _sha256(self.candidate_path), dtype="U64"
                ),
                "producer_git_commit": np.asarray("b" * 40, dtype="U40"),
            },
        )

    def _write_sidecar(
        self,
        *,
        mutate_sample_seed: bool = False,
        mutate_alpha: bool = False,
        mutate_prediction_sha256: bool = False,
    ) -> None:
        sample_seeds = self.sample_seeds.copy()
        if mutate_sample_seed:
            sample_seeds[0, 0] += 1
        alphas = self.alphas.copy()
        if mutate_alpha:
            alphas[2] = 0.03
        prediction_sha256 = _sha256(self.residual_path)
        if mutate_prediction_sha256:
            prediction_sha256 = "e" * 64
        relative = np.zeros((8, 32, 8), dtype=np.float64)
        relative[:, :, 1:] = np.arange(1, 8, dtype=np.float64)[None, None] / 100.0
        gt_c2w = np.zeros((8, 50, 4, 4), dtype=np.float64)
        gt_c2w[..., 3, 3] = 1.0
        _write_npz(
            self.sidecar_path,
            {
                "alphas": alphas,
                "source_sample_ids": self.sample_ids,
                "sample_seeds": sample_seeds,
                "gt_frame_ids": self.overlap_ids,
                "gt_c2w": gt_c2w,
                "baseline_rms": np.ones(8, dtype=np.float64),
                "candidate_rms": np.ones((8, 32, 8), dtype=np.float64),
                "relative_improvement": relative,
                "accept_correction": np.zeros(8, dtype=np.bool_),
                "best_sample_index": np.full(8, -1, dtype=np.int64),
                "best_alpha_index": np.zeros(8, dtype=np.int64),
                "best_alpha": np.zeros(8, dtype=np.float64),
                "best_sample_seed": np.full(8, -1, dtype=np.int64),
                "best_latent_cluster_id": np.full(8, -1, dtype=np.int64),
                "best_candidate_rms": np.ones(8, dtype=np.float64),
                "best_relative_improvement": np.zeros(8, dtype=np.float64),
                "best_nonzero_sample_index": np.zeros(8, dtype=np.int64),
                "best_nonzero_alpha_index": np.ones(8, dtype=np.int64),
                "best_nonzero_alpha": np.full(8, float(alphas[1]), dtype=np.float64),
                "best_nonzero_sample_seed": sample_seeds[:, 0],
                "best_nonzero_latent_cluster_id": np.zeros(8, dtype=np.int64),
                "best_nonzero_candidate_rms": np.ones(8, dtype=np.float64),
                "best_nonzero_relative_improvement": np.zeros(8, dtype=np.float64),
                "prediction_sha256": np.asarray(prediction_sha256, dtype="U64"),
                "prepared_gt_sha256": np.asarray("f" * 64, dtype="U64"),
                "oracle_scale": np.asarray(1.0, dtype=np.float64),
                "oracle_rotation": np.eye(3, dtype=np.float64),
                "oracle_translation": np.zeros(3, dtype=np.float64),
                "oracle_digest": np.asarray("d" * 64, dtype="U64"),
            },
        )

    def _write_manifests(
        self, *, privileged_scene: str | None = None, privileged_role: str | None = None
    ) -> None:
        prediction_record = {
            "scene": self.scene,
            "role": self.role,
            "long_context_path": str(self.long_path),
            "long_context_sha256": _sha256(self.long_path),
            "source_sha256": _sha256(self.source_path),
            "candidate_path": str(self.candidate_path),
            "candidate_sha256": _sha256(self.candidate_path),
            "residual_prediction_path": str(self.residual_path),
            "residual_prediction_sha256": _sha256(self.residual_path),
        }
        self.prediction_manifest.write_text(
            json.dumps(
                {
                    "schema": "variational_camera_selector.candidate_binding_manifest.v1",
                    "alphas": list(DEFAULT_ALPHAS),
                    "records": [prediction_record],
                }
            ),
            encoding="utf-8",
        )
        privileged_record = {
            "scene": privileged_scene or self.scene,
            "role": privileged_role or self.role,
            "path": str(self.sidecar_path),
            "sha256": _sha256(self.sidecar_path),
            "prediction_sha256": _sha256(self.residual_path),
            "source_sha256": _sha256(self.source_path),
            "candidate_sha256": _sha256(self.candidate_path),
        }
        self.privileged_manifest.write_text(
            json.dumps(
                {
                    "schema": "variational_camera_selector.privileged_binding_manifest.v1",
                    "records": [privileged_record],
                }
            ),
            encoding="utf-8",
        )

    def test_prediction_group_has_one_noop_and_224_nonzero_choices(self) -> None:
        group = PredictionCandidateDataset(self.prediction_manifest)[0]

        self.assertEqual(group.delta_tokens.shape, (225, 50, 2048))
        self.assertEqual(group.alphas.shape, (225,))
        self.assertEqual(int(np.count_nonzero(group.alphas == 0.0)), 1)
        np.testing.assert_array_equal(group.delta_tokens[0], 0.0)
        np.testing.assert_allclose(group.delta_tokens[1, 0, 0], 0.01)
        np.testing.assert_allclose(group.delta_tokens[32, 0, 0], 0.32)
        np.testing.assert_allclose(group.delta_tokens[33, 0, 0], 0.02)
        self.assertEqual(str(group.choice_ids[0]), f"{self.sample_ids[0]}:noop")
        self.assertIsNone(group.utilities)

    def test_prediction_dataset_cannot_accept_privileged_path(self) -> None:
        signature = inspect.signature(PredictionCandidateDataset)
        self.assertFalse(
            any("privileged" in name.lower() for name in signature.parameters)
        )

    def test_training_join_emits_alpha_major_utilities(self) -> None:
        group = SelectorTrainingDataset(
            self.prediction_manifest, self.privileged_manifest
        )[0]
        self.assertEqual(group.utilities.shape, (225,))
        self.assertEqual(float(group.utilities[0]), 0.0)
        np.testing.assert_allclose(group.utilities[1:33], 0.01)
        np.testing.assert_allclose(group.utilities[33:65], 0.02)

    def test_training_join_rejects_scene_seed_alpha_role_and_digest_mismatch(self) -> None:
        mutations = (
            "scene",
            "sample_seed",
            "alpha",
            "role",
            "prediction_sha256",
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                self._write_sidecar(
                    mutate_sample_seed=mutation == "sample_seed",
                    mutate_alpha=mutation == "alpha",
                    mutate_prediction_sha256=mutation == "prediction_sha256",
                )
                self._write_manifests(
                    privileged_scene="scene9999_99" if mutation == "scene" else None,
                    privileged_role="validation" if mutation == "role" else None,
                )
                with self.assertRaises(ValueError):
                    SelectorTrainingDataset(
                        self.prediction_manifest, self.privileged_manifest
                    )


if __name__ == "__main__":
    unittest.main()
