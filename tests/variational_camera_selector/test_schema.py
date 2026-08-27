from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from pre_experiments.variational_camera_latent.candidates import (
    _save_candidate_shard,
)
from pre_experiments.variational_camera_latent.source import save_source_shard
from pre_experiments.variational_camera_selector.schema import (
    build_long_context_shard,
    load_long_context_shard,
    write_prediction_binding_manifest,
)


class LongContextSchemaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source_path = self.root / "source.npz"
        self.candidate_path = self.root / "candidate.npz"
        self.output_path = self.root / "long_context.npz"
        self.frame_ids = np.arange(500, dtype=np.int64)
        self.global_tokens = np.repeat(
            self.frame_ids[:, None], 2048, axis=1
        ).astype(np.float32)
        self.overlap_frame_ids = np.stack(
            [self.frame_ids[start : start + 50] for start in range(50, 401, 50)]
        )
        self.overlap_tokens = np.stack(
            [self.global_tokens[start : start + 50] for start in range(50, 401, 50)]
        )
        self.sample_ids = np.asarray(
            [f"scene0000_00:overlap_{index:03d}" for index in range(8)],
            dtype="U64",
        )
        short_ids = np.stack(
            [self.frame_ids[start : start + 100] for start in range(0, 401, 50)]
        )
        short_tokens = np.zeros((9, 100, 2048), dtype=np.float32)
        save_source_shard(
            self.source_path,
            {
                "global_frame_ids": self.frame_ids,
                "global_camera_tokens": self.global_tokens,
                "short_frame_ids": short_ids,
                "short_camera_tokens": short_tokens,
                "overlap_frame_ids": self.overlap_frame_ids,
                "overlap_long_tokens": self.overlap_tokens,
                "overlap_left_tokens": short_tokens[:-1, 50:],
                "overlap_right_tokens": short_tokens[1:, :50],
                "span_starts": np.arange(0, 400, 50, dtype=np.int64),
                "sample_ids": self.sample_ids,
            },
        )
        self._write_candidate(self.candidate_path, self.sample_ids)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_candidate(self, path: Path, sample_ids: np.ndarray) -> None:
        samples = 2
        corrected = np.repeat(
            self.overlap_tokens[:, None], samples, axis=1
        ).astype(np.float32)
        _save_candidate_shard(
            path,
            {
                "z": np.zeros((8, samples, 2), dtype=np.float32),
                "corrected_camera_tokens": corrected,
                "latent_cluster_ids": np.zeros((8, samples), dtype=np.int64),
                "latent_cluster_centers": np.repeat(
                    self.overlap_tokens[:, None], 2, axis=1
                ).astype(np.float32),
                "source_long_tokens": self.overlap_tokens.copy(),
                "source_sample_ids": sample_ids.copy(),
                "span_starts": np.arange(0, 400, 50, dtype=np.int64),
                "sample_seeds": np.arange(16, dtype=np.int64).reshape(8, 2),
                "checkpoint_sha256": np.asarray("a" * 64, dtype="U64"),
            },
        )

    def _build(self):
        return build_long_context_shard(
            self.source_path,
            self.candidate_path,
            self.output_path,
            role="train",
            producer_git_commit="b" * 40,
        )

    def test_build_copies_only_long_context_and_binds_candidate(self) -> None:
        record = self._build()
        arrays = load_long_context_shard(record.path)

        self.assertEqual(arrays["global_camera_tokens"].shape, (500, 2048))
        self.assertEqual(arrays["overlap_long_tokens"].shape, (8, 50, 2048))
        self.assertEqual(str(arrays["role"]), "train")
        self.assertEqual(str(arrays["candidate_shard_sha256"]), record.candidate_sha256)
        self.assertEqual(set(arrays), {
            "global_frame_ids",
            "global_camera_tokens",
            "overlap_frame_ids",
            "overlap_long_tokens",
            "span_starts",
            "source_sample_ids",
            "scene",
            "role",
            "source_shard_sha256",
            "candidate_shard_sha256",
            "producer_git_commit",
        })
        self.assertFalse(any("short" in name.lower() for name in arrays))
        self.assertEqual(record.scene, "scene0000_00")
        self.assertEqual(record.role, "train")

    def test_loader_rejects_forbidden_members(self) -> None:
        self._build()
        clean = load_long_context_shard(self.output_path)
        for forbidden in ("short_camera_tokens", "gt_c2w", "quality", "error"):
            with self.subTest(forbidden=forbidden):
                tampered = self.root / f"{forbidden}.npz"
                with tampered.open("wb") as handle:
                    np.savez_compressed(
                        handle,
                        **clean,
                        **{forbidden: np.zeros((1,), dtype=np.float32)},
                    )
                with self.assertRaisesRegex(ValueError, "forbidden|members"):
                    load_long_context_shard(tampered)

    def test_builder_rejects_candidate_sample_id_mismatch(self) -> None:
        other = self.root / "other_candidate.npz"
        mismatched = self.sample_ids.copy()
        mismatched[0] = "scene0000_00:different"
        self._write_candidate(other, mismatched)

        with self.assertRaisesRegex(ValueError, "sample IDs"):
            build_long_context_shard(
                self.source_path,
                other,
                self.output_path,
                role="train",
                producer_git_commit="b" * 40,
            )

    def test_manifest_rejects_duplicate_scene_and_keeps_no_privileged_path(self) -> None:
        record = self._build()
        manifest_path = self.root / "prediction_binding_manifest.json"
        payload = write_prediction_binding_manifest(
            manifest_path,
            records=[record],
            upstream_run_root=self.root / "sealed_phase1",
            upstream_completion_sha256="c" * 64,
            producer_git_commit="b" * 40,
        )
        loaded = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.assertEqual(payload, loaded)
        self.assertEqual(loaded["scene_count"], 1)
        self.assertEqual(loaded["records"][0]["scene"], "scene0000_00")
        self.assertNotIn("privileged", json.dumps(loaded).lower())
        with self.assertRaisesRegex(ValueError, "duplicate scene"):
            write_prediction_binding_manifest(
                manifest_path,
                records=[record, record],
                upstream_run_root=self.root / "sealed_phase1",
                upstream_completion_sha256="c" * 64,
                producer_git_commit="b" * 40,
            )


if __name__ == "__main__":
    unittest.main()

