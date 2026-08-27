from __future__ import annotations

import inspect
import json
import tempfile
from pathlib import Path
import unittest

import numpy as np

from pre_experiments.variational_camera_latent.candidates import generate_scene_candidates
from pre_experiments.variational_camera_latent.privileged import (
    load_privileged_sidecar,
    write_privileged_deterministic_sidecar,
    write_privileged_scene_sidecar,
)
from pre_experiments.variational_camera_latent.report import summarize_run
from pre_experiments.variational_camera_latent.source import save_source_shard


class PrivilegedReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_prediction_only_candidate_generation_cannot_receive_gt(self) -> None:
        parameters = inspect.signature(generate_scene_candidates).parameters
        self.assertNotIn("gt", parameters)
        self.assertNotIn("privileged", parameters)

    def test_report_keeps_weak_signal_as_successful_completion(self) -> None:
        prediction = self.root / "prediction_manifest.json"
        privileged = self.root / "privileged_manifest.json"
        prediction.write_text(
            json.dumps(
                {
                    "schema": "variational_camera_latent.prediction_manifest.v1",
                    "scene_count": 10,
                    "candidate_count": 2560,
                    "z_sensitivity": 0.02,
                    "median_one_to_two_sse_ratio": 1.03,
                }
            ),
            encoding="utf-8",
        )
        privileged.write_text(
            json.dumps(
                {
                    "schema": "variational_camera_latent.privileged_manifest.v1",
                    "best_relative_improvements": [0.01, 0.0, -0.01],
                    "deterministic_relative_improvements": [0.0, -0.01, -0.02],
                }
            ),
            encoding="utf-8",
        )

        report = summarize_run(prediction, privileged, self.root / "report.json")

        self.assertEqual(report["signal"], "WEAK_SIGNAL")
        self.assertTrue(report["technically_complete"])
        self.assertAlmostEqual(report["vrfm_minus_deterministic_improvement"], 0.01)
        self.assertTrue((self.root / "report.json").is_file())

    def test_privileged_sidecar_is_separate_and_keyed_by_sample_id(self) -> None:
        source_path = self.root / "prediction_only" / "scene0000_00.npz"
        candidate_path = self.root / "prediction_only" / "candidates.npz"
        privileged_path = self.root / "privileged_labels" / "scene0000_00.npz"
        global_ids = np.arange(500, dtype=np.int64)
        short_ids = np.stack([global_ids[start : start + 100] for start in range(0, 401, 50)])
        tokens = np.zeros((500, 2048), dtype=np.float32)
        short_tokens = np.zeros((9, 100, 2048), dtype=np.float32)
        c2w = np.repeat(np.eye(4, dtype=np.float64)[None], 500, axis=0)
        angle = np.linspace(0.0, 4.0 * np.pi, 500)
        c2w[:, 0, 3] = np.cos(angle)
        c2w[:, 1, 3] = np.sin(angle)
        c2w[:, 2, 3] = np.linspace(-1.0, 1.0, 500) ** 2
        overlap_c2w = np.stack([c2w[start : start + 50] for start in range(50, 401, 50)])
        sample_ids = np.asarray(
            [f"scene0000_00:overlap_{index:03d}" for index in range(8)], dtype="U64"
        )
        save_source_shard(
            source_path,
            {
                "global_frame_ids": global_ids,
                "global_camera_tokens": tokens,
                "short_frame_ids": short_ids,
                "short_camera_tokens": short_tokens,
                "overlap_frame_ids": np.stack(
                    [global_ids[start : start + 50] for start in range(50, 401, 50)]
                ),
                "overlap_long_tokens": np.stack(
                    [tokens[start : start + 50] for start in range(50, 401, 50)]
                ),
                "overlap_left_tokens": short_tokens[:-1, 50:],
                "overlap_right_tokens": short_tokens[1:, :50],
                "span_starts": np.arange(0, 400, 50, dtype=np.int64),
                "sample_ids": sample_ids,
                "global_pred_c2w": c2w,
                "overlap_long_c2w": overlap_c2w,
            },
        )
        candidate_path.parent.mkdir(parents=True, exist_ok=True)
        candidate_c2w = np.repeat(overlap_c2w[:, None], 2, axis=1)
        with candidate_path.open("wb") as handle:
            np.savez_compressed(
                handle,
                z=np.zeros((8, 2, 2), dtype=np.float32),
                corrected_camera_tokens=np.zeros((8, 2, 50, 2048), dtype=np.float32),
                latent_cluster_ids=np.zeros((8, 2), dtype=np.int64),
                latent_cluster_centers=np.zeros((8, 2, 50, 2048), dtype=np.float32),
                source_long_tokens=np.zeros((8, 50, 2048), dtype=np.float32),
                source_sample_ids=sample_ids,
                span_starts=np.arange(0, 400, 50, dtype=np.int64),
                sample_seeds=np.arange(16, dtype=np.int64).reshape(8, 2),
                checkpoint_sha256=np.asarray("f" * 64, dtype="U64"),
                decoded_camera_c2w=candidate_c2w,
            )
        prepared = self.root / "prepared" / "scene0000_00"
        prepared.mkdir(parents=True)
        np.save(prepared / "frame_ids.npy", global_ids, allow_pickle=False)
        np.save(prepared / "raw_gt_c2w.npy", c2w, allow_pickle=False)

        record = write_privileged_scene_sidecar(
            source_path, candidate_path, prepared, privileged_path
        )
        arrays = load_privileged_sidecar(record.path)

        np.testing.assert_array_equal(arrays["sample_ids"], sample_ids)
        self.assertEqual(arrays["candidate_rms"].shape, (8, 2))
        self.assertTrue(record.path.is_relative_to(self.root / "privileged_labels"))

        deterministic_path = self.root / "prediction_only" / "deterministic.npz"
        with deterministic_path.open("wb") as handle:
            deterministic_raw = np.zeros((8, 50, 9), dtype=np.float32)
            deterministic_raw[..., 3] = 1.0
            np.savez_compressed(
                handle,
                corrected_camera_tokens=np.zeros((8, 50, 2048), dtype=np.float32),
                source_long_tokens=np.zeros((8, 50, 2048), dtype=np.float32),
                source_sample_ids=sample_ids,
                span_starts=np.arange(0, 400, 50, dtype=np.int64),
                checkpoint_sha256=np.asarray("f" * 64, dtype="U64"),
                decoded_camera_raw=deterministic_raw,
            )
        deterministic_record = write_privileged_deterministic_sidecar(
            source_path,
            deterministic_path,
            prepared,
            self.root / "privileged_labels" / "scene0000_00_deterministic.npz",
        )
        deterministic = load_privileged_sidecar(deterministic_record.path)
        self.assertEqual(deterministic["candidate_rms"].shape, (8, 1))


if __name__ == "__main__":
    unittest.main()
