from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

import numpy as np

from pre_experiments.camera_velocity_ambiguity_02.artifacts import (
    build_prediction_identity,
    save_completed_prediction,
)
from pre_experiments.variational_camera_latent.source import (
    build_scene_source_shard,
    load_source_shard,
)


class SourceShardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.predictions = self.root / "predictions" / "scene0000_00"
        self.output = self.root / "source" / "scene0000_00.npz"
        self.frame_ids = np.arange(500, dtype=np.int64)
        self.global_tokens = np.repeat(
            self.frame_ids[:, None], 2048, axis=1
        ).astype(np.float32)
        self._write_prediction(
            self.predictions / "global",
            frame_ids=self.frame_ids,
            tokens=self.global_tokens,
            kind="global",
            window_index=None,
        )
        for index, start in enumerate(range(0, 401, 50)):
            local_ids = self.frame_ids[start : start + 100]
            tokens = np.full((100, 2048), index + 10, dtype=np.float32)
            self._write_prediction(
                self.predictions / "local" / f"window_{index:03d}",
                frame_ids=local_ids,
                tokens=tokens,
                kind="local",
                window_index=index,
            )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_prediction(
        self,
        directory: Path,
        *,
        frame_ids: np.ndarray,
        tokens: np.ndarray,
        kind: str,
        window_index: int | None,
    ) -> None:
        identity = build_prediction_identity(
            run_id="fixture_run",
            scene="scene0000_00",
            artifact_kind=kind,
            window_index=window_index,
            frame_ids=frame_ids,
            checkpoint_sha256="a" * 64,
            git_commit="b" * 40,
            protocol_digest="c" * 64,
            preprocess="crop",
            camera_iterations=4,
        )
        save_completed_prediction(
            directory / "prediction.npz",
            directory / "complete.json",
            {
                "frame_ids": frame_ids,
                "normalized_camera_tokens": tokens,
                "pred_c2w_raw": np.repeat(
                    np.eye(4, dtype=np.float32)[None], len(frame_ids), axis=0
                ),
            },
            identity,
        )

    def test_builds_eight_shared_50_pairs_from_nine_windows(self) -> None:
        record = build_scene_source_shard(
            self.predictions, self.output, role="train"
        )
        arrays = load_source_shard(record.path)

        self.assertEqual(arrays["global_camera_tokens"].shape, (500, 2048))
        self.assertEqual(arrays["short_camera_tokens"].shape, (9, 100, 2048))
        self.assertEqual(arrays["overlap_long_tokens"].shape, (8, 50, 2048))
        np.testing.assert_array_equal(
            arrays["overlap_left_tokens"][0],
            arrays["short_camera_tokens"][0, 50:],
        )
        np.testing.assert_array_equal(
            arrays["overlap_right_tokens"][0],
            arrays["short_camera_tokens"][1, :50],
        )
        np.testing.assert_array_equal(arrays["overlap_frame_ids"][0], np.arange(50, 100))
        self.assertEqual(record.overlap_count, 8)

    def test_rejects_frame_misalignment(self) -> None:
        bad_directory = self.predictions / "local" / "window_001"
        bad_ids = np.arange(51, 151, dtype=np.int64)
        self._write_prediction(
            bad_directory,
            frame_ids=bad_ids,
            tokens=np.ones((100, 2048), dtype=np.float32),
            kind="local",
            window_index=1,
        )

        with self.assertRaisesRegex(ValueError, "frame IDs"):
            build_scene_source_shard(self.predictions, self.output, role="train")

    def test_loader_rejects_object_arrays_and_gt_members(self) -> None:
        self.output.parent.mkdir(parents=True, exist_ok=True)
        with self.output.open("wb") as handle:
            np.savez_compressed(
                handle,
                sample_ids=np.asarray([object()], dtype=object),
                gt_pose=np.zeros((1, 4, 4), dtype=np.float32),
            )

        with self.assertRaisesRegex(ValueError, "object|GT|privileged"):
            load_source_shard(self.output)


if __name__ == "__main__":
    unittest.main()
