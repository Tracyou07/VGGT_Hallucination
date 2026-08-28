from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from pre_experiments.long_short_camera_head.data import (
    load_long_context,
    load_source_records,
    publish_long_context,
)
from pre_experiments.variational_camera_latent.source import save_source_shard


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _source_arrays(scene: str = "scene0000_00") -> dict[str, np.ndarray]:
    frame_ids = np.arange(500, dtype=np.int64)
    global_tokens = np.zeros((500, 2048), dtype=np.float32)
    short_tokens = np.zeros((9, 100, 2048), dtype=np.float32)
    short_ids = np.stack([frame_ids[start : start + 100] for start in range(0, 401, 50)])
    overlap_ids = np.stack([frame_ids[start : start + 50] for start in range(50, 401, 50)])
    overlap_long = np.stack([global_tokens[start : start + 50] for start in range(50, 401, 50)])
    overlap_left = short_tokens[:-1, 50:].copy()
    overlap_right = short_tokens[1:, :50].copy()
    c2w = np.repeat(np.eye(4, dtype=np.float64)[None], 500, axis=0)
    return {
        "global_frame_ids": frame_ids,
        "global_camera_tokens": global_tokens,
        "short_frame_ids": short_ids,
        "short_camera_tokens": short_tokens,
        "overlap_frame_ids": overlap_ids,
        "overlap_long_tokens": overlap_long,
        "overlap_left_tokens": overlap_left,
        "overlap_right_tokens": overlap_right,
        "span_starts": np.arange(0, 400, 50, dtype=np.int64),
        "sample_ids": np.asarray(
            [f"{scene}:overlap_{index:03d}" for index in range(8)], dtype="U64"
        ),
        "global_pred_c2w": c2w,
        "overlap_long_c2w": np.stack(
            [c2w[start : start + 50] for start in range(50, 401, 50)]
        ),
    }


class LongContextContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source_run = self.root / "source_run"
        self.source_path = (
            self.source_run / "prediction_only" / "source" / "scene0000_00.npz"
        )
        save_source_shard(self.source_path, _source_arrays())
        manifest = {
            "schema": "variational_camera_latent.source.v1",
            "dataset_root": "/stale/pre-migration/root",
            "source_run_digest": "a" * 64,
            "records": [
                {
                    "scene": "scene0000_00",
                    "role": "train",
                    "path": "/stale/pre-migration/root/scene0000_00.npz",
                    "overlap_count": 8,
                    "sha256": _sha256(self.source_path),
                }
            ],
        }
        manifest_path = self.source_run / "manifests" / "source_manifest.json"
        manifest_path.parent.mkdir(parents=True)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_rebases_stale_manifest_path_and_checks_digest(self) -> None:
        records = load_source_records(self.source_run)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].path.resolve(), self.source_path.resolve())
        self.assertEqual(records[0].role, "train")

        with self.source_path.open("ab") as handle:
            handle.write(b"corrupt")
        with self.assertRaisesRegex(ValueError, "digest"):
            load_source_records(self.source_run)

    def test_long_context_excludes_short_and_gt_members(self) -> None:
        record = load_source_records(self.source_run)[0]
        destination = self.root / "long" / "scene0000_00.npz"

        published = publish_long_context(record, destination)
        arrays = load_long_context(published.path)

        self.assertEqual(
            set(arrays),
            {"scene", "frame_ids", "camera_tokens", "baseline_c2w", "source_sha256"},
        )
        self.assertEqual(arrays["camera_tokens"].shape, (500, 2048))
        self.assertFalse(any("short" in name or "gt" in name for name in arrays))

    def test_long_context_loader_rejects_extra_privileged_member(self) -> None:
        destination = self.root / "bad.npz"
        with destination.open("wb") as handle:
            np.savez_compressed(
                handle,
                scene=np.asarray("scene0000_00", dtype="U32"),
                frame_ids=np.arange(500, dtype=np.int64),
                camera_tokens=np.zeros((500, 2048), dtype=np.float32),
                baseline_c2w=np.repeat(np.eye(4)[None], 500, axis=0),
                source_sha256=np.asarray("a" * 64, dtype="U64"),
                gt_c2w=np.repeat(np.eye(4)[None], 500, axis=0),
            )

        with self.assertRaisesRegex(ValueError, "members"):
            load_long_context(destination)


if __name__ == "__main__":
    unittest.main()
