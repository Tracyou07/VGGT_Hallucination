from __future__ import annotations

import gzip
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from pre_experiments.camera_refiner_data_construction.co3d_manifest import (
    build_clip_manifest,
    load_clip_manifest,
    pytorch3d_viewpoint_to_c2w,
)


def _write_jgz(path: Path, payload: object) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)


def _viewpoint(frame: int) -> dict[str, object]:
    return {
        "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "T": [-float(frame), 0.0, 0.0],
        "focal_length": [1.2, 1.1],
        "principal_point": [0.0, 0.0],
    }


class Co3DClipManifestTest(unittest.TestCase):
    def _dataset(self, root: Path, *, frames: int = 8) -> Path:
        category = root / "apple"
        image_dir = category / "sequence_a" / "images"
        image_dir.mkdir(parents=True)
        annotations = []
        for frame in reversed(range(frames)):
            relative = f"apple/sequence_a/images/frame{frame:06d}.jpg"
            (root / relative).write_bytes(b"rgb")
            annotations.append(
                {
                    "sequence_name": "sequence_a",
                    "frame_number": frame,
                    "image": {"path": relative, "size": [48, 64]},
                    "viewpoint": _viewpoint(frame),
                }
            )
        _write_jgz(category / "frame_annotations.jgz", annotations)
        _write_jgz(
            category / "sequence_annotations.jgz",
            [{"sequence_name": "sequence_a", "viewpoint_quality_score": 0.9}],
        )
        download = root / "download_manifest.json"
        download.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "dataset": "CO3Dv2",
                    "selection_sha256": "a" * 64,
                    "sequences": [
                        {
                            "category": "apple",
                            "sequence_name": "sequence_a",
                            "relative_image_dir": "apple/sequence_a/images",
                            "quality_score": 0.9,
                            "valid_frame_count": frames,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        return download

    def test_pytorch3d_camera_conversion_preserves_center_and_axes(self) -> None:
        c2w = pytorch3d_viewpoint_to_c2w(
            np.eye(3), np.asarray([1.0, 2.0, 3.0])
        )

        np.testing.assert_allclose(c2w[:3, 3], [-1.0, -2.0, -3.0])
        np.testing.assert_allclose(c2w[:3, :3], np.diag([-1.0, -1.0, 1.0]))
        np.testing.assert_allclose(c2w[3], [0.0, 0.0, 0.0, 1.0])

    def test_builds_deterministic_ordered_sequence_disjoint_clips(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            download = self._dataset(root)
            first_path = root / "first.json"
            second_path = root / "second.json"
            build_clip_manifest(
                data_root=root,
                download_manifest=download,
                output_path=first_path,
                clip_length=4,
                max_clips_per_sequence=2,
                temporal_strides=(1, 2),
                validation_fraction=0.2,
                seed=33,
            )
            build_clip_manifest(
                data_root=root,
                download_manifest=download,
                output_path=second_path,
                clip_length=4,
                max_clips_per_sequence=2,
                temporal_strides=(1, 2),
                validation_fraction=0.2,
                seed=33,
            )
            first = load_clip_manifest(first_path, root)
            second = load_clip_manifest(second_path, root)

        self.assertEqual(first.digest, second.digest)
        self.assertEqual(len(first.clips), 2)
        self.assertEqual({clip.role for clip in first.clips}, {first.clips[0].role})
        for clip in first.clips:
            self.assertEqual(len(clip.frame_numbers), 4)
            self.assertTrue(np.all(np.diff(clip.frame_numbers) > 0))
            self.assertIn(clip.temporal_stride, (1, 2))

    def test_records_short_sequences_instead_of_looping_frames(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            download = self._dataset(root, frames=3)
            output = root / "clips.json"
            payload = build_clip_manifest(
                data_root=root,
                download_manifest=download,
                output_path=output,
                clip_length=4,
                max_clips_per_sequence=1,
                temporal_strides=(1,),
                validation_fraction=0.2,
                seed=33,
            )

        self.assertEqual(payload["clips"], [])
        self.assertEqual(payload["rejections"][0]["reason"], "insufficient_frames")


if __name__ == "__main__":
    unittest.main()
