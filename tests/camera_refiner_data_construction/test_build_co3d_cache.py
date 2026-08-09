from __future__ import annotations

from pathlib import Path
import re
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from pre_experiments.camera_refiner_data_construction.build_co3d_cache import (
    CameraPrediction,
    build_cache,
    canonical_short_windows,
    find_checkpoint,
)
from pre_experiments.camera_refiner_data_construction.cache_schema import (
    load_sequence_shard,
)
from pre_experiments.camera_refiner_data_construction.co3d_manifest import (
    ClipManifest,
    ClipSpec,
)
from pre_experiments.camera_refiner_data_construction.geometry import (
    c2w_to_pose_encoding,
    pose_encoding_to_c2w,
)


def _c2w(frame_ids: np.ndarray, *, moving: bool) -> np.ndarray:
    poses = np.tile(np.eye(4, dtype=np.float64), (len(frame_ids), 1, 1))
    centers = np.stack(
        (
            frame_ids.astype(np.float64),
            0.1 * frame_ids.astype(np.float64) ** 2,
            0.05 * frame_ids.astype(np.float64),
        ),
        axis=1,
    )
    if moving:
        centers = 2.5 * centers + np.asarray([7.0, -3.0, 1.0])
    poses[:, :3, 3] = centers
    return poses


class _FakeRunner:
    def __init__(self) -> None:
        self.calls = 0
        self.camera_iterations = 4
        self.pose_projection = np.full((9, 1024), 0.01, dtype=np.float32)

    def predict(self, image_paths: tuple[Path, ...], *, trace: bool) -> CameraPrediction:
        self.calls += 1
        frame_ids = np.asarray(
            [int(re.search(r"(\d+)$", path.stem).group(1)) for path in image_paths],
            dtype=np.int64,
        )
        pose = c2w_to_pose_encoding(
            _c2w(frame_ids, moving=not trace),
            np.full((len(frame_ids), 2), 0.9, dtype=np.float32),
        )
        return CameraPrediction(
            activated_pose=pose,
            raw_pose=pose.copy() if trace else None,
            hidden=(
                np.full((len(frame_ids), 1024), 0.25, dtype=np.float32)
                if trace
                else None
            ),
            camera_tokens=(
                np.full((len(frame_ids), 2048), 0.5, dtype=np.float32)
                if trace
                else None
            ),
            diagnostics=(
                np.full((len(frame_ids), 4), 0.1, dtype=np.float32)
                if trace
                else None
            ),
        )


class _MixedRunner(_FakeRunner):
    def predict(self, image_paths: tuple[Path, ...], *, trace: bool) -> CameraPrediction:
        if not trace and any("bad_sequence" in path.parts for path in image_paths):
            self.calls += 1
            count = len(image_paths)
            stationary = np.tile(np.eye(4, dtype=np.float64), (count, 1, 1))
            pose = c2w_to_pose_encoding(
                stationary, np.full((count, 2), 0.9, dtype=np.float32)
            )
            return CameraPrediction(activated_pose=pose)
        return super().predict(image_paths, trace=trace)


class Co3DCacheBuilderTest(unittest.TestCase):
    def _manifest(self, root: Path) -> ClipManifest:
        image_paths = []
        for frame in range(6):
            path = root / f"frame{frame:06d}.jpg"
            path.write_bytes(b"rgb")
            image_paths.append(path)
        frame_ids = np.arange(6, dtype=np.int64)
        clip = ClipSpec(
            clip_id="clip_a",
            category="apple",
            sequence_name="sequence_a",
            role="train",
            start_index=0,
            temporal_stride=1,
            frame_numbers=frame_ids,
            image_paths=tuple(image_paths),
            gt_c2w=_c2w(frame_ids, moving=False),
            focal_length=np.ones((6, 2), dtype=np.float32),
            principal_point=np.zeros((6, 2), dtype=np.float32),
            image_size=np.full((6, 2), 64, dtype=np.int64),
        )
        return ClipManifest(
            clips=(clip,),
            digest="a" * 64,
            clip_length=6,
            source_selection_digest="b" * 64,
        )

    def test_short_windows_cover_the_sequence_and_include_the_tail(self) -> None:
        windows = canonical_short_windows(frame_count=11, length=4, stride=3)

        self.assertEqual([window.tolist() for window in windows], [
            [0, 1, 2, 3],
            [3, 4, 5, 6],
            [6, 7, 8, 9],
            [7, 8, 9, 10],
        ])
        np.testing.assert_array_equal(
            np.unique(np.concatenate(windows)), np.arange(11)
        )

    def test_resolves_huggingface_cache_snapshot_without_network(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            revision = "1234abcd"
            snapshot = root / "snapshots" / revision
            snapshot.mkdir(parents=True)
            checkpoint = snapshot / "model.safetensors"
            checkpoint.write_bytes(b"local")
            (root / "refs").mkdir()
            (root / "refs" / "main").write_text(revision, encoding="utf-8")

            resolved = find_checkpoint(root)

        self.assertEqual(resolved, checkpoint.resolve())

    def test_builds_compatible_shard_and_resumes_completed_sequence(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            runner = _FakeRunner()
            manifest = self._manifest(root)
            output = root / "cache"
            first = build_cache(
                manifest,
                output_dir=output,
                runner=runner,
                short_window=4,
                short_stride=2,
                feature_dtype="float16",
                build_digest="c" * 64,
            )
            calls_after_first = runner.calls
            second = build_cache(
                manifest,
                output_dir=output,
                runner=runner,
                short_window=4,
                short_stride=2,
                feature_dtype="float16",
                build_digest="c" * 64,
            )
            shard_path = output / first["shards"][0]["path"]
            shard = load_sequence_shard(shard_path)

        self.assertEqual(calls_after_first, 3)
        self.assertEqual(runner.calls, calls_after_first)
        self.assertEqual(first, second)
        self.assertEqual(shard["long_hidden"].dtype, np.float16)
        self.assertNotIn("short_hidden", shard)
        baseline_c2w = pose_encoding_to_c2w(shard["baseline_pose"][0])
        target_c2w = pose_encoding_to_c2w(shard["short_pose"][0])
        np.testing.assert_allclose(
            target_c2w[:, :3, 3], baseline_c2w[:, :3, 3], atol=1e-5
        )
        self.assertEqual(shard["short_pose_observations"].shape, (1, 2, 4, 9))
        np.testing.assert_array_equal(shard["short_observation_count"], [[1, 1, 2, 2, 1, 1]])

    def test_records_degenerate_sequence_and_resumes_without_retrying_it(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = self._manifest(root).clips[0]
            bad_paths = []
            bad_dir = root / "bad_sequence"
            bad_dir.mkdir()
            for frame in range(6):
                path = bad_dir / f"frame{frame:06d}.jpg"
                path.write_bytes(b"rgb")
                bad_paths.append(path)
            bad = ClipSpec(
                clip_id="clip_bad",
                category="apple",
                sequence_name="bad_sequence",
                role="train",
                start_index=0,
                temporal_stride=1,
                frame_numbers=base.frame_numbers,
                image_paths=tuple(bad_paths),
                gt_c2w=base.gt_c2w,
                focal_length=base.focal_length,
                principal_point=base.principal_point,
                image_size=base.image_size,
            )
            manifest = ClipManifest(
                clips=(bad, base),
                digest="d" * 64,
                clip_length=6,
                source_selection_digest="b" * 64,
            )
            runner = _MixedRunner()
            output = root / "cache"
            first = build_cache(
                manifest,
                output_dir=output,
                runner=runner,
                short_window=4,
                short_stride=2,
                feature_dtype="float16",
                build_digest="e" * 64,
            )
            calls_after_first = runner.calls
            second = build_cache(
                manifest,
                output_dir=output,
                runner=runner,
                short_window=4,
                short_stride=2,
                feature_dtype="float16",
                build_digest="e" * 64,
            )
            rejection = (output / "cache_rejections.json").read_text(encoding="utf-8")

        self.assertEqual(len(first["shards"]), 1)
        self.assertEqual(first, second)
        self.assertEqual(runner.calls, calls_after_first)
        self.assertIn("bad_sequence", rejection)
        self.assertIn("insufficient translation variance", rejection)

    def test_gt_changes_only_audit_arrays_not_conditions_or_short_target(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            first_manifest = self._manifest(root)
            original = first_manifest.clips[0]
            shifted_gt = original.gt_c2w.copy()
            shifted_gt[:, 2, 3] += 100.0
            second_clip = ClipSpec(
                clip_id=original.clip_id,
                category=original.category,
                sequence_name=original.sequence_name,
                role=original.role,
                start_index=original.start_index,
                temporal_stride=original.temporal_stride,
                frame_numbers=original.frame_numbers,
                image_paths=original.image_paths,
                gt_c2w=shifted_gt,
                focal_length=original.focal_length,
                principal_point=original.principal_point,
                image_size=original.image_size,
            )
            second_manifest = ClipManifest(
                clips=(second_clip,),
                digest="f" * 64,
                clip_length=6,
                source_selection_digest="b" * 64,
            )
            first_output = root / "first"
            second_output = root / "second"
            first_result = build_cache(
                first_manifest,
                output_dir=first_output,
                runner=_FakeRunner(),
                short_window=4,
                short_stride=2,
                feature_dtype="float16",
                build_digest="1" * 64,
            )
            second_result = build_cache(
                second_manifest,
                output_dir=second_output,
                runner=_FakeRunner(),
                short_window=4,
                short_stride=2,
                feature_dtype="float16",
                build_digest="2" * 64,
            )
            first = load_sequence_shard(first_output / first_result["shards"][0]["path"])
            second = load_sequence_shard(second_output / second_result["shards"][0]["path"])

        for name in ("long_hidden", "camera_tokens", "baseline_pose", "short_pose"):
            np.testing.assert_array_equal(first[name], second[name])
        self.assertFalse(np.array_equal(first["gt_c2w_raw"], second["gt_c2w_raw"]))


if __name__ == "__main__":
    unittest.main()
