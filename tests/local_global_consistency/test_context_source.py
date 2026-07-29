from pathlib import Path
import json
import tempfile
import unittest
from unittest import mock

import numpy as np

from pre_experiments.local_global_consistency.context_source import (
    expected_context_frame_count,
    load_context_frame_ids,
    validate_context_source,
)
from pre_experiments.local_global_consistency.split import build_split_manifest


class _GuardedArchive:
    files = [
        "frame_ids",
        "normalized_camera_tokens",
        "pred_c2w_raw",
        "gt_c2w_raw",
    ]

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def __getitem__(self, name):
        if name != "frame_ids":
            raise AssertionError(f"prediction member was read: {name}")
        return np.arange(500, dtype=np.int64)


class ContextFrameIdTest(unittest.TestCase):
    def test_load_context_frame_ids_reads_no_prediction_member(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "context_diagnostics.npz"
            path.touch()
            with mock.patch(
                "pre_experiments.local_global_consistency.context_source.np.load",
                return_value=_GuardedArchive(),
            ):
                frame_ids = load_context_frame_ids(path)

        np.testing.assert_array_equal(frame_ids, np.arange(500))

    def test_load_context_frame_ids_rejects_non_integer_or_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "context_diagnostics.npz"
            path.touch()
            for kind, values in (
                ("non_integer", np.array([0.0, 1.5])),
                ("duplicate", np.array([0, 0], dtype=np.int64)),
            ):
                with self.subTest(kind=kind):
                    with mock.patch(
                        "pre_experiments.local_global_consistency.context_source.np.load"
                    ) as loader:
                        archive = loader.return_value.__enter__.return_value
                        archive.files = ["frame_ids"]
                        archive.__getitem__.return_value = values
                        with self.assertRaises(ValueError):
                            load_context_frame_ids(path)


class ContextSourceContractTest(unittest.TestCase):
    def setUp(self):
        self.scenes = [
            "scene0000_00",
            "scene0013_02",
            "scene0029_01",
            "scene0691_00",
            "scene0150_00",
        ] + [f"scene{index:04d}_00" for index in range(100, 145)]
        trajectories = {}
        for scene_index, scene in enumerate(self.scenes):
            count = 430 if scene == "scene0150_00" else 500
            poses = np.repeat(np.eye(4)[None], count, axis=0)
            poses[:, 0, 3] = np.arange(count) * (scene_index + 1)
            trajectories[scene] = poses
        self.trajectories = trajectories
        self.split = build_split_manifest(
            self.scenes,
            trajectories,
            source_run_id="source-run",
        )

    def test_expected_context_frame_count_has_one_exact_exception(self):
        self.assertEqual(expected_context_frame_count("scene0150_00"), 430)
        self.assertEqual(expected_context_frame_count("scene0000_00"), 500)

    def _write_source(self, root: Path) -> Path:
        source = root / "source"
        metadata = {
            "run_id": "source-run",
            "invocation": {
                "scenes": self.scenes,
                "frame_counts": [500],
                "iterations": [4],
                "sampling": "nested_uniform",
                "preprocess_mode": "pad",
                "save_context_diagnostics": True,
            },
        }
        source.mkdir()
        (source / "run_metadata.json").write_text(
            json.dumps(metadata), encoding="utf-8"
        )
        for scene in self.scenes:
            directory = source / scene / "frames_500"
            directory.mkdir(parents=True)
            poses = self.trajectories[scene]
            count = len(poses)
            np.savez_compressed(
                directory / "context_diagnostics.npz",
                frame_ids=np.arange(count, dtype=np.int64),
                normalized_camera_tokens=np.ones((count, 2)),
                pred_c2w_raw=poses,
                pred_c2w_aligned=poses,
                gt_c2w_raw=poses,
                translation_error_aligned=np.zeros(count),
                rotation_error_deg_aligned=np.zeros(count),
                delta_norm=np.zeros(count),
                sim3_scale=np.array(1.0),
                sim3_rotation=np.eye(3),
                sim3_translation=np.zeros(3),
            )
        return source

    def _scene_frames(self, _data_dir, scene):
        poses = self.trajectories[scene]
        count = len(poses)
        image_by_id = {index: Path(f"{index}.jpg") for index in range(count)}
        poses_by_id = {index: poses[index] for index in range(count)}
        return image_by_id, poses_by_id, list(range(count))

    def test_validate_context_source_requires_exact_protocol_and_raw_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self._write_source(root)
            data_dir = root / "processed"
            data_dir.mkdir()
            with mock.patch(
                "pre_experiments.local_global_consistency.context_source.load_scene_frames",
                side_effect=self._scene_frames,
            ):
                result = validate_context_source(source, self.split, data_dir)

        self.assertEqual(result["source_run_id"], "source-run")
        self.assertEqual(result["scenes"], self.scenes)
        self.assertEqual(len(result["frame_ids_by_scene"]), 50)
        self.assertEqual(len(result["frame_ids_by_scene"]["scene0150_00"]), 430)

    def test_validate_context_source_rejects_wrong_exception_or_short_normal_scene(self):
        cases = (
            ("scene0150_00", 429),
            ("scene0150_00", 431),
            ("scene0100_00", 430),
        )
        for scene, count in cases:
            with self.subTest(scene=scene, count=count):
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    source = self._write_source(root)
                    artifact = (
                        source / scene / "frames_500" / "context_diagnostics.npz"
                    )
                    with np.load(artifact, allow_pickle=False) as archive:
                        arrays = {name: archive[name] for name in archive.files}
                    for name, value in list(arrays.items()):
                        if np.asarray(value).ndim > 0 and len(value) in (430, 500):
                            if count <= len(value):
                                arrays[name] = value[:count]
                            else:
                                arrays[name] = np.concatenate(
                                    [value, value[-1:]], axis=0
                                )
                    arrays["frame_ids"] = np.arange(count, dtype=np.int64)
                    np.savez_compressed(artifact, **arrays)
                    with mock.patch(
                        "pre_experiments.local_global_consistency.context_source.load_scene_frames",
                        side_effect=self._scene_frames,
                    ):
                        with self.assertRaisesRegex(ValueError, "frame IDs|frames"):
                            validate_context_source(
                                source, self.split, root / "processed"
                            )

    def test_validate_context_source_rejects_protocol_or_scene_set_changes(self):
        changes = {
            "frame_counts": [100, 500],
            "iterations": [1, 4],
            "sampling": "random",
            "preprocess_mode": "crop",
            "save_context_diagnostics": False,
        }
        for field, value in changes.items():
            with self.subTest(field=field):
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    source = self._write_source(root)
                    metadata_path = source / "run_metadata.json"
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    metadata["invocation"][field] = value
                    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        validate_context_source(source, self.split, root / "processed")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self._write_source(root)
            extra = source / "scene9999_00" / "frames_500"
            extra.mkdir(parents=True)
            (extra / "context_diagnostics.npz").touch()
            with self.assertRaises(ValueError):
                validate_context_source(source, self.split, root / "processed")

    def test_validate_context_source_rejects_frame_or_raw_gt_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self._write_source(root)
            artifact = source / self.scenes[0] / "frames_500" / "context_diagnostics.npz"
            with np.load(artifact, allow_pickle=False) as archive:
                arrays = {name: archive[name] for name in archive.files}
            arrays["frame_ids"] = np.arange(1, 501, dtype=np.int64)
            np.savez_compressed(artifact, **arrays)
            with mock.patch(
                "pre_experiments.local_global_consistency.context_source.load_scene_frames",
                side_effect=self._scene_frames,
            ):
                with self.assertRaisesRegex(ValueError, "frame IDs"):
                    validate_context_source(source, self.split, root / "processed")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self._write_source(root)
            artifact = source / self.scenes[0] / "frames_500" / "context_diagnostics.npz"
            with np.load(artifact, allow_pickle=False) as archive:
                arrays = {name: archive[name] for name in archive.files}
            arrays["gt_c2w_raw"] = arrays["gt_c2w_raw"].copy()
            arrays["gt_c2w_raw"][10, 0, 3] += 1.0
            np.savez_compressed(artifact, **arrays)
            with mock.patch(
                "pre_experiments.local_global_consistency.context_source.load_scene_frames",
                side_effect=self._scene_frames,
            ):
                with self.assertRaisesRegex(ValueError, "raw GT"):
                    validate_context_source(source, self.split, root / "processed")


if __name__ == "__main__":
    unittest.main()
