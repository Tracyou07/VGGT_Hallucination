from __future__ import annotations

from io import BytesIO
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import warnings
import zipfile

import numpy as np

from pre_experiments.camera_translation_hvrfm import artifacts as artifacts_module
from pre_experiments.camera_translation_hvrfm.artifacts import (
    LONG_CONTEXT_MEMBERS,
    QUALITY_SIDECAR_MEMBERS,
    SHORT_CONTEXT_MEMBERS,
    TRANSLATION_TARGET_MEMBERS,
    load_bound_bundle,
    load_long_context,
    load_quality_sidecar,
    load_short_context,
    load_translation_target,
    save_long_context,
    save_quality_sidecar,
    save_short_context,
    save_translation_target,
)


FRAME_COUNT = 500


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _frame_digest(frame_ids: np.ndarray) -> str:
    payload = json.dumps(
        [int(value) for value in frame_ids], separators=(",", ":")
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _oracle_digest(frame_ids: np.ndarray) -> str:
    payload = {
        "scene": "scene0029_01",
        "frame_digest": _frame_digest(frame_ids),
        "fit_count": 500,
        "scale": 1.0,
        "rotation": ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        "translation": (0.0, 0.0, 0.0),
    }
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _base_camera() -> tuple[np.ndarray, np.ndarray, float]:
    centers = np.zeros((FRAME_COUNT, 3), dtype=np.float64)
    centers[:, 0] = np.arange(FRAME_COUNT, dtype=np.float64) - 249.5
    c2w = np.broadcast_to(np.eye(4, dtype=np.float64), (FRAME_COUNT, 4, 4)).copy()
    c2w[:, :3, 3] = centers
    pose = np.zeros((FRAME_COUNT, 9), dtype=np.float32)
    pose[:, :3] = (-centers).astype(np.float32)
    pose[:, 6] = 1.0
    pose[:, 7:] = 1.0
    scale = float(np.sqrt(np.mean(np.sum((centers - centers.mean(0)) ** 2, axis=1))))
    return c2w, pose, scale


class ArtifactFixture:
    sample_id = "scene0029_01:frames_500"
    scene = "scene0029_01"
    source_sha256 = "1" * 64
    checkpoint_sha256 = "2" * 64
    formal_label_sha256 = "3" * 64
    teacher_reference_sha256 = "4" * 64
    git_commit = "5" * 40

    def __init__(self) -> None:
        self.frame_ids = np.arange(2000, 2500, dtype=np.int64)
        self.c2w, self.pose, self.scale = _base_camera()

    def long(self) -> dict[str, np.ndarray]:
        return {
            "sample_id": np.asarray(self.sample_id, dtype="U96"),
            "scene": np.asarray(self.scene, dtype="U32"),
            "frame_ids": self.frame_ids.copy(),
            "camera_tokens": np.zeros((500, 2048), dtype=np.float32),
            "baseline_pose_encoding": self.pose.copy(),
            "baseline_c2w": self.c2w.copy(),
            "prediction_scale": np.asarray(self.scale, dtype=np.float64),
            "source_sha256": np.asarray(self.source_sha256, dtype="U64"),
            "checkpoint_sha256": np.asarray(self.checkpoint_sha256, dtype="U64"),
            "git_commit": np.asarray(self.git_commit, dtype="U40"),
        }

    def short(self, long_sha256: str) -> dict[str, np.ndarray]:
        starts = range(0, 401, 50)
        return {
            "sample_id": np.asarray(self.sample_id, dtype="U96"),
            "scene": np.asarray(self.scene, dtype="U32"),
            "short_frame_ids": np.stack(
                [self.frame_ids[start : start + 100] for start in starts]
            ),
            "short_camera_tokens": np.zeros((9, 100, 2048), dtype=np.float32),
            "long_context_sha256": np.asarray(long_sha256, dtype="U64"),
            "source_sha256": np.asarray(self.source_sha256, dtype="U64"),
            "checkpoint_sha256": np.asarray(self.checkpoint_sha256, dtype="U64"),
            "git_commit": np.asarray(self.git_commit, dtype="U40"),
        }

    def quality(self) -> dict[str, np.ndarray]:
        coverage = np.ones((4, 500), dtype=np.float64)
        coverage[0, ::5] = 0.0
        coverage[1, 100:150] = 0.0
        coverage[2, 300:] = 0.0
        coverage[3, :20] = 0.0
        teacher_translation = np.full((4, 500), 0.25, dtype=np.float64)
        teacher_rotation = np.full((4, 500), 0.5, dtype=np.float64)
        teacher_translation[coverage == 0.0] = np.nan
        teacher_rotation[coverage == 0.0] = np.nan
        return {
            "sample_id": np.asarray(self.sample_id, dtype="U96"),
            "scene": np.asarray(self.scene, dtype="U32"),
            "frame_ids": self.frame_ids.copy(),
            "teacher_variant_ids": np.arange(4, dtype=np.int64),
            "gt_c2w": self.c2w.copy(),
            "gt_scene_scale": np.asarray(self.scale, dtype=np.float64),
            "oracle_scene": np.asarray(self.scene, dtype="U32"),
            "oracle_frame_digest": np.asarray(_frame_digest(self.frame_ids), dtype="U64"),
            "oracle_fit_count": np.asarray(500, dtype=np.int64),
            "oracle_scale": np.asarray(1.0, dtype=np.float64),
            "oracle_rotation": np.eye(3, dtype=np.float64),
            "oracle_translation": np.zeros(3, dtype=np.float64),
            "oracle_rank": np.asarray(3, dtype=np.int64),
            "oracle_condition": np.asarray(1.0, dtype=np.float64),
            "oracle_digest": np.asarray(_oracle_digest(self.frame_ids), dtype="U64"),
            "window_weights": np.ones(9, dtype=np.float64),
            "window_masks": np.ones((4, 9), dtype=np.uint8),
            "coverage_weights": coverage,
            "variant_utilities": np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float64),
            "baseline_translation_error_normalized": np.ones(500, dtype=np.float64),
            "baseline_rotation_error_deg": np.ones(500, dtype=np.float64),
            "teacher_translation_error_normalized": teacher_translation,
            "teacher_rotation_error_deg": teacher_rotation,
            "source_sha256": np.asarray(self.source_sha256, dtype="U64"),
            "formal_label_sha256": np.asarray(self.formal_label_sha256, dtype="U64"),
            "teacher_reference_sha256": np.asarray(
                self.teacher_reference_sha256, dtype="U64"
            ),
            "checkpoint_sha256": np.asarray(self.checkpoint_sha256, dtype="U64"),
            "git_commit": np.asarray(self.git_commit, dtype="U40"),
        }

    def target(
        self, long_sha256: str, short_sha256: str, quality_sha256: str
    ) -> dict[str, np.ndarray]:
        quality = self.quality()
        coverage = (quality["coverage_weights"] > 0.0).astype(np.uint8)
        endpoints = np.zeros((4, 500, 3), dtype=np.float32)
        endpoints[coverage != 0] = np.asarray([0.1, -0.2, 0.3], dtype=np.float32)
        centers = np.broadcast_to(self.c2w[None, :, :3, 3], (4, 500, 3)).copy()
        return {
            "sample_id": np.asarray(self.sample_id, dtype="U96"),
            "scene": np.asarray(self.scene, dtype="U32"),
            "frame_ids": self.frame_ids.copy(),
            "teacher_variant_ids": np.arange(4, dtype=np.int64),
            "coverage_mask": coverage,
            "translation_endpoints": endpoints,
            "teacher_centers_raw_filled": centers,
            "prediction_scale": np.asarray(self.scale, dtype=np.float64),
            "long_context_sha256": np.asarray(long_sha256, dtype="U64"),
            "short_context_sha256": np.asarray(short_sha256, dtype="U64"),
            "quality_sha256": np.asarray(quality_sha256, dtype="U64"),
            "teacher_reference_sha256": np.asarray(
                self.teacher_reference_sha256, dtype="U64"
            ),
            "source_sha256": np.asarray(self.source_sha256, dtype="U64"),
            "checkpoint_sha256": np.asarray(self.checkpoint_sha256, dtype="U64"),
            "git_commit": np.asarray(self.git_commit, dtype="U40"),
        }


class StrictArtifactTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.fixture = ArtifactFixture()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _save_bundle(self) -> tuple[Path, Path, Path, Path]:
        long_path = self.root / "long.npz"
        short_path = self.root / "short.npz"
        quality_path = self.root / "quality.npz"
        target_path = self.root / "target.npz"
        long_digest = save_long_context(long_path, self.fixture.long())
        short_digest = save_short_context(short_path, self.fixture.short(long_digest))
        quality_digest = save_quality_sidecar(quality_path, self.fixture.quality())
        save_translation_target(
            target_path,
            self.fixture.target(long_digest, short_digest, quality_digest),
        )
        return long_path, short_path, target_path, quality_path

    def _bytes_api(self):
        self.assertTrue(
            hasattr(artifacts_module, "load_bound_bundle_bytes"),
            "Task 3a requires a strict in-memory bound-bundle loader",
        )
        return artifacts_module.load_bound_bundle_bytes

    def test_bound_bundle_bytes_parses_authenticated_archives_without_paths(self) -> None:
        load_bytes = self._bytes_api()
        paths = self._save_bundle()
        payloads = [path.read_bytes() for path in paths]
        for path in paths:
            path.unlink()
        bundle = load_bytes(*payloads)
        self.assertEqual(set(bundle), {"long", "short", "target", "quality"})
        self.assertEqual(str(bundle["long"]["sample_id"]), self.fixture.sample_id)
        np.testing.assert_array_equal(
            bundle["target"]["coverage_mask"],
            (bundle["quality"]["coverage_weights"] > 0.0).astype(np.uint8),
        )

    def test_bound_bundle_bytes_rejects_duplicate_object_and_extra_archives(self) -> None:
        load_bytes = self._bytes_api()
        paths = self._save_bundle()
        valid = [path.read_bytes() for path in paths]
        duplicate = self.root / "bytes-duplicate.npz"
        self._write_raw_zip(duplicate, self.fixture.long(), duplicate="sample_id")
        object_path = self.root / "bytes-object.npz"
        object_arrays = self.fixture.long()
        object_arrays["sample_id"] = np.asarray(self.fixture.sample_id, dtype=object)
        self._write_raw_zip(object_path, object_arrays)
        extra = self.root / "bytes-extra.npz"
        with extra.open("wb") as handle:
            np.savez_compressed(handle, **self.fixture.long(), extra=np.asarray(1))
        for name, malicious in (
            ("duplicate", duplicate.read_bytes()),
            ("object", object_path.read_bytes()),
            ("extra", extra.read_bytes()),
        ):
            payloads = list(valid)
            payloads[0] = malicious
            with self.subTest(name=name), self.assertRaises(ValueError):
                load_bytes(*payloads)

    def test_bound_bundle_bytes_replays_every_actual_digest_binding(self) -> None:
        load_bytes = self._bytes_api()
        long_path, short_path, target_path, quality_path = self._save_bundle()
        original = [
            path.read_bytes()
            for path in (long_path, short_path, target_path, quality_path)
        ]
        original_long_sha256 = _sha256(long_path)

        changed_long = self.fixture.long()
        changed_long["camera_tokens"] = changed_long["camera_tokens"].copy()
        changed_long["camera_tokens"][0, 0] = 1.0
        save_long_context(long_path, changed_long)

        short = self.fixture.short(original_long_sha256)
        short["short_camera_tokens"] = short["short_camera_tokens"].copy()
        short["short_camera_tokens"][0, 0, 0] = 1.0
        save_short_context(short_path, short)

        quality = self.fixture.quality()
        quality["variant_utilities"] = quality["variant_utilities"].copy()
        quality["variant_utilities"][0] += 0.01
        save_quality_sidecar(quality_path, quality)

        target = load_translation_target(target_path)
        target["quality_sha256"] = np.asarray("f" * 64, dtype="U64")
        save_translation_target(target_path, target)

        malicious = [
            long_path.read_bytes(),
            short_path.read_bytes(),
            target_path.read_bytes(),
            quality_path.read_bytes(),
        ]
        for index, name in enumerate(("long", "short", "target", "quality")):
            payloads = list(original)
            payloads[index] = malicious[index]
            with self.subTest(name=name), self.assertRaises(ValueError):
                load_bytes(*payloads)

    def test_round_trip_uses_all_four_exact_schemas_and_dtypes(self) -> None:
        paths = self._save_bundle()
        loaders = (
            (load_long_context, paths[0], LONG_CONTEXT_MEMBERS),
            (load_short_context, paths[1], SHORT_CONTEXT_MEMBERS),
            (load_translation_target, paths[2], TRANSLATION_TARGET_MEMBERS),
            (load_quality_sidecar, paths[3], QUALITY_SIDECAR_MEMBERS),
        )
        for loader, path, members in loaders:
            with self.subTest(path=path.name):
                loaded = loader(path)
                self.assertEqual(set(loaded), set(members))
                self.assertEqual(_sha256(path), hashlib.sha256(path.read_bytes()).hexdigest())
        bundle = load_bound_bundle(*paths)
        self.assertEqual(set(bundle), {"long", "short", "target", "quality"})

    def test_save_rejects_extra_missing_object_non_array_wrong_dtype_and_shape(self) -> None:
        cases = []
        for name, save, arrays in (
            ("long", save_long_context, self.fixture.long()),
            ("short", save_short_context, self.fixture.short("a" * 64)),
            ("target", save_translation_target, self.fixture.target("a" * 64, "b" * 64, "c" * 64)),
            ("quality", save_quality_sidecar, self.fixture.quality()),
        ):
            extra = dict(arrays)
            extra["extra"] = np.asarray(0, dtype=np.int64)
            cases.append((f"{name}_extra", save, extra))
            missing = dict(arrays)
            missing.pop(next(iter(missing)))
            cases.append((f"{name}_missing", save, missing))
        object_long = self.fixture.long()
        object_long["sample_id"] = np.asarray(self.fixture.sample_id, dtype=object)
        cases.append(("object", save_long_context, object_long))
        list_long = self.fixture.long()
        list_long["frame_ids"] = list(range(500))  # type: ignore[assignment]
        cases.append(("non_array", save_long_context, list_long))
        dtype_long = self.fixture.long()
        dtype_long["frame_ids"] = dtype_long["frame_ids"].astype(np.int32)
        cases.append(("dtype", save_long_context, dtype_long))
        shape_short = self.fixture.short("a" * 64)
        shape_short["short_camera_tokens"] = shape_short["short_camera_tokens"][:, :-1]
        cases.append(("shape", save_short_context, shape_short))
        for name, save, arrays in cases:
            with self.subTest(name=name), self.assertRaises(ValueError):
                save(self.root / f"{name}.npz", arrays)

    @staticmethod
    def _npy_bytes(value: np.ndarray) -> bytes:
        handle = BytesIO()
        np.save(handle, value, allow_pickle=True)
        return handle.getvalue()

    def _write_raw_zip(
        self, path: Path, arrays: dict[str, np.ndarray], *, duplicate: str | None = None,
        traversal: bool = False,
    ) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with zipfile.ZipFile(path, "w") as archive:
                for name, value in arrays.items():
                    archive.writestr(f"{name}.npy", self._npy_bytes(value))
                if duplicate is not None:
                    archive.writestr(f"{duplicate}.npy", self._npy_bytes(arrays[duplicate]))
                if traversal:
                    archive.writestr("../escape.npy", self._npy_bytes(np.asarray(1)))

    def test_load_rejects_duplicate_traversal_extra_and_object_zip_members(self) -> None:
        arrays = self.fixture.long()
        duplicate = self.root / "duplicate.npz"
        self._write_raw_zip(duplicate, arrays, duplicate="sample_id")
        traversal = self.root / "traversal.npz"
        self._write_raw_zip(traversal, arrays, traversal=True)
        extra = self.root / "extra.npz"
        with extra.open("wb") as handle:
            np.savez_compressed(handle, **arrays, extra=np.asarray(1))
        object_path = self.root / "object.npz"
        object_arrays = dict(arrays)
        object_arrays["sample_id"] = np.asarray(self.fixture.sample_id, dtype=object)
        with object_path.open("wb") as handle:
            np.savez_compressed(handle, **object_arrays)
        for name, path in (
            ("duplicate", duplicate),
            ("traversal", traversal),
            ("extra", extra),
            ("object", object_path),
        ):
            with self.subTest(name=name), self.assertRaises(ValueError):
                load_long_context(path)

    def test_load_and_save_reject_file_and_parent_symlinks(self) -> None:
        real = self.root / "real.npz"
        save_long_context(real, self.fixture.long())
        file_link = self.root / "file_link.npz"
        linked_directory = self.root / "real_directory"
        linked_directory.mkdir()
        directory_link = self.root / "directory_link"
        try:
            os.symlink(real, file_link)
            os.symlink(linked_directory, directory_link, target_is_directory=True)
        except OSError as error:
            self.skipTest(f"symlink creation unavailable: {error}")
        with self.assertRaises(ValueError):
            load_long_context(file_link)
        with self.assertRaises(ValueError):
            save_long_context(directory_link / "new.npz", self.fixture.long())

    def test_every_artifact_api_rejects_lexical_parent_traversal(self) -> None:
        safe = self.root / "safe"
        safe.mkdir()
        save_cases = (
            (save_long_context, self.fixture.long()),
            (save_short_context, self.fixture.short("a" * 64)),
            (
                save_translation_target,
                self.fixture.target("a" * 64, "b" * 64, "c" * 64),
            ),
            (save_quality_sidecar, self.fixture.quality()),
        )
        for index, (save, arrays) in enumerate(save_cases):
            path = safe / ".." / f"save-{index}.npz"
            with self.subTest(api=save.__name__), self.assertRaisesRegex(
                ValueError, "parent traversal"
            ):
                save(path, arrays)

        real_paths = self._save_bundle()
        load_cases = (
            (load_long_context, real_paths[0]),
            (load_short_context, real_paths[1]),
            (load_translation_target, real_paths[2]),
            (load_quality_sidecar, real_paths[3]),
        )
        lexical_paths = tuple(safe / ".." / path.name for _, path in load_cases)
        for (load, _), path in zip(load_cases, lexical_paths):
            with self.subTest(api=load.__name__), self.assertRaisesRegex(
                ValueError, "parent traversal"
            ):
                load(path)
        with self.assertRaisesRegex(ValueError, "parent traversal"):
            load_bound_bundle(*lexical_paths)

    def test_link_parent_traversal_cannot_escape_on_posix(self) -> None:
        safe = self.root / "safe-link-parent"
        outside = self.root / "outside"
        safe.mkdir()
        (outside / "subdirectory").mkdir(parents=True)
        secret = outside / "secret.npz"
        save_long_context(secret, self.fixture.long())
        link = safe / "link"
        try:
            os.symlink(outside / "subdirectory", link, target_is_directory=True)
        except OSError as error:
            self.skipTest(f"symlink creation unavailable: {error}")
        attack = link / ".." / secret.name
        self.assertTrue(attack.is_file())
        with self.assertRaisesRegex(ValueError, "parent traversal"):
            load_long_context(attack)

    def test_failed_write_is_atomic_and_cleans_unique_same_directory_tempfile(self) -> None:
        target = self.root / "long.npz"
        sentinel = b"existing-target-bytes"
        target.write_bytes(sentinel)

        def fail_after_partial_write(handle: object, **_: np.ndarray) -> None:
            handle.write(b"partial")  # type: ignore[attr-defined]
            raise OSError("injected write failure")

        with mock.patch(
            "pre_experiments.camera_translation_hvrfm.artifacts.np.savez_compressed",
            side_effect=fail_after_partial_write,
        ):
            with self.assertRaises(OSError):
                save_long_context(target, self.fixture.long())
        self.assertEqual(target.read_bytes(), sentinel)
        self.assertEqual([path.name for path in self.root.iterdir()], ["long.npz"])

    def test_content_validators_reject_bitwise_and_diagnostic_contract_breaks(self) -> None:
        bad_scale = self.fixture.long()
        bad_scale["prediction_scale"] = np.asarray(self.fixture.scale + 1.0, dtype=np.float64)
        with self.assertRaises(ValueError):
            save_long_context(self.root / "bad_scale.npz", bad_scale)

        target = self.fixture.target("a" * 64, "b" * 64, "c" * 64)
        uncovered = target["coverage_mask"] == 0
        target["translation_endpoints"][uncovered] = np.float32(-0.0)
        with self.assertRaises(ValueError):
            save_translation_target(self.root / "negative_zero.npz", target)

        quality = self.fixture.quality()
        first_uncovered = np.argwhere(quality["coverage_weights"] == 0.0)[0]
        quality["teacher_translation_error_normalized"][tuple(first_uncovered)] = 0.0
        with self.assertRaises(ValueError):
            save_quality_sidecar(self.root / "finite_uncovered.npz", quality)

        quality = self.fixture.quality()
        quality["teacher_rotation_error_deg"][0, 1] = np.nan
        with self.assertRaises(ValueError):
            save_quality_sidecar(self.root / "nan_covered.npz", quality)

        quality = self.fixture.quality()
        quality["oracle_digest"] = np.asarray("f" * 64, dtype="U64")
        with self.assertRaises(ValueError):
            save_quality_sidecar(self.root / "oracle_digest.npz", quality)

    def test_bundle_binds_actual_hashes_ids_frames_coverage_scale_and_teacher_reference(self) -> None:
        long_path, short_path, target_path, quality_path = self._save_bundle()
        original_target = load_translation_target(target_path)

        mismatched_target = dict(original_target)
        mismatched_target["long_context_sha256"] = np.asarray("a" * 64, dtype="U64")
        save_translation_target(target_path, mismatched_target)
        with self.assertRaises(ValueError):
            load_bound_bundle(long_path, short_path, target_path, quality_path)

        mismatched_target = dict(original_target)
        mismatched_target["teacher_reference_sha256"] = np.asarray("b" * 64, dtype="U64")
        save_translation_target(target_path, mismatched_target)
        with self.assertRaises(ValueError):
            load_bound_bundle(long_path, short_path, target_path, quality_path)

        mismatched_target = dict(original_target)
        mismatched_target["frame_ids"] = mismatched_target["frame_ids"].copy()
        mismatched_target["frame_ids"][-1] += 1
        save_translation_target(target_path, mismatched_target)
        with self.assertRaises(ValueError):
            load_bound_bundle(long_path, short_path, target_path, quality_path)

        mismatched_target = dict(original_target)
        mismatched_target["coverage_mask"] = mismatched_target["coverage_mask"].copy()
        mismatched_target["coverage_mask"][0, 0] = 1
        save_translation_target(target_path, mismatched_target)
        with self.assertRaises(ValueError):
            load_bound_bundle(long_path, short_path, target_path, quality_path)

        mismatched_target = dict(original_target)
        mismatched_target["prediction_scale"] = np.asarray(
            np.nextafter(
                mismatched_target["prediction_scale"],
                np.asarray(np.inf, dtype=np.float64),
            ),
            dtype=np.float64,
        )
        save_translation_target(target_path, mismatched_target)
        with self.assertRaises(ValueError):
            load_bound_bundle(long_path, short_path, target_path, quality_path)

        save_translation_target(target_path, original_target)
        with short_path.open("ab") as handle:
            handle.write(b"digest-tamper")
        with self.assertRaises(ValueError):
            load_bound_bundle(long_path, short_path, target_path, quality_path)

    def test_bundle_rejects_short_window_frame_or_common_identity_mismatch(self) -> None:
        long_path, short_path, target_path, quality_path = self._save_bundle()
        short = load_short_context(short_path)
        short["short_frame_ids"] = short["short_frame_ids"].copy()
        short["short_frame_ids"][0, 0] -= 1
        short_digest = save_short_context(short_path, short)
        target = load_translation_target(target_path)
        target["short_context_sha256"] = np.asarray(short_digest, dtype="U64")
        save_translation_target(target_path, target)
        with self.assertRaises(ValueError):
            load_bound_bundle(long_path, short_path, target_path, quality_path)

        short = self.fixture.short(_sha256(long_path))
        short["sample_id"] = np.asarray("scene0029_01:different", dtype="U96")
        short_digest = save_short_context(short_path, short)
        target["short_context_sha256"] = np.asarray(short_digest, dtype="U64")
        save_translation_target(target_path, target)
        with self.assertRaises(ValueError):
            load_bound_bundle(long_path, short_path, target_path, quality_path)


if __name__ == "__main__":
    unittest.main()
