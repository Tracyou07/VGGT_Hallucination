from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import zipfile

from pre_experiments.camera_refiner_data_construction import co3d_download
from pre_experiments.camera_refiner_data_construction.co3d_download import (
    archive_url,
    build_curl_command,
    build_dataset_manifest,
    extract_archive_images,
    inspect_data_archive,
    load_eligible_sequences,
    select_archive_sequences,
)


def _viewpoint() -> dict[str, object]:
    return {
        "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "T": [0.0, 0.0, 1.0],
        "focal_length": [1.0, 1.0],
        "principal_point": [0.0, 0.0],
    }


def _frame(category: str, sequence: str, index: int, *, pose: bool = True) -> dict:
    return {
        "sequence_name": sequence,
        "frame_number": index,
        "image": {"path": f"{category}/{sequence}/images/frame{index:06d}.jpg"},
        "viewpoint": _viewpoint() if pose else None,
    }


def _write_jgz(path: Path, payload: object) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)


class Co3DMetadataTest(unittest.TestCase):
    def test_filters_by_quality_frame_count_and_valid_camera_pose(self):
        with TemporaryDirectory() as temporary:
            category_dir = Path(temporary) / "apple"
            category_dir.mkdir()
            frames = []
            frames.extend(_frame("apple", "good", index) for index in range(3))
            frames.extend(_frame("apple", "low_quality", index) for index in range(3))
            frames.extend(_frame("apple", "missing_pose", index, pose=False) for index in range(3))
            frames.extend(_frame("apple", "too_short", index) for index in range(2))
            _write_jgz(category_dir / "frame_annotations.jgz", frames)
            _write_jgz(
                category_dir / "sequence_annotations.jgz",
                [
                    {"sequence_name": "good", "viewpoint_quality_score": 0.9},
                    {"sequence_name": "low_quality", "viewpoint_quality_score": 0.49},
                    {"sequence_name": "missing_pose", "viewpoint_quality_score": 0.9},
                    {"sequence_name": "too_short", "viewpoint_quality_score": 0.9},
                ],
            )

            candidates = load_eligible_sequences(
                category_dir,
                category="apple",
                min_frames=3,
                min_quality=0.5,
            )

        self.assertEqual(tuple(candidates), ("good",))
        self.assertEqual(candidates["good"].valid_frame_count, 3)
        self.assertEqual(candidates["good"].quality_score, 0.9)


class Co3DArchiveTest(unittest.TestCase):
    def test_inspects_and_extracts_only_selected_rgb_members(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata_dir = root / "metadata" / "apple"
            metadata_dir.mkdir(parents=True)
            frames = []
            for sequence in ("seq_a", "seq_b"):
                frames.extend(_frame("apple", sequence, index) for index in range(3))
            _write_jgz(metadata_dir / "frame_annotations.jgz", frames)
            _write_jgz(
                metadata_dir / "sequence_annotations.jgz",
                [
                    {"sequence_name": "seq_a", "viewpoint_quality_score": 0.8},
                    {"sequence_name": "seq_b", "viewpoint_quality_score": 0.9},
                ],
            )
            candidates = load_eligible_sequences(
                metadata_dir,
                category="apple",
                min_frames=3,
                min_quality=0.5,
            )
            archive = root / "apple_001.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                for sequence in ("seq_a", "seq_b"):
                    for index in range(3):
                        handle.writestr(
                            f"payload/apple/{sequence}/images/frame{index:06d}.jpg",
                            f"{sequence}-{index}".encode(),
                        )
                    handle.writestr(
                        f"payload/apple/{sequence}/masks/frame000000.png",
                        b"mask",
                    )
                handle.writestr("../escape.jpg", b"escape")

            available = inspect_data_archive(archive, "apple", candidates)
            selected = select_archive_sequences(
                category="apple",
                available=available,
                needed=1,
                seed=33,
                excluded=set(),
            )
            output_root = root / "dataset"
            counts = extract_archive_images(
                archive,
                output_root,
                {name: available[name] for name in selected},
            )

            self.assertEqual(len(selected), 1)
            selected_name = selected[0]
            self.assertEqual(counts, {selected_name: 3})
            self.assertEqual(
                len(list((output_root / "apple" / selected_name / "images").glob("*.jpg"))),
                3,
            )
            self.assertFalse(any(output_root.rglob("masks")))
            self.assertFalse((root / "escape.jpg").exists())

    def test_selection_is_deterministic_and_excludes_completed_sequences(self):
        available = {
            "seq_a": (),
            "seq_b": (),
            "seq_c": (),
            "seq_d": (),
        }
        expected = tuple(
            sorted(
                ("seq_a", "seq_c", "seq_d"),
                key=lambda name: hashlib.sha256(
                    f"33\0apple\0{name}".encode("utf-8")
                ).hexdigest(),
            )[:2]
        )
        selected = select_archive_sequences(
            category="apple",
            available=available,
            needed=2,
            seed=33,
            excluded={"seq_b"},
        )
        self.assertEqual(selected, expected)

    def test_urls_and_curl_command_use_numbered_archives_and_resume(self):
        self.assertEqual(
            archive_url("https://example.invalid/co3d", "apple", 0),
            "https://example.invalid/co3d/apple_000.zip",
        )
        self.assertEqual(
            archive_url("https://example.invalid/co3d/", "apple", 17),
            "https://example.invalid/co3d/apple_017.zip",
        )
        command = build_curl_command(
            "https://example.invalid/co3d/apple_001.zip",
            Path("apple_001.zip.part"),
        )
        self.assertIn("--continue-at", command)
        self.assertIn("-", command)
        self.assertIn("--retry-all-errors", command)
        self.assertIn("--fail", command)


class Co3DManifestTest(unittest.TestCase):
    def test_manifest_requires_exact_quota_and_has_an_authenticated_selection(self):
        states = {}
        for category in ("apple", "banana"):
            states[category] = {
                "completed": True,
                "selected": [
                    {
                        "sequence_name": f"{category}_{index}",
                        "quality_score": 0.8,
                        "valid_frame_count": 60,
                        "extracted_frame_count": 60,
                        "source_archive": f"{category}_001.zip",
                    }
                    for index in range(2)
                ],
            }
        manifest = build_dataset_manifest(
            categories=("apple", "banana"),
            category_states=states,
            sequences_per_category=2,
            min_frames=50,
            min_quality=0.5,
            seed=33,
            source_base_url="https://example.invalid/co3d",
        )
        self.assertEqual(manifest["category_count"], 2)
        self.assertEqual(manifest["sequence_count"], 4)
        self.assertEqual(len(manifest["selection_sha256"]), 64)

        states["banana"]["selected"].pop()
        with self.assertRaisesRegex(ValueError, "banana.*2"):
            build_dataset_manifest(
                categories=("apple", "banana"),
                category_states=states,
                sequences_per_category=2,
                min_frames=50,
                min_quality=0.5,
                seed=33,
                source_base_url="https://example.invalid/co3d",
            )


class Co3DDownloadIntegrationTest(unittest.TestCase):
    def test_cli_builds_exact_subset_and_rerun_needs_no_download(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixtures = root / "fixtures"
            fixtures.mkdir()
            frames = []
            for sequence in ("seq_a", "seq_b"):
                frames.extend(_frame("apple", sequence, index) for index in range(3))
            metadata_archive = fixtures / "apple_000.zip"
            with zipfile.ZipFile(metadata_archive, "w") as handle:
                handle.writestr(
                    "apple/frame_annotations.jgz",
                    gzip.compress(json.dumps(frames).encode("utf-8")),
                )
                handle.writestr(
                    "apple/sequence_annotations.jgz",
                    gzip.compress(
                        json.dumps(
                            [
                                {
                                    "sequence_name": "seq_a",
                                    "viewpoint_quality_score": 0.8,
                                },
                                {
                                    "sequence_name": "seq_b",
                                    "viewpoint_quality_score": 0.9,
                                },
                            ]
                        ).encode("utf-8")
                    ),
                )
            data_archive = fixtures / "apple_001.zip"
            with zipfile.ZipFile(data_archive, "w") as handle:
                for sequence in ("seq_a", "seq_b"):
                    for index in range(3):
                        handle.writestr(
                            f"apple/{sequence}/images/frame{index:06d}.jpg",
                            f"{sequence}-{index}".encode(),
                        )
                    handle.writestr(
                        f"apple/{sequence}/depths/frame000000.jpg.geometric.png",
                        b"depth",
                    )
            category_file = root / "categories.txt"
            category_file.write_text("apple\n", encoding="utf-8")
            output_root = root / "output"

            def fake_download(url: str, destination: Path, *, curl_bin: str) -> Path:
                del curl_bin
                source = fixtures / url.rsplit("/", 1)[-1]
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
                return destination

            argv = [
                "--output-root",
                str(output_root),
                "--category-file",
                str(category_file),
                "--sequences-per-category",
                "2",
                "--min-frames",
                "3",
                "--min-quality",
                "0.5",
            ]
            with patch.object(co3d_download.shutil, "which", return_value="curl"), patch.object(
                co3d_download, "download_archive", side_effect=fake_download
            ) as downloader:
                co3d_download.main(argv)
                self.assertEqual(downloader.call_count, 2)

            manifest = json.loads(
                (output_root / "download_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["sequence_count"], 2)
            self.assertFalse(any((output_root / ".archives").rglob("*.zip")))
            self.assertFalse(any(output_root.rglob("depths")))

            with patch.object(co3d_download.shutil, "which", return_value="curl"), patch.object(
                co3d_download, "download_archive", side_effect=fake_download
            ) as downloader:
                co3d_download.main(argv)
                downloader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
