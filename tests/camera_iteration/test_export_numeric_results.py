import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zipfile


MODULE = "scripts.autodl.camera_iteration.export_numeric_results"
ROOT_FILES = ("run_metadata.json", "summary.json", "summary.csv")
SELECTION_FILES = (
    "iteration_metrics.json",
    "iteration_metrics.csv",
    "selected_frame_ids.json",
    "complete.json",
    "camera_trace.npz",
)
TRACE_MEMBERS = (
    "frame_ids.npy",
    "pose_enc_by_iteration.npy",
    "raw_pose_enc_by_iteration.npy",
    "pose_delta_by_iteration.npy",
    "delta_norm.npy",
)
CONTEXT_ROOT_FILES = (
    "context_per_frame.csv",
    "context_summary.csv",
    "context_summary.json",
)
CONTEXT_MEMBERS = (
    "frame_ids.npy",
    "normalized_camera_tokens.npy",
    "pred_c2w_raw.npy",
    "pred_c2w_aligned.npy",
    "gt_c2w_raw.npy",
    "translation_error_aligned.npy",
    "rotation_error_deg_aligned.npy",
    "delta_norm.npy",
    "sim3_scale.npy",
    "sim3_rotation.npy",
    "sim3_translation.npy",
)


def load_exporter():
    if importlib.util.find_spec(MODULE) is None:
        raise AssertionError(f"missing exporter module: {MODULE}")
    return importlib.import_module(MODULE)


def write_trace(path: Path, members=TRACE_MEMBERS, payload: bytes = b"numbers") -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for member in members:
            archive.writestr(member, payload)


def make_source(root: Path, run_id: str = "commit_1234") -> Path:
    source = root / "external" / run_id
    selection = source / "scene0000_00" / "frames_25"
    selection.mkdir(parents=True)
    (source / "run_metadata.json").write_text(
        json.dumps({"run_id": run_id, "git_commit": "a" * 40}),
        encoding="utf-8",
    )
    (source / "summary.json").write_text("[]", encoding="utf-8")
    (source / "summary.csv").write_text("scene,iteration\n", encoding="utf-8")
    (selection / "iteration_metrics.json").write_text("[]", encoding="utf-8")
    (selection / "iteration_metrics.csv").write_text(
        "scene,iteration\n", encoding="utf-8"
    )
    (selection / "selected_frame_ids.json").write_text("[0]", encoding="utf-8")
    (selection / "complete.json").write_text(
        json.dumps({"run_id": run_id}), encoding="utf-8"
    )
    write_trace(selection / "camera_trace.npz")
    (source / "point_cloud.ply").write_bytes(b"large point cloud")
    (selection / "preview.jpg").write_bytes(b"image")
    return source


def add_context_artifacts(source: Path) -> None:
    metadata_path = source / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["invocation"] = {"save_context_diagnostics": True}
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    for name in CONTEXT_ROOT_FILES:
        content = "[]" if name.endswith(".json") else "scene,short_frames,long_frames\n"
        (source / name).write_text(content, encoding="utf-8")
    selection = source / "scene0000_00" / "frames_25"
    write_trace(selection / "context_diagnostics.npz", CONTEXT_MEMBERS)


class NumericResultExporterTest(unittest.TestCase):
    def test_exports_only_numeric_whitelist_and_writes_manifest(self):
        exporter = load_exporter()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = make_source(root)
            destination = exporter.export_numeric_results(
                source, root / "published", max_file_bytes=1024 * 1024
            )

            expected = set(ROOT_FILES)
            expected.update(
                f"scene0000_00/frames_25/{name}" for name in SELECTION_FILES
            )
            copied = {
                path.relative_to(destination).as_posix()
                for path in destination.rglob("*")
                if path.is_file() and path.name != "publish_manifest.json"
            }
            self.assertEqual(copied, expected)
            self.assertFalse((destination / "point_cloud.ply").exists())
            self.assertFalse(
                (destination / "scene0000_00" / "frames_25" / "preview.jpg").exists()
            )

            manifest = json.loads(
                (destination / "publish_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["run_id"], "commit_1234")
            self.assertEqual(manifest["file_count"], len(expected))
            self.assertEqual(
                [entry["path"] for entry in manifest["files"]], sorted(expected)
            )
            for entry in manifest["files"]:
                content = (destination / entry["path"]).read_bytes()
                self.assertEqual(entry["bytes"], len(content))
                self.assertEqual(entry["sha256"], hashlib.sha256(content).hexdigest())

    def test_exports_declared_context_diagnostics(self):
        exporter = load_exporter()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = make_source(root)
            add_context_artifacts(source)

            destination = exporter.export_numeric_results(
                source, root / "published", max_file_bytes=1024 * 1024
            )

            for name in CONTEXT_ROOT_FILES:
                self.assertTrue((destination / name).is_file())
            self.assertTrue(
                (
                    destination
                    / "scene0000_00"
                    / "frames_25"
                    / "context_diagnostics.npz"
                ).is_file()
            )

    def test_rejects_existing_destination(self):
        exporter = load_exporter()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = make_source(root)
            (root / "published" / "commit_1234").mkdir(parents=True)
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                exporter.export_numeric_results(source, root / "published", 1024 * 1024)

    def test_rejects_mismatched_completion_run_id(self):
        exporter = load_exporter()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = make_source(root)
            complete = source / "scene0000_00" / "frames_25" / "complete.json"
            complete.write_text(json.dumps({"run_id": "wrong"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "run_id"):
                exporter.export_numeric_results(source, root / "published", 1024 * 1024)

    def test_rejects_camera_token_arrays(self):
        exporter = load_exporter()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = make_source(root)
            trace = source / "scene0000_00" / "frames_25" / "camera_trace.npz"
            write_trace(trace, TRACE_MEMBERS + ("normalized_camera_tokens.npy",))
            with self.assertRaisesRegex(ValueError, "Camera Token"):
                exporter.export_numeric_results(source, root / "published", 1024 * 1024)

    def test_rejects_oversized_allowed_file(self):
        exporter = load_exporter()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = make_source(root)
            summary = source / "summary.csv"
            summary.write_bytes(b"x" * 513)
            with self.assertRaisesRegex(ValueError, "size limit"):
                exporter.export_numeric_results(source, root / "published", 512)


if __name__ == "__main__":
    unittest.main()
