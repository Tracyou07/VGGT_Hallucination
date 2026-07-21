import json
from pathlib import Path
import tempfile
import unittest

from scripts.autodl.camera_head_amplification.export_numeric_results import (
    export_numeric_results,
)


FILES = (
    "run_metadata.json",
    "complete.json",
    "baseline_checks.json",
    "amplification_summary.json",
    "amplification_summary.csv",
    "amplification_per_frame.csv",
    "pose_metrics.json",
    "pose_metrics.csv",
    "pose_per_frame.csv",
)


class CameraHeadAmplificationExportTest(unittest.TestCase):
    def make_run(self, root: Path) -> Path:
        run = root / "abc1234_deadbeef0000"
        run.mkdir()
        metadata = {
            "study_name": "camera_head_amplification",
            "run_id": run.name,
            "git_commit": "a" * 40,
        }
        complete = {"run_id": run.name, "baseline_checks_passed": True}
        for name in FILES:
            if name == "run_metadata.json":
                value = json.dumps(metadata)
            elif name == "complete.json":
                value = json.dumps(complete)
            elif name.endswith(".json"):
                value = "[]"
            else:
                value = "scene,value\nscene0000_00,1\n"
            (run / name).write_text(value + "\n", encoding="utf-8")
        return run

    def test_exports_only_fixed_numeric_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self.make_run(root)
            destination = export_numeric_results(source, root / "published")

            self.assertEqual({path.name for path in destination.iterdir()}, set(FILES) | {"publish_manifest.json"})
            manifest = json.loads(
                (destination / "publish_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["file_count"], len(FILES))

    def test_rejects_high_dimensional_artifacts_anywhere_in_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self.make_run(root)
            (source / "activations.npz").write_bytes(b"not allowed")

            with self.assertRaisesRegex(ValueError, "forbidden artifact"):
                export_numeric_results(source, root / "published")

    def test_requires_successful_baseline_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self.make_run(root)
            (source / "complete.json").write_text(
                json.dumps(
                    {"run_id": source.name, "baseline_checks_passed": False}
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "baseline"):
                export_numeric_results(source, root / "published")


if __name__ == "__main__":
    unittest.main()
