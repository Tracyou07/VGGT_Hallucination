import json
from pathlib import Path
import tempfile
import unittest

from scripts.autodl.local_global_consistency.export_numeric_results import (
    FILES,
    export_numeric_results,
)


class LocalGlobalExportTest(unittest.TestCase):
    def make_run(self, root: Path) -> Path:
        run = root / "abc1234_deadbeef0000"
        run.mkdir()
        metadata = {
            "study_name": "local_global_consistency",
            "run_id": run.name,
            "git_commit": "a" * 40,
        }
        complete = {"run_id": run.name, "analysis_complete": True}
        for name in FILES:
            if name == "run_metadata.json":
                content = json.dumps(metadata)
            elif name == "complete.json":
                content = json.dumps(complete)
            elif name.endswith(".json"):
                content = "[]" if "summary" in name else "{}"
            else:
                content = "scene,value\nscene0000_00,1"
            (run / name).write_text(content + "\n", encoding="utf-8")
        raw = run / "scene0000_00" / "window_000"
        raw.mkdir(parents=True)
        (raw / "window_diagnostics.npz").write_bytes(b"external raw input")
        return run

    def test_exports_only_root_scalar_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self.make_run(root)

            destination = export_numeric_results(source, root / "published")

            self.assertEqual(
                {path.name for path in destination.iterdir()},
                set(FILES) | {"publish_manifest.json"},
            )
            self.assertFalse(any(path.suffix == ".npz" for path in destination.rglob("*")))

    def test_rejects_unexpected_high_dimensional_root_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self.make_run(root)
            (source / "tokens.npz").write_bytes(b"forbidden")

            with self.assertRaisesRegex(ValueError, "unexpected high-dimensional"):
                export_numeric_results(source, root / "published")

    def test_requires_completed_analysis(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self.make_run(root)
            (source / "complete.json").write_text(
                json.dumps({"run_id": source.name, "analysis_complete": False}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "analysis"):
                export_numeric_results(source, root / "published")


if __name__ == "__main__":
    unittest.main()
