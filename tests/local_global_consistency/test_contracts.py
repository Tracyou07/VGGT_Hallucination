import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from pre_experiments.common.contracts import (
    atomic_write_json,
    read_git_commit,
)


class CommonContractTest(unittest.TestCase):
    def test_git_only_tracks_published_numeric_result_types(self):
        repo_root = Path(__file__).resolve().parents[2]

        def is_ignored(path: str) -> bool:
            result = subprocess.run(
                [
                    "git",
                    "-c",
                    "safe.directory=*",
                    "check-ignore",
                    "--no-index",
                    "--quiet",
                    path,
                ],
                cwd=repo_root,
                check=False,
            )
            return result.returncode == 0

        for path in (
            "results/camera_context/run/run_metadata.json",
            "results/camera_context/run/scene0000_00/frames_500/context_diagnostics.npz",
            "results/local_global_consistency/run/prediction_scores_per_frame.csv",
            "results/local_global_consistency/run/local_global_summary.json",
        ):
            self.assertFalse(is_ignored(path), path)
        for path in (
            "results/unpublished/output.json",
            "results/local_global_consistency/run/window_diagnostics.npz",
        ):
            self.assertTrue(is_ignored(path), path)

    def test_atomic_json_replaces_temporary_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "nested" / "metadata.json"
            atomic_write_json(output, {"iteration": 4})

            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), {"iteration": 4})
            self.assertFalse(output.with_suffix(".json.tmp").exists())

    def test_read_git_commit_returns_full_lowercase_hash(self):
        repo_root = Path(__file__).resolve().parents[2]
        self.assertRegex(read_git_commit(repo_root), r"^[0-9a-f]{40}$")


if __name__ == "__main__":
    unittest.main()
