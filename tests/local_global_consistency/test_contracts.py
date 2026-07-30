import ast
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
    def test_run_study_imports_every_artifact_helper_it_calls(self):
        repo_root = Path(__file__).resolve().parents[2]
        path = (
            repo_root
            / "pre_experiments"
            / "local_global_consistency"
            / "run_study.py"
        )
        tree = ast.parse(path.read_text(encoding="utf-8"))
        artifact_imports = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            and node.module
            == "pre_experiments.local_global_consistency.artifacts"
            for alias in node.names
        }
        called_names = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }

        self.assertIn("load_window_diagnostics", called_names)
        self.assertIn("load_window_diagnostics", artifact_imports)

    def test_git_ignores_generated_results_but_tracks_reproducibility_inputs(self):
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
            "results/camera_hidden_causal_preference/run/per_position.csv",
            "results/camera_hidden_causal_preference/run/summary.json",
            "results/unpublished/output.json",
            "results/local_global_consistency/run/window_diagnostics.npz",
            "results/camera_hidden_causal_preference/run/scene0000_00/causal_unit_effects.npz",
        ):
            self.assertTrue(is_ignored(path), path)
        for path in (
            "configs/scannet50_local_global_split.json",
            "doc/2026-07-30_Camera_Refiner_Data_Construction_Design.md",
        ):
            self.assertFalse(is_ignored(path), path)

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
