import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from pre_experiments.camera_iteration.contracts import (
    atomic_write_json,
    build_run_metadata,
    make_run_id,
    read_git_commit,
    validate_output_root,
)


class CameraIterationContractTest(unittest.TestCase):
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
            "results/camera_iteration/run/run_metadata.json",
            "results/camera_iteration/run/summary.csv",
            "results/camera_iteration/run/scene0000_00/frames_25/camera_trace.npz",
            "results/camera_head_amplification/run/amplification_summary.json",
            "results/camera_head_amplification/run/amplification_per_frame.csv",
        ):
            self.assertFalse(is_ignored(path), path)
        for path in (
            "results/unpublished/output.json",
            "results/camera_iteration/run/scene0000_00/frames_25/cloud.ply",
            "results/camera_iteration/run/scene0000_00/frames_25/preview.jpg",
            "results/camera_head_amplification/run/activations.npz",
        ):
            self.assertTrue(is_ignored(path), path)

    def test_output_root_accepts_method_and_external_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            repo_root = workspace / "repo"
            repo_root.mkdir()
            allowed = repo_root / "results" / "pre_experiments" / "camera_iteration"
            external = workspace / "autodl_results"

            self.assertEqual(validate_output_root(allowed, repo_root), allowed.resolve())
            self.assertEqual(validate_output_root(external, repo_root), external.resolve())

    def test_output_root_rejects_other_repository_and_phenomenon_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            repo_root = workspace / "repo"
            repo_root.mkdir()

            with self.assertRaisesRegex(ValueError, "repository output"):
                validate_output_root(repo_root / "results" / "other", repo_root)
            with self.assertRaisesRegex(ValueError, "phenomenon"):
                validate_output_root(workspace / "scannet_hallucination", repo_root)
            with self.assertRaisesRegex(ValueError, "absolute"):
                validate_output_root(Path("..") / "external", repo_root)

    def test_run_id_and_metadata_are_deterministic(self):
        commit = "2b1e6fc3a7e46e8bc4e628c4ce4f8e1a49373032"
        invocation = {
            "scenes": ["scene0000_00"],
            "frame_counts": [25],
            "iterations": [1, 2, 4],
            "sampling": "nested_uniform",
        }

        run_id = make_run_id(commit, invocation)
        self.assertRegex(run_id, r"^2b1e6fc_[0-9a-f]{12}$")
        self.assertEqual(run_id, make_run_id(commit, dict(reversed(list(invocation.items())))))
        self.assertNotEqual(run_id, make_run_id(commit, {**invocation, "iterations": [1, 4]}))

        metadata = build_run_metadata(commit, invocation)
        self.assertEqual(metadata["study_type"], "method_pre_experiment")
        self.assertEqual(metadata["study_name"], "camera_iteration")
        self.assertEqual(metadata["run_id"], run_id)
        self.assertEqual(metadata["invocation"], invocation)
        self.assertIn("aligned", metadata["primary_metric_policy"])

    def test_invalid_commit_and_empty_scenes_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "40-character"):
            make_run_id("not-a-commit", {"scenes": ["scene0000_00"]})
        with self.assertRaisesRegex(ValueError, "at least one scene"):
            build_run_metadata("a" * 40, {"scenes": []})

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
