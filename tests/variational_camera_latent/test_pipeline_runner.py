from __future__ import annotations

import inspect
import tempfile
from pathlib import Path
import subprocess
import unittest

import numpy as np

from pre_experiments.variational_camera_latent import pipeline as pipeline_module
from pre_experiments.variational_camera_latent.pipeline import (
    load_exact_completion,
    parse_args,
    run_stage,
    write_completion,
)


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts" / "h20" / "run_variational_camera_latent.sh"


class PipelineRunnerTests(unittest.TestCase):
    def test_runner_uses_h20_and_new_output_root(self) -> None:
        text = RUNNER.read_text(encoding="utf-8")
        self.assertIn(
            'RESULT_ROOT="${RESULT_ROOT:-/data/yjh/output/variational_camera_latent}"', text
        )
        self.assertIn('[[ "$(hostname)" == "VM-0-11-ubuntu" ]]', text)
        self.assertIn('SMOKE_SCENE_LIMIT="1"', text)
        self.assertIn('CALIBRATION_SCENE_LIMIT="10"', text)
        self.assertNotIn("HF_TOKEN", text)
        self.assertIn("--stage verify", text)

    def test_completion_resumes_only_an_exact_payload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "complete.json"
            payload = {"schema": "fixture.v1", "stage": "source", "upstream": "a" * 64}
            written = write_completion(path, payload)

            self.assertEqual(load_exact_completion(path, payload), written)
            self.assertIsNone(
                load_exact_completion(path, {**payload, "upstream": "b" * 64})
            )

    def test_pipeline_exposes_resumable_alpha_scan_stage(self) -> None:
        try:
            args = parse_args(
                [
                    "--stage",
                    "alpha-scan",
                    "--run-root",
                    "/data/yjh/output/variational_camera_latent/fixture",
                ]
            )
        except SystemExit:
            self.fail("pipeline does not expose the alpha-scan stage")

        self.assertEqual(args.stage, "alpha-scan")
        self.assertEqual(args.alpha_min_improvement, 0.01)

    def test_pipeline_exposes_resumable_vrfm_residual_alpha_scan_stage(self) -> None:
        try:
            args = parse_args(
                [
                    "--stage",
                    "vrfm-residual-alpha-scan",
                    "--run-root",
                    "/data/yjh/output/variational_camera_latent/fixture",
                ]
            )
        except SystemExit:
            self.fail("pipeline does not expose the VRFM residual alpha scan stage")

        self.assertEqual(args.stage, "vrfm-residual-alpha-scan")
        self.assertEqual(args.residual_scan_batch_size, 8)

    def test_pipeline_exposes_resumable_matched_random_ablation_stage(self) -> None:
        try:
            args = parse_args(
                [
                    "--stage",
                    "matched-random-ablation",
                    "--run-root",
                    "/data/yjh/output/variational_camera_latent/fixture",
                ]
            )
        except SystemExit:
            self.fail("pipeline does not expose the matched random ablation stage")

        self.assertEqual(args.stage, "matched-random-ablation")
        self.assertEqual(args.matched_random_batch_size, 8)
        self.assertEqual(args.matched_random_seed, 20260827)

    def test_matched_random_stage_dispatches_to_its_fail_closed_runner(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = parse_args(
                [
                    "--stage",
                    "matched-random-ablation",
                    "--run-root",
                    directory,
                    "--scene-limit",
                    "1",
                ]
            )

            with self.assertRaisesRegex(
                ValueError, "matched random ablation requires exactly ten scenes"
            ):
                run_stage(args)

    def test_matched_random_budget_rejects_shards_without_exactly_32_samples(self) -> None:
        candidate = {"z": np.zeros((8, 31, 16), dtype=np.float32)}
        prediction = {"z": np.zeros((8, 31, 16), dtype=np.float32)}
        privileged = {"candidate_rms": np.zeros((8, 31, 8), dtype=np.float64)}

        with self.assertRaisesRegex(
            ValueError, "exactly 32 samples per overlap"
        ):
            pipeline_module._require_matched_random_sample_budget(
                candidate,
                prediction,
                privileged,
            )

    def test_provenance_rejects_a_dirty_git_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            tracked = root / "tracked.txt"
            tracked.write_text("clean\n", encoding="utf-8")
            subprocess.run(["git", "add", "tracked.txt"], cwd=root, check=True)
            subprocess.run(
                [
                    "git",
                    "-c",
                    "user.name=Codex Test",
                    "-c",
                    "user.email=codex@example.invalid",
                    "commit",
                    "-qm",
                    "fixture",
                ],
                cwd=root,
                check=True,
            )

            commit = pipeline_module._require_clean_git_checkout(root)
            self.assertEqual(len(commit), 40)
            tracked.write_text("dirty\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "clean git checkout"):
                pipeline_module._require_clean_git_checkout(root)

    def test_control_identity_can_only_depend_on_prediction_only_manifests(self) -> None:
        parameters = inspect.signature(
            pipeline_module._matched_random_transform_identity
        ).parameters
        self.assertEqual(
            set(parameters),
            {
                "source_manifest_sha256",
                "candidate_manifest_sha256",
                "vrfm_prediction_manifest_sha256",
            },
        )
        self.assertFalse(
            any(
                forbidden in name
                for name in parameters
                for forbidden in ("privileged", "report", "completion", "gt")
            )
        )

    def test_prediction_manifest_is_sealed_before_privileged_artifacts_are_loaded(self) -> None:
        source = inspect.getsource(pipeline_module.run_matched_random_ablation)
        seal = source.index("_atomic_json(prediction_manifest_path")
        privileged_load = source.index("load_vrfm_residual_privileged")
        self.assertGreater(privileged_load, seal)


if __name__ == "__main__":
    unittest.main()
