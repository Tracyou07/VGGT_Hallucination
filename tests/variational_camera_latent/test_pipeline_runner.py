from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

from pre_experiments.variational_camera_latent.pipeline import (
    load_exact_completion,
    parse_args,
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


if __name__ == "__main__":
    unittest.main()
