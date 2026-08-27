from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

from pre_experiments.variational_camera_latent.pipeline import (
    load_exact_completion,
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


if __name__ == "__main__":
    unittest.main()
