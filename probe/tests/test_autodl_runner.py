from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts" / "autodl" / "run_scannet_hallucination.sh"


class AutoDLRunnerTest(unittest.TestCase):
    def test_runner_uses_existing_environment_and_checkpoint(self):
        content = RUNNER.read_text(encoding="utf-8")
        for expected in (
            'CONDA_ENV_NAME="${CONDA_ENV_NAME:-vggt}"',
            "conda activate",
            "check_runtime_deps.py",
            "model.safetensors",
            "experiments.scannet_hallucination.run_eval",
        ):
            self.assertIn(expected, content)
        for forbidden in (
            "conda create",
            "pip install",
            "pip uninstall",
            "download_vggt_weights.sh",
            "INSTALL_ENV",
            "REPAIR_MISSING_DEPS",
        ):
            self.assertNotIn(forbidden, content)


if __name__ == "__main__":
    unittest.main()
