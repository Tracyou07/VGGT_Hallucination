from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
AUTODL = ROOT / "scripts" / "autodl" / "camera_refiner_data_construction"
RUNNER = AUTODL / "run_multiscale_study.sh"
VALIDATOR = AUTODL / "validate_dataset.py"


class AutoDLEntryPointTest(unittest.TestCase):
    def test_runner_is_strict_resumable_and_uses_only_frozen_inputs(self):
        content = RUNNER.read_text(encoding="utf-8")
        for required in (
            "set -euo pipefail",
            '${SOURCE_RUN_DIR:?Set SOURCE_RUN_DIR',
            '${FROZEN_UNITS:?Set FROZEN_UNITS',
            'SCALE_PAIRS=("100:50" "200:100" "300:150")',
            "pre_experiments.local_global_consistency.run_study",
            "pre_experiments.camera_refiner_data_construction.run_study",
            'partition="calibration"',
            'partition="holdout"',
            '--run-dir-file "$pointer"',
            'scale_args+=(--scale-run "$scale=$scale_run_dir")',
            '--frozen-policy "$FROZEN_POLICY"',
        ):
            self.assertIn(required, content)
        for forbidden in (
            "conda create",
            "pip install",
            "git clone",
            "snapshot_download",
            "huggingface",
            "prepare_scannet50",
            "download_vggt",
        ):
            self.assertNotIn(forbidden, content.lower())

    def test_runner_has_valid_bash_syntax(self):
        bash = shutil.which("bash")
        if bash is None:
            self.skipTest("bash is unavailable")
        result = subprocess.run(
            [bash, "-n", str(RUNNER)],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_dataset_validator_is_cpu_only_and_has_a_standalone_cli(self):
        content = VALIDATOR.read_text(encoding="utf-8")
        self.assertIn("validate_dataset_manifest", content)
        self.assertIn('split["holdout_scenes"]', content)
        self.assertNotIn("import torch", content)
        result = subprocess.run(
            [sys.executable, str(VALIDATOR), "--help"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--manifest", result.stdout)
        self.assertIn("--dataset-root", result.stdout)
        self.assertIn("--split-manifest", result.stdout)


if __name__ == "__main__":
    unittest.main()
