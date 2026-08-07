from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
AUTODL = ROOT / "scripts" / "autodl" / "camera_refiner_data_construction"
RUNNER = AUTODL / "run_multiscale_study.sh"
CO3D_DOWNLOADER = AUTODL / "download_co3d_2050.sh"
CO3D_CATEGORIES = ROOT / "configs" / "co3d_train41.txt"
VALIDATOR = AUTODL / "validate_dataset.py"
README = ROOT / "README.md"
REFINER_RUNNER = (
    ROOT
    / "pre_experiments"
    / "camera_refiner_data_construction"
    / "run_study.py"
)
LOCAL_GLOBAL_RUNNER = (
    ROOT / "pre_experiments" / "local_global_consistency" / "run_study.py"
)


class AutoDLEntryPointTest(unittest.TestCase):
    def test_co3d_downloader_reuses_vggt_environment_and_targets_autodl_tmp(self):
        content = CO3D_DOWNLOADER.read_text(encoding="utf-8")
        self.assertIn("set -euo pipefail", content)
        self.assertIn('CONDA_ENV_NAME="${CONDA_ENV_NAME:-vggt}"', content)
        self.assertIn(
            'OUTPUT_ROOT="${OUTPUT_ROOT:-$AUTODL_TMP/datasets/co3dv2_2050}"',
            content,
        )
        self.assertIn(
            "pre_experiments.camera_refiner_data_construction.co3d_download",
            content,
        )
        self.assertIn("SEQUENCES_PER_CATEGORY", content)
        for forbidden in (
            "conda create",
            "pip install",
            "git clone",
            "snapshot_download",
            "download_vggt",
        ):
            self.assertNotIn(forbidden, content.lower())

    def test_co3d_category_config_is_the_41_category_training_split(self):
        categories = [
            line.strip()
            for line in CO3D_CATEGORIES.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        self.assertEqual(len(categories), 41)
        self.assertEqual(len(set(categories)), 41)
        for heldout in (
            "ball",
            "book",
            "couch",
            "frisbee",
            "hotdog",
            "kite",
            "remote",
            "sandwich",
            "skateboard",
            "suitcase",
        ):
            self.assertNotIn(heldout, categories)

    def test_all_output_defaults_use_the_canonical_results_root(self):
        shell = RUNNER.read_text(encoding="utf-8")
        refiner = REFINER_RUNNER.read_text(encoding="utf-8")
        local_global = LOCAL_GLOBAL_RUNNER.read_text(encoding="utf-8")
        readme = README.read_text(encoding="utf-8")
        self.assertIn(
            'RESULTS_ROOT="${RESULTS_ROOT:-$AUTODL_TMP/results}"',
            shell,
        )
        self.assertIn(
            'WORK_ROOT="${WORK_ROOT:-$RESULTS_ROOT/camera_refiner_data_construction}"',
            shell,
        )
        self.assertIn('os.environ.get("RESULTS_ROOT"', refiner)
        self.assertIn('os.environ.get("RESULTS_ROOT"', local_global)
        for legacy in (
            "$AUTODL_TMP/camera_refiner_data_construction",
            'AUTODL_TMP / "camera_refiner_data_construction" / "results"',
            'AUTODL_TMP / "local_global_consistency" / "results"',
            "/root/autodl-tmp/camera_context/results",
            "/root/autodl-tmp/camera_hidden_replacement/state",
        ):
            self.assertNotIn(legacy, "\n".join((shell, refiner, local_global, readme)))

    def test_readme_commands_use_resolvable_paths_not_shell_placeholders(self):
        content = README.read_text(encoding="utf-8")
        self.assertNotIn("results/<run_id>", content)
        self.assertNotIn("<frozen_units.json>", content)
        self.assertIn(
            "camera_context/results/d33d98b_309a9a586242",
            content,
        )
        self.assertIn(
            "camera_hidden_replacement/state/calibration_run.txt",
            content,
        )

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

    def test_co3d_downloader_has_valid_bash_syntax(self):
        bash = shutil.which("bash")
        if bash is None:
            self.skipTest("bash is unavailable")
        result = subprocess.run(
            [bash, "-n", str(CO3D_DOWNLOADER)],
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
