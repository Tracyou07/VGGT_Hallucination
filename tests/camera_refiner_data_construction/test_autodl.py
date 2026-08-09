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
SCANNET_PREPARE = AUTODL / "prepare_scannet_adaptation200.sh"
SCANNET_WATCHER = AUTODL / "wait_for_co3d_then_prepare_scannet200.sh"
OFFICIAL_SCANNET_TRAIN = ROOT / "configs" / "scannetv2_train_official.txt"
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

    def test_scannet_adaptation_runner_is_authorized_serial_and_space_bounded(self):
        content = SCANNET_PREPARE.read_text(encoding="utf-8")
        for required in (
            "set -euo pipefail",
            'SCANNET_TOS_ACCEPTED:-0',
            'MIN_FREE_GIB="${MIN_FREE_GIB:-60}"',
            'TARGET_SCENES="${TARGET_SCENES:-200}"',
            'REFINER_TRAIN_SCENES="${REFINER_TRAIN_SCENES:-160}"',
            'VALIDATION_SCENES="${VALIDATION_SCENES:-20}"',
            'SELECTOR_TRAIN_SCENES="${SELECTOR_TRAIN_SCENES:-20}"',
            "flock --nonblock",
            "download_asset",
            "extract_scannet_sens.py",
            "processed-scene-frame-count",
            'rm -f -- "$sens"',
            'cleanup_scene_download "$scene"',
            "conda activate \"$CONDA_ENV_NAME\"",
        ):
            self.assertIn(required, content)
        download = content.index('download_asset "$scene" .sens "$sens"')
        extraction = content.index('extract_scannet_sens.py')
        cleanup = content.rindex('cleanup_scene_download "$scene"')
        self.assertLess(download, extraction)
        self.assertLess(extraction, cleanup)
        for forbidden in ("conda create", "pip install", "snapshot_download", "download_vggt"):
            self.assertNotIn(forbidden, content.lower())

    def test_scannet_watcher_requires_successful_co3d_completion_before_switch(self):
        content = SCANNET_WATCHER.read_text(encoding="utf-8")
        for required in (
            "set -euo pipefail",
            'SCANNET_TOS_ACCEPTED:-0',
            "download_manifest.json",
            "co3d_download",
            "git switch 016-camera-refiner-multiscale",
            "prepare_scannet_adaptation200.sh",
            'SCANNET_TOS_ACCEPTED="$SCANNET_TOS_ACCEPTED"',
        ):
            self.assertIn(required, content)
        self.assertNotIn("exec env SCANNET_TOS_ACCEPTED=1", content)
        self.assertLess(content.index("download_manifest.json"), content.index("git switch"))
        self.assertLess(content.index("git switch"), content.index("prepare_scannet_adaptation200.sh"))

    def test_scannet_adaptation_scripts_have_valid_bash_syntax(self):
        bash = shutil.which("bash")
        if bash is None:
            self.skipTest("bash is unavailable")
        for script in (SCANNET_PREPARE, SCANNET_WATCHER):
            result = subprocess.run(
                [bash, "-n", str(script)],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, f"{script}: {result.stderr}")

    def test_vendored_official_scannet_train_split_is_complete(self):
        scenes = [
            line.strip()
            for line in OFFICIAL_SCANNET_TRAIN.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(len(scenes), 1201)
        self.assertEqual(len(set(scenes)), 1201)


if __name__ == "__main__":
    unittest.main()
