from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[2]
AUTODL = ROOT / "scripts" / "autodl" / "camera_refiner_data_construction"
DOWNLOADER = AUTODL / "download_co3d_2050.sh"
CATEGORIES = ROOT / "configs" / "co3d_train41.txt"


class AutoDLEntryPointTest(unittest.TestCase):
    def test_downloader_reuses_vggt_environment_and_targets_autodl_tmp(self):
        content = DOWNLOADER.read_text(encoding="utf-8")
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
        for forbidden in (
            "conda create",
            "pip install",
            "git clone",
            "snapshot_download",
            "download_vggt",
        ):
            self.assertNotIn(forbidden, content.lower())

    def test_category_config_is_the_41_category_training_split(self):
        categories = [
            line.strip()
            for line in CATEGORIES.read_text(encoding="utf-8").splitlines()
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

    def test_downloader_has_valid_bash_syntax(self):
        bash = shutil.which("bash")
        if bash is None:
            self.skipTest("bash is unavailable")
        result = subprocess.run(
            [bash, "-n", str(DOWNLOADER)],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_data_branch_has_no_multiscale_or_scannet_entry_points(self):
        forbidden = (
            AUTODL / "run_multiscale_study.sh",
            AUTODL / "validate_dataset.py",
            ROOT / "pre_experiments" / "local_global_consistency",
            ROOT / "pre_experiments" / "camera_hidden_state_attribution",
            ROOT / "configs" / "scannet50_local_global_split.json",
        )
        self.assertFalse([str(path) for path in forbidden if path.exists()])


if __name__ == "__main__":
    unittest.main()
