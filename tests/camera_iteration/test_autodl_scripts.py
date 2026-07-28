from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[2]
AUTODL = ROOT / "scripts" / "autodl"


class AutoDLScriptsTest(unittest.TestCase):
    def read(self, name: str) -> str:
        return (AUTODL / name).read_text(encoding="utf-8")

    def test_retired_setup_scripts_are_absent(self):
        self.assertFalse((AUTODL / "setup_vggt_env.sh").exists())
        self.assertFalse((AUTODL / "download_vggt_weights.sh").exists())

    def test_scannet_setup_requires_tos_and_only_requests_sens(self):
        content = self.read("prepare_scannet_camera_iteration.sh")
        for value in ("SCANNET_TOS_ACCEPTED", "http://kaldir.vc.in.tum.de/scannet/download-scannet.py", "camera_iteration_scannet.txt", "--type .sens", "extract_scannet_sens.py", "missing_processed_scenes"):
            self.assertIn(value, content)
        for forbidden in ("export_depth", "ply", "download_vggt_weights", "conda create"):
            self.assertNotIn(forbidden, content.lower())
        self.assertLess(content.index("SCANNET_TOS_ACCEPTED"), content.index("http://kaldir"))

    def test_runner_only_validates_and_executes(self):
        content = self.read("run_camera_iteration.sh")
        for value in ("CONDA_ENV_NAME=\"${CONDA_ENV_NAME:-vggt}\"", "preflight.py", "run_study", "existing conda environment", "CKPT_DIR", "SCANNET_ROOT", "25 50 100 200 500", "1 2 4 8 16"):
            self.assertIn(value, content)
        for forbidden in ("setup_vggt_env", "download_vggt_weights", "conda create", "pip install", "snapshot_download", "extract_scannet", "RUN_EXTRACT", "wget", "curl"):
            self.assertNotIn(forbidden, content)

    def test_shell_syntax(self):
        for path in AUTODL.glob("*.sh"):
            subprocess.run(
                ["bash", "-n"],
                input=path.read_text(encoding="utf-8").replace("\r", "").encode(),
                check=True,
            )


if __name__ == "__main__":
    unittest.main()
