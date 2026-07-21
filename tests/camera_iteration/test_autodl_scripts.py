from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[2]
AUTODL = ROOT / "scripts" / "autodl"


class AutoDLScriptsTest(unittest.TestCase):
    def read(self, name: str) -> str:
        return (AUTODL / name).read_text(encoding="utf-8")

    def test_environment_setup_preserves_base_torch(self):
        content = self.read("setup_vggt_env.sh")
        for value in ("CONDA_ENV_NAME=\"${CONDA_ENV_NAME:-vggt}\"", "--clone", "--no-deps", "--no-build-isolation", "--print-missing", "torch.cuda.is_available"):
            self.assertIn(value, content)
        self.assertNotIn("pip install torch", content.lower())

    def test_weight_setup_is_independent_and_resumable(self):
        content = self.read("download_vggt_weights.sh")
        for value in ("facebook/VGGT-1B", "https://hf-mirror.com", "snapshot_download", "max_workers=1", "model.safetensors", "model.pt", "HF_MAX_RETRIES"):
            self.assertIn(value, content)
        for forbidden in ("conda create", "scannet", "run_camera_iteration"):
            self.assertNotIn(forbidden, content.lower())

    def test_scannet_setup_requires_tos_and_only_requests_sens(self):
        content = self.read("prepare_scannet_camera_iteration.sh")
        for value in ("SCANNET_TOS_ACCEPTED", "http://kaldir.vc.in.tum.de/scannet/download-scannet.py", "camera_iteration_scannet.txt", "--type .sens", "extract_scannet_sens.py", "missing_processed_scenes"):
            self.assertIn(value, content)
        for forbidden in ("export_depth", "ply", "download_vggt_weights", "conda create"):
            self.assertNotIn(forbidden, content.lower())
        self.assertLess(content.index("SCANNET_TOS_ACCEPTED"), content.index("http://kaldir"))

    def test_runner_only_validates_and_executes(self):
        content = self.read("run_camera_iteration.sh")
        for value in ("CONDA_ENV_NAME=\"${CONDA_ENV_NAME:-vggt}\"", "preflight.py", "run_study", "setup_vggt_env.sh", "prepare_scannet_camera_iteration.sh", "25 50 100 200 500", "1 2 4 8 16"):
            self.assertIn(value, content)
        for forbidden in ("conda create", "pip install", "snapshot_download", "extract_scannet", "RUN_EXTRACT", "wget", "curl"):
            self.assertNotIn(forbidden, content)

    def test_context_runner_fixes_round_1_5_protocol(self):
        content = self.read("run_camera_context.sh")
        for value in (
            "camera_context_scannet.txt",
            'FRAME_COUNTS="${FRAME_COUNTS:-25 50 100 200 500}"',
            'ITERATIONS="4"',
            'SAVE_CONTEXT_DIAGNOSTICS="1"',
            "run_camera_iteration.sh",
            "pre_experiments.camera_context.analyze",
        ):
            self.assertIn(value, content)
        self.assertNotIn("SAVE_CAMERA_TOKENS=1", content)

    def test_camera_head_amplification_runner_is_replay_only(self):
        content = self.read("run_camera_head_amplification.sh")
        for value in (
            "camera_head_amplification.run_replay",
            "results/camera_context/911b598_f4577f584448",
            'SHORT_FRAMES="${SHORT_FRAMES:-200}"',
            'LONG_FRAMES="${LONG_FRAMES:-500}"',
            'ITERATIONS="${ITERATIONS:-4}"',
            "context_diagnostics.npz",
            "model.safetensors",
        ):
            self.assertIn(value, content)
        for forbidden in (
            "run_camera_iteration.sh",
            "load_and_preprocess_images",
            "prepare_scannet",
            "pip install",
            "conda create",
        ):
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
