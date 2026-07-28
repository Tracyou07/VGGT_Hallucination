from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[2]
AUTODL = ROOT / "scripts" / "autodl"


class AutoDLScriptsTest(unittest.TestCase):
    def test_only_camera_head_runner_remains(self):
        scripts = sorted(path.name for path in AUTODL.glob("*.sh"))
        self.assertEqual(scripts, ["run_camera_head_amplification.sh"])

    def test_camera_head_runner_is_replay_only(self):
        path = AUTODL / "run_camera_head_amplification.sh"
        content = path.read_text(encoding="utf-8")
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
        path = AUTODL / "run_camera_head_amplification.sh"
        subprocess.run(
            ["bash", "-n"],
            input=path.read_text(encoding="utf-8").replace("\r", "").encode(),
            check=True,
        )


if __name__ == "__main__":
    unittest.main()
