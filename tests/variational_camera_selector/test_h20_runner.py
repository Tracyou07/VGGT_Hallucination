from __future__ import annotations

from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts" / "h20" / "run_variational_camera_selector.sh"
GIT_BASH = Path(r"C:\Program Files\Git\bin\bash.exe")


class SelectorH20RunnerTests(unittest.TestCase):
    def test_runner_is_one_auto_call_after_h20_preflight(self) -> None:
        text = RUNNER.read_text(encoding="utf-8")
        self.assertIn('[[ "$(hostname)" == "VM-0-11-ubuntu" ]]', text)
        self.assertIn('[[ "$(id -un)" == "ubuntu" ]]', text)
        self.assertIn("/data/yjh/output/variational_camera_selector", text)
        self.assertIn("/data/yjh/output/variational_camera_latent/vrfm_camera_20260827T044926Z", text)
        self.assertEqual(text.count("--stage auto"), 1)
        self.assertNotIn("matched_random", text)
        self.assertIn("unset HF_TOKEN HUGGING_FACE_HUB_TOKEN", text)
        self.assertIn("CUDA_VISIBLE_DEVICES", text)

    def test_runner_has_valid_bash_syntax(self) -> None:
        if not GIT_BASH.is_file():
            self.skipTest("Git Bash is unavailable")
        result = subprocess.run(
            [str(GIT_BASH), "-n", str(RUNNER)],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
