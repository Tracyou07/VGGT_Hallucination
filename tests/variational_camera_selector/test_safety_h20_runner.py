from __future__ import annotations

from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts" / "h20" / "run_variational_camera_selector_safety.sh"
GIT_BASH = Path(r"C:\Program Files\Git\bin\bash.exe")


class SafetyH20RunnerTests(unittest.TestCase):
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

    def test_runner_fails_closed_before_compute_off_h20(self) -> None:
        # Catches accidentally launching eight folds on the local Windows machine.
        if not GIT_BASH.is_file():
            self.skipTest("Git Bash is unavailable")
        result = subprocess.run(
            [str(GIT_BASH), str(RUNNER)],
            text=True,
            capture_output=True,
            check=False,
            timeout=20,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("VM-0-11-ubuntu", result.stderr)


if __name__ == "__main__":
    unittest.main()
