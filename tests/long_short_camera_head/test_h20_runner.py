from __future__ import annotations

from pathlib import Path
import unittest


class H20RunnerContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = Path("scripts/h20/run_long_short_camera_head.sh")

    def test_runner_uses_requested_output_umbrella(self) -> None:
        text = self.runner.read_text(encoding="utf-8")
        self.assertIn("/data/yjh/output/vggt/long_short_camera_head", text)

    def test_runner_requires_clean_branch_h20_and_100_gib(self) -> None:
        text = self.runner.read_text(encoding="utf-8")
        for required in ("VM-0-11-ubuntu", "NVIDIA H20", "status --short", "-ge 100"):
            self.assertIn(required, text)

    def test_runner_never_uses_hugging_face_credentials(self) -> None:
        text = self.runner.read_text(encoding="utf-8").lower()
        for forbidden in ("hf_token", "huggingface_token", "token="):
            self.assertNotIn(forbidden, text)


if __name__ == "__main__":
    unittest.main()
