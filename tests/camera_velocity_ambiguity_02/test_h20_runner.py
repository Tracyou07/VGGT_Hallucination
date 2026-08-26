from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]


class H20RunnerTest(unittest.TestCase):
    def test_runner_is_fail_closed_smoke_then_exact_calibration(self) -> None:
        path = ROOT / "scripts" / "h20" / "run_camera_velocity_ambiguity_02.sh"
        text = path.read_text(encoding="utf-8")
        for required in (
            "set -euo pipefail",
            "REPO_ROOT=\"${REPO_ROOT:-/home/ubuntu/yjh/vggt/.worktrees/camera_velocity_ambiguity_02_pre_experiment}\"",
            "DATA_ROOT=\"${DATA_ROOT:-/data/yjh/share/datasets/ScanNet}\"",
            "CKPT_DIR=\"${CKPT_DIR:-/data/yjh/share/pretrained/VGGT-1B}\"",
            "RESULT_ROOT=\"${RESULT_ROOT:-/data/output/camera_velocity_ambiguity}\"",
            "CONDA_ENV=\"${CONDA_ENV:-/home/ubuntu/anaconda3/envs/vggt-gx}\"",
            "DEVICE=\"${DEVICE:-cuda}\"",
            "SMOKE_SCENE_LIMIT=\"1\"",
            "CALIBRATION_SCENE_LIMIT=\"10\"",
            "verified_completion.json",
            "smoke_complete.json",
            "calibration_complete.json",
            "--stage smoke",
            "--stage calibration",
            "--run-id \"$RUN_ID\"",
            "CUDA_VISIBLE_DEVICES",
            "nvidia-smi",
            "git branch --show-current",
            "codex/camera_velocity_ambiguity_02_pre_experiment",
        ):
            with self.subTest(required=required):
                self.assertIn(required, text)
        smoke = text.index("--stage smoke")
        gate = text.index("smoke_complete.json")
        calibration = text.index("--stage calibration")
        self.assertLess(smoke, gate)
        self.assertLess(gate, calibration)
        for forbidden in ("conda create", "pip install", "wget ", "huggingface", "--device cpu"):
            self.assertNotIn(forbidden, text)


if __name__ == "__main__":
    unittest.main()
