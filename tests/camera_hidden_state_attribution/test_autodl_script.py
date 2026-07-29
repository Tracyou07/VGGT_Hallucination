from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]


class AutoDLScriptTest(unittest.TestCase):
    def test_entrypoint_has_ordered_stages_and_required_inputs(self):
        path = ROOT / "scripts" / "autodl" / "run_camera_hidden_state_attribution.sh"
        text = path.read_text(encoding="utf-8")
        self.assertIn("set -euo pipefail", text)
        for name in (
            "SOURCE_RUN_DIR",
            "CALIBRATION_LOCAL_RUN_DIR",
            "HOLDOUT_LOCAL_RUN_DIR",
            "SPLIT_MANIFEST",
            "CKPT_DIR",
        ):
            self.assertIn(name, text)
        self.assertLess(text.index("run_smoke"), text.index("run_calibration"))
        self.assertLess(text.index("run_calibration"), text.index("run_holdout"))
        self.assertLess(text.index("run_holdout"), text.index("run_export"))
        self.assertIn('STAGE="${STAGE:-all}"', text)


if __name__ == "__main__":
    unittest.main()
