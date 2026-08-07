from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]


class AutoDLScriptsTest(unittest.TestCase):
    def test_training_and_inference_use_existing_vggt_environment(self):
        for name in ("train.sh", "infer.sh"):
            script = ROOT / "scripts" / "autodl" / "camera_refiner_training" / name
            text = script.read_text(encoding="utf-8")
            self.assertIn("conda activate vggt", text)
            self.assertNotIn("conda create", text)
            self.assertNotIn("pip install", text)
            self.assertNotIn("huggingface", text.lower())

    def test_artifacts_default_to_unified_results_root(self):
        for name in ("train.sh", "infer.sh"):
            text = (
                ROOT / "scripts" / "autodl" / "camera_refiner_training" / name
            ).read_text(encoding="utf-8")
            self.assertIn("/root/autodl-tmp/results/camera_refiner_training", text)


if __name__ == "__main__":
    unittest.main()
