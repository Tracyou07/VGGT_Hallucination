from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts.autodl.camera_iteration.preflight import (
    detect_scannet_layout,
    find_checkpoint,
    missing_package_specs,
)
from scripts.autodl.camera_iteration.extract_scannet_sens import (
    extract_scene,
    find_sens_file,
    scene_is_complete,
)


ROOT = Path(__file__).resolve().parents[2]


class AutoDLPreflightTest(unittest.TestCase):
    def test_checkpoint_prefers_safetensors_and_accepts_model_pt(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = Path(tmp)
            model_pt = checkpoint_dir / "model.pt"
            model_pt.touch()
            self.assertEqual(find_checkpoint(checkpoint_dir), model_pt)

            safetensors = checkpoint_dir / "model.safetensors"
            safetensors.touch()
            self.assertEqual(find_checkpoint(checkpoint_dir), safetensors)

    def test_missing_checkpoint_error_names_ckpt_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(FileNotFoundError, "CKPT_DIR"):
                find_checkpoint(Path(tmp))

    def test_processed_layout_is_preferred_for_configured_scene(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scene = root / "process_scannet" / "scene0000_00"
            (scene / "color").mkdir(parents=True)
            (scene / "pose").mkdir()
            raw_scene = root / "raw_sens" / "scans" / "scene0000_00"
            raw_scene.mkdir(parents=True)
            (raw_scene / "scene0000_00.sens").touch()

            self.assertEqual(
                detect_scannet_layout(root, ["scene0000_00"]),
                "processed",
            )

    def test_raw_layout_is_detected_recursively(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_scene = root / "raw_sens" / "scans" / "scene0013_02"
            raw_scene.mkdir(parents=True)
            (raw_scene / "scene0013_02.sens").touch()

            self.assertEqual(
                detect_scannet_layout(root, ["scene0013_02"]),
                "raw",
            )

    def test_missing_dataset_error_names_scannet_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(FileNotFoundError, "SCANNET_ROOT"):
                detect_scannet_layout(Path(tmp), ["scene0000_00"])

    def test_missing_packages_return_install_specs(self):
        available = {"numpy", "PIL", "huggingface_hub", "einops", "imageio"}

        with mock.patch(
            "scripts.autodl.camera_iteration.preflight.importlib.util.find_spec",
            side_effect=lambda module: object() if module in available else None,
        ):
            self.assertEqual(
                missing_package_specs(),
                ["safetensors", "opencv-python-headless==4.11.0.86"],
            )

    def test_runner_has_fixed_defaults_and_no_dataset_or_weight_downloads(self):
        runner = ROOT / "scripts" / "autodl" / "run_camera_iteration.sh"
        content = runner.read_text(encoding="utf-8")

        for value in (
            "RUN_EXTRACT",
            "SCANNET_ROOT",
            "CKPT_DIR",
            "RESULT_DIR",
            "25 50 100 200 500",
            "1 2 4 8 16",
            "vggt_camera_iteration",
        ):
            self.assertIn(value, content)
        for forbidden in ("wget", "curl", "huggingface-cli", "download_scannet"):
            self.assertNotIn(forbidden, content.lower())


class SensExtractionTest(unittest.TestCase):
    def test_sens_file_is_found_in_scene_directory_or_raw_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw_dir = Path(tmp)
            nested = raw_dir / "scene0000_00" / "scene0000_00.sens"
            nested.parent.mkdir()
            nested.touch()
            self.assertEqual(find_sens_file(raw_dir, "scene0000_00"), nested)

            flat = raw_dir / "scene0013_02.sens"
            flat.touch()
            self.assertEqual(find_sens_file(raw_dir, "scene0013_02"), flat)
            with self.assertRaisesRegex(FileNotFoundError, "scene9999_00"):
                find_sens_file(raw_dir, "scene9999_00")

    def test_complete_scene_is_skipped_and_partial_scene_is_reextracted(self):
        class FakeSensorData:
            constructions = 0

            def __init__(self, sens_file):
                del sens_file
                type(self).constructions += 1

            def export_color_images(self, output_dir):
                output = Path(output_dir)
                output.mkdir(parents=True, exist_ok=True)
                (output / "0.jpg").touch()

            def export_poses(self, output_dir):
                output = Path(output_dir)
                output.mkdir(parents=True, exist_ok=True)
                (output / "0.txt").write_text("pose", encoding="utf-8")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sens_file = root / "scene.sens"
            sens_file.touch()
            output_dir = root / "processed" / "scene0000_00"
            (output_dir / "color").mkdir(parents=True)
            (output_dir / "color" / "stale.jpg").touch()

            self.assertFalse(scene_is_complete(output_dir))
            self.assertTrue(extract_scene(FakeSensorData, sens_file, output_dir))
            self.assertTrue(scene_is_complete(output_dir))
            self.assertEqual(FakeSensorData.constructions, 1)
            self.assertFalse(extract_scene(FakeSensorData, sens_file, output_dir))
            self.assertEqual(FakeSensorData.constructions, 1)


if __name__ == "__main__":
    unittest.main()
