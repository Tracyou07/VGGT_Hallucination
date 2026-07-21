from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts.autodl.camera_iteration.preflight import (
    detect_scannet_layout,
    find_checkpoint,
    missing_package_specs,
    processed_scene_is_complete,
)
from scripts.autodl.camera_iteration.extract_scannet_sens import (
    extract_scene,
    find_sens_file,
    scene_is_complete,
)


ROOT = Path(__file__).resolve().parents[2]


def write_processed_frame(root: Path, scene: str, frame_id: int = 0) -> None:
    scene_dir = root / "process_scannet" / scene
    (scene_dir / "color").mkdir(parents=True)
    (scene_dir / "pose").mkdir()
    (scene_dir / "color" / f"{frame_id}.jpg").write_bytes(b"image")
    pose = "\n".join(
        " ".join("1" if row == col else "0" for col in range(4))
        for row in range(4)
    )
    (scene_dir / "pose" / f"{frame_id}.txt").write_text(pose, encoding="utf-8")


class AutoDLPreflightTest(unittest.TestCase):
    def test_checkpoint_prefers_nonempty_safetensors_and_accepts_model_pt(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = Path(tmp)
            model_pt = checkpoint_dir / "model.pt"
            model_pt.write_bytes(b"pt")
            self.assertEqual(find_checkpoint(checkpoint_dir), model_pt)

            safetensors = checkpoint_dir / "model.safetensors"
            safetensors.write_bytes(b"safe")
            self.assertEqual(find_checkpoint(checkpoint_dir), safetensors)

            safetensors.write_bytes(b"")
            self.assertEqual(find_checkpoint(checkpoint_dir), model_pt)

    def test_empty_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = Path(tmp)
            (checkpoint_dir / "model.safetensors").touch()
            with self.assertRaisesRegex(FileNotFoundError, "non-empty"):
                find_checkpoint(checkpoint_dir)

    def test_missing_checkpoint_error_names_ckpt_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(FileNotFoundError, "CKPT_DIR"):
                find_checkpoint(Path(tmp))

    def test_processed_layout_is_preferred_for_configured_scene(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_processed_frame(root, "scene0000_00")
            raw_scene = root / "raw_sens" / "scans" / "scene0000_00"
            raw_scene.mkdir(parents=True)
            (raw_scene / "scene0000_00.sens").write_bytes(b"sens")

            self.assertEqual(
                detect_scannet_layout(root, ["scene0000_00"]),
                "processed",
            )

    def test_processed_layout_requires_every_configured_scene(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_processed_frame(root, "scene0000_00")
            with self.assertRaisesRegex(FileNotFoundError, "scene0013_02"):
                detect_scannet_layout(root, ["scene0000_00", "scene0013_02"])

    def test_processed_scene_rejects_mismatched_and_nonfinite_pose(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene = Path(tmp)
            (scene / "color").mkdir()
            (scene / "pose").mkdir()
            (scene / "color" / "0.jpg").write_bytes(b"image")
            (scene / "pose" / "1.txt").write_text(" ".join(["0"] * 16))
            self.assertFalse(processed_scene_is_complete(scene))
            (scene / "pose" / "0.txt").write_text(" ".join(["nan"] * 16))
            self.assertFalse(processed_scene_is_complete(scene))

    def test_raw_layout_is_detected_recursively(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_scene = root / "raw_sens" / "scans" / "scene0013_02"
            raw_scene.mkdir(parents=True)
            (raw_scene / "scene0013_02.sens").write_bytes(b"sens")

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
                (output / "0.jpg").write_bytes(b"image")

            def export_poses(self, output_dir):
                output = Path(output_dir)
                output.mkdir(parents=True, exist_ok=True)
                (output / "0.txt").write_text(" ".join(["0"] * 16), encoding="utf-8")

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
