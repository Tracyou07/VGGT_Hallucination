from pathlib import Path
import tempfile
import unittest

import numpy as np

from pre_experiments.camera_iteration.scannet import (
    get_sorted_image_paths,
    load_poses,
    make_frame_selections,
    read_scene_list,
)


class ScanNetInputTest(unittest.TestCase):
    def test_scene_list_ignores_comments_and_applies_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene_list = Path(tmp) / "scenes.txt"
            scene_list.write_text(
                "# camera study\nscene0000_00\n\nscene0013_02\n",
                encoding="utf-8",
            )

            self.assertEqual(read_scene_list(scene_list, limit=1), ["scene0000_00"])
            self.assertEqual(
                read_scene_list(scene_list, limit=0),
                ["scene0000_00", "scene0013_02"],
            )

    def test_images_are_sorted_by_numeric_frame_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            color_dir = Path(tmp)
            for name in ("10.jpg", "2.png", "1.jpeg"):
                (color_dir / name).touch()

            self.assertEqual(
                [path.stem for path in get_sorted_image_paths(color_dir)],
                ["1", "2", "10"],
            )

    def test_pose_loader_skips_malformed_and_nonfinite_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            pose_dir = Path(tmp)
            np.savetxt(pose_dir / "1.txt", np.eye(4))
            np.savetxt(pose_dir / "2.txt", np.full((4, 4), np.nan))
            np.savetxt(pose_dir / "3.txt", np.eye(3))

            poses = load_poses(pose_dir)

            self.assertEqual(list(poses), [1])
            np.testing.assert_allclose(poses[1], np.eye(4))

    def test_nested_uniform_selections_share_the_largest_grid(self):
        selections = make_frame_selections(
            valid_ids=list(range(20)),
            frame_counts=[4, 8],
            sampling="nested_uniform",
        )

        self.assertEqual(len(selections[4]), 4)
        self.assertEqual(len(selections[8]), 8)
        self.assertTrue(set(selections[4]).issubset(selections[8]))

    def test_invalid_sampling_inputs_raise(self):
        with self.assertRaisesRegex(ValueError, "frame_counts"):
            make_frame_selections([1, 2], [], "prefix")
        with self.assertRaisesRegex(ValueError, "sampling"):
            make_frame_selections([1, 2], [1], "unknown")


if __name__ == "__main__":
    unittest.main()
