from pathlib import Path
import tempfile
import unittest

import numpy as np

from pre_experiments.common.scannet import get_sorted_image_paths, load_poses


class ScanNetInputTest(unittest.TestCase):
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

if __name__ == "__main__":
    unittest.main()
