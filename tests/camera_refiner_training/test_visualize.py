from pathlib import Path
import tempfile
import unittest

import numpy as np

from pre_experiments.camera_refiner_training.visualize import (
    write_history_plot,
    write_trajectory_plot,
)


def poses(centers: np.ndarray) -> np.ndarray:
    value = np.tile(np.eye(4), (len(centers), 1, 1))
    value[:, :3, 3] = centers
    return value


class VisualizationTest(unittest.TestCase):
    def test_writes_training_and_trajectory_pngs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            history = root / "history.png"
            trajectory = root / "trajectory.png"
            centers = np.stack(
                [np.arange(8), np.sin(np.arange(8)), np.zeros(8)], axis=1
            ).astype(float)

            write_history_plot(
                history,
                [{"epoch": 1, "train_loss": 0.4, "validation_loss": 0.6}],
            )
            write_trajectory_plot(
                trajectory,
                baseline_c2w=poses(centers + 0.2),
                refined_c2w=poses(centers + 0.1),
                gt_c2w=poses(centers),
            )

            self.assertGreater(history.stat().st_size, 1000)
            self.assertGreater(trajectory.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
