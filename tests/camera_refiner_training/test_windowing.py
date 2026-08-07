import unittest

import numpy as np

from pre_experiments.camera_refiner_training.windowing import (
    assemble_windows_in_reference_gauge,
    build_sliding_windows,
)


def poses(centers: np.ndarray) -> np.ndarray:
    value = np.tile(np.eye(4), (len(centers), 1, 1))
    value[:, :3, 3] = centers
    return value


class WindowingTest(unittest.TestCase):
    def test_windows_cover_tail_without_gaps(self):
        windows = build_sliding_windows(np.arange(230), length=100, stride=50)

        self.assertEqual([window.start for window in windows], [0, 50, 100, 130])
        self.assertEqual(windows[-1].stop, 230)

    def test_local_windows_are_assembled_in_prediction_reference_gauge(self):
        frame_ids = np.arange(8)
        centers = np.stack(
            [frame_ids, np.sin(frame_ids), np.zeros(8)], axis=1
        ).astype(float)
        reference = poses(centers)
        windows = [
            {"frame_ids": frame_ids[:5], "pred_c2w_raw": poses(centers[:5] * 2 + 3)},
            {"frame_ids": frame_ids[3:], "pred_c2w_raw": poses(centers[3:] * 0.5 - 2)},
        ]

        result = assemble_windows_in_reference_gauge(
            windows, reference_frame_ids=frame_ids, reference_c2w=reference
        )

        np.testing.assert_allclose(result["assembled_c2w"][:, :3, 3], centers, atol=1e-9)
        np.testing.assert_array_equal(
            result["observation_count"], np.asarray([1, 1, 1, 2, 2, 1, 1, 1])
        )


if __name__ == "__main__":
    unittest.main()
