import tempfile
from pathlib import Path
import unittest

import numpy as np

from pre_experiments.camera_hidden_state_attribution.replacement_artifacts import (
    load_replacement_scene,
    save_replacement_scene,
)


class HiddenReplacementArtifactTest(unittest.TestCase):
    def test_round_trip_uses_strict_numeric_members(self):
        result = {
            "condition_names": np.array(
                ["baseline", "selected", "control_00"]
            ),
            "replacement_count": np.array([0, 2, 2]),
            "frame_ids": np.arange(4),
            "selected_window_index": np.array([0, 0, 1, 1]),
            "selected_boundary_distance": np.array([0, 1, 1, 0]),
            "local_observation_count": np.array([1, 2, 2, 1]),
            "pred_c2w_raw": np.tile(np.eye(4), (3, 4, 1, 1)),
            "pose_enc": np.zeros((3, 4, 9)),
            "translation_error_aligned": np.zeros((3, 4)),
            "rotation_error_deg_aligned": np.zeros((3, 4)),
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "replacement.npz"
            save_replacement_scene(path, result)
            loaded = load_replacement_scene(path, "scene")
        self.assertEqual(loaded["scene"], "scene")
        np.testing.assert_array_equal(
            loaded["condition_names"],
            result["condition_names"],
        )
        np.testing.assert_allclose(
            loaded["pred_c2w_raw"],
            result["pred_c2w_raw"],
        )

    def test_rejects_nonfinite_or_wrong_members(self):
        result = {
            "condition_names": np.array(["baseline", "selected"]),
            "replacement_count": np.array([0, 1]),
            "frame_ids": np.arange(2),
            "selected_window_index": np.array([0, 0]),
            "selected_boundary_distance": np.array([0, 0]),
            "local_observation_count": np.array([1, 1]),
            "pred_c2w_raw": np.tile(np.eye(4), (2, 2, 1, 1)),
            "pose_enc": np.zeros((2, 2, 9)),
            "translation_error_aligned": np.zeros((2, 2)),
            "rotation_error_deg_aligned": np.zeros((2, 2)),
        }
        result["pose_enc"][0, 0, 0] = np.nan
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "replacement.npz"
            with self.assertRaisesRegex(ValueError, "finite"):
                save_replacement_scene(path, result)


if __name__ == "__main__":
    unittest.main()
