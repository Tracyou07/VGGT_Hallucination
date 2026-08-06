import tempfile
import unittest
from pathlib import Path

import numpy as np

from pre_experiments.camera_refiner_data_construction.artifacts import (
    load_scene_shard,
    save_scene_shard,
)


def _payload() -> dict[str, object]:
    frame_count = 5
    candidate_count = 3
    iterations = 2
    hidden_dim = 4
    identities = np.tile(np.eye(4), (candidate_count, frame_count, 1, 1))
    gt = np.tile(np.eye(4), (frame_count, 1, 1))
    return {
        "scene": "scene0000_00",
        "frame_ids": np.arange(frame_count),
        "scales": np.array([100, 200, 300]),
        "candidate_names": np.array(
            ["baseline", "a0p02_b1_0_0", "a0p05_b0p2_0p3_0p5"]
        ),
        "candidate_alpha": np.array([0.0, 0.02, 0.05]),
        "candidate_beta": np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.2, 0.3, 0.5]]
        ),
        "global_hidden": np.zeros(
            (iterations, frame_count, hidden_dim), dtype=np.float32
        ),
        "local_hidden": np.ones(
            (3, iterations, frame_count, hidden_dim), dtype=np.float32
        ),
        "selected_window_index": np.zeros((3, frame_count), dtype=np.int64),
        "selected_boundary_distance": np.ones(
            (3, frame_count), dtype=np.int64
        ),
        "selected_window_start": np.zeros((3, frame_count), dtype=np.int64),
        "selected_window_stop": np.array(
            [[100] * frame_count, [200] * frame_count, [300] * frame_count]
        ),
        "local_observation_count": np.ones(
            (3, frame_count), dtype=np.int64
        ),
        "pred_c2w_raw": identities,
        "pose_enc": np.zeros((candidate_count, frame_count, 9)),
        "gt_c2w_raw": gt,
        "translation_error_aligned": np.zeros(
            (candidate_count, frame_count)
        ),
        "rotation_error_deg_aligned": np.zeros(
            (candidate_count, frame_count)
        ),
        "hidden_displacement_rms": np.array([0.0, 0.02, 0.05]),
        "camera_center_displacement_mean": np.array([0.0, 0.01, 0.02]),
        "rotation_change_deg_mean": np.array([0.0, 0.01, 0.02]),
        "fov_change_mean": np.array([0.0, 0.001, 0.002]),
    }


class SceneShardArtifactTest(unittest.TestCase):
    def test_round_trip_preserves_strict_multiscale_schema(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "scene_shard.npz"
            save_scene_shard(path, _payload())
            loaded = load_scene_shard(path, "scene0000_00")

        self.assertEqual(loaded["scene"], "scene0000_00")
        self.assertEqual(loaded["global_hidden"].shape, (2, 5, 4))
        self.assertEqual(loaded["local_hidden"].shape, (3, 2, 5, 4))
        np.testing.assert_array_equal(loaded["scales"], [100, 200, 300])
        np.testing.assert_array_equal(
            loaded["candidate_names"],
            ["baseline", "a0p02_b1_0_0", "a0p05_b0p2_0p3_0p5"],
        )

    def test_rejects_nonfinite_hidden_and_invalid_candidate_simplex(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "scene_shard.npz"
            payload = _payload()
            payload["local_hidden"][0, 0, 0, 0] = np.nan
            with self.assertRaisesRegex(ValueError, "finite"):
                save_scene_shard(path, payload)

            payload = _payload()
            payload["candidate_beta"][2] = [0.2, 0.2, 0.2]
            with self.assertRaisesRegex(ValueError, "simplex"):
                save_scene_shard(path, payload)

    def test_rejects_tampered_members_and_scene_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "scene_shard.npz"
            save_scene_shard(path, _payload())
            with np.load(path, allow_pickle=False) as archive:
                arrays = {name: np.asarray(archive[name]) for name in archive.files}
            arrays["aligned_gt"] = np.zeros((5, 4, 4))
            np.savez_compressed(path, **arrays)

            with self.assertRaisesRegex(ValueError, "members"):
                load_scene_shard(path, "scene0000_00")

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "scene_shard.npz"
            save_scene_shard(path, _payload())
            with self.assertRaisesRegex(ValueError, "scene identity"):
                load_scene_shard(path, "scene9999_00")


if __name__ == "__main__":
    unittest.main()
