import unittest

import numpy as np


def _poses(centers: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    poses = np.tile(np.eye(4), (len(centers), 1, 1))
    cosine = np.cos(yaw)
    sine = np.sin(yaw)
    poses[:, 0, 0] = cosine
    poses[:, 0, 1] = -sine
    poses[:, 1, 0] = sine
    poses[:, 1, 1] = cosine
    poses[:, :3, 3] = centers
    return poses


def _inverse_sim3(
    canonical: np.ndarray,
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    local = canonical.copy()
    local[:, :3, :3] = np.einsum(
        "ij,sjk->sik", rotation.T, canonical[:, :3, :3]
    )
    local[:, :3, 3] = (
        (canonical[:, :3, 3] - translation) @ rotation
    ) / scale
    return local


class StitchLocalWindowsTest(unittest.TestCase):
    def test_recovers_one_prediction_gauge_from_overlapping_windows(self):
        from pre_experiments.local_global_consistency.stitch import (
            stitch_local_windows,
        )

        frame_ids = np.arange(10)
        centers = np.column_stack(
            [0.3 * frame_ids, np.sin(frame_ids * 0.4), 0.1 * frame_ids**1.2]
        )
        canonical = _poses(centers, yaw=frame_ids * 0.07)
        angle = 0.4
        rotation = np.asarray(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        windows = [
            {
                "frame_ids": frame_ids[:6],
                "pred_c2w_raw": canonical[:6],
            },
            {
                "frame_ids": frame_ids[3:9],
                "pred_c2w_raw": _inverse_sim3(
                    canonical[3:9],
                    1.7,
                    rotation,
                    np.asarray([2.0, -1.0, 0.4]),
                ),
            },
            {
                "frame_ids": frame_ids[6:],
                "pred_c2w_raw": _inverse_sim3(
                    canonical[6:],
                    0.8,
                    rotation.T,
                    np.asarray([-0.5, 1.2, -0.3]),
                ),
            },
        ]

        result = stitch_local_windows(windows)

        np.testing.assert_array_equal(result["frame_ids"], frame_ids)
        np.testing.assert_allclose(
            result["stitched_c2w"][:, :3, 3],
            canonical[:, :3, 3],
            atol=1e-9,
        )
        np.testing.assert_allclose(
            result["stitched_c2w"][:, :3, :3],
            canonical[:, :3, :3],
            atol=1e-9,
        )
        self.assertEqual(result["overlap_counts"], [3, 3])

    def test_rejects_a_window_without_prediction_overlap(self):
        from pre_experiments.local_global_consistency.stitch import (
            stitch_local_windows,
        )

        poses = _poses(np.column_stack([np.arange(6), np.zeros((6, 2))]), np.zeros(6))
        windows = [
            {"frame_ids": np.arange(3), "pred_c2w_raw": poses[:3]},
            {"frame_ids": np.arange(3, 6), "pred_c2w_raw": poses[3:]},
        ]

        with self.assertRaisesRegex(ValueError, "at least three overlapping"):
            stitch_local_windows(windows)

    def test_assembles_windows_in_a_prediction_only_reference_gauge(self):
        from pre_experiments.local_global_consistency.stitch import (
            assemble_windows_in_reference_gauge,
        )

        frame_ids = np.arange(8)
        centers = np.column_stack(
            [0.2 * frame_ids, np.sin(frame_ids * 0.5), 0.05 * frame_ids**1.3]
        )
        reference = _poses(centers, yaw=frame_ids * 0.04)
        angle = -0.35
        rotation = np.asarray(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        windows = [
            {
                "frame_ids": frame_ids[:5],
                "pred_c2w_raw": _inverse_sim3(
                    reference[:5],
                    1.4,
                    rotation,
                    np.asarray([0.7, -0.4, 0.2]),
                ),
            },
            {
                "frame_ids": frame_ids[3:],
                "pred_c2w_raw": _inverse_sim3(
                    reference[3:],
                    0.6,
                    rotation.T,
                    np.asarray([-1.0, 0.5, -0.1]),
                ),
            },
        ]

        result = assemble_windows_in_reference_gauge(
            windows,
            reference_frame_ids=frame_ids,
            reference_c2w=reference,
        )

        np.testing.assert_array_equal(result["frame_ids"], frame_ids)
        np.testing.assert_allclose(
            result["assembled_c2w"],
            reference,
            atol=1e-9,
        )
        np.testing.assert_array_equal(
            result["observation_count"],
            np.asarray([1, 1, 1, 2, 2, 1, 1, 1]),
        )

    def test_hybrid_keeps_reference_rotation_and_assembled_translation(self):
        from pre_experiments.local_global_consistency.stitch import (
            build_translation_hybrid,
        )

        frame_ids = np.arange(5)
        reference = _poses(
            np.column_stack([frame_ids, np.zeros((5, 2))]),
            yaw=frame_ids * 0.1,
        )
        assembled = _poses(
            np.column_stack([np.zeros(5), frame_ids * 0.2, frame_ids * 0.3]),
            yaw=-frame_ids * 0.2,
        )

        hybrid = build_translation_hybrid(reference, assembled)

        np.testing.assert_allclose(hybrid[:, :3, :3], reference[:, :3, :3])
        np.testing.assert_allclose(hybrid[:, :3, 3], assembled[:, :3, 3])


if __name__ == "__main__":
    unittest.main()
