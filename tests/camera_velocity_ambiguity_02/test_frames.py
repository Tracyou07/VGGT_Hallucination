from __future__ import annotations

from pathlib import Path
import unittest

from pre_experiments.camera_velocity_ambiguity_02.frames import (
    build_fastvggt_frame_selection,
    build_protocol_windows,
)


def _images(frame_ids: list[int]) -> list[Path]:
    return [Path("color") / f"{frame_id}.jpg" for frame_id in frame_ids]


class FastVggtFrameSelectionTest(unittest.TestCase):
    def test_intersects_images_with_finite_pose_ids_and_preserves_paths(self) -> None:
        selection = build_fastvggt_frame_selection(
            _images([2, 5, 9, 14, 20]),
            finite_pose_frame_ids=[1, 2, 9, 14, 99],
            input_frames=3,
        )

        self.assertEqual(selection.frame_ids, (2, 9, 14))
        self.assertEqual(
            selection.image_paths,
            tuple(_images([2, 9, 14])),
        )
        self.assertEqual(selection.pose_indices, (1, 2, 3))

    def test_matches_fastvggt_preserve_first_floor_stride_behavior(self) -> None:
        raw_ids = list(range(0, 2000, 2))
        selection = build_fastvggt_frame_selection(
            _images(raw_ids),
            finite_pose_frame_ids=raw_ids,
            input_frames=500,
        )

        expected = (raw_ids[0], *raw_ids[1::2][:499])
        self.assertEqual(selection.frame_ids, expected)
        self.assertEqual(len(selection.frame_ids), 500)
        self.assertTrue(
            all(left < right for left, right in zip(selection.frame_ids, selection.frame_ids[1:]))
        )

    def test_exact_count_is_supported_for_normal_and_exception_scenes(self) -> None:
        ids = list(range(600))
        normal = build_fastvggt_frame_selection(
            _images(ids), ids, input_frames=500
        )
        exceptional = build_fastvggt_frame_selection(
            _images(ids), ids, input_frames=430
        )

        self.assertEqual(len(normal.frame_ids), 500)
        self.assertEqual(len(exceptional.frame_ids), 430)
        self.assertEqual(len(build_protocol_windows(normal)), 9)
        exception_windows = build_protocol_windows(exceptional)
        self.assertEqual(len(exception_windows), 8)
        self.assertEqual((exception_windows[-1].start, exception_windows[-1].stop), (330, 430))

    def test_rejects_insufficient_duplicate_or_non_numeric_image_ids(self) -> None:
        with self.assertRaisesRegex(ValueError, "need at least 4"):
            build_fastvggt_frame_selection(_images([0, 2, 4]), [0, 2, 4], input_frames=4)
        with self.assertRaisesRegex(ValueError, "duplicate image frame ID"):
            build_fastvggt_frame_selection(
                [Path("a/7.jpg"), Path("b/7.png"), Path("a/9.jpg")],
                [7, 9],
                input_frames=2,
            )
        with self.assertRaisesRegex(ValueError, "numeric stem"):
            build_fastvggt_frame_selection(
                [Path("color/not-a-frame.jpg")], [0], input_frames=1
            )

    def test_rejects_duplicate_or_invalid_pose_ids(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite_pose_frame_ids must be unique"):
            build_fastvggt_frame_selection(_images([1, 2]), [1, 1, 2], input_frames=2)
        with self.assertRaisesRegex(ValueError, "finite_pose_frame_ids must contain integers"):
            build_fastvggt_frame_selection(_images([1, 2]), [1, 2.5], input_frames=2)


if __name__ == "__main__":
    unittest.main()
