from __future__ import annotations

import unittest

import numpy as np

from pre_experiments.camera_velocity_ambiguity_02.units import build_overlap_units
from pre_experiments.local_global_consistency.windows import (
    FrameWindow,
    build_sliding_windows,
)


class OverlapUnitTest(unittest.TestCase):
    def test_preserves_exact_adjacent_pair_identity_for_noncontiguous_frames(self) -> None:
        frame_ids = np.asarray([2, 5, 9, 14, 20, 27], dtype=np.int64)
        windows = build_sliding_windows(frame_ids, length=4, stride=2)
        units = build_overlap_units("scene0000_00", windows, primary_overlap=2)

        self.assertEqual(len(units), 1)
        unit = units[0]
        self.assertEqual(unit.pair_id, "scene0000_00/window_000__window_001")
        self.assertEqual((unit.left_window_index, unit.right_window_index), (0, 1))
        self.assertEqual(unit.shared_frame_ids, (9, 14))
        self.assertEqual(unit.left_shared_indices, (2, 3))
        self.assertEqual(unit.right_shared_indices, (0, 1))
        self.assertEqual(unit.global_shared_indices, (2, 3))
        self.assertEqual(unit.route, "primary")

    def test_protocol_routes_exactly_398_primary_and_one_secondary_pair(self) -> None:
        primary = 0
        secondary = 0
        for scene_index in range(50):
            scene = f"scene{scene_index:04d}_00"
            count = 430 if scene_index == 15 else 500
            windows = build_sliding_windows(
                np.arange(count, dtype=np.int64), length=100, stride=50
            )
            for unit in build_overlap_units(scene, windows, primary_overlap=50):
                primary += unit.route == "primary"
                secondary += unit.route == "secondary"

        self.assertEqual((primary, secondary), (398, 1))

    def test_exception_keeps_twenty_triply_covered_frames_in_both_pairs(self) -> None:
        windows = build_sliding_windows(
            np.arange(430, dtype=np.int64), length=100, stride=50
        )
        units = build_overlap_units("scene0150_00", windows, primary_overlap=50)
        penultimate = units[-2]
        final = units[-1]
        repeated = set(penultimate.shared_frame_ids).intersection(final.shared_frame_ids)

        self.assertEqual(penultimate.pair_id, "scene0150_00/window_005__window_006")
        self.assertEqual(final.pair_id, "scene0150_00/window_006__window_007")
        self.assertEqual(penultimate.route, "primary")
        self.assertEqual(final.route, "secondary")
        self.assertEqual(len(final.shared_frame_ids), 70)
        self.assertEqual(repeated, set(range(330, 350)))

    def test_rejects_nonadjacent_duplicate_or_inconsistent_windows(self) -> None:
        base = build_sliding_windows(np.arange(6), length=4, stride=2)
        with self.assertRaisesRegex(ValueError, "adjacent window indices"):
            build_overlap_units("scene0000_00", [base[1], base[0]], primary_overlap=2)

        duplicate = FrameWindow(
            index=1,
            start=2,
            stop=6,
            frame_ids=(2, 3, 3, 5),
            boundary_distance=base[1].boundary_distance,
        )
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            build_overlap_units("scene0000_00", [base[0], duplicate], primary_overlap=2)

        inconsistent = FrameWindow(
            index=1,
            start=3,
            stop=7,
            frame_ids=(2, 3, 4, 5),
            boundary_distance=base[1].boundary_distance,
        )
        with self.assertRaisesRegex(ValueError, "window boundaries"):
            build_overlap_units("scene0000_00", [base[0], inconsistent], primary_overlap=2)


if __name__ == "__main__":
    unittest.main()
