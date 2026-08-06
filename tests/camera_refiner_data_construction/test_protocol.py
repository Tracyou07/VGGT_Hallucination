import unittest

import numpy as np

from pre_experiments.camera_refiner_data_construction.protocol import (
    Candidate,
    LOCAL_SCALES,
    assemble_multiscale_hidden,
    default_mixture_candidates,
    default_pure_candidates,
    mix_local_hidden,
)
from pre_experiments.local_global_consistency.windows import (
    build_sliding_windows,
)


class MultiscaleProtocolTest(unittest.TestCase):
    def test_candidates_have_stable_unique_identity_and_valid_simplex(self):
        candidate = Candidate(alpha=0.02, beta=(1.0, 0.0, 0.0))

        self.assertEqual(candidate.name, "a0p02_b1_0_0")
        self.assertEqual(len(default_pure_candidates()), 12)
        self.assertEqual(len(default_mixture_candidates()), 12)
        self.assertEqual(
            len({item.name for item in default_pure_candidates()}),
            12,
        )

        for alpha, beta in (
            (0.0, (1.0, 0.0, 0.0)),
            (1.1, (1.0, 0.0, 0.0)),
            (0.1, (0.4, 0.4, 0.4)),
            (0.1, (1.0, -0.1, 0.1)),
            (0.1, (1.0, 0.0)),
        ):
            with self.subTest(alpha=alpha, beta=beta):
                with self.assertRaises(ValueError):
                    Candidate(alpha=alpha, beta=beta)  # type: ignore[arg-type]

    def test_assembles_exact_multiscale_windows_by_interior_distance(self):
        frame_ids = np.arange(430, dtype=np.int64)
        scale_windows = {}
        for scale in LOCAL_SCALES:
            records = []
            for window in build_sliding_windows(
                frame_ids,
                length=scale,
                stride=scale // 2,
            ):
                hidden = np.full(
                    (2, scale, 3),
                    fill_value=scale + window.index,
                    dtype=np.float32,
                )
                records.append(
                    {
                        "window_index": window.index,
                        "frame_ids": np.asarray(window.frame_ids),
                        "hidden": hidden,
                    }
                )
            scale_windows[scale] = records

        result = assemble_multiscale_hidden(frame_ids, scale_windows)

        self.assertEqual(result["hidden"].shape, (3, 2, 430, 3))
        np.testing.assert_array_equal(result["scales"], np.array(LOCAL_SCALES))
        self.assertEqual(result["selected_window_index"].shape, (3, 430))
        self.assertEqual(result["selected_window_start"].shape, (3, 430))
        self.assertEqual(result["selected_window_stop"].shape, (3, 430))
        self.assertEqual(result["hidden"][0, 0, 0, 0], 100.0)
        self.assertEqual(result["hidden"][0, 0, 75, 0], 101.0)
        self.assertEqual(result["selected_window_start"][0, 75], 50)
        self.assertEqual(result["selected_window_stop"][0, 75], 150)
        self.assertGreaterEqual(result["observation_count"].min(), 1)

    def test_rejects_missing_scale_and_noncanonical_windows(self):
        frame_ids = np.arange(430, dtype=np.int64)
        with self.assertRaisesRegex(ValueError, "exactly scales"):
            assemble_multiscale_hidden(frame_ids, {100: []})

        scale_windows = {}
        for scale in LOCAL_SCALES:
            windows = build_sliding_windows(
                frame_ids,
                length=scale,
                stride=scale // 2,
            )
            scale_windows[scale] = [
                {
                    "window_index": window.index,
                    "frame_ids": np.asarray(window.frame_ids),
                    "hidden": np.zeros((1, scale, 2), dtype=np.float32),
                }
                for window in windows
            ]
        scale_windows[200][0]["frame_ids"] = np.arange(1, 201)
        with self.assertRaisesRegex(ValueError, "canonical"):
            assemble_multiscale_hidden(frame_ids, scale_windows)

    def test_mixes_scale_axis_and_rejects_nonfinite_hidden(self):
        hidden = np.zeros((3, 1, 2, 1), dtype=np.float32)
        hidden[0] = 2.0
        hidden[1] = 4.0
        hidden[2] = 10.0

        mixed = mix_local_hidden(hidden, (0.2, 0.3, 0.5))

        np.testing.assert_allclose(mixed, np.full((1, 2, 1), 6.6))
        hidden[2, 0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            mix_local_hidden(hidden, (0.2, 0.3, 0.5))


if __name__ == "__main__":
    unittest.main()
