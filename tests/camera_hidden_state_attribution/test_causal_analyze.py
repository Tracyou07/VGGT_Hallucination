import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from pre_experiments.camera_hidden_state_attribution.causal_analyze import (
    aggregate_scene_effects,
    freeze_causal_normalization,
    validate_frozen_causal_normalization,
    write_causal_numeric_summary,
)


def _scene(
    name: str,
    translation: list[float],
    rotation: list[float],
    fov: list[float],
) -> dict[str, object]:
    direct_count = 1
    hidden_dim = len(translation)
    return {
        "scene": name,
        "activation_scale": np.arange(
            1,
            hidden_dim + 1,
            dtype=np.float64,
        )[None, :],
        "translation_effect": np.asarray([translation], dtype=np.float64),
        "rotation_effect_deg": np.asarray([rotation], dtype=np.float64),
        "fov_effect": np.asarray([fov], dtype=np.float64),
        "measured_basis_mask": np.ones((1, 9), dtype=bool),
        "direct_iteration": np.zeros(direct_count, dtype=np.int64),
        "direct_unit": np.zeros(direct_count, dtype=np.int64),
        "direct_projected_translation": np.asarray([translation[0]]),
        "direct_measured_translation": np.asarray([translation[0] * 1.1]),
        "direct_projected_rotation_deg": np.asarray([rotation[0]]),
        "direct_measured_rotation_deg": np.asarray([rotation[0] * 1.1]),
        "direct_projected_fov": np.asarray([fov[0]]),
        "direct_measured_fov": np.asarray([fov[0] * 1.1]),
    }


class CausalAnalyzeTest(unittest.TestCase):
    def setUp(self):
        self.calibration_scenes = [
            _scene(
                "a",
                [1.0, 3.0, 5.0],
                [2.0, 6.0, 10.0],
                [4.0, 12.0, 20.0],
            ),
            _scene(
                "b",
                [3.0, 5.0, 7.0],
                [4.0, 8.0, 12.0],
                [8.0, 16.0, 24.0],
            ),
        ]
        self.measurement = {
            "method": "centered_pose_delta_jacobian_projection",
            "num_iterations": 1,
            "target_dim": 9,
            "basis_step": 0.001,
        }

    def test_scene_equal_aggregate_and_frozen_provenance(self):
        aggregate = aggregate_scene_effects(
            self.calibration_scenes,
            require_complete_basis=True,
        )
        np.testing.assert_allclose(
            aggregate["effects_mean"]["translation"],
            [[2.0, 4.0, 6.0]],
        )
        np.testing.assert_allclose(
            aggregate["effects_mean"]["rotation"],
            [[3.0, 7.0, 11.0]],
        )
        np.testing.assert_allclose(
            aggregate["effects_mean"]["fov"],
            [[6.0, 14.0, 22.0]],
        )

        frozen = freeze_causal_normalization(
            aggregate,
            split_digest="split",
            calibration_scenes=["a", "b"],
            measurement_config=self.measurement,
            quantile=0.5,
        )
        self.assertEqual(
            frozen["normalization_scales"],
            {"translation": 4.0, "rotation": 7.0, "fov": 14.0},
        )
        validated = validate_frozen_causal_normalization(
            frozen,
            split_digest="split",
            calibration_scenes=["a", "b"],
            measurement_config=self.measurement,
        )
        self.assertEqual(validated["frozen_digest"], frozen["frozen_digest"])
        changed = dict(self.measurement)
        changed["basis_step"] = 0.01
        with self.assertRaisesRegex(ValueError, "provenance"):
            validate_frozen_causal_normalization(
                frozen,
                split_digest="split",
                calibration_scenes=["a", "b"],
                measurement_config=changed,
            )

    def test_numeric_summary_reuses_frozen_scales_on_holdout(self):
        aggregate = aggregate_scene_effects(self.calibration_scenes)
        frozen = freeze_causal_normalization(
            aggregate,
            split_digest="split",
            calibration_scenes=["a", "b"],
            measurement_config=self.measurement,
            quantile=0.5,
        )
        holdout = [
            _scene(
                "h",
                [30.0, 2.0, 1.0],
                [14.0, 5.0, 3.0],
                [7.0, 4.0, 2.0],
            )
        ]
        output_weight = np.ones((9, 3), dtype=np.float64)
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            summary = write_causal_numeric_summary(
                run_dir,
                holdout,
                output_weight=output_weight,
                partition="holdout",
                frozen=frozen,
            )
            with (run_dir / "per_position.csv").open(
                newline="",
                encoding="utf-8",
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[0]["iteration"], "0")
            self.assertEqual(rows[0]["unit"], "0")
            self.assertAlmostEqual(
                float(rows[0]["normalized_translation_effect"]),
                7.5,
            )
            self.assertEqual(
                summary["calibration_comparison"]["translation"][
                    "spearman"
                ],
                -1.0,
            )
            self.assertEqual(
                summary["calibration_comparison"]["translation"][
                    "top_k_overlap"
                ],
                3,
            )
            self.assertEqual(
                json.loads(
                    (run_dir / "frozen_causal_normalization.json").read_text(
                        encoding="utf-8"
                    )
                )["frozen_digest"],
                frozen["frozen_digest"],
            )
            with (run_dir / "direct_checks.csv").open(
                newline="",
                encoding="utf-8",
            ) as handle:
                direct_rows = list(csv.DictReader(handle))
            self.assertEqual(len(direct_rows), 1)
            self.assertAlmostEqual(
                float(direct_rows[0]["translation_relative_error"]),
                1.0 / 11.0,
            )

    def test_formal_aggregate_rejects_incomplete_basis(self):
        incomplete = _scene(
            "bad",
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],
        )
        incomplete["measured_basis_mask"][0, 8] = False
        with self.assertRaisesRegex(ValueError, "basis"):
            aggregate_scene_effects(
                [incomplete],
                require_complete_basis=True,
            )
        incomplete["measured_basis_mask"] = np.ones((1, 8), dtype=bool)
        with self.assertRaisesRegex(ValueError, "basis"):
            aggregate_scene_effects(
                [incomplete],
                require_complete_basis=True,
            )


if __name__ == "__main__":
    unittest.main()
