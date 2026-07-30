import tempfile
import unittest
from pathlib import Path

from pre_experiments.camera_hidden_state_attribution.visualize_causal_trace import (
    build_trace_summary,
    render_trace_overview,
)


POSITIONS = ((0, 0), (0, 1), (1, 0), (1, 1))


def old_rows(rankings):
    rows = []
    for group, ordered_positions in rankings.items():
        for rank, (iteration, unit) in enumerate(ordered_positions, start=1):
            rows.append(
                {
                    "group": group,
                    "partition_rank": str(rank),
                    "iteration": str(iteration),
                    "unit": str(unit),
                }
            )
    return rows


def causal_rows(group_orders, preferred):
    effects = {
        group: {
            position: float(len(POSITIONS) - rank)
            for rank, position in enumerate(order)
        }
        for group, order in group_orders.items()
    }
    return [
        {
            "iteration": str(iteration),
            "unit": str(unit),
            "translation_effect_mean": str(effects["translation"][(iteration, unit)]),
            "rotation_effect_mean": str(effects["rotation"][(iteration, unit)]),
            "fov_effect_mean": str(effects["fov"][(iteration, unit)]),
            "preferred_group": preferred[(iteration, unit)],
        }
        for iteration, unit in POSITIONS
    ]


class CausalTraceSummaryTests(unittest.TestCase):
    def test_freezes_calibration_overlap_before_holdout_validation(self):
        groups = ("translation", "rotation", "fov")
        old_calibration = old_rows(
            {
                group: ((0, 0), (0, 1), (1, 0), (1, 1))
                for group in groups
            }
        )
        old_holdout = old_rows(
            {
                group: ((0, 0), (1, 0), (0, 1), (1, 1))
                for group in groups
            }
        )
        causal_calibration = causal_rows(
            {
                group: ((0, 0), (1, 0), (0, 1), (1, 1))
                for group in groups
            },
            {
                (0, 0): "translation",
                (0, 1): "rotation",
                (1, 0): "translation",
                (1, 1): "fov",
            },
        )
        causal_holdout = causal_rows(
            {
                group: ((0, 0), (1, 1), (0, 1), (1, 0))
                for group in groups
            },
            {
                (0, 0): "translation",
                (0, 1): "rotation",
                (1, 0): "fov",
                (1, 1): "translation",
            },
        )

        summary = build_trace_summary(
            old_calibration,
            old_holdout,
            causal_calibration,
            causal_holdout,
            top_k=2,
        )

        translation = summary["groups"]["translation"]
        self.assertEqual(summary["universe_size"], 4)
        self.assertEqual(translation["calibration_overlap"], 1)
        self.assertEqual(translation["holdout_overlap"], 1)
        self.assertEqual(translation["frozen_candidate_count"], 1)
        self.assertEqual(translation["frozen_recovered_count"], 1)
        self.assertEqual(translation["stable_preference_count"], 1)
        self.assertEqual(
            translation["holdout_causal_iteration_counts"],
            {"0": 1, "1": 1},
        )

    def test_renders_nonempty_overview_png(self):
        summary = {
            "top_k": 64,
            "universe_size": 4096,
            "groups": {
                "translation": {
                    "calibration_overlap": 41,
                    "holdout_overlap": 41,
                    "frozen_candidate_count": 41,
                    "frozen_recovered_count": 39,
                },
                "rotation": {
                    "calibration_overlap": 23,
                    "holdout_overlap": 26,
                    "frozen_candidate_count": 23,
                    "frozen_recovered_count": 23,
                },
                "fov": {
                    "calibration_overlap": 12,
                    "holdout_overlap": 12,
                    "frozen_candidate_count": 12,
                    "frozen_recovered_count": 11,
                },
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "overview.png"

            rendered = render_trace_overview(summary, output)

            self.assertEqual(rendered, output)
            self.assertGreater(output.stat().st_size, 10_000)


if __name__ == "__main__":
    unittest.main()
