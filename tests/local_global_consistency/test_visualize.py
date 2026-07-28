import csv
import json
from pathlib import Path
import tempfile
import unittest

from pre_experiments.local_global_consistency.visualize import (
    write_visualizations,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class VisualizationTest(unittest.TestCase):
    def test_holdout_visualizations_are_nonempty_and_numeric_only_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "holdout"
            run_dir.mkdir()
            scenes = ["scene0000_00", "scene0001_00"]
            score_rows = []
            validation_rows = []
            scene_rows = []
            for scene_index, scene in enumerate(scenes):
                for frame_id in range(20):
                    value = float(scene_index + frame_id + 1)
                    score_rows.append(
                        {
                            "scene": scene,
                            "frame_id": frame_id,
                            "global_local_token_cosine": value / 20.0,
                            "global_local_pose_translation": value / 10.0,
                            "global_local_pose_rotation_deg": value,
                            "token_local_reliable": frame_id < 18,
                            "pose_local_reliable": frame_id < 16,
                        }
                    )
                    validation_rows.append(
                        {
                            "scene": scene,
                            "frame_id": frame_id,
                            "translation_error_growth_global_minus_local": value / 8.0
                            - 1.0,
                            "rotation_error_growth_global_minus_local_deg": value - 5.0,
                        }
                    )
                scene_rows.append(
                    {
                        "scene": scene,
                        "translation_growth_mean": float(scene_index + 1),
                        "rotation_growth_mean": float((scene_index + 1) * 2),
                        "token_reliable_coverage": 0.9 - scene_index * 0.1,
                        "pose_reliable_coverage": 0.8 - scene_index * 0.1,
                    }
                )
            aggregate_rows = [
                {
                    "metric": "translation_growth_mean",
                    "estimate": 1.5,
                    "ci95_low": 1.0,
                    "ci95_high": 2.0,
                },
                {
                    "metric": "rotation_growth_mean",
                    "estimate": 3.0,
                    "ci95_low": 2.0,
                    "ci95_high": 4.0,
                },
                {
                    "metric": "token_reliable_coverage",
                    "estimate": 0.85,
                    "ci95_low": 0.8,
                    "ci95_high": 0.9,
                },
                {
                    "metric": "pose_reliable_coverage",
                    "estimate": 0.75,
                    "ci95_low": 0.7,
                    "ci95_high": 0.8,
                },
            ]
            _write_csv(
                run_dir / "holdout_prediction_scores_per_frame.csv", score_rows
            )
            _write_csv(
                run_dir / "holdout_gt_validation_per_frame.csv", validation_rows
            )
            _write_csv(run_dir / "holdout_per_scene_summary.csv", scene_rows)
            _write_csv(run_dir / "holdout_aggregate_summary.csv", aggregate_rows)
            split = {
                "calibration_scenes": [scenes[0]],
                "scene_difficulty": {
                    scenes[0]: {
                        "difficulty_score": 0.2,
                        "stratum": "easy",
                        "selected_for_calibration": True,
                    },
                    scenes[1]: {
                        "difficulty_score": 0.8,
                        "stratum": "hard",
                        "selected_for_calibration": False,
                    },
                },
            }
            split_path = root / "split.json"
            split_path.write_text(json.dumps(split), encoding="utf-8")

            outputs = write_visualizations(
                run_dir,
                mode="holdout",
                split_manifest=split_path,
            )

            self.assertEqual(len(outputs), 5)
            self.assertEqual(
                {path.name for path in outputs},
                {
                    "split_difficulty.png",
                    "holdout_error_growth_by_scene.png",
                    "holdout_score_vs_error_growth.png",
                    "holdout_reliability_coverage.png",
                    "holdout_aggregate_ci.png",
                },
            )
            self.assertTrue(all(path.stat().st_size > 1000 for path in outputs))


if __name__ == "__main__":
    unittest.main()
