import unittest

from pre_experiments.camera_hidden_state_attribution.adaptive_alpha import (
    FEATURE_FIELDS,
)
from pre_experiments.camera_hidden_state_attribution.run_adaptive_replacement import (
    assign_holdout_alphas,
    build_calibration_bundle,
    parse_args,
)


def _score_rows(scene_count):
    rows = []
    for index in range(scene_count):
        scene = f"scene_{index:02d}"
        for frame in range(3):
            rows.append(
                {
                    "scene": scene,
                    "frame_id": frame,
                    "global_local_token_cosine": index + frame / 10,
                    "global_local_pose_translation": index + frame / 5,
                    "local_local_pose_translation": (
                        None if frame == 0 else index + frame / 3
                    ),
                }
            )
    return rows


def _replacement_rows(scene_count):
    rows = []
    choices = (0.01, 0.02, 0.05)
    for index in range(scene_count):
        oracle = choices[index % len(choices)]
        for alpha in choices:
            rows.append(
                {
                    "scene": f"scene_{index:02d}",
                    "condition_family": "selected",
                    "alpha": alpha,
                    "aligned_translation_error_delta": abs(alpha - oracle),
                }
            )
    return rows


class AdaptiveRunnerTest(unittest.TestCase):
    def test_calibration_bundle_fits_and_reports_loocv(self):
        bundle = build_calibration_bundle(
            _score_rows(10),
            _replacement_rows(10),
            expected_scenes=[f"scene_{index:02d}" for index in range(10)],
            split_digest="split",
            score_run_id="scores",
            replacement_run_id="replacement",
        )

        self.assertEqual(bundle["report"]["scene_count"], 10)
        self.assertEqual(
            bundle["frozen"]["calibration_scenes"],
            [f"scene_{index:02d}" for index in range(10)],
        )
        self.assertEqual(len(bundle["features"]), 10)

    def test_holdout_assignment_uses_only_frozen_features(self):
        bundle = build_calibration_bundle(
            _score_rows(10),
            _replacement_rows(10),
            expected_scenes=[f"scene_{index:02d}" for index in range(10)],
            split_digest="split",
            score_run_id="scores",
            replacement_run_id="replacement",
        )
        holdout = [
            {
                "scene": "holdout",
                FEATURE_FIELDS[0]: 1.0,
                FEATURE_FIELDS[1]: 1.0,
                FEATURE_FIELDS[2]: 1.0,
            }
        ]

        assignments = assign_holdout_alphas(
            bundle["frozen"],
            holdout,
            expected_scenes=["holdout"],
        )

        self.assertEqual(set(assignments), {"holdout"})
        self.assertIn(assignments["holdout"], (0.01, 0.02, 0.05))

    def test_cli_separates_calibration_and_holdout_requirements(self):
        common = [
            "--stage",
            "calibration",
            "--score-run-dir",
            "scores",
            "--split-manifest",
            "split.json",
        ]
        with self.assertRaises(SystemExit):
            parse_args(common)
        calibration = parse_args(
            [*common, "--replacement-calibration-dir", "replacement"]
        )
        self.assertEqual(calibration.stage, "calibration")
        with self.assertRaises(SystemExit):
            parse_args(
                [
                    "--stage",
                    "holdout",
                    "--score-run-dir",
                    "scores",
                    "--split-manifest",
                    "split.json",
                ]
            )
        holdout = parse_args(
            [
                "--stage",
                "holdout",
                "--score-run-dir",
                "scores",
                "--split-manifest",
                "split.json",
                "--selector",
                "selector.json",
                "--frozen-replacement",
                "frozen.json",
                "--fixed-holdout-dir",
                "fixed",
                "--source-run-dir",
                "source",
                "--local-run-dir",
                "local",
            ]
        )
        self.assertEqual(holdout.fixed_holdout_dir.name, "fixed")


if __name__ == "__main__":
    unittest.main()
