import unittest
import csv
import json
from pathlib import Path
import tempfile

import numpy as np
import torch

from pre_experiments.camera_hidden_state_attribution.run_replacement import (
    _finalize_alpha_selection,
    _freeze_from_calibration_dirs,
    _validate_frozen_replacement,
    parse_args,
    run_scene_replacement,
)
from pre_experiments.camera_hidden_state_attribution.run_study import (
    replay_tokens,
)
from vggt.heads.camera_head import CameraHead


class HiddenReplacementRunnerTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        self.device = torch.device("cpu")
        self.head = CameraHead(
            dim_in=16,
            trunk_depth=1,
            num_heads=4,
            mlp_ratio=2,
        ).eval()
        self.global_tokens = np.random.default_rng(4).normal(
            size=(6, 16)
        ).astype(np.float32)

    def test_scene_runner_replaces_short_hidden_and_reports_aligned_metrics(self):
        baseline = replay_tokens(
            self.head,
            self.global_tokens,
            self.device,
        )
        gt_c2w_raw = baseline["pred_c2w_raw"].copy()
        global_artifact = {
            "frame_ids": np.arange(6),
            "normalized_camera_tokens": self.global_tokens,
            "pred_c2w_raw": baseline["pred_c2w_raw"],
            "gt_c2w_raw": gt_c2w_raw,
        }
        local_records = []
        for window_index, start in enumerate((0, 2)):
            stop = start + 4
            local_tokens = self.global_tokens[start:stop] + (
                0.15 * (window_index + 1)
            )
            local_replay = replay_tokens(
                self.head,
                local_tokens,
                self.device,
            )
            local_records.append(
                {
                    "window_index": window_index,
                    "artifact": {
                        "frame_ids": np.arange(start, stop),
                        "normalized_camera_tokens": local_tokens,
                        "pred_c2w_raw": local_replay["pred_c2w_raw"],
                        "gt_c2w_raw": gt_c2w_raw[start:stop],
                    },
                }
            )
        frozen = {
            "selected": [{"iteration": 0, "unit": 0}],
            "control_sets": [
                {
                    "name": "control_00",
                    "positions": [{"iteration": 0, "unit": 1}],
                }
            ],
        }

        result = run_scene_replacement(
            self.head,
            global_artifact,
            local_records,
            frozen,
            self.device,
            scene="scene",
            alphas=(0.25, 1.0),
        )

        self.assertEqual(
            [row["condition"] for row in result["rows"]],
            [
                "baseline",
                "selected_a0p25",
                "control_00_a0p25",
                "selected_a1",
                "control_00_a1",
            ],
        )
        selected = result["rows"][1]
        self.assertEqual(selected["replacement_count"], 1)
        self.assertEqual(selected["condition_family"], "selected")
        self.assertEqual(selected["alpha"], 0.25)
        self.assertGreater(
            selected["camera_center_displacement_mean"],
            0.0,
        )
        self.assertTrue(
            np.isfinite(selected["aligned_translation_error_mean"])
        )
        self.assertEqual(
            result["pred_c2w_raw"].shape,
            (5, 6, 4, 4),
        )
        np.testing.assert_allclose(
            result["condition_alpha"],
            np.array([0.0, 0.25, 0.25, 1.0, 1.0]),
        )
        np.testing.assert_array_equal(
            result["selected_window_index"],
            np.array([0, 0, 0, 1, 1, 1]),
        )

    def test_cli_requires_calibration_sources_or_frozen_holdout(self):
        common = [
            "--source-run-dir",
            "source",
            "--local-run-dir",
            "local",
            "--split-manifest",
            "split.json",
            "--ckpt-dir",
            "ckpt",
        ]
        with self.assertRaises(SystemExit):
            parse_args(["--stage", "calibration", *common])
        with self.assertRaises(SystemExit):
            parse_args(["--stage", "holdout", *common])
        args = parse_args(
            [
                "--stage",
                "calibration",
                "--attribution-calibration-dir",
                "attribution",
                "--causal-calibration-dir",
                "causal",
                "--alphas",
                "0.01,0.1,1",
                *common,
            ]
        )
        self.assertEqual(args.alphas, (0.01, 0.1, 1.0))
        with self.assertRaises(SystemExit):
            parse_args(
                [
                    "--stage",
                    "calibration",
                    "--attribution-calibration-dir",
                    "attribution",
                    "--causal-calibration-dir",
                    "causal",
                    "--alphas",
                    "0.1,0.1",
                    *common,
                ]
            )

    def test_freeze_reads_authenticated_calibration_numeric_runs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            attribution = root / "attribution"
            causal = root / "causal"
            attribution.mkdir()
            causal.mkdir()
            for directory, study in (
                (attribution, "camera_hidden_state_attribution"),
                (causal, "camera_hidden_causal_preference"),
            ):
                (directory / "run_metadata.json").write_text(
                    json.dumps(
                        {
                            "run_id": directory.name,
                            "study_name": study,
                            "partition": "calibration",
                            "split_digest": "split",
                            "protocol_complete": True,
                        }
                    ),
                    encoding="utf-8",
                )
                (directory / "complete.json").write_text(
                    json.dumps(
                        {
                            "run_id": directory.name,
                            "partition": "calibration",
                            "analysis_complete": True,
                            "protocol_complete": True,
                        }
                    ),
                    encoding="utf-8",
                )
            with (attribution / "per_unit.csv").open(
                "w",
                newline="",
                encoding="utf-8",
            ) as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=(
                        "group",
                        "iteration",
                        "unit",
                        "partition_rank",
                    ),
                )
                writer.writeheader()
                for rank, unit in enumerate(range(8), start=1):
                    writer.writerow(
                        {
                            "group": "translation",
                            "iteration": 0,
                            "unit": unit,
                            "partition_rank": rank,
                        }
                    )
            with (causal / "per_position.csv").open(
                "w",
                newline="",
                encoding="utf-8",
            ) as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=(
                        "iteration",
                        "unit",
                        "translation_effect_mean",
                    ),
                )
                writer.writeheader()
                for unit, score in enumerate((1, 8, 7, 2, 6, 5, 4, 3)):
                    writer.writerow(
                        {
                            "iteration": 0,
                            "unit": unit,
                            "translation_effect_mean": score,
                        }
                    )

            frozen = _freeze_from_calibration_dirs(
                attribution,
                causal,
                split_digest="split",
                calibration_scenes=["scene"],
                source_top_k=3,
                control_repeats=1,
                seed=3,
            )
            validated = _validate_frozen_replacement(
                frozen,
                split_digest="split",
                calibration_scenes=["scene"],
            )
            finalized = _finalize_alpha_selection(
                validated,
                {
                    "selected_alpha": 0.1,
                    "alpha_selection_metric": "metric",
                    "calibration_selected_delta": -0.02,
                },
                alpha_grid=(0.01, 0.1, 1.0),
            )
            validated_final = _validate_frozen_replacement(
                finalized,
                split_digest="split",
                calibration_scenes=["scene"],
                require_selected_alpha=True,
            )

        self.assertEqual(validated["source_runs"]["attribution"], "attribution")
        self.assertEqual(validated["source_runs"]["causal"], "causal")
        self.assertEqual(validated_final["selected_alpha"], 0.1)
        tampered = dict(validated)
        tampered["selected_count"] = 999
        with self.assertRaisesRegex(ValueError, "provenance"):
            _validate_frozen_replacement(
                tampered,
                split_digest="split",
                calibration_scenes=["scene"],
            )


if __name__ == "__main__":
    unittest.main()
