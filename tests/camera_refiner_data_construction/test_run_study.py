import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from pre_experiments.camera_hidden_state_attribution.run_study import replay_tokens
from pre_experiments.camera_refiner_data_construction.protocol import Candidate
from pre_experiments.camera_refiner_data_construction.run_study import (
    parse_args,
    run_scene_candidates,
    validate_scale_runs,
)
from pre_experiments.local_global_consistency.windows import build_sliding_windows
from vggt.heads.camera_head import CameraHead


class ScaleRunProvenanceTest(unittest.TestCase):
    def test_accepts_only_exact_scale_stride_and_source_protocol(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = {}
            for scale in (100, 200, 300):
                path = root / str(scale)
                path.mkdir()
                metadata = {
                    "study_name": "local_global_consistency",
                    "run_id": f"run_{scale}",
                    "partition": "calibration",
                    "split_digest": "split",
                    "source_run_id": "source",
                    "protocol_complete": True,
                    "invocation": {
                        "window_length": scale,
                        "window_stride": scale // 2,
                        "camera_iterations": 4,
                        "preprocess_mode": "pad",
                        "partition_scenes": ["scene_a", "scene_b"],
                        "scenes": ["scene_a", "scene_b"],
                    },
                }
                (path / "run_metadata.json").write_text(
                    json.dumps(metadata), encoding="utf-8"
                )
                paths[scale] = path

            validated = validate_scale_runs(
                paths,
                partition="calibration",
                split_digest="split",
                source_run_id="source",
                expected_scenes=["scene_a", "scene_b"],
            )
            self.assertEqual(
                validated,
                {100: "run_100", 200: "run_200", 300: "run_300"},
            )

            metadata = json.loads((paths[200] / "run_metadata.json").read_text())
            metadata["invocation"]["window_stride"] = 50
            (paths[200] / "run_metadata.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "200/100"):
                validate_scale_runs(
                    paths,
                    partition="calibration",
                    split_digest="split",
                    source_run_id="source",
                    expected_scenes=["scene_a", "scene_b"],
                )

    def test_rejects_missing_scale_and_partition_scene_mismatch(self):
        with self.assertRaisesRegex(ValueError, "exactly scales"):
            validate_scale_runs(
                {},
                partition="calibration",
                split_digest="split",
                source_run_id="source",
                expected_scenes=["scene"],
            )

    def test_accepts_one_scene_incomplete_runs_only_for_smoke(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = {}
            for scale in (100, 200, 300):
                path = root / str(scale)
                path.mkdir()
                (path / "run_metadata.json").write_text(
                    json.dumps(
                        {
                            "study_name": "local_global_consistency",
                            "run_id": f"smoke_{scale}",
                            "partition": "calibration",
                            "split_digest": "split",
                            "source_run_id": "source",
                            "protocol_complete": False,
                            "invocation": {
                                "window_length": scale,
                                "window_stride": scale // 2,
                                "camera_iterations": 4,
                                "preprocess_mode": "pad",
                                "partition_scenes": ["scene_a", "scene_b"],
                                "scenes": ["scene_a"],
                            },
                        }
                    ),
                    encoding="utf-8",
                )
                paths[scale] = path

            validated = validate_scale_runs(
                paths,
                partition="calibration",
                split_digest="split",
                source_run_id="source",
                expected_scenes=["scene_a", "scene_b"],
                run_scenes=["scene_a"],
                require_complete=False,
            )
            self.assertEqual(validated[300], "smoke_300")
            with self.assertRaisesRegex(ValueError, "100/50"):
                validate_scale_runs(
                    paths,
                    partition="calibration",
                    split_digest="split",
                    source_run_id="source",
                    expected_scenes=["scene_a", "scene_b"],
                )

    def test_cli_requires_three_scale_runs_and_holdout_policy(self):
        common = [
            "--source-run-dir",
            "source",
            "--split-manifest",
            "split.json",
            "--ckpt-dir",
            "ckpt",
            "--frozen-units",
            "units.json",
            "--scale-run",
            "100=run100",
            "--scale-run",
            "200=run200",
            "--scale-run",
            "300=run300",
        ]
        calibration = parse_args(["--stage", "calibration", *common])
        self.assertEqual(set(calibration.scale_runs), {100, 200, 300})
        self.assertEqual(calibration.candidate_family, "pure")
        with self.assertRaises(SystemExit):
            parse_args(["--stage", "holdout", *common])
        holdout = parse_args(
            ["--stage", "holdout", "--frozen-policy", "policy.json", *common]
        )
        self.assertEqual(holdout.stage, "holdout")


class SceneCandidateReplayTest(unittest.TestCase):
    def test_replays_multiscale_hidden_candidates_through_real_camera_head(self):
        torch.manual_seed(7)
        rng = np.random.default_rng(11)
        device = torch.device("cpu")
        head = CameraHead(
            dim_in=16,
            trunk_depth=1,
            num_heads=4,
            mlp_ratio=2,
        ).eval()
        frame_count = 300
        frame_ids = np.arange(frame_count, dtype=np.int64)
        global_tokens = rng.normal(size=(frame_count, 16)).astype(np.float32)
        global_replay = replay_tokens(head, global_tokens, device)
        global_artifact = {
            "frame_ids": frame_ids,
            "normalized_camera_tokens": global_tokens,
            "pred_c2w_raw": global_replay["pred_c2w_raw"],
            "gt_c2w_raw": global_replay["pred_c2w_raw"].copy(),
        }

        scale_records = {}
        for scale in (100, 200, 300):
            records = []
            for window in build_sliding_windows(
                frame_ids, length=scale, stride=scale // 2
            ):
                ids = np.asarray(window.frame_ids)
                tokens = global_tokens[window.start : window.stop] + scale / 10000.0
                replay = replay_tokens(head, tokens, device)
                records.append(
                    {
                        "window_index": window.index,
                        "artifact": {
                            "frame_ids": ids,
                            "normalized_camera_tokens": tokens,
                            "pred_c2w_raw": replay["pred_c2w_raw"],
                            "gt_c2w_raw": global_artifact["gt_c2w_raw"][
                                window.start : window.stop
                            ],
                        },
                    }
                )
            scale_records[scale] = records

        result = run_scene_candidates(
            head,
            global_artifact,
            scale_records,
            {"selected": [{"iteration": 0, "unit": 0}]},
            (Candidate(alpha=0.02, beta=(1.0, 0.0, 0.0)),),
            device,
            scene="scene0000_00",
        )

        np.testing.assert_array_equal(
            result["candidate_names"], ["baseline", "a0p02_b1_0_0"]
        )
        self.assertEqual(result["global_hidden"].shape, (4, 300, 8))
        self.assertEqual(result["local_hidden"].shape, (3, 4, 300, 8))
        self.assertEqual(result["pred_c2w_raw"].shape, (2, 300, 4, 4))
        self.assertEqual(result["translation_error_aligned"].shape, (2, 300))
        self.assertGreater(result["hidden_displacement_rms"][1], 0.0)
        self.assertTrue(np.isfinite(result["rotation_error_deg_aligned"]).all())


if __name__ == "__main__":
    unittest.main()
