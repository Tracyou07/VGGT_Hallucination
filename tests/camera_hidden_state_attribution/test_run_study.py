import unittest
import json
from pathlib import Path
import tempfile

import numpy as np
import torch

from pre_experiments.camera_hidden_state_attribution.run_study import (
    _validate_local_run,
    collect_scene_statistics,
    replay_tokens,
)
from vggt.heads.camera_head import CameraHead


class ReplayStudyTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(5)
        self.head = CameraHead(
            dim_in=16,
            trunk_depth=1,
            num_heads=4,
            mlp_ratio=2,
        ).eval()
        self.global_tokens = np.random.default_rng(2).normal(
            size=(6, 16)
        ).astype(np.float32)
        self.local_tokens = self.global_tokens[1:5].copy()

    def test_replay_returns_iteration_hidden_and_pose(self):
        replay = replay_tokens(self.head, self.global_tokens, torch.device("cpu"))
        self.assertEqual(replay["hidden"].shape, (4, 6, 8))
        self.assertEqual(replay["trunk"].shape, (4, 6, 16))
        self.assertEqual(replay["pose_delta"].shape, (4, 6, 9))
        self.assertEqual(replay["pose_enc"].shape, (6, 9))
        self.assertEqual(replay["pred_c2w_raw"].shape, (6, 4, 4))

    def test_statistics_match_frame_ids_and_ignore_gt_values(self):
        global_replay = replay_tokens(
            self.head, self.global_tokens, torch.device("cpu")
        )
        local_replay = replay_tokens(
            self.head, self.local_tokens, torch.device("cpu")
        )
        global_artifact = {
            "frame_ids": np.arange(6),
            "normalized_camera_tokens": self.global_tokens,
            "pred_c2w_raw": global_replay["pred_c2w_raw"],
            "gt_c2w_raw": np.tile(np.eye(4), (6, 1, 1)),
        }
        local_artifact = {
            "frame_ids": np.arange(1, 5),
            "normalized_camera_tokens": self.local_tokens,
            "pred_c2w_raw": local_replay["pred_c2w_raw"],
            "gt_c2w_raw": np.tile(np.eye(4), (4, 1, 1)),
        }
        first = collect_scene_statistics(
            self.head,
            global_artifact,
            [local_artifact],
            torch.device("cpu"),
            scene="scene",
        )
        global_artifact["gt_c2w_raw"][:, :3, 3] = 999.0
        local_artifact["gt_c2w_raw"][:, :3, 3] = -999.0
        second = collect_scene_statistics(
            self.head,
            global_artifact,
            [local_artifact],
            torch.device("cpu"),
            scene="scene",
        )
        for group in ("translation", "rotation", "fov"):
            np.testing.assert_allclose(
                first["drift"][group], second["drift"][group]
            )

    def test_replay_rejects_pose_or_frame_identity_mismatch(self):
        replay = replay_tokens(self.head, self.global_tokens, torch.device("cpu"))
        global_artifact = {
            "frame_ids": np.arange(6),
            "normalized_camera_tokens": self.global_tokens,
            "pred_c2w_raw": replay["pred_c2w_raw"].copy(),
            "gt_c2w_raw": np.tile(np.eye(4), (6, 1, 1)),
        }
        local_replay = replay_tokens(
            self.head, self.local_tokens[:2], torch.device("cpu")
        )
        local = {
            "frame_ids": np.array([99, 100]),
            "normalized_camera_tokens": self.local_tokens[:2],
            "pred_c2w_raw": local_replay["pred_c2w_raw"],
            "gt_c2w_raw": np.tile(np.eye(4), (2, 1, 1)),
        }
        with self.assertRaisesRegex(ValueError, "frame ID"):
            collect_scene_statistics(
                self.head,
                global_artifact,
                [local],
                torch.device("cpu"),
                scene="scene",
            )
        global_artifact["pred_c2w_raw"][0, 0, 3] += 1.0
        with self.assertRaisesRegex(ValueError, "replay"):
            collect_scene_statistics(
                self.head,
                global_artifact,
                [],
                torch.device("cpu"),
                scene="scene",
            )

    def test_local_run_provenance_is_partition_and_split_bound(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "run_metadata.json").write_text(
                json.dumps(
                    {
                        "study_name": "local_global_consistency",
                        "partition": "calibration",
                        "split_digest": "split",
                    }
                ),
                encoding="utf-8",
            )
            _validate_local_run(
                root, partition="calibration", split_digest="split"
            )
            with self.assertRaisesRegex(ValueError, "provenance"):
                _validate_local_run(
                    root, partition="holdout", split_digest="split"
                )


if __name__ == "__main__":
    unittest.main()
