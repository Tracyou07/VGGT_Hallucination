import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from pre_experiments.camera_iteration.model_io import find_checkpoint, load_local_model
from pre_experiments.camera_iteration.run_study import (
    parse_args,
    run_selection,
    selection_is_complete,
)
from vggt.utils.pose_enc import extri_intri_to_pose_encoding


class FakeCameraModel:
    def __init__(self, pose_enc, iterations):
        self.pose_enc = pose_enc
        self.iterations = iterations
        self.calls = 0

    def __call__(self, images, **options):
        self.calls += 1
        if options["camera_num_iterations"] != self.iterations:
            raise AssertionError("unexpected camera iteration count")
        poses = [self.pose_enc.clone() for _ in range(self.iterations)]
        delta_norm = torch.stack(
            [
                torch.full((1, self.pose_enc.shape[1]), float(index))
                for index in range(1, self.iterations + 1)
            ]
        )
        return {
            "pose_enc": poses[-1],
            "pose_enc_list": poses,
            "camera_trace": {
                "normalized_camera_tokens": torch.zeros(1, self.pose_enc.shape[1], 8),
                "raw_pose_enc_list": poses,
                "pose_delta_list": poses,
                "delta_norm": delta_norm,
                "pose_tokens_modulated_list": [],
            },
        }


def make_pose_fixture():
    gt_c2w = np.tile(np.eye(4, dtype=np.float64), (3, 1, 1))
    gt_c2w[1, 0, 3] = 1.0
    gt_c2w[2, 1, 3] = 1.0
    pred_w2c = np.linalg.inv(gt_c2w)
    extrinsics = torch.from_numpy(pred_w2c[:, :3, :4]).unsqueeze(0).float()
    intrinsics = torch.eye(3).repeat(1, 3, 1, 1)
    intrinsics[..., 0, 0] = 100.0
    intrinsics[..., 1, 1] = 100.0
    intrinsics[..., 0, 2] = 50.0
    intrinsics[..., 1, 2] = 50.0
    return gt_c2w, extri_intri_to_pose_encoding(extrinsics, intrinsics, (100, 100))


class StudyCLITest(unittest.TestCase):
    def test_parser_defaults_match_camera_protocol(self):
        args = parse_args([])
        self.assertEqual(args.frame_counts, [25, 50, 100, 200, 500])
        self.assertEqual(args.iterations, [1, 2, 4, 8, 16])
        self.assertEqual(args.sampling, "nested_uniform")
        self.assertEqual(args.scene_limit, 10)
        self.assertFalse(args.save_camera_tokens)
        self.assertFalse(args.save_context_diagnostics)

    def test_parser_enables_context_diagnostics_explicitly(self):
        args = parse_args(["--iterations", "4", "--save-context-diagnostics"])
        self.assertEqual(args.iterations, [4])
        self.assertTrue(args.save_context_diagnostics)

    def test_checkpoint_is_validated_before_model_construction(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = Path(tmp)
            with self.assertRaisesRegex(FileNotFoundError, "model.safetensors or model.pt"):
                find_checkpoint(checkpoint_dir)
            with mock.patch("pre_experiments.camera_iteration.model_io.VGGT") as model_type:
                with self.assertRaises(FileNotFoundError):
                    load_local_model(checkpoint_dir)
                model_type.assert_not_called()

    def test_run_selection_calls_model_once_and_writes_complete_artifacts(self):
        gt_c2w, pose_enc = make_pose_fixture()
        model = FakeCameraModel(pose_enc, iterations=4)
        selected_ids = [1, 2, 3]
        image_by_id = {frame_id: Path(f"{frame_id}.jpg") for frame_id in selected_ids}
        poses_by_id = {
            frame_id: gt_c2w[index] for index, frame_id in enumerate(selected_ids)
        }

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "selection"
            rows = run_selection(
                model=model,
                scene="scene0000_00",
                requested_frame_count=3,
                selected_ids=selected_ids,
                image_by_id=image_by_id,
                poses_by_id=poses_by_id,
                iterations=[1, 4],
                device=torch.device("cpu"),
                preprocess_mode="pad",
                output_dir=output_dir,
                run_id="abcdef0_123456789abc",
                save_camera_tokens=False,
                save_context_diagnostics=True,
                image_loader=lambda paths, mode: torch.zeros(len(paths), 3, 100, 100),
            )

            self.assertEqual(model.calls, 1)
            self.assertEqual([row["iteration"] for row in rows], [1, 4])
            for name in (
                "iteration_metrics.json",
                "iteration_metrics.csv",
                "camera_trace.npz",
                "context_diagnostics.npz",
                "selected_frame_ids.json",
                "complete.json",
            ):
                self.assertTrue((output_dir / name).is_file(), name)
            completion = json.loads((output_dir / "complete.json").read_text(encoding="utf-8"))
            self.assertEqual(completion["run_id"], "abcdef0_123456789abc")
            self.assertTrue(
                selection_is_complete(
                    output_dir,
                    run_id="abcdef0_123456789abc",
                    selected_ids=selected_ids,
                    iterations=[1, 4],
                    require_context_diagnostics=True,
                )
            )
            with np.load(output_dir / "context_diagnostics.npz") as diagnostics:
                np.testing.assert_array_equal(diagnostics["frame_ids"], selected_ids)
                np.testing.assert_allclose(diagnostics["gt_c2w_raw"], gt_c2w)
                np.testing.assert_allclose(
                    diagnostics["translation_error_aligned"], 0.0, atol=1e-5
                )
            self.assertFalse(
                selection_is_complete(
                    output_dir,
                    run_id="different_123456789abc",
                    selected_ids=selected_ids,
                    iterations=[1, 4],
                )
            )


if __name__ == "__main__":
    unittest.main()
