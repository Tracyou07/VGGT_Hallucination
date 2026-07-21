import math
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from pre_experiments.local_global_consistency.alignment import (
    align_prediction_trajectories,
)
from pre_experiments.local_global_consistency.artifacts import (
    build_window_diagnostics,
    load_window_diagnostics,
)
from pre_experiments.local_global_consistency.analyze import write_analysis
from pre_experiments.local_global_consistency.run_study import (
    configure_camera_only,
    parse_args,
    run_window,
    window_is_complete,
)
from pre_experiments.local_global_consistency.metrics import (
    build_scene_rows,
    fit_reliability_thresholds,
    summarize_scores,
)
from pre_experiments.local_global_consistency.windows import build_sliding_windows


def rotation_z(angle: float) -> np.ndarray:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return np.array(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


class SlidingWindowTest(unittest.TestCase):
    def test_fixed_protocol_produces_nine_windows_and_expected_coverage(self):
        frame_ids = np.arange(1000, 1500, dtype=np.int64)

        windows = build_sliding_windows(frame_ids, length=100, stride=50)

        self.assertEqual([window.start for window in windows], list(range(0, 401, 50)))
        self.assertEqual([window.index for window in windows], list(range(9)))
        coverage = np.zeros(500, dtype=np.int64)
        for window in windows:
            coverage[window.start : window.stop] += 1
            np.testing.assert_array_equal(
                window.frame_ids,
                frame_ids[window.start : window.stop],
            )
            self.assertEqual(window.boundary_distance[0], 0)
            self.assertEqual(window.boundary_distance[-1], 0)
            self.assertEqual(max(window.boundary_distance), 49)

        np.testing.assert_array_equal(coverage[:50], 1)
        np.testing.assert_array_equal(coverage[50:450], 2)
        np.testing.assert_array_equal(coverage[450:], 1)

    def test_non_divisible_tail_is_covered_deterministically(self):
        windows = build_sliding_windows(np.arange(23), length=10, stride=6)
        self.assertEqual([window.start for window in windows], [0, 6, 12, 13])
        self.assertEqual(windows[-1].stop, 23)

    def test_invalid_windows_fail_closed(self):
        for length, stride in ((1, 1), (10, 0), (10, 11), (501, 50)):
            with self.subTest(length=length, stride=stride):
                with self.assertRaises(ValueError):
                    build_sliding_windows(np.arange(500), length=length, stride=stride)
        with self.assertRaisesRegex(ValueError, "unique"):
            build_sliding_windows(np.array([1, 1, 2]), length=2, stride=1)


class PredictionAlignmentTest(unittest.TestCase):
    def test_removes_prediction_coordinate_sim3_without_gt(self):
        reference = np.tile(np.eye(4, dtype=np.float64), (5, 1, 1))
        reference[:, :3, 3] = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.2, 0.0],
                [1.8, 1.2, 0.1],
                [2.2, 2.0, 0.4],
                [3.0, 2.3, 0.8],
            ]
        )
        for index in range(len(reference)):
            reference[index, :3, :3] = rotation_z(0.1 * index)

        scale = 2.5
        rotation = rotation_z(0.35)
        translation = np.array([4.0, -2.0, 0.7])
        moving = reference.copy()
        moving[:, :3, 3] = (
            (reference[:, :3, 3] - translation) @ rotation
        ) / scale
        moving[:, :3, :3] = np.einsum(
            "ij,sjk->sik", rotation.T, reference[:, :3, :3]
        )

        result = align_prediction_trajectories(reference, moving)

        np.testing.assert_allclose(result["aligned_c2w"], reference, atol=1e-10)
        np.testing.assert_allclose(result["translation_residual"], 0.0, atol=1e-10)
        np.testing.assert_allclose(result["rotation_residual_deg"], 0.0, atol=1e-6)
        self.assertAlmostEqual(result["sim3_scale"], scale, places=10)

    def test_rejects_mismatched_or_degenerate_trajectories(self):
        poses = np.tile(np.eye(4, dtype=np.float64), (3, 1, 1))
        with self.assertRaisesRegex(ValueError, "same shape"):
            align_prediction_trajectories(poses, poses[:2])
        with self.assertRaisesRegex(ValueError, "variance"):
            align_prediction_trajectories(poses, poses)


class WindowArtifactTest(unittest.TestCase):
    def test_stores_raw_gt_and_raw_prediction_without_aligned_gt(self):
        gt = np.tile(np.eye(4, dtype=np.float64), (3, 1, 1))
        gt[:, 0, 3] = [0.0, 1.0, 2.0]
        pred_c2w = gt.copy()
        pred_c2w[:, :3, 3] *= 2.0
        artifact = build_window_diagnostics(
            frame_ids=np.array([10, 20, 30]),
            normalized_camera_tokens=np.ones((3, 4), dtype=np.float32),
            pred_w2c=np.linalg.inv(pred_c2w),
            gt_c2w_raw=gt,
        )

        self.assertEqual(
            set(artifact),
            {"frame_ids", "normalized_camera_tokens", "pred_c2w_raw", "gt_c2w_raw"},
        )
        np.testing.assert_allclose(artifact["pred_c2w_raw"], pred_c2w)
        np.testing.assert_allclose(artifact["gt_c2w_raw"], gt)

    def test_rejects_mismatched_window_arrays(self):
        poses = np.tile(np.eye(4, dtype=np.float64), (3, 1, 1))
        with self.assertRaisesRegex(ValueError, "sequence length"):
            build_window_diagnostics(
                frame_ids=np.array([10, 20]),
                normalized_camera_tokens=np.ones((3, 4)),
                pred_w2c=np.linalg.inv(poses),
                gt_c2w_raw=poses,
            )


class LocalWindowRunnerContractTest(unittest.TestCase):
    def test_camera_only_runner_disables_non_camera_heads(self):
        model = SimpleNamespace(
            camera_head=object(), depth_head=object(), point_head=object(), track_head=object()
        )

        configured = configure_camera_only(model)

        self.assertIs(configured, model)
        self.assertIsNotNone(model.camera_head)
        self.assertIsNone(model.depth_head)
        self.assertIsNone(model.point_head)
        self.assertIsNone(model.track_head)

    def test_parser_fixes_round2a_defaults(self):
        args = parse_args([])
        self.assertEqual(args.window_length, 100)
        self.assertEqual(args.window_stride, 50)
        self.assertEqual(args.camera_iterations, 4)
        self.assertEqual(args.scene_limit, 4)

    def test_completion_requires_matching_window_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            poses = np.tile(np.eye(4, dtype=np.float64), (2, 1, 1))
            poses[:, 0, 3] = [0.0, 1.0]
            np.savez_compressed(
                directory / "window_diagnostics.npz",
                frame_ids=np.array([10, 20]),
                normalized_camera_tokens=np.ones((2, 4)),
                pred_c2w_raw=poses,
                gt_c2w_raw=poses,
            )
            (directory / "complete.json").write_text(
                '{"run_id":"run","scene":"scene","window_index":0,'
                '"start":0,"stop":2,"frame_ids":[10,20]}\n',
                encoding="utf-8",
            )

            self.assertTrue(
                window_is_complete(
                    directory,
                    run_id="run",
                    scene="scene",
                    window_index=0,
                    start=0,
                    stop=2,
                    frame_ids=[10, 20],
                )
            )
            self.assertFalse(
                window_is_complete(
                    directory,
                    run_id="run",
                    scene="scene",
                    window_index=0,
                    start=0,
                    stop=2,
                    frame_ids=[10, 30],
                )
            )
            np.savez_compressed(
                directory / "window_diagnostics.npz",
                frame_ids=np.array([10, 20]),
            )
            self.assertFalse(
                window_is_complete(
                    directory,
                    run_id="run",
                    scene="scene",
                    window_index=0,
                    start=0,
                    stop=2,
                    frame_ids=[10, 20],
                )
            )

    def test_run_window_writes_camera_trace_and_raw_pose_artifact(self):
        class FakeModel:
            def __call__(self, images, **kwargs):
                self.images_shape = tuple(images.shape)
                self.kwargs = kwargs
                return {
                    "pose_enc_list": [torch.zeros((1, 2, 9))],
                    "camera_trace": {
                        "normalized_camera_tokens": torch.tensor(
                            [[[1.0, 0.0], [0.0, 1.0]]]
                        )
                    },
                }

        model = FakeModel()
        extrinsic = torch.tensor(
            [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
             [[1.0, 0.0, 0.0, -1.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]]
        ).unsqueeze(0)
        window = build_sliding_windows(np.array([10, 20]), length=2, stride=1)[0]
        gt = np.tile(np.eye(4, dtype=np.float64), (2, 1, 1))
        gt[:, 0, 3] = [0.0, 1.0]
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            with mock.patch(
                "pre_experiments.local_global_consistency.run_study.pose_encoding_to_extri_intri",
                return_value=(extrinsic, None),
            ):
                completion = run_window(
                    model=model,
                    window=window,
                    scene="scene",
                    image_by_id={10: Path("10.jpg"), 20: Path("20.jpg")},
                    gt_c2w_raw=gt,
                    device=torch.device("cpu"),
                    preprocess_mode="pad",
                    output_dir=output,
                    run_id="run",
                    camera_iterations=4,
                    image_loader=lambda paths, mode: torch.zeros((2, 3, 8, 8)),
                )

            artifact = load_window_diagnostics(output / "window_diagnostics.npz")
            np.testing.assert_allclose(artifact["pred_c2w_raw"], gt)
            np.testing.assert_allclose(artifact["gt_c2w_raw"], gt)
            self.assertEqual(model.images_shape, (2, 3, 8, 8))
            self.assertEqual(model.kwargs["camera_num_iterations"], 4)
            self.assertTrue(model.kwargs["return_camera_trace"])
            self.assertEqual(completion["frame_ids"], [10, 20])


class LocalGlobalMetricTest(unittest.TestCase):
    def make_poses(self, count: int) -> np.ndarray:
        poses = np.tile(np.eye(4, dtype=np.float64), (count, 1, 1))
        poses[:, 0, 3] = np.arange(count, dtype=np.float64)
        poses[:, 1, 3] = np.square(np.arange(count, dtype=np.float64)) * 0.1
        return poses

    def test_scene_rows_keep_prediction_scores_separate_from_gt_validation(self):
        gt = self.make_poses(5)
        global_pred = gt.copy()
        global_pred[2, 1, 3] += 1.0
        global_tokens = np.tile(np.array([[1.0, 0.0]]), (5, 1))
        global_tokens[2] = [0.0, 1.0]
        global_artifact = {
            "frame_ids": np.arange(10, 15),
            "normalized_camera_tokens": global_tokens,
            "pred_c2w_raw": global_pred,
            "gt_c2w_raw": gt,
        }
        windows = []
        for index, start in enumerate((0, 1)):
            stop = start + 4
            windows.append(
                {
                    "index": index,
                    "start": start,
                    "stop": stop,
                    "artifact": {
                        "frame_ids": np.arange(10 + start, 10 + stop),
                        "normalized_camera_tokens": np.tile(
                            np.array([[1.0, 0.0]]), (4, 1)
                        ),
                        "pred_c2w_raw": gt[start:stop],
                        "gt_c2w_raw": gt[start:stop],
                    },
                }
            )

        observations, overlaps, score_rows, validation_rows = build_scene_rows(
            "scene", global_artifact, windows
        )

        self.assertEqual(len(observations), 8)
        self.assertEqual(len(overlaps), 3)
        self.assertEqual(len(score_rows), 5)
        self.assertEqual(len(validation_rows), 5)
        self.assertNotIn("gt", " ".join(score_rows[0]))
        self.assertNotIn("error", " ".join(score_rows[0]))
        self.assertIn("global_translation_error_aligned", validation_rows[0])
        frame_12_score = next(row for row in score_rows if row["frame_id"] == 12)
        frame_12_validation = next(
            row for row in validation_rows if row["frame_id"] == 12
        )
        self.assertGreater(frame_12_score["global_local_token_cosine"], 0.9)
        self.assertGreater(
            frame_12_validation["translation_error_growth_global_minus_local"],
            0.0,
        )

    def test_thresholds_use_only_named_stable_controls(self):
        rows = [
            {
                "scene": "stable",
                "local_local_token_cosine": value,
                "local_local_pose_translation": value * 2,
                "local_local_pose_rotation_deg": value * 3,
            }
            for value in (0.1, 0.2, 0.3, 0.4)
        ]
        rows.append(
            {
                "scene": "failure",
                "local_local_token_cosine": 99.0,
                "local_local_pose_translation": 99.0,
                "local_local_pose_rotation_deg": 99.0,
            }
        )

        thresholds = fit_reliability_thresholds(rows, stable_scenes={"stable"})

        self.assertLess(thresholds["token_cosine_p95"], 0.5)
        self.assertLess(thresholds["pose_translation_p95"], 1.0)
        self.assertLess(thresholds["pose_rotation_deg_p95"], 2.0)

    def test_prediction_scores_are_invariant_to_raw_gt_changes(self):
        poses = self.make_poses(5)
        global_artifact = {
            "frame_ids": np.arange(10, 15),
            "normalized_camera_tokens": np.tile(np.array([[1.0, 0.0]]), (5, 1)),
            "pred_c2w_raw": poses.copy(),
            "gt_c2w_raw": poses.copy(),
        }
        windows = []
        for index, start in enumerate((0, 1)):
            windows.append(
                {
                    "index": index,
                    "start": start,
                    "stop": start + 4,
                    "artifact": {
                        "frame_ids": np.arange(10 + start, 14 + start),
                        "normalized_camera_tokens": np.tile(
                            np.array([[1.0, 0.0]]), (4, 1)
                        ),
                        "pred_c2w_raw": poses[start : start + 4].copy(),
                        "gt_c2w_raw": poses[start : start + 4].copy(),
                    },
                }
            )
        original_scores = build_scene_rows("scene", global_artifact, windows)[2]
        changed_gt = poses.copy()
        changed_gt[:, 1, 3] += np.linspace(0.0, 3.0, 5) ** 2
        global_artifact["gt_c2w_raw"] = changed_gt
        for record in windows:
            start = int(record["start"])
            record["artifact"]["gt_c2w_raw"] = changed_gt[start : start + 4]

        changed_scores = build_scene_rows("scene", global_artifact, windows)[2]

        self.assertEqual(original_scores, changed_scores)

    def test_summary_correlates_prediction_score_with_separate_label(self):
        score_rows = [
            {
                "scene": "scene",
                "frame_id": index,
                "global_local_token_cosine": float(index),
                "global_local_pose_translation": float(index),
                "global_local_pose_rotation_deg": float(index),
                "local_local_token_cosine": 0.0,
                "local_local_pose_translation": 0.0,
                "local_local_pose_rotation_deg": 0.0,
                "token_local_reliable": True,
                "pose_local_reliable": True,
            }
            for index in range(1, 6)
        ]
        validation_rows = [
            {
                "scene": "scene",
                "frame_id": index,
                "translation_error_growth_global_minus_local": float(index * 2),
                "rotation_error_growth_global_minus_local_deg": float(index * 3),
            }
            for index in range(1, 6)
        ]

        rows = summarize_scores(score_rows, validation_rows)

        token = next(row for row in rows if row["score"] == "global_local_token_cosine")
        self.assertAlmostEqual(token["translation_growth_spearman"], 1.0)
        self.assertAlmostEqual(token["translation_growth_pearson"], 1.0)


class LocalGlobalAnalysisTest(unittest.TestCase):
    def test_complete_analysis_requires_every_declared_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            run_dir = root / "run"
            scene = "scene"
            poses = np.tile(np.eye(4, dtype=np.float64), (5, 1, 1))
            poses[:, 0, 3] = np.arange(5, dtype=np.float64)
            global_artifact = build_window_diagnostics(
                frame_ids=np.arange(10, 15),
                normalized_camera_tokens=np.tile(np.array([[1.0, 0.0]]), (5, 1)),
                pred_w2c=np.linalg.inv(poses),
                gt_c2w_raw=poses,
            )
            global_artifact.update(
                {
                    "pred_c2w_aligned": poses.copy(),
                    "translation_error_aligned": np.zeros(5),
                    "rotation_error_deg_aligned": np.zeros(5),
                    "delta_norm": np.zeros(5),
                    "sim3_scale": np.array(1.0),
                    "sim3_rotation": np.eye(3),
                    "sim3_translation": np.zeros(3),
                }
            )
            global_dir = source / scene / "frames_500"
            global_dir.mkdir(parents=True)
            np.savez_compressed(global_dir / "context_diagnostics.npz", **global_artifact)
            run_dir.mkdir()
            (run_dir / "run_metadata.json").write_text(
                '{"run_id":"run","invocation":{"source_run_dir":"'
                + source.as_posix()
                + '","scenes":["scene"],"window_length":4,"window_stride":1}}\n',
                encoding="utf-8",
            )
            (run_dir / "complete.json").write_text(
                '{"run_id":"run","analysis_complete":true}\n', encoding="utf-8"
            )
            for index, start in enumerate((0, 1)):
                directory = run_dir / scene / f"window_{index:03d}"
                directory.mkdir(parents=True)
                local = build_window_diagnostics(
                    frame_ids=np.arange(10 + start, 14 + start),
                    normalized_camera_tokens=np.tile(
                        np.array([[1.0, 0.0]]), (4, 1)
                    ),
                    pred_w2c=np.linalg.inv(poses[start : start + 4]),
                    gt_c2w_raw=poses[start : start + 4],
                )
                np.savez_compressed(directory / "window_diagnostics.npz", **local)
                (directory / "complete.json").write_text(
                    '{"run_id":"run","window_index":'
                    + str(index)
                    + ',"start":'
                    + str(start)
                    + ',"stop":'
                    + str(start + 4)
                    + "}\n",
                    encoding="utf-8",
                )

            missing = run_dir / scene / "window_001"
            renamed = run_dir / scene / "omitted_window"
            missing.rename(renamed)
            with self.assertRaisesRegex(ValueError, "window set"):
                write_analysis(run_dir, stable_scenes=set())
            invalidated = (run_dir / "complete.json").read_text(encoding="utf-8")
            self.assertIn('"analysis_complete": false', invalidated)
            renamed.rename(missing)

            completion = write_analysis(run_dir, stable_scenes=set())

            self.assertTrue(completion["analysis_complete"])
            self.assertEqual(completion["window_count"], 2)
            score_header = (run_dir / "prediction_scores_per_frame.csv").read_text(
                encoding="utf-8"
            ).splitlines()[0]
            self.assertNotIn("gt", score_header)
            validation_header = (run_dir / "gt_validation_per_frame.csv").read_text(
                encoding="utf-8"
            ).splitlines()[0]
            self.assertIn("error_aligned", validation_header)
            self.assertNotIn("gt_aligned", validation_header)


if __name__ == "__main__":
    unittest.main()
