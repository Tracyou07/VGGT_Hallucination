import json
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
from pre_experiments.local_global_consistency.analyze import (
    collect_run_rows,
    write_analysis,
)
from pre_experiments.local_global_consistency.run_study import (
    _run_id,
    configure_camera_only,
    parse_args,
    run_window,
    window_is_complete,
)
from pre_experiments.local_global_consistency.metrics import (
    build_scene_rows,
    summarize_scores,
)
from pre_experiments.local_global_consistency.thresholds import (
    fit_frozen_thresholds,
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

    def test_parser_requires_partitioned_scannet50_inputs(self):
        args = parse_args(
            [
                "--source-run-dir",
                "source",
                "--split-manifest",
                "split.json",
                "--partition",
                "calibration",
            ]
        )
        self.assertEqual(args.window_length, 100)
        self.assertEqual(args.window_stride, 50)
        self.assertEqual(args.camera_iterations, 4)
        self.assertEqual(args.scene_limit, 0)
        self.assertEqual(args.partition, "calibration")
        with self.assertRaises(SystemExit):
            parse_args([])

    def test_partition_split_and_source_are_part_of_run_id(self):
        base = {
            "source_run_id": "source-a",
            "split_digest": "a" * 64,
            "partition": "calibration",
        }
        calibration = _run_id("1" * 40, base)
        self.assertNotEqual(
            calibration,
            _run_id("1" * 40, {**base, "partition": "holdout"}),
        )
        self.assertNotEqual(
            calibration,
            _run_id("1" * 40, {**base, "split_digest": "b" * 64}),
        )
        self.assertNotEqual(
            calibration,
            _run_id("1" * 40, {**base, "source_run_id": "source-b"}),
        )

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
                '{"run_id":"run","partition":"calibration",'
                '"split_digest":"digest","scene":"scene","window_index":0,'
                '"start":0,"stop":2,"frame_ids":[10,20]}\n',
                encoding="utf-8",
            )

            self.assertTrue(
                window_is_complete(
                    directory,
                    run_id="run",
                    partition="calibration",
                    split_digest="digest",
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
                    partition="calibration",
                    split_digest="digest",
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
                    partition="calibration",
                    split_digest="digest",
                    scene="scene",
                    window_index=0,
                    start=0,
                    stop=2,
                    frame_ids=[10, 20],
                )
            )

    def test_completion_requires_partition_and_split_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            poses = np.tile(np.eye(4, dtype=np.float64), (2, 1, 1))
            np.savez_compressed(
                directory / "window_diagnostics.npz",
                frame_ids=np.array([10, 20]),
                normalized_camera_tokens=np.ones((2, 4)),
                pred_c2w_raw=poses,
                gt_c2w_raw=poses,
            )
            (directory / "complete.json").write_text(
                '{"run_id":"run","partition":"calibration",'
                '"split_digest":"digest","scene":"scene","window_index":0,'
                '"start":0,"stop":2,"frame_ids":[10,20]}\n',
                encoding="utf-8",
            )

            common = {
                "directory": directory,
                "run_id": "run",
                "scene": "scene",
                "window_index": 0,
                "start": 0,
                "stop": 2,
                "frame_ids": [10, 20],
            }
            self.assertTrue(
                window_is_complete(
                    partition="calibration", split_digest="digest", **common
                )
            )
            self.assertFalse(
                window_is_complete(
                    partition="holdout", split_digest="digest", **common
                )
            )
            self.assertFalse(
                window_is_complete(
                    partition="calibration", split_digest="changed", **common
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
                    partition="calibration",
                    split_digest="digest",
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
            self.assertEqual(completion["partition"], "calibration")
            self.assertEqual(completion["split_digest"], "digest")


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
                    '{"run_id":"run","scene":"scene","window_index":'
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
                collect_run_rows(run_dir)
            renamed.rename(missing)

            collected = collect_run_rows(run_dir)

            self.assertEqual(collected["window_count"], 2)
            self.assertEqual(collected["expected_window_count"], 2)
            self.assertNotIn("gt", " ".join(collected["scores"][0]))
            self.assertIn("global_translation_error_aligned", collected["validation"][0])

    def test_calibration_analysis_requires_complete_partition_and_freezes_thresholds(self):
        scenes = [f"scene{index:04d}_00" for index in range(10)]
        score_rows = []
        validation_rows = []
        for scene_index, scene in enumerate(scenes):
            for frame_id in range(3):
                value = float(scene_index + frame_id + 1)
                score_rows.append(
                    {
                        "scene": scene,
                        "frame_id": frame_id,
                        "global_local_token_cosine": value,
                        "global_local_pose_translation": value,
                        "global_local_pose_rotation_deg": value,
                        "local_local_token_cosine": value,
                        "local_local_pose_translation": value,
                        "local_local_pose_rotation_deg": value,
                    }
                )
                validation_rows.append(
                    {
                        "scene": scene,
                        "frame_id": frame_id,
                        "translation_error_growth_global_minus_local": value,
                        "rotation_error_growth_global_minus_local_deg": value,
                    }
                )
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            metadata = {
                "run_id": "calibration-run",
                "git_commit": "b" * 40,
                "source_run_id": "source-run",
                "split_digest": "a" * 64,
                "partition": "calibration",
                "protocol_complete": True,
                "invocation": {
                    "scenes": scenes,
                    "partition_scenes": scenes,
                    "source_run_id": "source-run",
                    "split_digest": "a" * 64,
                    "partition": "calibration",
                    "protocol_complete": True,
                    "window_length": 100,
                    "window_stride": 50,
                    "camera_iterations": 4,
                    "preprocess_mode": "pad",
                },
            }
            (run_dir / "run_metadata.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )
            collected = {
                "observations": [],
                "overlaps": [],
                "scores": score_rows,
                "validation": validation_rows,
                "window_count": 89,
                "expected_window_count": 89,
            }
            with mock.patch(
                "pre_experiments.local_global_consistency.analyze.collect_run_rows",
                return_value=collected,
            ):
                completion = write_analysis(run_dir, mode="calibration")

            self.assertTrue(completion["analysis_complete"])
            self.assertTrue(
                (run_dir / "frozen_reliability_thresholds.json").is_file()
            )
            self.assertTrue(
                (run_dir / "calibration_prediction_scores_per_frame.csv").is_file()
            )
            self.assertTrue(
                (run_dir / "calibration_gt_validation_per_frame.csv").is_file()
            )
            threshold_payload = json.loads(
                (run_dir / "frozen_reliability_thresholds.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                completion["threshold_digest"],
                threshold_payload["threshold_digest"],
            )
            self.assertEqual(completion["window_count"], 89)
            self.assertEqual(completion["expected_window_count"], 89)

            collected["window_count"] = 90
            with mock.patch(
                "pre_experiments.local_global_consistency.analyze.collect_run_rows",
                return_value=collected,
            ):
                with self.assertRaisesRegex(ValueError, "expected window"):
                    write_analysis(run_dir, mode="calibration")
            collected["window_count"] = 89

            metadata["protocol_complete"] = False
            (run_dir / "run_metadata.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "smoke"):
                write_analysis(run_dir, mode="calibration")

    def test_calibration_analysis_rejects_protocol_metadata_tampering(self):
        scenes = [f"scene{index:04d}_00" for index in range(10)]
        base = {
            "run_id": "calibration-run",
            "git_commit": "b" * 40,
            "source_run_id": "source-run",
            "split_digest": "a" * 64,
            "partition": "calibration",
            "protocol_complete": True,
            "invocation": {
                "scenes": scenes,
                "partition_scenes": scenes,
                "source_run_id": "source-run",
                "split_digest": "a" * 64,
                "partition": "calibration",
                "protocol_complete": True,
                "window_length": 100,
                "window_stride": 50,
                "camera_iterations": 4,
                "preprocess_mode": "pad",
            },
        }
        changes = {
            "window_length": 99,
            "window_stride": 49,
            "camera_iterations": 3,
            "preprocess_mode": "crop",
            "split_digest": "c" * 64,
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run_metadata.json"
            for field, value in changes.items():
                with self.subTest(field=field):
                    metadata = json.loads(json.dumps(base))
                    metadata["invocation"][field] = value
                    path.write_text(json.dumps(metadata), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        write_analysis(Path(tmp), mode="calibration")

    def test_holdout_analysis_uses_frozen_thresholds_without_refitting(self):
        calibration_scenes = [f"calibration{index:02d}" for index in range(10)]
        holdout_scenes = [f"holdout{index:02d}" for index in range(40)]
        threshold_rows = [
            {
                "scene": scene,
                "frame_id": 0,
                "local_local_token_cosine": float(index + 1),
                "local_local_pose_translation": float(index + 1),
                "local_local_pose_rotation_deg": float(index + 1),
            }
            for index, scene in enumerate(calibration_scenes)
        ]
        threshold_payload = fit_frozen_thresholds(
            threshold_rows,
            {
                "calibration_scenes": calibration_scenes,
                "source_run_id": "source-run",
                "calibration_run_id": "calibration-run",
                "split_digest": "a" * 64,
                "code_commit": "b" * 40,
            },
        )
        scores = []
        validation = []
        for scene_index, scene in enumerate(holdout_scenes):
            for frame_id in range(3):
                value = float(scene_index + frame_id + 1)
                scores.append(
                    {
                        "scene": scene,
                        "frame_id": frame_id,
                        "global_local_token_cosine": value,
                        "global_local_pose_translation": value,
                        "global_local_pose_rotation_deg": value,
                        "local_local_token_cosine": value,
                        "local_local_pose_translation": value,
                        "local_local_pose_rotation_deg": value,
                    }
                )
                validation.append(
                    {
                        "scene": scene,
                        "frame_id": frame_id,
                        "translation_error_growth_global_minus_local": value,
                        "rotation_error_growth_global_minus_local_deg": value,
                    }
                )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            metadata = {
                "run_id": "holdout-run",
                "git_commit": "b" * 40,
                "source_run_id": "source-run",
                "split_digest": "a" * 64,
                "partition": "holdout",
                "protocol_complete": True,
                "invocation": {
                    "scenes": holdout_scenes,
                    "partition_scenes": holdout_scenes,
                    "source_run_id": "source-run",
                    "split_digest": "a" * 64,
                    "partition": "holdout",
                    "protocol_complete": True,
                    "window_length": 100,
                    "window_stride": 50,
                    "camera_iterations": 4,
                    "preprocess_mode": "pad",
                },
            }
            (run_dir / "run_metadata.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )
            thresholds_path = run_dir / "external_thresholds.json"
            thresholds_path.write_text(
                json.dumps(threshold_payload, sort_keys=True), encoding="utf-8"
            )
            original_threshold_bytes = thresholds_path.read_bytes()
            collected = {
                "observations": [],
                "overlaps": [],
                "scores": scores,
                "validation": validation,
                "window_count": 359,
                "expected_window_count": 359,
            }
            with mock.patch(
                "pre_experiments.local_global_consistency.analyze.collect_run_rows",
                return_value=collected,
            ), mock.patch(
                "pre_experiments.local_global_consistency.analyze.fit_frozen_thresholds",
                side_effect=AssertionError("holdout attempted threshold fitting"),
            ):
                completion = write_analysis(
                    run_dir,
                    mode="holdout",
                    thresholds_path=thresholds_path,
                )

            self.assertEqual(thresholds_path.read_bytes(), original_threshold_bytes)
            self.assertTrue(completion["analysis_complete"])
            self.assertEqual(
                completion["threshold_digest"],
                threshold_payload["threshold_digest"],
            )
            self.assertEqual(completion["window_count"], 359)
            self.assertEqual(completion["expected_window_count"], 359)
            for filename in (
                "holdout_prediction_scores_per_frame.csv",
                "holdout_gt_validation_per_frame.csv",
                "holdout_per_scene_summary.csv",
                "holdout_aggregate_summary.csv",
                "holdout_aggregate_summary.json",
                "holdout_complete.json",
            ):
                self.assertTrue((run_dir / filename).is_file(), filename)

            collected["window_count"] = 360
            with mock.patch(
                "pre_experiments.local_global_consistency.analyze.collect_run_rows",
                return_value=collected,
            ):
                with self.assertRaisesRegex(ValueError, "expected window"):
                    write_analysis(
                        run_dir,
                        mode="holdout",
                        thresholds_path=thresholds_path,
                    )

    def test_holdout_analysis_rejects_calibration_scene_overlap(self):
        calibration_scenes = [f"calibration{index:02d}" for index in range(10)]
        threshold_payload = fit_frozen_thresholds(
            [
                {
                    "scene": scene,
                    "local_local_token_cosine": 1.0,
                    "local_local_pose_translation": 1.0,
                    "local_local_pose_rotation_deg": 1.0,
                }
                for scene in calibration_scenes
            ],
            {
                "calibration_scenes": calibration_scenes,
                "source_run_id": "source-run",
                "calibration_run_id": "calibration-run",
                "split_digest": "a" * 64,
                "code_commit": "b" * 40,
            },
        )
        holdout_scenes = [calibration_scenes[0]] + [
            f"holdout{index:02d}" for index in range(39)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            metadata = {
                "run_id": "holdout-run",
                "source_run_id": "source-run",
                "split_digest": "a" * 64,
                "partition": "holdout",
                "protocol_complete": True,
                "invocation": {
                    "scenes": holdout_scenes,
                    "partition_scenes": holdout_scenes,
                    "source_run_id": "source-run",
                    "split_digest": "a" * 64,
                    "partition": "holdout",
                    "protocol_complete": True,
                    "window_length": 100,
                    "window_stride": 50,
                    "camera_iterations": 4,
                    "preprocess_mode": "pad",
                },
            }
            (run_dir / "run_metadata.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )
            thresholds_path = run_dir / "thresholds.json"
            thresholds_path.write_text(json.dumps(threshold_payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "overlap"):
                write_analysis(
                    run_dir,
                    mode="holdout",
                    thresholds_path=thresholds_path,
                )


if __name__ == "__main__":
    unittest.main()
