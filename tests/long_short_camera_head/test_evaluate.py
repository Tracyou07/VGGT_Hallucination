from __future__ import annotations

import inspect
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from pre_experiments.long_short_camera_head.evaluate import (
    evaluate_prediction,
    load_prediction,
    run_long_only_inference,
    run_long_only_inference_batch,
)
from pre_experiments.long_short_camera_head.pipeline import run_evaluation
from tests.long_short_camera_head.test_train import TinyPoseHead


class LongOnlyEvaluationTests(unittest.TestCase):
    def test_pipeline_evaluates_all_ten_manifest_scenes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "training" / "gt_only" / "checkpoints" / "best.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"checkpoint")
            records = [
                {
                    "scene": f"scene{i:04d}_00",
                    "role": "train" if i < 8 else "validation",
                    "long_context_path": str(root / f"long{i}.npz"),
                    "privileged_path": str(root / f"privileged{i}.npz"),
                }
                for i in range(10)
            ]

            def fake_batch(
                long_paths,
                checkpoint_path,
                checkpoint_dir,
                destinations,
                device,
            ):
                records = []
                for destination in destinations:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(b"prediction")
                    records.append(SimpleNamespace(path=destination, sha256="a" * 64))
                return tuple(records)

            def fake_evaluate(prediction_path, privileged_path, destination):
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text("{}", encoding="utf-8")
                return SimpleNamespace(path=destination)

            with mock.patch(
                "pre_experiments.long_short_camera_head.pipeline._load_manifest",
                return_value={"records": records},
            ), mock.patch(
                "pre_experiments.long_short_camera_head.pipeline._load_formal_config",
                return_value={},
            ), mock.patch(
                "pre_experiments.long_short_camera_head.pipeline.run_long_only_inference_batch",
                side_effect=fake_batch,
            ) as inference_batch, mock.patch(
                "pre_experiments.long_short_camera_head.pipeline.evaluate_prediction",
                side_effect=fake_evaluate,
            ), mock.patch(
                "pre_experiments.long_short_camera_head.pipeline.sha256_file",
                return_value="b" * 64,
            ):
                completion = run_evaluation(
                    run_root=root,
                    checkpoint_dir=root / "base",
                    variant="gt_only",
                    device=torch.device("cpu"),
                )

            payload = __import__("json").loads(completion.read_text())
            self.assertEqual(inference_batch.call_count, 1)
            self.assertEqual(len(payload["records"]), 10)
            self.assertEqual(
                sum(row["role"] == "locked_replay" for row in payload["records"]),
                2,
            )

    def test_inference_signature_has_no_privileged_or_short_argument(self) -> None:
        names = set(inspect.signature(run_long_only_inference).parameters)
        self.assertFalse(
            names
            & {
                "gt",
                "prepared_root",
                "short_tokens",
                "privileged",
                "privileged_path",
                "model",
            }
        )

    def test_batch_inference_signature_is_checkpoint_bound(self) -> None:
        names = set(inspect.signature(run_long_only_inference_batch).parameters)
        self.assertFalse(names & {"model", "gt", "short_tokens", "privileged"})

    def test_prediction_contract_and_frozen_oracle_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame_ids = np.arange(500, dtype=np.int64)
            long_path = root / "long.npz"
            with long_path.open("wb") as handle:
                np.savez_compressed(
                    handle,
                    scene=np.asarray("scene0000_00", dtype="U32"),
                    frame_ids=frame_ids,
                    camera_tokens=np.zeros((500, 2048), dtype=np.float32),
                    baseline_c2w=np.repeat(np.eye(4)[None], 500, axis=0),
                    source_sha256=np.asarray("a" * 64, dtype="U64"),
                )
            checkpoint = root / "best.pt"
            checkpoint.write_bytes(b"test checkpoint identity")
            prediction_path = root / "prediction.npz"
            model = TinyPoseHead(500)
            with mock.patch(
                "pre_experiments.long_short_camera_head.evaluate.load_camera_head_checkpoint",
                return_value=model,
            ):
                record = run_long_only_inference(
                    long_path,
                    checkpoint,
                    root / "base",
                    prediction_path,
                    torch.device("cpu"),
                )
            prediction = load_prediction(record.path)
            self.assertEqual(
                set(prediction),
                {
                    "scene",
                    "frame_ids",
                    "pose_encoding",
                    "predicted_c2w",
                    "source_sha256",
                    "checkpoint_sha256",
                },
            )

            privileged_path = root / "labels.npz"
            gt = np.repeat(np.eye(4)[None], 500, axis=0)
            gt[:, 0, 3] = np.linspace(0.0, 1.0, 500)
            baseline_pose = np.zeros((500, 9), dtype=np.float32)
            baseline_pose[:, 6] = 1.0
            with privileged_path.open("wb") as handle:
                np.savez_compressed(
                    handle,
                    scene=np.asarray("scene0000_00", dtype="U32"),
                    frame_ids=frame_ids,
                    gt_c2w=gt,
                    oracle_scale=np.asarray(1.0),
                    oracle_rotation=np.eye(3),
                    oracle_translation=np.zeros(3),
                    oracle_digest=np.asarray("c" * 64, dtype="U64"),
                    gt_scene_scale=np.asarray(1.0),
                    baseline_pose_encoding=baseline_pose,
                    teacher_c2w_gt_gauge=np.full((500, 4, 4), np.nan),
                    teacher_weight=np.zeros(500),
                    window_teacher_weight=np.zeros(9),
                    window_baseline_rms=np.ones(9),
                    window_teacher_rms=np.ones(9),
                    source_sha256=np.asarray("a" * 64, dtype="U64"),
                    checkpoint_sha256=np.asarray("d" * 64, dtype="U64"),
                )
            metrics_path = root / "metrics.json"
            evaluated = evaluate_prediction(prediction_path, privileged_path, metrics_path)
            self.assertTrue(metrics_path.is_file())
            self.assertEqual(evaluated.scene, "scene0000_00")
            self.assertAlmostEqual(evaluated.metrics["utility"], 0.0)


if __name__ == "__main__":
    unittest.main()
