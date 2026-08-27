from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from pre_experiments.variational_camera_selector.dataset import CandidateGroup
from pre_experiments.variational_camera_selector.train import (
    FROZEN_TRAIN_SCENES,
    FROZEN_VALIDATION_SCENES,
    SelectorTrainConfig,
    train_selectors,
)


class _FakeTrainingDataset:
    def __init__(self, prediction_manifest: Path, privileged_manifest: Path, *, roles):
        if tuple(roles) != ("train",):
            raise AssertionError("training must request only the train role")
        self.scenes = ("scene0000_00",)
        self.roles = ("train",)
        self.groups = [self._group(index) for index in range(8)]

    @staticmethod
    def _group(index: int) -> CandidateGroup:
        rng = np.random.default_rng(100 + index)
        choices = 5
        z = rng.normal(size=(choices, 2)).astype(np.float32)
        return CandidateGroup(
            scene="scene0000_00",
            role="train",
            overlap_index=index,
            sample_id=f"scene0000_00:overlap_{index:03d}",
            span_start=index * 50,
            global_tokens=rng.normal(size=(4, 2048)).astype(np.float32),
            x0=rng.normal(size=(3, 2048)).astype(np.float32),
            delta_tokens=(
                rng.normal(size=(choices, 3, 2048)).astype(np.float32) * 0.01
            ),
            alphas=np.asarray([0.0, 0.01, 0.1, 0.5, 1.0], dtype=np.float32),
            z=z,
            sample_seeds=np.asarray([-1, 1, 2, 3, 4], dtype=np.int64),
            choice_ids=np.asarray(
                [f"scene0000_00:overlap_{index:03d}:choice_{i}" for i in range(choices)],
                dtype="U96",
            ),
            source_sha256="a" * 64,
            candidate_sha256="b" * 64,
            residual_prediction_sha256="c" * 64,
            utilities=np.asarray([0.0, 0.10, 0.09, -0.02, 0.03], dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, index: int) -> CandidateGroup:
        return self.groups[index]


class SelectorTrainingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.prediction_manifest = self.root / "prediction.json"
        self.privileged_manifest = self.root / "labels.json"
        self.prediction_manifest.write_text('{"fixture":"prediction"}\n', encoding="utf-8")
        self.privileged_manifest.write_text('{"fixture":"labels"}\n', encoding="utf-8")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def config(self, run_name: str = "run", **changes) -> SelectorTrainConfig:
        base = SelectorTrainConfig(
            prediction_manifest=self.prediction_manifest,
            privileged_manifest=self.privileged_manifest,
            run_root=self.root / run_name,
            max_steps=1,
            batch_size=2,
            learning_rate=1e-3,
            tau=0.05,
            seed=1234,
            d_model=4,
            device="cpu",
            checkpoint_interval=1,
            git_commit="d" * 40,
            train_scenes=("scene0000_00",),
        )
        return replace(base, **changes)

    @mock.patch(
        "pre_experiments.variational_camera_selector.train.SelectorTrainingDataset",
        _FakeTrainingDataset,
    )
    def test_one_step_trains_both_models_with_finite_loss(self) -> None:
        result = train_selectors(self.config(max_steps=1, device="cpu"))

        self.assertEqual(result.completed_step, 1)
        rows = [
            json.loads(line)
            for line in result.metrics_path.read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(len(rows), 1)
        self.assertTrue(np.isfinite(rows[0]["full_context_loss"]))
        self.assertTrue(np.isfinite(rows[0]["residual_only_loss"]))
        self.assertTrue(result.checkpoint_path.is_file())

    @mock.patch(
        "pre_experiments.variational_camera_selector.train.SelectorTrainingDataset",
        _FakeTrainingDataset,
    )
    def test_resume_starts_at_exact_next_step_and_rejects_changed_config(self) -> None:
        first = train_selectors(self.config(max_steps=2, checkpoint_interval=1))
        second = train_selectors(self.config(max_steps=4, checkpoint_interval=1))

        self.assertEqual((first.start_step, second.start_step), (0, 2))
        self.assertEqual(second.completed_step, 4)
        rows = second.metrics_path.read_text(encoding="utf-8").splitlines()
        self.assertEqual([json.loads(row)["step"] for row in rows], [1, 2, 3, 4])
        with self.assertRaisesRegex(ValueError, "config or input digest"):
            train_selectors(self.config(max_steps=4, tau=0.1))
        self.prediction_manifest.write_text(
            '{"fixture":"tampered-prediction"}\n', encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "config or input digest"):
            train_selectors(self.config(max_steps=4))

    @mock.patch(
        "pre_experiments.variational_camera_selector.train.SelectorTrainingDataset",
        _FakeTrainingDataset,
    )
    def test_resume_matches_uninterrupted_model_state(self) -> None:
        train_selectors(self.config("resumed", max_steps=2))
        resumed = train_selectors(self.config("resumed", max_steps=4))
        uninterrupted = train_selectors(self.config("uninterrupted", max_steps=4))
        resumed_state = torch.load(
            resumed.checkpoint_path, map_location="cpu", weights_only=False
        )
        uninterrupted_state = torch.load(
            uninterrupted.checkpoint_path, map_location="cpu", weights_only=False
        )

        for model_name in ("full_context_model", "residual_only_model"):
            self.assertEqual(
                set(resumed_state[model_name]), set(uninterrupted_state[model_name])
            )
            for name, tensor in resumed_state[model_name].items():
                torch.testing.assert_close(
                    tensor, uninterrupted_state[model_name][name], atol=0.0, rtol=0.0
                )

    def test_frozen_scene_split_is_exact_and_disjoint(self) -> None:
        self.assertEqual(
            FROZEN_TRAIN_SCENES,
            (
                "scene0000_00",
                "scene0013_02",
                "scene0029_01",
                "scene0691_00",
                "scene0084_01",
                "scene0121_01",
                "scene0207_01",
                "scene0280_00",
            ),
        )
        self.assertEqual(FROZEN_VALIDATION_SCENES, ("scene0325_01", "scene0675_00"))
        self.assertTrue(set(FROZEN_TRAIN_SCENES).isdisjoint(FROZEN_VALIDATION_SCENES))


if __name__ == "__main__":
    unittest.main()
