from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
from torch import nn

from pre_experiments.long_short_camera_head.train import (
    TrainConfig,
    TrainingExample,
    configure_trainable_scope,
    load_training_example,
    load_training_checkpoint,
    run_training_loop,
    save_training_checkpoint,
    train_camera_head,
)
from vggt.heads.camera_head import CameraHead


class TinyPoseHead(nn.Module):
    def __init__(self, frames: int) -> None:
        super().__init__()
        self.translation = nn.Parameter(torch.zeros(frames))

    def decode_pose_tokens(
        self, tokens: torch.Tensor, *, num_iterations: int
    ) -> list[torch.Tensor]:
        pose = torch.zeros(tokens.shape[0], tokens.shape[1], 9, device=tokens.device)
        pose[..., 0] = self.translation
        pose[..., 6] = 1.0
        return [pose for _ in range(num_iterations)]


class CameraHeadTrainingTests(unittest.TestCase):
    def test_train_camera_head_writes_strict_provenance(self) -> None:
        train_long = self._write_long_context(
            frames=np.arange(500), source="a" * 64, scene="scene0000_00"
        )
        train_labels = self._write_privileged(
            frames=np.arange(500), source="a" * 64, scene="scene0000_00"
        )
        validation_long = self._write_long_context(
            frames=np.arange(500), source="b" * 64, scene="scene0001_00"
        )
        validation_labels = self._write_privileged(
            frames=np.arange(500), source="b" * 64, scene="scene0001_00"
        )
        run_root = Path(self.temporary.name) / "run"
        config = TrainConfig(
            checkpoint_dir=Path(self.temporary.name) / "checkpoint",
            run_root=run_root,
            variant="gt_only",
            train_pairs=((train_long, train_labels),),
            validation_pairs=((validation_long, validation_labels),),
            max_steps=1,
            checkpoint_interval=1,
            patience=1,
            device=torch.device("cpu"),
        )
        with mock.patch(
            "pre_experiments.long_short_camera_head.train.load_base_camera_head",
            return_value=(TinyPoseHead(500), "d" * 64),
        ), mock.patch(
            "pre_experiments.long_short_camera_head.train.configure_trainable_scope",
            return_value=("translation",),
        ):
            result = train_camera_head(config)

        provenance = json.loads((run_root / "training_provenance.json").read_text())
        self.assertEqual(provenance["variant"], "gt_only")
        self.assertEqual(provenance["train_scenes"], ["scene0000_00"])
        self.assertEqual(provenance["validation_scenes"], ["scene0001_00"])
        self.assertEqual(provenance["base_checkpoint_sha256"], "d" * 64)
        self.assertTrue(result.best_checkpoint.is_file())

    def test_training_example_join_requires_exact_scene_frames_and_source(self) -> None:
        frames = np.arange(500, dtype=np.int64)
        long_path = self._write_long_context(frames=frames, source="a" * 64)
        privileged_path = self._write_privileged(frames=frames, source="a" * 64)

        example = load_training_example(long_path, privileged_path)

        self.assertEqual(example.scene, "scene0000_00")
        self.assertEqual(example.tokens.shape, (1, 500, 2048))
        with np.load(privileged_path, allow_pickle=False) as archive:
            changed = {name: np.asarray(archive[name]).copy() for name in archive.files}
        changed["source_sha256"] = np.asarray("b" * 64, dtype="U64")
        with privileged_path.open("wb") as handle:
            np.savez_compressed(handle, **changed)
        with self.assertRaisesRegex(ValueError, "source"):
            load_training_example(long_path, privileged_path)

    def _write_long_context(
        self, *, frames: np.ndarray, source: str, scene: str = "scene0000_00"
    ) -> Path:
        path = Path(self.temporary.name) / f"{scene}.long.npz"
        with path.open("wb") as handle:
            np.savez_compressed(
                handle,
                scene=np.asarray(scene, dtype="U32"),
                frame_ids=frames,
                camera_tokens=np.zeros((500, 2048), dtype=np.float32),
                baseline_c2w=np.repeat(np.eye(4)[None], 500, axis=0),
                source_sha256=np.asarray(source, dtype="U64"),
            )
        return path

    def _write_privileged(
        self, *, frames: np.ndarray, source: str, scene: str = "scene0000_00"
    ) -> Path:
        path = Path(self.temporary.name) / f"{scene}.privileged.npz"
        with path.open("wb") as handle:
            np.savez_compressed(
                handle,
                scene=np.asarray(scene, dtype="U32"),
                frame_ids=frames,
                gt_c2w=np.repeat(np.eye(4)[None], 500, axis=0),
                oracle_scale=np.asarray(1.0),
                oracle_rotation=np.eye(3),
                oracle_translation=np.zeros(3),
                oracle_digest=np.asarray("c" * 64, dtype="U64"),
                gt_scene_scale=np.asarray(1.0),
                baseline_pose_encoding=np.pad(
                    np.ones((500, 1), dtype=np.float32), ((0, 0), (6, 2))
                ),
                teacher_c2w_gt_gauge=np.repeat(np.eye(4)[None], 500, axis=0),
                teacher_weight=np.ones(500),
                window_teacher_weight=np.ones(9),
                window_baseline_rms=np.ones(9),
                window_teacher_rms=np.zeros(9),
                source_sha256=np.asarray(source, dtype="U64"),
                checkpoint_sha256=np.asarray("d" * 64, dtype="U64"),
            )
        return path

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_tiny_training_loop_reduces_loss_and_writes_best_checkpoint(self) -> None:
        frames = 30
        gt = torch.eye(4).repeat(1, frames, 1, 1)
        gt[:, :, 0, 3] = torch.linspace(0.0, 1.0, frames)
        teacher = gt.clone()
        baseline = torch.zeros(1, frames, 9)
        baseline[..., 6] = 1.0
        example = TrainingExample(
            scene="scene0000_00",
            tokens=torch.zeros(1, frames, 2048),
            baseline_pose=baseline,
            gt_c2w=gt,
            oracle_scale=torch.tensor(1.0),
            oracle_rotation=torch.eye(3),
            oracle_translation=torch.zeros(3),
            gt_scene_scale=torch.tensor(1.0),
            teacher_c2w=teacher,
            teacher_weight=torch.ones(1, frames),
        )
        with tempfile.TemporaryDirectory() as directory:
            model = TinyPoseHead(frames)
            result = run_training_loop(
                model=model,
                train_examples=(example,),
                validation_examples=(example,),
                run_root=Path(directory),
                variant="gt_only",
                max_steps=20,
                learning_rate=0.1,
                weight_decay=0.0,
                checkpoint_interval=5,
                patience=20,
                seed=11,
                device=torch.device("cpu"),
                config_digest="a" * 64,
                data_digest="b" * 64,
            )

            self.assertLess(result.final_training_loss, result.initial_training_loss)
            self.assertTrue(result.best_checkpoint.is_file())
            self.assertEqual(result.completed_step, 20)

    def test_only_declared_camera_head_parameters_train(self) -> None:
        head = CameraHead(dim_in=32, trunk_depth=4, num_heads=4, mlp_ratio=1)

        names = configure_trainable_scope(head)

        self.assertTrue(any(name.startswith("trunk.3") for name in names))
        self.assertFalse(any(name.startswith("trunk.0") for name in names))
        self.assertFalse(head.token_norm.weight.requires_grad)
        self.assertTrue(head.pose_branch.fc2.weight.requires_grad)
        expected = {name for name, parameter in head.named_parameters() if parameter.requires_grad}
        self.assertEqual(set(names), expected)

    def test_checkpoint_restores_exact_model_optimizer_and_next_step(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "latest.pt"
            model = nn.Linear(3, 2)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            loss = model(torch.ones(1, 3)).square().mean()
            loss.backward()
            optimizer.step()
            expected = {name: value.detach().clone() for name, value in model.state_dict().items()}
            save_training_checkpoint(
                path,
                step=7,
                model=model,
                optimizer=optimizer,
                config_digest="a" * 64,
                data_digest="b" * 64,
                best_validation_rms=0.25,
            )
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.zero_()

            state = load_training_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                config_digest="a" * 64,
                data_digest="b" * 64,
                device=torch.device("cpu"),
            )

            self.assertEqual(state.step, 7)
            self.assertEqual(state.best_validation_rms, 0.25)
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, expected[name])

    def test_checkpoint_rejects_changed_data_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "latest.pt"
            model = nn.Linear(2, 1)
            optimizer = torch.optim.AdamW(model.parameters())
            save_training_checkpoint(
                path,
                step=1,
                model=model,
                optimizer=optimizer,
                config_digest="a" * 64,
                data_digest="b" * 64,
                best_validation_rms=1.0,
            )
            with self.assertRaisesRegex(ValueError, "digest"):
                load_training_checkpoint(
                    path,
                    model=model,
                    optimizer=optimizer,
                    config_digest="a" * 64,
                    data_digest="c" * 64,
                    device=torch.device("cpu"),
                )


if __name__ == "__main__":
    unittest.main()
