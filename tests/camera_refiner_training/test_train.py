from pathlib import Path
import tempfile
import unittest

import torch

from pre_experiments.camera_refiner_training.checkpoint import (
    load_checkpoint,
    run_config_digest,
    save_checkpoint,
)
from pre_experiments.camera_refiner_training.diffusion import DiffusionSchedule
from pre_experiments.camera_refiner_training.model import ModelConfig, ResidualDiT
from pre_experiments.camera_refiner_training.train import (
    fit_condition_stats,
    make_scene_batch,
    train_batch,
    validation_loss,
)


class TrainingTest(unittest.TestCase):
    def test_diffusion_validation_uses_repeatable_forward_noising(self):
        class Recorder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.inputs = []

            def forward(self, noisy, condition, timestep):
                self.inputs.append(noisy.clone())
                return torch.ones_like(noisy), torch.ones((*noisy.shape[:2], 1))

        scene = type(
            "Scene",
            (),
            {
                "scene": "scene_a",
                "condition": torch.zeros(1, 30, 2).numpy(),
                "target_residual": torch.ones(1, 30, 3).numpy(),
                "global_centers": torch.zeros(1, 30, 3).numpy(),
                "starts": torch.tensor([0]).numpy(),
            },
        )()
        model = Recorder()
        schedule = DiffusionSchedule.cosine(20)

        first = validation_loss(
            model, [scene], torch.zeros(2), torch.ones(2), torch.device("cpu"),
            schedule=schedule, model_kind="diffusion", seed=9,
        )
        first_input = model.inputs[-1]
        second = validation_loss(
            model, [scene], torch.zeros(2), torch.ones(2), torch.device("cpu"),
            schedule=schedule, model_kind="diffusion", seed=9,
        )

        self.assertEqual(first, second)
        self.assertFalse(torch.equal(first_input, torch.zeros_like(first_input)))
        torch.testing.assert_close(first_input, model.inputs[-1])

    def test_condition_statistics_use_training_values_only(self):
        scenes = [
            type("Scene", (), {"condition": torch.arange(24).reshape(1, 4, 6).numpy()})(),
            type("Scene", (), {"condition": torch.arange(24, 48).reshape(1, 4, 6).numpy()})(),
        ]

        mean, std = fit_condition_stats(scenes)

        expected = torch.arange(48).reshape(8, 6).float()
        torch.testing.assert_close(mean, expected.mean(dim=0))
        torch.testing.assert_close(std, expected.std(dim=0, unbiased=False))

    def test_scene_batch_normalizes_conditions(self):
        scene = type(
            "Scene",
            (),
            {
                "scene": "scene_a",
                "condition": torch.arange(12).reshape(1, 2, 6).numpy(),
                "target_residual": torch.ones(1, 2, 3).numpy(),
                "global_centers": torch.zeros(1, 2, 3).numpy(),
                "starts": torch.tensor([0]).numpy(),
            },
        )()
        mean = torch.arange(6).float()
        std = torch.full((6,), 6.0)

        batch = make_scene_batch(scene, mean, std, torch.device("cpu"))

        torch.testing.assert_close(
            batch["condition"],
            torch.tensor([[[0.0] * 6, [1.0] * 6]]),
        )
        self.assertEqual(batch["scene_ids"], ("scene_a",))

    def test_one_cpu_batch_updates_zero_initialized_head(self):
        torch.manual_seed(3)
        model = ResidualDiT(ModelConfig(condition_dim=7, hidden_size=16, depth=1, num_heads=4, max_frames=30))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
        batch = {
            "condition": torch.randn(2, 30, 7),
            "target_residual": torch.full((2, 30, 3), 0.2),
            "global_centers": torch.zeros(2, 30, 3),
            "scene_ids": ("scene", "scene"),
            "starts": torch.tensor([0, 15]),
        }
        before = model.residual_head.weight.detach().clone()

        losses = train_batch(
            model,
            optimizer,
            batch,
            DiffusionSchedule.cosine(20),
            model_kind="diffusion",
            generator=torch.Generator().manual_seed(5),
            lags=(1, 5, 10, 25),
        )

        self.assertTrue(torch.isfinite(losses["total"]))
        self.assertFalse(torch.equal(before, model.residual_head.weight))

    def test_checkpoint_round_trip_and_resume_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.pt"
            config = ModelConfig(condition_dim=7, hidden_size=16, depth=1, num_heads=4, max_frames=30)
            model = ResidualDiT(config)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            run_config = {"dataset_digest": "data", "unit_digest": "units", "epochs": 5}
            save_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                epoch=2,
                model_config=config,
                condition_mean=torch.zeros(7),
                condition_std=torch.ones(7),
                run_config=run_config,
            )
            restored = ResidualDiT(config)
            restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)

            state = load_checkpoint(
                path,
                model=restored,
                optimizer=restored_optimizer,
                expected_run_digest=run_config_digest(run_config),
            )

            self.assertEqual(state.epoch, 2)
            self.assertEqual(state.model_config, config)
            torch.testing.assert_close(state.condition_mean, torch.zeros(7))
            with self.assertRaisesRegex(ValueError, "configuration"):
                load_checkpoint(
                    path,
                    model=restored,
                    optimizer=restored_optimizer,
                    expected_run_digest=run_config_digest({**run_config, "epochs": 6}),
                )


if __name__ == "__main__":
    unittest.main()
