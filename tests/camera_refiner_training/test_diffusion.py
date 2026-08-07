import unittest

import torch
from torch import nn

from pre_experiments.camera_refiner_training.diffusion import (
    DiffusionSchedule,
    ddim_sample,
    q_sample,
)


class ZeroDenoiser(nn.Module):
    def forward(self, noisy, condition, timestep):
        return torch.zeros_like(noisy), torch.full_like(noisy[..., :1], 0.25)


class DiffusionTest(unittest.TestCase):
    def test_cosine_schedule_and_forward_noise_are_finite(self):
        schedule = DiffusionSchedule.cosine(100)
        clean = torch.ones(2, 5, 3)
        noise = torch.zeros_like(clean)

        sample = q_sample(clean, torch.tensor([0, 99]), noise, schedule)

        self.assertEqual(sample.shape, clean.shape)
        self.assertTrue(torch.isfinite(sample).all())
        self.assertGreater(schedule.alpha_cumprod[0], schedule.alpha_cumprod[-1])
        with self.assertRaisesRegex(ValueError, "timestep"):
            q_sample(clean, torch.tensor([100, 0]), noise, schedule)

    def test_ddim_is_deterministic_and_returns_final_confidence(self):
        schedule = DiffusionSchedule.cosine(20)
        condition = torch.zeros(2, 8, 4)
        model = ZeroDenoiser()

        first = ddim_sample(
            model,
            condition,
            schedule,
            sample_steps=5,
            generator=torch.Generator().manual_seed(7),
        )
        second = ddim_sample(
            model,
            condition,
            schedule,
            sample_steps=5,
            generator=torch.Generator().manual_seed(7),
        )

        torch.testing.assert_close(first[0], second[0])
        torch.testing.assert_close(first[0], torch.zeros_like(first[0]))
        torch.testing.assert_close(first[1], torch.full_like(first[1], 0.25))


if __name__ == "__main__":
    unittest.main()
