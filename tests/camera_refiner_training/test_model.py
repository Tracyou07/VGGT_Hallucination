import unittest

import torch

from pre_experiments.camera_refiner_training.model import ModelConfig, ResidualDiT


class ResidualDiTTest(unittest.TestCase):
    def test_forward_is_shape_stable_and_zero_initialized(self):
        config = ModelConfig(
            condition_dim=19,
            hidden_size=32,
            depth=2,
            num_heads=4,
            max_frames=100,
        )
        model = ResidualDiT(config)
        noisy = torch.randn(2, 100, 3)
        condition = torch.randn(2, 100, 19)
        timestep = torch.tensor([0, 50])

        clean, confidence = model(noisy, condition, timestep)

        self.assertEqual(clean.shape, (2, 100, 3))
        self.assertEqual(confidence.shape, (2, 100, 1))
        torch.testing.assert_close(clean, torch.zeros_like(clean))
        torch.testing.assert_close(confidence, torch.full_like(confidence, 0.5))

    def test_forward_rejects_mismatched_inputs(self):
        model = ResidualDiT(ModelConfig(condition_dim=5, hidden_size=16, depth=1, num_heads=4, max_frames=8))
        with self.assertRaisesRegex(ValueError, "condition"):
            model(torch.zeros(1, 8, 3), torch.zeros(1, 7, 5), torch.zeros(1, dtype=torch.long))
        with self.assertRaisesRegex(ValueError, "timestep"):
            model(torch.zeros(1, 8, 3), torch.zeros(1, 8, 5), torch.zeros(2, dtype=torch.long))


if __name__ == "__main__":
    unittest.main()
