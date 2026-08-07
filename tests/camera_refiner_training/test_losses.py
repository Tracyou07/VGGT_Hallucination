import unittest

import torch

from pre_experiments.camera_refiner_training.losses import LossWeights, training_losses


class TrainingLossTest(unittest.TestCase):
    def test_exact_target_has_zero_supervised_losses(self):
        target = torch.randn(2, 30, 3)
        global_centers = torch.randn(2, 30, 3)
        losses = training_losses(
            target,
            torch.ones(2, 30, 1),
            target,
            global_centers,
            scene_ids=("a", "b"),
            starts=torch.tensor([0, 0]),
            weights=LossWeights(gate=0.0),
            lags=(1, 5, 10, 25),
        )

        for name in ("total", "denoising", "center", "relative_motion", "overlap"):
            torch.testing.assert_close(losses[name], torch.zeros_like(losses[name]))

    def test_overlap_compares_same_absolute_frames(self):
        prediction = torch.zeros(2, 4, 3)
        prediction[1, :, 0] = 2.0
        losses = training_losses(
            prediction,
            torch.ones(2, 4, 1),
            torch.zeros_like(prediction),
            torch.zeros_like(prediction),
            scene_ids=("scene", "scene"),
            starts=torch.tensor([0, 2]),
            weights=LossWeights(denoising=0.0, center=0.0, relative_motion=0.0, overlap=1.0, gate=0.0),
            lags=(1,),
        )

        self.assertGreater(float(losses["overlap"]), 0.0)
        torch.testing.assert_close(losses["total"], losses["overlap"])

    def test_invalid_lag_is_rejected(self):
        values = torch.zeros(1, 4, 3)
        with self.assertRaisesRegex(ValueError, "lags"):
            training_losses(
                values,
                torch.ones(1, 4, 1),
                values,
                values,
                scene_ids=("scene",),
                starts=torch.tensor([0]),
                lags=(4,),
            )


if __name__ == "__main__":
    unittest.main()
