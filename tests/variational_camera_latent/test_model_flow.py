from __future__ import annotations

import inspect
import unittest

import torch
from torch import nn

from pre_experiments.variational_camera_latent.flow import (
    heun_sample,
    make_training_pairs,
    vrfm_loss,
)
from pre_experiments.variational_camera_latent.model import (
    DeterministicRFMModel,
    RecognitionPosterior,
    VRFMModel,
)


class RecordingVelocity(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.z_calls: list[torch.Tensor] = []

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        z: torch.Tensor,
        context: torch.Tensor,
        span: torch.Tensor,
    ) -> torch.Tensor:
        self.z_calls.append(z.detach().clone())
        return torch.ones_like(x_t)


class ModelFlowTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(5)
        self.x = torch.randn(2, 4, 2048)
        self.context = torch.randn(2, 8, 2048)
        self.t = torch.tensor([0.2, 0.8])
        self.z = torch.randn(2, 4)
        self.span = torch.tensor([0, 50])

    def test_vrfm_shapes_and_single_segment_z(self) -> None:
        model = VRFMModel(d_model=32, z_dim=4, layers=1, heads=4)

        output = model(self.x, self.t, self.z, self.context, self.span)

        self.assertEqual(output.shape, self.x.shape)
        self.assertTrue(torch.isfinite(output).all())

    def test_heun_reuses_one_z_for_all_steps(self) -> None:
        recorder = RecordingVelocity()

        output = heun_sample(
            recorder,
            self.x,
            self.context,
            self.span,
            self.z,
            steps=4,
        )

        self.assertEqual(len(recorder.z_calls), 8)
        self.assertTrue(all(torch.equal(call, self.z) for call in recorder.z_calls))
        torch.testing.assert_close(output, self.x + 1.0)

    def test_left_and_right_pairs_are_equal_weight(self) -> None:
        long = torch.zeros(1, 4, 2048)
        left = torch.ones(1, 4, 2048)
        right = torch.full((1, 4, 2048), 2.0)

        batch = make_training_pairs(
            long,
            left,
            right,
            context=torch.zeros(1, 8, 2048),
            span_starts=torch.tensor([100]),
        )

        self.assertEqual(batch.endpoint_side.tolist(), [0, 1])
        self.assertEqual(batch.weights.tolist(), [1.0, 1.0])
        self.assertTrue(torch.equal(batch.x1[0], left[0]))
        self.assertTrue(torch.equal(batch.x1[1], right[0]))

    def test_vrfm_loss_warms_kl_without_weighting_endpoint_side(self) -> None:
        batch = make_training_pairs(
            self.x,
            self.x + 0.5,
            self.x - 0.5,
            context=self.context,
            span_starts=self.span,
        )
        model = VRFMModel(d_model=32, z_dim=4, layers=1, heads=4)
        posterior = RecognitionPosterior(d_model=32, z_dim=4)

        early = vrfm_loss(model, posterior, batch, progress=0.1, beta_max=1e-4)
        late = vrfm_loss(model, posterior, batch, progress=1.0, beta_max=1e-4)

        self.assertAlmostEqual(early.beta, 5e-5)
        self.assertAlmostEqual(late.beta, 1e-4)
        self.assertTrue(torch.isfinite(early.total))

    def test_deterministic_baseline_has_no_z_input(self) -> None:
        model = DeterministicRFMModel(d_model=32, layers=1, heads=4)
        self.assertNotIn("z", inspect.signature(model.forward).parameters)
        self.assertEqual(
            model(self.x, self.t, self.context, self.span).shape,
            self.x.shape,
        )


if __name__ == "__main__":
    unittest.main()
