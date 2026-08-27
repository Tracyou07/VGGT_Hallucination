from __future__ import annotations

import unittest

import torch

from pre_experiments.variational_camera_selector.loss import listwise_quality_loss
from pre_experiments.variational_camera_selector.model import (
    CandidateRanker,
    summarize_sequence,
)


class CandidateRankerTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.batch = 2
        self.choices = 5
        self.global_tokens = torch.randn(self.batch, 7, 2048)
        self.x0 = torch.randn(self.batch, 4, 2048)
        self.delta = torch.randn(self.batch, self.choices, 4, 2048) * 0.05
        self.alpha = torch.tensor(
            [[0.0, 0.01, 0.1, 0.5, 1.0], [0.0, 0.02, 0.2, 0.5, 1.0]]
        )
        self.span_starts = torch.tensor([0, 350], dtype=torch.long)
        self.z = torch.randn(self.batch, self.choices, 2)
        self.full_model = CandidateRanker(
            d_model=8, z_dim=2, include_global_context=True
        )
        self.residual_model = CandidateRanker(
            d_model=8, z_dim=2, include_global_context=False
        )

    @property
    def candidate_inputs(self):
        return self.x0, self.delta, self.alpha, self.span_starts, self.z

    def test_ranker_returns_one_finite_score_per_choice(self) -> None:
        scores = self.full_model(self.global_tokens, *self.candidate_inputs)

        self.assertEqual(scores.shape, (2, 5))
        self.assertTrue(torch.isfinite(scores).all())

    def test_residual_only_ranker_is_invariant_to_global_context(self) -> None:
        first = self.residual_model(self.global_tokens, *self.candidate_inputs)
        second = self.residual_model(
            self.global_tokens + 100.0, *self.candidate_inputs
        )

        torch.testing.assert_close(first, second, atol=0.0, rtol=0.0)

    def test_backward_reaches_shared_projector_and_head(self) -> None:
        scores = self.full_model(self.global_tokens, *self.candidate_inputs)
        scores.square().mean().backward()

        self.assertIsNotNone(self.full_model.token_projector.weight.grad)
        self.assertTrue(torch.isfinite(self.full_model.token_projector.weight.grad).all())
        self.assertTrue(
            all(
                parameter.grad is None or torch.isfinite(parameter.grad).all()
                for parameter in self.full_model.parameters()
            )
        )

    def test_summarize_sequence_preserves_temporal_statistics(self) -> None:
        sequence = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 8.0]]])
        summary = summarize_sequence(sequence)

        self.assertEqual(summary.shape, (1, 8))
        torch.testing.assert_close(summary[..., :2], sequence.mean(dim=-2))
        torch.testing.assert_close(summary[..., 4:6], torch.tensor([[2.0, 3.0]]))

    def test_listwise_target_prefers_higher_utility_without_hard_one_hot(self) -> None:
        scores = torch.zeros(1, 3, requires_grad=True)
        utilities = torch.tensor([[0.0, 0.10, 0.09]])
        loss, target = listwise_quality_loss(
            scores, utilities, tau=0.05, return_target=True
        )

        self.assertGreater(float(target[0, 1]), float(target[0, 2]))
        self.assertGreater(float(target[0, 2]), 0.0)
        self.assertLess(float(target[0, 1]), 1.0)
        loss.backward()
        self.assertTrue(torch.isfinite(scores.grad).all())

    def test_listwise_loss_rejects_nonfinite_utility_and_bad_tau(self) -> None:
        with self.assertRaises(ValueError):
            listwise_quality_loss(torch.zeros(1, 2), torch.zeros(1, 2), tau=0.0)
        with self.assertRaises(ValueError):
            listwise_quality_loss(
                torch.zeros(1, 2), torch.tensor([[0.0, float("nan")]]), tau=0.05
            )


if __name__ == "__main__":
    unittest.main()
