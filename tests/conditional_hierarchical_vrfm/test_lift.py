from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch
from torch import nn

from pre_experiments.camera_velocity_ambiguity_02.frozen_oracle import FrozenOracle
from pre_experiments.conditional_hierarchical_vrfm.lift import (
    LiftConfig,
    decode_coefficients,
    latent_lift_loss,
    load_lift_checkpoint,
    optimize_latent_target,
    save_lift_checkpoint,
)


def _identity_oracle() -> FrozenOracle:
    return FrozenOracle(
        scene="scene0000_00", frame_digest="a" * 64, fit_count=500,
        scale=1.0,
        rotation=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        translation=(0.0, 0.0, 0.0), rank=3, condition=1.0,
        transform_digest="b" * 64,
    )


def _raw_to_c2w(raw: torch.Tensor) -> torch.Tensor:
    poses = torch.eye(4, dtype=torch.float32, device=raw.device).repeat(raw.shape[0], raw.shape[1], 1, 1)
    poses[..., :3, 3] = raw[..., 6:9]
    return poses


class _FakeHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(1.0))
        self.register_buffer("counter", torch.zeros(()))

    def decode_pose_tokens(self, tokens: torch.Tensor, *, num_iterations: int) -> list[torch.Tensor]:
        self.counter.add_(1)
        return [tokens[..., :9] + self.anchor * 0.0]


class LiftTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.head = _FakeHead()
        self.long_tokens = torch.zeros(1, 500, 2048, dtype=torch.float32)
        self.coverage = torch.zeros(500, dtype=torch.float32)
        self.coverage[100:400] = 1.0
        self.oracle = _identity_oracle()
        self.baseline = _raw_to_c2w(self.long_tokens[..., :9])
        self.teacher = self.baseline.clone()
        self.teacher[:, 100:400, 0, 3] = 0.5
        self.teacher[:, self.coverage == 0] = torch.nan

    def test_zero_coefficients_reproduce_baseline_exactly(self) -> None:
        with mock.patch("pre_experiments.conditional_hierarchical_vrfm.lift.pose_encoding_to_c2w", side_effect=_raw_to_c2w):
            decoded = decode_coefficients(self.head, self.long_tokens, torch.zeros(1, 32, 2048))
        torch.testing.assert_close(decoded, self.baseline, atol=0.0, rtol=0.0)

    def test_nan_teacher_gaps_are_masked_before_loss_arithmetic(self) -> None:
        corrected = self.baseline.clone()
        corrected[:, 100:400, 0, 3] = 0.25
        corrected[:, :100, 0, 3] = 0.1
        with mock.patch("pre_experiments.conditional_hierarchical_vrfm.lift.pose_encoding_to_c2w", side_effect=_raw_to_c2w):
            losses = latent_lift_loss(
                corrected_c2w_raw=corrected,
                baseline_c2w_raw=self.baseline,
                teacher_c2w_gt_gauge=self.teacher,
                coverage_weight=self.coverage,
                oracle=self.oracle,
                residual=torch.zeros(1, 500, 2048),
                config=LiftConfig(max_steps=2),
            )
        self.assertTrue(all(torch.isfinite(value).item() for value in losses.values()))
        self.assertGreater(float(losses["uncovered_center_anchor"]), 0.0)

    def test_optimizer_reduces_loss_and_preserves_head_state(self) -> None:
        parameter_before = self.head.anchor.detach().clone()
        buffer_before = self.head.counter.detach().clone()
        mode_before = self.head.training
        config = LiftConfig(max_steps=20, learning_rate=0.08, smoothness=0.0, residual_norm=0.0)
        with mock.patch("pre_experiments.conditional_hierarchical_vrfm.lift.pose_encoding_to_c2w", side_effect=_raw_to_c2w):
            result = optimize_latent_target(
                self.head, self.long_tokens, self.teacher, self.oracle, config,
                coverage_weight=self.coverage,
            )
        self.assertTrue(result.finite)
        self.assertLess(result.final_loss, result.initial_loss)
        self.assertEqual(result.completed_steps, 20)
        torch.testing.assert_close(self.head.anchor, parameter_before)
        torch.testing.assert_close(self.head.counter, buffer_before)
        self.assertEqual(self.head.training, mode_before)

    def test_resume_is_bitwise_identical_and_rejects_corrupt_binding(self) -> None:
        config = LiftConfig(max_steps=20, learning_rate=0.08, smoothness=0.0, residual_norm=0.0)
        source_sha256 = hashlib.sha256(b"source").hexdigest()
        teacher_sha256 = hashlib.sha256(b"teacher").hexdigest()
        with tempfile.TemporaryDirectory() as directory, mock.patch(
            "pre_experiments.conditional_hierarchical_vrfm.lift.pose_encoding_to_c2w", side_effect=_raw_to_c2w
        ):
            direct = optimize_latent_target(
                self.head, self.long_tokens, self.teacher, self.oracle, config,
                coverage_weight=self.coverage, source_sha256=source_sha256, teacher_sha256=teacher_sha256,
            )
            checkpoint = Path(directory) / "lift.pt"
            first = optimize_latent_target(
                self.head, self.long_tokens, self.teacher, self.oracle,
                LiftConfig(**{**config.__dict__, "max_steps": 8}), coverage_weight=self.coverage,
                checkpoint_path=checkpoint, source_sha256=source_sha256, teacher_sha256=teacher_sha256,
            )
            resumed = optimize_latent_target(
                self.head, self.long_tokens, self.teacher, self.oracle, config,
                coverage_weight=self.coverage, checkpoint_path=checkpoint, resume=True,
                source_sha256=source_sha256, teacher_sha256=teacher_sha256,
            )
            torch.testing.assert_close(direct.coefficients, resumed.coefficients, atol=0.0, rtol=0.0)
            self.assertEqual(direct.loss_trace, resumed.loss_trace)
            self.assertEqual(first.completed_steps, 8)
            payload = load_lift_checkpoint(checkpoint)
            with self.assertRaisesRegex(ValueError, "teacher digest"):
                optimize_latent_target(
                    self.head, self.long_tokens, self.teacher, self.oracle, config,
                    coverage_weight=self.coverage, checkpoint_path=checkpoint, resume=True,
                    source_sha256=source_sha256, teacher_sha256="c" * 64,
                )
            self.assertEqual(payload["next_step"], 20)

    def test_checkpoint_load_rejects_malformed_internal_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.pt"
            path.write_bytes(b"not a checkpoint")
            with self.assertRaisesRegex(ValueError, "checkpoint"):
                load_lift_checkpoint(path)


if __name__ == "__main__":
    unittest.main()
