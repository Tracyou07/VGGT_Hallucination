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
        return [tokens[..., :9] + self.anchor * 0.0]


class _UnsafePayload:
    def __init__(self, marker: str) -> None:
        self.marker = marker

    def __reduce__(self) -> tuple[object, tuple[str]]:
        return (Path.write_text, (Path(self.marker), "executed"))


class _DataMutatingBufferHead(_FakeHead):
    def decode_pose_tokens(self, tokens: torch.Tensor, *, num_iterations: int) -> list[torch.Tensor]:
        self.counter.data.add_(1.0)
        return super().decode_pose_tokens(tokens, num_iterations=num_iterations)


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
        self.source_sha256 = hashlib.sha256(b"source").hexdigest()
        self.teacher_sha256 = hashlib.sha256(b"teacher").hexdigest()

    def _checkpoint_fixture(self, directory: str) -> tuple[Path, LiftConfig]:
        config = LiftConfig(max_steps=3, learning_rate=0.08, smoothness=0.0, residual_norm=0.0)
        checkpoint = Path(directory) / "state.pt"
        with mock.patch("pre_experiments.conditional_hierarchical_vrfm.lift.pose_encoding_to_c2w", side_effect=_raw_to_c2w):
            optimize_latent_target(
                self.head, self.long_tokens, self.teacher, self.oracle, config,
                coverage_weight=self.coverage, checkpoint_path=checkpoint,
                source_sha256=self.source_sha256, teacher_sha256=self.teacher_sha256,
            )
        return checkpoint, config

    @staticmethod
    def _rewrite_checkpoint(path: Path, mutate: object) -> None:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        mutate(payload)
        torch.save(payload, path)

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
                coverage_weight=self.coverage, source_sha256=self.source_sha256,
                teacher_sha256=self.teacher_sha256,
            )
        self.assertTrue(result.finite)
        self.assertLess(result.final_loss, result.initial_loss)
        self.assertEqual(result.completed_steps, 20)
        torch.testing.assert_close(self.head.anchor, parameter_before)
        torch.testing.assert_close(self.head.counter, buffer_before)
        self.assertEqual(self.head.training, mode_before)

    def test_resume_is_bitwise_identical_and_rejects_corrupt_binding(self) -> None:
        config = LiftConfig(max_steps=20, learning_rate=0.08, smoothness=0.0, residual_norm=0.0)
        source_sha256 = self.source_sha256
        teacher_sha256 = self.teacher_sha256
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

    def test_resume_rejects_changed_adamw_hyperparameters_before_loading(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint, config = self._checkpoint_fixture(directory)
            self._rewrite_checkpoint(
                checkpoint, lambda payload: payload["optimizer"]["param_groups"][0].__setitem__("lr", 123.0)
            )
            with mock.patch("pre_experiments.conditional_hierarchical_vrfm.lift.pose_encoding_to_c2w", side_effect=_raw_to_c2w):
                with self.assertRaisesRegex(ValueError, "optimizer hyperparameter"):
                    optimize_latent_target(
                        self.head, self.long_tokens, self.teacher, self.oracle, config,
                        coverage_weight=self.coverage, checkpoint_path=checkpoint, resume=True,
                        source_sha256=self.source_sha256, teacher_sha256=self.teacher_sha256,
                    )

    def test_checkpoint_rejects_injected_optimizer_state_and_wrong_tensor_shape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint, _ = self._checkpoint_fixture(directory)
            self._rewrite_checkpoint(
                checkpoint, lambda payload: payload["optimizer"]["state"][0].__setitem__("injected", torch.tensor(1))
            )
            with self.assertRaisesRegex(ValueError, "checkpoint"):
                load_lift_checkpoint(checkpoint)
            checkpoint, _ = self._checkpoint_fixture(directory)
            self._rewrite_checkpoint(
                checkpoint,
                lambda payload: payload["optimizer"]["state"][0].__setitem__("exp_avg", torch.zeros(1)),
            )
            with self.assertRaisesRegex(ValueError, "checkpoint"):
                load_lift_checkpoint(checkpoint)

    def test_checkpoint_rejects_noncanonical_adamw_step_tensor(self) -> None:
        reference_parameter = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        reference_optimizer = torch.optim.AdamW([reference_parameter], lr=1.0, weight_decay=0.0)
        reference_parameter.grad = torch.ones_like(reference_parameter)
        reference_optimizer.step()
        reference_step = reference_optimizer.state_dict()["state"][0]["step"]
        self.assertEqual(reference_step.shape, torch.Size([]))
        self.assertEqual(reference_step.dtype, torch.float32)

        with tempfile.TemporaryDirectory() as directory:
            checkpoint, _ = self._checkpoint_fixture(directory)
            self._rewrite_checkpoint(
                checkpoint,
                lambda payload: payload["optimizer"]["state"][0].__setitem__(
                    "step", torch.tensor([payload["next_step"]], dtype=torch.int64)
                ),
            )
            with self.assertRaisesRegex(ValueError, "checkpoint"):
                load_lift_checkpoint(checkpoint)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required to exercise AdamW restore placement")
    def test_cuda_resume_preserves_installed_adamw_step_placement(self) -> None:
        config = LiftConfig(max_steps=2, learning_rate=0.08, smoothness=0.0, residual_norm=0.0)
        head = _FakeHead().cuda()
        long_tokens = self.long_tokens.cuda()
        teacher = self.teacher.cuda()
        coverage = self.coverage.cuda()
        reference_parameter = nn.Parameter(torch.zeros(1, device="cuda"))
        reference_optimizer = torch.optim.AdamW([reference_parameter], lr=1.0, weight_decay=0.0)
        reference_parameter.grad = torch.ones_like(reference_parameter)
        reference_optimizer.step()
        reference_step = reference_optimizer.state[reference_parameter]["step"]
        observed: list[dict[str, torch.device]] = []
        original_step = torch.optim.AdamW.step

        def observe_state(optimizer: torch.optim.AdamW, *args: object, **kwargs: object) -> object:
            state = next(iter(optimizer.state.values()))
            observed.append({name: value.device for name, value in state.items() if isinstance(value, torch.Tensor)})
            return original_step(optimizer, *args, **kwargs)

        with tempfile.TemporaryDirectory() as directory, mock.patch(
            "pre_experiments.conditional_hierarchical_vrfm.lift.pose_encoding_to_c2w", side_effect=_raw_to_c2w
        ):
            checkpoint = Path(directory) / "cuda-state.pt"
            optimize_latent_target(
                head, long_tokens, teacher, self.oracle,
                LiftConfig(**{**config.__dict__, "max_steps": 1}), coverage_weight=coverage,
                checkpoint_path=checkpoint, source_sha256=self.source_sha256,
                teacher_sha256=self.teacher_sha256,
            )
            with mock.patch.object(torch.optim.AdamW, "step", new=observe_state):
                optimize_latent_target(
                    head, long_tokens, teacher, self.oracle, config, coverage_weight=coverage,
                    checkpoint_path=checkpoint, resume=True, source_sha256=self.source_sha256,
                    teacher_sha256=self.teacher_sha256,
                )
        self.assertEqual(observed[0]["exp_avg"].type, "cuda")
        self.assertEqual(observed[0]["exp_avg_sq"].type, "cuda")
        self.assertEqual(observed[0]["step"], reference_step.device)

    def test_checkpoint_rejects_malformed_cpu_rng_state_before_resume_mutates_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint, config = self._checkpoint_fixture(directory)
            self._rewrite_checkpoint(checkpoint, lambda payload: payload.__setitem__("rng_state", torch.zeros(1, dtype=torch.uint8)))
            with self.assertRaisesRegex(ValueError, "checkpoint"):
                load_lift_checkpoint(checkpoint)
            with mock.patch("pre_experiments.conditional_hierarchical_vrfm.lift.pose_encoding_to_c2w", side_effect=_raw_to_c2w):
                with self.assertRaisesRegex(ValueError, "checkpoint"):
                    optimize_latent_target(
                        self.head, self.long_tokens, self.teacher, self.oracle, config,
                        coverage_weight=self.coverage, checkpoint_path=checkpoint, resume=True,
                        source_sha256=self.source_sha256, teacher_sha256=self.teacher_sha256,
                    )

    def test_checkpoint_rejects_same_shape_invalid_cpu_rng_without_changing_global_rng(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint, _ = self._checkpoint_fixture(directory)
            valid_rng = torch.get_rng_state()
            self._rewrite_checkpoint(checkpoint, lambda payload: payload.__setitem__("rng_state", torch.zeros_like(valid_rng)))
            before = torch.get_rng_state().clone()
            with self.assertRaisesRegex(ValueError, "checkpoint"):
                load_lift_checkpoint(checkpoint)
            torch.testing.assert_close(torch.get_rng_state(), before, atol=0.0, rtol=0.0)

    def test_rng_setter_failure_rolls_back_cpu_and_cuda_state(self) -> None:
        import pre_experiments.conditional_hierarchical_vrfm.lift as lift
        saved_cpu = torch.get_rng_state().clone()
        original_cpu = torch.full_like(saved_cpu, 7)
        saved_cuda = torch.ones(8, dtype=torch.uint8)
        original_cuda = torch.full((8,), 2, dtype=torch.uint8)
        cpu_sets: list[torch.Tensor] = []
        cuda_sets: list[torch.Tensor] = []

        def set_cpu(value: torch.Tensor) -> None:
            cpu_sets.append(value.clone())

        def set_cuda(value: torch.Tensor, *, device: torch.device) -> None:
            cuda_sets.append(value.clone())
            if len(cuda_sets) == 1:
                raise RuntimeError("controlled CUDA setter failure")

        with mock.patch.object(lift.torch, "get_rng_state", return_value=original_cpu), mock.patch.object(
            lift.torch, "set_rng_state", side_effect=set_cpu
        ), mock.patch.object(lift.torch.cuda, "get_rng_state", return_value=original_cuda), mock.patch.object(
            lift.torch.cuda, "set_rng_state", side_effect=set_cuda
        ):
            with self.assertRaisesRegex(ValueError, "RNG state"):
                lift._apply_rng_states_atomically(saved_cpu, saved_cuda, torch.device("cuda", 0))
        torch.testing.assert_close(cpu_sets[0], saved_cpu)
        torch.testing.assert_close(cpu_sets[-1], original_cpu)
        torch.testing.assert_close(cuda_sets[-1], original_cuda)

    def test_nonmonotonic_resume_keeps_the_original_best_state(self) -> None:
        config = LiftConfig(max_steps=20, learning_rate=4.0, smoothness=0.0, residual_norm=0.0)
        with tempfile.TemporaryDirectory() as directory, mock.patch(
            "pre_experiments.conditional_hierarchical_vrfm.lift.pose_encoding_to_c2w", side_effect=_raw_to_c2w
        ):
            direct = optimize_latent_target(
                self.head, self.long_tokens, self.teacher, self.oracle, config,
                coverage_weight=self.coverage, source_sha256=self.source_sha256,
                teacher_sha256=self.teacher_sha256,
            )
            checkpoint = Path(directory) / "nonmonotonic.pt"
            optimize_latent_target(
                self.head, self.long_tokens, self.teacher, self.oracle,
                LiftConfig(**{**config.__dict__, "max_steps": 13}), coverage_weight=self.coverage,
                checkpoint_path=checkpoint, source_sha256=self.source_sha256,
                teacher_sha256=self.teacher_sha256,
            )
            resumed = optimize_latent_target(
                self.head, self.long_tokens, self.teacher, self.oracle, config,
                coverage_weight=self.coverage, checkpoint_path=checkpoint, resume=True,
                source_sha256=self.source_sha256, teacher_sha256=self.teacher_sha256,
            )
        torch.testing.assert_close(resumed.coefficients, direct.coefficients, atol=0.0, rtol=0.0)
        self.assertEqual(resumed.final_loss, direct.final_loss)
        self.assertEqual(resumed.loss_trace, direct.loss_trace)

    def test_completed_checkpoint_replays_the_saved_best_state(self) -> None:
        config = LiftConfig(max_steps=8, learning_rate=4.0, smoothness=0.0, residual_norm=0.0)
        with tempfile.TemporaryDirectory() as directory, mock.patch(
            "pre_experiments.conditional_hierarchical_vrfm.lift.pose_encoding_to_c2w", side_effect=_raw_to_c2w
        ):
            checkpoint = Path(directory) / "complete.pt"
            direct = optimize_latent_target(
                self.head, self.long_tokens, self.teacher, self.oracle, config,
                coverage_weight=self.coverage, checkpoint_path=checkpoint,
                source_sha256=self.source_sha256, teacher_sha256=self.teacher_sha256,
            )
            resumed = optimize_latent_target(
                self.head, self.long_tokens, self.teacher, self.oracle, config,
                coverage_weight=self.coverage, checkpoint_path=checkpoint, resume=True,
                source_sha256=self.source_sha256, teacher_sha256=self.teacher_sha256,
            )
        torch.testing.assert_close(resumed.coefficients, direct.coefficients, atol=0.0, rtol=0.0)
        self.assertEqual(resumed.final_loss, direct.final_loss)
        self.assertEqual(resumed.loss_trace, direct.loss_trace)

    def test_optimizer_takes_one_full_head_snapshot_not_one_per_step(self) -> None:
        config = LiftConfig(max_steps=5, learning_rate=0.08, smoothness=0.0, residual_norm=0.0)
        import pre_experiments.conditional_hierarchical_vrfm.lift as lift
        with mock.patch("pre_experiments.conditional_hierarchical_vrfm.lift.pose_encoding_to_c2w", side_effect=_raw_to_c2w), mock.patch.object(
            lift, "_snapshot_head", wraps=lift._snapshot_head
        ) as snapshot:
            optimize_latent_target(
                self.head, self.long_tokens, self.teacher, self.oracle, config,
                coverage_weight=self.coverage, source_sha256=self.source_sha256,
                teacher_sha256=self.teacher_sha256,
            )
        self.assertEqual(snapshot.call_count, 1)

    def test_final_snapshot_detects_buffer_data_mutation_that_avoids_version_changes(self) -> None:
        head = _DataMutatingBufferHead()
        with mock.patch("pre_experiments.conditional_hierarchical_vrfm.lift.pose_encoding_to_c2w", side_effect=_raw_to_c2w):
            with self.assertRaisesRegex(ValueError, "Camera Head state"):
                optimize_latent_target(
                    head, self.long_tokens, self.teacher, self.oracle,
                    LiftConfig(max_steps=2), coverage_weight=self.coverage,
                    source_sha256=self.source_sha256, teacher_sha256=self.teacher_sha256,
                )
        torch.testing.assert_close(head.counter, torch.zeros(()))

    def test_checkpoint_loading_uses_weights_only_and_never_executes_reduce(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "executed.txt"
            path = Path(directory) / "unsafe.pt"
            torch.save({"unsafe": _UnsafePayload(str(marker))}, path)
            with self.assertRaisesRegex(ValueError, "checkpoint"):
                load_lift_checkpoint(path)
            self.assertFalse(marker.exists())

    def test_rejects_invalid_covered_teacher_and_oracle_pose(self) -> None:
        invalid_teacher = self.teacher.clone()
        invalid_teacher[:, 100, :3, :3] = 2.0 * torch.eye(3)
        with self.assertRaisesRegex(ValueError, "SO\(3\)"):
            latent_lift_loss(
                corrected_c2w_raw=self.baseline, baseline_c2w_raw=self.baseline,
                teacher_c2w_gt_gauge=invalid_teacher, coverage_weight=self.coverage,
                oracle=self.oracle, residual=torch.zeros(1, 500, 2048), config=LiftConfig(),
            )
        bad_oracle = FrozenOracle(**{**self.oracle.__dict__, "scale": -1.0})
        with self.assertRaisesRegex(ValueError, "scale"):
            latent_lift_loss(
                corrected_c2w_raw=self.baseline, baseline_c2w_raw=self.baseline,
                teacher_c2w_gt_gauge=self.teacher, coverage_weight=self.coverage,
                oracle=bad_oracle, residual=torch.zeros(1, 500, 2048), config=LiftConfig(),
            )
        bad_rotation = FrozenOracle(
            **{**self.oracle.__dict__, "rotation": ((2.0, 0.0, 0.0), (0.0, 2.0, 0.0), (0.0, 0.0, 2.0))}
        )
        with self.assertRaisesRegex(ValueError, "SO\(3\)"):
            latent_lift_loss(
                corrected_c2w_raw=self.baseline, baseline_c2w_raw=self.baseline,
                teacher_c2w_gt_gauge=self.teacher, coverage_weight=self.coverage,
                oracle=bad_rotation, residual=torch.zeros(1, 500, 2048), config=LiftConfig(),
            )

    def test_rejects_nan_in_positive_coverage_and_uses_coverage_magnitudes(self) -> None:
        broken = self.teacher.clone()
        broken[:, 100] = torch.nan
        with self.assertRaisesRegex(ValueError, "covered teacher"):
            latent_lift_loss(
                corrected_c2w_raw=self.baseline, baseline_c2w_raw=self.baseline,
                teacher_c2w_gt_gauge=broken, coverage_weight=self.coverage,
                oracle=self.oracle, residual=torch.zeros(1, 500, 2048), config=LiftConfig(),
            )
        corrected = self.baseline.clone()
        corrected[:, 101:400, 0, 3] = 0.5
        equal = torch.zeros(500)
        equal[100:400] = 1.0
        weighted = equal.clone()
        weighted[100] = 100.0
        equal_loss = latent_lift_loss(
            corrected_c2w_raw=corrected, baseline_c2w_raw=self.baseline,
            teacher_c2w_gt_gauge=self.teacher, coverage_weight=equal,
            oracle=self.oracle, residual=torch.zeros(1, 500, 2048), config=LiftConfig(),
        )["teacher_center_loss"]
        weighted_loss = latent_lift_loss(
            corrected_c2w_raw=corrected, baseline_c2w_raw=self.baseline,
            teacher_c2w_gt_gauge=self.teacher, coverage_weight=weighted,
            oracle=self.oracle, residual=torch.zeros(1, 500, 2048), config=LiftConfig(),
        )["teacher_center_loss"]
        self.assertGreater(float(weighted_loss), float(equal_loss))

    def test_optimizer_requires_explicit_provenance_digests(self) -> None:
        with mock.patch("pre_experiments.conditional_hierarchical_vrfm.lift.pose_encoding_to_c2w", side_effect=_raw_to_c2w):
            with self.assertRaisesRegex(TypeError, "source_sha256"):
                optimize_latent_target(
                    self.head, self.long_tokens, self.teacher, self.oracle, LiftConfig(max_steps=2),
                    coverage_weight=self.coverage,
                )


if __name__ == "__main__":
    unittest.main()
