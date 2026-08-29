from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import inspect
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
from torch import nn

from pre_experiments.camera_translation_hvrfm import teacher as teacher_module
from pre_experiments.camera_translation_hvrfm.teacher import (
    TeacherControls,
    build_raw_gauge_teacher,
    load_teacher_controls,
    verify_legacy_teacher_witness,
)
from pre_experiments.camera_velocity_ambiguity_02.artifacts import frame_digest
from pre_experiments.camera_velocity_ambiguity_02.contracts import canonical_json_digest
from pre_experiments.camera_velocity_ambiguity_02.geometry import global_scene_scale
from pre_experiments.conditional_hierarchical_vrfm.artifacts import save_teacher_artifact
from pre_experiments.conditional_hierarchical_vrfm.teacher import (
    build_variant_window_masks,
)
from pre_experiments.variational_camera_latent.camera import pose_encoding_to_c2w
from pre_experiments.variational_camera_latent.source import save_source_shard
from vggt.utils.rotation import mat_to_quat


SCENE = "scene0029_01"
CHECKPOINT_SHA256 = "c" * 64
FORMAL_LABEL_SHA256 = "f" * 64
GIT_COMMIT = "1" * 40


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def coverage_from_controls(weights: np.ndarray, masks: np.ndarray) -> np.ndarray:
    coverage = np.zeros((4, 500), dtype=np.float64)
    for endpoint in range(4):
        for window in range(9):
            if masks[endpoint, window]:
                start = window * 50
                coverage[endpoint, start : start + 100] += weights[window]
    return coverage


def default_controls_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    weights = np.asarray(
        [0.25, 0.4, 0.6, 0.35, 0.8, 0.5, 0.3, 0.7, 0.45],
        dtype=np.float64,
    )
    masks = np.asarray(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 0, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 0, 1, 1, 1],
        ],
        dtype=np.uint8,
    )
    return weights, masks, coverage_from_controls(weights, masks)


def c2w_stack(centers: np.ndarray, rotations: np.ndarray | None = None) -> np.ndarray:
    result = np.broadcast_to(
        np.eye(4, dtype=np.float64), (len(centers), 4, 4)
    ).copy()
    result[:, :3, 3] = centers
    if rotations is not None:
        result[:, :3, :3] = rotations
    return result


def pose_encoding(poses: np.ndarray) -> np.ndarray:
    c2w = np.asarray(poses, dtype=np.float64)
    w2c_rotation = np.swapaxes(c2w[:, :3, :3], -1, -2)
    w2c_translation = -np.einsum(
        "fij,fj->fi", w2c_rotation, c2w[:, :3, 3]
    )
    quaternion = (
        mat_to_quat(torch.from_numpy(w2c_rotation.astype(np.float32)))
        .cpu()
        .numpy()
    )
    raw = np.zeros((len(c2w), 9), dtype=np.float32)
    raw[:, :3] = w2c_translation.astype(np.float32)
    raw[:, 3:7] = quaternion.astype(np.float32)
    raw[:, 7:] = np.asarray([0.8, 0.9], dtype=np.float32)
    return raw


def decode_pose(raw: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return (
            pose_encoding_to_c2w(torch.from_numpy(raw[None]))[0]
            .to(dtype=torch.float64)
            .cpu()
            .numpy()
        )


def tokens_from_pose(raw: np.ndarray) -> np.ndarray:
    tokens = np.zeros((len(raw), 2048), dtype=np.float32)
    tokens[:, :9] = raw
    return tokens


def make_source_arrays(*, perturbed_windows: bool = False) -> dict[str, np.ndarray]:
    frame_ids = np.arange(2000, 2500, dtype=np.int64)
    u = np.linspace(-2.0, 2.0, 500, dtype=np.float64)
    centers = np.stack(
        (u, np.sin(2.3 * u), 0.35 * np.cos(1.7 * u) + 0.1 * u * u),
        axis=1,
    )
    long_raw = pose_encoding(c2w_stack(centers))
    baseline = decode_pose(long_raw)
    global_tokens = tokens_from_pose(long_raw)

    short_tokens: list[np.ndarray] = []
    for window, start in enumerate(range(0, 401, 50)):
        desired = baseline[start : start + 100, :3, 3].copy()
        if perturbed_windows:
            phase = np.linspace(0.0, 3.0 * np.pi, 100, dtype=np.float64)
            amplitude = 0.003 * (window + 1)
            desired += amplitude * np.stack(
                (np.sin(phase), np.cos(0.7 * phase), np.sin(1.3 * phase)), axis=1
            )
        angle = 0.025 * (window - 4)
        rotation = np.asarray(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        scale = 0.8 + 0.05 * window
        translation = np.asarray(
            [0.1 * window, -0.04 * window, 0.03 * (window - 2)], dtype=np.float64
        )
        local_centers = ((desired - translation) @ rotation) / scale
        local_rotations = np.broadcast_to(rotation.T, (100, 3, 3)).copy()
        short_tokens.append(
            tokens_from_pose(pose_encoding(c2w_stack(local_centers, local_rotations)))
        )
    shorts = np.stack(short_tokens).astype(np.float32, copy=False)
    short_ids = np.stack(
        [frame_ids[start : start + 100] for start in range(0, 401, 50)]
    )
    return {
        "global_frame_ids": frame_ids,
        "global_camera_tokens": global_tokens,
        "short_frame_ids": short_ids,
        "short_camera_tokens": shorts,
        "overlap_frame_ids": np.stack(
            [frame_ids[start : start + 50] for start in range(50, 401, 50)]
        ),
        "overlap_long_tokens": np.stack(
            [global_tokens[start : start + 50] for start in range(50, 401, 50)]
        ),
        "overlap_left_tokens": np.stack(
            [shorts[index, 50:] for index in range(8)]
        ),
        "overlap_right_tokens": np.stack(
            [shorts[index + 1, :50] for index in range(8)]
        ),
        "span_starts": np.arange(0, 400, 50, dtype=np.int64),
        "sample_ids": np.asarray(
            [f"{SCENE}:overlap_{index:03d}" for index in range(8)], dtype="U64"
        ),
        "global_pred_c2w": baseline.astype(np.float64, copy=False),
        "overlap_long_c2w": np.stack(
            [baseline[start : start + 50] for start in range(50, 401, 50)]
        ).astype(np.float64, copy=False),
    }


def make_reference_arrays(
    source: dict[str, np.ndarray],
    *,
    source_sha256: str,
    formal_label_sha256: str = FORMAL_LABEL_SHA256,
    weights: np.ndarray | None = None,
    masks: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    default_weights, default_masks, _ = default_controls_arrays()
    weights = default_weights if weights is None else np.asarray(weights, dtype=np.float64)
    masks = default_masks if masks is None else np.asarray(masks, dtype=np.uint8)
    coverage = coverage_from_controls(weights, masks)
    frame_ids = source["global_frame_ids"].astype(np.int64, copy=True)
    baseline = source["global_pred_c2w"].astype(np.float64, copy=True)
    gt = baseline.copy()
    gt[:, 0, 3] += 0.02 * np.sin(np.linspace(0.0, 5.0, 500))
    fused = np.full((4, 500, 4, 4), np.nan, dtype=np.float64)
    for endpoint in range(4):
        covered = coverage[endpoint] > 0.0
        fused[endpoint, covered] = baseline[covered]
    oracle_payload = {
        "scene": SCENE,
        "frame_digest": frame_digest(frame_ids),
        "fit_count": 500,
        "scale": 1.0,
        "rotation": ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        "translation": (0.0, 0.0, 0.0),
    }
    return {
        "scene": np.asarray(SCENE, dtype="U32"),
        "frame_ids": frame_ids,
        "gt_c2w": gt,
        "gt_scene_scale": np.asarray(global_scene_scale(gt), dtype=np.float64),
        "baseline_c2w_raw": baseline,
        "oracle_scene": np.asarray(SCENE, dtype="U32"),
        "oracle_frame_digest": np.asarray(frame_digest(frame_ids), dtype="U64"),
        "oracle_fit_count": np.asarray(500, dtype=np.int64),
        "oracle_scale": np.asarray(1.0, dtype=np.float64),
        "oracle_rotation": np.eye(3, dtype=np.float64),
        "oracle_translation": np.zeros(3, dtype=np.float64),
        "oracle_rank": np.asarray(3, dtype=np.int64),
        "oracle_condition": np.asarray(2.0, dtype=np.float64),
        "oracle_digest": np.asarray(
            canonical_json_digest(oracle_payload), dtype="U64"
        ),
        "window_weights": weights.astype(np.float64, copy=True),
        "window_masks": masks.astype(np.uint8, copy=True),
        "coverage_weights": coverage,
        "fused_c2w": fused,
        "variant_utilities": np.zeros(4, dtype=np.float64),
        "source_sha256": np.asarray(source_sha256, dtype="U64"),
        "formal_label_sha256": np.asarray(formal_label_sha256, dtype="U64"),
        "checkpoint_sha256": np.asarray(CHECKPOINT_SHA256, dtype="U64"),
        "git_commit": np.asarray(GIT_COMMIT, dtype="U40"),
    }


def authenticate_controls(
    directory: Path,
    source: dict[str, np.ndarray],
    source_sha256: str,
    *,
    weights: np.ndarray | None = None,
    masks: np.ndarray | None = None,
) -> TeacherControls:
    path = directory / f"controls-{len(list(directory.glob('controls-*.npz'))):03d}.npz"
    save_teacher_artifact(
        path,
        make_reference_arrays(
            source,
            source_sha256=source_sha256,
            weights=weights,
            masks=masks,
        ),
    )
    return load_teacher_controls(
        path,
        expected_sha256=sha256_file(path),
        expected_source_sha256=source_sha256,
        expected_checkpoint_sha256=CHECKPOINT_SHA256,
        expected_formal_label_sha256=FORMAL_LABEL_SHA256,
    )


class TokenCameraHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))

    def decode_pose_tokens(
        self, tokens: torch.Tensor, *, num_iterations: int
    ) -> list[torch.Tensor]:
        self.last_iterations = num_iterations
        return [tokens[..., :9].clone()]


class StatefulTokenCameraHead(TokenCameraHead):
    def __init__(self, *, fail: bool = False) -> None:
        super().__init__()
        self.fail = fail
        self.register_buffer("running", torch.zeros(1))
        self.saw_eval = False
        self.grad_modes: list[bool] = []

    def decode_pose_tokens(
        self, tokens: torch.Tensor, *, num_iterations: int
    ) -> list[torch.Tensor]:
        self.saw_eval = self.saw_eval or not self.training
        self.grad_modes.append(torch.is_grad_enabled())
        self.anchor.data.add_(1.0)
        self.running.add_(1.0)
        if self.fail:
            raise RuntimeError("controlled Camera Head failure")
        return super().decode_pose_tokens(tokens, num_iterations=num_iterations)


class BufferDependentTokenCameraHead(TokenCameraHead):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("decode_epoch", torch.zeros((), dtype=torch.float32))
        self.seen_epochs: list[float] = []

    def decode_pose_tokens(
        self, tokens: torch.Tensor, *, num_iterations: int
    ) -> list[torch.Tensor]:
        epoch = float(self.decode_epoch.item())
        self.seen_epochs.append(epoch)
        output = tokens[..., :9].clone()
        phase = torch.linspace(
            -1.0, 1.0, output.shape[1], dtype=output.dtype, device=output.device
        )
        output[..., 0] += epoch * 0.02 * phase.square()
        self.decode_epoch.add_(1.0)
        return [output]


class InPlaceControlsMutatingHead(TokenCameraHead):
    def __init__(self, controls: TeacherControls) -> None:
        super().__init__()
        self.controls = controls
        self.mutated = False

    def decode_pose_tokens(
        self, tokens: torch.Tensor, *, num_iterations: int
    ) -> list[torch.Tensor]:
        output = super().decode_pose_tokens(tokens, num_iterations=num_iterations)
        if not self.mutated:
            self.controls.window_masks.setflags(write=True)
            self.controls.expected_coverage_weights.setflags(write=True)
            self.controls.window_masks[[1, 2]] = self.controls.window_masks[[2, 1]]
            self.controls.expected_coverage_weights[[1, 2]] = (
                self.controls.expected_coverage_weights[[2, 1]]
            )
            self.mutated = True
        return output


class ReplacingControlsHead(TokenCameraHead):
    def __init__(
        self, controls: TeacherControls, *, restore_before_return: bool
    ) -> None:
        super().__init__()
        self.controls = controls
        self.restore_before_return = restore_before_return
        self.mutated = False

    def decode_pose_tokens(
        self, tokens: torch.Tensor, *, num_iterations: int
    ) -> list[torch.Tensor]:
        output = super().decode_pose_tokens(tokens, num_iterations=num_iterations)
        if not self.mutated:
            original_masks = self.controls.window_masks
            original_coverage = self.controls.expected_coverage_weights
            changed_masks = original_masks.copy()
            changed_coverage = original_coverage.copy()
            changed_masks[[1, 2]] = changed_masks[[2, 1]]
            changed_coverage[[1, 2]] = changed_coverage[[2, 1]]
            object.__setattr__(self.controls, "window_masks", changed_masks)
            object.__setattr__(
                self.controls, "expected_coverage_weights", changed_coverage
            )
            if self.restore_before_return:
                object.__setattr__(self.controls, "window_masks", original_masks)
                object.__setattr__(
                    self.controls,
                    "expected_coverage_weights",
                    original_coverage,
                )
            self.mutated = True
        return output


class RawGaugeTeacherTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = make_source_arrays()
        self.source_path = self.root / "source.npz"
        save_source_shard(self.source_path, self.source)
        self.source_sha256 = sha256_file(self.source_path)
        self.controls = authenticate_controls(
            self.root, self.source, self.source_sha256
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def build(self, **overrides: object):
        arguments = {
            "source_path": self.source_path,
            "controls": self.controls,
            "camera_head": TokenCameraHead(),
            "expected_source_sha256": self.source_sha256,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "device": torch.device("cpu"),
        }
        arguments.update(overrides)
        return build_raw_gauge_teacher(**arguments)

    def test_numeric_builder_has_a_prediction_and_controls_only_signature(self) -> None:
        names = set(inspect.signature(build_raw_gauge_teacher).parameters)
        self.assertEqual(
            names,
            {
                "source_path",
                "controls",
                "camera_head",
                "expected_source_sha256",
                "checkpoint_sha256",
                "device",
            },
        )
        forbidden = ("gt", "oracle", "prepared", "quality", "utility", "fused")
        self.assertFalse(any(part in name for name in names for part in forbidden))

    def test_direct_control_construction_cannot_forge_legacy_authority(self) -> None:
        weights, masks, coverage = default_controls_arrays()
        with self.assertRaisesRegex((TypeError, ValueError), "authenticated"):
            TeacherControls(
                scene=SCENE,
                frame_ids=np.arange(2000, 2500, dtype=np.int64),
                window_weights=weights,
                window_masks=masks,
                expected_coverage_weights=coverage,
                source_sha256=self.source_sha256,
                checkpoint_sha256=CHECKPOINT_SHA256,
                formal_label_sha256=FORMAL_LABEL_SHA256,
                teacher_reference_sha256="0" * 64,
            )

    def test_replacement_and_post_authentication_mutation_invalidate_controls(self) -> None:
        with self.assertRaisesRegex((TypeError, ValueError), "authenticated"):
            replace(self.controls, teacher_reference_sha256="0" * 64)
        tampered = copy.copy(self.controls)
        object.__setattr__(tampered, "teacher_reference_sha256", "0" * 64)
        with self.assertRaisesRegex(ValueError, "authenticated"):
            self.build(controls=tampered)

    def test_load_controls_authenticates_actual_reference_and_replays_coverage(self) -> None:
        reference_path = self.root / "teacher.npz"
        save_teacher_artifact(
            reference_path,
            make_reference_arrays(self.source, source_sha256=self.source_sha256),
        )
        digest = sha256_file(reference_path)
        controls = load_teacher_controls(
            reference_path,
            expected_sha256=digest,
            expected_source_sha256=self.source_sha256,
            expected_checkpoint_sha256=CHECKPOINT_SHA256,
            expected_formal_label_sha256=FORMAL_LABEL_SHA256,
        )
        self.assertEqual(controls.teacher_reference_sha256, digest)
        np.testing.assert_array_equal(
            controls.expected_coverage_weights,
            coverage_from_controls(controls.window_weights, controls.window_masks),
        )
        reference_path.write_bytes(reference_path.read_bytes() + b"mutated")
        with self.assertRaisesRegex(ValueError, "digest"):
            load_teacher_controls(
                reference_path,
                expected_sha256=digest,
                expected_source_sha256=self.source_sha256,
                expected_checkpoint_sha256=CHECKPOINT_SHA256,
                expected_formal_label_sha256=FORMAL_LABEL_SHA256,
            )

    def test_load_controls_rejects_noncanonical_mask_row_order(self) -> None:
        weights, masks, _ = default_controls_arrays()
        masks = masks.copy()
        masks[[1, 2]] = masks[[2, 1]]
        reference_path = self.root / "reordered-masks.npz"
        save_teacher_artifact(
            reference_path,
            make_reference_arrays(
                self.source,
                source_sha256=self.source_sha256,
                weights=weights,
                masks=masks,
            ),
        )
        with self.assertRaisesRegex(ValueError, "canonical"):
            load_teacher_controls(
                reference_path,
                expected_sha256=sha256_file(reference_path),
                expected_source_sha256=self.source_sha256,
                expected_checkpoint_sha256=CHECKPOINT_SHA256,
                expected_formal_label_sha256=FORMAL_LABEL_SHA256,
            )

    def test_load_controls_rejects_noncanonical_legacy_dtypes_before_casting(self) -> None:
        cases = {
            "scene": lambda value: np.asarray(str(value), dtype="U16"),
            "frame_ids": lambda value: value.astype(np.int32),
            "window_weights": lambda value: value.astype(np.float32),
            "window_masks": lambda value: value.astype(np.int16),
            "coverage_weights": lambda value: value.astype(np.float32),
            "source_sha256": lambda value: np.asarray(str(value), dtype="U65"),
        }
        for field, mutate in cases.items():
            arrays = make_reference_arrays(
                self.source, source_sha256=self.source_sha256
            )
            arrays[field] = mutate(arrays[field])
            path = self.root / f"bad-dtype-{field}.npz"
            save_teacher_artifact(path, arrays)
            with self.subTest(field=field), self.assertRaisesRegex(
                ValueError, "dtype|canonical"
            ):
                load_teacher_controls(
                    path,
                    expected_sha256=sha256_file(path),
                    expected_source_sha256=self.source_sha256,
                    expected_checkpoint_sha256=CHECKPOINT_SHA256,
                    expected_formal_label_sha256=FORMAL_LABEL_SHA256,
                )

    def test_controls_reject_invalid_masks_weights_and_replay(self) -> None:
        weights, masks, coverage = default_controls_arrays()
        cases: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
        bad_weights = weights.copy()
        bad_weights[2] = 1.01
        cases.append(("weight", bad_weights, masks, coverage))
        bad_row_zero = masks.copy()
        bad_row_zero[0, 3] = 0
        cases.append(("row zero", weights, bad_row_zero, coverage))
        duplicate = masks.copy()
        duplicate[3] = duplicate[2]
        cases.append(("unique", weights, duplicate, coverage))
        selected_zero = weights.copy()
        selected_zero[2] = 0.0
        cases.append(("zero-weight", selected_zero, masks, coverage))
        replay = coverage.copy()
        replay[0, 0] += np.finfo(np.float64).eps
        cases.append(("coverage", weights, masks, replay))
        for label, candidate_weights, candidate_masks, candidate_coverage in cases:
            arrays = make_reference_arrays(
                self.source,
                source_sha256=self.source_sha256,
                weights=candidate_weights,
                masks=candidate_masks,
            )
            arrays["coverage_weights"] = candidate_coverage.copy()
            path = self.root / f"invalid-controls-{label.replace(' ', '-')}.npz"
            save_teacher_artifact(path, arrays)
            with self.subTest(label=label), self.assertRaises(ValueError):
                load_teacher_controls(
                    path,
                    expected_sha256=sha256_file(path),
                    expected_source_sha256=self.source_sha256,
                    expected_checkpoint_sha256=CHECKPOINT_SHA256,
                    expected_formal_label_sha256=FORMAL_LABEL_SHA256,
                )

    def test_recovers_all_nine_full_window_sim3s_and_fuses_fixed_four_masks(self) -> None:
        with (
            mock.patch.object(
                teacher_module,
                "fit_frozen_oracle",
                create=True,
                side_effect=AssertionError("numeric path fitted an oracle"),
            ),
            mock.patch.object(
                teacher_module,
                "load_prepared_gt",
                create=True,
                side_effect=AssertionError("numeric path loaded GT"),
            ),
            mock.patch.object(
                teacher_module,
                "apply_frozen_oracle",
                side_effect=AssertionError("numeric path applied an oracle"),
            ),
        ):
            teacher = self.build()
        self.assertEqual(teacher.aligned_short_centers.shape, (9, 100, 3))
        baseline_centers = teacher.baseline_c2w[:, :3, 3]
        for window, start in enumerate(range(0, 401, 50)):
            np.testing.assert_allclose(
                teacher.aligned_short_centers[window],
                baseline_centers[start : start + 100],
                atol=3e-6,
                rtol=0.0,
            )
        np.testing.assert_array_equal(
            teacher.coverage_weights, self.controls.expected_coverage_weights
        )
        np.testing.assert_array_equal(
            teacher.coverage_mask, (teacher.coverage_weights > 0.0).astype(np.uint8)
        )
        uncovered = teacher.coverage_mask == 0
        self.assertTrue(np.isnan(teacher.raw_teacher_centers[uncovered]).all())
        self.assertTrue(
            np.all(teacher.translation_endpoints.view(np.uint32)[uncovered] == 0)
        )
        self.assertTrue(np.isfinite(teacher.filled_teacher_centers).all())

    def test_gt_oracle_utility_and_fused_witness_mutations_cannot_reach_numeric_builder(self) -> None:
        baseline = self.build()
        legacy = make_reference_arrays(self.source, source_sha256=self.source_sha256)
        legacy["gt_c2w"][:, :3, 3] += 1000.0
        legacy["oracle_scale"] = np.asarray(999.0, dtype=np.float64)
        legacy["oracle_rotation"] = np.diag([-1.0, 1.0, 1.0]).astype(np.float64)
        legacy["oracle_translation"] = np.asarray([4.0, 5.0, 6.0], dtype=np.float64)
        legacy["variant_utilities"][:] = [-10.0, 20.0, 30.0, 40.0]
        covered = legacy["coverage_weights"] > 0.0
        legacy["fused_c2w"][covered, :3, 3] += 500.0

        with (
            mock.patch.object(
                teacher_module,
                "fit_frozen_oracle",
                side_effect=AssertionError("numeric path fitted an oracle"),
            ),
            mock.patch.object(
                teacher_module,
                "apply_frozen_oracle",
                side_effect=AssertionError("numeric path applied an oracle"),
            ),
            mock.patch.object(
                teacher_module,
                "load_prepared_gt",
                side_effect=AssertionError("numeric path loaded GT"),
            ),
        ):
            mutated = self.build(controls=self.controls)
        np.testing.assert_array_equal(
            mutated.coverage_weights, baseline.coverage_weights
        )
        np.testing.assert_array_equal(
            mutated.translation_endpoints, baseline.translation_endpoints
        )
        np.testing.assert_array_equal(
            mutated.filled_teacher_centers, baseline.filled_teacher_centers
        )

    def test_builder_restores_head_state_on_success_and_failure(self) -> None:
        for fail in (False, True):
            head = StatefulTokenCameraHead(fail=fail)
            head.train()
            parameter = head.anchor.detach().clone()
            buffer = head.running.detach().clone()
            context = (
                self.assertRaisesRegex(RuntimeError, "controlled")
                if fail
                else _NullContext()
            )
            with context:
                self.build(camera_head=head)
            self.assertTrue(head.training)
            self.assertTrue(head.saw_eval)
            self.assertTrue(head.grad_modes)
            self.assertFalse(any(head.grad_modes))
            torch.testing.assert_close(head.anchor, parameter)
            torch.testing.assert_close(head.running, buffer)

    def test_long_decode_mutation_cannot_change_short_decode_state(self) -> None:
        reference = self.build(camera_head=TokenCameraHead())
        head = BufferDependentTokenCameraHead()
        candidate = self.build(camera_head=head)
        self.assertEqual(head.seen_epochs, [0.0, 0.0])
        self.assertEqual(float(head.decode_epoch.item()), 0.0)
        np.testing.assert_array_equal(
            candidate.translation_endpoints, reference.translation_endpoints
        )

    def test_camera_head_cannot_toggle_controls_writable_and_mutate_in_place(self) -> None:
        with self.assertRaises(ValueError):
            self.build(camera_head=InPlaceControlsMutatingHead(self.controls))

    def test_camera_head_control_replacement_is_detected_after_forward(self) -> None:
        head = ReplacingControlsHead(
            self.controls, restore_before_return=False
        )
        with self.assertRaisesRegex(ValueError, "authenticated"):
            self.build(camera_head=head)

    def test_transient_control_replacement_cannot_affect_numeric_result(self) -> None:
        reference = self.build(camera_head=TokenCameraHead())
        head = ReplacingControlsHead(
            self.controls, restore_before_return=True
        )
        candidate = self.build(camera_head=head)
        np.testing.assert_array_equal(
            candidate.coverage_weights, reference.coverage_weights
        )
        np.testing.assert_array_equal(
            candidate.translation_endpoints, reference.translation_endpoints
        )
        np.testing.assert_array_equal(
            candidate.filled_teacher_centers, reference.filled_teacher_centers
        )

    def test_entire_frozen_decode_scope_disables_autograd(self) -> None:
        converter_grad_modes: list[bool] = []
        real_converter = teacher_module.pose_encoding_to_c2w

        def capture_converter(raw: torch.Tensor) -> torch.Tensor:
            converter_grad_modes.append(torch.is_grad_enabled())
            return real_converter(raw)

        with mock.patch.object(
            teacher_module, "pose_encoding_to_c2w", side_effect=capture_converter
        ):
            self.build(camera_head=StatefulTokenCameraHead())
        self.assertEqual(len(converter_grad_modes), 2)
        self.assertFalse(any(converter_grad_modes))

    def test_authenticated_baseline_is_only_a_witness(self) -> None:
        reference = self.build()
        perturbed = {name: value.copy() for name, value in self.source.items()}
        scale = reference.prediction_scale
        perturbation = 1e-7 * scale * np.sin(np.arange(500, dtype=np.float64))
        perturbed["global_pred_c2w"][:, 0, 3] += perturbation
        perturbed["overlap_long_c2w"] = np.stack(
            [
                perturbed["global_pred_c2w"][start : start + 50]
                for start in range(50, 401, 50)
            ]
        )
        path = self.root / "perturbed.npz"
        save_source_shard(path, perturbed)
        digest = sha256_file(path)
        candidate = self.build(
            source_path=path,
            controls=authenticate_controls(self.root, perturbed, digest),
            expected_source_sha256=digest,
        )
        np.testing.assert_array_equal(
            candidate.translation_endpoints, reference.translation_endpoints
        )
        np.testing.assert_array_equal(
            candidate.filled_teacher_centers, reference.filled_teacher_centers
        )

        rejected = {name: value.copy() for name, value in self.source.items()}
        rejected["global_pred_c2w"][:, 0, 3] += 1e-2 * scale
        rejected["overlap_long_c2w"] = np.stack(
            [
                rejected["global_pred_c2w"][start : start + 50]
                for start in range(50, 401, 50)
            ]
        )
        rejected_path = self.root / "rejected.npz"
        save_source_shard(rejected_path, rejected)
        rejected_digest = sha256_file(rejected_path)
        with self.assertRaisesRegex(ValueError, "baseline"):
            self.build(
                source_path=rejected_path,
                controls=authenticate_controls(self.root, rejected, rejected_digest),
                expected_source_sha256=rejected_digest,
            )

    def test_selected_invalid_alignment_fails_but_unused_invalid_window_is_allowed(self) -> None:
        invalid = {name: value.copy() for name, value in self.source.items()}
        linear_centers = np.zeros((100, 3), dtype=np.float64)
        linear_centers[:, 0] = np.linspace(-1.0, 1.0, 100)
        invalid["short_camera_tokens"][4] = tokens_from_pose(
            pose_encoding(c2w_stack(linear_centers))
        )
        invalid["overlap_left_tokens"][4] = invalid["short_camera_tokens"][4, 50:]
        invalid["overlap_right_tokens"][3] = invalid["short_camera_tokens"][4, :50]
        path = self.root / "invalid-window.npz"
        save_source_shard(path, invalid)
        digest = sha256_file(path)
        with self.assertRaisesRegex(ValueError, "window 4"):
            self.build(
                source_path=path,
                controls=authenticate_controls(self.root, invalid, digest),
                expected_source_sha256=digest,
            )

        weights, masks, _ = default_controls_arrays()
        weights[4] = 0.0
        masks = build_variant_window_masks(SCENE, weights).astype(np.uint8)
        controls = authenticate_controls(
            self.root,
            invalid,
            digest,
            weights=weights,
            masks=masks,
        )
        teacher = self.build(
            source_path=path,
            controls=controls,
            expected_source_sha256=digest,
        )
        self.assertTrue(np.isnan(teacher.aligned_short_centers[4]).all())

    def test_meaningful_weight_and_mask_changes_alter_numeric_teacher(self) -> None:
        noisy_path = self.root / "noisy.npz"
        noisy_source = make_source_arrays(perturbed_windows=True)
        save_source_shard(noisy_path, noisy_source)
        digest = sha256_file(noisy_path)
        controls = authenticate_controls(self.root, noisy_source, digest)
        baseline = self.build(
            source_path=noisy_path,
            controls=controls,
            expected_source_sha256=digest,
        )
        weights = controls.window_weights.copy()
        weights[4] *= 0.5
        changed_weights = authenticate_controls(
            self.root,
            noisy_source,
            digest,
            weights=weights,
            masks=controls.window_masks,
        )
        weighted = self.build(
            source_path=noisy_path,
            controls=changed_weights,
            expected_source_sha256=digest,
        )
        self.assertFalse(
            np.array_equal(weighted.coverage_weights, baseline.coverage_weights)
        )
        self.assertFalse(
            np.array_equal(weighted.translation_endpoints, baseline.translation_endpoints)
        )

        mask_weights = controls.window_weights.copy()
        mask_weights[0] = 0.0
        masks = build_variant_window_masks(SCENE, mask_weights).astype(np.uint8)
        changed_masks = authenticate_controls(
            self.root,
            noisy_source,
            digest,
            weights=mask_weights,
            masks=masks,
        )
        masked = self.build(
            source_path=noisy_path,
            controls=changed_masks,
            expected_source_sha256=digest,
        )
        self.assertFalse(np.array_equal(masked.coverage_mask, baseline.coverage_mask))

    def test_legacy_oracle_is_forward_replay_witness_only(self) -> None:
        reference_path = self.root / "teacher.npz"
        arrays = make_reference_arrays(self.source, source_sha256=self.source_sha256)
        save_teacher_artifact(reference_path, arrays)
        digest = sha256_file(reference_path)
        controls = load_teacher_controls(
            reference_path,
            expected_sha256=digest,
            expected_source_sha256=self.source_sha256,
            expected_checkpoint_sha256=CHECKPOINT_SHA256,
            expected_formal_label_sha256=FORMAL_LABEL_SHA256,
        )
        teacher = self.build(controls=controls)
        verify_legacy_teacher_witness(
            teacher, reference_path, expected_sha256=digest
        )

        mutated = make_reference_arrays(self.source, source_sha256=self.source_sha256)
        covered = mutated["coverage_weights"][0] > 0.0
        mutated["fused_c2w"][0, covered, 0, 3] += 0.01
        bad_path = self.root / "bad-teacher.npz"
        save_teacher_artifact(bad_path, mutated)
        with self.assertRaisesRegex(ValueError, "witness"):
            verify_legacy_teacher_witness(
                teacher, bad_path, expected_sha256=sha256_file(bad_path)
            )

    def test_source_and_reference_path_security_and_exact_schema(self) -> None:
        extra_source = {name: value.copy() for name, value in self.source.items()}
        extra_source["innocent"] = np.asarray(1, dtype=np.int64)
        extra_path = self.root / "extra-source.npz"
        with extra_path.open("wb") as handle:
            np.savez_compressed(handle, **extra_source)
        digest = sha256_file(extra_path)
        with self.assertRaises(ValueError):
            self.build(
                source_path=extra_path,
                controls=authenticate_controls(self.root, self.source, digest),
                expected_source_sha256=digest,
            )

    def test_authenticated_teacher_inputs_reject_lexical_parent_traversal(self) -> None:
        safe = self.root / "safe"
        safe.mkdir()
        reference_path = self.root / "teacher.npz"
        save_teacher_artifact(
            reference_path,
            make_reference_arrays(self.source, source_sha256=self.source_sha256),
        )
        reference_digest = sha256_file(reference_path)
        lexical_reference = safe / ".." / reference_path.name
        lexical_source = safe / ".." / self.source_path.name
        with self.assertRaisesRegex(ValueError, "parent traversal"):
            load_teacher_controls(
                lexical_reference,
                expected_sha256=reference_digest,
                expected_source_sha256=self.source_sha256,
                expected_checkpoint_sha256=CHECKPOINT_SHA256,
                expected_formal_label_sha256=FORMAL_LABEL_SHA256,
            )
        teacher = self.build()
        with self.assertRaisesRegex(ValueError, "parent traversal"):
            self.build(source_path=lexical_source)
        with self.assertRaisesRegex(ValueError, "parent traversal"):
            verify_legacy_teacher_witness(
                teacher, lexical_reference, expected_sha256=reference_digest
            )


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


if __name__ == "__main__":
    unittest.main()
