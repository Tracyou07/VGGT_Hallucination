from __future__ import annotations

import ast
import copy
from dataclasses import replace
from pathlib import Path
import hashlib
import inspect
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from pre_experiments.camera_translation_hvrfm.artifacts import (
    load_quality_sidecar,
    load_translation_target,
    save_long_context,
    save_quality_sidecar,
    save_short_context,
    save_translation_target,
)
from pre_experiments.camera_translation_hvrfm.data import (
    PublishedTranslationSample,
    calibration_role,
)
from pre_experiments.camera_translation_hvrfm.geometry import build_translation_endpoint
from pre_experiments.camera_velocity_ambiguity_02.artifacts import frame_digest
from pre_experiments.camera_velocity_ambiguity_02.contracts import canonical_json_digest
from pre_experiments.camera_velocity_ambiguity_02.frozen_oracle import FrozenOracle
from pre_experiments.variational_camera_latent.camera import (
    pose_encoding_to_c2w as real_pose_encoding_to_c2w,
)

try:
    from pre_experiments.camera_translation_hvrfm import evaluate
except (ImportError, ModuleNotFoundError):
    evaluate = None  # type: ignore[assignment]


SCENE = "scene0029_01"
SAMPLE_ID = f"{SCENE}:frames_500"
SOURCE_SHA256 = "1" * 64
CHECKPOINT_SHA256 = "2" * 64
FORMAL_LABEL_SHA256 = "3" * 64
TEACHER_REFERENCE_SHA256 = "4" * 64
GIT_COMMIT = "5" * 40
ALPHAS = np.asarray([0.25, 0.5, 0.75, 1.0], dtype=np.float64)
CALIBRATION_SCENES = (
    "scene0000_00",
    "scene0013_02",
    "scene0029_01",
    "scene0084_01",
    "scene0121_01",
    "scene0207_01",
    "scene0280_00",
    "scene0325_01",
    "scene0675_00",
    "scene0691_00",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _camera_fixture(
    quaternion: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    frame_ids = np.arange(500, dtype=np.int64)
    centers = np.zeros((500, 3), dtype=np.float64)
    centers[:, 0] = np.arange(500, dtype=np.float64) - 249.5
    pose = np.zeros((500, 9), dtype=np.float32)
    if quaternion is None:
        pose[:, 6] = 1.0
    else:
        value = np.asarray(quaternion, dtype=np.float32)
        value = value / np.linalg.norm(value)
        pose[:, 3:7] = value
    pose[:, 7] = np.linspace(0.7, 1.1, 500, dtype=np.float32)
    pose[:, 8] = np.linspace(0.8, 1.2, 500, dtype=np.float32)
    if quaternion is None:
        pose[:, :3] = (-centers).astype(np.float32)
    else:
        with torch.no_grad():
            rotation_only = real_pose_encoding_to_c2w(
                torch.from_numpy(pose[None])
            )[0].to(dtype=torch.float64).numpy()
        rotations_w2c = np.swapaxes(rotation_only[:, :3, :3], -1, -2)
        pose[:, :3] = (-np.einsum("tij,tj->ti", rotations_w2c, centers)).astype(
            np.float32
        )
    with torch.no_grad():
        c2w = real_pose_encoding_to_c2w(torch.from_numpy(pose[None]))[0].to(
            dtype=torch.float64
        ).numpy()
    actual_centers = c2w[:, :3, 3]
    scale = float(
        np.sqrt(
            np.mean(
                np.sum((actual_centers - actual_centers.mean(0)) ** 2, axis=1)
            )
        )
    )
    return frame_ids, c2w, pose, scale


def _coverage() -> np.ndarray:
    coverage = np.zeros((4, 500), dtype=np.uint8)
    coverage[0, :100] = 1
    coverage[1, 50:250] = 1
    coverage[2, 100:400] = 1
    coverage[3, 100:] = 1
    return coverage


def _oracle_digest(frame_ids: np.ndarray) -> str:
    payload = {
        "scene": SCENE,
        "frame_digest": frame_digest(frame_ids),
        "fit_count": 500,
        "scale": 1.0,
        "rotation": ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        "translation": (0.0, 0.0, 0.0),
    }
    return canonical_json_digest(payload)


def _direct_so3_error_deg(candidate: np.ndarray, reference: np.ndarray) -> np.ndarray:
    relative = np.einsum(
        "...ij,...kj->...ik",
        candidate[..., :3, :3],
        reference[..., :3, :3],
    )
    cosine = np.clip(
        (np.trace(relative, axis1=-2, axis2=-1) - 1.0) * 0.5,
        -1.0,
        1.0,
    )
    return np.rad2deg(np.arccos(cosine))


def _gate_endpoint(endpoint_id: int) -> dict[str, object]:
    return {
        "endpoint_id": endpoint_id,
        "covered_utility": 0.96,
        "teacher_covered_utility": 1.0,
        "full_scene_utility": 0.02,
        "covered_roundtrip_fraction": 0.0,
        "uncovered_drift_fraction": 0.0,
        "rotation_delta_deg": 0.0,
        "quaternion_bytes_equal": True,
        "fov_bytes_equal": True,
        "uncovered_positive_zero": True,
        "endpoint_rms": 0.125,
        "coverage_fraction": 0.5,
        "all_finite": True,
    }


def _gate_scene(scene: str) -> dict[str, object]:
    return {
        "scene": scene,
        "sample_id": f"{scene}:frames_500",
        "role": "validation"
        if scene in {"scene0325_01", "scene0675_00"}
        else "train",
        "endpoint_count": 4,
        "endpoint_ids": [0, 1, 2, 3],
        "endpoints": [_gate_endpoint(endpoint) for endpoint in range(4)],
        "mean_covered_utility": 0.96,
        "mean_teacher_covered_utility": 1.0,
        "teacher_retention": 0.96,
        "mean_full_scene_utility": 0.02,
        "max_covered_roundtrip_fraction": 0.0,
        "max_uncovered_drift_fraction": 0.0,
        "max_rotation_delta_deg": 0.0,
        "quaternion_bytes_equal": True,
        "fov_bytes_equal": True,
        "uncovered_positive_zero": True,
        "all_finite": True,
        "provenance": {
            "long_sha256": "1" * 64,
            "short_sha256": "2" * 64,
            "quality_sha256": "3" * 64,
            "target_sha256": "4" * 64,
            "source_sha256": "5" * 64,
            "checkpoint_sha256": "6" * 64,
            "teacher_reference_sha256": "7" * 64,
            "git_commit": "8" * 40,
        },
    }


def _passing_gate_scenes() -> list[dict[str, object]]:
    return [_gate_scene(scene) for scene in CALIBRATION_SCENES]


def _gate_cohort() -> list[PublishedTranslationSample]:
    return [
        replace(
            PublishedTranslationSample.placeholder(scene, calibration_role(scene)),
            long_sha256="1" * 64,
            short_sha256="2" * 64,
            quality_sha256="3" * 64,
            target_sha256="4" * 64,
        )
        for scene in CALIBRATION_SCENES
    ]


class EvaluationBundle:
    def __init__(
        self,
        root: Path,
        *,
        quaternion: np.ndarray | None = None,
        alphas: np.ndarray = ALPHAS,
    ) -> None:
        self.root = root
        self.alphas = np.asarray(alphas, dtype=np.float64)
        self.frame_ids, self.baseline, self.pose, self.scale = _camera_fixture(
            quaternion
        )
        self.coverage = _coverage()
        self.gt = self.baseline.copy()
        self.gt[:, 1, 3] = 1.0
        raw_teacher = np.full((4, 500, 3), np.nan, dtype=np.float64)
        for endpoint, alpha in enumerate(self.alphas):
            mask = self.coverage[endpoint] != 0
            raw_teacher[endpoint, mask] = self.baseline[mask, :3, 3]
            raw_teacher[endpoint, mask, 1] += alpha
        self.endpoints, self.filled, replayed_scale = build_translation_endpoint(
            long_frame_ids=self.frame_ids,
            teacher_frame_ids=self.frame_ids.copy(),
            baseline_c2w=self.baseline,
            baseline_pose_encoding=self.pose,
            teacher_centers=raw_teacher,
            coverage_mask=self.coverage,
        )
        assert replayed_scale == self.scale
        self.long_path = root / "long.npz"
        self.short_path = root / "short.npz"
        self.target_path = root / "target.npz"
        self.quality_path = root / "quality.npz"
        self.sample = self._write_bundle()

    def _shared(self) -> dict[str, np.ndarray]:
        return {
            "sample_id": np.asarray(SAMPLE_ID, dtype="U96"),
            "scene": np.asarray(SCENE, dtype="U32"),
            "source_sha256": np.asarray(SOURCE_SHA256, dtype="U64"),
            "checkpoint_sha256": np.asarray(CHECKPOINT_SHA256, dtype="U64"),
            "git_commit": np.asarray(GIT_COMMIT, dtype="U40"),
        }

    def quality_arrays(self) -> dict[str, np.ndarray]:
        baseline_error = (
            np.linalg.norm(
                self.baseline[:, :3, 3] - self.gt[:, :3, 3], axis=1
            )
            / self.scale
        )
        teacher_error = np.full((4, 500), np.nan, dtype=np.float64)
        teacher_rotation = np.full((4, 500), np.nan, dtype=np.float64)
        utilities = np.empty(4, dtype=np.float64)
        baseline_rotation = _direct_so3_error_deg(self.baseline, self.gt)
        for endpoint, _ in enumerate(self.alphas):
            mask = self.coverage[endpoint] != 0
            teacher_error[endpoint, mask] = (
                np.linalg.norm(
                    self.filled[endpoint, mask] - self.gt[mask, :3, 3], axis=1
                )
                / self.scale
            )
            teacher_rotation[endpoint, mask] = baseline_rotation[mask]
            baseline_rms = float(np.sqrt(np.mean(baseline_error[mask] ** 2)))
            teacher_rms = float(
                np.sqrt(np.mean(teacher_error[endpoint, mask] ** 2))
            )
            utilities[endpoint] = (baseline_rms - teacher_rms) / baseline_rms
        return {
            **self._shared(),
            "frame_ids": self.frame_ids.copy(),
            "teacher_variant_ids": np.arange(4, dtype=np.int64),
            "gt_c2w": self.gt.copy(),
            "gt_scene_scale": np.asarray(self.scale, dtype=np.float64),
            "oracle_scene": np.asarray(SCENE, dtype="U32"),
            "oracle_frame_digest": np.asarray(frame_digest(self.frame_ids), dtype="U64"),
            "oracle_fit_count": np.asarray(500, dtype=np.int64),
            "oracle_scale": np.asarray(1.0, dtype=np.float64),
            "oracle_rotation": np.eye(3, dtype=np.float64),
            "oracle_translation": np.zeros(3, dtype=np.float64),
            "oracle_rank": np.asarray(3, dtype=np.int64),
            "oracle_condition": np.asarray(1.0, dtype=np.float64),
            "oracle_digest": np.asarray(_oracle_digest(self.frame_ids), dtype="U64"),
            "window_weights": np.ones(9, dtype=np.float64),
            "window_masks": np.asarray(
                [[1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 0, 1, 0, 1, 0, 1, 0, 1],
                 [0, 1, 0, 1, 0, 1, 0, 1, 0], [1, 1, 0, 0, 1, 1, 0, 0, 1]],
                dtype=np.uint8,
            ),
            "coverage_weights": self.coverage.astype(np.float64),
            "variant_utilities": utilities,
            "baseline_translation_error_normalized": baseline_error,
            "baseline_rotation_error_deg": baseline_rotation,
            "teacher_translation_error_normalized": teacher_error,
            "teacher_rotation_error_deg": teacher_rotation,
            "formal_label_sha256": np.asarray(FORMAL_LABEL_SHA256, dtype="U64"),
            "teacher_reference_sha256": np.asarray(
                TEACHER_REFERENCE_SHA256, dtype="U64"
            ),
        }

    def _write_bundle(self) -> PublishedTranslationSample:
        long_sha = save_long_context(
            self.long_path,
            {
                **self._shared(),
                "frame_ids": self.frame_ids.copy(),
                "camera_tokens": np.zeros((500, 2048), dtype=np.float32),
                "baseline_pose_encoding": self.pose.copy(),
                "baseline_c2w": self.baseline.copy(),
                "prediction_scale": np.asarray(self.scale, dtype=np.float64),
            },
        )
        short_sha = save_short_context(
            self.short_path,
            {
                **self._shared(),
                "short_frame_ids": np.stack(
                    [self.frame_ids[start : start + 100] for start in range(0, 401, 50)]
                ),
                "short_camera_tokens": np.zeros((9, 100, 2048), dtype=np.float32),
                "long_context_sha256": np.asarray(long_sha, dtype="U64"),
            },
        )
        quality_sha = save_quality_sidecar(self.quality_path, self.quality_arrays())
        target_sha = save_translation_target(
            self.target_path,
            {
                **self._shared(),
                "frame_ids": self.frame_ids.copy(),
                "teacher_variant_ids": np.arange(4, dtype=np.int64),
                "coverage_mask": self.coverage.copy(),
                "translation_endpoints": self.endpoints.copy(),
                "teacher_centers_raw_filled": self.filled.copy(),
                "prediction_scale": np.asarray(self.scale, dtype=np.float64),
                "long_context_sha256": np.asarray(long_sha, dtype="U64"),
                "short_context_sha256": np.asarray(short_sha, dtype="U64"),
                "quality_sha256": np.asarray(quality_sha, dtype="U64"),
                "teacher_reference_sha256": np.asarray(
                    TEACHER_REFERENCE_SHA256, dtype="U64"
                ),
            },
        )
        return PublishedTranslationSample(
            sample_id=SAMPLE_ID,
            scene=SCENE,
            role="train",
            long_path=self.long_path,
            short_path=self.short_path,
            quality_path=self.quality_path,
            target_path=self.target_path,
            long_sha256=long_sha,
            short_sha256=short_sha,
            quality_sha256=quality_sha,
            target_sha256=target_sha,
        )


class TranslationEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.bundle = EvaluationBundle(Path(self.temporary.name))

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def api(self):
        self.assertIsNotNone(evaluate, "Task 3 evaluate module is missing")
        return evaluate

    def test_evaluator_exposes_only_saved_oracle_decode_not_fitting(self) -> None:
        module = self.api()
        self.assertFalse(
            hasattr(module, "fit_frozen_oracle"),
            "evaluation namespace must not expose oracle fitting",
        )
        self.assertTrue(callable(module.decode_saved_oracle))
        oracle = module.decode_saved_oracle(
            load_quality_sidecar(self.bundle.quality_path)
        )
        self.assertIsInstance(oracle, FrozenOracle)
        self.assertEqual(oracle.scene, SCENE)
        self.assertEqual(oracle.fit_count, 500)

    def test_evaluator_ast_imports_no_teacher_private_or_fitting_symbol(self) -> None:
        module = self.api()
        tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
        frozen_names = [
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            and node.module
            == "pre_experiments.camera_velocity_ambiguity_02.frozen_oracle"
            for alias in node.names
        ]
        teacher_names = [
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            and node.module == "pre_experiments.camera_translation_hvrfm.teacher"
            for alias in node.names
        ]
        self.assertEqual(frozen_names, ["FrozenOracle", "apply_frozen_oracle"])
        self.assertEqual(teacher_names, [])

    def test_real_replay_uses_own_masks_ratio_of_means_and_one_cpu_decode(self) -> None:
        module = self.api()
        calls: list[tuple[tuple[int, ...], str, bool]] = []

        def checked_decode(value: torch.Tensor) -> torch.Tensor:
            calls.append((tuple(value.shape), value.device.type, torch.is_grad_enabled()))
            return real_pose_encoding_to_c2w(value)

        real_decode_saved_oracle = module.decode_saved_oracle
        with mock.patch.object(
            module,
            "decode_saved_oracle",
            wraps=real_decode_saved_oracle,
        ) as decode_saved, mock.patch.object(
            module, "pose_encoding_to_c2w", side_effect=checked_decode
        ):
            metrics = module.evaluate_translation_sample(self.bundle.sample)

        self.assertEqual(decode_saved.call_count, 1)
        self.assertEqual(calls, [((5, 500, 9), "cpu", False)])
        self.assertEqual(metrics["endpoint_ids"], [0, 1, 2, 3])
        self.assertEqual(
            [row["coverage_fraction"] for row in metrics["endpoints"]],
            [0.2, 0.4, 0.6, 0.8],
        )
        np.testing.assert_allclose(
            [row["covered_utility"] for row in metrics["endpoints"]],
            ALPHAS,
            atol=2e-6,
            rtol=0.0,
        )
        self.assertAlmostEqual(metrics["teacher_retention"], 1.0, places=5)
        self.assertEqual(metrics["provenance"]["target_sha256"], self.bundle.sample.target_sha256)

    def test_identical_non_axis_float32_rotations_have_zero_so3_delta(self) -> None:
        module = self.api()
        non_axis = EvaluationBundle(
            Path(self.temporary.name) / "non-axis",
            quaternion=np.asarray([0.13, -0.29, 0.41, 0.85], dtype=np.float32),
        )
        try:
            metrics = module.evaluate_translation_sample(non_axis.sample)
        except ValueError as error:
            self.fail(f"identical decoded rotations were rejected: {error}")
        self.assertEqual(metrics["max_rotation_delta_deg"], 0.0)
        self.assertTrue(metrics["quaternion_bytes_equal"])

    def test_scene_aggregation_is_ratio_of_means_not_mean_of_ratios(self) -> None:
        module = self.api()
        rows = [
            {"covered_utility": 0.9, "teacher_covered_utility": 0.6, "full_scene_utility": 0.4},
            {"covered_utility": 0.1, "teacher_covered_utility": 0.2, "full_scene_utility": 0.2},
            {"covered_utility": 0.5, "teacher_covered_utility": 0.5, "full_scene_utility": 0.0},
            {"covered_utility": 0.5, "teacher_covered_utility": 0.7, "full_scene_utility": -0.2},
        ]
        aggregate = module._aggregate_scene_utilities(rows)
        self.assertEqual(aggregate["mean_covered_utility"], 0.5)
        self.assertEqual(aggregate["mean_teacher_covered_utility"], 0.5)
        self.assertEqual(aggregate["teacher_retention"], 1.0)
        self.assertAlmostEqual(aggregate["mean_full_scene_utility"], 0.1)

    def test_zero_nonfinite_baseline_and_teacher_denominators_are_rejected(self) -> None:
        module = self.api()
        for value in (0.0, -0.0, -1.0, np.nan, np.inf):
            with self.subTest(baseline=value), self.assertRaisesRegex(
                ValueError, "finite and positive"
            ):
                module._relative_utility(value, 0.5, name="covered")
        rows = [
            {"covered_utility": 0.1, "teacher_covered_utility": value, "full_scene_utility": 0.1}
            for value in (0.1, -0.1, 0.1, -0.1)
        ]
        with self.assertRaisesRegex(ValueError, "teacher.*positive"):
            module._aggregate_scene_utilities(rows)
        rows[0]["teacher_covered_utility"] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            module._aggregate_scene_utilities(rows)

    def test_sample_digest_and_recomputed_diagnostic_tampering_are_rejected(self) -> None:
        module = self.api()
        with self.bundle.short_path.open("ab") as handle:
            handle.write(b"tamper")
        with self.assertRaisesRegex(ValueError, "digest"):
            module.evaluate_translation_sample(self.bundle.sample)

        replacement_root = Path(self.temporary.name) / "replacement"
        replacement_root.mkdir()
        clean = EvaluationBundle(replacement_root)
        quality = load_quality_sidecar(clean.quality_path)
        quality["variant_utilities"] = quality["variant_utilities"].copy()
        quality["variant_utilities"][0] += 0.01
        quality_sha = save_quality_sidecar(clean.quality_path, quality)
        target = load_translation_target(clean.target_path)
        target["quality_sha256"] = np.asarray(quality_sha, dtype="U64")
        target_sha = save_translation_target(clean.target_path, target)
        tampered = replace(clean.sample, quality_sha256=quality_sha, target_sha256=target_sha)
        with self.assertRaisesRegex(ValueError, "diagnostic|utility"):
            module.evaluate_translation_sample(tampered)

    def test_bundle_swap_between_authentication_and_load_fails_closed(self) -> None:
        module = self.api()
        replacement_bundle = EvaluationBundle(
            Path(self.temporary.name) / "swap-replacement",
            alphas=np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float64),
        )
        original_authenticate = module._authenticate_published_sample

        def authenticate_then_swap(sample: PublishedTranslationSample):
            authenticated = original_authenticate(sample)
            for source, target in (
                (replacement_bundle.long_path, self.bundle.long_path),
                (replacement_bundle.short_path, self.bundle.short_path),
                (replacement_bundle.quality_path, self.bundle.quality_path),
                (replacement_bundle.target_path, self.bundle.target_path),
            ):
                shutil.copyfile(source, target)
            return authenticated

        with mock.patch.object(
            module,
            "_authenticate_published_sample",
            side_effect=authenticate_then_swap,
        ), self.assertRaisesRegex(ValueError, "changed|digest"):
            module.evaluate_translation_sample(self.bundle.sample)

    def test_authenticated_bytes_are_parsed_without_temp_materialization(self) -> None:
        module = self.api()
        with mock.patch(
            "tempfile.TemporaryDirectory",
            side_effect=AssertionError("authenticated bytes must not become temp paths"),
        ):
            metrics = module.evaluate_translation_sample(self.bundle.sample)
        self.assertEqual(
            metrics["provenance"]["target_sha256"],
            self.bundle.sample.target_sha256,
        )

    def test_negative_zero_endpoint_and_q_fov_bit_mutation_fail_closed(self) -> None:
        module = self.api()
        target = load_translation_target(self.bundle.target_path)
        first = tuple(np.argwhere(target["coverage_mask"] == 0)[0])
        target["translation_endpoints"][first] = np.float32(-0.0)
        with self.bundle.target_path.open("wb") as handle:
            np.savez_compressed(handle, **target)
        negative_zero = replace(self.bundle.sample, target_sha256=_sha256(self.bundle.target_path))
        with self.assertRaisesRegex(ValueError, "positive zero"):
            module.evaluate_translation_sample(negative_zero)

        clean = EvaluationBundle(Path(self.temporary.name) / "clean")
        real_apply = module.apply_translation_endpoint

        def corrupt_tail(*args: object, **kwargs: object) -> np.ndarray:
            result = real_apply(*args, **kwargs)
            bits = result.view(np.uint32)
            bits[0, 0, 3] ^= np.uint32(1)
            bits[1, 0, 7] ^= np.uint32(1)
            return result

        with mock.patch.object(module, "apply_translation_endpoint", side_effect=corrupt_tail):
            metrics = module.evaluate_translation_sample(clean.sample)
        self.assertFalse(metrics["quaternion_bytes_equal"])
        self.assertFalse(metrics["fov_bytes_equal"])


class StageAPrimeGateTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(evaluate, "Task 3 evaluate module is missing")
        return evaluate

    def test_classifier_uses_the_frozen_keyword_only_cohort_signature(self) -> None:
        module = self.api()
        parameters = inspect.signature(module.classify_stage_a_prime).parameters
        self.assertEqual(
            tuple(parameters),
            ("scene_metrics", "cohort", "physical_leakage_clean"),
        )
        self.assertIs(
            parameters["cohort"].kind,
            inspect.Parameter.KEYWORD_ONLY,
        )
        self.assertIs(
            parameters["physical_leakage_clean"].kind,
            inspect.Parameter.KEYWORD_ONLY,
        )

    def test_classifier_binds_exact_cohort_and_all_four_sample_digests(self) -> None:
        module = self.api()
        passing = _passing_gate_scenes()
        cohort = _gate_cohort()
        result = module.classify_stage_a_prime(
            passing,
            cohort=cohort,
            physical_leakage_clean=True,
        )
        self.assertEqual(result["classification"], "TRANSLATION_ENDPOINTS_READY")

        malformed_cohorts = (
            ("missing", cohort[:-1]),
            ("extra", cohort + [cohort[0]]),
            ("duplicate", cohort[:-1] + [cohort[0]]),
            (
                "wrong_role",
                [
                    replace(
                        cohort[0],
                        role=(
                            "validation"
                            if cohort[0].role == "train"
                            else "train"
                        ),
                    ),
                    *cohort[1:],
                ],
            ),
        )
        for name, malformed in malformed_cohorts:
            with self.subTest(name=name), self.assertRaisesRegex(
                ValueError, "cohort|role|scene"
            ):
                module.classify_stage_a_prime(
                    passing,
                    cohort=malformed,
                    physical_leakage_clean=True,
                )

        for field in (
            "long_sha256",
            "short_sha256",
            "quality_sha256",
            "target_sha256",
        ):
            wrong_sample_digest = [
                replace(cohort[0], **{field: "9" * 64}),
                *cohort[1:],
            ]
            with self.subTest(sample_digest=field), self.assertRaisesRegex(
                ValueError, "digest|provenance"
            ):
                module.classify_stage_a_prime(
                    passing,
                    cohort=wrong_sample_digest,
                    physical_leakage_clean=True,
                )

            wrong_metric_digest = copy.deepcopy(passing)
            wrong_metric_digest[0]["provenance"][field] = "9" * 64
            with self.subTest(metric_digest=field), self.assertRaisesRegex(
                ValueError, "digest|provenance"
            ):
                module.classify_stage_a_prime(
                    wrong_metric_digest,
                    cohort=cohort,
                    physical_leakage_clean=True,
                )

        identity_mismatch = copy.deepcopy(passing)
        identity_mismatch[0]["sample_id"] = passing[1]["sample_id"]
        with self.assertRaisesRegex(ValueError, "cohort|sample"):
            module.classify_stage_a_prime(
                identity_mismatch,
                cohort=cohort,
                physical_leakage_clean=True,
            )

        with self.assertRaisesRegex(ValueError, "cohort"):
            module.classify_stage_a_prime(
                object(),  # type: ignore[arg-type]
                cohort=[],
                physical_leakage_clean=True,
            )

    def test_requires_exact_frozen_cohort_four_endpoint_ids_and_bool_leakage(self) -> None:
        module = self.api()
        passing = _passing_gate_scenes()
        result = module.classify_stage_a_prime(
            passing,
            cohort=_gate_cohort(),
            physical_leakage_clean=True,
        )
        self.assertEqual(result["classification"], "TRANSLATION_ENDPOINTS_READY")
        self.assertEqual(result["scene_count"], 10)
        self.assertEqual(result["endpoint_count"], 40)

        malformed = (
            passing[:-1],
            passing + [_gate_scene("scene9999_99")],
            passing[:-1] + [copy.deepcopy(passing[0])],
        )
        for rows in malformed:
            with self.subTest(count=len(rows)), self.assertRaisesRegex(
                ValueError, "exact.*cohort|scene"
            ):
                module.classify_stage_a_prime(
                    rows,
                    cohort=_gate_cohort(),
                    physical_leakage_clean=True,
                )
        duplicate_endpoint = copy.deepcopy(passing)
        duplicate_endpoint[0]["endpoint_ids"] = [0, 1, 2, 2]
        duplicate_endpoint[0]["endpoints"][3]["endpoint_id"] = 2
        with self.assertRaisesRegex(ValueError, "endpoint"):
            module.classify_stage_a_prime(
                duplicate_endpoint,
                cohort=_gate_cohort(),
                physical_leakage_clean=True,
            )
        with self.assertRaisesRegex(ValueError, "Boolean"):
            module.classify_stage_a_prime(
                passing,
                cohort=_gate_cohort(),
                physical_leakage_clean=1,
            )
        failed = module.classify_stage_a_prime(
            passing,
            cohort=_gate_cohort(),
            physical_leakage_clean=False,
        )
        self.assertEqual(failed["classification"], "TRANSLATION_ENDPOINTS_FAILED")
        self.assertEqual(failed["failed_gates"], ["physical_leakage_clean"])

    def test_equal_scene_retention_boundary_is_inclusive_and_not_endpoint_weighted(self) -> None:
        module = self.api()
        rows = _passing_gate_scenes()
        rows[0]["mean_covered_utility"] = 0.5
        rows[0]["teacher_retention"] = 0.5
        for endpoint in rows[0]["endpoints"]:
            endpoint["covered_utility"] = 0.5
        for row in rows[1:]:
            row["mean_covered_utility"] = 1.0
            row["teacher_retention"] = 1.0
            for endpoint in row["endpoints"]:
                endpoint["covered_utility"] = 1.0
        result = module.classify_stage_a_prime(
            rows,
            cohort=_gate_cohort(),
            physical_leakage_clean=True,
        )
        self.assertEqual(result["mean_teacher_retention"], 0.95)
        self.assertTrue(result["gates"]["teacher_retention"])
        below = np.nextafter(0.95, -np.inf)
        for row in rows:
            row["mean_covered_utility"] = below
            row["teacher_retention"] = below
            for endpoint in row["endpoints"]:
                endpoint["covered_utility"] = below
        result = module.classify_stage_a_prime(
            rows,
            cohort=_gate_cohort(),
            physical_leakage_clean=True,
        )
        self.assertFalse(result["gates"]["teacher_retention"])
        self.assertEqual(result["classification"], "TRANSLATION_ENDPOINTS_FAILED")

    def test_structural_gate_thresholds_use_the_frozen_strictness(self) -> None:
        module = self.api()
        cases = (
            ("max_covered_roundtrip_fraction", 1e-5, "covered_roundtrip", False),
            (
                "max_covered_roundtrip_fraction",
                np.nextafter(1e-5, -np.inf),
                "covered_roundtrip",
                True,
            ),
            ("max_uncovered_drift_fraction", 1e-8, "uncovered_anchor", False),
            (
                "max_uncovered_drift_fraction",
                np.nextafter(1e-8, -np.inf),
                "uncovered_anchor",
                True,
            ),
            ("max_rotation_delta_deg", 1e-6, "rotation_guard", True),
            (
                "max_rotation_delta_deg",
                np.nextafter(1e-6, np.inf),
                "rotation_guard",
                False,
            ),
        )
        for field, value, gate, expected in cases:
            rows = _passing_gate_scenes()
            rows[0][field] = value
            endpoint_field = {
                "max_covered_roundtrip_fraction": "covered_roundtrip_fraction",
                "max_uncovered_drift_fraction": "uncovered_drift_fraction",
                "max_rotation_delta_deg": "rotation_delta_deg",
            }[field]
            rows[0]["endpoints"][0][endpoint_field] = value
            with self.subTest(field=field, value=value):
                result = module.classify_stage_a_prime(
                    rows,
                    cohort=_gate_cohort(),
                    physical_leakage_clean=True,
                )
                self.assertIs(result["gates"][gate], expected)

        rows = _passing_gate_scenes()
        rows[0]["mean_full_scene_utility"] = 0.0
        for endpoint in rows[0]["endpoints"]:
            endpoint["full_scene_utility"] = 0.0
        result = module.classify_stage_a_prime(
            rows,
            cohort=_gate_cohort(),
            physical_leakage_clean=True,
        )
        self.assertFalse(result["gates"]["positive_scene_count"])
        self.assertTrue(result["gates"]["minimum_full_utility"])
        rows[0]["mean_full_scene_utility"] = np.nextafter(0.0, np.inf)
        for endpoint in rows[0]["endpoints"]:
            endpoint["full_scene_utility"] = np.nextafter(0.0, np.inf)
        result = module.classify_stage_a_prime(
            rows,
            cohort=_gate_cohort(),
            physical_leakage_clean=True,
        )
        self.assertTrue(result["gates"]["positive_scene_count"])

    def test_zero_coverage_endpoint_is_malformed_evidence(self) -> None:
        module = self.api()
        rows = _passing_gate_scenes()
        rows[0]["endpoints"][0]["coverage_fraction"] = 0.0
        with self.assertRaisesRegex(ValueError, "coverage"):
            module.classify_stage_a_prime(
                rows,
                cohort=_gate_cohort(),
                physical_leakage_clean=True,
            )

    def test_each_boolean_and_remaining_utility_gate_is_wired(self) -> None:
        module = self.api()
        boolean_cases = (
            ("all_finite", "finite"),
            ("uncovered_positive_zero", "uncovered_positive_zero"),
            ("quaternion_bytes_equal", "quaternion_bytes_equal"),
            ("fov_bytes_equal", "fov_bytes_equal"),
        )
        for field, gate in boolean_cases:
            rows = _passing_gate_scenes()
            rows[0][field] = False
            rows[0]["endpoints"][0][field] = False
            with self.subTest(gate=gate):
                result = module.classify_stage_a_prime(
                    rows,
                    cohort=_gate_cohort(),
                    physical_leakage_clean=True,
                )
                self.assertFalse(result["gates"][gate])
                self.assertEqual(
                    result["classification"], "TRANSLATION_ENDPOINTS_FAILED"
                )

        rows = _passing_gate_scenes()
        for row in rows:
            row["mean_full_scene_utility"] = 0.0
            for endpoint in row["endpoints"]:
                endpoint["full_scene_utility"] = 0.0
        result = module.classify_stage_a_prime(
            rows,
            cohort=_gate_cohort(),
            physical_leakage_clean=True,
        )
        self.assertFalse(result["gates"]["positive_mean"])

        rows = _passing_gate_scenes()
        rows[0]["mean_full_scene_utility"] = -np.nextafter(0.0, np.inf)
        for endpoint in rows[0]["endpoints"]:
            endpoint["full_scene_utility"] = -np.nextafter(0.0, np.inf)
        result = module.classify_stage_a_prime(
            rows,
            cohort=_gate_cohort(),
            physical_leakage_clean=True,
        )
        self.assertFalse(result["gates"]["minimum_full_utility"])


if __name__ == "__main__":
    unittest.main()
