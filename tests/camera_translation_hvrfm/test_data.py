from __future__ import annotations

import inspect
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np
import torch

from pre_experiments.camera_translation_hvrfm.artifacts import (
    LONG_CONTEXT_MEMBERS,
    QUALITY_SIDECAR_MEMBERS,
    SHORT_CONTEXT_MEMBERS,
    TRANSLATION_TARGET_MEMBERS,
    load_bound_bundle,
)
from pre_experiments.camera_translation_hvrfm.data import (
    PublishedTranslationSample,
    calibration_role,
    publish_translation_sample,
    validate_calibration_cohort,
)
from pre_experiments.conditional_hierarchical_vrfm.artifacts import save_teacher_artifact
from pre_experiments.variational_camera_latent.contracts import SourceShardRecord
from pre_experiments.variational_camera_latent.source import save_source_shard
from tests.camera_translation_hvrfm.test_teacher import (
    CHECKPOINT_SHA256,
    FORMAL_LABEL_SHA256,
    SCENE,
    TokenCameraHead,
    make_reference_arrays,
    make_source_arrays,
    sha256_file,
)


def current_git_commit(worktree: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=worktree,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


class PublicationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.output = self.root / "published"
        self.source = make_source_arrays()
        self.source_path = self.root / "relocated-source.npz"
        save_source_shard(self.source_path, self.source)
        self.source_sha256 = sha256_file(self.source_path)
        self.formal_path = self.root / "formal-label.json"
        self.formal_path.write_bytes(b"authenticated formal label\n")
        self.formal_sha256 = sha256_file(self.formal_path)
        self.reference_path = self.root / "teacher-reference.npz"
        save_teacher_artifact(
            self.reference_path,
            make_reference_arrays(
                self.source,
                source_sha256=self.source_sha256,
                formal_label_sha256=self.formal_sha256,
            ),
        )
        self.reference_sha256 = sha256_file(self.reference_path)
        self.worktree = Path(__file__).resolve().parents[2]
        self.git_commit = current_git_commit(self.worktree)
        self.record = SourceShardRecord(
            scene=SCENE,
            role="train",
            path=Path("/stale/absent/source.npz"),
            overlap_count=8,
            sha256=self.source_sha256,
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def publish(self, **overrides: object) -> PublishedTranslationSample:
        arguments = {
            "output_root": self.output,
            "role": "train",
            "source_path": self.source_path,
            "source_record": self.record,
            "teacher_reference_path": self.reference_path,
            "expected_teacher_reference_sha256": self.reference_sha256,
            "formal_label_path": self.formal_path,
            "expected_formal_label_sha256": self.formal_sha256,
            "camera_head": TokenCameraHead(),
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "git_commit": self.git_commit,
            "device": torch.device("cpu"),
        }
        arguments.update(overrides)
        return publish_translation_sample(**arguments)

    def test_publication_signature_has_no_prepared_scene_or_gt_input(self) -> None:
        names = set(inspect.signature(publish_translation_sample).parameters)
        self.assertFalse(
            any(
                forbidden in name
                for name in names
                for forbidden in ("prepared", "gt_path", "oracle_path")
            )
        )

    def test_publishes_four_physically_separated_bound_artifacts(self) -> None:
        published = self.publish()
        self.assertEqual(published.sample_id, f"{SCENE}:frames_500")
        self.assertEqual(published.scene, SCENE)
        self.assertEqual(published.role, "train")
        expected_paths = {
            "long": self.output / "prediction_only" / "long_context" / f"{SCENE}.npz",
            "short": self.output / "privileged_training" / "short_context" / f"{SCENE}.npz",
            "quality": self.output / "privileged_labels" / "quality" / f"{SCENE}.npz",
            "target": self.output / "privileged_labels" / "translation_targets" / f"{SCENE}.npz",
        }
        self.assertEqual(published.long_path, expected_paths["long"])
        self.assertEqual(published.short_path, expected_paths["short"])
        self.assertEqual(published.quality_path, expected_paths["quality"])
        self.assertEqual(published.target_path, expected_paths["target"])
        bundle = load_bound_bundle(
            published.long_path,
            published.short_path,
            published.target_path,
            published.quality_path,
        )
        self.assertEqual(set(bundle["long"]), set(LONG_CONTEXT_MEMBERS))
        self.assertEqual(set(bundle["short"]), set(SHORT_CONTEXT_MEMBERS))
        self.assertEqual(set(bundle["quality"]), set(QUALITY_SIDECAR_MEMBERS))
        self.assertEqual(set(bundle["target"]), set(TRANSLATION_TARGET_MEMBERS))
        self.assertNotIn("gt_c2w", bundle["long"])
        self.assertNotIn("window_weights", bundle["target"])
        self.assertEqual(str(bundle["target"]["teacher_reference_sha256"]), self.reference_sha256)
        self.assertEqual(str(bundle["quality"]["formal_label_sha256"]), self.formal_sha256)
        self.assertEqual(str(bundle["long"]["git_commit"]), self.git_commit)
        self.assertEqual(str(bundle["long"]["source_sha256"]), self.source_sha256)
        self.assertEqual(published.long_sha256, sha256_file(published.long_path))
        self.assertEqual(published.short_sha256, sha256_file(published.short_path))
        self.assertEqual(published.quality_sha256, sha256_file(published.quality_path))
        self.assertEqual(published.target_sha256, sha256_file(published.target_path))

    def test_explicit_relocation_ignores_stale_manifest_path_but_authenticates_record(self) -> None:
        self.assertFalse(self.record.path.exists())
        published = self.publish()
        self.assertTrue(published.long_path.is_file())
        wrong = SourceShardRecord(
            scene=SCENE,
            role="train",
            path=self.source_path,
            overlap_count=8,
            sha256="0" * 64,
        )
        with self.assertRaisesRegex(ValueError, "source.*digest"):
            self.publish(output_root=self.root / "wrong", source_record=wrong)

    def test_refuses_wrong_reference_formal_checkpoint_and_current_git_bindings(self) -> None:
        cases = {
            "teacher": {"expected_teacher_reference_sha256": "0" * 64},
            "formal": {"expected_formal_label_sha256": "0" * 64},
            "checkpoint": {"checkpoint_sha256": "0" * 64},
            "git": {"git_commit": "0" * 40},
        }
        for label, override in cases.items():
            with self.subTest(label=label), self.assertRaises(ValueError):
                self.publish(output_root=self.root / label, **override)

    def test_smoke_is_a_pipeline_stage_not_a_persisted_data_role(self) -> None:
        with self.assertRaisesRegex(ValueError, "train.*validation"):
            self.publish(output_root=self.root / "smoke", role="smoke")

    def test_quality_rejects_zero_baseline_rms_denominator(self) -> None:
        degenerate = make_reference_arrays(
            self.source,
            source_sha256=self.source_sha256,
            formal_label_sha256=self.formal_sha256,
        )
        degenerate["gt_c2w"] = degenerate["baseline_c2w_raw"].copy()
        centers = degenerate["gt_c2w"][:, :3, 3]
        centered = centers - centers.mean(axis=0)
        degenerate["gt_scene_scale"] = np.asarray(
            np.sqrt(np.mean(np.sum(centered * centered, axis=1))), dtype=np.float64
        )
        path = self.root / "zero-rms-reference.npz"
        save_teacher_artifact(path, degenerate)
        with self.assertRaisesRegex(ValueError, "baseline RMS"):
            self.publish(
                output_root=self.root / "zero-rms",
                teacher_reference_path=path,
                expected_teacher_reference_sha256=sha256_file(path),
            )

    def test_gt_oracle_diagnostic_and_legacy_utility_mutations_do_not_change_numeric_target(self) -> None:
        first = self.publish()
        first_bundle = load_bound_bundle(
            first.long_path, first.short_path, first.target_path, first.quality_path
        )
        mutated_arrays = make_reference_arrays(
            self.source,
            source_sha256=self.source_sha256,
            formal_label_sha256=self.formal_sha256,
        )
        mutated_arrays["gt_c2w"][:, 1, 3] += 0.03 * np.sin(
            np.linspace(0.0, 11.0, 500)
        )
        centers = mutated_arrays["gt_c2w"][:, :3, 3]
        centered = centers - centers.mean(axis=0)
        mutated_arrays["gt_scene_scale"] = np.asarray(
            np.sqrt(np.mean(np.sum(centered * centered, axis=1))), dtype=np.float64
        )
        mutated_arrays["oracle_condition"] = np.asarray(9.0, dtype=np.float64)
        mutated_arrays["variant_utilities"] = np.asarray(
            [99.0, -40.0, 3.0, 7.0], dtype=np.float64
        )
        mutated_path = self.root / "mutated-reference.npz"
        save_teacher_artifact(mutated_path, mutated_arrays)
        second = self.publish(
            output_root=self.root / "mutated-output",
            teacher_reference_path=mutated_path,
            expected_teacher_reference_sha256=sha256_file(mutated_path),
        )
        second_bundle = load_bound_bundle(
            second.long_path, second.short_path, second.target_path, second.quality_path
        )
        np.testing.assert_array_equal(
            second_bundle["target"]["translation_endpoints"],
            first_bundle["target"]["translation_endpoints"],
        )
        np.testing.assert_array_equal(
            second_bundle["target"]["teacher_centers_raw_filled"],
            first_bundle["target"]["teacher_centers_raw_filled"],
        )
        np.testing.assert_array_equal(
            second_bundle["quality"]["coverage_weights"],
            first_bundle["quality"]["coverage_weights"],
        )
        self.assertFalse(
            np.array_equal(
                second_bundle["quality"]["baseline_translation_error_normalized"],
                first_bundle["quality"]["baseline_translation_error_normalized"],
            )
        )
        self.assertNotEqual(second.quality_sha256, first.quality_sha256)
        self.assertNotEqual(second.target_sha256, first.target_sha256)

    def test_quality_metrics_and_utilities_are_recomputed_independently(self) -> None:
        published = self.publish()
        bundle = load_bound_bundle(
            published.long_path,
            published.short_path,
            published.target_path,
            published.quality_path,
        )
        quality = bundle["quality"]
        long = bundle["long"]
        baseline_centers = long["baseline_c2w"][:, :3, 3]
        gt_centers = quality["gt_c2w"][:, :3, 3]
        expected_baseline = np.linalg.norm(
            baseline_centers - gt_centers, axis=1
        ) / float(quality["gt_scene_scale"])
        np.testing.assert_allclose(
            quality["baseline_translation_error_normalized"],
            expected_baseline,
            atol=1e-12,
            rtol=0.0,
        )
        covered = quality["coverage_weights"] > 0.0
        expected_utilities = np.zeros(4, dtype=np.float64)
        teacher_error = quality["teacher_translation_error_normalized"]
        for endpoint in range(4):
            mask = covered[endpoint]
            baseline_rms = np.sqrt(np.mean(expected_baseline[mask] ** 2))
            teacher_rms = np.sqrt(np.mean(teacher_error[endpoint, mask] ** 2))
            expected_utilities[endpoint] = (
                baseline_rms - teacher_rms
            ) / max(float(baseline_rms), 1e-12)
        np.testing.assert_allclose(
            quality["variant_utilities"], expected_utilities, atol=1e-12, rtol=0.0
        )

    def test_preexisting_conflict_is_rejected_before_any_new_publication(self) -> None:
        target = (
            self.output
            / "privileged_labels"
            / "translation_targets"
            / f"{SCENE}.npz"
        )
        target.parent.mkdir(parents=True)
        target.write_bytes(b"do not replace")
        before = target.read_bytes()
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.publish()
        self.assertEqual(target.read_bytes(), before)
        self.assertFalse(
            (self.output / "prediction_only" / "long_context" / f"{SCENE}.npz").exists()
        )

    def test_publication_rejects_lexical_parent_traversal_on_every_path_boundary(self) -> None:
        safe = self.root / "safe"
        safe.mkdir()
        cases = (
            {
                "output_root": safe / ".." / "lexical-output",
            },
            {
                "output_root": self.root / "source-output",
                "source_path": safe / ".." / self.source_path.name,
            },
            {
                "output_root": self.root / "reference-output",
                "teacher_reference_path": safe / ".." / self.reference_path.name,
            },
            {
                "output_root": self.root / "formal-output",
                "formal_label_path": safe / ".." / self.formal_path.name,
            },
        )
        for index, overrides in enumerate(cases):
            with self.subTest(index=index), self.assertRaisesRegex(
                ValueError, "parent traversal"
            ):
                self.publish(**overrides)


class CalibrationCohortTests(unittest.TestCase):
    def test_frozen_ten_scene_roles_are_exactly_eight_two_with_smoke_scene(self) -> None:
        scenes = (
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
        roles = [calibration_role(scene) for scene in scenes]
        self.assertEqual(roles.count("train"), 8)
        self.assertEqual(roles.count("validation"), 2)
        self.assertEqual(calibration_role(SCENE), "train")
        samples = [
            PublishedTranslationSample.placeholder(scene, calibration_role(scene))
            for scene in scenes
        ]
        validate_calibration_cohort(samples)
        with self.assertRaises(ValueError):
            validate_calibration_cohort(samples[:-1])
        with self.assertRaises(ValueError):
            calibration_role("scene9999_99")


if __name__ == "__main__":
    unittest.main()
