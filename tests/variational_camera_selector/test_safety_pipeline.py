from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path
import tempfile
from unittest import mock
import unittest

import numpy as np

from pre_experiments.variational_camera_selector import safety_pipeline
from pre_experiments.variational_camera_selector.safety_calibration import (
    FrozenGateResult,
    GateMetrics,
    gated_evaluation_from_arrays,
    summarize_gate_validation,
)
from pre_experiments.variational_camera_selector.safety_gate import GatePolicy
from pre_experiments.variational_camera_selector.train import (
    FROZEN_TRAIN_SCENES,
    FROZEN_VALIDATION_SCENES,
)


class SafetyPipelineTests(unittest.TestCase):
    def test_fit_report_is_stable_across_json_round_trip(self) -> None:
        policy = GatePolicy.fail_closed()
        metrics = GateMetrics(
            scene_count=1,
            overlap_count=8,
            selected_count=0,
            positive_count=0,
            catastrophic_count=0,
            coverage=0.0,
            positive_precision=0.0,
            catastrophic_rate=0.0,
            mean_utility=0.0,
            median_utility=0.0,
            worst_scene_mean=0.0,
            per_scene_mean=(("scene_a", 0.0),),
        )
        result = FrozenGateResult(
            policy=policy,
            calibration_metrics=metrics,
            crossfit_metrics=metrics,
            fold_policies=(("scene_a", policy),),
        )

        report = safety_pipeline._fit_report(result)

        self.assertEqual(report, json.loads(json.dumps(report)))

    def test_prediction_only_base_context_does_not_require_label_manifests(self) -> None:
        # The apply stage must work before privileged/evaluation manifests are readable.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifests = root / "manifests"
            manifests.mkdir()
            (root / "verified_completion.json").write_text(
                json.dumps(
                    {
                        "schema": "variational_camera_selector.verified_completion.v1",
                        "train_scenes": list(FROZEN_TRAIN_SCENES),
                        "validation_scenes": list(FROZEN_VALIDATION_SCENES),
                        "completed_step": 800,
                    }
                ),
                encoding="utf-8",
            )

            candidate_records = []
            for scene in FROZEN_TRAIN_SCENES + FROZEN_VALIDATION_SCENES:
                row = {"scene": scene}
                for stem in ("long_context", "candidate", "residual_prediction"):
                    path = root / f"{scene}.{stem}.bin"
                    path.write_bytes(f"{scene}:{stem}".encode("utf-8"))
                    row[f"{stem}_path"] = str(path)
                    row[f"{stem}_sha256"] = safety_pipeline._sha256_file(path)
                candidate_records.append(row)
            (manifests / "candidate_binding_manifest.json").write_text(
                json.dumps(
                    {
                        "schema": "variational_camera_selector.candidate_binding_manifest.v1",
                        "records": candidate_records,
                    }
                ),
                encoding="utf-8",
            )

            score_records = []
            for scene in FROZEN_VALIDATION_SCENES:
                path = root / f"{scene}.scores.bin"
                path.write_bytes(scene.encode("utf-8"))
                score_records.append(
                    {
                        "scene": scene,
                        "path": str(path),
                        "sha256": safety_pipeline._sha256_file(path),
                    }
                )
            (manifests / "score_manifest.json").write_text(
                json.dumps(
                    {
                        "schema": "variational_camera_selector.score_manifest.v1",
                        "records": score_records,
                    }
                ),
                encoding="utf-8",
            )

            context = safety_pipeline._base_context(
                root, require_privileged=False, require_evaluation=False
            )

            self.assertNotIn("privileged_manifest", context)
            self.assertNotIn("evaluation_manifest", context)

    def test_oof_base_context_hashes_only_the_fold_training_labels(self) -> None:
        held = FROZEN_TRAIN_SCENES[0]
        training = tuple(scene for scene in FROZEN_TRAIN_SCENES if scene != held)
        verified = {
            "schema": "variational_camera_selector.verified_completion.v1",
            "train_scenes": list(FROZEN_TRAIN_SCENES),
            "validation_scenes": list(FROZEN_VALIDATION_SCENES),
            "completed_step": 800,
        }
        candidate_records = [
            {"scene": scene}
            for scene in FROZEN_TRAIN_SCENES + FROZEN_VALIDATION_SCENES
        ]
        score_records = [{"scene": scene} for scene in FROZEN_VALIDATION_SCENES]
        privileged_records = [
            {"scene": scene, "path": f"/{scene}.labels", "sha256": "a" * 64}
            for scene in FROZEN_TRAIN_SCENES + FROZEN_VALIDATION_SCENES
        ]

        def records_for(path: Path, **_kwargs):
            if path.name == "candidate_binding_manifest.json":
                return {}, candidate_records
            if path.name == "score_manifest.json":
                return {}, score_records
            if path.name == "privileged_binding_manifest.json":
                return {}, privileged_records
            raise AssertionError(path)

        with (
            mock.patch.object(safety_pipeline, "_read_json", return_value=verified),
            mock.patch.object(safety_pipeline, "_records", side_effect=records_for),
            mock.patch.object(
                safety_pipeline, "_verify_base_artifact_records"
            ) as verifier,
        ):
            safety_pipeline._base_context(
                Path("/fake/base"),
                require_privileged=True,
                privileged_scenes=training,
            )

        privileged_call = verifier.call_args_list[-1]
        observed = tuple(row["scene"] for row in privileged_call.args[1])
        self.assertEqual(observed, training)
        self.assertNotIn(held, observed)

    def test_base_artifact_verifier_understands_candidate_binding_fields(self) -> None:
        # Candidate bindings name three artifacts explicitly; they have no generic path.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifacts = {}
            for name in ("long", "candidate", "residual", "privileged"):
                path = root / f"{name}.bin"
                path.write_bytes(name.encode("utf-8"))
                artifacts[name] = path
            candidate_record = {
                "long_context_path": str(artifacts["long"]),
                "long_context_sha256": safety_pipeline._sha256_file(artifacts["long"]),
                "candidate_path": str(artifacts["candidate"]),
                "candidate_sha256": safety_pipeline._sha256_file(
                    artifacts["candidate"]
                ),
                "residual_prediction_path": str(artifacts["residual"]),
                "residual_prediction_sha256": safety_pipeline._sha256_file(
                    artifacts["residual"]
                ),
            }
            generic_record = {
                "path": str(artifacts["privileged"]),
                "sha256": safety_pipeline._sha256_file(artifacts["privileged"]),
            }

            safety_pipeline._verify_base_artifact_records(
                [candidate_record], [generic_record]
            )

            candidate_record["candidate_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "candidate_path"):
                safety_pipeline._verify_base_artifact_records(
                    [candidate_record], [generic_record]
                )

    def test_each_oof_fold_excludes_exactly_its_held_scene(self) -> None:
        # Catches training a fold on the scene later presented as out-of-fold.
        for held in FROZEN_TRAIN_SCENES:
            observed = safety_pipeline.fold_training_scenes(held)
            self.assertEqual(len(observed), 7)
            self.assertNotIn(held, observed)
            self.assertEqual(
                observed,
                tuple(scene for scene in FROZEN_TRAIN_SCENES if scene != held),
            )
        with self.assertRaisesRegex(ValueError, "frozen train"):
            safety_pipeline.fold_training_scenes(FROZEN_VALIDATION_SCENES[0])

    def test_finalize_preserves_prediction_barrier_before_label_access(self) -> None:
        # Catches fitting or evaluating with labels before all OOF scores are sealed.
        calls: list[str] = []

        def mark(name: str):
            return lambda _args: calls.append(name)

        with (
            mock.patch.object(safety_pipeline, "run_collect", mark("collect")),
            mock.patch.object(safety_pipeline, "run_privileged", mark("privileged")),
            mock.patch.object(safety_pipeline, "run_fit", mark("fit")),
            mock.patch.object(safety_pipeline, "run_apply", mark("apply")),
            mock.patch.object(safety_pipeline, "run_evaluate", mark("evaluate")),
            mock.patch.object(safety_pipeline, "run_verify", mark("verify")),
        ):
            safety_pipeline.run_stage(Namespace(stage="finalize"))

        self.assertEqual(
            calls,
            ["collect", "privileged", "fit", "apply", "evaluate", "verify"],
        )

    def test_gated_evaluation_joins_by_digest_ids_and_decision(self) -> None:
        # Catches applying the gate mask to a different score/evaluation artifact.
        sample_ids = np.asarray([f"sample:{index}" for index in range(8)], dtype="U32")
        raw = {
            "scene": np.asarray("scene_a", dtype="U32"),
            "source_sample_ids": sample_ids,
            "full_context_selected_indices": np.arange(1, 9, dtype=np.int64),
            "full_context_utility": np.asarray(
                [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8], dtype=np.float64
            ),
            "score_sha256": np.asarray("a" * 64, dtype="U64"),
        }
        gated = {
            "scene": np.asarray("scene_a", dtype="U32"),
            "source_sample_ids": sample_ids.copy(),
            "proposed_indices": np.arange(1, 9, dtype=np.int64),
            "selected_indices": np.asarray([1, 0, 3, 0, 5, 0, 7, 0], dtype=np.int64),
            "gate_pass": np.asarray([True, False] * 4, dtype=np.bool_),
            "score_sha256": np.asarray("a" * 64, dtype="U64"),
            "policy_sha256": np.asarray("b" * 64, dtype="U64"),
        }
        evaluation = gated_evaluation_from_arrays(gated, raw)
        np.testing.assert_allclose(
            evaluation["gated_utility"],
            np.asarray([0.1, 0.0, 0.3, 0.0, 0.5, 0.0, 0.7, 0.0]),
        )

        gated["score_sha256"] = np.asarray("c" * 64, dtype="U64")
        with self.assertRaisesRegex(ValueError, "score"):
            gated_evaluation_from_arrays(gated, raw)

    def test_validation_summary_counts_noops_in_the_denominator(self) -> None:
        # Catches reporting only executed corrections and hiding abstentions.
        records = []
        for scene in ("scene_a", "scene_b"):
            records.append(
                {
                    "scene": np.asarray(scene, dtype="U32"),
                    "gate_pass": np.asarray([True, False] * 4, dtype=np.bool_),
                    "gated_utility": np.asarray([0.1, 0.0] * 4, dtype=np.float64),
                }
            )
        summary = summarize_gate_validation(records)

        self.assertEqual(summary["scene_count"], 2)
        self.assertEqual(summary["overlap_count"], 16)
        self.assertEqual(summary["selected_count"], 8)
        self.assertAlmostEqual(summary["coverage"], 0.5)
        self.assertAlmostEqual(summary["mean_utility"], 0.05)

    def test_validation_summary_does_not_call_one_lucky_choice_safe(self) -> None:
        # One positive correction out of 16 is below the frozen 12.5% coverage floor.
        records = []
        for scene_index, scene in enumerate(("scene_a", "scene_b")):
            gate_pass = np.zeros(8, dtype=np.bool_)
            utility = np.zeros(8, dtype=np.float64)
            if scene_index == 0:
                gate_pass[0] = True
                utility[0] = 0.2
            records.append(
                {
                    "scene": np.asarray(scene, dtype="U32"),
                    "gate_pass": gate_pass,
                    "gated_utility": utility,
                }
            )

        summary = summarize_gate_validation(records)

        self.assertEqual(summary["selected_count"], 1)
        self.assertEqual(summary["classification"], "UNSAFE_GENERALIZATION")


if __name__ == "__main__":
    unittest.main()
