from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from pre_experiments.variational_camera_selector import pipeline


class SelectorPipelineTests(unittest.TestCase):
    @staticmethod
    def _sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def test_privileged_stage_requires_signed_prediction_barrier(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = pipeline.parse_args(
                [
                    "--stage",
                    "privileged",
                    "--run-root",
                    str(root / "run"),
                    "--input-root",
                    str(root / "sealed_phase1"),
                ]
            )
            evaluator = mock.Mock(
                side_effect=AssertionError("labels were loaded before score barrier")
            )
            with mock.patch.object(pipeline, "evaluate_scene_scores", evaluator):
                with self.assertRaisesRegex(ValueError, "prediction barrier"):
                    pipeline.run_stage(args)
            evaluator.assert_not_called()

    def test_verify_requires_exact_two_validation_scenes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory)
            manifests = run_root / "manifests"
            manifests.mkdir()
            (manifests / "report_complete.json").write_text(
                json.dumps(
                    {
                        "schema": "variational_camera_selector.report_complete.v1",
                        "validation_scenes": ["scene0325_01"],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "validation scenes"):
                pipeline.verify_completed_run(run_root)

    def test_completion_is_idempotent_and_refuses_changed_payload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "complete.json"
            payload = {"schema": "fixture.v1", "digest": "a" * 64}
            first = pipeline.write_exact_json(path, payload)
            second = pipeline.write_exact_json(path, payload)
            self.assertEqual(first, second)
            with self.assertRaisesRegex(ValueError, "immutable JSON"):
                pipeline.write_exact_json(path, {**payload, "digest": "b" * 64})

    def test_score_barrier_rejects_missing_selection_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory)
            manifests = run_root / "manifests"
            manifests.mkdir()
            records = []
            for scene in pipeline.FROZEN_VALIDATION_SCENES:
                score_path = run_root / f"{scene}.score.npz"
                score_path.write_bytes(b"sealed-score")
                records.append(
                    {
                        "scene": scene,
                        "path": str(score_path),
                        "sha256": self._sha256(score_path),
                        "selection_path": str(run_root / f"{scene}.selection.npz"),
                        "selection_sha256": "a" * 64,
                    }
                )
            manifest_path = manifests / "score_manifest.json"
            pipeline.write_exact_json(
                manifest_path,
                {
                    "schema": "variational_camera_selector.score_manifest.v1",
                    "validation_scenes": list(pipeline.FROZEN_VALIDATION_SCENES),
                    "records": records,
                },
            )
            pipeline.write_exact_json(
                manifests / "score_complete.json",
                {
                    "schema": "variational_camera_selector.score_complete.v1",
                    "validation_scenes": list(pipeline.FROZEN_VALIDATION_SCENES),
                    "score_manifest_sha256": self._sha256(manifest_path),
                },
            )

            with mock.patch.object(pipeline, "load_score_shard", return_value={}):
                with self.assertRaisesRegex(ValueError, "selection"):
                    pipeline._require_score_barrier(run_root)

    def test_score_barrier_rejects_selection_bound_to_another_score(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory)
            manifests = run_root / "manifests"
            manifests.mkdir()
            records = []
            for scene in pipeline.FROZEN_VALIDATION_SCENES:
                score_path = run_root / f"{scene}.score.npz"
                selection_path = run_root / f"{scene}.selection.npz"
                score_path.write_bytes(b"sealed-score")
                selection_path.write_bytes(b"sealed-selection")
                records.append(
                    {
                        "scene": scene,
                        "path": str(score_path),
                        "sha256": self._sha256(score_path),
                        "selection_path": str(selection_path),
                        "selection_sha256": self._sha256(selection_path),
                    }
                )
            manifest_path = manifests / "score_manifest.json"
            pipeline.write_exact_json(
                manifest_path,
                {
                    "schema": "variational_camera_selector.score_manifest.v1",
                    "validation_scenes": list(pipeline.FROZEN_VALIDATION_SCENES),
                    "records": records,
                },
            )
            pipeline.write_exact_json(
                manifests / "score_complete.json",
                {
                    "schema": "variational_camera_selector.score_complete.v1",
                    "validation_scenes": list(pipeline.FROZEN_VALIDATION_SCENES),
                    "score_manifest_sha256": self._sha256(manifest_path),
                },
            )

            with (
                mock.patch.object(pipeline, "load_score_shard", return_value={}),
                mock.patch.object(
                    pipeline,
                    "load_selection_shard",
                    return_value={"score_sha256": "b" * 64},
                ),
            ):
                with self.assertRaisesRegex(ValueError, "bind"):
                    pipeline._require_score_barrier(run_root)

    def test_auto_stage_calls_every_barrier_in_order(self) -> None:
        args = pipeline.parse_args(
            [
                "--stage",
                "auto",
                "--run-root",
                "/data/yjh/output/variational_camera_selector/fixture",
            ]
        )
        calls: list[str] = []

        def mark(name):
            return lambda _args: calls.append(name) or _args.run_root

        with (
            mock.patch.object(pipeline, "run_prepare", mark("prepare")),
            mock.patch.object(pipeline, "run_smoke", mark("smoke")),
            mock.patch.object(pipeline, "run_calibration", mark("calibration")),
            mock.patch.object(pipeline, "run_score", mark("score")),
            mock.patch.object(pipeline, "run_privileged", mark("privileged")),
            mock.patch.object(pipeline, "run_report", mark("report")),
            mock.patch.object(pipeline, "verify_completed_run", lambda _root: calls.append("verify") or {}),
        ):
            pipeline.run_stage(args)

        self.assertEqual(
            calls,
            ["prepare", "smoke", "calibration", "score", "privileged", "report", "verify"],
        )


if __name__ == "__main__":
    unittest.main()
