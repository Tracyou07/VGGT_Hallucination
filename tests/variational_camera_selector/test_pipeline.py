from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from pre_experiments.variational_camera_selector import pipeline


class SelectorPipelineTests(unittest.TestCase):
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
