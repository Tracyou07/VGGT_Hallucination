from __future__ import annotations

from contextlib import ExitStack
from contextlib import redirect_stdout
import hashlib
import inspect
from io import StringIO
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import numpy as np

from pre_experiments.camera_translation_hvrfm.artifacts import LONG_CONTEXT_MEMBERS
from pre_experiments.camera_translation_hvrfm.data import PublishedTranslationSample

try:
    from pre_experiments.camera_translation_hvrfm import stages
except (ImportError, ModuleNotFoundError):
    stages = None  # type: ignore[assignment]


SCENES = (
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
SMOKE_SCENE = "scene0029_01"
READY = "TRANSLATION_ENDPOINTS_READY"
FAILED = "TRANSLATION_ENDPOINTS_FAILED"
GIT_COMMIT = "7" * 40


def _digest(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()


def _json_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
            ensure_ascii=True,
        )
        + "\n"
    ).encode("utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(payload))


def _signed_stage(
    *,
    schema: str,
    stage: str,
    run_id: str,
    git_commit: str,
    run_config_sha256: str,
    previous_marker_sha256: str,
    files: dict[str, str],
    metadata: dict[str, object],
) -> dict[str, object]:
    unsigned = {
        "schema": schema,
        "stage": stage,
        "run_id": run_id,
        "git_commit": git_commit,
        "run_config_sha256": run_config_sha256,
        "previous_marker_sha256": previous_marker_sha256,
        "files": files,
        "metadata": metadata,
    }
    return {**unsigned, "completion_digest": _digest(unsigned)}


def _metric(scene: str, *, covered: float = 1e-7) -> dict[str, object]:
    endpoints = [
        {
            "endpoint_id": endpoint,
            "covered_utility": 0.2,
            "teacher_covered_utility": 0.2,
            "full_scene_utility": 0.1,
            "covered_roundtrip_fraction": covered,
            "uncovered_drift_fraction": 0.0,
            "rotation_delta_deg": 0.0,
            "quaternion_bytes_equal": True,
            "fov_bytes_equal": True,
            "uncovered_positive_zero": True,
            "endpoint_rms": 0.1,
            "coverage_fraction": 0.5,
            "all_finite": True,
        }
        for endpoint in range(4)
    ]
    return {
        "scene": scene,
        "sample_id": f"{scene}:frames_500",
        "role": "validation" if scene in {"scene0325_01", "scene0675_00"} else "train",
        "endpoint_count": 4,
        "endpoint_ids": [0, 1, 2, 3],
        "endpoints": endpoints,
        "mean_covered_utility": 0.2,
        "mean_teacher_covered_utility": 0.2,
        "teacher_retention": 1.0,
        "mean_full_scene_utility": 0.1,
        "max_covered_roundtrip_fraction": covered,
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
            "teacher_reference_sha256": "8" * 64,
            "git_commit": GIT_COMMIT,
        },
    }


def _classification(value: str = READY) -> dict[str, object]:
    return {
        "classification": value,
        "failed_gates": [] if value == READY else ["physical_leakage_clean"],
        "gates": {"physical_leakage_clean": value == READY},
        "scene_count": 10,
        "endpoint_count": 40,
        "mean_teacher_retention": 1.0,
        "mean_full_scene_utility": 0.1,
        "minimum_full_scene_utility": 0.1,
        "positive_scene_count": 10,
    }


class PreparedRun:
    def __init__(self, parent: Path) -> None:
        self.root = parent / "stage-a-prime-test"
        self.root.mkdir()
        self.inputs = SimpleNamespace(
            run_root=self.root,
            git_commit=GIT_COMMIT,
            checkpoint_dir=parent / "checkpoint",
        )
        self.inputs.checkpoint_dir.mkdir()
        self.samples: list[PublishedTranslationSample] = []

        preflight = self.root / "manifests" / "preflight_evidence.json"
        config = self.root / "config.json"
        long_manifest = self.root / "manifests" / "long_context.json"
        cohort_manifest = self.root / "manifests" / "cohort.json"
        _write_json(preflight, {"fixture": "authenticated preflight"})
        _write_json(config, {"fixture": "run config"})
        _write_json(long_manifest, {"fixture": "long manifest"})
        _write_json(cohort_manifest, {"fixture": "cohort manifest"})

        files = {
            "config.json": _sha(config),
            "manifests/long_context.json": _sha(long_manifest),
            "manifests/cohort.json": _sha(cohort_manifest),
        }
        for scene in SCENES:
            paths = {
                "long": self.root / "prediction_only" / "long_context" / f"{scene}.npz",
                "short": self.root / "privileged_training" / "short_context" / f"{scene}.npz",
                "quality": self.root / "privileged_labels" / "quality" / f"{scene}.npz",
                "target": self.root
                / "privileged_labels"
                / "translation_targets"
                / f"{scene}.npz",
            }
            digests: dict[str, str] = {}
            for kind, path in paths.items():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(f"{kind}:{scene}\n".encode("ascii"))
                digests[kind] = _sha(path)
                files[path.relative_to(self.root).as_posix()] = digests[kind]
            self.samples.append(
                PublishedTranslationSample(
                    sample_id=f"{scene}:frames_500",
                    scene=scene,
                    role=(
                        "validation"
                        if scene in {"scene0325_01", "scene0675_00"}
                        else "train"
                    ),
                    long_path=paths["long"],
                    short_path=paths["short"],
                    quality_path=paths["quality"],
                    target_path=paths["target"],
                    long_sha256=digests["long"],
                    short_sha256=digests["short"],
                    quality_sha256=digests["quality"],
                    target_sha256=digests["target"],
                )
            )
        self.prepare = self.root / "prepare" / "completed.json"
        _write_json(
            self.prepare,
            _signed_stage(
                schema="camera_translation_hvrfm.prepare_completion.v1",
                stage="prepare",
                run_id=self.root.name,
                git_commit=GIT_COMMIT,
                run_config_sha256=_sha(config),
                previous_marker_sha256=_sha(preflight),
                files=dict(sorted(files.items())),
                metadata={
                    "scene_count": 10,
                    "endpoint_count": 40,
                    "smoke_scene": SMOKE_SCENE,
                },
            ),
        )

    def loader(self, run_root: Path, **kwargs: object):
        if Path(run_root) != self.root:
            raise AssertionError("loader received the wrong run root")
        allowed = kwargs.get("allowed_downstream_files", frozenset())
        if not isinstance(allowed, frozenset):
            raise AssertionError("downstream allowlist must be immutable")
        return tuple(self.samples)

    def write_report(self, run_root: Path, payload: dict[str, object]):
        root = Path(run_root)
        report_json = root / "reports" / "stage_a_prime.json"
        report_markdown = root / "reports" / "stage_a_prime.md"
        completion = root / "reports" / "completed.json"
        report_json.parent.mkdir(parents=True, exist_ok=True)
        report_json.write_bytes(_json_bytes(payload))
        report_markdown.write_text(
            f"classification: {payload['classification']}\n", encoding="utf-8"
        )
        unsigned = {
            "schema": "camera_translation_hvrfm.stage_a_prime_completion.v1",
            "run_id": root.name,
            "git_commit": GIT_COMMIT,
            "classification": payload["classification"],
            "scene_count": 10,
            "endpoint_count": 40,
            "report_json_path": "reports/stage_a_prime.json",
            "report_json_sha256": _sha(report_json),
            "report_markdown_path": "reports/stage_a_prime.md",
            "report_markdown_sha256": _sha(report_markdown),
        }
        _write_json(completion, {**unsigned, "completion_digest": _digest(unsigned)})
        return report_json, report_markdown, completion


class StageOrchestratorTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(stages, "Task 4c stage orchestrator module is missing")
        return stages

    def patch_dependencies(
        self,
        module,
        fixture: PreparedRun,
        *,
        classification: str = READY,
        audit: bool = True,
        evaluator=None,
    ) -> ExitStack:
        stack = ExitStack()
        stack.enter_context(
            mock.patch.object(module, "load_published_cohort", side_effect=fixture.loader)
        )
        stack.enter_context(
            mock.patch.object(
                module,
                "evaluate_translation_sample",
                side_effect=(evaluator or (lambda sample: _metric(sample.scene))),
            )
        )
        stack.enter_context(
            mock.patch.object(
                module,
                "classify_stage_a_prime",
                side_effect=lambda *args, **kwargs: _classification(classification),
            )
        )
        stack.enter_context(
            mock.patch.object(
                module, "_audit_prediction_only_tree", return_value=audit
            )
        )
        stack.enter_context(
            mock.patch.object(
                module,
                "build_stage_a_prime_report",
                side_effect=lambda rows, *, cohort, run_id, git_commit,
                physical_leakage_clean: {
                    "schema": "camera_translation_hvrfm.stage_a_prime_report.v1",
                    "run_id": run_id,
                    "git_commit": git_commit,
                    **_classification(classification),
                    "physical_leakage_clean": physical_leakage_clean,
                    "scene_metrics": list(rows),
                    "cohort": [
                        {
                            "scene": sample.scene,
                            "sample_id": sample.sample_id,
                            "role": sample.role,
                            "long_sha256": sample.long_sha256,
                            "short_sha256": sample.short_sha256,
                            "quality_sha256": sample.quality_sha256,
                            "target_sha256": sample.target_sha256,
                        }
                        for sample in cohort
                    ],
                },
            )
        )
        stack.enter_context(
            mock.patch.object(
                module,
                "write_stage_a_prime_report",
                side_effect=fixture.write_report,
            )
        )
        return stack

    def test_public_api_is_narrow_and_runner_orders_every_fresh_stage(self) -> None:
        module = self.api()
        self.assertEqual(
            str(inspect.signature(module.run_smoke)),
            "(inputs: 'PipelineInputs') -> 'Path'",
        )
        self.assertEqual(
            str(inspect.signature(module.run_calibration)),
            "(inputs: 'PipelineInputs') -> 'Path'",
        )
        self.assertEqual(
            str(inspect.signature(module.run_report)),
            "(inputs: 'PipelineInputs') -> 'tuple[Path, Path, Path]'",
        )
        self.assertEqual(
            str(inspect.signature(module.run_all)),
            "(inputs: 'PipelineInputs') -> 'Path'",
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "ordered-run"
            inputs = SimpleNamespace(
                run_root=root,
                git_commit=GIT_COMMIT,
                checkpoint_dir=Path(directory) / "checkpoint",
            )
            events: list[str] = []

            def stage(name: str, result: Path):
                def invoke(*args, **kwargs):
                    events.append(name)
                    return result

                return invoke

            calibration = root / "calibration" / "completed.json"

            def calibration_stage(*args, **kwargs):
                events.append("calibration")
                calibration.parent.mkdir(parents=True, exist_ok=True)
                _write_json(calibration, {"metadata": {"classification": READY}})
                return calibration

            with (
                mock.patch.object(
                    module,
                    "run_preflight",
                    side_effect=stage(
                        "preflight", root / "manifests" / "preflight_evidence.json"
                    ),
                ),
                mock.patch.object(
                    module,
                    "run_prepare",
                    side_effect=stage("prepare", root / "prepare" / "completed.json"),
                ),
                mock.patch.object(
                    module,
                    "run_smoke",
                    side_effect=stage("smoke", root / "smoke" / "completed.json"),
                ),
                mock.patch.object(
                    module, "run_calibration", side_effect=calibration_stage
                ),
                mock.patch.object(
                    module,
                    "run_report",
                    side_effect=stage(
                        "report",
                        (
                            root / "reports" / "stage_a_prime.json",
                            root / "reports" / "stage_a_prime.md",
                            root / "reports" / "completed.json",
                        ),
                    ),
                ),
                mock.patch.object(
                    module,
                    "verify_completed_run",
                    side_effect=stage("verify", root / "verified_completion.json"),
                ),
            ):
                result = module.run_all(inputs)
            self.assertEqual(result, root / "verified_completion.json")
            self.assertEqual(
                events,
                ["preflight", "prepare", "smoke", "calibration", "report", "verify"],
            )

    def test_module_cli_builds_exact_inputs_and_dispatches_all(self) -> None:
        module = self.api()
        self.assertEqual(
            str(inspect.signature(module.main)),
            "(argv: 'Sequence[str] | None' = None) -> 'int'",
        )
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            run_root = base / "cli-run"
            expected = run_root / "verified_completion.json"
            captured: list[object] = []

            def invoke(inputs):
                captured.append(inputs)
                return expected

            argv = [
                "all",
                "--run-root",
                str(run_root),
                "--git-commit",
                GIT_COMMIT,
                "--source-run",
                str(base / "source"),
                "--reference-run",
                str(base / "reference"),
                "--formal-run",
                str(base / "formal"),
                "--checkpoint-dir",
                str(base / "checkpoint"),
                "--expected-source-completion-sha256",
                "1" * 64,
                "--expected-reference-completion-sha256",
                "2" * 64,
                "--expected-formal-completion-sha256",
                "3" * 64,
                "--expected-checkpoint-sha256",
                "4" * 64,
                "--device",
                "cpu",
            ]
            output = StringIO()
            with (
                mock.patch.object(module, "run_all", side_effect=invoke),
                redirect_stdout(output),
            ):
                result = module.main(argv)
            self.assertEqual(result, 0)
            self.assertEqual(output.getvalue(), f"{expected}\n")
            self.assertEqual(len(captured), 1)
            inputs = captured[0]
            self.assertEqual(inputs.run_root, run_root)
            self.assertEqual(inputs.git_commit, GIT_COMMIT)
            self.assertEqual(inputs.source_run, base / "source")
            self.assertEqual(inputs.reference_run, base / "reference")
            self.assertEqual(inputs.formal_run, base / "formal")
            self.assertEqual(inputs.checkpoint_dir, base / "checkpoint")
            self.assertEqual(inputs.expected_source_completion_sha256, "1" * 64)
            self.assertEqual(inputs.expected_reference_completion_sha256, "2" * 64)
            self.assertEqual(inputs.expected_formal_completion_sha256, "3" * 64)
            self.assertEqual(inputs.expected_checkpoint_sha256, "4" * 64)
            self.assertEqual(str(inputs.device), "cpu")

    def test_smoke_evaluates_only_frozen_scene_and_requires_structural_gates(self) -> None:
        module = self.api()
        with tempfile.TemporaryDirectory() as directory:
            fixture = PreparedRun(Path(directory))
            evaluated: list[str] = []

            def fail_gate(sample: PublishedTranslationSample):
                evaluated.append(sample.scene)
                return _metric(sample.scene, covered=1e-5)

            with self.patch_dependencies(module, fixture, evaluator=fail_gate):
                with self.assertRaisesRegex(ValueError, "smoke structural gates"):
                    module.run_smoke(fixture.inputs)
            self.assertEqual(evaluated, [SMOKE_SCENE])
            self.assertFalse((fixture.root / "smoke" / "completed.json").exists())

            evaluated.clear()
            with self.patch_dependencies(
                module,
                fixture,
                evaluator=lambda sample: (
                    evaluated.append(sample.scene) or _metric(sample.scene)
                ),
            ):
                marker = module.run_smoke(fixture.inputs)
            self.assertEqual(evaluated, [SMOKE_SCENE])
            payload = json.loads(marker.read_text(encoding="utf-8"))
            self.assertEqual(
                payload["metadata"],
                {
                    "scene": SMOKE_SCENE,
                    "endpoint_count": 4,
                    "classification": READY,
                },
            )
            self.assertEqual(set(payload), module.STAGE_COMPLETION_FIELDS)
            self.assertEqual(payload["previous_marker_sha256"], _sha(fixture.prepare))

    def test_completed_smoke_replays_live_gate_and_rejects_marker_divergence(self) -> None:
        module = self.api()
        with tempfile.TemporaryDirectory() as directory:
            fixture = PreparedRun(Path(directory))
            with self.patch_dependencies(module, fixture):
                marker = module.run_smoke(fixture.inputs)
            marker_bytes = marker.read_bytes()
            evaluated: list[str] = []

            def fail_live_gate(sample: PublishedTranslationSample):
                evaluated.append(sample.scene)
                return _metric(sample.scene, covered=1e-5)

            with self.patch_dependencies(module, fixture, evaluator=fail_live_gate):
                with self.assertRaisesRegex(ValueError, "smoke.*structural gates"):
                    module.run_smoke(fixture.inputs)
            self.assertEqual(evaluated, [SMOKE_SCENE])
            self.assertEqual(marker.read_bytes(), marker_bytes)

    def test_exact_cohort_is_required_before_any_smoke_evaluation(self) -> None:
        module = self.api()
        with tempfile.TemporaryDirectory() as directory:
            fixture = PreparedRun(Path(directory))
            with (
                mock.patch.object(
                    module,
                    "load_published_cohort",
                    return_value=tuple(fixture.samples[:-1]),
                ),
                mock.patch.object(module, "evaluate_translation_sample") as evaluate,
            ):
                with self.assertRaisesRegex(ValueError, "exact ten-scene"):
                    module.run_smoke(fixture.inputs)
            evaluate.assert_not_called()

    def test_calibration_evaluates_exact_ten_by_four_and_audits_real_tree(self) -> None:
        module = self.api()
        with tempfile.TemporaryDirectory() as directory:
            fixture = PreparedRun(Path(directory))
            with self.patch_dependencies(module, fixture):
                module.run_smoke(fixture.inputs)

            evaluated: list[str] = []
            classifier_inputs: list[tuple[int, bool]] = []

            def classify(rows, *, cohort, physical_leakage_clean):
                classifier_inputs.append((len(rows), physical_leakage_clean))
                return _classification()

            arrays = {name: np.asarray(0) for name in LONG_CONTEXT_MEMBERS}
            with (
                mock.patch.object(
                    module, "load_published_cohort", side_effect=fixture.loader
                ),
                mock.patch.object(
                    module,
                    "evaluate_translation_sample",
                    side_effect=lambda sample: (
                        evaluated.append(sample.scene) or _metric(sample.scene)
                    ),
                ),
                mock.patch.object(module, "load_long_context", return_value=arrays),
                mock.patch.object(
                    module, "classify_stage_a_prime", side_effect=classify
                ),
            ):
                marker = module.run_calibration(fixture.inputs)
            self.assertEqual(evaluated, list(SCENES))
            self.assertEqual(classifier_inputs, [(10, True)])
            self.assertEqual(
                json.loads(marker.read_text(encoding="utf-8"))["metadata"],
                {
                    "scene_count": 10,
                    "endpoint_count": 40,
                    "classification": READY,
                },
            )

    def test_completed_calibration_replays_live_cohort_audit_and_classifier(self) -> None:
        module = self.api()
        with tempfile.TemporaryDirectory() as directory:
            fixture = PreparedRun(Path(directory))
            with self.patch_dependencies(module, fixture):
                module.run_smoke(fixture.inputs)
                marker = module.run_calibration(fixture.inputs)
            marker_bytes = marker.read_bytes()
            evaluated: list[str] = []
            classifier_inputs: list[tuple[list[str], list[list[int]], bool]] = []

            def classify(rows, *, cohort, physical_leakage_clean):
                classifier_inputs.append(
                    (
                        [str(row["scene"]) for row in rows],
                        [list(row["endpoint_ids"]) for row in rows],
                        physical_leakage_clean,
                    )
                )
                return _classification(FAILED)

            with (
                self.patch_dependencies(
                    module,
                    fixture,
                    classification=FAILED,
                    audit=False,
                    evaluator=lambda sample: (
                        evaluated.append(sample.scene) or _metric(sample.scene)
                    ),
                ),
                mock.patch.object(
                    module, "_audit_prediction_only_tree", return_value=False
                ) as audit,
                mock.patch.object(
                    module, "classify_stage_a_prime", side_effect=classify
                ),
            ):
                with self.assertRaisesRegex(
                    ValueError, "completed calibration.*live"
                ):
                    module.run_calibration(fixture.inputs)
            self.assertEqual(evaluated, list(SCENES))
            self.assertEqual(
                classifier_inputs,
                [(list(SCENES), [[0, 1, 2, 3]] * 10, False)],
            )
            audit.assert_called_once()
            self.assertEqual(marker.read_bytes(), marker_bytes)

    def test_failed_calibration_is_reported_and_run_all_never_calls_verifier(self) -> None:
        module = self.api()
        with tempfile.TemporaryDirectory() as directory:
            fixture = PreparedRun(Path(directory))
            with self.patch_dependencies(module, fixture, classification=FAILED):
                module.run_smoke(fixture.inputs)
                with mock.patch.object(module, "verify_completed_run") as verify:
                    with self.assertRaises(module.StageAPrimeFailed):
                        module.run_all(fixture.inputs)
            verify.assert_not_called()
            calibration = json.loads(
                (fixture.root / "calibration" / "completed.json").read_text(
                    encoding="utf-8"
                )
            )
            report = json.loads(
                (fixture.root / "reports" / "stage_a_prime.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(calibration["metadata"]["classification"], FAILED)
            self.assertEqual(report["classification"], FAILED)
            self.assertFalse((fixture.root / "verified_completion.json").exists())

    def test_report_semantic_failure_rolls_back_only_completion_last(self) -> None:
        module = self.api()
        with tempfile.TemporaryDirectory() as directory:
            fixture = PreparedRun(Path(directory))
            with self.patch_dependencies(module, fixture):
                module.run_smoke(fixture.inputs)
                module.run_calibration(fixture.inputs)

            def malformed_writer(run_root: Path, payload: dict[str, object]):
                paths = fixture.write_report(run_root, payload)
                paths[0].write_text("{}\n", encoding="utf-8")
                return paths

            with (
                self.patch_dependencies(module, fixture),
                mock.patch.object(
                    module,
                    "write_stage_a_prime_report",
                    side_effect=malformed_writer,
                ),
            ):
                with self.assertRaises(ValueError):
                    module.run_report(fixture.inputs)
            self.assertTrue((fixture.root / "reports" / "stage_a_prime.json").is_file())
            self.assertTrue((fixture.root / "reports" / "stage_a_prime.md").is_file())
            self.assertFalse((fixture.root / "reports" / "completed.json").exists())

    def test_report_writer_exception_after_valid_triplet_rolls_back_completion(self) -> None:
        module = self.api()
        with tempfile.TemporaryDirectory() as directory:
            fixture = PreparedRun(Path(directory))
            with self.patch_dependencies(module, fixture):
                module.run_smoke(fixture.inputs)
                module.run_calibration(fixture.inputs)

            def writer_raises_after_publication(
                run_root: Path, payload: dict[str, object]
            ):
                fixture.write_report(run_root, payload)
                raise RuntimeError("writer failed after publishing completion")

            with (
                self.patch_dependencies(module, fixture),
                mock.patch.object(
                    module,
                    "write_stage_a_prime_report",
                    side_effect=writer_raises_after_publication,
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "after publishing completion"):
                    module.run_report(fixture.inputs)
            self.assertTrue((fixture.root / "reports" / "stage_a_prime.json").is_file())
            self.assertTrue((fixture.root / "reports" / "stage_a_prime.md").is_file())
            self.assertFalse((fixture.root / "reports" / "completed.json").exists())

    def test_report_noncanonical_writer_paths_roll_back_completion(self) -> None:
        module = self.api()
        with tempfile.TemporaryDirectory() as directory:
            fixture = PreparedRun(Path(directory))
            with self.patch_dependencies(module, fixture):
                module.run_smoke(fixture.inputs)
                module.run_calibration(fixture.inputs)

            def writer_returns_noncanonical_paths(
                run_root: Path, payload: dict[str, object]
            ):
                return list(fixture.write_report(run_root, payload))

            with (
                self.patch_dependencies(module, fixture),
                mock.patch.object(
                    module,
                    "write_stage_a_prime_report",
                    side_effect=writer_returns_noncanonical_paths,
                ),
            ):
                with self.assertRaisesRegex(ValueError, "noncanonical publication paths"):
                    module.run_report(fixture.inputs)
            self.assertTrue((fixture.root / "reports" / "stage_a_prime.json").is_file())
            self.assertTrue((fixture.root / "reports" / "stage_a_prime.md").is_file())
            self.assertFalse((fixture.root / "reports" / "completed.json").exists())

    def test_report_rollback_preserves_later_same_content_completion(self) -> None:
        module = self.api()
        with tempfile.TemporaryDirectory() as directory:
            fixture = PreparedRun(Path(directory))
            with self.patch_dependencies(module, fixture):
                module.run_smoke(fixture.inputs)
                module.run_calibration(fixture.inputs)

            completion = fixture.root / "reports" / "completed.json"
            original_capture = module._snapshot_expected_report_completion
            replacement_bytes: list[bytes] = []

            def writer_returns_noncanonical_paths(
                run_root: Path, payload: dict[str, object]
            ):
                return list(fixture.write_report(run_root, payload))

            def replace_after_capture(*args, **kwargs):
                captured = original_capture(*args, **kwargs)
                self.assertIsNotNone(captured)
                completion_snapshot, expected_bytes = captured
                replacement_bytes.append(expected_bytes)
                replacement = completion.with_name("replacement.tmp")
                replacement.write_bytes(expected_bytes)
                os.replace(replacement, completion)
                self.assertNotEqual(
                    module._file_identity(completion.stat(follow_symlinks=False)),
                    completion_snapshot.identity,
                )
                return captured

            with (
                self.patch_dependencies(module, fixture),
                mock.patch.object(
                    module,
                    "write_stage_a_prime_report",
                    side_effect=writer_returns_noncanonical_paths,
                ),
                mock.patch.object(
                    module,
                    "_snapshot_expected_report_completion",
                    side_effect=replace_after_capture,
                ),
            ):
                with self.assertRaisesRegex(ValueError, "rollback failed"):
                    module.run_report(fixture.inputs)
            self.assertTrue(completion.is_file())
            self.assertEqual(completion.read_bytes(), replacement_bytes[0])

    def test_fresh_report_change_after_semantic_validation_rolls_back_completion(self) -> None:
        module = self.api()
        for relative in ("reports/stage_a_prime.json", "reports/stage_a_prime.md"):
            with self.subTest(relative=relative), tempfile.TemporaryDirectory() as directory:
                fixture = PreparedRun(Path(directory))
                with self.patch_dependencies(module, fixture):
                    module.run_smoke(fixture.inputs)
                    module.run_calibration(fixture.inputs)

                original_validate_report = module._validate_report
                changed = False

                def change_after_semantic_validation(*args, **kwargs):
                    nonlocal changed
                    result = original_validate_report(*args, **kwargs)
                    if not changed:
                        changed = True
                        (fixture.root / relative).write_text(
                            "changed after semantic validation\n", encoding="utf-8"
                        )
                    return result

                with (
                    self.patch_dependencies(module, fixture),
                    mock.patch.object(
                        module,
                        "_validate_report",
                        side_effect=change_after_semantic_validation,
                    ),
                ):
                    with self.assertRaises(ValueError):
                        module.run_report(fixture.inputs)
                self.assertTrue((fixture.root / "reports" / "stage_a_prime.json").is_file())
                self.assertTrue((fixture.root / "reports" / "stage_a_prime.md").is_file())
                self.assertFalse((fixture.root / "reports" / "completed.json").exists())

    def test_prediction_only_npz_schema_failure_reaches_classifier_as_false(self) -> None:
        module = self.api()
        with tempfile.TemporaryDirectory() as directory:
            fixture = PreparedRun(Path(directory))
            with self.patch_dependencies(module, fixture):
                module.run_smoke(fixture.inputs)

            observed: list[bool] = []

            def classify(rows, *, cohort, physical_leakage_clean):
                observed.append(physical_leakage_clean)
                return _classification(FAILED)

            incomplete_arrays = {
                name: np.asarray(0) for name in sorted(LONG_CONTEXT_MEMBERS)[1:]
            }
            with (
                mock.patch.object(
                    module, "load_published_cohort", side_effect=fixture.loader
                ),
                mock.patch.object(
                    module,
                    "evaluate_translation_sample",
                    side_effect=lambda sample: _metric(sample.scene),
                ),
                mock.patch.object(
                    module, "load_long_context", return_value=incomplete_arrays
                ),
                mock.patch.object(
                    module, "classify_stage_a_prime", side_effect=classify
                ),
            ):
                marker = module.run_calibration(fixture.inputs)
            self.assertEqual(observed, [False])
            self.assertEqual(
                json.loads(marker.read_text(encoding="utf-8"))["metadata"][
                    "classification"
                ],
                FAILED,
            )

    def test_fully_resigned_smoke_marker_metadata_mismatch_blocks_calibration(self) -> None:
        module = self.api()
        with tempfile.TemporaryDirectory() as directory:
            fixture = PreparedRun(Path(directory))
            with self.patch_dependencies(module, fixture):
                marker = module.run_smoke(fixture.inputs)
            payload = json.loads(marker.read_text(encoding="utf-8"))
            payload["metadata"]["scene"] = "scene0000_00"
            unsigned = dict(payload)
            unsigned.pop("completion_digest")
            payload["completion_digest"] = _digest(unsigned)
            marker.write_bytes(_json_bytes(payload))

            with self.patch_dependencies(module, fixture):
                with self.assertRaisesRegex(ValueError, "smoke.*metadata"):
                    module.run_calibration(fixture.inputs)
            self.assertFalse((fixture.root / "calibration" / "completed.json").exists())

    def test_swap_restore_dependency_during_smoke_is_detected(self) -> None:
        module = self.api()
        with tempfile.TemporaryDirectory() as directory:
            fixture = PreparedRun(Path(directory))
            config = fixture.root / "config.json"

            def mutate(sample: PublishedTranslationSample):
                original = config.read_bytes()
                config.write_bytes(b"swapped config\n")
                config.write_bytes(original)
                return _metric(sample.scene)

            with self.patch_dependencies(module, fixture, evaluator=mutate):
                with self.assertRaisesRegex(ValueError, "changed during smoke"):
                    module.run_smoke(fixture.inputs)
            self.assertFalse((fixture.root / "smoke" / "completed.json").exists())

    def test_snapshot_bytes_rejects_a_different_middle_read(self) -> None:
        module = self.api()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "authenticated.json"
            original_bytes = _json_bytes({"value": "A"})
            alternate_bytes = _json_bytes({"value": "B"})
            path.write_bytes(original_bytes)
            snapshot = module._snapshot_file(path, label="authenticated fixture")
            real_read_bytes = type(path).read_bytes

            def transient_read(candidate: Path) -> bytes:
                if Path(candidate) == path:
                    return alternate_bytes
                return real_read_bytes(candidate)

            with mock.patch.object(
                type(path),
                "read_bytes",
                autospec=True,
                side_effect=transient_read,
            ):
                with self.assertRaisesRegex(ValueError, "changed during authentication"):
                    module._read_snapshot_bytes(snapshot)
            self.assertEqual(path.read_bytes(), original_bytes)

    def test_new_stages_snapshot_before_semantic_authentication(self) -> None:
        module = self.api()
        cases = (
            ("smoke", module.run_smoke, "smoke/completed.json"),
            ("calibration", module.run_calibration, "calibration/completed.json"),
            ("report", module.run_report, "reports/completed.json"),
        )
        for stage_name, runner, completion_relative in cases:
            with self.subTest(stage=stage_name), tempfile.TemporaryDirectory() as directory:
                fixture = PreparedRun(Path(directory))
                with self.patch_dependencies(module, fixture):
                    if stage_name in {"calibration", "report"}:
                        module.run_smoke(fixture.inputs)
                    if stage_name == "report":
                        module.run_calibration(fixture.inputs)

                original_authenticate = module._authenticate_prefix
                config = fixture.root / "config.json"

                def mutate_after_authentication(*args, **kwargs):
                    result = original_authenticate(*args, **kwargs)
                    config.write_text(
                        f"changed after {stage_name} authentication\n",
                        encoding="utf-8",
                    )
                    return result

                with (
                    self.patch_dependencies(module, fixture),
                    mock.patch.object(
                        module,
                        "_authenticate_prefix",
                        side_effect=mutate_after_authentication,
                    ),
                ):
                    with self.assertRaisesRegex(
                        ValueError, f"changed during {stage_name}"
                    ):
                        runner(fixture.inputs)
                self.assertFalse((fixture.root / completion_relative).exists())

    def test_completed_stages_terminally_revalidate_after_last_validation(self) -> None:
        module = self.api()
        for stage_name in ("smoke", "calibration", "report"):
            with self.subTest(stage=stage_name), tempfile.TemporaryDirectory() as directory:
                fixture = PreparedRun(Path(directory))
                with self.patch_dependencies(module, fixture):
                    module.run_smoke(fixture.inputs)
                    if stage_name in {"calibration", "report"}:
                        module.run_calibration(fixture.inputs)
                    if stage_name == "report":
                        module.run_report(fixture.inputs)

                if stage_name in {"smoke", "calibration"}:
                    original_snapshot_auth = module._snapshot_authenticated_state
                    long_path = fixture.samples[0].long_path

                    def mutate_after_snapshot_auth(*args, **kwargs):
                        result = original_snapshot_auth(*args, **kwargs)
                        long_path.write_text(
                            f"changed after completed {stage_name} validation\n",
                            encoding="utf-8",
                        )
                        return result

                    mutation = mock.patch.object(
                        module,
                        "_snapshot_authenticated_state",
                        side_effect=mutate_after_snapshot_auth,
                    )
                else:
                    original_validate_report = module._validate_report
                    report_json = fixture.root / "reports" / "stage_a_prime.json"

                    def mutate_after_report_read(*args, **kwargs):
                        result = original_validate_report(*args, **kwargs)
                        report_json.write_text(
                            "{}\n",
                            encoding="utf-8",
                        )
                        return result

                    mutation = mock.patch.object(
                        module,
                        "_validate_report",
                        side_effect=mutate_after_report_read,
                    )

                runner = {
                    "smoke": module.run_smoke,
                    "calibration": module.run_calibration,
                    "report": module.run_report,
                }[stage_name]
                with self.patch_dependencies(module, fixture), mutation:
                    with self.assertRaisesRegex(
                        ValueError, f"changed during {stage_name}"
                    ):
                        runner(fixture.inputs)

    def test_completed_report_recomputes_ready_and_failed_live_evidence(self) -> None:
        module = self.api()
        for classification, audit in ((READY, True), (FAILED, False)):
            with (
                self.subTest(classification=classification),
                tempfile.TemporaryDirectory() as directory,
            ):
                fixture = PreparedRun(Path(directory))
                with self.patch_dependencies(
                    module,
                    fixture,
                    classification=classification,
                    audit=audit,
                ):
                    module.run_smoke(fixture.inputs)
                    module.run_calibration(fixture.inputs)
                    paths = module.run_report(fixture.inputs)
                original_bytes = tuple(path.read_bytes() for path in paths)

                evaluated: list[str] = []
                with self.patch_dependencies(
                    module,
                    fixture,
                    classification=classification,
                    audit=audit,
                    evaluator=lambda sample: (
                        evaluated.append(sample.scene) or _metric(sample.scene)
                    ),
                ):
                    resumed = module.run_report(fixture.inputs)
                self.assertEqual(resumed, paths)
                self.assertEqual(evaluated, list(SCENES))
                self.assertEqual(
                    tuple(path.read_bytes() for path in resumed), original_bytes
                )

    def test_fully_resigned_ready_report_with_failed_leakage_is_rejected(self) -> None:
        module = self.api()
        with tempfile.TemporaryDirectory() as directory:
            fixture = PreparedRun(Path(directory))
            with self.patch_dependencies(module, fixture):
                module.run_smoke(fixture.inputs)
                module.run_calibration(fixture.inputs)
                module.run_report(fixture.inputs)

            report_path = fixture.root / "reports" / "stage_a_prime.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["physical_leakage_clean"] = False
            report["gates"]["physical_leakage_clean"] = False
            report["failed_gates"] = ["physical_leakage_clean"]
            self.assertEqual(report["classification"], READY)
            report_path.write_bytes(_json_bytes(report))

            completion_path = fixture.root / "reports" / "completed.json"
            completion = json.loads(completion_path.read_text(encoding="utf-8"))
            completion["report_json_sha256"] = _sha(report_path)
            unsigned = dict(completion)
            unsigned.pop("completion_digest")
            completion["completion_digest"] = _digest(unsigned)
            completion_path.write_bytes(_json_bytes(completion))

            with self.patch_dependencies(module, fixture):
                with self.assertRaisesRegex(
                    ValueError, "deterministic|classification|reclassification"
                ):
                    module.run_report(fixture.inputs)

    def test_fully_resigned_finite_metric_rewrite_fails_live_report_replay(self) -> None:
        module = self.api()
        with tempfile.TemporaryDirectory() as directory:
            fixture = PreparedRun(Path(directory))
            with self.patch_dependencies(module, fixture):
                module.run_smoke(fixture.inputs)
                module.run_calibration(fixture.inputs)
                module.run_report(fixture.inputs)

            report_path = fixture.root / "reports" / "stage_a_prime.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["scene_metrics"][0]["mean_full_scene_utility"] = 0.2
            report_path.write_bytes(_json_bytes(report))

            completion_path = fixture.root / "reports" / "completed.json"
            completion = json.loads(completion_path.read_text(encoding="utf-8"))
            completion["report_json_sha256"] = _sha(report_path)
            unsigned = dict(completion)
            unsigned.pop("completion_digest")
            completion["completion_digest"] = _digest(unsigned)
            completion_path.write_bytes(_json_bytes(completion))

            evaluated: list[str] = []
            with self.patch_dependencies(
                module,
                fixture,
                evaluator=lambda sample: (
                    evaluated.append(sample.scene) or _metric(sample.scene)
                ),
            ):
                with self.assertRaisesRegex(ValueError, "deterministic|live"):
                    module.run_report(fixture.inputs)
            self.assertEqual(evaluated, list(SCENES))

    def test_lexical_ancestor_escape_is_rejected_before_authentication(self) -> None:
        module = self.api()
        with tempfile.TemporaryDirectory() as directory:
            fixture = PreparedRun(Path(directory))
            escaped = SimpleNamespace(
                **vars(fixture.inputs),
            )
            escaped.run_root = fixture.root.parent / "unused" / ".." / fixture.root.name
            with mock.patch.object(module, "load_published_cohort") as load:
                with self.assertRaisesRegex(ValueError, "parent traversal"):
                    module.run_smoke(escaped)
            load.assert_not_called()

    def test_late_foreign_file_rolls_back_only_new_marker_and_is_preserved(self) -> None:
        module = self.api()
        with tempfile.TemporaryDirectory() as directory:
            fixture = PreparedRun(Path(directory))
            original = module._atomic_create_exact
            late = fixture.root / "late-foreign.txt"

            def publish(path: Path, content: bytes) -> bool:
                created = original(path, content)
                if Path(path) == fixture.root / "smoke" / "completed.json":
                    late.write_text("preserve me\n", encoding="utf-8")
                return created

            with (
                self.patch_dependencies(module, fixture),
                mock.patch.object(module, "_atomic_create_exact", side_effect=publish),
            ):
                with self.assertRaisesRegex(ValueError, "noncanonical.*inventory"):
                    module.run_smoke(fixture.inputs)
            self.assertTrue(late.is_file())
            self.assertFalse((fixture.root / "smoke" / "completed.json").exists())

    def test_stage_rollback_preserves_later_same_content_marker(self) -> None:
        module = self.api()
        with tempfile.TemporaryDirectory() as directory:
            fixture = PreparedRun(Path(directory))
            marker = fixture.root / "smoke" / "completed.json"
            late = fixture.root / "late-foreign.txt"
            original_require_inventory = module._require_inventory
            replaced = False
            replacement_identity: list[tuple[int, int, int, int, int, int]] = []
            replacement_bytes: list[bytes] = []

            def replace_before_completion_inventory(
                root: Path, files: frozenset[str], *, label: str
            ) -> None:
                nonlocal replaced
                if (
                    not replaced
                    and label == "smoke completion"
                    and "smoke/completed.json" in files
                    and marker.is_file()
                ):
                    replaced = True
                    content = marker.read_bytes()
                    original_identity = module._file_identity(
                        marker.stat(follow_symlinks=False)
                    )
                    replacement = marker.with_name("replacement.tmp")
                    replacement.write_bytes(content)
                    os.replace(replacement, marker)
                    current_identity = module._file_identity(
                        marker.stat(follow_symlinks=False)
                    )
                    self.assertNotEqual(current_identity, original_identity)
                    replacement_identity.append(current_identity)
                    replacement_bytes.append(content)
                    late.write_text("force terminal inventory failure\n", encoding="utf-8")
                original_require_inventory(root, files, label=label)

            with (
                self.patch_dependencies(module, fixture),
                mock.patch.object(
                    module,
                    "_require_inventory",
                    side_effect=replace_before_completion_inventory,
                ),
            ):
                with self.assertRaisesRegex(ValueError, "rollback failed"):
                    module.run_smoke(fixture.inputs)
            self.assertTrue(replaced)
            self.assertTrue(marker.is_file())
            self.assertEqual(marker.read_bytes(), replacement_bytes[0])
            self.assertEqual(
                module._file_identity(marker.stat(follow_symlinks=False)),
                replacement_identity[0],
            )

    def test_temporary_cleanup_failure_rolls_back_linked_marker(self) -> None:
        module = self.api()
        with tempfile.TemporaryDirectory() as directory:
            fixture = PreparedRun(Path(directory))
            marker = fixture.root / "smoke" / "completed.json"
            real_unlink = type(marker).unlink
            failed_temporary_cleanup = False

            def fail_first_publication_temporary_unlink(
                candidate: Path, *args: object, **kwargs: object
            ) -> None:
                nonlocal failed_temporary_cleanup
                path = Path(candidate)
                if (
                    not failed_temporary_cleanup
                    and path.parent == marker.parent
                    and path.name.startswith(".completed.json.")
                    and path.name.endswith(".tmp")
                ):
                    failed_temporary_cleanup = True
                    raise PermissionError("simulated temporary cleanup failure")
                real_unlink(candidate, *args, **kwargs)

            with (
                self.patch_dependencies(module, fixture),
                mock.patch.object(
                    type(marker),
                    "unlink",
                    autospec=True,
                    side_effect=fail_first_publication_temporary_unlink,
                ),
            ):
                with self.assertRaisesRegex(ValueError, "temporary cleanup failed"):
                    module.run_smoke(fixture.inputs)
            self.assertTrue(failed_temporary_cleanup)
            self.assertFalse(marker.exists())

    def test_exact_publication_race_and_live_resume_are_idempotent(self) -> None:
        module = self.api()
        with tempfile.TemporaryDirectory() as directory:
            fixture = PreparedRun(Path(directory))
            real_link = os.link
            raced = False

            def race_link(source, target, *args, **kwargs):
                nonlocal raced
                target_path = Path(target)
                if target_path == fixture.root / "smoke" / "completed.json" and not raced:
                    raced = True
                    target_path.write_bytes(Path(source).read_bytes())
                    raise FileExistsError("simulated exact publication race")
                return real_link(source, target, *args, **kwargs)

            evaluated: list[str] = []
            with (
                self.patch_dependencies(
                    module,
                    fixture,
                    evaluator=lambda sample: (
                        evaluated.append(sample.scene) or _metric(sample.scene)
                    ),
                ),
                mock.patch.object(module.os, "link", side_effect=race_link),
            ):
                first = module.run_smoke(fixture.inputs)
            first_bytes = first.read_bytes()

            with self.patch_dependencies(
                module,
                fixture,
                evaluator=lambda sample: (
                    evaluated.append(sample.scene) or _metric(sample.scene)
                ),
            ):
                second = module.run_smoke(fixture.inputs)
            self.assertEqual(first, second)
            self.assertEqual(second.read_bytes(), first_bytes)
            self.assertEqual(evaluated, [SMOKE_SCENE, SMOKE_SCENE])

    def test_rollback_mismatch_is_reported_as_secondary_failure(self) -> None:
        module = self.api()
        with tempfile.TemporaryDirectory() as directory:
            fixture = PreparedRun(Path(directory))
            config = fixture.root / "config.json"
            marker = fixture.root / "smoke" / "completed.json"
            original = module._atomic_create_exact

            def corrupt_after_publish(path: Path, content: bytes) -> bool:
                created = original(path, content)
                if Path(path) == marker:
                    config.write_text("changed after publication\n", encoding="utf-8")
                    marker.write_text("changed marker\n", encoding="utf-8")
                return created

            with (
                self.patch_dependencies(module, fixture),
                mock.patch.object(
                    module, "_atomic_create_exact", side_effect=corrupt_after_publish
                ),
            ):
                with self.assertRaisesRegex(ValueError, "rollback failed"):
                    module.run_smoke(fixture.inputs)
            self.assertTrue(marker.is_file())
            self.assertEqual(marker.read_text(encoding="utf-8"), "changed marker\n")


if __name__ == "__main__":
    unittest.main()
