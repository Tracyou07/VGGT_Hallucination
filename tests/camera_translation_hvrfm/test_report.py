from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from tests.camera_translation_hvrfm.test_evaluate import (
    _gate_cohort,
    _passing_gate_scenes,
)

try:
    from pre_experiments.camera_translation_hvrfm import report
except (ImportError, ModuleNotFoundError):
    report = None  # type: ignore[assignment]


RUN_ID = "stage-a-prime-test"
GIT_COMMIT = "8" * 40
REPORT_KEYS = {
    "schema",
    "run_id",
    "git_commit",
    "classification",
    "failed_gates",
    "gates",
    "scene_count",
    "endpoint_count",
    "mean_teacher_retention",
    "mean_full_scene_utility",
    "minimum_full_scene_utility",
    "positive_scene_count",
    "physical_leakage_clean",
    "scene_metrics",
    "cohort",
}
COHORT_KEYS = {
    "scene",
    "sample_id",
    "role",
    "long_sha256",
    "short_sha256",
    "quality_sha256",
    "target_sha256",
}
COMPLETION_KEYS = {
    "schema",
    "run_id",
    "git_commit",
    "classification",
    "scene_count",
    "endpoint_count",
    "report_json_path",
    "report_json_sha256",
    "report_markdown_path",
    "report_markdown_sha256",
    "completion_digest",
}


class StageAPrimeReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def api(self):
        self.assertIsNotNone(report, "Task 3b report module is missing")
        return report

    def build(
        self,
        module,
        *,
        scene_metrics=None,
        cohort=None,
        run_id: str = RUN_ID,
        git_commit: str = GIT_COMMIT,
        physical_leakage_clean: bool = True,
    ):
        return module.build_stage_a_prime_report(
            _passing_gate_scenes() if scene_metrics is None else scene_metrics,
            cohort=_gate_cohort() if cohort is None else cohort,
            run_id=run_id,
            git_commit=git_commit,
            physical_leakage_clean=physical_leakage_clean,
        )

    def test_build_calls_classifier_and_is_shuffle_deterministic_exact_schema(self) -> None:
        module = self.api()
        rows = list(reversed(_passing_gate_scenes()))
        cohort = list(reversed(_gate_cohort()))
        with mock.patch.object(
            module,
            "classify_stage_a_prime",
            wraps=module.classify_stage_a_prime,
        ) as classify:
            shuffled = self.build(module, scene_metrics=rows, cohort=cohort)

        self.assertEqual(classify.call_count, 1)
        called_rows = classify.call_args.args[0]
        called_keywords = classify.call_args.kwargs
        self.assertEqual(
            [row["scene"] for row in called_rows],
            sorted(row["scene"] for row in rows),
        )
        self.assertEqual(
            [sample.scene for sample in called_keywords["cohort"]],
            sorted(sample.scene for sample in cohort),
        )
        self.assertIs(called_keywords["physical_leakage_clean"], True)

        canonical = self.build(module)
        self.assertEqual(shuffled, canonical)
        self.assertEqual(set(canonical), REPORT_KEYS)
        self.assertEqual(
            canonical["schema"],
            "camera_translation_hvrfm.stage_a_prime_report.v1",
        )
        self.assertEqual(canonical["run_id"], RUN_ID)
        self.assertEqual(canonical["git_commit"], GIT_COMMIT)
        self.assertEqual(canonical["scene_count"], 10)
        self.assertEqual(canonical["endpoint_count"], 40)
        self.assertIs(canonical["physical_leakage_clean"], True)
        self.assertEqual(
            [row["scene"] for row in canonical["scene_metrics"]],
            sorted(row["scene"] for row in _passing_gate_scenes()),
        )
        self.assertEqual(
            [row["scene"] for row in canonical["cohort"]],
            sorted(sample.scene for sample in _gate_cohort()),
        )
        self.assertTrue(
            all(set(row) == COHORT_KEYS for row in canonical["cohort"])
        )
        expected_by_scene = {
            row["scene"]: row for row in _passing_gate_scenes()
        }
        for row in canonical["scene_metrics"]:
            self.assertEqual(row, expected_by_scene[row["scene"]])

    def test_build_rejects_unknown_malformed_nonfinite_and_bool_numeric(self) -> None:
        module = self.api()
        cases = []

        unknown = _passing_gate_scenes()
        unknown[0]["unknown"] = 1
        cases.append(("unknown", unknown))

        missing = _passing_gate_scenes()
        missing[0]["endpoints"][0].pop("endpoint_rms")
        cases.append(("missing", missing))

        for name, value in (
            ("nan", float("nan")),
            ("positive_inf", float("inf")),
            ("negative_inf", -float("inf")),
            ("bool_numeric", True),
        ):
            rows = _passing_gate_scenes()
            rows[0]["endpoints"][0]["endpoint_rms"] = value
            cases.append((name, rows))

        for name, rows in cases:
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.build(module, scene_metrics=rows)

        payload = self.build(module)
        payload["scene_count"] = True
        with self.assertRaises(ValueError):
            module.write_stage_a_prime_report(self.root, payload)

    def test_build_rejects_cohort_identity_digest_and_commit_tampering(self) -> None:
        module = self.api()
        cohort = _gate_cohort()
        wrong_role = [
            replace(cohort[0], role="validation"),
            *cohort[1:],
        ]
        wrong_digest = [
            replace(cohort[0], quality_sha256="9" * 64),
            *cohort[1:],
        ]
        for name, malformed in (
            ("role", wrong_role),
            ("digest", wrong_digest),
        ):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.build(module, cohort=malformed)

        metrics_digest = _passing_gate_scenes()
        metrics_digest[0]["provenance"]["short_sha256"] = "9" * 64
        with self.assertRaises(ValueError):
            self.build(module, scene_metrics=metrics_digest)

        with self.assertRaises(ValueError):
            self.build(module, git_commit="9" * 40)
        with self.assertRaises(ValueError):
            self.build(module, run_id="../escape")

    def test_write_hashes_reports_and_self_digests_completion(self) -> None:
        module = self.api()
        payload = self.build(module)
        json_path, markdown_path, completion_path = (
            module.write_stage_a_prime_report(self.root, payload)
        )

        self.assertEqual(json_path, self.root / "reports" / "stage_a_prime.json")
        self.assertEqual(markdown_path, self.root / "reports" / "stage_a_prime.md")
        self.assertEqual(completion_path, self.root / "reports" / "completed.json")
        json_bytes = json_path.read_bytes()
        markdown_bytes = markdown_path.read_bytes()
        completion_bytes = completion_path.read_bytes()
        for content in (json_bytes, markdown_bytes, completion_bytes):
            self.assertTrue(content.endswith(b"\n"))
            self.assertNotIn(b"\r\n", content)

        expected_json = (
            json.dumps(
                payload,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        self.assertEqual(json_bytes, expected_json)
        self.assertEqual(json.loads(json_bytes), payload)
        self.assertIn(
            format(float(payload["mean_teacher_retention"]), ".17g"),
            markdown_bytes.decode("utf-8"),
        )

        completion = json.loads(completion_bytes)
        self.assertEqual(set(completion), COMPLETION_KEYS)
        self.assertEqual(
            completion["schema"],
            "camera_translation_hvrfm.stage_a_prime_completion.v1",
        )
        self.assertEqual(completion["report_json_path"], "reports/stage_a_prime.json")
        self.assertEqual(
            completion["report_markdown_path"], "reports/stage_a_prime.md"
        )
        self.assertEqual(
            completion["report_json_sha256"], hashlib.sha256(json_bytes).hexdigest()
        )
        self.assertEqual(
            completion["report_markdown_sha256"],
            hashlib.sha256(markdown_bytes).hexdigest(),
        )
        unsigned = dict(completion)
        completion_digest = unsigned.pop("completion_digest")
        expected_digest = hashlib.sha256(
            json.dumps(
                unsigned,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        self.assertEqual(completion_digest, expected_digest)

    def test_any_existing_conflict_is_preflighted_before_new_writes(self) -> None:
        module = self.api()
        payload = self.build(module)
        names = ("stage_a_prime.json", "stage_a_prime.md", "completed.json")
        for name in names:
            with self.subTest(name=name):
                root = self.root / name.replace(".", "-")
                reports = root / "reports"
                reports.mkdir(parents=True)
                conflict = reports / name
                conflict.write_bytes(b"conflict\n")
                with mock.patch.object(
                    module,
                    "_atomic_publish_bytes",
                    side_effect=AssertionError("publication began before preflight"),
                ) as publish, self.assertRaises(ValueError):
                    module.write_stage_a_prime_report(root, payload)
                publish.assert_not_called()
                self.assertEqual(conflict.read_bytes(), b"conflict\n")
                self.assertEqual(list(reports.iterdir()), [conflict])

    def test_publication_paths_reject_parent_traversal_and_nonregular_target(self) -> None:
        module = self.api()
        payload = self.build(module)
        safe = self.root / "safe"
        safe.mkdir()
        lexical_root = safe / ".." / "escaped"
        with mock.patch.object(
            module,
            "_atomic_publish_bytes",
            side_effect=AssertionError("unsafe path reached publication"),
        ) as publish, self.assertRaisesRegex(ValueError, "parent traversal"):
            module.write_stage_a_prime_report(lexical_root, payload)
        publish.assert_not_called()
        self.assertFalse((self.root / "escaped").exists())

        directory_root = self.root / "directory-target"
        target = directory_root / "reports" / "stage_a_prime.json"
        target.mkdir(parents=True)
        with mock.patch.object(
            module,
            "_atomic_publish_bytes",
            side_effect=AssertionError("nonregular target reached publication"),
        ) as publish, self.assertRaisesRegex(ValueError, "regular file"):
            module.write_stage_a_prime_report(directory_root, payload)
        publish.assert_not_called()
        self.assertTrue(target.is_dir())

    def test_publication_rejects_existing_reports_symlink_before_temp_creation(self) -> None:
        module = self.api()
        payload = self.build(module)
        run_root = self.root / "linked-run"
        run_root.mkdir()
        real_reports = self.root / "real-reports"
        real_reports.mkdir()
        try:
            os.symlink(real_reports, run_root / "reports", target_is_directory=True)
        except OSError as error:
            self.skipTest(f"symlink creation unavailable: {error}")
        with mock.patch.object(
            module.tempfile,
            "NamedTemporaryFile",
            side_effect=AssertionError("symlink path reached temp creation"),
        ) as temporary, self.assertRaisesRegex(ValueError, "symlink"):
            module.write_stage_a_prime_report(run_root, payload)
        temporary.assert_not_called()
        self.assertEqual(list(real_reports.iterdir()), [])

    def test_publication_uses_unique_same_directory_temps_and_completion_last(self) -> None:
        module = self.api()
        payload = self.build(module)
        real_publish = module._atomic_publish_bytes
        with mock.patch.object(
            module,
            "_atomic_publish_bytes",
            wraps=real_publish,
        ) as publish, mock.patch.object(
            module.tempfile,
            "NamedTemporaryFile",
            wraps=tempfile.NamedTemporaryFile,
        ) as temporary:
            module.write_stage_a_prime_report(self.root, payload)

        self.assertEqual(
            [call.args[0].name for call in publish.call_args_list],
            ["stage_a_prime.json", "stage_a_prime.md", "completed.json"],
        )
        self.assertEqual(temporary.call_count, 3)
        prefixes = []
        for call in temporary.call_args_list:
            self.assertEqual(Path(call.kwargs["dir"]), self.root / "reports")
            self.assertIs(call.kwargs["delete"], False)
            self.assertEqual(call.kwargs["suffix"], ".tmp")
            prefixes.append(call.kwargs["prefix"])
        self.assertEqual(len(set(prefixes)), 3)
        self.assertEqual(
            sorted(path.name for path in (self.root / "reports").iterdir()),
            ["completed.json", "stage_a_prime.json", "stage_a_prime.md"],
        )

    def test_report_mutation_after_preflight_blocks_completion_publication(self) -> None:
        module = self.api()
        payload = self.build(module)

        fresh_root = self.root / "fresh-mutation"
        fresh_json = fresh_root / "reports" / "stage_a_prime.json"
        fresh_completion = fresh_root / "reports" / "completed.json"
        real_publish = module._atomic_publish_bytes

        def publish_then_mutate(path: Path, content: bytes) -> None:
            real_publish(path, content)
            if path.name == "stage_a_prime.md":
                fresh_json.write_bytes(b"mutated after fresh publication\n")

        with mock.patch.object(
            module,
            "_atomic_publish_bytes",
            side_effect=publish_then_mutate,
        ), self.assertRaisesRegex(ValueError, "changed|match|report"):
            module.write_stage_a_prime_report(fresh_root, payload)
        self.assertEqual(
            fresh_json.read_bytes(), b"mutated after fresh publication\n"
        )
        self.assertFalse(fresh_completion.exists())

        reused_root = self.root / "reused-mutation"
        reused_json, _, reused_completion = module.write_stage_a_prime_report(
            reused_root, payload
        )
        reused_completion.unlink()
        real_preflight = module._preflight_publication

        def preflight_then_mutate(*args, **kwargs):
            existing = real_preflight(*args, **kwargs)
            reused_json.write_bytes(b"mutated after reuse preflight\n")
            return existing

        with mock.patch.object(
            module,
            "_preflight_publication",
            side_effect=preflight_then_mutate,
        ), self.assertRaisesRegex(ValueError, "changed|match|report"):
            module.write_stage_a_prime_report(reused_root, payload)
        self.assertEqual(
            reused_json.read_bytes(), b"mutated after reuse preflight\n"
        )
        self.assertFalse(reused_completion.exists())

    def test_report_mutation_during_completion_publication_rolls_back_marker(self) -> None:
        module = self.api()
        payload = self.build(module)
        run_root = self.root / "completion-publication-mutation"
        report_json = run_root / "reports" / "stage_a_prime.json"
        completion = run_root / "reports" / "completed.json"
        real_publish = module._atomic_publish_bytes

        def mutate_dependency_then_publish(path: Path, content: bytes) -> None:
            if path.name == "completed.json":
                report_json.write_bytes(b"mutated during completion publication\n")
            real_publish(path, content)

        with mock.patch.object(
            module,
            "_atomic_publish_bytes",
            side_effect=mutate_dependency_then_publish,
        ), self.assertRaisesRegex(ValueError, "changed|match|report"):
            module.write_stage_a_prime_report(run_root, payload)

        self.assertEqual(
            report_json.read_bytes(), b"mutated during completion publication\n"
        )
        self.assertFalse(completion.exists())

    def test_completion_rollback_double_failure_preserves_validation_cause(self) -> None:
        module = self.api()
        payload = self.build(module)
        run_root = self.root / "completion-rollback-double-failure"
        report_json = run_root / "reports" / "stage_a_prime.json"
        real_publish = module._atomic_publish_bytes

        def mutate_dependency_then_publish(path: Path, content: bytes) -> None:
            if path.name == "completed.json":
                report_json.write_bytes(b"mutated during completion publication\n")
            real_publish(path, content)

        with mock.patch.object(
            module,
            "_atomic_publish_bytes",
            side_effect=mutate_dependency_then_publish,
        ), mock.patch.object(
            module,
            "_rollback_exact_new_file",
            side_effect=ValueError("forced completion rollback failure"),
        ), self.assertRaisesRegex(
            ValueError, "dependency validation failed.*rollback failed"
        ) as raised:
            module.write_stage_a_prime_report(run_root, payload)

        self.assertIsInstance(raised.exception.__cause__, ValueError)
        self.assertRegex(
            str(raised.exception.__cause__), "match|report|dependency"
        )
        self.assertNotIsInstance(raised.exception, AttributeError)

    def test_atomic_publish_reports_cleanup_failure_without_masking_primary_error(self) -> None:
        module = self.api()
        destination = self.root / "cleanup" / "artifact.json"
        destination.parent.mkdir()
        real_unlink = Path.unlink

        def reject_temp_unlink(path: Path, *args, **kwargs) -> None:
            if path.name.endswith(".tmp"):
                raise PermissionError("forced temp cleanup failure")
            real_unlink(path, *args, **kwargs)

        with mock.patch.object(
            Path,
            "unlink",
            autospec=True,
            side_effect=reject_temp_unlink,
        ), self.assertRaisesRegex(ValueError, "cleanup|temporary"):
            module._atomic_publish_bytes(destination, b"complete\n")
        self.assertEqual(destination.read_bytes(), b"complete\n")
        leftovers = list(destination.parent.glob(".artifact.json.*.tmp"))
        self.assertEqual(len(leftovers), 1)
        leftovers[0].unlink()

        failed_destination = self.root / "cleanup" / "failed.json"
        link_error = OSError("forced primary link failure")
        with mock.patch.object(
            module.os,
            "link",
            side_effect=link_error,
        ), mock.patch.object(
            Path,
            "unlink",
            autospec=True,
            side_effect=reject_temp_unlink,
        ), self.assertRaisesRegex(ValueError, "publish") as raised:
            module._atomic_publish_bytes(failed_destination, b"complete\n")
        self.assertIs(raised.exception.__cause__, link_error)
        self.assertFalse(failed_destination.exists())
        failed_leftovers = list(destination.parent.glob(".failed.json.*.tmp"))
        self.assertEqual(len(failed_leftovers), 1)
        failed_leftovers[0].unlink()

    def test_repeated_write_reuses_all_three_byte_exact_files(self) -> None:
        module = self.api()
        payload = self.build(module)
        paths = module.write_stage_a_prime_report(self.root, payload)
        original = {path: path.read_bytes() for path in paths}
        with mock.patch.object(
            module,
            "_atomic_publish_bytes",
            side_effect=AssertionError("byte-exact reports must be reused"),
        ) as publish:
            repeated = module.write_stage_a_prime_report(self.root, payload)
        publish.assert_not_called()
        self.assertEqual(repeated, paths)
        self.assertEqual({path: path.read_bytes() for path in paths}, original)


if __name__ == "__main__":
    unittest.main()
