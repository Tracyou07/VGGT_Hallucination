from __future__ import annotations

import inspect
import os
import tempfile
from pathlib import Path
import subprocess
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from pre_experiments.variational_camera_latent import pipeline as pipeline_module
from pre_experiments.variational_camera_latent.pipeline import (
    load_exact_completion,
    parse_args,
    run_stage,
    write_completion,
)


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts" / "h20" / "run_variational_camera_latent.sh"
NULL20_RUNNER = ROOT / "scripts" / "h20" / "run_matched_random_null20.sh"


class PipelineRunnerTests(unittest.TestCase):
    def test_runner_uses_h20_and_new_output_root(self) -> None:
        text = RUNNER.read_text(encoding="utf-8")
        self.assertIn(
            'RESULT_ROOT="${RESULT_ROOT:-/data/yjh/output/variational_camera_latent}"', text
        )
        self.assertIn('[[ "$(hostname)" == "VM-0-11-ubuntu" ]]', text)
        self.assertIn('SMOKE_SCENE_LIMIT="1"', text)
        self.assertIn('CALIBRATION_SCENE_LIMIT="10"', text)
        self.assertNotIn("HF_TOKEN", text)
        self.assertIn("--stage verify", text)

    def test_null20_runner_exposes_resumable_three_phase_protocol(self) -> None:
        text = NULL20_RUNNER.read_text(encoding="utf-8")
        self.assertIn('[[ "$(hostname)" == "VM-0-11-ubuntu" ]]', text)
        self.assertIn("codex/vrfm-random-null20", text)
        self.assertIn(
            "/data/yjh/output/variational_camera_latent/vrfm_camera_20260827T044926Z",
            text,
        )
        self.assertIn("matched-random-plan", text)
        self.assertIn("matched-random-predict", text)
        self.assertIn("matched-random-finalize", text)
        self.assertIn("--matched-random-replicate-index", text)
        self.assertIn("CUDA_VISIBLE_DEVICES", text)
        self.assertIn('matched_random_null20_gpu_${GPU_INDEX}.lock', text)
        self.assertIn("unset HF_TOKEN", text)

    def test_contract_path_rejects_a_symlinked_parent_inside_the_run_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary_root = Path(directory)
            run_root = temporary_root / "run"
            outside = temporary_root / "outside"
            run_root.mkdir()
            outside.mkdir()
            linked_parent = run_root / "prediction_only"
            try:
                os.symlink(outside, linked_parent, target_is_directory=True)
            except OSError as error:
                self.skipTest(f"directory symlinks are unavailable: {error}")
            candidate = linked_parent / "replicate_000" / "scene.npz"

            with self.assertRaisesRegex(ValueError, "symlink"):
                pipeline_module._require_contract_path(
                    candidate,
                    candidate,
                    run_root=run_root,
                    label="prediction artifact",
                )

    def test_exact_directory_contract_rejects_extra_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = [root / "scene0.npz", root / "scene1.npz"]
            for path in expected:
                path.write_bytes(b"fixture")

            pipeline_module._require_exact_directory_entries(
                root,
                expected,
                entry_kind="file",
                label="prediction replicate",
            )
            (root / "orphan.npz").write_bytes(b"orphan")
            with self.assertRaisesRegex(ValueError, "exactly"):
                pipeline_module._require_exact_directory_entries(
                    root,
                    expected,
                    entry_kind="file",
                    label="prediction replicate",
                )

    def test_aggregate_prediction_barrier_has_a_signed_exact_completion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory)
            manifest = {
                "schema": pipeline_module._MATCHED_RANDOM_20Q_AGGREGATE_PREDICTION_SCHEMA,
                "replicate_count": 20,
                "scene_count": 10,
                "prediction_artifact_count": 200,
                "plan_path": str(run_root / "manifests" / "matched_random_20q_plan.json"),
                "plan_sha256": "a" * 64,
                "plan_digest": "b" * 64,
                "producer_git_commit": "c" * 40,
                "source_manifest_sha256": "d" * 64,
                "candidate_manifest_sha256": "e" * 64,
                "vrfm_prediction_manifest_sha256": "f" * 64,
                "transform_identity_sha256": "1" * 64,
                "replicates": [
                    {
                        "replicate_index": index,
                        "replicate_id": f"formal_null_{index:03d}",
                        "replicate_seed": index,
                        "transform_sha256": f"{index + 2:064x}",
                        "prediction_manifest_path": str(
                            run_root
                            / "manifests"
                            / "matched_random_20q"
                            / f"replicate_{index:03d}_prediction_manifest.json"
                        ),
                        "prediction_manifest_sha256": f"{index + 22:064x}",
                        "prediction_completion_path": str(
                            run_root
                            / "manifests"
                            / "matched_random_20q"
                            / f"replicate_{index:03d}_prediction_complete.json"
                        ),
                        "prediction_completion_sha256": f"{index + 42:064x}",
                    }
                    for index in range(20)
                ],
            }

            first = pipeline_module._seal_matched_random_aggregate_prediction_barrier(
                run_root=run_root,
                manifest=manifest,
            )
            second = pipeline_module._seal_matched_random_aggregate_prediction_barrier(
                run_root=run_root,
                manifest=manifest,
            )

            self.assertEqual(first, second)
            completion = pipeline_module._load_signed_completion(
                first["completion_path"],
                schema=pipeline_module._MATCHED_RANDOM_20Q_AGGREGATE_PREDICTION_COMPLETION_SCHEMA,
            )
            self.assertEqual(completion["prediction_artifact_count"], 200)
            self.assertEqual(
                completion["prediction_manifest_sha256"],
                first["manifest_sha256"],
            )
            self.assertEqual(first["completion_sha256"], pipeline_module._sha256_file(first["completion_path"]))

            with self.assertRaisesRegex(ValueError, "aggregate prediction manifest"):
                pipeline_module._seal_matched_random_aggregate_prediction_barrier(
                    run_root=run_root,
                    manifest={**manifest, "prediction_artifact_count": 199},
                )

    def test_completion_resumes_only_an_exact_payload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "complete.json"
            payload = {"schema": "fixture.v1", "stage": "source", "upstream": "a" * 64}
            written = write_completion(path, payload)

            self.assertEqual(load_exact_completion(path, payload), written)
            self.assertIsNone(
                load_exact_completion(path, {**payload, "upstream": "b" * 64})
            )

    def test_pipeline_exposes_resumable_alpha_scan_stage(self) -> None:
        try:
            args = parse_args(
                [
                    "--stage",
                    "alpha-scan",
                    "--run-root",
                    "/data/yjh/output/variational_camera_latent/fixture",
                ]
            )
        except SystemExit:
            self.fail("pipeline does not expose the alpha-scan stage")

        self.assertEqual(args.stage, "alpha-scan")
        self.assertEqual(args.alpha_min_improvement, 0.01)

    def test_pipeline_exposes_resumable_vrfm_residual_alpha_scan_stage(self) -> None:
        try:
            args = parse_args(
                [
                    "--stage",
                    "vrfm-residual-alpha-scan",
                    "--run-root",
                    "/data/yjh/output/variational_camera_latent/fixture",
                ]
            )
        except SystemExit:
            self.fail("pipeline does not expose the VRFM residual alpha scan stage")

        self.assertEqual(args.stage, "vrfm-residual-alpha-scan")
        self.assertEqual(args.residual_scan_batch_size, 8)

    def test_pipeline_exposes_resumable_matched_random_ablation_stage(self) -> None:
        try:
            args = parse_args(
                [
                    "--stage",
                    "matched-random-ablation",
                    "--run-root",
                    "/data/yjh/output/variational_camera_latent/fixture",
                ]
            )
        except SystemExit:
            self.fail("pipeline does not expose the matched random ablation stage")

        self.assertEqual(args.stage, "matched-random-ablation")
        self.assertEqual(args.matched_random_batch_size, 8)
        self.assertEqual(args.matched_random_seed, 20260827)

    def test_pipeline_exposes_three_phase_matched_random_20q_stages(self) -> None:
        for stage in (
            "matched-random-plan",
            "matched-random-predict",
            "matched-random-finalize",
        ):
            with self.subTest(stage=stage):
                try:
                    args = parse_args(
                        [
                            "--stage",
                            stage,
                            "--run-root",
                            "/data/yjh/output/variational_camera_latent/fixture",
                        ]
                    )
                except SystemExit:
                    self.fail(f"pipeline does not expose {stage}")
                self.assertEqual(args.stage, stage)
        self.assertEqual(args.matched_random_master_seed, 2026082701)

    def test_prediction_stage_requires_a_replicate_index_in_range(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing = parse_args(
                ["--stage", "matched-random-predict", "--run-root", directory]
            )
            with self.assertRaisesRegex(ValueError, "replicate index"):
                run_stage(missing)

            outside = parse_args(
                [
                    "--stage",
                    "matched-random-predict",
                    "--run-root",
                    directory,
                    "--matched-random-replicate-index",
                    "20",
                ]
            )
            with self.assertRaisesRegex(ValueError, "replicate index"):
                run_stage(outside)

    def test_finalize_cannot_load_privileged_state_before_prediction_barrier(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            context = {"run_root": Path(directory)}
            plan_path = Path(directory) / "plan.json"
            privileged_loader = mock.Mock(
                side_effect=AssertionError("privileged state loaded too early")
            )
            with (
                mock.patch.object(
                    pipeline_module,
                    "_load_matched_random_prediction_context",
                    return_value=context,
                ),
                mock.patch.object(
                    pipeline_module,
                    "_load_current_matched_random_20q_plan",
                    return_value=(plan_path, {}, "a" * 64),
                ),
                mock.patch.object(
                    pipeline_module,
                    "_validate_matched_random_prediction_replicates",
                    side_effect=ValueError("missing replicate 019 prediction"),
                ),
                mock.patch.object(
                    pipeline_module,
                    "_load_matched_random_privileged_context",
                    privileged_loader,
                ),
            ):
                with self.assertRaisesRegex(ValueError, "missing replicate 019"):
                    pipeline_module.finalize_matched_random_ensemble(
                        SimpleNamespace()
                    )
            privileged_loader.assert_not_called()

    def test_matched_random_stage_dispatches_to_its_fail_closed_runner(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = parse_args(
                [
                    "--stage",
                    "matched-random-ablation",
                    "--run-root",
                    directory,
                    "--scene-limit",
                    "1",
                ]
            )

            with self.assertRaisesRegex(
                ValueError, "matched random ablation requires exactly ten scenes"
            ):
                run_stage(args)

    def test_matched_random_budget_rejects_shards_without_exactly_32_samples(self) -> None:
        candidate = {"z": np.zeros((8, 31, 16), dtype=np.float32)}
        prediction = {"z": np.zeros((8, 31, 16), dtype=np.float32)}
        privileged = {"candidate_rms": np.zeros((8, 31, 8), dtype=np.float64)}

        with self.assertRaisesRegex(
            ValueError, "exactly 32 samples per overlap"
        ):
            pipeline_module._require_matched_random_sample_budget(
                candidate,
                prediction,
                privileged,
            )

    def test_provenance_rejects_a_dirty_git_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            tracked = root / "tracked.txt"
            tracked.write_text("clean\n", encoding="utf-8")
            subprocess.run(["git", "add", "tracked.txt"], cwd=root, check=True)
            subprocess.run(
                [
                    "git",
                    "-c",
                    "user.name=Codex Test",
                    "-c",
                    "user.email=codex@example.invalid",
                    "commit",
                    "-qm",
                    "fixture",
                ],
                cwd=root,
                check=True,
            )

            commit = pipeline_module._require_clean_git_checkout(root)
            self.assertEqual(len(commit), 40)
            tracked.write_text("dirty\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "clean git checkout"):
                pipeline_module._require_clean_git_checkout(root)

    def test_control_identity_can_only_depend_on_prediction_only_manifests(self) -> None:
        parameters = inspect.signature(
            pipeline_module._matched_random_transform_identity
        ).parameters
        self.assertEqual(
            set(parameters),
            {
                "source_manifest_sha256",
                "candidate_manifest_sha256",
                "vrfm_prediction_manifest_sha256",
            },
        )
        self.assertFalse(
            any(
                forbidden in name
                for name in parameters
                for forbidden in ("privileged", "report", "completion", "gt")
            )
        )

    def test_prediction_manifest_is_sealed_before_privileged_artifacts_are_loaded(self) -> None:
        source = inspect.getsource(pipeline_module.run_matched_random_ablation)
        seal = source.index("_atomic_json(prediction_manifest_path")
        privileged_load = source.index("load_vrfm_residual_privileged")
        self.assertGreater(privileged_load, seal)


if __name__ == "__main__":
    unittest.main()
