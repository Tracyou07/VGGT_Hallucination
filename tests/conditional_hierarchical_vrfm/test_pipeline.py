from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from pre_experiments.conditional_hierarchical_vrfm.pipeline import (
    EXPECTED_SCENES,
    PREFLIGHT_SUITES,
    authenticate_formal_run,
    git_tree_identity,
    audit_long_context_manifest,
    build_long_context_manifest,
    is_expected_formal_file,
    preflight_source_inventory,
    preflight_test_inventory,
    run_calibration,
    run_prepare,
    run_preflight,
    reuse_or_publish_long_context,
    select_resume_checkpoint,
    validate_preflight_evidence,
    validate_frozen_scene_identity,
    validate_source_scene_cohort,
    validate_target_for_stage,
    validate_target_checkpoint_witness,
    validate_variant_zero_against_formal,
    validate_long_context_publication_root,
    verify_target_redecode,
    verify_completed_run,
    verify_inventory_exactness,
)
from pre_experiments.long_short_camera_head.data import SceneRecord
from pre_experiments.long_short_camera_head.labels import save_privileged_labels
from pre_experiments.variational_camera_latent.source import save_source_shard


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_long(path: Path, scene: str, source_sha: str) -> None:
    poses = np.broadcast_to(np.eye(4), (500, 4, 4)).copy()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, scene=np.asarray(scene, dtype="U32"), frame_ids=np.arange(500),
        camera_tokens=np.zeros((500, 2048), dtype=np.float32), baseline_c2w=poses,
        source_sha256=np.asarray(source_sha, dtype="U64"),
    )


def _write_source(path: Path, scene: str) -> str:
    global_ids = np.arange(500, dtype=np.int64)
    global_tokens = np.zeros((500, 2048), dtype=np.float32)
    short_ids = np.stack([global_ids[start:start + 100] for start in range(0, 401, 50)])
    short_tokens = np.stack([global_tokens[start:start + 100] for start in range(0, 401, 50)])
    overlap_ids = np.stack([global_ids[start:start + 50] for start in range(50, 401, 50)])
    overlap = np.stack([global_tokens[start:start + 50] for start in range(50, 401, 50)])
    poses = np.broadcast_to(np.eye(4), (500, 4, 4)).copy()
    save_source_shard(path, {
        "global_frame_ids": global_ids, "global_camera_tokens": global_tokens,
        "short_frame_ids": short_ids, "short_camera_tokens": short_tokens,
        "overlap_frame_ids": overlap_ids, "overlap_long_tokens": overlap,
        "overlap_left_tokens": short_tokens[:-1, 50:],
        "overlap_right_tokens": short_tokens[1:, :50],
        "span_starts": np.arange(0, 400, 50, dtype=np.int64),
        "sample_ids": np.asarray([f"{scene}:overlap_{i:03d}" for i in range(8)]),
        "global_pred_c2w": poses,
        "overlap_long_c2w": np.stack([poses[start:start + 50] for start in range(50, 401, 50)]),
    })
    return _sha(path)


def _write_privileged(path: Path, scene: str, source_sha: str, checkpoint_sha: str) -> str:
    poses = np.repeat(np.eye(4, dtype=np.float64)[None], 500, axis=0)
    return save_privileged_labels(path, {
        "scene": np.asarray(scene, dtype="U32"),
        "frame_ids": np.arange(500, dtype=np.int64),
        "gt_c2w": poses.copy(),
        "oracle_scale": np.asarray(1.0, dtype=np.float64),
        "oracle_rotation": np.eye(3, dtype=np.float64),
        "oracle_translation": np.zeros(3, dtype=np.float64),
        "oracle_digest": np.asarray("d" * 64, dtype="U64"),
        "gt_scene_scale": np.asarray(1.0, dtype=np.float64),
        "baseline_pose_encoding": np.zeros((500, 9), dtype=np.float32),
        "teacher_c2w_gt_gauge": poses.copy(),
        "teacher_weight": np.ones(500, dtype=np.float64),
        "window_teacher_weight": np.ones(9, dtype=np.float64),
        "window_baseline_rms": np.ones(9, dtype=np.float64),
        "window_teacher_rms": np.zeros(9, dtype=np.float64),
        "source_sha256": np.asarray(source_sha, dtype="U64"),
        "checkpoint_sha256": np.asarray(checkpoint_sha, dtype="U64"),
    })


class PipelineBarrierTests(unittest.TestCase):
    @staticmethod
    def _synthetic_preflight_inventory() -> dict[str, list[str]]:
        return {
            suite: [
                f"{suite.replace('/', '.')}.Synthetic.test_{index:03d}"
                for index in range(count)
            ]
            for suite, count in PREFLIGHT_SUITES
        }

    def _check_failed_preflight_command_preserves_raw_log(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inventory = self._synthetic_preflight_inventory()
            completed = [
                SimpleNamespace(
                    returncode=1,
                    stdout="suite failure: FAIL\n",
                    stderr="traceback: ERROR\n",
                ),
                *[
                    SimpleNamespace(returncode=0, stdout="passing suite\n", stderr="")
                    for _ in range(3)
                ],
            ]

            def parse_only_success(
                content: str, suite: str, expected_ids: list[str]
            ) -> list[dict[str, str]]:
                if "FAIL" in content or "ERROR" in content:
                    raise AssertionError("failed command entered pass-only parser")
                return [
                    {"id": identifier, "status": "ok"}
                    for identifier in expected_ids
                ]

            with patch(
                "pre_experiments.conditional_hierarchical_vrfm.pipeline._validate_git"
            ), patch(
                "pre_experiments.conditional_hierarchical_vrfm.pipeline.preflight_test_inventory",
                return_value=inventory,
            ), patch(
                "pre_experiments.conditional_hierarchical_vrfm.pipeline.subprocess.run",
                side_effect=completed,
            ), patch(
                "pre_experiments.conditional_hierarchical_vrfm.pipeline._parse_unittest_results",
                side_effect=parse_only_success,
            ) as parser:
                with self.assertRaisesRegex(ValueError, "preflight command failed"):
                    run_preflight(SimpleNamespace(run_root=root, git_commit="a" * 40))

            log = root / "logs" / "preflight_0.log"
            self.assertEqual(
                log.read_text(encoding="utf-8"),
                "suite failure: FAIL\ntraceback: ERROR\n",
            )
            self.assertEqual(parser.call_count, 2)

    def _check_malformed_success_preserves_raw_log(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inventory = self._synthetic_preflight_inventory()
            completed = [
                SimpleNamespace(
                    returncode=0,
                    stdout="malformed success output\n",
                    stderr="parser context\n",
                ),
                *[
                    SimpleNamespace(returncode=0, stdout="unused\n", stderr="")
                    for _ in range(3)
                ],
            ]
            with patch(
                "pre_experiments.conditional_hierarchical_vrfm.pipeline._validate_git"
            ), patch(
                "pre_experiments.conditional_hierarchical_vrfm.pipeline.preflight_test_inventory",
                return_value=inventory,
            ), patch(
                "pre_experiments.conditional_hierarchical_vrfm.pipeline.subprocess.run",
                side_effect=completed,
            ):
                with self.assertRaisesRegex(ValueError, "stable test IDs"):
                    run_preflight(SimpleNamespace(run_root=root, git_commit="a" * 40))

            log = root / "logs" / "preflight_0.log"
            self.assertEqual(
                log.read_text(encoding="utf-8"),
                "malformed success output\nparser context\n",
            )

    def test_forged_smoke_completion_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root / "smoke" / "completed.json"
            marker.parent.mkdir(parents=True)
            marker.write_text(json.dumps({
                "schema": "conditional_hierarchical_vrfm.smoke_completion.v1",
                "git_commit": "a" * 40, "files": {},
                "metadata": {"scene": "scene0000_00", "variant_count": 4, "steps": 20, "exact_resume": True},
                "record_digest": "0" * 64,
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "completion digest"):
                run_calibration(SimpleNamespace(run_root=root, git_commit="a" * 40))

    def test_authoritative_live_gate_rejects_forged_handwritten_preflight(self) -> None:
        # Regression: FORGED_HANDWRITTEN_PREFLIGHT_ACCEPTED.
        self.assertEqual(dict(PREFLIGHT_SUITES)["tests/conditional_hierarchical_vrfm"], 72)
        self._check_failed_preflight_command_preserves_raw_log()
        self._check_malformed_success_preserves_raw_log()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inventory = preflight_test_inventory()
            rows = []
            commands = [
                [os.sys.executable, "-m", "unittest", "discover", "-s", suite, "-v"]
                for suite in (
                    "tests/conditional_hierarchical_vrfm", "tests/variational_camera_latent",
                    "tests/long_short_camera_head",
                )
            ] + [[os.sys.executable, "-m", "compileall", "-q", "pre_experiments"]]
            for index, command in enumerate(commands):
                log = root / "logs" / f"preflight_{index}.log"
                log.parent.mkdir(parents=True, exist_ok=True)
                if index < 3:
                    suite = command[-2]
                    test_results = [{"id": identifier, "status": "ok"} for identifier in inventory[suite]]
                    lines = []
                    for result in test_results:
                        owner, method = result["id"].rsplit(".", 1)
                        lines.append(f"{method} ({owner}) ... ok")
                    content = "\n".join(lines) + f"\nRan {len(test_results)} tests in 0.001s\n\nOK\n"
                else:
                    test_results = []
                    content = "compile ok\n"
                log.write_text(content, encoding="utf-8")
                rows.append({
                    "command": command, "returncode": 0,
                    "test_count": len(test_results), "skipped_count": 0,
                    "test_results": test_results,
                    "log": log.relative_to(root).as_posix(), "log_sha256": _sha(log),
                })
            unsigned = {"schema": "conditional_hierarchical_vrfm.preflight_evidence.v1",
                        "git_commit": "a" * 40,
                        "source_inventory": preflight_source_inventory(),
                        "test_inventory": inventory,
                        "git_tree": git_tree_identity(),
                        "commands": rows}
            digest = hashlib.sha256(json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            evidence = root / "manifests" / "preflight_evidence.json"
            evidence.parent.mkdir(parents=True)
            evidence.write_text(json.dumps({**unsigned, "record_digest": digest}), encoding="utf-8")
            # Static hashes are intentionally not treated as execution proof.
            validate_preflight_evidence(root, "a" * 40)
            with patch(
                "pre_experiments.conditional_hierarchical_vrfm.pipeline._validate_git"
            ), patch(
                "pre_experiments.conditional_hierarchical_vrfm.pipeline.git_tree_identity",
                return_value=git_tree_identity(),
            ), patch(
                "pre_experiments.conditional_hierarchical_vrfm.pipeline.subprocess.run",
                side_effect=AssertionError("valid preflight evidence must be reused"),
            ):
                self.assertEqual(
                    run_preflight(SimpleNamespace(run_root=root, git_commit="a" * 40)),
                    evidence.resolve(),
                )

            genuine_live = [
                SimpleNamespace(
                    returncode=0,
                    stdout=(root / row["log"]).read_text(encoding="utf-8"),
                    stderr="",
                )
                for row in rows
            ]
            self.assertEqual(len(rows[0]["test_results"]), 72)
            with patch(
                "pre_experiments.conditional_hierarchical_vrfm.pipeline._validate_git"
            ), patch(
                "pre_experiments.conditional_hierarchical_vrfm.pipeline.git_tree_identity",
                return_value=unsigned["git_tree"],
            ), patch(
                "pre_experiments.conditional_hierarchical_vrfm.pipeline.load_source_records",
                side_effect=ValueError("genuine current-tree live authorization accepted"),
            ), patch(
                "pre_experiments.conditional_hierarchical_vrfm.pipeline.subprocess.run",
                side_effect=genuine_live,
            ) as invoked:
                with self.assertRaisesRegex(ValueError, "genuine current-tree live authorization accepted"):
                    run_prepare(SimpleNamespace(
                        run_root=root, git_commit="a" * 40, source_run=root / "source",
                        formal_label_root=root / "formal", prepared_root=root / "prepared",
                        checkpoint_dir=root / "checkpoint", device="cpu",
                    ))
                self.assertEqual(invoked.call_count, 4)
                self.assertEqual([call.args[0] for call in invoked.call_args_list], commands)

            first_result = rows[0]["test_results"][0]
            owner, method = first_result["id"].rsplit(".", 1)
            first_line = f"{method} ({owner}) ... ok\n"
            original = genuine_live[0].stdout
            malformed_outputs = (
                original.replace(first_line, "", 1),
                original.replace(
                    "Ran 72 tests",
                    "test_unexpected (tests.conditional_hierarchical_vrfm.test_pipeline.Unexpected) ... ok\nRan 72 tests",
                    1,
                ),
            )
            for malformed in malformed_outputs:
                attempted = [
                    SimpleNamespace(returncode=0, stdout=malformed, stderr=""),
                    *genuine_live[1:],
                ]
                with patch(
                    "pre_experiments.conditional_hierarchical_vrfm.pipeline._validate_git"
                ), patch(
                    "pre_experiments.conditional_hierarchical_vrfm.pipeline.git_tree_identity",
                    return_value=unsigned["git_tree"],
                ), patch(
                    "pre_experiments.conditional_hierarchical_vrfm.pipeline.load_source_records",
                    side_effect=AssertionError("malformed live results escaped authorization"),
                ), patch(
                    "pre_experiments.conditional_hierarchical_vrfm.pipeline.subprocess.run",
                    side_effect=attempted,
                ) as invoked:
                    with self.assertRaisesRegex(ValueError, "stable test IDs"):
                        run_prepare(SimpleNamespace(
                            run_root=root, git_commit="a" * 40, source_run=root / "source",
                            formal_label_root=root / "formal", prepared_root=root / "prepared",
                            checkpoint_dir=root / "checkpoint", device="cpu",
                        ))
                    self.assertEqual(invoked.call_count, 4)

            live = []
            for index, row in enumerate(rows):
                content = (root / row["log"]).read_text(encoding="utf-8")
                if index == 0:
                    first = row["test_results"][0]
                    owner, method = first["id"].rsplit(".", 1)
                    content = content.replace(
                        f"{method} ({owner}) ... ok",
                        f"{method} ({owner}) ... skipped 'platform capability unavailable'",
                        1,
                    ).replace("\n\nOK\n", "\n\nOK (skipped=1)\n")
                live.append(SimpleNamespace(returncode=0, stdout=content, stderr=""))

            with patch(
                "pre_experiments.conditional_hierarchical_vrfm.pipeline._validate_git"
            ), patch(
                "pre_experiments.conditional_hierarchical_vrfm.pipeline.git_tree_identity",
                return_value=unsigned["git_tree"],
            ), patch(
                "pre_experiments.conditional_hierarchical_vrfm.pipeline.load_source_records",
                side_effect=ValueError("escaped authoritative live preflight"),
            ), patch(
                "pre_experiments.conditional_hierarchical_vrfm.pipeline.subprocess.run",
                side_effect=live,
            ) as invoked:
                with self.assertRaisesRegex(ValueError, "authoritative live preflight"):
                    run_prepare(SimpleNamespace(
                        run_root=root, git_commit="a" * 40, source_run=root / "source",
                        formal_label_root=root / "formal", prepared_root=root / "prepared",
                        checkpoint_dir=root / "checkpoint", device="cpu",
                    ))
                self.assertEqual(invoked.call_count, 4)
                self.assertEqual([call.args[0] for call in invoked.call_args_list], commands)

            smoke_unsigned = {
                "schema": "conditional_hierarchical_vrfm.smoke_completion.v1",
                "git_commit": "a" * 40, "files": {},
                "metadata": {
                    "scene": "scene0000_00", "variant_count": 4,
                    "steps": 20, "exact_resume": True,
                },
            }
            smoke_digest = hashlib.sha256(json.dumps(
                smoke_unsigned, sort_keys=True, separators=(",", ":")
            ).encode()).hexdigest()
            smoke_marker = root / "smoke" / "completed.json"
            smoke_marker.parent.mkdir(parents=True)
            smoke_marker.write_text(json.dumps({
                **smoke_unsigned, "record_digest": smoke_digest,
            }), encoding="utf-8")
            with patch(
                "pre_experiments.conditional_hierarchical_vrfm.pipeline._validate_git"
            ), patch(
                "pre_experiments.conditional_hierarchical_vrfm.pipeline.git_tree_identity",
                return_value=unsigned["git_tree"],
            ), patch(
                "pre_experiments.conditional_hierarchical_vrfm.pipeline.subprocess.run",
                side_effect=live,
            ) as invoked:
                with self.assertRaisesRegex(ValueError, "authoritative live preflight"):
                    run_calibration(SimpleNamespace(
                        run_root=root, git_commit="a" * 40,
                        calibration_executor=lambda _: (_ for _ in ()).throw(
                            ValueError("escaped calibration live gate")
                        ),
                    ))
                self.assertEqual(invoked.call_count, 4)
                self.assertEqual([call.args[0] for call in invoked.call_args_list], commands)

            (root / "config.json").write_text("{}\n", encoding="utf-8")
            inventory_path = root / "manifests" / "verification_inventory.json"
            inventory_path.write_text(json.dumps({
                "schema": "conditional_hierarchical_vrfm.verification_inventory.v1",
                "git_commit": "a" * 40, "classification": "LATENT_TARGETS_READY",
                "files": {},
            }), encoding="utf-8")
            with patch(
                "pre_experiments.conditional_hierarchical_vrfm.pipeline._validate_git"
            ), patch(
                "pre_experiments.conditional_hierarchical_vrfm.pipeline.git_tree_identity",
                return_value=unsigned["git_tree"],
            ), patch(
                "pre_experiments.conditional_hierarchical_vrfm.pipeline.subprocess.run",
                side_effect=live,
            ) as invoked:
                with self.assertRaisesRegex(ValueError, "authoritative live preflight"):
                    verify_completed_run(root, expected_git_commit="a" * 40)
                self.assertEqual(invoked.call_count, 4)
                self.assertEqual([call.args[0] for call in invoked.call_args_list], commands)

    def test_prepare_rejects_rederived_variant_zero_that_differs_from_formal_label(self) -> None:
        pose = np.broadcast_to(np.eye(4), (500, 4, 4)).copy()
        teacher = SimpleNamespace(
            frame_ids=np.arange(500), window_weights=np.ones(9),
            fused_c2w=np.broadcast_to(pose, (4, 500, 4, 4)).copy(),
            coverage_weights=np.ones((4, 500)),
        )
        formal = {
            "frame_ids": np.arange(500), "window_teacher_weight": np.ones(9),
            "teacher_c2w_gt_gauge": pose.copy(), "teacher_weight": np.ones(500),
        }
        validate_variant_zero_against_formal(teacher, formal)
        teacher.fused_c2w[0, 10, 0, 3] = 0.01
        with self.assertRaisesRegex(ValueError, "variant zero"):
            validate_variant_zero_against_formal(teacher, formal)

        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            source_run = workspace / "source_run"
            source_manifest = source_run / "manifests" / "source_manifest.json"
            source_manifest.parent.mkdir(parents=True)
            source_manifest.write_text('{"authenticated":true}\n', encoding="utf-8")
            source_manifest_sha = _sha(source_manifest)
            formal_root = workspace / "long_short_head_formal_20260828T072407Z"
            labels_root = formal_root / "data" / "privileged_labels"
            labels_root.mkdir(parents=True)
            checkpoint_sha = "f164acf60724910d8fe1578bb499d800850c7bb0948db7555c413f9fbe60467e"
            records = []
            rows = []
            for index, scene in enumerate(EXPECTED_SCENES):
                role = "validation" if scene in {"scene0325_01", "scene0675_00"} else "train"
                source_sha = hashlib.sha256(scene.encode()).hexdigest()
                label = labels_root / f"{scene}.npz"
                label_sha = _write_privileged(label, scene, source_sha, checkpoint_sha)
                records.append(SimpleNamespace(scene=scene, role=role, sha256=source_sha))
                rows.append({
                    "scene": scene, "role": role,
                    "source_path": str(source_run / "data" / f"{scene}.npz"),
                    "source_sha256": source_sha,
                    "long_context_path": str(formal_root / "data" / "long_context" / f"{scene}.npz"),
                    "long_context_sha256": "1" * 64,
                    "privileged_path": str(label.resolve()),
                    "privileged_sha256": label_sha,
                    "teacher_frame_count": 500,
                })
            formal_manifest = {
                "schema": "long_short_camera_head.data_manifest.v1",
                "git_revision": "2476a59f583ce4c39bbe66dc65d6a8e5cddfb52e",
                "source_run": str(source_run.resolve()),
                "source_manifest_sha256": source_manifest_sha,
                "prepared_root": str(workspace / "prepared"),
                "checkpoint_dir": str(workspace / "checkpoint"),
                "base_checkpoint_sha256": checkpoint_sha,
                "records": rows,
            }
            manifest_path = formal_root / "manifests" / "data_manifest.json"
            manifest_path.parent.mkdir(parents=True)
            manifest_path.write_text(json.dumps(formal_manifest), encoding="utf-8")
            marker = {
                "schema": "long_short_camera_head.verified_completion.v1",
                "git_revision": "2476a59f583ce4c39bbe66dc65d6a8e5cddfb52e",
                "verifier_git_revision": "2476a59f583ce4c39bbe66dc65d6a8e5cddfb52e",
                "source_manifest_sha256": source_manifest_sha,
                "base_checkpoint_sha256": checkpoint_sha,
                "config_sha256": "2" * 64,
                "data_manifest_sha256": _sha(manifest_path),
                "test_evidence_sha256": "3" * 64,
                "stage_completion_sha256": "4" * 64,
                "scene_count": 10, "train_scene_count": 8,
                "locked_replay_scene_count": 2,
                "classification": "NO_SOURCE_HEAD_SIGNAL",
                "report_sha256": "5" * 64,
                "artifacts": {},
                "inference_leakage_audit": True,
                "formal_protocol_sha256": "6" * 64,
            }
            marker_path = formal_root / "verified_completion.json"
            marker_path.write_text(json.dumps(marker), encoding="utf-8")
            authenticated = authenticate_formal_run(
                formal_root, source_run, records, checkpoint_sha
            )
            self.assertEqual(set(authenticated["labels"]), set(EXPECTED_SCENES))
            self.assertEqual(authenticated["completion_sha256"], _sha(marker_path))

            isolated = workspace / "isolated_labels"
            isolated.mkdir()
            _write_privileged(isolated / "scene0000_00.npz", "scene0000_00", records[0].sha256, checkpoint_sha)
            with self.assertRaisesRegex(ValueError, "verified formal run"):
                authenticate_formal_run(isolated, source_run, records, checkpoint_sha)
            marker["data_manifest_sha256"] = "0" * 64
            marker_path.write_text(json.dumps(marker), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "formal data manifest digest"):
                authenticate_formal_run(formal_root, source_run, records, checkpoint_sha)
    def test_smoke_must_complete_before_calibration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(run_root=Path(directory))
            with self.assertRaisesRegex(ValueError, "smoke completion"):
                run_calibration(args)

    def test_prediction_only_manifest_contains_no_short_or_privileged_path(self) -> None:
        source_manifest = {
            "records": [{"scene": "scene0000_00", "role": "train", "sha256": "a" * 64}]
        }
        manifest = build_long_context_manifest(source_manifest)
        serialized = json.dumps(manifest).lower()
        for forbidden in ("short", "teacher", "privileged", "gt", "prepared"):
            self.assertNotIn(forbidden, serialized)
        expected = (
            "scene0000_00", "scene0013_02", "scene0029_01", "scene0084_01",
            "scene0121_01", "scene0207_01", "scene0280_00", "scene0325_01",
            "scene0675_00", "scene0691_00",
        )
        self.assertEqual(EXPECTED_SCENES, expected)
        validate_frozen_scene_identity(
            Path("configs/scannet50_camera_velocity_ambiguity_02_split_v2.json")
        )
        records = [
            SimpleNamespace(
                scene=scene,
                role="validation" if scene in {"scene0325_01", "scene0675_00"} else "train",
            )
            for scene in expected
        ]
        validate_source_scene_cohort(records)
        records[-1] = SimpleNamespace(scene="scene9999_99", role="validation")
        with self.assertRaisesRegex(ValueError, "exact ten calibration scenes"):
            validate_source_scene_cohort(records)

    def test_audit_requires_physical_strict_long_only_shard(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_sha = "a" * 64
            path = root / "prediction_only" / "long_context" / "scene0000_00.npz"
            _write_long(path, "scene0000_00", source_sha)
            manifest = build_long_context_manifest({
                "records": [{"scene": "scene0000_00", "role": "train", "sha256": source_sha}]
            })
            manifest["records"][0]["sha256"] = _sha(path)
            audit_long_context_manifest(root, manifest)
            with np.load(path, allow_pickle=False) as archive:
                arrays = {name: np.asarray(archive[name]) for name in archive.files}
            arrays["benign"] = np.zeros((9, 100, 2048), dtype=np.float32)
            np.savez_compressed(path, **arrays)
            manifest["records"][0]["sha256"] = _sha(path)
            with self.assertRaisesRegex(ValueError, "strict long-only"):
                audit_long_context_manifest(root, manifest)
            source_path = root / "source.npz"
            source_sha = _write_source(source_path, "scene0000_00")
            record = SceneRecord("scene0000_00", "train", source_path, source_sha)
            resumable = root / "resumable" / "scene0000_00.npz"
            first = reuse_or_publish_long_context(record, resumable)
            second = reuse_or_publish_long_context(record, resumable)
            self.assertEqual(first.sha256, second.sha256)
            with np.load(resumable, allow_pickle=False) as archive:
                arrays = {name: np.asarray(archive[name]).copy() for name in archive.files}
            arrays["camera_tokens"][0, 0] = 1.0
            np.savez_compressed(resumable, **arrays)
            with self.assertRaisesRegex(ValueError, "existing long-context"):
                reuse_or_publish_long_context(record, resumable)

    def test_long_only_audit_rejects_symlink_and_path_escape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_sha = "a" * 64
            outside = root / "outside.npz"
            _write_long(outside, "scene0000_00", source_sha)
            link = root / "prediction_only" / "long_context" / "scene0000_00.npz"
            link.parent.mkdir(parents=True)
            try:
                os.symlink(outside, link)
            except OSError:
                self.skipTest("symlink privilege is unavailable")
            manifest = build_long_context_manifest({
                "records": [{"scene": "scene0000_00", "role": "train", "sha256": source_sha}]
            })
            manifest["records"][0]["sha256"] = _sha(outside)
            with self.assertRaisesRegex(ValueError, "symlink"):
                audit_long_context_manifest(root, manifest)
            manifest["records"][0]["file"] = "../outside.npz"
            with self.assertRaisesRegex(ValueError, "file identity"):
                audit_long_context_manifest(root, manifest)
            outside_directory = root / "outside_directory"
            outside_directory.mkdir()
            linked_root = root / "linked_run"
            os.symlink(outside_directory, linked_root)
            with self.assertRaisesRegex(ValueError, "symlink"):
                validate_long_context_publication_root(linked_root)

    def test_invalid_final_and_smoke_target_are_not_calibration_finals(self) -> None:
        target = {"optimization_steps": np.full(4, 20), "initial_losses": np.ones(4),
                  "final_losses": np.zeros(4)}
        with self.assertRaisesRegex(ValueError, "250-step final"):
            validate_target_for_stage(target, steps=250)
        target["optimization_steps"] = np.full(4, 250)
        target["final_losses"][2] = 2.0
        with self.assertRaisesRegex(ValueError, "decreasing"):
            validate_target_for_stage(target, steps=250)
        coefficients = np.zeros((4, 32, 2048), dtype=np.float32)
        bound_target = {
            "optimization_steps": np.full(4, 250, dtype=np.int64),
            "initial_losses": np.ones(4, dtype=np.float64),
            "final_losses": np.full(4, 0.25, dtype=np.float64),
            "residual_coefficients": coefficients,
            "source_sha256": np.asarray("a" * 64),
            "teacher_sha256": np.asarray("b" * 64),
            "basis_sha256": np.asarray("c" * 64),
            "checkpoint_sha256": np.asarray("d" * 64),
            "git_commit": np.asarray("e" * 40),
        }
        checkpoints = [
            {
                "variant_index": variant, "next_step": 250,
                "loss_trace": [1.0] + [0.25] * 249,
                "initial_loss": 1.0, "best_loss": 0.25,
                "best_coefficients": torch.from_numpy(coefficients[variant]).unsqueeze(0),
                "source_sha256": "a" * 64, "teacher_sha256": "b" * 64,
                "basis_sha256": "c" * 64,
                "camera_head_checkpoint_sha256": "d" * 64,
                "git_commit": "e" * 40,
            }
            for variant in range(4)
        ]
        validate_target_checkpoint_witness(bound_target, checkpoints, steps=250)
        tampered = dict(bound_target)
        tampered["residual_coefficients"] = coefficients.copy()
        tampered["residual_coefficients"][2, 0, 0] = 1.0
        with self.assertRaisesRegex(ValueError, "checkpoint witness"):
            validate_target_checkpoint_witness(tampered, checkpoints, steps=250)
        invalid_dtype = [dict(row) for row in checkpoints]
        invalid_dtype[0]["best_coefficients"] = invalid_dtype[0]["best_coefficients"].double()
        with self.assertRaisesRegex(ValueError, "checkpoint witness"):
            validate_target_checkpoint_witness(bound_target, invalid_dtype, steps=250)
        invalid_provenance = [dict(row) for row in checkpoints]
        invalid_provenance[3]["teacher_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "checkpoint witness"):
            validate_target_checkpoint_witness(bound_target, invalid_provenance, steps=250)

    def test_calibration_resume_source_is_checkpoint_never_smoke_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            smoke_target = root / "smoke" / "latent_targets" / "scene0000_00.npz"
            checkpoint = root / "smoke" / "checkpoints" / "scene0000_00" / "variant_0.pt"
            smoke_target.parent.mkdir(parents=True)
            checkpoint.parent.mkdir(parents=True)
            smoke_target.write_bytes(b"summary")
            checkpoint.write_bytes(b"resume-state")
            selected = select_resume_checkpoint(
                final_target=root / "privileged_labels" / "latent_targets" / "scene0000_00.npz",
                checkpoint=checkpoint,
            )
            self.assertEqual(selected, checkpoint)
            checkpoint.unlink()
            self.assertIsNone(select_resume_checkpoint(
                final_target=root / "privileged_labels" / "latent_targets" / "scene0000_00.npz",
                checkpoint=checkpoint,
            ))
    def test_fabricated_decoded_pose_is_rejected_by_real_redecode(self) -> None:
        class Head(nn.Module):
            def decode_pose_tokens(self, tokens: torch.Tensor, *, num_iterations: int):
                raw = torch.zeros((*tokens.shape[:2], 9), device=tokens.device)
                raw[..., 3] = 1.0
                return [raw]
        tokens = torch.zeros(1, 500, 2048)
        coefficients = torch.zeros(1, 32, 2048)
        expected = verify_target_redecode(Head(), tokens, coefficients, expected=None)
        expected[0, 0, 3] += 1.0
        with self.assertRaisesRegex(ValueError, "re-decode"):
            verify_target_redecode(Head(), tokens, coefficients, expected=expected)

    def test_verify_rehashes_every_artifact_and_rejects_extra_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            allowed = root / "reports" / "stage_a.json"
            allowed.parent.mkdir(parents=True)
            allowed.write_text('{"classification":"LATENT_LIFT_FAILED"}\n', encoding="utf-8")
            inventory = {
                "schema": "conditional_hierarchical_vrfm.verification_inventory.v1",
                "git_commit": "a" * 40,
                "classification": "LATENT_LIFT_FAILED",
                "files": {"reports/stage_a.json": _sha(allowed)},
            }
            manifest = root / "manifests" / "verification_inventory.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text(json.dumps(inventory, sort_keys=True), encoding="utf-8")
            verify_inventory_exactness(root, inventory)
            self.assertTrue(is_expected_formal_file(
                "checkpoints/calibration/scene0013_02/variant_3.pt"
            ))
            self.assertFalse(is_expected_formal_file(
                "checkpoints/calibration/scene9999_99/variant_3.pt"
            ))
            extra_directory = root / "checkpoints" / "calibration" / "scene9999_99"
            extra_directory.mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "exact directory"):
                verify_inventory_exactness(root, inventory)
            extra_directory.rmdir()
            extra = root / "privileged_labels" / "latent_targets" / "extra.npz"
            extra.parent.mkdir(parents=True)
            extra.write_bytes(b"x")
            with self.assertRaisesRegex(ValueError, "exact directory"):
                verify_inventory_exactness(root, inventory)

    def test_verify_completed_run_rejects_missing_formal_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = root / "reports" / "stage_a.json"
            report.parent.mkdir(parents=True)
            report.write_text('{}\n', encoding="utf-8")
            inventory = {
                "schema": "conditional_hierarchical_vrfm.verification_inventory.v1",
                "git_commit": "a" * 40,
                "classification": "LATENT_LIFT_FAILED",
                "files": {"reports/stage_a.json": _sha(report)},
            }
            manifest = root / "manifests" / "verification_inventory.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text(json.dumps(inventory, sort_keys=True), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "formal config"):
                verify_completed_run(root, expected_git_commit="a" * 40)


if __name__ == "__main__":
    unittest.main()
