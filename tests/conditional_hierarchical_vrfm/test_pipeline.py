from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import os
import tempfile
import unittest

import numpy as np
import torch
from torch import nn

from pre_experiments.conditional_hierarchical_vrfm.pipeline import (
    audit_long_context_manifest,
    build_long_context_manifest,
    run_calibration,
    select_resume_checkpoint,
    validate_preflight_evidence,
    validate_target_for_stage,
    validate_variant_zero_against_formal,
    verify_target_redecode,
    verify_completed_run,
    verify_inventory_exactness,
)


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


class PipelineBarrierTests(unittest.TestCase):
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

    def test_stale_or_fabricated_preflight_log_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
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
                log.write_text("fabricated\n", encoding="utf-8")
                rows.append({"command": command, "returncode": 0, "test_count": 1 if index < 3 else 0,
                             "log": log.relative_to(root).as_posix(), "log_sha256": _sha(log)})
            unsigned = {"schema": "conditional_hierarchical_vrfm.preflight_evidence.v1",
                        "git_commit": "a" * 40, "commands": rows}
            digest = hashlib.sha256(json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            evidence = root / "manifests" / "preflight_evidence.json"
            evidence.parent.mkdir(parents=True)
            evidence.write_text(json.dumps({**unsigned, "record_digest": digest}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not prove"):
                validate_preflight_evidence(root, "a" * 40)

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

    def test_invalid_final_and_smoke_target_are_not_calibration_finals(self) -> None:
        target = {"optimization_steps": np.full(4, 20), "initial_losses": np.ones(4),
                  "final_losses": np.zeros(4)}
        with self.assertRaisesRegex(ValueError, "250-step final"):
            validate_target_for_stage(target, steps=250)
        target["optimization_steps"] = np.full(4, 250)
        target["final_losses"][2] = 2.0
        with self.assertRaisesRegex(ValueError, "decreasing"):
            validate_target_for_stage(target, steps=250)

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
