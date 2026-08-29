from __future__ import annotations

import ast
from contextlib import ExitStack, contextmanager
from io import BytesIO
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock
import warnings
import zipfile

import numpy as np
import torch

from pre_experiments.camera_translation_hvrfm.data import (
    PublishedTranslationSample,
    publish_translation_sample,
)
from pre_experiments.camera_translation_hvrfm.evaluate import (
    evaluate_translation_sample,
)
from pre_experiments.camera_translation_hvrfm.geometry import prediction_scale
from pre_experiments.camera_translation_hvrfm.report import (
    _markdown_bytes,
    build_stage_a_prime_report,
    write_stage_a_prime_report,
)
from pre_experiments.camera_translation_hvrfm.teacher import (
    build_raw_gauge_teacher,
    load_teacher_controls,
)
from pre_experiments.camera_velocity_ambiguity_02.artifacts import frame_digest
from pre_experiments.camera_velocity_ambiguity_02.contracts import (
    canonical_json_digest,
)
from pre_experiments.conditional_hierarchical_vrfm.artifacts import (
    save_teacher_artifact,
)
from pre_experiments.conditional_hierarchical_vrfm.teacher import (
    build_variant_window_masks,
)
from pre_experiments.variational_camera_latent.contracts import SourceShardRecord
from tests.camera_translation_hvrfm.test_teacher import (
    TokenCameraHead,
    default_controls_arrays,
    make_reference_arrays,
    make_source_arrays,
)


try:
    from pre_experiments.camera_translation_hvrfm import verify
except (ImportError, ModuleNotFoundError):
    verify = None  # type: ignore[assignment]


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
VALIDATION_SCENES = frozenset({"scene0325_01", "scene0675_00"})


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _rewrite_npz(path: Path, mutate) -> None:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    mutate(arrays)
    with path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[2],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _role(scene: str) -> str:
    return "validation" if scene in VALIDATION_SCENES else "train"


class ValidRunFixture:
    """Build a production-published cohort that the verifier must replay itself."""

    def __init__(self, root: Path) -> None:
        self.workspace = root
        self.run_id = "translation-endpoints-ready-test"
        self.run_root = root / self.run_id
        self.git_commit = _git_commit()
        self.checkpoint_dir = root / "VGGT-1B"
        self.checkpoint_file = self.checkpoint_dir / "model.safetensors"
        self.checkpoint_file.parent.mkdir(parents=True)
        self.checkpoint_file.write_bytes(b"independent verifier fake checkpoint\n")
        self.checkpoint_sha256 = _sha(self.checkpoint_file)
        self.source_root = root / "vrfm_camera_20260827T044926Z"
        self.reference_root = (
            root / "privileged_teacher_lift_20260829T012716Z_tolfix"
        )
        self.formal_root = root / "long_short_head_formal_20260828T072407Z"
        self.source_paths: dict[str, Path] = {}
        self.source_sha256: dict[str, str] = {}
        self.reference_teacher_paths: dict[str, Path] = {}
        self.reference_teacher_sha256: dict[str, str] = {}
        self.reference_long_paths: dict[str, Path] = {}
        self.reference_long_sha256: dict[str, str] = {}
        self.formal_paths: dict[str, Path] = {}
        self.formal_sha256: dict[str, str] = {}
        self.samples = [self._publish_scene(scene) for scene in SCENES]
        self._publish_upstream_metadata()
        self._publish_metadata()

    def _publish_scene(self, scene: str) -> PublishedTranslationSample:
        source = make_source_arrays(perturbed_windows=True)
        source["sample_ids"] = np.asarray(
            [f"{scene}:overlap_{index:03d}" for index in range(8)], dtype="U64"
        )
        source_path = (
            self.source_root / "prediction_only" / "source" / f"{scene}.npz"
        )
        source_path.parent.mkdir(parents=True, exist_ok=True)
        with source_path.open("wb") as handle:
            np.savez_compressed(handle, **source)
        source_sha256 = _sha(source_path)
        self.source_paths[scene] = source_path
        self.source_sha256[scene] = source_sha256

        weights, _, _ = default_controls_arrays()
        masks = build_variant_window_masks(scene, weights).astype(np.uint8)
        reference = make_reference_arrays(
            source,
            source_sha256=source_sha256,
            formal_label_sha256="f" * 64,
            weights=weights,
            masks=masks,
        )
        reference["scene"] = np.asarray(scene, dtype="U32")
        reference["oracle_scene"] = np.asarray(scene, dtype="U32")
        reference["checkpoint_sha256"] = np.asarray(
            self.checkpoint_sha256, dtype="U64"
        )
        oracle_payload = {
            "scene": scene,
            "frame_digest": frame_digest(source["global_frame_ids"]),
            "fit_count": 500,
            "scale": 1.0,
            "rotation": (
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
            ),
            "translation": (0.0, 0.0, 0.0),
        }
        reference["oracle_digest"] = np.asarray(
            canonical_json_digest(oracle_payload), dtype="U64"
        )
        reference_path = (
            self.reference_root
            / "privileged_labels"
            / "teacher"
            / f"{scene}.npz"
        )
        reference_path.parent.mkdir(parents=True, exist_ok=True)
        save_teacher_artifact(reference_path, reference)
        controls = load_teacher_controls(
            reference_path,
            expected_sha256=_sha(reference_path),
            expected_source_sha256=source_sha256,
            expected_checkpoint_sha256=self.checkpoint_sha256,
            expected_formal_label_sha256="f" * 64,
        )
        teacher = build_raw_gauge_teacher(
            source_path,
            controls,
            TokenCameraHead(),
            expected_source_sha256=source_sha256,
            checkpoint_sha256=self.checkpoint_sha256,
            device=torch.device("cpu"),
        )
        ground_truth = teacher.baseline_c2w.copy()
        ground_truth[:, :3, 3] = teacher.filled_teacher_centers[0]
        reference["gt_c2w"] = ground_truth
        reference["gt_scene_scale"] = np.asarray(
            prediction_scale(ground_truth), dtype=np.float64
        )
        fused = np.broadcast_to(
            teacher.baseline_c2w[None], (4, 500, 4, 4)
        ).copy()
        fused[:, :, :3, 3] = teacher.filled_teacher_centers
        reference["fused_c2w"] = fused

        formal_path = (
            self.formal_root / "data" / "privileged_labels" / f"{scene}.npz"
        )
        formal_path.parent.mkdir(parents=True, exist_ok=True)
        formal_teacher = teacher.baseline_c2w.copy()
        formal_teacher[:, :3, 3] = teacher.filled_teacher_centers[0]
        formal_arrays = {
            "scene": np.asarray(scene, dtype="U32"),
            "frame_ids": source["global_frame_ids"].astype(np.int64, copy=True),
            "gt_c2w": ground_truth.copy(),
            "oracle_scale": reference["oracle_scale"].copy(),
            "oracle_rotation": reference["oracle_rotation"].copy(),
            "oracle_translation": reference["oracle_translation"].copy(),
            "oracle_digest": reference["oracle_digest"].copy(),
            "gt_scene_scale": reference["gt_scene_scale"].copy(),
            "baseline_pose_encoding": teacher.baseline_pose_encoding.copy(),
            "teacher_c2w_gt_gauge": formal_teacher,
            "teacher_weight": reference["coverage_weights"][0].copy(),
            "window_teacher_weight": weights.copy(),
            "window_baseline_rms": np.ones(9, dtype=np.float64),
            "window_teacher_rms": np.full(9, 0.5, dtype=np.float64),
            "source_sha256": np.asarray(source_sha256, dtype="U64"),
            "checkpoint_sha256": np.asarray(
                self.checkpoint_sha256, dtype="U64"
            ),
        }
        with formal_path.open("wb") as handle:
            np.savez_compressed(handle, **formal_arrays)
        formal_sha256 = _sha(formal_path)
        self.formal_paths[scene] = formal_path
        self.formal_sha256[scene] = formal_sha256
        reference["formal_label_sha256"] = np.asarray(
            formal_sha256, dtype="U64"
        )
        save_teacher_artifact(reference_path, reference)
        reference_sha256 = _sha(reference_path)
        self.reference_teacher_paths[scene] = reference_path
        self.reference_teacher_sha256[scene] = reference_sha256

        record = SourceShardRecord(
            scene=scene,
            role=_role(scene),
            path=source_path,
            overlap_count=8,
            sha256=source_sha256,
        )
        with mock.patch(
            "pre_experiments.camera_translation_hvrfm.data._current_git_commit",
            return_value=self.git_commit,
        ):
            return publish_translation_sample(
                self.run_root,
                role=_role(scene),
                source_path=source_path,
                source_record=record,
                teacher_reference_path=reference_path,
                expected_teacher_reference_sha256=reference_sha256,
                formal_label_path=formal_path,
                expected_formal_label_sha256=formal_sha256,
                camera_head=TokenCameraHead(),
                checkpoint_sha256=self.checkpoint_sha256,
                git_commit=self.git_commit,
                device=torch.device("cpu"),
            )

    def _publish_upstream_metadata(self) -> None:
        stale_root = Path("/stale/migrated/vrfm/prediction_only/source")
        source_rows = [
            {
                "scene": scene,
                "role": _role(scene),
                "path": str(stale_root / f"{scene}.npz"),
                "overlap_count": 8,
                "sha256": self.source_sha256[scene],
            }
            for scene in SCENES
        ]
        self.source_manifest = self.source_root / "manifests/source_manifest.json"
        _write_json(
            self.source_manifest,
            {
                "schema": "variational_camera_latent.source.v1",
                "dataset_root": str(stale_root),
                "source_run_digest": "6" * 64,
                "records": source_rows,
            },
        )
        source_unsigned = {
            "schema": "variational_camera_latent.verified_completion.v1",
            "signal": "WEAK_SIGNAL",
            "scene_count": 10,
            "overlap_count": 80,
            "candidate_count": 2560,
            "prediction_manifest_sha256": "1" * 64,
            "privileged_manifest_sha256": "2" * 64,
            "report_sha256": "3" * 64,
        }
        self.source_completion = self.source_root / "verified_completion.json"
        _write_json(
            self.source_completion,
            {
                **source_unsigned,
                "completion_digest": _digest(source_unsigned),
            },
        )
        self.source_manifest_sha256 = _sha(self.source_manifest)
        self.source_completion_sha256 = _sha(self.source_completion)

        for sample in self.samples:
            destination = (
                self.reference_root
                / "prediction_only"
                / "long_context"
                / f"{sample.scene}.npz"
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(sample.long_path.read_bytes())
            self.reference_long_paths[sample.scene] = destination
            self.reference_long_sha256[sample.scene] = _sha(destination)

        formal_rows = [
            {
                "scene": scene,
                "role": _role(scene),
                "source_path": str(self.source_paths[scene].resolve()),
                "source_sha256": self.source_sha256[scene],
                "long_context_path": str(
                    (
                        self.formal_root
                        / "data"
                        / "long_context"
                        / f"{scene}.npz"
                    ).resolve()
                ),
                "long_context_sha256": self.reference_long_sha256[scene],
                "privileged_path": str(self.formal_paths[scene].resolve()),
                "privileged_sha256": self.formal_sha256[scene],
                "teacher_frame_count": 500,
            }
            for scene in SCENES
        ]
        self.formal_manifest = self.formal_root / "manifests/data_manifest.json"
        _write_json(
            self.formal_manifest,
            {
                "schema": "long_short_camera_head.data_manifest.v1",
                "git_revision": "2476a59f583ce4c39bbe66dc65d6a8e5cddfb52e",
                "source_run": str(self.source_root.resolve()),
                "source_manifest_sha256": self.source_manifest_sha256,
                "prepared_root": str((self.workspace / "prepared").resolve()),
                "checkpoint_dir": str(self.checkpoint_dir.resolve()),
                "base_checkpoint_sha256": self.checkpoint_sha256,
                "records": formal_rows,
            },
        )
        self.formal_manifest_sha256 = _sha(self.formal_manifest)
        formal_completion = {
            "schema": "long_short_camera_head.verified_completion.v1",
            "git_revision": "2476a59f583ce4c39bbe66dc65d6a8e5cddfb52e",
            "verifier_git_revision": "2476a59f583ce4c39bbe66dc65d6a8e5cddfb52e",
            "source_manifest_sha256": self.source_manifest_sha256,
            "base_checkpoint_sha256": self.checkpoint_sha256,
            "config_sha256": "4" * 64,
            "data_manifest_sha256": self.formal_manifest_sha256,
            "test_evidence_sha256": "5" * 64,
            "stage_completion_sha256": {
                "evaluation_gt_only": "6" * 64,
                "evaluation_long_short": "7" * 64,
                "smoke": "8" * 64,
                "training_gt_only": "9" * 64,
                "training_long_short": "a" * 64,
            },
            "scene_count": 10,
            "train_scene_count": 8,
            "locked_replay_scene_count": 2,
            "classification": "NO_SOURCE_HEAD_SIGNAL",
            "report_sha256": "b" * 64,
            "artifacts": [
                {
                    "scene": scene,
                    "variant": variant,
                    "checkpoint_sha256": "c" * 64,
                    "prediction_sha256": "d" * 64,
                    "evaluation_sha256": "e" * 64,
                }
                for scene in SCENES
                for variant in ("gt_only", "long_short")
            ],
            "inference_leakage_audit": True,
            "formal_protocol_sha256": "f" * 64,
        }
        self.formal_completion = self.formal_root / "verified_completion.json"
        _write_json(self.formal_completion, formal_completion)
        self.formal_completion_sha256 = _sha(self.formal_completion)

        long_rows = [
            {
                "scene": scene,
                "role": _role(scene),
                "file": f"{scene}.npz",
                "sha256": self.reference_long_sha256[scene],
                "source_sha256": self.source_sha256[scene],
            }
            for scene in SCENES
        ]
        teacher_rows = [
            {
                "scene": scene,
                "role": _role(scene),
                "file": f"privileged_labels/teacher/{scene}.npz",
                "sha256": self.reference_teacher_sha256[scene],
                "formal_label_sha256": self.formal_sha256[scene],
            }
            for scene in SCENES
        ]
        self.reference_long_manifest = (
            self.reference_root / "manifests/long_context.json"
        )
        _write_json(
            self.reference_long_manifest,
            {
                "schema": "conditional_hierarchical_vrfm.long_context_manifest.v1",
                "records": long_rows,
            },
        )
        self.reference_teacher_manifest = (
            self.reference_root / "manifests/teacher.json"
        )
        _write_json(
            self.reference_teacher_manifest,
            {
                "schema": "conditional_hierarchical_vrfm.teacher_manifest.v1",
                "git_commit": "cee41a09ac4085c8d6b0b343ca07d8e8c53ace3c",
                "checkpoint_sha256": self.checkpoint_sha256,
                "teacher_upper_bound": {
                    "scene_count": 10,
                    "positive_scene_count": 10,
                    "mean_coverage": 0.89,
                    "mean_utility": 0.1293578270771188,
                },
                "formal_completion_sha256": self.formal_completion_sha256,
                "formal_data_manifest_sha256": self.formal_manifest_sha256,
                "records": teacher_rows,
            },
        )
        self.reference_config = self.reference_root / "config.json"
        _write_json(
            self.reference_config,
            {
                "schema": "conditional_hierarchical_vrfm.run_config.v1",
                "git_commit": "cee41a09ac4085c8d6b0b343ca07d8e8c53ace3c",
                "checkpoint_sha256": self.checkpoint_sha256,
                "basis_sha256": "89fecc83d51be8a1923a0c177c20b45dec8d8aa611fae0b615ef9293511dd213",
                "long_manifest_sha256": _sha(self.reference_long_manifest),
                "teacher_manifest_sha256": _sha(self.reference_teacher_manifest),
                "source_run": str(self.source_root.resolve()),
                "source_manifest_sha256": self.source_manifest_sha256,
                "formal_run_root": str(self.formal_root.resolve()),
                "formal_completion_sha256": self.formal_completion_sha256,
                "formal_data_manifest_sha256": self.formal_manifest_sha256,
                "smoke_scene": "scene0000_00",
                "smoke_steps": 20,
                "calibration_steps": 250,
                "scene_count": 10,
                "variant_count": 4,
            },
        )
        self.reference_report_json = self.reference_root / "reports/stage_a.json"
        _write_json(
            self.reference_report_json,
            {
                "schema": "conditional_hierarchical_vrfm.stage_a_report.v1",
                "git_commit": "cee41a09ac4085c8d6b0b343ca07d8e8c53ace3c",
                "classification": "LATENT_LIFT_FAILED",
                "failed_gates": [
                    "teacher_retention",
                    "per_scene_harm",
                    "rotation_guard",
                    "uncovered_anchor",
                ],
                "scene_metrics": [{"scene": scene} for scene in SCENES],
                "provenance": {
                    "checkpoint_sha256": self.checkpoint_sha256,
                    "basis_sha256": "89fecc83d51be8a1923a0c177c20b45dec8d8aa611fae0b615ef9293511dd213",
                    "long_manifest_sha256": _sha(self.reference_long_manifest),
                    "teacher_manifest_sha256": _sha(
                        self.reference_teacher_manifest
                    ),
                },
            },
        )
        self.reference_report_markdown = self.reference_root / "reports/stage_a.md"
        self.reference_report_markdown.parent.mkdir(parents=True, exist_ok=True)
        self.reference_report_markdown.write_text(
            "# LATENT_LIFT_FAILED\n", encoding="utf-8"
        )
        inventory_files = {
            f"prediction_only/long_context/{scene}.npz": self.reference_long_sha256[
                scene
            ]
            for scene in SCENES
        }
        inventory_files.update(
            {
                f"privileged_labels/teacher/{scene}.npz": self.reference_teacher_sha256[
                    scene
                ]
                for scene in SCENES
            }
        )
        retained = {
            "config.json": self.reference_config,
            "manifests/long_context.json": self.reference_long_manifest,
            "manifests/teacher.json": self.reference_teacher_manifest,
            "reports/stage_a.json": self.reference_report_json,
            "reports/stage_a.md": self.reference_report_markdown,
        }
        inventory_files.update(
            {relative: _sha(path) for relative, path in retained.items()}
        )
        expected = {
            "config.json",
            "manifests/preflight_evidence.json",
            "manifests/long_context.json",
            "manifests/teacher.json",
            "smoke/completed.json",
            "calibration/completed.json",
            "reports/stage_a.json",
            "reports/stage_a.md",
            *(f"logs/preflight_{index}.log" for index in range(4)),
        }
        for scene in SCENES:
            expected.update(
                {
                    f"prediction_only/long_context/{scene}.npz",
                    f"privileged_labels/teacher/{scene}.npz",
                    f"privileged_labels/latent_targets/{scene}.npz",
                    *(
                        f"checkpoints/calibration/{scene}/variant_{variant}.pt"
                        for variant in range(4)
                    ),
                }
            )
        expected.add("smoke/latent_targets/scene0000_00.npz")
        expected.update(
            f"smoke/checkpoints/scene0000_00/variant_{variant}.pt"
            for variant in range(4)
        )
        for relative in sorted(expected - set(inventory_files)):
            path = self.reference_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"inventory:{relative}\n".encode())
            inventory_files[relative] = _sha(path)
        self.reference_inventory = (
            self.reference_root / "manifests/verification_inventory.json"
        )
        _write_json(
            self.reference_inventory,
            {
                "schema": "conditional_hierarchical_vrfm.verification_inventory.v1",
                "git_commit": "cee41a09ac4085c8d6b0b343ca07d8e8c53ace3c",
                "classification": "LATENT_LIFT_FAILED",
                "files": inventory_files,
            },
        )
        reference_unsigned = {
            "schema": "conditional_hierarchical_vrfm.verified_completion.v1",
            "git_commit": "cee41a09ac4085c8d6b0b343ca07d8e8c53ace3c",
            "classification": "LATENT_LIFT_FAILED",
            "inventory_sha256": _sha(self.reference_inventory),
            "file_count": 87,
        }
        self.reference_completion = self.reference_root / "verified_completion.json"
        _write_json(
            self.reference_completion,
            {
                **reference_unsigned,
                "completion_digest": _digest(reference_unsigned),
            },
        )
        self.reference_completion_sha256 = _sha(self.reference_completion)
        self.reference_inventory_sha256 = _sha(self.reference_inventory)
        self.reference_config_sha256 = _sha(self.reference_config)
        self.reference_report_json_sha256 = _sha(self.reference_report_json)
        self.reference_report_markdown_sha256 = _sha(
            self.reference_report_markdown
        )
        self.reference_long_manifest_sha256 = _sha(
            self.reference_long_manifest
        )
        self.reference_teacher_manifest_sha256 = _sha(
            self.reference_teacher_manifest
        )

    @contextmanager
    def patched_frozen(self, module):
        constants = {
            "_FROZEN_SOURCE_COMPLETION_SHA256": self.source_completion_sha256,
            "_FROZEN_SOURCE_MANIFEST_SHA256": self.source_manifest_sha256,
            "_FROZEN_REFERENCE_COMPLETION_SHA256": self.reference_completion_sha256,
            "_FROZEN_REFERENCE_LONG_MANIFEST_SHA256": self.reference_long_manifest_sha256,
            "_FROZEN_FORMAL_COMPLETION_SHA256": self.formal_completion_sha256,
            "_FROZEN_FORMAL_DATA_MANIFEST_SHA256": self.formal_manifest_sha256,
            "_FROZEN_CHECKPOINT_SHA256": self.checkpoint_sha256,
        }
        with ExitStack() as stack:
            for name, value in constants.items():
                stack.enter_context(mock.patch.object(module, name, value))
            yield

    def _stage_marker(
        self,
        path: Path,
        *,
        schema: str,
        stage: str,
        previous: Path,
        files: dict[str, str],
        metadata: dict[str, object],
    ) -> Path:
        unsigned = {
            "schema": schema,
            "stage": stage,
            "run_id": self.run_id,
            "git_commit": self.git_commit,
            "run_config_sha256": _sha(self.run_root / "config.json"),
            "previous_marker_sha256": _sha(previous),
            "files": files,
            "metadata": metadata,
        }
        _write_json(path, {**unsigned, "completion_digest": _digest(unsigned)})
        return path

    def _publish_metadata(self) -> None:
        metrics = [evaluate_translation_sample(sample) for sample in self.samples]
        report_payload = build_stage_a_prime_report(
            metrics,
            cohort=self.samples,
            run_id=self.run_id,
            git_commit=self.git_commit,
            physical_leakage_clean=True,
        )
        write_stage_a_prime_report(self.run_root, report_payload)

        long_records = []
        cohort_records = []
        for sample in self.samples:
            long_records.append(
                {
                    "sample_id": sample.sample_id,
                    "scene": sample.scene,
                    "role": sample.role,
                    "path": sample.long_path.relative_to(self.run_root).as_posix(),
                    "sha256": sample.long_sha256,
                    "source_sha256": str(
                        np.load(sample.long_path, allow_pickle=False)["source_sha256"]
                    ),
                }
            )
            cohort_records.append(
                {
                    "sample_id": sample.sample_id,
                    "scene": sample.scene,
                    "role": sample.role,
                    "long_path": sample.long_path.relative_to(self.run_root).as_posix(),
                    "short_path": sample.short_path.relative_to(self.run_root).as_posix(),
                    "quality_path": sample.quality_path.relative_to(self.run_root).as_posix(),
                    "target_path": sample.target_path.relative_to(self.run_root).as_posix(),
                    "long_sha256": sample.long_sha256,
                    "short_sha256": sample.short_sha256,
                    "quality_sha256": sample.quality_sha256,
                    "target_sha256": sample.target_sha256,
                }
            )
        long_manifest = {
            "schema": "camera_translation_hvrfm.long_context_manifest.v1",
            "run_id": self.run_id,
            "git_commit": self.git_commit,
            "records": long_records,
        }
        cohort_manifest = {
            "schema": "camera_translation_hvrfm.cohort_manifest.v1",
            "run_id": self.run_id,
            "git_commit": self.git_commit,
            "records": cohort_records,
        }
        long_path = self.run_root / "manifests/long_context.json"
        cohort_path = self.run_root / "manifests/cohort.json"
        _write_json(long_path, long_manifest)
        _write_json(cohort_path, cohort_manifest)

        scene_bindings = []
        for sample in self.samples:
            with np.load(sample.quality_path, allow_pickle=False) as quality:
                scene_bindings.append(
                    {
                        "scene": sample.scene,
                        "role": sample.role,
                        "source_sha256": str(quality["source_sha256"]),
                        "long_context_sha256": self.reference_long_sha256[
                            sample.scene
                        ],
                        "teacher_reference_sha256": str(
                            quality["teacher_reference_sha256"]
                        ),
                        "formal_label_sha256": str(quality["formal_label_sha256"]),
                    }
                )
        preflight_unsigned = {
            "schema": "camera_translation_hvrfm.preflight_evidence.v1",
            "stage": "preflight",
            "run_id": self.run_id,
            "git_commit": self.git_commit,
            "source_run": str(self.source_root.resolve()),
            "source_completion_sha256": self.source_completion_sha256,
            "source_manifest_sha256": self.source_manifest_sha256,
            "reference_run": str(self.reference_root.resolve()),
            "reference_completion_sha256": self.reference_completion_sha256,
            "reference_inventory_sha256": self.reference_inventory_sha256,
            "reference_config_sha256": self.reference_config_sha256,
            "reference_report_json_sha256": self.reference_report_json_sha256,
            "reference_report_markdown_sha256": self.reference_report_markdown_sha256,
            "reference_long_manifest_sha256": self.reference_long_manifest_sha256,
            "reference_teacher_manifest_sha256": self.reference_teacher_manifest_sha256,
            "formal_run": str(self.formal_root.resolve()),
            "formal_completion_sha256": self.formal_completion_sha256,
            "formal_data_manifest_sha256": self.formal_manifest_sha256,
            "checkpoint_file": str(self.checkpoint_file.resolve()),
            "checkpoint_sha256": self.checkpoint_sha256,
            "scene_bindings": scene_bindings,
        }
        preflight_path = self.run_root / "manifests/preflight_evidence.json"
        _write_json(
            preflight_path,
            {
                **preflight_unsigned,
                "completion_digest": _digest(preflight_unsigned),
            },
        )
        config = {
            "schema": "camera_translation_hvrfm.run_config.v1",
            "run_id": self.run_id,
            "git_commit": self.git_commit,
            "source_run": str(self.source_root.resolve()),
            "source_completion_sha256": self.source_completion_sha256,
            "source_manifest_sha256": self.source_manifest_sha256,
            "reference_run": str(self.reference_root.resolve()),
            "reference_completion_sha256": self.reference_completion_sha256,
            "reference_inventory_sha256": self.reference_inventory_sha256,
            "reference_long_manifest_sha256": self.reference_long_manifest_sha256,
            "reference_teacher_manifest_sha256": self.reference_teacher_manifest_sha256,
            "formal_run": str(self.formal_root.resolve()),
            "formal_completion_sha256": self.formal_completion_sha256,
            "formal_data_manifest_sha256": self.formal_manifest_sha256,
            "checkpoint_file": str(self.checkpoint_file.resolve()),
            "checkpoint_sha256": self.checkpoint_sha256,
            "preflight_evidence_sha256": _sha(preflight_path),
            "long_context_manifest_sha256": _sha(long_path),
            "cohort_manifest_sha256": _sha(cohort_path),
            "scene_count": 10,
            "endpoint_count": 40,
            "smoke_scene": "scene0029_01",
        }
        config_path = self.run_root / "config.json"
        _write_json(config_path, config)

        prepare_files = {
            "config.json": _sha(config_path),
            "manifests/long_context.json": _sha(long_path),
            "manifests/cohort.json": _sha(cohort_path),
        }
        for row in cohort_records:
            for prefix in ("long", "short", "quality", "target"):
                prepare_files[row[f"{prefix}_path"]] = row[f"{prefix}_sha256"]
        prepare = self._stage_marker(
            self.run_root / "prepare/completed.json",
            schema="camera_translation_hvrfm.prepare_completion.v1",
            stage="prepare",
            previous=preflight_path,
            files=prepare_files,
            metadata={
                "scene_count": 10,
                "endpoint_count": 40,
                "smoke_scene": "scene0029_01",
            },
        )
        smoke = self._stage_marker(
            self.run_root / "smoke/completed.json",
            schema="camera_translation_hvrfm.smoke_completion.v1",
            stage="smoke",
            previous=prepare,
            files={},
            metadata={
                "scene": "scene0029_01",
                "endpoint_count": 4,
                "classification": "TRANSLATION_ENDPOINTS_READY",
            },
        )
        self._stage_marker(
            self.run_root / "calibration/completed.json",
            schema="camera_translation_hvrfm.calibration_completion.v1",
            stage="calibration",
            previous=smoke,
            files={},
            metadata={
                "scene_count": 10,
                "endpoint_count": 40,
                "classification": "TRANSLATION_ENDPOINTS_READY",
            },
        )

    def _bundle_paths(self, scene: str) -> dict[str, Path]:
        return {
            "long": self.run_root / f"prediction_only/long_context/{scene}.npz",
            "short": self.run_root / f"privileged_training/short_context/{scene}.npz",
            "quality": self.run_root / f"privileged_labels/quality/{scene}.npz",
            "target": self.run_root
            / f"privileged_labels/translation_targets/{scene}.npz",
        }

    def mutate_bundle(self, scene: str, kind: str, mutate) -> None:
        paths = self._bundle_paths(scene)
        _rewrite_npz(paths[kind], mutate)
        if kind == "long":
            long_sha256 = _sha(paths["long"])
            _rewrite_npz(
                paths["short"],
                lambda arrays: arrays.__setitem__(
                    "long_context_sha256",
                    np.asarray(long_sha256, dtype="U64"),
                ),
            )
        if kind in {"long", "short", "quality"}:
            _rewrite_npz(
                paths["target"],
                lambda arrays: arrays.update(
                    {
                        "long_context_sha256": np.asarray(
                            _sha(paths["long"]), dtype="U64"
                        ),
                        "short_context_sha256": np.asarray(
                            _sha(paths["short"]), dtype="U64"
                        ),
                        "quality_sha256": np.asarray(
                            _sha(paths["quality"]), dtype="U64"
                        ),
                    }
                ),
            )
        self.resign_all()

    def resign_all(self) -> None:
        long_path = self.run_root / "manifests/long_context.json"
        cohort_path = self.run_root / "manifests/cohort.json"
        long_manifest = json.loads(long_path.read_text(encoding="utf-8"))
        cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
        by_scene = {row["scene"]: row for row in cohort["records"]}
        for row in long_manifest["records"]:
            paths = self._bundle_paths(row["scene"])
            row["sha256"] = _sha(paths["long"])
            with np.load(paths["long"], allow_pickle=False) as archive:
                row["source_sha256"] = str(archive["source_sha256"])
        for scene, row in by_scene.items():
            paths = self._bundle_paths(scene)
            for kind in ("long", "short", "quality", "target"):
                row[f"{kind}_sha256"] = _sha(paths[kind])
        _write_json(long_path, long_manifest)
        _write_json(cohort_path, cohort)

        report_path = self.run_root / "reports/stage_a_prime.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report_cohort = {row["scene"]: row for row in report["cohort"]}
        report_metrics = {row["scene"]: row for row in report["scene_metrics"]}
        for scene, row in by_scene.items():
            for kind in ("long", "short", "quality", "target"):
                report_cohort[scene][f"{kind}_sha256"] = row[f"{kind}_sha256"]
                report_metrics[scene]["provenance"][f"{kind}_sha256"] = row[
                    f"{kind}_sha256"
                ]
        _write_json(report_path, report)
        markdown_path = self.run_root / "reports/stage_a_prime.md"
        markdown_path.write_bytes(_markdown_bytes(report))
        completion_unsigned = {
            "schema": "camera_translation_hvrfm.stage_a_prime_completion.v1",
            "run_id": self.run_id,
            "git_commit": self.git_commit,
            "classification": report["classification"],
            "scene_count": report["scene_count"],
            "endpoint_count": report["endpoint_count"],
            "report_json_path": "reports/stage_a_prime.json",
            "report_json_sha256": _sha(report_path),
            "report_markdown_path": "reports/stage_a_prime.md",
            "report_markdown_sha256": _sha(markdown_path),
        }
        _write_json(
            self.run_root / "reports/completed.json",
            {
                **completion_unsigned,
                "completion_digest": _digest(completion_unsigned),
            },
        )

        config_path = self.run_root / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["preflight_evidence_sha256"] = _sha(
            self.run_root / "manifests/preflight_evidence.json"
        )
        config["long_context_manifest_sha256"] = _sha(long_path)
        config["cohort_manifest_sha256"] = _sha(cohort_path)
        _write_json(config_path, config)
        prepare_files = {
            "config.json": _sha(config_path),
            "manifests/long_context.json": _sha(long_path),
            "manifests/cohort.json": _sha(cohort_path),
        }
        for row in cohort["records"]:
            for kind in ("long", "short", "quality", "target"):
                prepare_files[row[f"{kind}_path"]] = row[f"{kind}_sha256"]
        prepare = self._stage_marker(
            self.run_root / "prepare/completed.json",
            schema="camera_translation_hvrfm.prepare_completion.v1",
            stage="prepare",
            previous=self.run_root / "manifests/preflight_evidence.json",
            files=prepare_files,
            metadata={
                "scene_count": 10,
                "endpoint_count": 40,
                "smoke_scene": "scene0029_01",
            },
        )
        smoke = self._stage_marker(
            self.run_root / "smoke/completed.json",
            schema="camera_translation_hvrfm.smoke_completion.v1",
            stage="smoke",
            previous=prepare,
            files={},
            metadata={
                "scene": "scene0029_01",
                "endpoint_count": 4,
                "classification": "TRANSLATION_ENDPOINTS_READY",
            },
        )
        self._stage_marker(
            self.run_root / "calibration/completed.json",
            schema="camera_translation_hvrfm.calibration_completion.v1",
            stage="calibration",
            previous=smoke,
            files={},
            metadata={
                "scene_count": 10,
                "endpoint_count": 40,
                "classification": "TRANSLATION_ENDPOINTS_READY",
            },
        )

    def tamper_report_metric_and_resign(self) -> None:
        report_path = self.run_root / "reports/stage_a_prime.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        first = report["scene_metrics"][0]
        first["endpoints"][0]["full_scene_utility"] += 0.01
        first["mean_full_scene_utility"] = sum(
            endpoint["full_scene_utility"] for endpoint in first["endpoints"]
        ) / 4.0
        scene_utilities = [
            row["mean_full_scene_utility"] for row in report["scene_metrics"]
        ]
        report["mean_full_scene_utility"] = sum(scene_utilities) / 10.0
        report["minimum_full_scene_utility"] = min(scene_utilities)
        report["positive_scene_count"] = sum(value > 0.0 for value in scene_utilities)
        _write_json(report_path, report)
        (self.run_root / "reports/stage_a_prime.md").write_bytes(
            _markdown_bytes(report)
        )
        self.resign_all()


def _npz_with_member(member: str) -> bytes:
    payload = BytesIO()
    npy = BytesIO()
    np.save(npy, np.arange(3, dtype=np.float32), allow_pickle=False)
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr(member, npy.getvalue())
    return payload.getvalue()


class IndependentVerifierContractTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(verify, "Task 4b independent verifier module is missing")
        return verify

    def assert_rejected_without_seal(
        self,
        module,
        fixture: ValidRunFixture,
        *,
        pattern: str,
    ) -> None:
        with fixture.patched_frozen(module):
            with mock.patch.object(
                module,
                "_load_camera_head",
                return_value=(
                    TokenCameraHead(),
                    fixture.checkpoint_sha256,
                    torch.device("cpu"),
                ),
            ):
                with self.assertRaisesRegex(ValueError, pattern):
                    module.verify_completed_run(
                        fixture.run_root,
                        expected_run_id=fixture.run_id,
                        expected_git_commit=fixture.git_commit,
                        checkpoint_dir=fixture.checkpoint_dir,
                    )
        self.assertFalse((fixture.run_root / "verified_completion.json").exists())
        self.assertFalse(
            (fixture.run_root / "manifests/verification_inventory.json").exists()
        )

    def test_public_api_and_frozen_output_schemas(self) -> None:
        module = self.api()
        self.assertEqual(
            str(inspect.signature(module.verify_completed_run)),
            "(run_root: 'Path', *, expected_run_id: 'str', "
            "expected_git_commit: 'str', checkpoint_dir: 'Path') -> 'Path'",
        )
        self.assertEqual(
            module.INVENTORY_SCHEMA,
            "camera_translation_hvrfm.verification_inventory.v1",
        )
        self.assertEqual(
            module.VERIFIED_SCHEMA,
            "camera_translation_hvrfm.verified_completion.v1",
        )

    def test_ast_forbids_production_publish_evaluate_classify_builders_and_old_verifier(self) -> None:
        module = self.api()
        tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
        forbidden_symbols = {
            "publish_translation_sample",
            "evaluate_translation_sample",
            "classify_stage_a_prime",
            "build_stage_a_prime_report",
            "build_raw_gauge_teacher",
            "fit_frozen_oracle",
            "verify_completed_run",
        }
        forbidden_modules = {
            "pre_experiments.camera_translation_hvrfm.pipeline",
            "pre_experiments.camera_translation_hvrfm.evaluate",
            "pre_experiments.camera_translation_hvrfm.report",
            "pre_experiments.camera_translation_hvrfm.teacher",
            "pre_experiments.conditional_hierarchical_vrfm.pipeline",
        }
        imports: list[tuple[str, str | None]] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    imports.append((alias.name, node.module))
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append((alias.name, alias.name))
        self.assertFalse(
            {name for name, _ in imports} & forbidden_symbols,
            imports,
        )
        self.assertFalse(
            {source for _, source in imports if source is not None}
            & forbidden_modules,
            imports,
        )
        calls = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        self.assertFalse(calls & forbidden_symbols, calls)

    def test_missing_run_fails_without_creating_any_file(self) -> None:
        module = self.api()
        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory) / "missing"
            with self.assertRaises(ValueError):
                module.verify_completed_run(
                    run_root,
                    expected_run_id="run-20260829",
                    expected_git_commit="1" * 40,
                    checkpoint_dir=Path(directory) / "checkpoint",
                )
            self.assertFalse(run_root.exists())

    def test_json_snapshot_requires_canonical_bytes_exact_fields_and_no_duplicates(self) -> None:
        module = self.api()
        canonical = b'{\n  "alpha": 1,\n  "beta": "two"\n}\n'
        self.assertEqual(
            module._decode_json_snapshot(
                canonical,
                label="fixture",
                expected_fields=frozenset({"alpha", "beta"}),
            ),
            {"alpha": 1, "beta": "two"},
        )
        malformed = (
            b'{"alpha":1,"beta":"two"}\n',
            b'{"alpha":1,"alpha":1,"beta":"two"}\n',
            b'{"alpha":NaN,"beta":"two"}\n',
            b'{"alpha":1,"beta":"two","extra":3}\n',
        )
        for payload in malformed:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                module._decode_json_snapshot(
                    payload,
                    label="fixture",
                    expected_fields=frozenset({"alpha", "beta"}),
                )

    def test_npz_snapshot_rejects_duplicate_unsafe_extra_and_object_members(self) -> None:
        module = self.api()
        buffer = BytesIO()
        np.savez_compressed(
            buffer,
            value=np.arange(3, dtype=np.float32),
            label=np.asarray("ok", dtype="U8"),
        )
        arrays = module._decode_npz_snapshot(
            buffer.getvalue(),
            label="fixture",
            expected_members=frozenset({"value", "label"}),
        )
        np.testing.assert_array_equal(arrays["value"], np.arange(3, dtype=np.float32))

        object_buffer = BytesIO()
        np.savez_compressed(
            object_buffer,
            value=np.asarray([object()], dtype=object),
            label=np.asarray("ok", dtype="U8"),
        )
        with self.assertRaisesRegex(ValueError, "object"):
            module._decode_npz_snapshot(
                object_buffer.getvalue(),
                label="fixture",
                expected_members=frozenset({"value", "label"}),
            )

        duplicate = BytesIO()
        npy = BytesIO()
        np.save(npy, np.arange(3, dtype=np.float32), allow_pickle=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with zipfile.ZipFile(duplicate, "w") as archive:
                archive.writestr("value.npy", npy.getvalue())
                archive.writestr("value.npy", npy.getvalue())
                archive.writestr("label.npy", npy.getvalue())
        for label, payload in (
            ("duplicate", duplicate.getvalue()),
            ("unsafe", _npz_with_member("../value.npy")),
            ("extra", _npz_with_member("extra.npy")),
        ):
            with self.subTest(label=label), self.assertRaises(ValueError):
                module._decode_npz_snapshot(
                    payload,
                    label="fixture",
                    expected_members=frozenset({"value", "label"}),
                )

    def test_fusion_skips_only_zero_weight_unselected_invalid_window(self) -> None:
        module = self.api()
        baseline = np.broadcast_to(
            np.eye(4, dtype=np.float64), (500, 4, 4)
        ).copy()
        phase = np.linspace(0.0, 4.0 * np.pi, 500, dtype=np.float64)
        baseline[:, 0, 3] = np.linspace(0.0, 10.0, 500, dtype=np.float64)
        baseline[:, 1, 3] = np.sin(phase)
        baseline[:, 2, 3] = np.cos(phase)
        short = np.stack(
            [baseline[start : start + 100].copy() for start in range(0, 401, 50)]
        )
        short[0, :, :3, 3] = 0.0
        weights = np.ones(9, dtype=np.float64)
        weights[0] = 0.0
        masks = module._canonical_window_masks("scene0000_00", weights)
        self.assertFalse(np.any(masks[:, 0]))
        coverage, *_ = module._fuse_teacher(
            baseline, short, weights=weights, masks=masks
        )
        self.assertEqual(coverage.shape, (4, 500))

        selected = masks.copy()
        selected[0, 0] = 1
        with self.assertRaisesRegex(ValueError, "rank|condition|selected"):
            module._fuse_teacher(
                baseline, short, weights=weights, masks=selected
            )
        positive = weights.copy()
        positive[0] = 1.0
        with self.assertRaisesRegex(ValueError, "rank|condition|selected"):
            module._fuse_teacher(
                baseline, short, weights=positive, masks=masks
            )

    def test_live_frozen_source_reference_and_formal_tamper_are_rejected(self) -> None:
        module = self.api()
        attacks = (
            (
                "source",
                lambda fixture: fixture.source_paths["scene0000_00"],
            ),
            (
                "reference",
                lambda fixture: fixture.reference_long_paths["scene0000_00"],
            ),
            (
                "reference_teacher",
                lambda fixture: fixture.reference_teacher_paths["scene0000_00"],
            ),
            (
                "formal",
                lambda fixture: fixture.formal_paths["scene0000_00"],
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            fixture = ValidRunFixture(Path(directory))
            for label, locate in attacks:
                with self.subTest(label=label):
                    path = locate(fixture)
                    original = path.read_bytes()
                    path.write_bytes(original + b"tamper\n")
                    try:
                        self.assert_rejected_without_seal(
                            module,
                            fixture,
                            pattern="upstream|source|reference|formal|digest",
                        )
                    finally:
                        path.write_bytes(original)
                        (fixture.run_root / "verified_completion.json").unlink(
                            missing_ok=True
                        )
                        (
                            fixture.run_root
                            / "manifests/verification_inventory.json"
                        ).unlink(missing_ok=True)

            preflight_path = fixture.run_root / "manifests/preflight_evidence.json"
            preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
            preflight["scene_bindings"][0]["long_context_sha256"] = "f" * 64
            unsigned = dict(preflight)
            unsigned.pop("completion_digest")
            preflight["completion_digest"] = _digest(unsigned)
            _write_json(preflight_path, preflight)
            fixture.resign_all()
            self.assert_rejected_without_seal(
                module,
                fixture,
                pattern="upstream|reference|binding|digest",
            )

    def test_late_ordinary_file_is_detected_before_final_seal(self) -> None:
        module = self.api()
        with tempfile.TemporaryDirectory() as directory:
            fixture = ValidRunFixture(Path(directory))
            original = module._inventory_bytes

            def inject_late_file(*args, **kwargs):
                result = original(*args, **kwargs)
                (fixture.run_root / "prediction_only/late.json").write_text(
                    "{}\n", encoding="utf-8"
                )
                return result

            with fixture.patched_frozen(module):
                with mock.patch.object(
                    module,
                    "_load_camera_head",
                    return_value=(
                        TokenCameraHead(),
                        fixture.checkpoint_sha256,
                        torch.device("cpu"),
                    ),
                ), mock.patch.object(
                    module, "_inventory_bytes", side_effect=inject_late_file
                ):
                    with self.assertRaisesRegex(ValueError, "inventory|topology|late"):
                        module.verify_completed_run(
                            fixture.run_root,
                            expected_run_id=fixture.run_id,
                            expected_git_commit=fixture.git_commit,
                            checkpoint_dir=fixture.checkpoint_dir,
                        )
            self.assertFalse((fixture.run_root / "verified_completion.json").exists())

    def test_upstream_change_after_authentication_is_rechecked_before_seal(self) -> None:
        module = self.api()
        with tempfile.TemporaryDirectory() as directory:
            fixture = ValidRunFixture(Path(directory))
            source = fixture.source_paths["scene0000_00"]

            def mutate_after_authentication(_private_dir: Path):
                source.write_bytes(source.read_bytes() + b"late upstream tamper\n")
                return (
                    TokenCameraHead(),
                    fixture.checkpoint_sha256,
                    torch.device("cpu"),
                )

            with fixture.patched_frozen(module):
                with mock.patch.object(
                    module,
                    "_load_camera_head",
                    side_effect=mutate_after_authentication,
                ):
                    with self.assertRaisesRegex(
                        ValueError, "upstream authentication|source shard.*changed"
                    ):
                        module.verify_completed_run(
                            fixture.run_root,
                            expected_run_id=fixture.run_id,
                            expected_git_commit=fixture.git_commit,
                            checkpoint_dir=fixture.checkpoint_dir,
                        )
            self.assertFalse((fixture.run_root / "verified_completion.json").exists())
            self.assertFalse(
                (fixture.run_root / "manifests/verification_inventory.json").exists()
            )

    def test_final_marker_is_exactly_rolled_back_on_late_failure(self) -> None:
        module = self.api()
        with tempfile.TemporaryDirectory() as directory:
            fixture = ValidRunFixture(Path(directory))
            original = module._require_snapshot_unchanged
            calls = 0

            def fail_after_final(root: Path, expected):
                nonlocal calls
                calls += 1
                if calls == 3:
                    raise ValueError("injected terminal failure")
                return original(root, expected)

            with fixture.patched_frozen(module):
                with mock.patch.object(
                    module,
                    "_load_camera_head",
                    return_value=(
                        TokenCameraHead(),
                        fixture.checkpoint_sha256,
                        torch.device("cpu"),
                    ),
                ), mock.patch.object(
                    module,
                    "_require_snapshot_unchanged",
                    side_effect=fail_after_final,
                ):
                    with self.assertRaisesRegex(ValueError, "terminal failure"):
                        module.verify_completed_run(
                            fixture.run_root,
                            expected_run_id=fixture.run_id,
                            expected_git_commit=fixture.git_commit,
                            checkpoint_dir=fixture.checkpoint_dir,
                        )
            self.assertFalse((fixture.run_root / "verified_completion.json").exists())

    def test_final_temp_cleanup_failure_runs_terminal_checks_and_hides_final(self) -> None:
        module = self.api()
        with tempfile.TemporaryDirectory() as directory:
            fixture = ValidRunFixture(Path(directory))
            original_snapshot_check = module._require_snapshot_unchanged
            original_unlink = Path.unlink
            terminal_checks = 0

            def count_snapshot_checks(root: Path, expected):
                nonlocal terminal_checks
                terminal_checks += 1
                return original_snapshot_check(root, expected)

            def fail_final_temp_cleanup(path: Path, *args, **kwargs):
                if (
                    path.parent == fixture.run_root
                    and path.name.startswith(".verified_completion.json.")
                    and path.name.endswith(".tmp")
                ):
                    raise OSError("injected final temp cleanup failure")
                return original_unlink(path, *args, **kwargs)

            with fixture.patched_frozen(module):
                with mock.patch.object(
                    module,
                    "_load_camera_head",
                    return_value=(
                        TokenCameraHead(),
                        fixture.checkpoint_sha256,
                        torch.device("cpu"),
                    ),
                ), mock.patch.object(
                    module,
                    "_require_snapshot_unchanged",
                    side_effect=count_snapshot_checks,
                ), mock.patch.object(
                    Path,
                    "unlink",
                    new=fail_final_temp_cleanup,
                ):
                    with self.assertRaisesRegex(ValueError, "temporary|publication"):
                        module.verify_completed_run(
                            fixture.run_root,
                            expected_run_id=fixture.run_id,
                            expected_git_commit=fixture.git_commit,
                            checkpoint_dir=fixture.checkpoint_dir,
                        )
            self.assertGreaterEqual(terminal_checks, 3)
            self.assertFalse((fixture.run_root / "verified_completion.json").exists())

    def test_rollback_does_not_delete_replacement_after_authenticated_read(self) -> None:
        module = self.api()
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "verified_completion.json"
            expected = b"expected final marker\n"
            replacement = b"foreign replacement marker\n"
            target.write_bytes(expected)
            original_read_bytes = Path.read_bytes
            replaced = False

            def replace_after_read(path: Path) -> bytes:
                nonlocal replaced
                payload = original_read_bytes(path)
                if path == target and not replaced:
                    temporary = target.with_name("replacement.tmp")
                    temporary.write_bytes(replacement)
                    temporary.replace(target)
                    replaced = True
                return payload

            with mock.patch.object(Path, "read_bytes", new=replace_after_read):
                with self.assertRaisesRegex(ValueError, "changed|rollback"):
                    module._rollback_exact_new_file(target, expected)
            self.assertTrue(replaced)
            self.assertEqual(target.read_bytes(), replacement)

    def test_changed_final_marker_reports_unsafe_rollback(self) -> None:
        module = self.api()
        with tempfile.TemporaryDirectory() as directory:
            fixture = ValidRunFixture(Path(directory))
            original = module._require_snapshot_unchanged
            calls = 0

            def replace_after_final(root: Path, expected):
                nonlocal calls
                calls += 1
                if calls == 3:
                    (fixture.run_root / "verified_completion.json").write_bytes(
                        b"foreign marker\n"
                    )
                    raise ValueError("injected terminal failure")
                return original(root, expected)

            with fixture.patched_frozen(module):
                with mock.patch.object(
                    module,
                    "_load_camera_head",
                    return_value=(
                        TokenCameraHead(),
                        fixture.checkpoint_sha256,
                        torch.device("cpu"),
                    ),
                ), mock.patch.object(
                    module,
                    "_require_snapshot_unchanged",
                    side_effect=replace_after_final,
                ):
                    with self.assertRaisesRegex(
                        ValueError, "rollback.*changed|changed.*rollback"
                    ):
                        module.verify_completed_run(
                            fixture.run_root,
                            expected_run_id=fixture.run_id,
                            expected_git_commit=fixture.git_commit,
                            checkpoint_dir=fixture.checkpoint_dir,
                        )
            self.assertEqual(
                (fixture.run_root / "verified_completion.json").read_bytes(),
                b"foreign marker\n",
            )

    def test_valid_run_is_numerically_replayed_inventoried_and_sealed_idempotently(self) -> None:
        module = self.api()
        with tempfile.TemporaryDirectory() as directory:
            fixture = ValidRunFixture(Path(directory))
            private_checkpoint_dirs: list[Path] = []

            def load_private(checkpoint_dir: Path):
                private = Path(checkpoint_dir)
                private_checkpoint_dirs.append(private)
                self.assertNotEqual(private.resolve(), fixture.checkpoint_dir.resolve())
                copied = private / fixture.checkpoint_file.name
                self.assertEqual(copied.read_bytes(), fixture.checkpoint_file.read_bytes())
                return (
                    TokenCameraHead(),
                    fixture.checkpoint_sha256,
                    torch.device("cpu"),
                )

            with fixture.patched_frozen(module):
                with mock.patch.object(
                    module,
                    "_load_camera_head",
                    side_effect=load_private,
                ) as load:
                    marker = module.verify_completed_run(
                        fixture.run_root,
                        expected_run_id=fixture.run_id,
                        expected_git_commit=fixture.git_commit,
                        checkpoint_dir=fixture.checkpoint_dir,
                    )
                    first = marker.read_bytes()
                    inventory_first = (
                        fixture.run_root / "manifests/verification_inventory.json"
                    ).read_bytes()
                    self.assertEqual(
                        module.verify_completed_run(
                            fixture.run_root,
                            expected_run_id=fixture.run_id,
                            expected_git_commit=fixture.git_commit,
                            checkpoint_dir=fixture.checkpoint_dir,
                        ),
                        marker,
                    )
            self.assertEqual(load.call_count, 2)
            self.assertEqual(len(private_checkpoint_dirs), 2)
            self.assertTrue(all(not path.exists() for path in private_checkpoint_dirs))
            self.assertEqual(marker.read_bytes(), first)
            self.assertEqual(
                (fixture.run_root / "manifests/verification_inventory.json").read_bytes(),
                inventory_first,
            )

            inventory = json.loads(inventory_first)
            self.assertEqual(
                set(inventory),
                {
                    "schema",
                    "run_id",
                    "git_commit",
                    "classification",
                    "report_completion_sha256",
                    "calibration_completion_sha256",
                    "files",
                    "file_count",
                    "total_bytes",
                    "completion_digest",
                },
            )
            self.assertEqual(
                inventory["schema"],
                "camera_translation_hvrfm.verification_inventory.v1",
            )
            self.assertEqual(inventory["classification"], "TRANSLATION_ENDPOINTS_READY")
            self.assertEqual(inventory["file_count"], 50)
            self.assertEqual(len(inventory["files"]), 50)
            unsigned_inventory = dict(inventory)
            inventory_digest = unsigned_inventory.pop("completion_digest")
            self.assertEqual(inventory_digest, _digest(unsigned_inventory))

            completion = json.loads(first)
            self.assertEqual(
                set(completion),
                {
                    "schema",
                    "run_id",
                    "git_commit",
                    "classification",
                    "inventory_path",
                    "inventory_sha256",
                    "report_completion_sha256",
                    "file_count",
                    "total_bytes",
                    "completion_digest",
                },
            )
            self.assertEqual(
                completion["schema"],
                "camera_translation_hvrfm.verified_completion.v1",
            )
            self.assertEqual(completion["classification"], "TRANSLATION_ENDPOINTS_READY")
            self.assertEqual(
                completion["inventory_sha256"],
                hashlib.sha256(inventory_first).hexdigest(),
            )
            unsigned_completion = dict(completion)
            completion_digest = unsigned_completion.pop("completion_digest")
            self.assertEqual(completion_digest, _digest(unsigned_completion))

    def test_fully_resigned_long_token_tamper_fails_independent_camera_replay(self) -> None:
        module = self.api()
        with tempfile.TemporaryDirectory() as directory:
            fixture = ValidRunFixture(Path(directory))

            def mutate(arrays: dict[str, np.ndarray]) -> None:
                arrays["camera_tokens"][0, 0] += np.float32(0.25)

            fixture.mutate_bundle("scene0000_00", "long", mutate)
            self.assert_rejected_without_seal(
                module,
                fixture,
                pattern="Camera Head pose encoding|baseline",
            )

    def test_fully_resigned_quality_weights_fail_independent_fusion_replay(self) -> None:
        module = self.api()
        with tempfile.TemporaryDirectory() as directory:
            fixture = ValidRunFixture(Path(directory))

            def mutate(arrays: dict[str, np.ndarray]) -> None:
                arrays["window_weights"][0] *= 0.8
                coverage = np.zeros((4, 500), dtype=np.float64)
                for endpoint in range(4):
                    for window in range(9):
                        if arrays["window_masks"][endpoint, window]:
                            start = window * 50
                            coverage[endpoint, start : start + 100] += arrays[
                                "window_weights"
                            ][window]
                arrays["coverage_weights"] = coverage

            fixture.mutate_bundle("scene0000_00", "quality", mutate)
            self.assert_rejected_without_seal(
                module,
                fixture,
                pattern="fusion|teacher centers|diagnostic",
            )

    def test_frozen_numeric_payload_rejects_co_resigned_gt_and_oracle_transform(self) -> None:
        module = self.api()
        with tempfile.TemporaryDirectory() as directory:
            fixture = ValidRunFixture(Path(directory))

            def mutate(arrays: dict[str, np.ndarray]) -> None:
                rotation = np.asarray(
                    [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                    dtype=np.float64,
                )
                arrays["gt_c2w"][:, :3, :3] = np.einsum(
                    "ij,fjk->fik", rotation, arrays["gt_c2w"][:, :3, :3]
                )
                arrays["gt_c2w"][:, :3, 3] = (
                    arrays["gt_c2w"][:, :3, 3] @ rotation.T
                )
                arrays["oracle_rotation"] = (
                    rotation @ arrays["oracle_rotation"]
                )
                arrays["oracle_translation"] = (
                    arrays["oracle_translation"] @ rotation.T
                )
                arrays["gt_scene_scale"] = np.asarray(
                    prediction_scale(arrays["gt_c2w"]), dtype=np.float64
                )
                oracle_payload = {
                    "scene": str(arrays["oracle_scene"]),
                    "frame_digest": str(arrays["oracle_frame_digest"]),
                    "fit_count": int(arrays["oracle_fit_count"]),
                    "scale": float(arrays["oracle_scale"]),
                    "rotation": tuple(
                        tuple(float(value) for value in row)
                        for row in arrays["oracle_rotation"]
                    ),
                    "translation": tuple(
                        float(value) for value in arrays["oracle_translation"]
                    ),
                }
                arrays["oracle_digest"] = np.asarray(
                    canonical_json_digest(oracle_payload), dtype="U64"
                )

            fixture.mutate_bundle("scene0000_00", "quality", mutate)
            self.assert_rejected_without_seal(
                module,
                fixture,
                pattern="frozen|upstream|ground truth|oracle|formal|teacher reference",
            )

    def test_fully_resigned_short_token_tamper_fails_independent_fusion_replay(self) -> None:
        module = self.api()
        with tempfile.TemporaryDirectory() as directory:
            fixture = ValidRunFixture(Path(directory))

            def mutate(arrays: dict[str, np.ndarray]) -> None:
                arrays["short_camera_tokens"][0, 0, 0] += np.float32(0.25)

            fixture.mutate_bundle("scene0000_00", "short", mutate)
            self.assert_rejected_without_seal(
                module,
                fixture,
                pattern="fusion teacher centers|endpoint replay",
            )

    def test_fully_resigned_center_and_endpoint_tamper_fails_unique_replay(self) -> None:
        module = self.api()
        with tempfile.TemporaryDirectory() as directory:
            fixture = ValidRunFixture(Path(directory))
            paths = fixture._bundle_paths("scene0000_00")
            with np.load(paths["long"], allow_pickle=False) as archive:
                baseline = archive["baseline_c2w"].copy()
                scale = float(archive["prediction_scale"])

            def mutate(arrays: dict[str, np.ndarray]) -> None:
                endpoint = 0
                frame = 0
                arrays["teacher_centers_raw_filled"][endpoint, frame, 0] += 0.01
                delta = (
                    arrays["teacher_centers_raw_filled"][endpoint, frame]
                    - baseline[frame, :3, 3]
                )
                arrays["translation_endpoints"][endpoint, frame] = (
                    -baseline[frame, :3, :3].T @ delta / scale
                ).astype(np.float32)

            fixture.mutate_bundle("scene0000_00", "target", mutate)
            self.assert_rejected_without_seal(
                module,
                fixture,
                pattern="fusion teacher centers|endpoint replay",
            )

    def test_fully_resigned_report_metric_tamper_fails_independent_metrics(self) -> None:
        module = self.api()
        with tempfile.TemporaryDirectory() as directory:
            fixture = ValidRunFixture(Path(directory))
            fixture.tamper_report_metric_and_resign()
            self.assert_rejected_without_seal(
                module,
                fixture,
                pattern="report numeric mismatch",
            )

    def test_extra_prediction_only_file_fails_physical_inventory_before_model_load(self) -> None:
        module = self.api()
        with tempfile.TemporaryDirectory() as directory:
            fixture = ValidRunFixture(Path(directory))
            (fixture.run_root / "prediction_only/leaked_labels.json").write_text(
                "{}\n", encoding="utf-8"
            )
            with mock.patch.object(module, "_load_camera_head") as load:
                with self.assertRaisesRegex(ValueError, "inventory|directory"):
                    module.verify_completed_run(
                        fixture.run_root,
                        expected_run_id=fixture.run_id,
                        expected_git_commit=fixture.git_commit,
                        checkpoint_dir=fixture.checkpoint_dir,
                    )
            load.assert_not_called()
            self.assertFalse((fixture.run_root / "verified_completion.json").exists())

    def test_checkpoint_swap_restore_after_private_copy_is_detected_before_seal(self) -> None:
        module = self.api()
        with tempfile.TemporaryDirectory() as directory:
            fixture = ValidRunFixture(Path(directory))
            original = fixture.checkpoint_file.read_bytes()

            def swap_restore(private_dir: Path):
                copied = Path(private_dir) / fixture.checkpoint_file.name
                self.assertEqual(copied.read_bytes(), original)
                fixture.checkpoint_file.write_bytes(b"malicious checkpoint bytes\n")
                fixture.checkpoint_file.write_bytes(original)
                return (
                    TokenCameraHead(),
                    fixture.checkpoint_sha256,
                    torch.device("cpu"),
                )

            with fixture.patched_frozen(module):
                with mock.patch.object(
                    module, "_load_camera_head", side_effect=swap_restore
                ):
                    with self.assertRaisesRegex(ValueError, "checkpoint identity changed"):
                        module.verify_completed_run(
                            fixture.run_root,
                            expected_run_id=fixture.run_id,
                            expected_git_commit=fixture.git_commit,
                            checkpoint_dir=fixture.checkpoint_dir,
                        )
            self.assertFalse((fixture.run_root / "verified_completion.json").exists())
            self.assertFalse(
                (fixture.run_root / "manifests/verification_inventory.json").exists()
            )


if __name__ == "__main__":
    unittest.main()
