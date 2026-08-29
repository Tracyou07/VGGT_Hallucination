"""Authenticated, physically separated publication for translation teachers."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import subprocess
from typing import Sequence

import numpy as np
import torch
from torch import nn

from pre_experiments.camera_translation_hvrfm.artifacts import (
    load_bound_bundle,
    save_long_context,
    save_quality_sidecar,
    save_short_context,
    save_translation_target,
)
from pre_experiments.camera_translation_hvrfm.geometry import prediction_scale
from pre_experiments.camera_translation_hvrfm.teacher import (
    RawGaugeTeacherSet,
    _authenticate_file,
    _canonical_digest,
    _load_reference,
    _load_source,
    _oracle_from_reference,
    _reject_symlink_components,
    _scene_from_source,
    _sha256_file,
    build_raw_gauge_teacher,
    load_teacher_controls,
    verify_legacy_teacher_witness,
)
from pre_experiments.camera_velocity_ambiguity_02.frozen_oracle import (
    apply_frozen_oracle,
)
from pre_experiments.variational_camera_latent.contracts import SourceShardRecord


_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_ENDPOINTS = 4
_FRAMES = 500
_CALIBRATION_SCENES = (
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
_VALIDATION_SCENES = frozenset({"scene0325_01", "scene0675_00"})
_SMOKE_SCENE = "scene0029_01"


@dataclass(frozen=True)
class PublishedTranslationSample:
    """Paths and actual digests for one four-artifact sample bundle."""

    sample_id: str
    scene: str
    role: str
    long_path: Path
    short_path: Path
    quality_path: Path
    target_path: Path
    long_sha256: str
    short_sha256: str
    quality_sha256: str
    target_sha256: str

    @classmethod
    def placeholder(cls, scene: str, role: str) -> PublishedTranslationSample:
        """Construct a path-free identity used only for cohort validation."""
        return cls(
            sample_id=f"{scene}:frames_500",
            scene=scene,
            role=role,
            long_path=Path(),
            short_path=Path(),
            quality_path=Path(),
            target_path=Path(),
            long_sha256="0" * 64,
            short_sha256="0" * 64,
            quality_sha256="0" * 64,
            target_sha256="0" * 64,
        )


def calibration_role(scene: str) -> str:
    """Return the frozen eight-train/two-validation role for a calibration scene."""
    if scene not in _CALIBRATION_SCENES:
        raise ValueError("scene is outside the frozen ten-scene calibration cohort")
    return "validation" if scene in _VALIDATION_SCENES else "train"


def validate_calibration_cohort(
    samples: Sequence[PublishedTranslationSample],
) -> None:
    """Validate exact scene identities, unique samples, roles, and the smoke member."""
    if not isinstance(samples, Sequence) or isinstance(samples, (str, bytes)):
        raise ValueError("samples must be a sequence")
    if len(samples) != len(_CALIBRATION_SCENES):
        raise ValueError("calibration cohort must contain exactly ten samples")
    if any(not isinstance(sample, PublishedTranslationSample) for sample in samples):
        raise ValueError("calibration cohort contains a malformed sample")
    by_scene = {sample.scene: sample for sample in samples}
    if len(by_scene) != len(samples) or set(by_scene) != set(_CALIBRATION_SCENES):
        raise ValueError("calibration cohort scene identities must match exactly")
    for scene, sample in by_scene.items():
        if sample.role != calibration_role(scene):
            raise ValueError("calibration cohort train/validation role mismatch")
        if sample.sample_id != f"{scene}:frames_500":
            raise ValueError("calibration cohort sample ID mismatch")
    if _SMOKE_SCENE not in by_scene:
        raise ValueError("calibration cohort must contain the frozen smoke scene")


def _current_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("could not determine current git commit") from error
    commit = result.stdout.strip()
    if _COMMIT_RE.fullmatch(commit) is None:
        raise ValueError("current git commit is malformed")
    return commit


def _validate_git_commit(value: str) -> str:
    if not isinstance(value, str) or _COMMIT_RE.fullmatch(value) is None:
        raise ValueError("git_commit must be a canonical lowercase 40-character commit")
    current = _current_git_commit()
    if value != current:
        raise ValueError("git_commit does not match the current repository HEAD")
    return value


def _validate_source_record(
    record: SourceShardRecord,
    *,
    scene: str,
    role: str,
    actual_source_sha256: str,
) -> None:
    if not isinstance(record, SourceShardRecord):
        raise ValueError("source_record must be a SourceShardRecord")
    if record.scene != scene:
        raise ValueError("source record scene mismatch")
    if record.role != role:
        raise ValueError("source record role mismatch")
    if record.overlap_count != 8:
        raise ValueError("source record must bind exactly eight overlaps")
    _canonical_digest(record.sha256, name="source record digest")
    if record.sha256 != actual_source_sha256:
        raise ValueError("source record digest does not match actual source digest")
    # record.path is historical provenance and is deliberately never dereferenced.


def _so3_error_deg(candidate: np.ndarray, ground_truth: np.ndarray) -> np.ndarray:
    relative = np.einsum(
        "...ij,...kj->...ik",
        candidate[..., :3, :3],
        ground_truth[..., :3, :3],
    )
    cosine = np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0)
    return np.rad2deg(np.arccos(cosine)).astype(np.float64, copy=False)


def _hybrid_in_oracle_gauge(
    teacher: RawGaugeTeacherSet, endpoint: int, covered: np.ndarray, oracle
) -> np.ndarray:
    raw = teacher.baseline_c2w[covered].copy()
    raw[:, :3, 3] = teacher.raw_teacher_centers[endpoint, covered]
    return apply_frozen_oracle(oracle, raw)


def _quality_arrays(
    *,
    sample_id: str,
    scene: str,
    frame_ids: np.ndarray,
    teacher: RawGaugeTeacherSet,
    reference: dict[str, np.ndarray],
    source_sha256: str,
    formal_label_sha256: str,
    teacher_reference_sha256: str,
    checkpoint_sha256: str,
    git_commit: str,
) -> dict[str, np.ndarray]:
    gt = np.asarray(reference["gt_c2w"], dtype=np.float64).copy()
    gt_scale = prediction_scale(gt)
    recorded_gt_scale = float(reference["gt_scene_scale"])
    if not np.isclose(recorded_gt_scale, gt_scale, atol=1e-12, rtol=1e-12):
        raise ValueError("teacher reference GT scene scale does not replay")
    oracle = _oracle_from_reference(reference)
    baseline_aligned = apply_frozen_oracle(oracle, teacher.baseline_c2w)
    baseline_translation = (
        np.linalg.norm(
            baseline_aligned[:, :3, 3] - gt[:, :3, 3], axis=1
        ).astype(np.float64, copy=False)
        / gt_scale
    )
    baseline_rotation = _so3_error_deg(baseline_aligned, gt)

    teacher_translation = np.full((_ENDPOINTS, _FRAMES), np.nan, dtype=np.float64)
    teacher_rotation = np.full((_ENDPOINTS, _FRAMES), np.nan, dtype=np.float64)
    utilities = np.zeros(_ENDPOINTS, dtype=np.float64)
    for endpoint in range(_ENDPOINTS):
        covered = teacher.coverage_mask[endpoint] != 0
        if not np.any(covered):
            continue
        aligned = _hybrid_in_oracle_gauge(teacher, endpoint, covered, oracle)
        teacher_translation[endpoint, covered] = (
            np.linalg.norm(aligned[:, :3, 3] - gt[covered, :3, 3], axis=1)
            / gt_scale
        )
        teacher_rotation[endpoint, covered] = _so3_error_deg(aligned, gt[covered])
        baseline_rms = float(
            np.sqrt(np.mean(baseline_translation[covered] ** 2))
        )
        teacher_rms = float(
            np.sqrt(np.mean(teacher_translation[endpoint, covered] ** 2))
        )
        if not np.isfinite(baseline_rms) or baseline_rms <= 0.0:
            raise ValueError(
                f"endpoint {endpoint} covered baseline RMS must be finite and strictly positive"
            )
        if not np.isfinite(teacher_rms):
            raise ValueError(f"endpoint {endpoint} teacher RMS must be finite")
        utilities[endpoint] = (baseline_rms - teacher_rms) / baseline_rms

    return {
        "sample_id": np.asarray(sample_id, dtype="U96"),
        "scene": np.asarray(scene, dtype="U32"),
        "frame_ids": frame_ids.astype(np.int64, copy=True),
        "teacher_variant_ids": np.arange(_ENDPOINTS, dtype=np.int64),
        "gt_c2w": gt,
        "gt_scene_scale": np.asarray(gt_scale, dtype=np.float64),
        "oracle_scene": np.asarray(str(reference["oracle_scene"]), dtype="U32"),
        "oracle_frame_digest": np.asarray(
            str(reference["oracle_frame_digest"]), dtype="U64"
        ),
        "oracle_fit_count": np.asarray(reference["oracle_fit_count"], dtype=np.int64),
        "oracle_scale": np.asarray(reference["oracle_scale"], dtype=np.float64),
        "oracle_rotation": np.asarray(
            reference["oracle_rotation"], dtype=np.float64
        ).copy(),
        "oracle_translation": np.asarray(
            reference["oracle_translation"], dtype=np.float64
        ).copy(),
        "oracle_rank": np.asarray(reference["oracle_rank"], dtype=np.int64),
        "oracle_condition": np.asarray(
            reference["oracle_condition"], dtype=np.float64
        ),
        "oracle_digest": np.asarray(str(reference["oracle_digest"]), dtype="U64"),
        "window_weights": np.asarray(
            reference["window_weights"], dtype=np.float64
        ).copy(),
        "window_masks": np.asarray(reference["window_masks"], dtype=np.uint8).copy(),
        "coverage_weights": teacher.coverage_weights.copy(),
        "variant_utilities": utilities,
        "baseline_translation_error_normalized": baseline_translation,
        "baseline_rotation_error_deg": baseline_rotation,
        "teacher_translation_error_normalized": teacher_translation,
        "teacher_rotation_error_deg": teacher_rotation,
        "source_sha256": np.asarray(source_sha256, dtype="U64"),
        "formal_label_sha256": np.asarray(formal_label_sha256, dtype="U64"),
        "teacher_reference_sha256": np.asarray(
            teacher_reference_sha256, dtype="U64"
        ),
        "checkpoint_sha256": np.asarray(checkpoint_sha256, dtype="U64"),
        "git_commit": np.asarray(git_commit, dtype="U40"),
    }


def _artifact_paths(root: Path, scene: str) -> dict[str, Path]:
    return {
        "long": root / "prediction_only" / "long_context" / f"{scene}.npz",
        "short": root
        / "privileged_training"
        / "short_context"
        / f"{scene}.npz",
        "quality": root / "privileged_labels" / "quality" / f"{scene}.npz",
        "target": root
        / "privileged_labels"
        / "translation_targets"
        / f"{scene}.npz",
    }


def publish_translation_sample(
    output_root: Path,
    *,
    role: str,
    source_path: Path,
    source_record: SourceShardRecord,
    teacher_reference_path: Path,
    expected_teacher_reference_sha256: str,
    formal_label_path: Path,
    expected_formal_label_sha256: str,
    camera_head: nn.Module,
    checkpoint_sha256: str,
    git_commit: str,
    device: torch.device,
) -> PublishedTranslationSample:
    """Publish one authenticated four-file sample under the frozen physical roots."""
    checkpoint_digest = _canonical_digest(
        checkpoint_sha256, name="checkpoint_sha256"
    )
    reference_digest = _canonical_digest(
        expected_teacher_reference_sha256,
        name="expected teacher-reference digest",
    )
    formal_digest = _canonical_digest(
        expected_formal_label_sha256, name="expected formal-label digest"
    )
    commit = _validate_git_commit(git_commit)

    actual_source_path = _authenticate_file(
        Path(source_path), source_record.sha256, label="source shard"
    )
    source_digest = _sha256_file(actual_source_path)
    source = _load_source(actual_source_path, source_digest)
    scene = _scene_from_source(source)
    if role not in {"train", "validation"}:
        raise ValueError("role must be train or validation")
    expected_role = calibration_role(scene)
    if role != expected_role:
        raise ValueError("requested role disagrees with the frozen calibration split")
    _validate_source_record(
        source_record,
        scene=scene,
        role=role,
        actual_source_sha256=source_digest,
    )
    _authenticate_file(
        Path(formal_label_path), formal_digest, label="formal label"
    )
    _authenticate_file(
        Path(teacher_reference_path), reference_digest, label="teacher reference"
    )
    controls = load_teacher_controls(
        teacher_reference_path,
        expected_sha256=reference_digest,
        expected_source_sha256=source_digest,
        expected_checkpoint_sha256=checkpoint_digest,
        expected_formal_label_sha256=formal_digest,
    )
    reference = _load_reference(Path(teacher_reference_path), reference_digest)

    root = Path(output_root)
    _reject_symlink_components(root)
    paths = _artifact_paths(root, scene)
    for path in paths.values():
        _reject_symlink_components(path)
        if path.exists():
            raise ValueError(f"publication target already exists: {path}")

    teacher = build_raw_gauge_teacher(
        actual_source_path,
        controls,
        camera_head,
        expected_source_sha256=source_digest,
        checkpoint_sha256=checkpoint_digest,
        device=device,
    )
    verify_legacy_teacher_witness(
        teacher,
        teacher_reference_path,
        expected_sha256=reference_digest,
    )

    frame_ids = source["global_frame_ids"].astype(np.int64, copy=True)
    sample_id = f"{scene}:frames_500"
    shared = {
        "sample_id": np.asarray(sample_id, dtype="U96"),
        "scene": np.asarray(scene, dtype="U32"),
        "source_sha256": np.asarray(source_digest, dtype="U64"),
        "checkpoint_sha256": np.asarray(checkpoint_digest, dtype="U64"),
        "git_commit": np.asarray(commit, dtype="U40"),
    }
    long_arrays = {
        **shared,
        "frame_ids": frame_ids,
        "camera_tokens": source["global_camera_tokens"].astype(np.float32, copy=True),
        "baseline_pose_encoding": teacher.baseline_pose_encoding.copy(),
        "baseline_c2w": teacher.baseline_c2w.copy(),
        "prediction_scale": np.asarray(teacher.prediction_scale, dtype=np.float64),
    }
    long_sha256 = save_long_context(paths["long"], long_arrays)

    short_arrays = {
        **shared,
        "short_frame_ids": source["short_frame_ids"].astype(np.int64, copy=True),
        "short_camera_tokens": source["short_camera_tokens"].astype(
            np.float32, copy=True
        ),
        "long_context_sha256": np.asarray(long_sha256, dtype="U64"),
    }
    short_sha256 = save_short_context(paths["short"], short_arrays)

    quality_arrays = _quality_arrays(
        sample_id=sample_id,
        scene=scene,
        frame_ids=frame_ids,
        teacher=teacher,
        reference=reference,
        source_sha256=source_digest,
        formal_label_sha256=formal_digest,
        teacher_reference_sha256=reference_digest,
        checkpoint_sha256=checkpoint_digest,
        git_commit=commit,
    )
    quality_sha256 = save_quality_sidecar(paths["quality"], quality_arrays)

    target_arrays = {
        **shared,
        "frame_ids": frame_ids,
        "teacher_variant_ids": np.arange(_ENDPOINTS, dtype=np.int64),
        "coverage_mask": teacher.coverage_mask.copy(),
        "translation_endpoints": teacher.translation_endpoints.copy(),
        "teacher_centers_raw_filled": teacher.filled_teacher_centers.copy(),
        "prediction_scale": np.asarray(teacher.prediction_scale, dtype=np.float64),
        "long_context_sha256": np.asarray(long_sha256, dtype="U64"),
        "short_context_sha256": np.asarray(short_sha256, dtype="U64"),
        "quality_sha256": np.asarray(quality_sha256, dtype="U64"),
        "teacher_reference_sha256": np.asarray(reference_digest, dtype="U64"),
    }
    target_sha256 = save_translation_target(paths["target"], target_arrays)
    load_bound_bundle(
        paths["long"], paths["short"], paths["target"], paths["quality"]
    )
    return PublishedTranslationSample(
        sample_id=sample_id,
        scene=scene,
        role=role,
        long_path=paths["long"],
        short_path=paths["short"],
        quality_path=paths["quality"],
        target_path=paths["target"],
        long_sha256=long_sha256,
        short_sha256=short_sha256,
        quality_sha256=quality_sha256,
        target_sha256=target_sha256,
    )


__all__ = [
    "PublishedTranslationSample",
    "calibration_role",
    "publish_translation_sample",
    "validate_calibration_cohort",
]
