from __future__ import annotations

import hashlib
from pathlib import Path
import re
from typing import Mapping

import numpy as np

from pre_experiments.camera_velocity_ambiguity_02.contracts import canonical_json_digest


LATENT_TARGET_MEMBERS = {
    "scene", "frame_ids", "teacher_variant_ids", "teacher_window_masks",
    "coverage_masks", "residual_coefficients", "decoded_c2w_raw",
    "optimization_steps", "initial_losses", "final_losses", "basis_sha256",
    "source_sha256", "teacher_sha256", "checkpoint_sha256", "git_commit",
}

TEACHER_ARTIFACT_MEMBERS = {
    "scene", "frame_ids", "gt_c2w", "gt_scene_scale", "baseline_c2w_raw",
    "oracle_scene", "oracle_frame_digest", "oracle_fit_count", "oracle_scale",
    "oracle_rotation", "oracle_translation", "oracle_rank", "oracle_condition",
    "oracle_digest", "window_weights", "window_masks", "coverage_weights",
    "fused_c2w", "variant_utilities", "source_sha256", "formal_label_sha256",
    "checkpoint_sha256", "git_commit",
}

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_EXPECTED_SHAPES = {
    "scene": (),
    "frame_ids": (500,),
    "teacher_variant_ids": (4,),
    "teacher_window_masks": (4, 9),
    "coverage_masks": (4, 500),
    "residual_coefficients": (4, 32, 2048),
    "decoded_c2w_raw": (4, 500, 4, 4),
    "optimization_steps": (4,),
    "initial_losses": (4,),
    "final_losses": (4,),
    "basis_sha256": (),
    "source_sha256": (),
    "teacher_sha256": (),
    "checkpoint_sha256": (),
    "git_commit": (),
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _unicode_scalar(array: np.ndarray, name: str) -> str:
    if array.shape != () or array.dtype.kind != "U":
        raise ValueError(f"{name} must be a Unicode scalar")
    return str(array)


def _validate_latent_targets(arrays: Mapping[str, np.ndarray]) -> None:
    if set(arrays) != LATENT_TARGET_MEMBERS:
        raise ValueError("latent targets must use the exact schema")
    if any(value.dtype.hasobject for value in arrays.values()):
        raise ValueError("latent targets may not use object dtypes")
    for name, shape in _EXPECTED_SHAPES.items():
        if arrays[name].shape != shape:
            raise ValueError(f"latent target {name} has invalid shape")

    for name in ("residual_coefficients", "decoded_c2w_raw", "initial_losses", "final_losses"):
        value = arrays[name]
        if not np.issubdtype(value.dtype, np.floating):
            raise ValueError(f"{name} must use a real floating dtype")
        if not np.isfinite(value).all():
            raise ValueError(f"latent target {name} must be finite")

    for name in ("frame_ids", "teacher_variant_ids", "optimization_steps"):
        if not np.issubdtype(arrays[name].dtype, np.integer):
            raise ValueError(f"{name} must be integers")
    if len(set(arrays["teacher_variant_ids"].tolist())) != 4:
        raise ValueError("teacher_variant_ids must be unique")

    for name in ("teacher_window_masks", "coverage_masks"):
        values = arrays[name]
        is_binary_type = np.issubdtype(values.dtype, np.integer) or np.issubdtype(
            values.dtype, np.bool_
        )
        if not is_binary_type or not np.isin(values, (0, 1)).all():
            raise ValueError(f"{name} must be binary")

    poses = arrays["decoded_c2w_raw"]
    if not np.allclose(poses[..., 3, :], [0.0, 0.0, 0.0, 1.0]):
        raise ValueError("decoded_c2w_raw poses must be homogeneous")

    scene = _unicode_scalar(arrays["scene"], "scene")
    if not scene:
        raise ValueError("scene must be non-empty")
    for name in ("basis_sha256", "source_sha256", "teacher_sha256", "checkpoint_sha256"):
        if _SHA256_RE.fullmatch(_unicode_scalar(arrays[name], name)) is None:
            raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    if _COMMIT_RE.fullmatch(_unicode_scalar(arrays["git_commit"], "git_commit")) is None:
        raise ValueError("git_commit must be a lowercase 40-character git commit")


def save_latent_targets(
    path: Path,
    arrays: Mapping[str, np.ndarray],
    *,
    teacher_artifact: Path | None = None,
) -> str:
    """Validate and atomically publish one strict latent-target archive."""
    normalized = {name: np.asarray(value) for name, value in arrays.items()}
    _validate_latent_targets(normalized)
    if teacher_artifact is not None:
        load_teacher_artifact(teacher_artifact)
        if str(normalized["teacher_sha256"]) != _sha256_file(Path(teacher_artifact)):
            raise ValueError("latent target teacher artifact digest mismatch")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **normalized)
    temporary.replace(path)
    return _sha256_file(path)


def load_latent_targets(path: Path) -> dict[str, np.ndarray]:
    """Load and revalidate one strict latent-target archive without pickles."""
    try:
        archive = np.load(Path(path), allow_pickle=False)
    except (OSError, ValueError, KeyError) as error:
        raise ValueError(f"invalid latent-target archive: {path}") from error
    with archive:
        names = list(archive.files)
        if len(names) != len(LATENT_TARGET_MEMBERS) or set(names) != LATENT_TARGET_MEMBERS:
            if len(names) != len(set(names)):
                raise ValueError("latent-target archive contains duplicate members")
            raise ValueError("latent-target archive must use the exact schema")
        try:
            arrays = {name: np.asarray(archive[name]).copy() for name in names}
        except (OSError, ValueError, KeyError) as error:
            raise ValueError(f"invalid latent-target archive: {path}") from error
    _validate_latent_targets(arrays)
    return arrays


def _validate_pose_stack(value: np.ndarray, shape: tuple[int, ...], name: str) -> None:
    if value.shape != shape or not np.issubdtype(value.dtype, np.floating):
        raise ValueError(f"teacher artifact {name} has invalid shape or dtype")


def _validate_so3(value: np.ndarray, name: str) -> None:
    rotations = np.asarray(value, dtype=np.float64)
    gram = np.einsum("...ji,...jk->...ik", rotations, rotations)
    if (
        not np.isfinite(rotations).all()
        or not np.allclose(gram, np.eye(3), atol=1e-7, rtol=1e-7)
        or not np.allclose(np.linalg.det(rotations), 1.0, atol=1e-7, rtol=1e-7)
    ):
        raise ValueError(f"teacher artifact {name} must contain proper SO(3) rotations")


def _validate_teacher_artifact(arrays: Mapping[str, np.ndarray]) -> None:
    if set(arrays) != TEACHER_ARTIFACT_MEMBERS:
        raise ValueError("teacher artifact must use the exact schema")
    if any(value.dtype.hasobject for value in arrays.values()):
        raise ValueError("teacher artifact may not use object dtypes")
    scalar_unicode = (
        "scene", "oracle_scene", "oracle_frame_digest", "oracle_digest",
        "source_sha256", "formal_label_sha256", "checkpoint_sha256", "git_commit",
    )
    for name in scalar_unicode:
        _unicode_scalar(arrays[name], name)
    scene = str(arrays["scene"])
    if not scene or str(arrays["oracle_scene"]) != scene:
        raise ValueError("teacher artifact oracle scene mismatch")
    for name in (
        "oracle_frame_digest", "oracle_digest", "source_sha256",
        "formal_label_sha256", "checkpoint_sha256",
    ):
        if _SHA256_RE.fullmatch(str(arrays[name])) is None:
            raise ValueError(f"teacher artifact {name} is malformed")
    if _COMMIT_RE.fullmatch(str(arrays["git_commit"])) is None:
        raise ValueError("teacher artifact git_commit is malformed")
    frame_ids = arrays["frame_ids"]
    if frame_ids.shape != (500,) or not np.issubdtype(frame_ids.dtype, np.integer):
        raise ValueError("teacher artifact frame_ids must be integer [500]")
    if np.any(frame_ids[1:] <= frame_ids[:-1]):
        raise ValueError("teacher artifact frame_ids must be strictly increasing")
    _validate_pose_stack(arrays["gt_c2w"], (500, 4, 4), "gt_c2w")
    _validate_pose_stack(arrays["baseline_c2w_raw"], (500, 4, 4), "baseline_c2w_raw")
    _validate_pose_stack(arrays["fused_c2w"], (4, 500, 4, 4), "fused_c2w")
    if not np.isfinite(arrays["gt_c2w"]).all() or not np.isfinite(arrays["baseline_c2w_raw"]).all():
        raise ValueError("teacher artifact baseline and GT poses must be finite")
    if not np.allclose(arrays["gt_c2w"][..., 3, :], [0, 0, 0, 1]) or not np.allclose(
        arrays["baseline_c2w_raw"][..., 3, :], [0, 0, 0, 1]
    ):
        raise ValueError("teacher artifact baseline and GT poses must be homogeneous")
    _validate_so3(arrays["gt_c2w"][..., :3, :3], "gt_c2w")
    _validate_so3(arrays["baseline_c2w_raw"][..., :3, :3], "baseline_c2w_raw")
    expected_shapes = {
        "gt_scene_scale": (), "oracle_fit_count": (), "oracle_scale": (),
        "oracle_rotation": (3, 3), "oracle_translation": (3,), "oracle_rank": (),
        "oracle_condition": (), "window_weights": (9,), "window_masks": (4, 9),
        "coverage_weights": (4, 500), "variant_utilities": (4,),
    }
    for name, shape in expected_shapes.items():
        if arrays[name].shape != shape:
            raise ValueError(f"teacher artifact {name} has invalid shape")
    for name in ("oracle_fit_count", "oracle_rank"):
        if not np.issubdtype(arrays[name].dtype, np.integer):
            raise ValueError(f"teacher artifact {name} must be integer")
    for name in (
        "gt_scene_scale", "oracle_scale", "oracle_rotation", "oracle_translation",
        "oracle_condition", "window_weights", "coverage_weights", "variant_utilities",
    ):
        if not np.issubdtype(arrays[name].dtype, np.floating) or not np.isfinite(arrays[name]).all():
            raise ValueError(f"teacher artifact {name} must be finite floating point")
    if float(arrays["gt_scene_scale"]) <= 0.0 or float(arrays["oracle_scale"]) <= 0.0:
        raise ValueError("teacher artifact scales must be positive")
    _validate_so3(arrays["oracle_rotation"], "oracle_rotation")
    oracle_payload = {
        "scene": str(arrays["oracle_scene"]),
        "frame_digest": str(arrays["oracle_frame_digest"]),
        "fit_count": int(arrays["oracle_fit_count"]),
        "scale": float(arrays["oracle_scale"]),
        "rotation": tuple(
            tuple(float(value) for value in row)
            for row in arrays["oracle_rotation"]
        ),
        "translation": tuple(float(value) for value in arrays["oracle_translation"]),
    }
    if canonical_json_digest(oracle_payload) != str(arrays["oracle_digest"]):
        raise ValueError("teacher artifact oracle digest mismatch")
    masks = arrays["window_masks"]
    if not (np.issubdtype(masks.dtype, np.integer) or np.issubdtype(masks.dtype, np.bool_)) or not np.isin(masks, (0, 1)).all():
        raise ValueError("teacher artifact window_masks must be binary")
    coverage = arrays["coverage_weights"]
    if np.any(coverage < 0.0):
        raise ValueError("teacher artifact coverage must be nonnegative")
    covered = coverage > 0.0
    fused = arrays["fused_c2w"]
    if np.any(covered) and not np.isfinite(fused[covered]).all():
        raise ValueError("teacher artifact covered fused poses must be finite")
    if np.any(~covered) and not np.isnan(fused[~covered]).all():
        raise ValueError("teacher artifact uncovered fused poses must be all-NaN")
    if np.any(covered) and not np.allclose(fused[covered][:, 3, :], [0, 0, 0, 1]):
        raise ValueError("teacher artifact covered fused poses must be homogeneous")
    if np.any(covered):
        _validate_so3(fused[covered][:, :3, :3], "covered fused_c2w")


def save_teacher_artifact(path: Path, arrays: Mapping[str, np.ndarray]) -> str:
    normalized = {name: np.asarray(value) for name, value in arrays.items()}
    _validate_teacher_artifact(normalized)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **normalized)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return _sha256_file(target)


def load_teacher_artifact(path: Path) -> dict[str, np.ndarray]:
    try:
        archive = np.load(Path(path), allow_pickle=False)
    except (OSError, ValueError, KeyError) as error:
        raise ValueError(f"invalid teacher artifact: {path}") from error
    with archive:
        names = list(archive.files)
        if len(names) != len(TEACHER_ARTIFACT_MEMBERS) or set(names) != TEACHER_ARTIFACT_MEMBERS:
            raise ValueError("teacher artifact must use the exact schema")
        arrays = {name: np.asarray(archive[name]).copy() for name in names}
    _validate_teacher_artifact(arrays)
    return arrays
