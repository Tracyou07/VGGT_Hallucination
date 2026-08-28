from __future__ import annotations

import hashlib
from pathlib import Path
import re
from typing import Mapping

import numpy as np


LATENT_TARGET_MEMBERS = {
    "scene", "frame_ids", "teacher_variant_ids", "teacher_window_masks",
    "coverage_masks", "residual_coefficients", "decoded_c2w_raw",
    "optimization_steps", "initial_losses", "final_losses", "basis_sha256",
    "source_sha256", "teacher_sha256", "checkpoint_sha256", "git_commit",
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

    for name, value in arrays.items():
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError(f"latent target {name} must be finite")

    if not np.issubdtype(arrays["frame_ids"].dtype, np.integer):
        raise ValueError("frame_ids must be integers")
    if not np.issubdtype(arrays["teacher_variant_ids"].dtype, np.integer):
        raise ValueError("teacher_variant_ids must be integers")
    if len(set(arrays["teacher_variant_ids"].tolist())) != 4:
        raise ValueError("teacher_variant_ids must be unique")
    if not np.issubdtype(arrays["optimization_steps"].dtype, np.integer):
        raise ValueError("optimization_steps must be integers")

    for name in ("teacher_window_masks", "coverage_masks"):
        values = arrays[name]
        if not np.issubdtype(values.dtype, np.number) or not np.isin(values, (0, 1)).all():
            raise ValueError(f"{name} must be binary")

    poses = arrays["decoded_c2w_raw"]
    if not np.issubdtype(poses.dtype, np.number):
        raise ValueError("decoded_c2w_raw must be numeric")
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


def save_latent_targets(path: Path, arrays: Mapping[str, np.ndarray]) -> str:
    """Validate and atomically publish one strict latent-target archive."""
    normalized = {name: np.asarray(value) for name, value in arrays.items()}
    _validate_latent_targets(normalized)
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
        with np.load(Path(path), allow_pickle=False) as archive:
            arrays = {name: np.asarray(archive[name]).copy() for name in archive.files}
    except (OSError, ValueError, KeyError) as error:
        raise ValueError(f"invalid latent-target archive: {path}") from error
    _validate_latent_targets(arrays)
    return arrays
