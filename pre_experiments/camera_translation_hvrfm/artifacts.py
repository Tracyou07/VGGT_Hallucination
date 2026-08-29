"""Exact, physically separated NPZ contracts for translation targets."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import tempfile
from typing import Callable, Mapping
import zipfile

import numpy as np

from pre_experiments.camera_velocity_ambiguity_02.artifacts import frame_digest
from pre_experiments.camera_velocity_ambiguity_02.contracts import canonical_json_digest
from pre_experiments.camera_translation_hvrfm.geometry import (
    _validate_baseline_c2w,
    _validate_pose_encoding_matches_c2w,
    _validate_so3,
    prediction_scale,
)


LONG_CONTEXT_MEMBERS = frozenset(
    {
        "sample_id",
        "scene",
        "frame_ids",
        "camera_tokens",
        "baseline_pose_encoding",
        "baseline_c2w",
        "prediction_scale",
        "source_sha256",
        "checkpoint_sha256",
        "git_commit",
    }
)
SHORT_CONTEXT_MEMBERS = frozenset(
    {
        "sample_id",
        "scene",
        "short_frame_ids",
        "short_camera_tokens",
        "long_context_sha256",
        "source_sha256",
        "checkpoint_sha256",
        "git_commit",
    }
)
TRANSLATION_TARGET_MEMBERS = frozenset(
    {
        "sample_id",
        "scene",
        "frame_ids",
        "teacher_variant_ids",
        "coverage_mask",
        "translation_endpoints",
        "teacher_centers_raw_filled",
        "prediction_scale",
        "long_context_sha256",
        "short_context_sha256",
        "quality_sha256",
        "teacher_reference_sha256",
        "source_sha256",
        "checkpoint_sha256",
        "git_commit",
    }
)
QUALITY_SIDECAR_MEMBERS = frozenset(
    {
        "sample_id",
        "scene",
        "frame_ids",
        "teacher_variant_ids",
        "gt_c2w",
        "gt_scene_scale",
        "oracle_scene",
        "oracle_frame_digest",
        "oracle_fit_count",
        "oracle_scale",
        "oracle_rotation",
        "oracle_translation",
        "oracle_rank",
        "oracle_condition",
        "oracle_digest",
        "window_weights",
        "window_masks",
        "coverage_weights",
        "variant_utilities",
        "baseline_translation_error_normalized",
        "baseline_rotation_error_deg",
        "teacher_translation_error_normalized",
        "teacher_rotation_error_deg",
        "source_sha256",
        "formal_label_sha256",
        "teacher_reference_sha256",
        "checkpoint_sha256",
        "git_commit",
    }
)


_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_SCENE_RE = re.compile(r"scene\d{4}_\d{2}")
_SAMPLE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]*")
_FRAMES = 500
_ENDPOINTS = 4
_SHORT_WINDOWS = 9
_SHORT_FRAMES = 100
_TOKEN_WIDTH = 2048


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _absolute_without_resolving(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _reject_symlink_components(path: Path) -> None:
    if ".." in Path(path).parts:
        raise ValueError("artifact paths may not contain lexical parent traversal")
    current = _absolute_without_resolving(path)
    for candidate in (current, *current.parents):
        if candidate.is_symlink():
            raise ValueError(f"artifact paths may not contain symlinks: {candidate}")


def _require_exact_arrays(
    arrays: Mapping[str, np.ndarray], members: frozenset[str], *, label: str
) -> dict[str, np.ndarray]:
    if not isinstance(arrays, Mapping) or set(arrays) != members:
        raise ValueError(f"{label} must use the exact schema")
    validated: dict[str, np.ndarray] = {}
    for name in members:
        value = arrays[name]
        if not isinstance(value, np.ndarray):
            raise ValueError(f"{label} {name} must be a NumPy array")
        if value.dtype.hasobject:
            raise ValueError(f"{label} may not contain object arrays")
        validated[name] = value.copy()
    return validated


def _expect(
    arrays: Mapping[str, np.ndarray],
    name: str,
    *,
    shape: tuple[int, ...],
    dtype: str | type[np.generic],
) -> np.ndarray:
    value = arrays[name]
    expected_dtype = np.dtype(dtype)
    if value.shape != shape or value.dtype != expected_dtype:
        raise ValueError(
            f"{name} must have exact shape {shape} and dtype {expected_dtype}"
        )
    return value


def _text(
    arrays: Mapping[str, np.ndarray], name: str, *, width: int
) -> str:
    value = _expect(arrays, name, shape=(), dtype=f"U{width}")
    return str(value)


def _digest(arrays: Mapping[str, np.ndarray], name: str) -> str:
    value = _text(arrays, name, width=64)
    if _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _identity(arrays: Mapping[str, np.ndarray]) -> tuple[str, str]:
    sample_id = _text(arrays, "sample_id", width=96)
    scene = _text(arrays, "scene", width=32)
    if _SAMPLE_RE.fullmatch(sample_id) is None:
        raise ValueError("sample_id must be non-empty and use only safe characters")
    if _SCENE_RE.fullmatch(scene) is None:
        raise ValueError("scene must use sceneNNNN_NN format")
    return sample_id, scene


def _provenance(arrays: Mapping[str, np.ndarray]) -> None:
    _digest(arrays, "source_sha256")
    _digest(arrays, "checkpoint_sha256")
    commit = _text(arrays, "git_commit", width=40)
    if _COMMIT_RE.fullmatch(commit) is None:
        raise ValueError("git_commit must be a lowercase 40-character commit")


def _frame_ids(arrays: Mapping[str, np.ndarray], name: str = "frame_ids") -> np.ndarray:
    values = _expect(arrays, name, shape=(_FRAMES,), dtype=np.int64)
    if np.any(values[1:] <= values[:-1]):
        raise ValueError(f"{name} must be strictly increasing and unique")
    return values


def _finite(array: np.ndarray, *, name: str) -> None:
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")


def _validate_long(arrays: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    values = _require_exact_arrays(arrays, LONG_CONTEXT_MEMBERS, label="long context")
    _identity(values)
    _provenance(values)
    _frame_ids(values)
    tokens = _expect(
        values, "camera_tokens", shape=(_FRAMES, _TOKEN_WIDTH), dtype=np.float32
    )
    pose = _expect(
        values, "baseline_pose_encoding", shape=(_FRAMES, 9), dtype=np.float32
    )
    c2w = _expect(
        values, "baseline_c2w", shape=(_FRAMES, 4, 4), dtype=np.float64
    )
    scale = _expect(values, "prediction_scale", shape=(), dtype=np.float64)
    _finite(tokens, name="camera_tokens")
    _finite(pose, name="baseline_pose_encoding")
    _validate_baseline_c2w(c2w)
    _validate_pose_encoding_matches_c2w(pose, c2w)
    expected_scale = np.asarray(prediction_scale(c2w), dtype=np.float64)
    if scale.tobytes() != expected_scale.tobytes():
        raise ValueError("prediction_scale must be the exact baseline C2W center RMS")
    return values


def _validate_short(arrays: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    values = _require_exact_arrays(arrays, SHORT_CONTEXT_MEMBERS, label="short context")
    _identity(values)
    _provenance(values)
    frame_ids = _expect(
        values,
        "short_frame_ids",
        shape=(_SHORT_WINDOWS, _SHORT_FRAMES),
        dtype=np.int64,
    )
    if np.any(frame_ids[:, 1:] <= frame_ids[:, :-1]):
        raise ValueError("short_frame_ids rows must be strictly increasing and unique")
    tokens = _expect(
        values,
        "short_camera_tokens",
        shape=(_SHORT_WINDOWS, _SHORT_FRAMES, _TOKEN_WIDTH),
        dtype=np.float32,
    )
    _finite(tokens, name="short_camera_tokens")
    _digest(values, "long_context_sha256")
    return values


def _validate_variant_ids(arrays: Mapping[str, np.ndarray]) -> np.ndarray:
    variants = _expect(
        arrays, "teacher_variant_ids", shape=(_ENDPOINTS,), dtype=np.int64
    )
    if not np.array_equal(variants, np.arange(_ENDPOINTS, dtype=np.int64)):
        raise ValueError("teacher_variant_ids must be exactly [0, 1, 2, 3]")
    return variants


def _validate_target(arrays: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    values = _require_exact_arrays(
        arrays, TRANSLATION_TARGET_MEMBERS, label="translation target"
    )
    _identity(values)
    _provenance(values)
    _frame_ids(values)
    _validate_variant_ids(values)
    coverage = _expect(
        values,
        "coverage_mask",
        shape=(_ENDPOINTS, _FRAMES),
        dtype=np.uint8,
    )
    if not np.isin(coverage, (0, 1)).all():
        raise ValueError("coverage_mask must be binary uint8")
    endpoints = _expect(
        values,
        "translation_endpoints",
        shape=(_ENDPOINTS, _FRAMES, 3),
        dtype=np.float32,
    )
    centers = _expect(
        values,
        "teacher_centers_raw_filled",
        shape=(_ENDPOINTS, _FRAMES, 3),
        dtype=np.float64,
    )
    _finite(endpoints, name="translation_endpoints")
    _finite(centers, name="teacher_centers_raw_filled")
    if not np.all(endpoints.view(np.uint32)[coverage == 0] == 0):
        raise ValueError("uncovered translation endpoints must be bitwise positive zero")
    scale = _expect(values, "prediction_scale", shape=(), dtype=np.float64)
    if not np.isfinite(scale).all() or float(scale) <= 0.0:
        raise ValueError("prediction_scale must be finite and positive")
    for name in (
        "long_context_sha256",
        "short_context_sha256",
        "quality_sha256",
        "teacher_reference_sha256",
    ):
        _digest(values, name)
    return values


def _nonnegative_finite(value: np.ndarray, *, name: str) -> None:
    if not np.isfinite(value).all() or np.any(value < 0.0):
        raise ValueError(f"{name} must be finite and nonnegative")


def _validate_quality(arrays: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    values = _require_exact_arrays(
        arrays, QUALITY_SIDECAR_MEMBERS, label="quality sidecar"
    )
    _, scene = _identity(values)
    _provenance(values)
    frames = _frame_ids(values)
    _validate_variant_ids(values)
    gt_c2w = _expect(values, "gt_c2w", shape=(_FRAMES, 4, 4), dtype=np.float64)
    _validate_baseline_c2w(gt_c2w)
    gt_scale = _expect(values, "gt_scene_scale", shape=(), dtype=np.float64)
    if not np.isfinite(gt_scale).all() or float(gt_scale) <= 0.0:
        raise ValueError("gt_scene_scale must be finite and positive")

    oracle_scene = _text(values, "oracle_scene", width=32)
    if oracle_scene != scene:
        raise ValueError("oracle_scene must match scene")
    oracle_frame_digest = _digest(values, "oracle_frame_digest")
    if oracle_frame_digest != frame_digest(frames):
        raise ValueError("oracle_frame_digest must bind frame_ids")
    fit_count = _expect(values, "oracle_fit_count", shape=(), dtype=np.int64)
    oracle_scale = _expect(values, "oracle_scale", shape=(), dtype=np.float64)
    oracle_rotation = _expect(
        values, "oracle_rotation", shape=(3, 3), dtype=np.float64
    )
    oracle_translation = _expect(
        values, "oracle_translation", shape=(3,), dtype=np.float64
    )
    oracle_rank = _expect(values, "oracle_rank", shape=(), dtype=np.int64)
    oracle_condition = _expect(
        values, "oracle_condition", shape=(), dtype=np.float64
    )
    if int(fit_count) < 1 or int(fit_count) > _FRAMES:
        raise ValueError("oracle_fit_count must be between 1 and 500")
    if int(oracle_rank) not in (1, 2, 3):
        raise ValueError("oracle_rank must be 1, 2, or 3")
    if not np.isfinite(oracle_scale).all() or float(oracle_scale) <= 0.0:
        raise ValueError("oracle_scale must be finite and positive")
    if not np.isfinite(oracle_condition).all() or float(oracle_condition) <= 0.0:
        raise ValueError("oracle_condition must be finite and positive")
    _finite(oracle_translation, name="oracle_translation")
    _validate_so3(oracle_rotation, name="oracle_rotation", atol=1e-7)
    oracle_payload = {
        "scene": oracle_scene,
        "frame_digest": oracle_frame_digest,
        "fit_count": int(fit_count),
        "scale": float(oracle_scale),
        "rotation": tuple(
            tuple(float(component) for component in row) for row in oracle_rotation
        ),
        "translation": tuple(float(component) for component in oracle_translation),
    }
    if _digest(values, "oracle_digest") != canonical_json_digest(oracle_payload):
        raise ValueError("oracle_digest does not match its canonical payload")

    window_weights = _expect(values, "window_weights", shape=(9,), dtype=np.float64)
    window_masks = _expect(
        values, "window_masks", shape=(_ENDPOINTS, 9), dtype=np.uint8
    )
    coverage = _expect(
        values,
        "coverage_weights",
        shape=(_ENDPOINTS, _FRAMES),
        dtype=np.float64,
    )
    utilities = _expect(
        values, "variant_utilities", shape=(_ENDPOINTS,), dtype=np.float64
    )
    _nonnegative_finite(window_weights, name="window_weights")
    if not np.isin(window_masks, (0, 1)).all():
        raise ValueError("window_masks must be binary uint8")
    _nonnegative_finite(coverage, name="coverage_weights")
    _finite(utilities, name="variant_utilities")

    for name in (
        "baseline_translation_error_normalized",
        "baseline_rotation_error_deg",
    ):
        value = _expect(values, name, shape=(_FRAMES,), dtype=np.float64)
        _nonnegative_finite(value, name=name)
    covered = coverage > 0.0
    for name in (
        "teacher_translation_error_normalized",
        "teacher_rotation_error_deg",
    ):
        value = _expect(
            values, name, shape=(_ENDPOINTS, _FRAMES), dtype=np.float64
        )
        if np.any(covered) and (
            not np.isfinite(value[covered]).all() or np.any(value[covered] < 0.0)
        ):
            raise ValueError(f"{name} must be finite and nonnegative where covered")
        if np.any(~covered) and not np.isnan(value[~covered]).all():
            raise ValueError(f"{name} must be NaN where uncovered")
    _digest(values, "formal_label_sha256")
    _digest(values, "teacher_reference_sha256")
    return values


_Validator = Callable[[Mapping[str, np.ndarray]], dict[str, np.ndarray]]


def _atomic_save(
    path: Path, arrays: Mapping[str, np.ndarray], validator: _Validator
) -> str:
    values = validator(arrays)
    target = Path(path)
    _reject_symlink_components(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(target)
    if target.exists() and not target.is_file():
        raise ValueError("artifact target must be a regular file")

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            np.savez_compressed(handle, **values)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, target)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return _sha256_file(target)


def _load(
    path: Path,
    members: frozenset[str],
    validator: _Validator,
    *,
    label: str,
) -> dict[str, np.ndarray]:
    source = Path(path)
    _reject_symlink_components(source)
    if not source.is_file():
        raise ValueError(f"{label} must be a regular NPZ file")
    expected_names = {f"{name}.npy" for name in members}
    try:
        with zipfile.ZipFile(source, "r") as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)):
                raise ValueError(f"{label} contains duplicate ZIP members")
            if any(
                info.is_dir()
                or "/" in info.filename
                or "\\" in info.filename
                or info.filename in {".", ".."}
                for info in infos
            ):
                raise ValueError(f"{label} contains an unsafe ZIP member path")
            if set(names) != expected_names or len(names) != len(expected_names):
                raise ValueError(f"{label} must use the exact schema")
        with np.load(source, allow_pickle=False) as archive:
            arrays = {name: archive[name].copy() for name in members}
    except ValueError:
        raise
    except (OSError, KeyError, EOFError, zipfile.BadZipFile, zipfile.LargeZipFile) as error:
        raise ValueError(f"invalid {label}: {source}") from error
    return validator(arrays)


def save_long_context(path: Path, arrays: Mapping[str, np.ndarray]) -> str:
    """Validate and atomically write one prediction-only long context."""
    return _atomic_save(path, arrays, _validate_long)


def load_long_context(path: Path) -> dict[str, np.ndarray]:
    """Load one strict prediction-only long context without pickle support."""
    return _load(path, LONG_CONTEXT_MEMBERS, _validate_long, label="long context")


def save_short_context(path: Path, arrays: Mapping[str, np.ndarray]) -> str:
    """Validate and atomically write one privileged-training short context."""
    return _atomic_save(path, arrays, _validate_short)


def load_short_context(path: Path) -> dict[str, np.ndarray]:
    """Load one strict privileged-training short context without pickles."""
    return _load(path, SHORT_CONTEXT_MEMBERS, _validate_short, label="short context")


def save_translation_target(path: Path, arrays: Mapping[str, np.ndarray]) -> str:
    """Validate and atomically write one translation-target sidecar."""
    return _atomic_save(path, arrays, _validate_target)


def load_translation_target(path: Path) -> dict[str, np.ndarray]:
    """Load one strict translation-target sidecar without pickles."""
    return _load(
        path,
        TRANSLATION_TARGET_MEMBERS,
        _validate_target,
        label="translation target",
    )


def save_quality_sidecar(path: Path, arrays: Mapping[str, np.ndarray]) -> str:
    """Validate and atomically write one physically separate quality sidecar."""
    return _atomic_save(path, arrays, _validate_quality)


def load_quality_sidecar(path: Path) -> dict[str, np.ndarray]:
    """Load one strict privileged quality sidecar without pickles."""
    return _load(
        path, QUALITY_SIDECAR_MEMBERS, _validate_quality, label="quality sidecar"
    )


def load_bound_bundle(
    long_path: Path,
    short_path: Path,
    target_path: Path,
    quality_path: Path,
) -> dict[str, dict[str, np.ndarray]]:
    """Load four artifacts and verify identities plus actual file-digest bindings."""
    paths = {
        "long": Path(long_path),
        "short": Path(short_path),
        "target": Path(target_path),
        "quality": Path(quality_path),
    }
    bundle = {
        "long": load_long_context(paths["long"]),
        "short": load_short_context(paths["short"]),
        "target": load_translation_target(paths["target"]),
        "quality": load_quality_sidecar(paths["quality"]),
    }
    actual = {name: _sha256_file(path) for name, path in paths.items()}
    long = bundle["long"]
    short = bundle["short"]
    target = bundle["target"]
    quality = bundle["quality"]

    if str(short["long_context_sha256"]) != actual["long"]:
        raise ValueError("short context does not bind the actual long-context file")
    for member, artifact_name in (
        ("long_context_sha256", "long"),
        ("short_context_sha256", "short"),
        ("quality_sha256", "quality"),
    ):
        if str(target[member]) != actual[artifact_name]:
            raise ValueError(f"translation target {member} does not bind the actual file")

    for field in ("sample_id", "scene", "source_sha256", "checkpoint_sha256", "git_commit"):
        if len({str(arrays[field]) for arrays in bundle.values()}) != 1:
            raise ValueError(f"bundle {field} values must match exactly")
    if not (
        np.array_equal(long["frame_ids"], target["frame_ids"])
        and np.array_equal(long["frame_ids"], quality["frame_ids"])
    ):
        raise ValueError("bundle long/target/quality frame IDs must match exactly")
    expected_short_frames = np.stack(
        [long["frame_ids"][start : start + _SHORT_FRAMES] for start in range(0, 401, 50)]
    )
    if not np.array_equal(short["short_frame_ids"], expected_short_frames):
        raise ValueError("short window frame IDs do not match the bound long context")
    if not np.array_equal(target["teacher_variant_ids"], quality["teacher_variant_ids"]):
        raise ValueError("target and quality teacher variant IDs must match")
    expected_coverage = (quality["coverage_weights"] > 0.0).astype(np.uint8)
    if not np.array_equal(target["coverage_mask"], expected_coverage):
        raise ValueError("target coverage mask does not match quality coverage weights")
    if target["prediction_scale"].tobytes() != long["prediction_scale"].tobytes():
        raise ValueError("target and long prediction scales must be bitwise equal")
    if str(target["teacher_reference_sha256"]) != str(
        quality["teacher_reference_sha256"]
    ):
        raise ValueError("target and quality teacher-reference digests must match")
    return bundle


__all__ = [
    "LONG_CONTEXT_MEMBERS",
    "QUALITY_SIDECAR_MEMBERS",
    "SHORT_CONTEXT_MEMBERS",
    "TRANSLATION_TARGET_MEMBERS",
    "load_bound_bundle",
    "load_long_context",
    "load_quality_sidecar",
    "load_short_context",
    "load_translation_target",
    "save_long_context",
    "save_quality_sidecar",
    "save_short_context",
    "save_translation_target",
]
