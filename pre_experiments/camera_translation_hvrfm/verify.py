"""Independent fail-closed verification for Stage A-prime translation endpoints."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from io import BytesIO
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Iterator, Mapping, Sequence
import zipfile

import numpy as np
import torch
from torch import nn

from pre_experiments.variational_camera_latent.camera import (
    pose_encoding_to_c2w,
)


INVENTORY_SCHEMA = "camera_translation_hvrfm.verification_inventory.v1"
VERIFIED_SCHEMA = "camera_translation_hvrfm.verified_completion.v1"

_REPORT_SCHEMA = "camera_translation_hvrfm.stage_a_prime_report.v1"
_REPORT_COMPLETION_SCHEMA = "camera_translation_hvrfm.stage_a_prime_completion.v1"
_READY = "TRANSLATION_ENDPOINTS_READY"
_FROZEN_SOURCE_RUN_NAME = "vrfm_camera_20260827T044926Z"
_FROZEN_REFERENCE_RUN_NAME = "privileged_teacher_lift_20260829T012716Z_tolfix"
_FROZEN_FORMAL_RUN_NAME = "long_short_head_formal_20260828T072407Z"
_FROZEN_SOURCE_COMPLETION_SHA256 = (
    "fd1b93caa16f45f0dbdc55fd7000aba9ab8bf166a7240f5ac2a716a0b3de9a32"
)
_FROZEN_SOURCE_MANIFEST_SHA256 = (
    "be5aaa1b61be5e25709e40b3912e48aab38b6bbfac4be3b7ed183140219d6054"
)
_FROZEN_REFERENCE_COMPLETION_SHA256 = (
    "7e63ca36e6fc4c08772e3356255f84c2853c9d46310ae546cc5e53dc1792048c"
)
_FROZEN_REFERENCE_LONG_MANIFEST_SHA256 = (
    "6b6ab434bb4cd8bd4afbeaf8a2d11354f321d8791501ada3ef2f9376eb064166"
)
_FROZEN_FORMAL_COMPLETION_SHA256 = (
    "4d24b944792f348ccc8c180a99f3e0ee11397ce472900eb6abe38f6924732667"
)
_FROZEN_FORMAL_DATA_MANIFEST_SHA256 = (
    "944ee57a75a68af45fc0ea6037070267552ea3f042bd2346638cdc65f2dd4a6e"
)
_FROZEN_CHECKPOINT_SHA256 = (
    "f164acf60724910d8fe1578bb499d800850c7bb0948db7555c413f9fbe60467e"
)
_FROZEN_REFERENCE_GIT = "cee41a09ac4085c8d6b0b343ca07d8e8c53ace3c"
_FROZEN_FORMAL_GIT = "2476a59f583ce4c39bbe66dc65d6a8e5cddfb52e"
_FROZEN_BASIS_SHA256 = (
    "89fecc83d51be8a1923a0c177c20b45dec8d8aa611fae0b615ef9293511dd213"
)
_FRAMES = 500
_WINDOWS = 9
_WINDOW_FRAMES = 100
_ENDPOINTS = 4
_TOKEN_WIDTH = 2048
_CROSS_DEVICE_CENTER_ATOL = 5e-6
_CROSS_DEVICE_ROTATION_ATOL_DEG = 2e-5
_SCENES = (
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
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")

_LONG_MEMBERS = frozenset(
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
_REFERENCE_LONG_MEMBERS = frozenset(
    {
        "scene",
        "frame_ids",
        "camera_tokens",
        "baseline_c2w",
        "source_sha256",
    }
)
_SHORT_MEMBERS = frozenset(
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
_TARGET_MEMBERS = frozenset(
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
_QUALITY_MEMBERS = frozenset(
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
_REFERENCE_TEACHER_MEMBERS = frozenset(
    {
        "scene",
        "frame_ids",
        "gt_c2w",
        "gt_scene_scale",
        "baseline_c2w_raw",
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
        "fused_c2w",
        "variant_utilities",
        "source_sha256",
        "formal_label_sha256",
        "checkpoint_sha256",
        "git_commit",
    }
)
_FORMAL_LABEL_MEMBERS = frozenset(
    {
        "scene",
        "frame_ids",
        "gt_c2w",
        "oracle_scale",
        "oracle_rotation",
        "oracle_translation",
        "oracle_digest",
        "gt_scene_scale",
        "baseline_pose_encoding",
        "teacher_c2w_gt_gauge",
        "teacher_weight",
        "window_teacher_weight",
        "window_baseline_rms",
        "window_teacher_rms",
        "source_sha256",
        "checkpoint_sha256",
    }
)

_CONFIG_FIELDS = frozenset(
    {
        "schema",
        "run_id",
        "git_commit",
        "source_run",
        "source_completion_sha256",
        "source_manifest_sha256",
        "reference_run",
        "reference_completion_sha256",
        "reference_inventory_sha256",
        "reference_long_manifest_sha256",
        "reference_teacher_manifest_sha256",
        "formal_run",
        "formal_completion_sha256",
        "formal_data_manifest_sha256",
        "checkpoint_file",
        "checkpoint_sha256",
        "preflight_evidence_sha256",
        "long_context_manifest_sha256",
        "cohort_manifest_sha256",
        "scene_count",
        "endpoint_count",
        "smoke_scene",
    }
)
_PREFLIGHT_FIELDS = frozenset(
    {
        "schema",
        "stage",
        "run_id",
        "git_commit",
        "source_run",
        "source_completion_sha256",
        "source_manifest_sha256",
        "reference_run",
        "reference_completion_sha256",
        "reference_inventory_sha256",
        "reference_config_sha256",
        "reference_report_json_sha256",
        "reference_report_markdown_sha256",
        "reference_long_manifest_sha256",
        "reference_teacher_manifest_sha256",
        "formal_run",
        "formal_completion_sha256",
        "formal_data_manifest_sha256",
        "checkpoint_file",
        "checkpoint_sha256",
        "scene_bindings",
        "completion_digest",
    }
)
_LONG_MANIFEST_FIELDS = frozenset({"schema", "run_id", "git_commit", "records"})
_COHORT_MANIFEST_FIELDS = _LONG_MANIFEST_FIELDS
_STAGE_FIELDS = frozenset(
    {
        "schema",
        "stage",
        "run_id",
        "git_commit",
        "run_config_sha256",
        "previous_marker_sha256",
        "files",
        "metadata",
        "completion_digest",
    }
)
_REPORT_COMPLETION_FIELDS = frozenset(
    {
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
)
_REPORT_FIELDS = frozenset(
    {
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
)
_GATE_NAMES = (
    "finite",
    "uncovered_positive_zero",
    "quaternion_bytes_equal",
    "fov_bytes_equal",
    "covered_roundtrip",
    "uncovered_anchor",
    "rotation_guard",
    "teacher_retention",
    "positive_scene_count",
    "positive_mean",
    "minimum_full_utility",
    "physical_leakage_clean",
)

_SOURCE_COMPLETION_FIELDS = frozenset(
    {
        "schema",
        "signal",
        "scene_count",
        "overlap_count",
        "candidate_count",
        "prediction_manifest_sha256",
        "privileged_manifest_sha256",
        "report_sha256",
        "completion_digest",
    }
)
_SOURCE_MANIFEST_FIELDS = frozenset(
    {"schema", "dataset_root", "source_run_digest", "records"}
)
_SOURCE_RECORD_FIELDS = frozenset(
    {"scene", "role", "path", "overlap_count", "sha256"}
)
_REFERENCE_COMPLETION_FIELDS = frozenset(
    {
        "schema",
        "git_commit",
        "classification",
        "inventory_sha256",
        "file_count",
        "completion_digest",
    }
)
_REFERENCE_INVENTORY_FIELDS = frozenset(
    {"schema", "git_commit", "classification", "files"}
)
_REFERENCE_CONFIG_FIELDS = frozenset(
    {
        "schema",
        "git_commit",
        "checkpoint_sha256",
        "basis_sha256",
        "long_manifest_sha256",
        "teacher_manifest_sha256",
        "source_run",
        "source_manifest_sha256",
        "formal_run_root",
        "formal_completion_sha256",
        "formal_data_manifest_sha256",
        "smoke_scene",
        "smoke_steps",
        "calibration_steps",
        "scene_count",
        "variant_count",
    }
)
_REFERENCE_LONG_FIELDS = frozenset({"schema", "records"})
_REFERENCE_LONG_RECORD_FIELDS = frozenset(
    {"scene", "role", "file", "sha256", "source_sha256"}
)
_REFERENCE_TEACHER_FIELDS = frozenset(
    {
        "schema",
        "git_commit",
        "checkpoint_sha256",
        "teacher_upper_bound",
        "formal_completion_sha256",
        "formal_data_manifest_sha256",
        "records",
    }
)
_REFERENCE_TEACHER_RECORD_FIELDS = frozenset(
    {"scene", "role", "file", "sha256", "formal_label_sha256"}
)
_REFERENCE_REPORT_FIELDS = frozenset(
    {
        "schema",
        "git_commit",
        "classification",
        "failed_gates",
        "scene_metrics",
        "provenance",
    }
)
_FORMAL_COMPLETION_FIELDS = frozenset(
    {
        "schema",
        "git_revision",
        "verifier_git_revision",
        "source_manifest_sha256",
        "base_checkpoint_sha256",
        "config_sha256",
        "data_manifest_sha256",
        "test_evidence_sha256",
        "stage_completion_sha256",
        "scene_count",
        "train_scene_count",
        "locked_replay_scene_count",
        "classification",
        "report_sha256",
        "artifacts",
        "inference_leakage_audit",
        "formal_protocol_sha256",
    }
)
_FORMAL_MANIFEST_FIELDS = frozenset(
    {
        "schema",
        "git_revision",
        "source_run",
        "source_manifest_sha256",
        "prepared_root",
        "checkpoint_dir",
        "base_checkpoint_sha256",
        "records",
    }
)
_FORMAL_RECORD_FIELDS = frozenset(
    {
        "scene",
        "role",
        "source_path",
        "source_sha256",
        "long_context_path",
        "long_context_sha256",
        "privileged_path",
        "privileged_sha256",
        "teacher_frame_count",
    }
)


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    try:
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
    except (TypeError, ValueError) as error:
        raise ValueError("JSON payload is not canonicalizable") from error


def _decode_json_snapshot(
    payload: bytes,
    *,
    label: str,
    expected_fields: frozenset[str],
) -> dict[str, object]:
    if type(payload) is not bytes:
        raise ValueError(f"{label} snapshot must be immutable bytes")
    try:
        decoded = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid {label} JSON snapshot") from error
    if not isinstance(decoded, dict) or set(decoded) != set(expected_fields):
        raise ValueError(f"{label} must use the exact JSON schema")
    if payload != _canonical_json_bytes(decoded):
        raise ValueError(f"{label} JSON bytes are not canonical")
    return decoded


def _decode_npz_snapshot(
    payload: bytes,
    *,
    label: str,
    expected_members: frozenset[str],
) -> dict[str, np.ndarray]:
    if type(payload) is not bytes:
        raise ValueError(f"{label} snapshot must be immutable bytes")
    expected_names = {f"{member}.npy" for member in expected_members}
    try:
        with zipfile.ZipFile(BytesIO(payload), "r") as archive:
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
                raise ValueError(f"{label} must use the exact NPZ schema")
        with np.load(BytesIO(payload), allow_pickle=False) as archive:
            arrays = {member: archive[member].copy() for member in expected_members}
    except ValueError as error:
        if "Object arrays cannot be loaded" in str(error):
            raise ValueError(f"{label} may not contain object arrays") from error
        raise
    except (OSError, EOFError, KeyError, zipfile.BadZipFile, zipfile.LargeZipFile) as error:
        raise ValueError(f"invalid {label} NPZ snapshot") from error
    if any(array.dtype.hasobject for array in arrays.values()):
        raise ValueError(f"{label} may not contain object arrays")
    return arrays


def _sha256(payload: bytes) -> str:
    if type(payload) is not bytes:
        raise ValueError("hash input must be immutable bytes")
    return hashlib.sha256(payload).hexdigest()


def _canonical_digest(payload: Mapping[str, object]) -> str:
    try:
        encoded = json.dumps(
            dict(payload),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
            ensure_ascii=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("completion payload is not canonicalizable") from error
    return hashlib.sha256(encoded).hexdigest()


def _require_digest(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a canonical lowercase SHA-256 digest")
    return value


def _require_commit(value: object, *, name: str = "git_commit") -> str:
    if not isinstance(value, str) or _COMMIT_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a canonical lowercase Git commit")
    return value


def _require_run_id(value: object) -> str:
    if not isinstance(value, str) or _RUN_ID_RE.fullmatch(value) is None:
        raise ValueError("run_id must be a canonical path-safe identifier")
    return value


def _require_exact_int(value: object, expected: int, *, name: str) -> None:
    if type(value) is not int or value != expected:
        raise ValueError(f"{name} must be exactly {expected}")


def _role(scene: str) -> str:
    return "validation" if scene in _VALIDATION_SCENES else "train"


def _expect_array(
    arrays: Mapping[str, np.ndarray],
    name: str,
    *,
    shape: tuple[int, ...],
    dtype: str | type[np.generic],
) -> np.ndarray:
    value = arrays[name]
    expected = np.dtype(dtype)
    if value.shape != shape or value.dtype != expected:
        raise ValueError(
            f"{name} must have exact shape {shape} and dtype {expected}"
        )
    return value


def _array_text(
    arrays: Mapping[str, np.ndarray], name: str, *, width: int
) -> str:
    return str(_expect_array(arrays, name, shape=(), dtype=f"U{width}"))


def _array_digest(arrays: Mapping[str, np.ndarray], name: str) -> str:
    return _require_digest(_array_text(arrays, name, width=64), name=name)


def _finite(value: np.ndarray, *, name: str) -> None:
    if not np.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")


def _validate_pose_stack(value: np.ndarray, *, name: str) -> np.ndarray:
    poses = np.asarray(value)
    if (
        poses.ndim != 3
        or poses.shape[-2:] != (4, 4)
        or poses.dtype != np.float64
        or not np.isfinite(poses).all()
    ):
        raise ValueError(f"{name} must be a finite float64 pose stack")
    if not np.allclose(
        poses[:, 3, :],
        np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
        atol=1e-10,
        rtol=0.0,
    ):
        raise ValueError(f"{name} must contain homogeneous poses")
    rotations = poses[:, :3, :3]
    gram = np.einsum("...ji,...jk->...ik", rotations, rotations)
    determinant = np.linalg.det(rotations)
    if (
        not np.allclose(gram, np.eye(3), atol=2e-6, rtol=0.0)
        or not np.allclose(determinant, 1.0, atol=2e-6, rtol=0.0)
        or np.any(determinant <= 0.0)
    ):
        raise ValueError(f"{name} must contain proper SO(3) rotations")
    return poses


def _prediction_scale(poses: np.ndarray) -> float:
    validated = _validate_pose_stack(poses, name="pose scale input")
    centers = validated[:, :3, 3]
    centered = centers - centers.mean(axis=0)
    scale = float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))
    if not math.isfinite(scale) or scale <= 1e-12:
        raise ValueError("pose stack has insufficient prediction scale")
    return scale


def _frame_digest(frame_ids: np.ndarray) -> str:
    values = np.asarray(frame_ids)
    if (
        values.shape != (_FRAMES,)
        or values.dtype != np.int64
        or np.any(values[1:] <= values[:-1])
    ):
        raise ValueError("frame IDs must be exact strictly increasing int64 values")
    encoded = json.dumps(
        [int(value) for value in values], separators=(",", ":")
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _validate_identity_and_provenance(
    arrays: Mapping[str, np.ndarray],
    *,
    scene: str,
    sample_id: str,
    git_commit: str,
    checkpoint_sha256: str,
) -> None:
    if _array_text(arrays, "scene", width=32) != scene:
        raise ValueError("bundle scene mismatch")
    if _array_text(arrays, "sample_id", width=96) != sample_id:
        raise ValueError("bundle sample ID mismatch")
    if _array_text(arrays, "git_commit", width=40) != git_commit:
        raise ValueError("bundle Git commit mismatch")
    if _array_digest(arrays, "checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("bundle checkpoint digest mismatch")
    _array_digest(arrays, "source_sha256")


def _reject_symlink_components(path: Path) -> None:
    candidate = Path(path)
    if ".." in candidate.parts:
        raise ValueError("verified paths may not contain lexical parent traversal")
    absolute = Path(os.path.abspath(os.fspath(candidate)))
    for component in (absolute, *absolute.parents):
        if component.is_symlink():
            raise ValueError(f"verified paths may not contain symlinks: {component}")


def _snapshot_file(path: Path, *, label: str) -> bytes:
    source = Path(path)
    _reject_symlink_components(source)
    if not source.is_file():
        raise ValueError(f"{label} must be a regular file")
    try:
        return source.read_bytes()
    except OSError as error:
        raise ValueError(f"could not snapshot {label}") from error


def _strict_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


@dataclass(frozen=True)
class _UpstreamFileSnapshot:
    path: Path
    sha256: str
    identity: tuple[int, int, int, int, int]
    label: str
    payload: bytes | None


@dataclass(frozen=True)
class _DirectoryIdentity:
    path: Path
    identity: tuple[int, int]
    label: str


@dataclass(frozen=True)
class _FrozenScenePayload:
    long: Mapping[str, np.ndarray]
    teacher: Mapping[str, np.ndarray]
    formal: Mapping[str, np.ndarray]


@dataclass(frozen=True)
class _UpstreamAuthentication:
    files: tuple[_UpstreamFileSnapshot, ...]
    directories: tuple[_DirectoryIdentity, ...]
    scenes: Mapping[str, _FrozenScenePayload]


def _require_exact_array_values(
    left: np.ndarray,
    right: np.ndarray,
    *,
    name: str,
) -> None:
    first = np.asarray(left)
    second = np.asarray(right)
    equal = (
        np.array_equal(first, second, equal_nan=True)
        if np.issubdtype(first.dtype, np.floating)
        and np.issubdtype(second.dtype, np.floating)
        else np.array_equal(first, second)
    )
    if (
        first.shape != second.shape
        or first.dtype != second.dtype
        or not equal
    ):
        raise ValueError(f"{name} does not match frozen upstream values")


def _readonly_arrays(
    arrays: Mapping[str, np.ndarray],
) -> Mapping[str, np.ndarray]:
    frozen: dict[str, np.ndarray] = {}
    for name, value in arrays.items():
        item = np.asarray(value).copy()
        item.setflags(write=False)
        frozen[name] = item
    return frozen


def _validate_frozen_scene_payload(
    *,
    scene: str,
    long: Mapping[str, np.ndarray],
    teacher: Mapping[str, np.ndarray],
    formal: Mapping[str, np.ndarray],
    source_sha256: str,
    formal_sha256: str,
    checkpoint_sha256: str,
) -> _FrozenScenePayload:
    if _array_text(long, "scene", width=32) != scene:
        raise ValueError(f"frozen long scene mismatch for {scene}")
    long_frames = _expect_array(
        long, "frame_ids", shape=(_FRAMES,), dtype=np.int64
    )
    if np.any(long_frames[1:] <= long_frames[:-1]):
        raise ValueError(f"frozen long frame IDs are noncanonical for {scene}")
    long_tokens = _expect_array(
        long,
        "camera_tokens",
        shape=(_FRAMES, _TOKEN_WIDTH),
        dtype=np.float32,
    )
    long_c2w = _expect_array(
        long, "baseline_c2w", shape=(_FRAMES, 4, 4), dtype=np.float64
    )
    _finite(long_tokens, name=f"frozen long tokens {scene}")
    _validate_pose_stack(long_c2w, name=f"frozen long baseline {scene}")
    if _array_digest(long, "source_sha256") != source_sha256:
        raise ValueError(f"frozen long provenance mismatch for {scene}")

    if (
        _array_text(teacher, "scene", width=32) != scene
        or _array_text(teacher, "oracle_scene", width=32) != scene
    ):
        raise ValueError(f"frozen teacher scene mismatch for {scene}")
    teacher_frames = _expect_array(
        teacher, "frame_ids", shape=(_FRAMES,), dtype=np.int64
    )
    gt = _expect_array(
        teacher, "gt_c2w", shape=(_FRAMES, 4, 4), dtype=np.float64
    )
    gt_scale = _expect_array(
        teacher, "gt_scene_scale", shape=(), dtype=np.float64
    )
    teacher_baseline = _expect_array(
        teacher,
        "baseline_c2w_raw",
        shape=(_FRAMES, 4, 4),
        dtype=np.float64,
    )
    oracle_scale = _expect_array(
        teacher, "oracle_scale", shape=(), dtype=np.float64
    )
    oracle_rotation = _expect_array(
        teacher, "oracle_rotation", shape=(3, 3), dtype=np.float64
    )
    oracle_translation = _expect_array(
        teacher, "oracle_translation", shape=(3,), dtype=np.float64
    )
    _expect_array(teacher, "oracle_fit_count", shape=(), dtype=np.int64)
    _expect_array(teacher, "oracle_rank", shape=(), dtype=np.int64)
    oracle_condition = _expect_array(
        teacher, "oracle_condition", shape=(), dtype=np.float64
    )
    window_weights = _expect_array(
        teacher, "window_weights", shape=(_WINDOWS,), dtype=np.float64
    )
    window_masks = _expect_array(
        teacher,
        "window_masks",
        shape=(_ENDPOINTS, _WINDOWS),
        dtype=np.uint8,
    )
    coverage = _expect_array(
        teacher,
        "coverage_weights",
        shape=(_ENDPOINTS, _FRAMES),
        dtype=np.float64,
    )
    fused = _expect_array(
        teacher,
        "fused_c2w",
        shape=(_ENDPOINTS, _FRAMES, 4, 4),
        dtype=np.float64,
    )
    utilities = _expect_array(
        teacher,
        "variant_utilities",
        shape=(_ENDPOINTS,),
        dtype=np.float64,
    )
    _validate_pose_stack(gt, name=f"frozen teacher GT {scene}")
    _validate_pose_stack(
        teacher_baseline, name=f"frozen teacher baseline {scene}"
    )
    if (
        not np.isfinite(gt_scale).all()
        or float(gt_scale) <= 0.0
        or not math.isclose(
            float(gt_scale),
            _prediction_scale(gt),
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        or not np.isfinite(oracle_scale).all()
        or float(oracle_scale) <= 0.0
        or not np.isfinite(oracle_rotation).all()
        or not np.isfinite(oracle_translation).all()
        or not np.isfinite(oracle_condition).all()
        or not np.isfinite(window_weights).all()
        or not np.isfinite(coverage).all()
        or not np.isfinite(utilities).all()
        or np.any(window_weights < 0.0)
        or np.any(window_weights > 1.0)
        or np.any(coverage < 0.0)
        or not np.isin(window_masks, (0, 1)).all()
    ):
        raise ValueError(f"frozen teacher numeric payload is malformed for {scene}")
    oracle_pose = np.eye(4, dtype=np.float64)[None]
    oracle_pose[0, :3, :3] = oracle_rotation
    _validate_pose_stack(oracle_pose, name=f"frozen teacher oracle {scene}")
    oracle_payload = {
        "scene": scene,
        "frame_digest": _array_text(
            teacher, "oracle_frame_digest", width=64
        ),
        "fit_count": int(teacher["oracle_fit_count"]),
        "scale": float(oracle_scale),
        "rotation": tuple(
            tuple(float(value) for value in row) for row in oracle_rotation
        ),
        "translation": tuple(float(value) for value in oracle_translation),
    }
    if (
        oracle_payload["frame_digest"] != _frame_digest(teacher_frames)
        or _array_digest(teacher, "oracle_digest")
        != _canonical_digest(oracle_payload)
    ):
        raise ValueError(f"frozen teacher oracle witness mismatch for {scene}")
    covered = coverage > 0.0
    if (
        (np.any(covered) and not np.isfinite(fused[covered]).all())
        or (np.any(~covered) and not np.isnan(fused[~covered]).all())
    ):
        raise ValueError(f"frozen teacher fused poses are malformed for {scene}")
    if np.any(covered):
        _validate_pose_stack(
            fused[covered], name=f"frozen teacher covered witness {scene}"
        )
    if (
        _array_digest(teacher, "source_sha256") != source_sha256
        or _array_digest(teacher, "formal_label_sha256") != formal_sha256
        or _array_digest(teacher, "checkpoint_sha256") != checkpoint_sha256
    ):
        raise ValueError(f"frozen teacher provenance mismatch for {scene}")

    if _array_text(formal, "scene", width=32) != scene:
        raise ValueError(f"frozen formal scene mismatch for {scene}")
    formal_frames = _expect_array(
        formal, "frame_ids", shape=(_FRAMES,), dtype=np.int64
    )
    formal_gt = _expect_array(
        formal, "gt_c2w", shape=(_FRAMES, 4, 4), dtype=np.float64
    )
    formal_scale = _expect_array(
        formal, "gt_scene_scale", shape=(), dtype=np.float64
    )
    formal_oracle_scale = _expect_array(
        formal, "oracle_scale", shape=(), dtype=np.float64
    )
    formal_oracle_rotation = _expect_array(
        formal, "oracle_rotation", shape=(3, 3), dtype=np.float64
    )
    formal_oracle_translation = _expect_array(
        formal, "oracle_translation", shape=(3,), dtype=np.float64
    )
    formal_pose = _expect_array(
        formal,
        "baseline_pose_encoding",
        shape=(_FRAMES, 9),
        dtype=np.float32,
    )
    formal_teacher = _expect_array(
        formal,
        "teacher_c2w_gt_gauge",
        shape=(_FRAMES, 4, 4),
        dtype=np.float64,
    )
    formal_weight = _expect_array(
        formal, "teacher_weight", shape=(_FRAMES,), dtype=np.float64
    )
    formal_window_weight = _expect_array(
        formal,
        "window_teacher_weight",
        shape=(_WINDOWS,),
        dtype=np.float64,
    )
    formal_baseline_rms = _expect_array(
        formal, "window_baseline_rms", shape=(_WINDOWS,), dtype=np.float64
    )
    formal_teacher_rms = _expect_array(
        formal, "window_teacher_rms", shape=(_WINDOWS,), dtype=np.float64
    )
    _validate_pose_stack(formal_gt, name=f"frozen formal GT {scene}")
    if (
        not np.isfinite(formal_scale).all()
        or float(formal_scale) <= 0.0
        or not np.isfinite(formal_oracle_scale).all()
        or float(formal_oracle_scale) <= 0.0
        or not np.isfinite(formal_oracle_rotation).all()
        or not np.isfinite(formal_oracle_translation).all()
        or not np.isfinite(formal_pose).all()
        or not np.isfinite(formal_weight).all()
        or np.any(formal_weight < 0.0)
        or not np.isfinite(formal_window_weight).all()
        or not np.isfinite(formal_baseline_rms).all()
        or not np.isfinite(formal_teacher_rms).all()
    ):
        raise ValueError(f"frozen formal numeric payload is malformed for {scene}")
    formal_covered = formal_weight > 0.0
    if (
        (np.any(formal_covered) and not np.isfinite(formal_teacher[formal_covered]).all())
        or (
            np.any(~formal_covered)
            and not np.isnan(formal_teacher[~formal_covered]).all()
        )
    ):
        raise ValueError(f"frozen formal teacher witness is malformed for {scene}")
    if np.any(formal_covered):
        _validate_pose_stack(
            formal_teacher[formal_covered],
            name=f"frozen formal covered witness {scene}",
        )
    if (
        _array_digest(formal, "source_sha256") != source_sha256
        or _array_digest(formal, "checkpoint_sha256") != checkpoint_sha256
    ):
        raise ValueError(f"frozen formal provenance mismatch for {scene}")

    try:
        formal_decoded = (
            pose_encoding_to_c2w(torch.from_numpy(formal_pose[None]))
            .detach()
            .to(device="cpu", dtype=torch.float64)
            .numpy()[0]
        )
    except (RuntimeError, ValueError) as error:
        raise ValueError(
            f"frozen formal baseline pose conversion failed for {scene}"
        ) from error
    _require_cross_device_baseline_match(
        formal_decoded,
        long_c2w,
        scale=_prediction_scale(long_c2w),
    )

    for left, right, label in (
        (teacher_frames, long_frames, "teacher/long frame IDs"),
        (formal_frames, long_frames, "formal/long frame IDs"),
        (teacher_baseline, long_c2w, "teacher/long baseline"),
        (formal_gt, gt, "formal/teacher GT"),
        (formal_scale, gt_scale, "formal/teacher GT scale"),
        (formal_oracle_scale, oracle_scale, "formal/teacher oracle scale"),
        (
            formal_oracle_rotation,
            oracle_rotation,
            "formal/teacher oracle rotation",
        ),
        (
            formal_oracle_translation,
            oracle_translation,
            "formal/teacher oracle translation",
        ),
        (formal_window_weight, window_weights, "formal/teacher window weights"),
        (formal_weight, coverage[0], "formal/teacher variant-zero weights"),
        (formal_teacher, fused[0], "formal/teacher variant-zero poses"),
    ):
        _require_exact_array_values(left, right, name=f"{scene} frozen {label}")
    if _array_digest(formal, "oracle_digest") != _array_digest(
        teacher, "oracle_digest"
    ):
        raise ValueError(f"frozen formal/teacher oracle digest mismatch for {scene}")
    return _FrozenScenePayload(
        long=_readonly_arrays(long),
        teacher=_readonly_arrays(teacher),
        formal=_readonly_arrays(formal),
    )


def _hash_regular_file(path: Path, *, label: str) -> str:
    target = Path(path)
    _reject_symlink_components(target)
    if not target.is_file():
        raise ValueError(f"{label} must remain a regular file")
    digest = hashlib.sha256()
    try:
        with target.open("rb", buffering=0) as handle:
            while True:
                block = handle.read(1024 * 1024)
                if not block:
                    break
                digest.update(block)
    except OSError as error:
        raise ValueError(f"could not hash {label}") from error
    return digest.hexdigest()


def _snapshot_upstream_file(
    path: Path,
    expected_sha256: object,
    *,
    label: str,
    retain: bool = False,
) -> _UpstreamFileSnapshot:
    expected = _require_digest(expected_sha256, name=f"expected {label} digest")
    target = Path(path)
    _reject_symlink_components(target)
    if not target.is_file():
        raise ValueError(f"{label} must be a regular file")
    digest = hashlib.sha256()
    chunks: list[bytes] | None = [] if retain else None
    try:
        before = target.stat(follow_symlinks=False)
        with target.open("rb", buffering=0) as handle:
            opened_before = os.fstat(handle.fileno())
            while True:
                block = handle.read(1024 * 1024)
                if not block:
                    break
                digest.update(block)
                if chunks is not None:
                    chunks.append(block)
            opened_after = os.fstat(handle.fileno())
        after = target.stat(follow_symlinks=False)
    except OSError as error:
        raise ValueError(f"could not snapshot {label}") from error
    identities = {
        _strict_identity(value)
        for value in (before, opened_before, opened_after, after)
    }
    if len(identities) != 1:
        raise ValueError(f"{label} changed while taking its immutable snapshot")
    actual = digest.hexdigest()
    if actual != expected:
        raise ValueError(f"{label} digest mismatch")
    payload = None if chunks is None else b"".join(chunks)
    return _UpstreamFileSnapshot(
        path=target.resolve(),
        sha256=actual,
        identity=identities.pop(),
        label=label,
        payload=payload,
    )


def _snapshot_directory_identity(path: Path, *, label: str) -> _DirectoryIdentity:
    target = Path(path)
    _reject_symlink_components(target)
    if not target.is_dir():
        raise ValueError(f"{label} must be a directory")
    try:
        value = target.stat(follow_symlinks=False)
    except OSError as error:
        raise ValueError(f"could not snapshot {label}") from error
    return _DirectoryIdentity(
        path=target.resolve(),
        identity=(int(value.st_dev), int(value.st_ino)),
        label=label,
    )


def _require_upstream_unchanged(authentication: _UpstreamAuthentication) -> None:
    for directory in authentication.directories:
        _reject_symlink_components(directory.path)
        try:
            current = directory.path.stat(follow_symlinks=False)
        except OSError as error:
            raise ValueError(
                f"{directory.label} changed during upstream authentication"
            ) from error
        if (
            not directory.path.is_dir()
            or (int(current.st_dev), int(current.st_ino)) != directory.identity
        ):
            raise ValueError(
                f"{directory.label} changed during upstream authentication"
            )
    for snapshot in authentication.files:
        _reject_symlink_components(snapshot.path)
        try:
            current = snapshot.path.stat(follow_symlinks=False)
        except OSError as error:
            raise ValueError(
                f"{snapshot.label} changed during upstream authentication"
            ) from error
        if (
            _strict_identity(current) != snapshot.identity
            or _hash_regular_file(snapshot.path, label=snapshot.label)
            != snapshot.sha256
        ):
            raise ValueError(
                f"{snapshot.label} changed during upstream authentication"
            )


def _decode_upstream_json(
    snapshot: _UpstreamFileSnapshot,
    *,
    fields: frozenset[str],
) -> dict[str, object]:
    if snapshot.payload is None:
        raise ValueError(f"{snapshot.label} was not retained for parsing")
    try:
        value = json.loads(
            snapshot.payload.decode("utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid {snapshot.label} JSON snapshot") from error
    if not isinstance(value, dict) or set(value) != set(fields):
        raise ValueError(f"{snapshot.label} must use the exact JSON schema")
    return value


def _same_file_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        left.st_dev == right.st_dev
        and left.st_ino == right.st_ino
        and left.st_size == right.st_size
        and left.st_mtime_ns == right.st_mtime_ns
    )


@contextmanager
def _private_checkpoint_copy(
    checkpoint_file: Path, *, checkpoint_dir: Path
) -> Iterator[tuple[Path, str, tuple[os.stat_result, os.stat_result]]]:
    """Copy one checkpoint from a stable fd while hashing; never load its live path."""
    source = Path(checkpoint_file)
    directory = Path(checkpoint_dir)
    _reject_symlink_components(source)
    _reject_symlink_components(directory)
    if source.parent.resolve() != directory.resolve():
        raise ValueError("configured checkpoint must be directly inside checkpoint_dir")
    if not source.is_file():
        raise ValueError("configured checkpoint must be a regular file")
    temporary = tempfile.TemporaryDirectory(prefix="hvrfm-verifier-checkpoint-")
    try:
        private_dir = Path(temporary.name)
        private_file = private_dir / source.name
        digest = hashlib.sha256()
        try:
            with source.open("rb", buffering=0) as read_handle:
                opened_stat = os.fstat(read_handle.fileno())
                with private_file.open("xb", buffering=0) as write_handle:
                    while True:
                        block = read_handle.read(1024 * 1024)
                        if not block:
                            break
                        digest.update(block)
                        write_handle.write(block)
                    os.fsync(write_handle.fileno())
                closed_stat = os.fstat(read_handle.fileno())
        except OSError as error:
            raise ValueError("could not create authenticated private checkpoint copy") from error
        if not _same_file_identity(opened_stat, closed_stat):
            raise ValueError("checkpoint changed while creating private copy")
        if private_file.stat().st_size != opened_stat.st_size:
            raise ValueError("private checkpoint copy size mismatch")
        yield private_dir, digest.hexdigest(), (opened_stat, closed_stat)
    finally:
        temporary.cleanup()


def _rehash_checkpoint_identity(
    checkpoint_file: Path,
    *,
    expected_stat: os.stat_result,
    expected_sha256: str,
) -> None:
    source = Path(checkpoint_file)
    _reject_symlink_components(source)
    try:
        before = source.stat()
        digest = hashlib.sha256()
        with source.open("rb", buffering=0) as handle:
            opened = os.fstat(handle.fileno())
            while True:
                block = handle.read(1024 * 1024)
                if not block:
                    break
                digest.update(block)
            after = os.fstat(handle.fileno())
        final = source.stat()
    except OSError as error:
        raise ValueError("could not reauthenticate checkpoint after model load") from error
    if not all(
        _same_file_identity(expected_stat, candidate)
        for candidate in (before, opened, after, final)
    ):
        raise ValueError("checkpoint identity changed during verification")
    if digest.hexdigest() != expected_sha256:
        raise ValueError("checkpoint bytes changed during verification")


def _load_camera_head(checkpoint_dir: Path) -> tuple[nn.Module, str, torch.device]:
    """Load the native frozen Camera Head from an authenticated private copy."""
    from pre_experiments.long_short_camera_head.train import load_base_camera_head

    head, digest = load_base_camera_head(Path(checkpoint_dir))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return head.to(device).eval(), digest, device


def _model_tensor_digest(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for kind, values in (("parameter", model.named_parameters()), ("buffer", model.named_buffers())):
        for name, tensor in values:
            value = tensor.detach().to(device="cpu").contiguous()
            descriptor = (
                f"{kind}:{name}:{value.dtype}:{tuple(value.shape)}"
            ).encode("utf-8")
            digest.update(len(descriptor).to_bytes(8, byteorder="big"))
            digest.update(descriptor)
            digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _decode_tokens(
    camera_head: nn.Module,
    tokens: np.ndarray,
    *,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    if (
        not isinstance(camera_head, nn.Module)
        or not isinstance(device, torch.device)
        or tokens.dtype != np.float32
        or tokens.ndim != 3
        or tokens.shape[-1] != _TOKEN_WIDTH
        or not np.isfinite(tokens).all()
    ):
        raise ValueError("Camera Head decode inputs are malformed")
    tensors = list(camera_head.parameters()) + list(camera_head.buffers())
    if any(tensor.device.type != device.type for tensor in tensors):
        raise ValueError("Camera Head tensors are not on the declared device")
    before = _model_tensor_digest(camera_head)
    modes = [(module, module.training) for module in camera_head.modules()]
    try:
        camera_head.eval()
        with torch.no_grad():
            token_tensor = torch.from_numpy(tokens).to(device=device)
            trace = camera_head.decode_pose_tokens(
                token_tensor, num_iterations=4
            )
            if not isinstance(trace, (list, tuple)) or not trace:
                raise ValueError("Camera Head returned a malformed decode trace")
            raw = trace[-1]
            if (
                not isinstance(raw, torch.Tensor)
                or raw.shape != (*tokens.shape[:2], 9)
                or not torch.isfinite(raw).all()
            ):
                raise ValueError("Camera Head returned malformed pose encodings")
            c2w = pose_encoding_to_c2w(raw)
            raw_array = raw.detach().to(device="cpu", dtype=torch.float32).numpy()
            c2w_array = c2w.detach().to(device="cpu", dtype=torch.float64).numpy()
    finally:
        for module, training in modes:
            module.training = training
    after = _model_tensor_digest(camera_head)
    if before != after:
        raise ValueError("Camera Head parameters or buffers changed during replay")
    if not np.isfinite(raw_array).all() or not np.isfinite(c2w_array).all():
        raise ValueError("Camera Head replay produced non-finite values")
    return raw_array, c2w_array


def _artifact_relpaths(scene: str) -> dict[str, str]:
    return {
        "long": f"prediction_only/long_context/{scene}.npz",
        "short": f"privileged_training/short_context/{scene}.npz",
        "quality": f"privileged_labels/quality/{scene}.npz",
        "target": f"privileged_labels/translation_targets/{scene}.npz",
    }


def _expected_base_files() -> frozenset[str]:
    paths = {
        "config.json",
        "manifests/preflight_evidence.json",
        "manifests/long_context.json",
        "manifests/cohort.json",
        "prepare/completed.json",
        "smoke/completed.json",
        "calibration/completed.json",
        "reports/stage_a_prime.json",
        "reports/stage_a_prime.md",
        "reports/completed.json",
    }
    for scene in _SCENES:
        paths.update(_artifact_relpaths(scene).values())
    return frozenset(paths)


_BASE_FILES = _expected_base_files()
_INVENTORY_PATH = "manifests/verification_inventory.json"
_FINAL_PATH = "verified_completion.json"


def _expected_directories() -> frozenset[str]:
    directories: set[str] = set()
    for path in (*_BASE_FILES, _INVENTORY_PATH, _FINAL_PATH):
        parent = Path(path).parent
        while parent != Path("."):
            directories.add(parent.as_posix())
            parent = parent.parent
    return frozenset(directories)


_DIRECTORIES = _expected_directories()


def _directory_inode(path: Path, *, label: str) -> tuple[int, int]:
    target = Path(path)
    _reject_symlink_components(target)
    if not target.is_dir():
        raise ValueError(f"{label} must be a directory")
    try:
        value = target.stat(follow_symlinks=False)
    except OSError as error:
        raise ValueError(f"could not inspect {label}") from error
    return int(value.st_dev), int(value.st_ino)


def _snapshot_run(
    root: Path,
) -> tuple[dict[str, bytes], dict[str, bytes], dict[str, tuple[int, int]]]:
    _reject_symlink_components(root)
    if not root.is_dir():
        raise ValueError("completed run root must be an existing directory")
    files: dict[str, bytes] = {}
    optional: dict[str, bytes] = {}
    directories: set[str] = set()
    directory_identities = {
        "": _directory_inode(root, label="completed run root")
    }
    try:
        entries = list(root.rglob("*"))
    except OSError as error:
        raise ValueError("could not inventory completed run") from error
    for entry in entries:
        if entry.is_symlink():
            raise ValueError("completed run may not contain symlinks")
        relative = entry.relative_to(root).as_posix()
        if entry.is_dir():
            directories.add(relative)
            directory_identities[relative] = _directory_inode(
                entry, label=f"run directory {relative}"
            )
            continue
        if not entry.is_file():
            raise ValueError("completed run may contain only regular files")
        payload = _snapshot_file(entry, label=f"run file {relative}")
        if relative in {_INVENTORY_PATH, _FINAL_PATH}:
            optional[relative] = payload
        else:
            files[relative] = payload
    if set(files) != set(_BASE_FILES):
        missing = sorted(set(_BASE_FILES) - set(files))
        extra = sorted(set(files) - set(_BASE_FILES))
        raise ValueError(
            f"completed run has noncanonical file inventory; missing={missing}, extra={extra}"
        )
    if directories != set(_DIRECTORIES):
        missing = sorted(set(_DIRECTORIES) - directories)
        extra = sorted(directories - set(_DIRECTORIES))
        raise ValueError(
            f"completed run has noncanonical directory inventory; missing={missing}, extra={extra}"
        )
    if _FINAL_PATH in optional and _INVENTORY_PATH not in optional:
        raise ValueError("verified completion exists without verification inventory")
    return files, optional, directory_identities


def _require_exact_run_topology(
    root: Path,
    *,
    optional_files: frozenset[str],
    directory_identities: Mapping[str, tuple[int, int]],
    transient_files: frozenset[str] = frozenset(),
) -> None:
    if not optional_files <= {_INVENTORY_PATH, _FINAL_PATH}:
        raise ValueError("internal optional run inventory is invalid")
    if any(
        not relative
        or Path(relative).is_absolute()
        or ".." in Path(relative).parts
        for relative in transient_files
    ):
        raise ValueError("internal transient run inventory is invalid")
    expected_files = set(_BASE_FILES) | set(optional_files) | set(transient_files)
    expected_directories = set(_DIRECTORIES)
    actual_files: set[str] = set()
    actual_directories: set[str] = set()
    _reject_symlink_components(root)
    if not root.is_dir():
        raise ValueError("completed run root changed during verification")
    try:
        entries = list(root.rglob("*"))
    except OSError as error:
        raise ValueError("could not rescan completed run topology") from error
    for entry in entries:
        if entry.is_symlink():
            raise ValueError("completed run topology gained a symlink")
        relative = entry.relative_to(root).as_posix()
        if entry.is_dir():
            actual_directories.add(relative)
        elif entry.is_file():
            actual_files.add(relative)
        else:
            raise ValueError("completed run topology gained a special file")
    if actual_files != expected_files or actual_directories != expected_directories:
        raise ValueError("completed run topology changed during verification")
    expected_identity_keys = {"", *expected_directories}
    if set(directory_identities) != expected_identity_keys:
        raise ValueError("initial run directory identity snapshot is incomplete")
    for relative in sorted(expected_identity_keys):
        path = root if not relative else root / relative
        if _directory_inode(path, label=f"run directory {relative or '.'}") != tuple(
            directory_identities[relative]
        ):
            raise ValueError(
                f"run directory identity changed during verification: {relative or '.'}"
            )


def _require_snapshot_unchanged(root: Path, expected: Mapping[str, bytes]) -> None:
    for relative, payload in expected.items():
        if _snapshot_file(root / relative, label=f"revalidation {relative}") != payload:
            raise ValueError(f"run file changed during verification: {relative}")


def _validate_self_digest(payload: Mapping[str, object], *, label: str) -> None:
    unsigned = dict(payload)
    try:
        recorded = unsigned.pop("completion_digest")
    except KeyError as error:
        raise ValueError(f"{label} lacks completion_digest") from error
    if _require_digest(recorded, name=f"{label} completion digest") != _canonical_digest(
        unsigned
    ):
        raise ValueError(f"{label} completion digest mismatch")


def _exact_mapping(value: object, fields: frozenset[str], *, name: str) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != set(fields):
        raise ValueError(f"{name} must use the exact schema")
    return value


def _parse_metadata(
    snapshots: Mapping[str, bytes],
    *,
    expected_run_id: str,
    expected_git_commit: str,
) -> dict[str, object]:
    config = _decode_json_snapshot(
        snapshots["config.json"],
        label="run config",
        expected_fields=_CONFIG_FIELDS,
    )
    if config["schema"] != "camera_translation_hvrfm.run_config.v1":
        raise ValueError("run config schema mismatch")
    if _require_run_id(config["run_id"]) != expected_run_id:
        raise ValueError("run config run ID mismatch")
    if _require_commit(config["git_commit"]) != expected_git_commit:
        raise ValueError("run config Git commit mismatch")
    _require_exact_int(config["scene_count"], 10, name="scene_count")
    _require_exact_int(config["endpoint_count"], 40, name="endpoint_count")
    if config["smoke_scene"] != _SMOKE_SCENE:
        raise ValueError("run config smoke scene mismatch")
    for name in (
        "source_completion_sha256",
        "source_manifest_sha256",
        "reference_completion_sha256",
        "reference_inventory_sha256",
        "reference_long_manifest_sha256",
        "reference_teacher_manifest_sha256",
        "formal_completion_sha256",
        "formal_data_manifest_sha256",
        "checkpoint_sha256",
        "preflight_evidence_sha256",
        "long_context_manifest_sha256",
        "cohort_manifest_sha256",
    ):
        _require_digest(config[name], name=f"config {name}")
    digest_bindings = {
        "preflight_evidence_sha256": "manifests/preflight_evidence.json",
        "long_context_manifest_sha256": "manifests/long_context.json",
        "cohort_manifest_sha256": "manifests/cohort.json",
    }
    for name, relative in digest_bindings.items():
        if config[name] != _sha256(snapshots[relative]):
            raise ValueError(f"run config {name} does not bind actual bytes")

    preflight = _decode_json_snapshot(
        snapshots["manifests/preflight_evidence.json"],
        label="preflight evidence",
        expected_fields=_PREFLIGHT_FIELDS,
    )
    if (
        preflight["schema"] != "camera_translation_hvrfm.preflight_evidence.v1"
        or preflight["stage"] != "preflight"
        or preflight["run_id"] != expected_run_id
        or preflight["git_commit"] != expected_git_commit
    ):
        raise ValueError("preflight identity or schema mismatch")
    _validate_self_digest(preflight, label="preflight evidence")
    shared_preflight = (
        "source_run",
        "source_completion_sha256",
        "source_manifest_sha256",
        "reference_run",
        "reference_completion_sha256",
        "reference_inventory_sha256",
        "reference_long_manifest_sha256",
        "reference_teacher_manifest_sha256",
        "formal_run",
        "formal_completion_sha256",
        "formal_data_manifest_sha256",
        "checkpoint_file",
        "checkpoint_sha256",
    )
    for name in shared_preflight:
        if preflight[name] != config[name]:
            raise ValueError(f"preflight/config binding mismatch: {name}")
    for name in (
        "reference_config_sha256",
        "reference_report_json_sha256",
        "reference_report_markdown_sha256",
    ):
        _require_digest(preflight[name], name=f"preflight {name}")

    long_manifest = _decode_json_snapshot(
        snapshots["manifests/long_context.json"],
        label="long-context manifest",
        expected_fields=_LONG_MANIFEST_FIELDS,
    )
    cohort_manifest = _decode_json_snapshot(
        snapshots["manifests/cohort.json"],
        label="cohort manifest",
        expected_fields=_COHORT_MANIFEST_FIELDS,
    )
    for manifest, schema, label in (
        (
            long_manifest,
            "camera_translation_hvrfm.long_context_manifest.v1",
            "long-context manifest",
        ),
        (
            cohort_manifest,
            "camera_translation_hvrfm.cohort_manifest.v1",
            "cohort manifest",
        ),
    ):
        if (
            manifest["schema"] != schema
            or manifest["run_id"] != expected_run_id
            or manifest["git_commit"] != expected_git_commit
        ):
            raise ValueError(f"{label} identity or schema mismatch")

    return {
        "config": config,
        "preflight": preflight,
        "long_manifest": long_manifest,
        "cohort_manifest": cohort_manifest,
    }


def _same_resolved_path(value: object, expected: Path) -> bool:
    if not isinstance(value, str) or not value:
        return False
    try:
        candidate = Path(value)
        _reject_symlink_components(candidate)
        return candidate.resolve() == Path(expected).resolve()
    except (OSError, ValueError):
        return False


def _upstream_rows_by_scene(
    value: object,
    *,
    fields: frozenset[str],
    label: str,
) -> dict[str, dict[str, object]]:
    if not isinstance(value, list) or len(value) != len(_SCENES):
        raise ValueError(f"{label} must contain exactly ten records")
    result: dict[str, dict[str, object]] = {}
    for row in value:
        if not isinstance(row, dict) or set(row) != set(fields):
            raise ValueError(f"{label} record schema is not exact")
        scene = row.get("scene")
        if not isinstance(scene, str) or scene not in _SCENES or scene in result:
            raise ValueError(f"{label} scene cohort is invalid")
        if row.get("role") != _role(scene):
            raise ValueError(f"{label} role does not match the frozen split")
        result[scene] = dict(row)
    if set(result) != set(_SCENES):
        raise ValueError(f"{label} scene cohort is incomplete")
    return result


def _expected_reference_files() -> frozenset[str]:
    files = {
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
    for scene in _SCENES:
        files.update(
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
    files.add("smoke/latent_targets/scene0000_00.npz")
    files.update(
        f"smoke/checkpoints/scene0000_00/variant_{variant}.pt"
        for variant in range(4)
    )
    if len(files) != 87:  # pragma: no cover - frozen constant guard.
        raise RuntimeError("reference inventory constant is inconsistent")
    return frozenset(files)


def _authenticate_frozen_upstream(
    metadata: Mapping[str, object],
    *,
    checkpoint_dir: Path,
) -> _UpstreamAuthentication:
    config = metadata.get("config")
    preflight = metadata.get("preflight")
    if not isinstance(config, dict) or not isinstance(preflight, dict):
        raise ValueError("upstream metadata is malformed")
    frozen = {
        "source_completion_sha256": _FROZEN_SOURCE_COMPLETION_SHA256,
        "source_manifest_sha256": _FROZEN_SOURCE_MANIFEST_SHA256,
        "reference_completion_sha256": _FROZEN_REFERENCE_COMPLETION_SHA256,
        "reference_long_manifest_sha256": _FROZEN_REFERENCE_LONG_MANIFEST_SHA256,
        "formal_completion_sha256": _FROZEN_FORMAL_COMPLETION_SHA256,
        "formal_data_manifest_sha256": _FROZEN_FORMAL_DATA_MANIFEST_SHA256,
        "checkpoint_sha256": _FROZEN_CHECKPOINT_SHA256,
    }
    for name, expected in frozen.items():
        if config.get(name) != expected or preflight.get(name) != expected:
            raise ValueError(f"run does not bind frozen upstream {name}")

    roots: dict[str, Path] = {}
    for field, basename in (
        ("source_run", _FROZEN_SOURCE_RUN_NAME),
        ("reference_run", _FROZEN_REFERENCE_RUN_NAME),
        ("formal_run", _FROZEN_FORMAL_RUN_NAME),
    ):
        value = config.get(field)
        if not isinstance(value, str):
            raise ValueError(f"frozen upstream {field} path is malformed")
        path = Path(value)
        _reject_symlink_components(path)
        if path.name != basename or not path.is_dir():
            raise ValueError(f"{field} is not the frozen upstream run")
        roots[field] = path.resolve()
    if len(set(roots.values())) != 3:
        raise ValueError("frozen upstream roots must be physically distinct")
    checkpoint_root = Path(checkpoint_dir)
    _reject_symlink_components(checkpoint_root)
    if not checkpoint_root.is_dir():
        raise ValueError("checkpoint directory is unavailable")

    files: list[_UpstreamFileSnapshot] = []
    directories = tuple(
        _snapshot_directory_identity(path, label=f"frozen {name}")
        for name, path in roots.items()
    )

    def take(
        path: Path,
        expected: object,
        *,
        label: str,
        retain: bool = False,
    ) -> _UpstreamFileSnapshot:
        snapshot = _snapshot_upstream_file(
            path, expected, label=label, retain=retain
        )
        files.append(snapshot)
        return snapshot

    source_root = roots["source_run"]
    source_completion_snapshot = take(
        source_root / "verified_completion.json",
        _FROZEN_SOURCE_COMPLETION_SHA256,
        label="frozen source completion",
        retain=True,
    )
    source_completion = _decode_upstream_json(
        source_completion_snapshot, fields=_SOURCE_COMPLETION_FIELDS
    )
    _validate_self_digest(source_completion, label="frozen source completion")
    if (
        source_completion.get("schema")
        != "variational_camera_latent.verified_completion.v1"
        or source_completion.get("signal") != "WEAK_SIGNAL"
        or source_completion.get("scene_count") != 10
        or source_completion.get("overlap_count") != 80
        or source_completion.get("candidate_count") != 2560
    ):
        raise ValueError("frozen source completion identity mismatch")
    for name in (
        "prediction_manifest_sha256",
        "privileged_manifest_sha256",
        "report_sha256",
    ):
        _require_digest(source_completion.get(name), name=f"source {name}")

    source_manifest_snapshot = take(
        source_root / "manifests/source_manifest.json",
        _FROZEN_SOURCE_MANIFEST_SHA256,
        label="frozen source manifest",
        retain=True,
    )
    source_manifest = _decode_upstream_json(
        source_manifest_snapshot, fields=_SOURCE_MANIFEST_FIELDS
    )
    if source_manifest.get("schema") != "variational_camera_latent.source.v1":
        raise ValueError("frozen source manifest schema mismatch")
    _require_digest(source_manifest.get("source_run_digest"), name="source run digest")
    source_rows = _upstream_rows_by_scene(
        source_manifest.get("records"),
        fields=_SOURCE_RECORD_FIELDS,
        label="frozen source manifest",
    )
    historic_root = str(source_manifest.get("dataset_root", "")).replace(
        "\\", "/"
    ).rstrip("/")
    if not historic_root:
        raise ValueError("source manifest historical root is missing")
    source_scene_snapshots: dict[str, _UpstreamFileSnapshot] = {}
    for scene in _SCENES:
        row = source_rows[scene]
        if (
            str(row["path"]).replace("\\", "/")
            != f"{historic_root}/{scene}.npz"
            or row["overlap_count"] != 8
        ):
            raise ValueError("source historical path/overlap binding mismatch")
        source_scene_snapshots[scene] = take(
            source_root / "prediction_only/source" / f"{scene}.npz",
            row["sha256"],
            label=f"frozen source shard {scene}",
        )

    reference_root = roots["reference_run"]
    reference_completion_snapshot = take(
        reference_root / "verified_completion.json",
        _FROZEN_REFERENCE_COMPLETION_SHA256,
        label="frozen reference completion",
        retain=True,
    )
    reference_completion = _decode_upstream_json(
        reference_completion_snapshot, fields=_REFERENCE_COMPLETION_FIELDS
    )
    _validate_self_digest(reference_completion, label="frozen reference completion")
    if (
        reference_completion.get("schema")
        != "conditional_hierarchical_vrfm.verified_completion.v1"
        or reference_completion.get("git_commit") != _FROZEN_REFERENCE_GIT
        or reference_completion.get("classification") != "LATENT_LIFT_FAILED"
        or reference_completion.get("file_count") != 87
    ):
        raise ValueError("frozen reference completion identity mismatch")
    inventory_digest = _require_digest(
        reference_completion.get("inventory_sha256"),
        name="reference inventory binding",
    )
    if inventory_digest != preflight.get("reference_inventory_sha256"):
        raise ValueError("reference inventory/preflight binding mismatch")
    inventory_snapshot = take(
        reference_root / "manifests/verification_inventory.json",
        inventory_digest,
        label="frozen reference inventory",
        retain=True,
    )
    inventory = _decode_upstream_json(
        inventory_snapshot, fields=_REFERENCE_INVENTORY_FIELDS
    )
    inventory_files = inventory.get("files")
    if (
        inventory.get("schema")
        != "conditional_hierarchical_vrfm.verification_inventory.v1"
        or inventory.get("git_commit") != _FROZEN_REFERENCE_GIT
        or inventory.get("classification") != "LATENT_LIFT_FAILED"
        or not isinstance(inventory_files, dict)
        or set(inventory_files) != set(_expected_reference_files())
    ):
        raise ValueError("reference inventory does not bind the exact 87-file cohort")
    retained_reference = {
        "config.json",
        "manifests/long_context.json",
        "manifests/teacher.json",
        "reports/stage_a.json",
    }
    reference_files: dict[str, _UpstreamFileSnapshot] = {}
    for relative in sorted(_expected_reference_files()):
        reference_files[relative] = take(
            reference_root / relative,
            inventory_files[relative],
            label=f"frozen reference artifact {relative}",
            retain=(
                relative in retained_reference
                or relative.startswith("prediction_only/long_context/")
                or relative.startswith("privileged_labels/teacher/")
            ),
        )
    if (
        reference_files["config.json"].sha256
        != preflight.get("reference_config_sha256")
        or reference_files["reports/stage_a.json"].sha256
        != preflight.get("reference_report_json_sha256")
        or reference_files["reports/stage_a.md"].sha256
        != preflight.get("reference_report_markdown_sha256")
    ):
        raise ValueError("reference live report/config digest binding mismatch")

    reference_config = _decode_upstream_json(
        reference_files["config.json"], fields=_REFERENCE_CONFIG_FIELDS
    )
    reference_long = _decode_upstream_json(
        reference_files["manifests/long_context.json"],
        fields=_REFERENCE_LONG_FIELDS,
    )
    reference_teacher = _decode_upstream_json(
        reference_files["manifests/teacher.json"],
        fields=_REFERENCE_TEACHER_FIELDS,
    )
    reference_report = _decode_upstream_json(
        reference_files["reports/stage_a.json"],
        fields=_REFERENCE_REPORT_FIELDS,
    )
    long_digest = reference_files["manifests/long_context.json"].sha256
    teacher_digest = reference_files["manifests/teacher.json"].sha256
    if (
        long_digest != _FROZEN_REFERENCE_LONG_MANIFEST_SHA256
        or long_digest != preflight.get("reference_long_manifest_sha256")
        or teacher_digest != preflight.get("reference_teacher_manifest_sha256")
    ):
        raise ValueError("reference manifest digest binding mismatch")
    if (
        reference_config.get("schema")
        != "conditional_hierarchical_vrfm.run_config.v1"
        or reference_config.get("git_commit") != _FROZEN_REFERENCE_GIT
        or reference_config.get("checkpoint_sha256") != _FROZEN_CHECKPOINT_SHA256
        or reference_config.get("basis_sha256") != _FROZEN_BASIS_SHA256
        or reference_config.get("long_manifest_sha256") != long_digest
        or reference_config.get("teacher_manifest_sha256") != teacher_digest
        or reference_config.get("source_manifest_sha256")
        != source_manifest_snapshot.sha256
        or reference_config.get("formal_completion_sha256")
        != _FROZEN_FORMAL_COMPLETION_SHA256
        or reference_config.get("formal_data_manifest_sha256")
        != _FROZEN_FORMAL_DATA_MANIFEST_SHA256
        or not _same_resolved_path(reference_config.get("source_run"), source_root)
        or not _same_resolved_path(
            reference_config.get("formal_run_root"), roots["formal_run"]
        )
        or reference_config.get("smoke_scene") != "scene0000_00"
        or reference_config.get("smoke_steps") != 20
        or reference_config.get("calibration_steps") != 250
        or reference_config.get("scene_count") != 10
        or reference_config.get("variant_count") != 4
    ):
        raise ValueError("reference config provenance binding mismatch")
    if (
        reference_long.get("schema")
        != "conditional_hierarchical_vrfm.long_context_manifest.v1"
    ):
        raise ValueError("reference long-context manifest schema mismatch")
    long_rows = _upstream_rows_by_scene(
        reference_long.get("records"),
        fields=_REFERENCE_LONG_RECORD_FIELDS,
        label="reference long-context manifest",
    )
    teacher_rows = _upstream_rows_by_scene(
        reference_teacher.get("records"),
        fields=_REFERENCE_TEACHER_RECORD_FIELDS,
        label="reference teacher manifest",
    )
    if (
        reference_teacher.get("schema")
        != "conditional_hierarchical_vrfm.teacher_manifest.v1"
        or reference_teacher.get("git_commit") != _FROZEN_REFERENCE_GIT
        or reference_teacher.get("checkpoint_sha256")
        != _FROZEN_CHECKPOINT_SHA256
        or reference_teacher.get("formal_completion_sha256")
        != _FROZEN_FORMAL_COMPLETION_SHA256
        or reference_teacher.get("formal_data_manifest_sha256")
        != _FROZEN_FORMAL_DATA_MANIFEST_SHA256
        or reference_teacher.get("teacher_upper_bound")
        != {
            "scene_count": 10,
            "positive_scene_count": 10,
            "mean_coverage": 0.89,
            "mean_utility": 0.1293578270771188,
        }
    ):
        raise ValueError("reference teacher manifest provenance mismatch")
    report_rows = reference_report.get("scene_metrics")
    if (
        reference_report.get("schema")
        != "conditional_hierarchical_vrfm.stage_a_report.v1"
        or reference_report.get("git_commit") != _FROZEN_REFERENCE_GIT
        or reference_report.get("classification") != "LATENT_LIFT_FAILED"
        or reference_report.get("failed_gates")
        != [
            "teacher_retention",
            "per_scene_harm",
            "rotation_guard",
            "uncovered_anchor",
        ]
        or reference_report.get("provenance")
        != {
            "checkpoint_sha256": _FROZEN_CHECKPOINT_SHA256,
            "basis_sha256": _FROZEN_BASIS_SHA256,
            "long_manifest_sha256": long_digest,
            "teacher_manifest_sha256": teacher_digest,
        }
        or not isinstance(report_rows, list)
        or len(report_rows) != 10
        or {
            row.get("scene")
            for row in report_rows
            if isinstance(row, dict)
        }
        != set(_SCENES)
    ):
        raise ValueError("reference report provenance mismatch")

    formal_root = roots["formal_run"]
    formal_completion_snapshot = take(
        formal_root / "verified_completion.json",
        _FROZEN_FORMAL_COMPLETION_SHA256,
        label="frozen formal completion",
        retain=True,
    )
    formal_manifest_snapshot = take(
        formal_root / "manifests/data_manifest.json",
        _FROZEN_FORMAL_DATA_MANIFEST_SHA256,
        label="frozen formal data manifest",
        retain=True,
    )
    formal_completion = _decode_upstream_json(
        formal_completion_snapshot, fields=_FORMAL_COMPLETION_FIELDS
    )
    formal_manifest = _decode_upstream_json(
        formal_manifest_snapshot, fields=_FORMAL_MANIFEST_FIELDS
    )
    if (
        formal_completion.get("schema")
        != "long_short_camera_head.verified_completion.v1"
        or formal_completion.get("git_revision") != _FROZEN_FORMAL_GIT
        or not isinstance(formal_completion.get("verifier_git_revision"), str)
        or _COMMIT_RE.fullmatch(str(formal_completion["verifier_git_revision"]))
        is None
        or formal_completion.get("source_manifest_sha256")
        != source_manifest_snapshot.sha256
        or formal_completion.get("base_checkpoint_sha256")
        != _FROZEN_CHECKPOINT_SHA256
        or formal_completion.get("data_manifest_sha256")
        != formal_manifest_snapshot.sha256
        or formal_completion.get("scene_count") != 10
        or formal_completion.get("train_scene_count") != 8
        or formal_completion.get("locked_replay_scene_count") != 2
        or formal_completion.get("classification") != "NO_SOURCE_HEAD_SIGNAL"
        or formal_completion.get("inference_leakage_audit") is not True
    ):
        raise ValueError("formal completion provenance mismatch")
    for name in (
        "config_sha256",
        "test_evidence_sha256",
        "report_sha256",
        "formal_protocol_sha256",
    ):
        _require_digest(formal_completion.get(name), name=f"formal {name}")
    stages = formal_completion.get("stage_completion_sha256")
    if not isinstance(stages, dict) or set(stages) != {
        "evaluation_gt_only",
        "evaluation_long_short",
        "smoke",
        "training_gt_only",
        "training_long_short",
    }:
        raise ValueError("formal completion stage bindings are not exact")
    for name, digest in stages.items():
        _require_digest(digest, name=f"formal stage {name}")
    artifacts = formal_completion.get("artifacts")
    artifact_fields = {
        "scene",
        "variant",
        "checkpoint_sha256",
        "prediction_sha256",
        "evaluation_sha256",
    }
    if not isinstance(artifacts, list) or len(artifacts) != 20:
        raise ValueError("formal completion artifact cohort is not exact")
    artifact_identities: set[tuple[object, object]] = set()
    for row in artifacts:
        if not isinstance(row, dict) or set(row) != artifact_fields:
            raise ValueError("formal completion artifact schema is not exact")
        identity = (row.get("scene"), row.get("variant"))
        if (
            identity[0] not in _SCENES
            or identity[1] not in {"gt_only", "long_short"}
            or identity in artifact_identities
        ):
            raise ValueError("formal completion artifact identity is invalid")
        artifact_identities.add(identity)
        for name in (
            "checkpoint_sha256",
            "prediction_sha256",
            "evaluation_sha256",
        ):
            _require_digest(row.get(name), name=f"formal artifact {name}")
    if (
        formal_manifest.get("schema")
        != "long_short_camera_head.data_manifest.v1"
        or formal_manifest.get("git_revision") != _FROZEN_FORMAL_GIT
        or formal_manifest.get("source_manifest_sha256")
        != source_manifest_snapshot.sha256
        or formal_manifest.get("base_checkpoint_sha256")
        != _FROZEN_CHECKPOINT_SHA256
        or not _same_resolved_path(formal_manifest.get("source_run"), source_root)
        or not _same_resolved_path(
            formal_manifest.get("checkpoint_dir"), checkpoint_root
        )
    ):
        raise ValueError("formal data manifest provenance mismatch")
    formal_rows = _upstream_rows_by_scene(
        formal_manifest.get("records"),
        fields=_FORMAL_RECORD_FIELDS,
        label="formal data manifest",
    )

    bindings = preflight.get("scene_bindings")
    if not isinstance(bindings, list) or len(bindings) != 10:
        raise ValueError("preflight scene bindings are malformed")
    frozen_scenes: dict[str, _FrozenScenePayload] = {}
    for index, scene in enumerate(_SCENES):
        source_row = source_rows[scene]
        long_row = long_rows[scene]
        teacher_row = teacher_rows[scene]
        formal_row = formal_rows[scene]
        binding = bindings[index]
        if not isinstance(binding, dict):
            raise ValueError("preflight scene binding is malformed")
        source_digest = source_scene_snapshots[scene].sha256
        long_relative = f"prediction_only/long_context/{scene}.npz"
        long_snapshot = reference_files[long_relative]
        teacher_relative = f"privileged_labels/teacher/{scene}.npz"
        teacher_snapshot = reference_files[teacher_relative]
        formal_path = formal_root / "data/privileged_labels" / f"{scene}.npz"
        formal_snapshot = take(
            formal_path,
            formal_row["privileged_sha256"],
            label=f"frozen formal label {scene}",
            retain=True,
        )
        if (
            long_row.get("file") != f"{scene}.npz"
            or long_row.get("source_sha256") != source_digest
            or long_row.get("sha256") != long_snapshot.sha256
            or teacher_row.get("file") != teacher_relative
            or teacher_row.get("sha256") != teacher_snapshot.sha256
            or teacher_row.get("formal_label_sha256") != formal_snapshot.sha256
            or formal_row.get("source_sha256") != source_digest
            or formal_row.get("long_context_sha256") != long_snapshot.sha256
            or formal_row.get("privileged_sha256") != formal_snapshot.sha256
            or not _same_resolved_path(
                formal_row.get("source_path"),
                source_root / "prediction_only/source" / f"{scene}.npz",
            )
            or not _same_resolved_path(
                formal_row.get("long_context_path"),
                formal_root / "data/long_context" / f"{scene}.npz",
            )
            or not _same_resolved_path(
                formal_row.get("privileged_path"), formal_path
            )
            or type(formal_row.get("teacher_frame_count")) is not int
            or int(formal_row["teacher_frame_count"]) <= 0
        ):
            raise ValueError(f"upstream per-scene provenance mismatch for {scene}")
        expected_binding = {
            "scene": scene,
            "role": _role(scene),
            "source_sha256": source_digest,
            "long_context_sha256": long_snapshot.sha256,
            "teacher_reference_sha256": teacher_snapshot.sha256,
            "formal_label_sha256": formal_snapshot.sha256,
        }
        if binding != expected_binding:
            raise ValueError(f"live upstream binding mismatch for {scene}")

        if (
            long_snapshot.payload is None
            or teacher_snapshot.payload is None
            or formal_snapshot.payload is None
        ):
            raise ValueError(f"frozen numeric snapshots were not retained for {scene}")
        frozen_scene = _validate_frozen_scene_payload(
            scene=scene,
            long=_decode_npz_snapshot(
                long_snapshot.payload,
                label=f"frozen long context {scene}",
                expected_members=_REFERENCE_LONG_MEMBERS,
            ),
            teacher=_decode_npz_snapshot(
                teacher_snapshot.payload,
                label=f"frozen teacher reference {scene}",
                expected_members=_REFERENCE_TEACHER_MEMBERS,
            ),
            formal=_decode_npz_snapshot(
                formal_snapshot.payload,
                label=f"frozen formal label {scene}",
                expected_members=_FORMAL_LABEL_MEMBERS,
            ),
            source_sha256=source_digest,
            formal_sha256=formal_snapshot.sha256,
            checkpoint_sha256=_FROZEN_CHECKPOINT_SHA256,
        )
        if int(np.count_nonzero(frozen_scene.formal["teacher_weight"])) != int(
            formal_row["teacher_frame_count"]
        ):
            raise ValueError(
                f"frozen formal teacher-frame count mismatch for {scene}"
            )
        frozen_scenes[scene] = frozen_scene

    authentication = _UpstreamAuthentication(
        files=tuple(files), directories=directories, scenes=frozen_scenes
    )
    _require_upstream_unchanged(authentication)
    return authentication


def _validate_manifests_and_chain(
    snapshots: Mapping[str, bytes],
    metadata: Mapping[str, object],
    *,
    expected_run_id: str,
    expected_git_commit: str,
) -> tuple[list[dict[str, object]], dict[str, object], dict[str, object]]:
    config = metadata["config"]
    preflight = metadata["preflight"]
    long_manifest = metadata["long_manifest"]
    cohort_manifest = metadata["cohort_manifest"]
    if not all(isinstance(value, dict) for value in metadata.values()):
        raise ValueError("parsed metadata is malformed")

    long_records_value = long_manifest["records"]
    cohort_records_value = cohort_manifest["records"]
    bindings_value = preflight["scene_bindings"]
    if not all(isinstance(value, list) for value in (long_records_value, cohort_records_value, bindings_value)):
        raise ValueError("manifest records and scene bindings must be lists")
    if not all(len(value) == len(_SCENES) for value in (long_records_value, cohort_records_value, bindings_value)):
        raise ValueError("manifests must contain exactly ten scene records")

    cohort_records: list[dict[str, object]] = []
    prepare_files: dict[str, str] = {
        "config.json": _sha256(snapshots["config.json"]),
        "manifests/long_context.json": _sha256(
            snapshots["manifests/long_context.json"]
        ),
        "manifests/cohort.json": _sha256(snapshots["manifests/cohort.json"]),
    }
    for index, scene in enumerate(_SCENES):
        role = _role(scene)
        sample_id = f"{scene}:frames_500"
        paths = _artifact_relpaths(scene)
        long_row = _exact_mapping(
            long_records_value[index],
            frozenset({"sample_id", "scene", "role", "path", "sha256", "source_sha256"}),
            name="long-context manifest record",
        )
        expected_long = {
            "sample_id": sample_id,
            "scene": scene,
            "role": role,
            "path": paths["long"],
        }
        for name, expected in expected_long.items():
            if long_row[name] != expected:
                raise ValueError(f"long-context manifest record mismatch: {name}")
        long_digest = _require_digest(long_row["sha256"], name="manifest long digest")
        source_digest = _require_digest(
            long_row["source_sha256"], name="manifest source digest"
        )
        if long_digest != _sha256(snapshots[paths["long"]]):
            raise ValueError("long-context manifest digest mismatch")

        cohort_row = _exact_mapping(
            cohort_records_value[index],
            frozenset(
                {
                    "sample_id",
                    "scene",
                    "role",
                    "long_path",
                    "short_path",
                    "quality_path",
                    "target_path",
                    "long_sha256",
                    "short_sha256",
                    "quality_sha256",
                    "target_sha256",
                }
            ),
            name="cohort manifest record",
        )
        for name, expected in {
            "sample_id": sample_id,
            "scene": scene,
            "role": role,
            "long_path": paths["long"],
            "short_path": paths["short"],
            "quality_path": paths["quality"],
            "target_path": paths["target"],
        }.items():
            if cohort_row[name] != expected:
                raise ValueError(f"cohort manifest record mismatch: {name}")
        for kind in ("long", "short", "quality", "target"):
            digest = _require_digest(
                cohort_row[f"{kind}_sha256"],
                name=f"cohort {kind} digest",
            )
            if digest != _sha256(snapshots[paths[kind]]):
                raise ValueError(f"cohort {kind} digest mismatch")
            prepare_files[paths[kind]] = digest
        if cohort_row["long_sha256"] != long_digest:
            raise ValueError("long and cohort manifests disagree")

        binding = _exact_mapping(
            bindings_value[index],
            frozenset(
                {
                    "scene",
                    "role",
                    "source_sha256",
                    "long_context_sha256",
                    "teacher_reference_sha256",
                    "formal_label_sha256",
                }
            ),
            name="preflight scene binding",
        )
        if binding["scene"] != scene or binding["role"] != role:
            raise ValueError("preflight scene binding order or role mismatch")
        if _require_digest(binding["source_sha256"], name="binding source digest") != source_digest:
            raise ValueError("preflight source digest mismatch")
        for name in (
            "long_context_sha256",
            "teacher_reference_sha256",
            "formal_label_sha256",
        ):
            _require_digest(binding[name], name=f"binding {name}")
        cohort_records.append(cohort_row)

    def load_stage(relative: str, *, schema: str, stage: str) -> dict[str, object]:
        value = _decode_json_snapshot(
            snapshots[relative], label=f"{stage} completion", expected_fields=_STAGE_FIELDS
        )
        if (
            value["schema"] != schema
            or value["stage"] != stage
            or value["run_id"] != expected_run_id
            or value["git_commit"] != expected_git_commit
            or value["run_config_sha256"] != _sha256(snapshots["config.json"])
        ):
            raise ValueError(f"{stage} completion identity or schema mismatch")
        _validate_self_digest(value, label=f"{stage} completion")
        return value

    prepare = load_stage(
        "prepare/completed.json",
        schema="camera_translation_hvrfm.prepare_completion.v1",
        stage="prepare",
    )
    if prepare["previous_marker_sha256"] != _sha256(
        snapshots["manifests/preflight_evidence.json"]
    ):
        raise ValueError("prepare marker predecessor mismatch")
    if prepare["files"] != prepare_files:
        raise ValueError("prepare marker does not bind the exact published files")
    if prepare["metadata"] != {
        "scene_count": 10,
        "endpoint_count": 40,
        "smoke_scene": _SMOKE_SCENE,
    }:
        raise ValueError("prepare marker metadata mismatch")

    smoke = load_stage(
        "smoke/completed.json",
        schema="camera_translation_hvrfm.smoke_completion.v1",
        stage="smoke",
    )
    if smoke["previous_marker_sha256"] != _sha256(
        snapshots["prepare/completed.json"]
    ):
        raise ValueError("smoke marker predecessor mismatch")
    if smoke["files"] != {} or smoke["metadata"] != {
        "scene": _SMOKE_SCENE,
        "endpoint_count": 4,
        "classification": _READY,
    }:
        raise ValueError("smoke marker payload mismatch")

    calibration = load_stage(
        "calibration/completed.json",
        schema="camera_translation_hvrfm.calibration_completion.v1",
        stage="calibration",
    )
    if calibration["previous_marker_sha256"] != _sha256(
        snapshots["smoke/completed.json"]
    ):
        raise ValueError("calibration marker predecessor mismatch")
    if calibration["files"] != {} or calibration["metadata"] != {
        "scene_count": 10,
        "endpoint_count": 40,
        "classification": _READY,
    }:
        raise ValueError("calibration marker payload mismatch")

    report = _decode_json_snapshot(
        snapshots["reports/stage_a_prime.json"],
        label="Stage A-prime report",
        expected_fields=_REPORT_FIELDS,
    )
    completion = _decode_json_snapshot(
        snapshots["reports/completed.json"],
        label="Stage A-prime report completion",
        expected_fields=_REPORT_COMPLETION_FIELDS,
    )
    _validate_self_digest(completion, label="Stage A-prime report completion")
    if (
        completion["schema"] != _REPORT_COMPLETION_SCHEMA
        or completion["run_id"] != expected_run_id
        or completion["git_commit"] != expected_git_commit
        or completion["classification"] != _READY
        or completion["scene_count"] != 10
        or completion["endpoint_count"] != 40
        or completion["report_json_path"] != "reports/stage_a_prime.json"
        or completion["report_markdown_path"] != "reports/stage_a_prime.md"
        or completion["report_json_sha256"]
        != _sha256(snapshots["reports/stage_a_prime.json"])
        or completion["report_markdown_sha256"]
        != _sha256(snapshots["reports/stage_a_prime.md"])
    ):
        raise ValueError("Stage A-prime report completion mismatch")
    if (
        report["schema"] != _REPORT_SCHEMA
        or report["run_id"] != expected_run_id
        or report["git_commit"] != expected_git_commit
    ):
        raise ValueError("Stage A-prime report identity or schema mismatch")
    return cohort_records, report, completion


def _validate_frame_ids(arrays: Mapping[str, np.ndarray], name: str) -> np.ndarray:
    value = _expect_array(arrays, name, shape=(_FRAMES,), dtype=np.int64)
    if np.any(value[1:] <= value[:-1]):
        raise ValueError(f"{name} must be strictly increasing")
    return value


def _bind_bundle_to_frozen_payload(
    bundle: Mapping[str, Mapping[str, np.ndarray]],
    frozen: _FrozenScenePayload,
    *,
    scene: str,
) -> None:
    long = bundle["long"]
    quality = bundle["quality"]
    target = bundle["target"]
    for name in ("frame_ids", "camera_tokens"):
        _require_exact_array_values(
            long[name],
            frozen.long[name],
            name=f"{scene} local/frozen long baseline {name}",
        )
    _require_exact_array_values(
        long["baseline_pose_encoding"],
        frozen.formal["baseline_pose_encoding"],
        name=f"{scene} local/frozen formal baseline pose encoding",
    )
    _require_cross_device_baseline_match(
        long["baseline_c2w"],
        frozen.long["baseline_c2w"],
        scale=float(long["prediction_scale"]),
    )

    for name in (
        "frame_ids",
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
    ):
        _require_exact_array_values(
            quality[name],
            frozen.teacher[name],
            name=f"{scene} local/frozen teacher fusion {name}",
        )

    coverage = np.asarray(frozen.teacher["coverage_weights"])
    expected_mask = (coverage > 0.0).astype(np.uint8)
    _require_exact_array_values(
        target["coverage_mask"],
        expected_mask,
        name=f"{scene} local/frozen teacher coverage mask",
    )
    oracle_scale = float(frozen.teacher["oracle_scale"])
    oracle_rotation = np.asarray(frozen.teacher["oracle_rotation"])
    oracle_translation = np.asarray(frozen.teacher["oracle_translation"])
    frozen_fused = np.asarray(frozen.teacher["fused_c2w"])
    normalizer = float(frozen.teacher["gt_scene_scale"])
    for endpoint in range(_ENDPOINTS):
        covered = expected_mask[endpoint] != 0
        if not np.any(covered):
            continue
        raw = np.asarray(long["baseline_c2w"])[covered].copy()
        raw[:, :3, 3] = np.asarray(
            target["teacher_centers_raw_filled"]
        )[endpoint, covered]
        forward = _apply_oracle(
            raw,
            scale=oracle_scale,
            rotation=oracle_rotation,
            translation=oracle_translation,
        )
        center_error = float(
            np.max(
                np.linalg.norm(
                    forward[:, :3, 3]
                    - frozen_fused[endpoint, covered, :3, 3],
                    axis=1,
                )
            )
            / normalizer
        )
        if not math.isfinite(center_error) or center_error > 1e-5:
            raise ValueError(
                f"{scene} local fusion teacher centers do not replay frozen teacher "
                f"variant {endpoint}"
            )


def _parse_bundle(
    snapshots: Mapping[str, bytes],
    *,
    scene: str,
    frozen: _FrozenScenePayload,
    cohort_row: Mapping[str, object],
    long_source_sha256: str,
    binding: Mapping[str, object],
    expected_git_commit: str,
    checkpoint_sha256: str,
) -> dict[str, dict[str, np.ndarray]]:
    sample_id = f"{scene}:frames_500"
    paths = _artifact_relpaths(scene)
    arrays = {
        "long": _decode_npz_snapshot(
            snapshots[paths["long"]],
            label=f"{scene} long context",
            expected_members=_LONG_MEMBERS,
        ),
        "short": _decode_npz_snapshot(
            snapshots[paths["short"]],
            label=f"{scene} short context",
            expected_members=_SHORT_MEMBERS,
        ),
        "quality": _decode_npz_snapshot(
            snapshots[paths["quality"]],
            label=f"{scene} quality sidecar",
            expected_members=_QUALITY_MEMBERS,
        ),
        "target": _decode_npz_snapshot(
            snapshots[paths["target"]],
            label=f"{scene} translation target",
            expected_members=_TARGET_MEMBERS,
        ),
    }
    for value in arrays.values():
        _validate_identity_and_provenance(
            value,
            scene=scene,
            sample_id=sample_id,
            git_commit=expected_git_commit,
            checkpoint_sha256=checkpoint_sha256,
        )
        if _array_digest(value, "source_sha256") != long_source_sha256:
            raise ValueError("bundle/source-manifest digest mismatch")
    if binding["source_sha256"] != long_source_sha256:
        raise ValueError("preflight source binding mismatch")

    long = arrays["long"]
    long_frames = _validate_frame_ids(long, "frame_ids")
    tokens = _expect_array(
        long, "camera_tokens", shape=(_FRAMES, _TOKEN_WIDTH), dtype=np.float32
    )
    baseline_pose = _expect_array(
        long, "baseline_pose_encoding", shape=(_FRAMES, 9), dtype=np.float32
    )
    baseline_c2w = _expect_array(
        long, "baseline_c2w", shape=(_FRAMES, 4, 4), dtype=np.float64
    )
    scale_array = _expect_array(long, "prediction_scale", shape=(), dtype=np.float64)
    _finite(tokens, name="long camera tokens")
    _finite(baseline_pose, name="baseline pose encoding")
    _validate_pose_stack(baseline_c2w, name="stored baseline C2W")
    stored_scale = float(scale_array)
    recomputed_scale = _prediction_scale(baseline_c2w)
    if scale_array.tobytes() != np.asarray(recomputed_scale, dtype=np.float64).tobytes():
        raise ValueError("stored prediction scale does not replay exactly")

    short = arrays["short"]
    short_frames = _expect_array(
        short,
        "short_frame_ids",
        shape=(_WINDOWS, _WINDOW_FRAMES),
        dtype=np.int64,
    )
    short_tokens = _expect_array(
        short,
        "short_camera_tokens",
        shape=(_WINDOWS, _WINDOW_FRAMES, _TOKEN_WIDTH),
        dtype=np.float32,
    )
    _finite(short_tokens, name="short camera tokens")
    expected_short_frames = np.stack(
        [long_frames[start : start + _WINDOW_FRAMES] for start in range(0, 401, 50)]
    )
    if not np.array_equal(short_frames, expected_short_frames):
        raise ValueError("short frame IDs do not match the long context")
    if _array_digest(short, "long_context_sha256") != cohort_row["long_sha256"]:
        raise ValueError("short context does not bind the actual long context")

    quality = arrays["quality"]
    target = arrays["target"]
    for value, label in ((quality, "quality"), (target, "target")):
        if not np.array_equal(_validate_frame_ids(value, "frame_ids"), long_frames):
            raise ValueError(f"{label} frame IDs do not match long context")
        variants = _expect_array(
            value, "teacher_variant_ids", shape=(_ENDPOINTS,), dtype=np.int64
        )
        if not np.array_equal(variants, np.arange(_ENDPOINTS, dtype=np.int64)):
            raise ValueError(f"{label} variant IDs are noncanonical")

    target_coverage = _expect_array(
        target,
        "coverage_mask",
        shape=(_ENDPOINTS, _FRAMES),
        dtype=np.uint8,
    )
    if not np.isin(target_coverage, (0, 1)).all():
        raise ValueError("target coverage mask must be binary")
    endpoints = _expect_array(
        target,
        "translation_endpoints",
        shape=(_ENDPOINTS, _FRAMES, 3),
        dtype=np.float32,
    )
    centers = _expect_array(
        target,
        "teacher_centers_raw_filled",
        shape=(_ENDPOINTS, _FRAMES, 3),
        dtype=np.float64,
    )
    _finite(endpoints, name="translation endpoints")
    _finite(centers, name="teacher centers")
    if not np.all(endpoints.view(np.uint32)[target_coverage == 0] == 0):
        raise ValueError("uncovered endpoint values must be bitwise positive zero")
    target_scale = _expect_array(
        target, "prediction_scale", shape=(), dtype=np.float64
    )
    if target_scale.tobytes() != np.asarray(stored_scale, dtype=np.float64).tobytes():
        raise ValueError("target prediction scale mismatch")
    for member, kind in (
        ("long_context_sha256", "long"),
        ("short_context_sha256", "short"),
        ("quality_sha256", "quality"),
    ):
        if _array_digest(target, member) != cohort_row[f"{kind}_sha256"]:
            raise ValueError(f"target {member} binding mismatch")
    teacher_reference_sha256 = _array_digest(target, "teacher_reference_sha256")
    if teacher_reference_sha256 != binding["teacher_reference_sha256"]:
        raise ValueError("target/preflight teacher-reference digest mismatch")

    gt = _expect_array(quality, "gt_c2w", shape=(_FRAMES, 4, 4), dtype=np.float64)
    _validate_pose_stack(gt, name="ground truth")
    gt_scale = _expect_array(quality, "gt_scene_scale", shape=(), dtype=np.float64)
    if not np.isfinite(gt_scale).all() or float(gt_scale) <= 0.0:
        raise ValueError("GT scale must be finite and positive")
    if not math.isclose(
        float(gt_scale), _prediction_scale(gt), rel_tol=1e-12, abs_tol=1e-12
    ):
        raise ValueError("GT scale diagnostic mismatch")
    if _array_text(quality, "oracle_scene", width=32) != scene:
        raise ValueError("oracle scene mismatch")
    if _array_digest(quality, "oracle_frame_digest") != _frame_digest(long_frames):
        raise ValueError("oracle frame digest mismatch")
    if int(_expect_array(quality, "oracle_fit_count", shape=(), dtype=np.int64)) != _FRAMES:
        raise ValueError("oracle must bind all 500 frames")
    oracle_scale = _expect_array(quality, "oracle_scale", shape=(), dtype=np.float64)
    oracle_rotation = _expect_array(
        quality, "oracle_rotation", shape=(3, 3), dtype=np.float64
    )
    oracle_translation = _expect_array(
        quality, "oracle_translation", shape=(3,), dtype=np.float64
    )
    oracle_rank = int(
        _expect_array(quality, "oracle_rank", shape=(), dtype=np.int64)
    )
    oracle_condition = _expect_array(
        quality, "oracle_condition", shape=(), dtype=np.float64
    )
    if (
        not np.isfinite(oracle_scale).all()
        or float(oracle_scale) <= 0.0
        or not np.isfinite(oracle_translation).all()
        or oracle_rank not in (1, 2, 3)
        or not np.isfinite(oracle_condition).all()
        or float(oracle_condition) <= 0.0
    ):
        raise ValueError("saved oracle fields are malformed")
    oracle_pose = np.eye(4, dtype=np.float64)[None]
    oracle_pose[0, :3, :3] = oracle_rotation
    _validate_pose_stack(oracle_pose, name="oracle rotation")
    oracle_payload = {
        "scene": scene,
        "frame_digest": str(quality["oracle_frame_digest"]),
        "fit_count": _FRAMES,
        "scale": float(oracle_scale),
        "rotation": tuple(
            tuple(float(component) for component in row) for row in oracle_rotation
        ),
        "translation": tuple(float(component) for component in oracle_translation),
    }
    if _array_digest(quality, "oracle_digest") != _canonical_digest(oracle_payload):
        raise ValueError("saved oracle digest mismatch")

    weights = _expect_array(
        quality, "window_weights", shape=(_WINDOWS,), dtype=np.float64
    )
    masks = _expect_array(
        quality,
        "window_masks",
        shape=(_ENDPOINTS, _WINDOWS),
        dtype=np.uint8,
    )
    coverage_weights = _expect_array(
        quality,
        "coverage_weights",
        shape=(_ENDPOINTS, _FRAMES),
        dtype=np.float64,
    )
    if (
        not np.isfinite(weights).all()
        or np.any(weights < 0.0)
        or np.any(weights > 1.0)
        or not np.isin(masks, (0, 1)).all()
        or not np.isfinite(coverage_weights).all()
        or np.any(coverage_weights < 0.0)
    ):
        raise ValueError("teacher fusion controls are malformed")
    if not np.array_equal(masks, _canonical_window_masks(scene, weights)):
        raise ValueError("teacher window masks are not the canonical registered variants")
    expected_coverage_mask = (coverage_weights > 0.0).astype(np.uint8)
    if not np.array_equal(target_coverage, expected_coverage_mask):
        raise ValueError("target/quality coverage mismatch")

    _expect_array(
        quality,
        "variant_utilities",
        shape=(_ENDPOINTS,),
        dtype=np.float64,
    )
    for name in (
        "baseline_translation_error_normalized",
        "baseline_rotation_error_deg",
    ):
        value = _expect_array(quality, name, shape=(_FRAMES,), dtype=np.float64)
        if not np.isfinite(value).all() or np.any(value < 0.0):
            raise ValueError(f"{name} diagnostic is malformed")
    covered = coverage_weights > 0.0
    for name in (
        "teacher_translation_error_normalized",
        "teacher_rotation_error_deg",
    ):
        value = _expect_array(
            quality, name, shape=(_ENDPOINTS, _FRAMES), dtype=np.float64
        )
        if (
            not np.isfinite(value[covered]).all()
            or np.any(value[covered] < 0.0)
            or not np.isnan(value[~covered]).all()
        ):
            raise ValueError(f"{name} diagnostic is malformed")
    if _array_digest(quality, "formal_label_sha256") != binding["formal_label_sha256"]:
        raise ValueError("quality/preflight formal-label digest mismatch")
    if _array_digest(quality, "teacher_reference_sha256") != teacher_reference_sha256:
        raise ValueError("quality/target teacher-reference digest mismatch")
    _bind_bundle_to_frozen_payload(arrays, frozen, scene=scene)
    return arrays


def _window_coverage(mask: np.ndarray) -> np.ndarray:
    coverage = np.zeros(_FRAMES, dtype=np.bool_)
    for index, selected in enumerate(mask):
        if selected:
            coverage[index * 50 : index * 50 + _WINDOW_FRAMES] = True
    return coverage


def _canonical_window_masks(scene: str, weights: np.ndarray) -> np.ndarray:
    positive = weights > 0.0
    if np.count_nonzero(positive) < 3:
        raise ValueError("at least three positive teacher windows are required")
    masks = [positive.copy()]
    seen = {positive.tobytes()}
    target_coverage = _window_coverage(positive)
    candidates: list[np.ndarray] = []
    positive_indices = np.flatnonzero(positive)
    for bits in range(1, 1 << len(positive_indices)):
        candidate = np.zeros(_WINDOWS, dtype=np.bool_)
        candidate[positive_indices] = np.asarray(
            [(bits >> offset) & 1 for offset in range(len(positive_indices))],
            dtype=np.bool_,
        )
        candidates.append(candidate)
    for index in range(1, _ENDPOINTS):
        seed = int.from_bytes(
            hashlib.sha256(
                f"{scene}:teacher_variant:{index}".encode("utf-8")
            ).digest()[:8],
            byteorder="big",
            signed=False,
        )
        generator = np.random.default_rng(seed)
        proposal = positive & (generator.random(_WINDOWS) < 0.75)
        available = [value for value in candidates if value.tobytes() not in seen]
        preserving = [
            value
            for value in available
            if np.array_equal(_window_coverage(value), target_coverage)
        ]
        pool = preserving or available
        if not pool:
            raise ValueError("could not replay canonical teacher masks")
        distances = np.asarray(
            [np.count_nonzero(value != proposal) for value in pool]
        )
        nearest = np.flatnonzero(distances == distances.min())
        candidate = pool[int(nearest[generator.integers(len(nearest))])].copy()
        masks.append(candidate)
        seen.add(candidate.tobytes())
    return np.stack(masks).astype(np.uint8)


def _umeyama(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    moving = np.asarray(source, dtype=np.float64)
    reference = np.asarray(target, dtype=np.float64)
    if (
        moving.shape != reference.shape
        or moving.ndim != 2
        or moving.shape[1] != 3
        or len(moving) < 2
        or not np.isfinite(moving).all()
        or not np.isfinite(reference).all()
    ):
        raise ValueError("Sim(3) inputs must be matching finite [frames,3] arrays")
    moving_mean = moving.mean(axis=0)
    reference_mean = reference.mean(axis=0)
    moving_centered = moving - moving_mean
    reference_centered = reference - reference_mean
    moving_variance = float(
        np.mean(np.sum(moving_centered * moving_centered, axis=1))
    )
    if moving_variance <= 1e-12:
        raise ValueError("short trajectory has insufficient translation variance")
    covariance = (reference_centered.T @ moving_centered) / len(moving)
    left, singular_values, right_transpose = np.linalg.svd(covariance)
    signs = np.ones(3, dtype=np.float64)
    if np.linalg.det(left @ right_transpose) < 0.0:
        signs[-1] = -1.0
    rotation = left @ np.diag(signs) @ right_transpose
    scale = float(np.sum(singular_values * signs) / moving_variance)
    translation = reference_mean - scale * (rotation @ moving_mean)
    return scale, rotation, translation


def _align_short_centers(
    long_segment: np.ndarray,
    short_poses: np.ndarray,
    *,
    scene_scale: float,
) -> np.ndarray:
    reference = _validate_pose_stack(long_segment, name="long alignment segment")
    moving = _validate_pose_stack(short_poses, name="short alignment window")
    if reference.shape != moving.shape or reference.shape[0] != _WINDOW_FRAMES:
        raise ValueError("short/long alignment windows must match exactly")
    centers = moving[:, :3, 3]
    centered = centers - centers.mean(axis=0)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    largest = float(singular_values[0])
    threshold = largest * 1e-12
    rank = int(np.count_nonzero(singular_values > threshold)) if largest > 0.0 else 0
    condition = (
        float(largest / singular_values[rank - 1])
        if rank > 0 and singular_values[rank - 1] > 0.0
        else float("inf")
    )
    if rank < 2 or condition > 1e6:
        raise ValueError("selected short window failed Sim(3) rank/condition gates")
    scale, rotation, translation = _umeyama(
        centers, reference[:, :3, 3]
    )
    determinant = float(np.linalg.det(rotation))
    if (
        not math.isfinite(scale)
        or not 1e-4 <= scale <= 1e4
        or not math.isfinite(determinant)
        or abs(determinant - 1.0) > 1e-8
    ):
        raise ValueError("selected short window failed Sim(3) scale/rotation gates")
    aligned = scale * (centers @ rotation.T) + translation
    residual = aligned - reference[:, :3, 3]
    normalized_rms = float(
        np.sqrt(np.mean(np.sum(residual * residual, axis=1))) / scene_scale
    )
    if not math.isfinite(normalized_rms) or normalized_rms > 0.5:
        raise ValueError("selected short window failed normalized Sim(3) RMS gate")
    return aligned


def _fuse_teacher(
    decoded_long: np.ndarray,
    decoded_short: np.ndarray,
    *,
    weights: np.ndarray,
    masks: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    np.ndarray,
]:
    baseline = _validate_pose_stack(decoded_long, name="decoded long context")
    if decoded_short.shape != (_WINDOWS, _WINDOW_FRAMES, 4, 4):
        raise ValueError("decoded short contexts have noncanonical shape")
    scale = _prediction_scale(baseline)
    aligned = np.full(
        (_WINDOWS, _WINDOW_FRAMES, 3), np.nan, dtype=np.float64
    )
    for window, start in enumerate(range(0, 401, 50)):
        try:
            aligned[window] = _align_short_centers(
                baseline[start : start + _WINDOW_FRAMES],
                decoded_short[window],
                scene_scale=scale,
            )
        except ValueError:
            unused = float(weights[window]) == 0.0 and not np.any(
                masks[:, window]
            )
            if not unused:
                raise
    coverage = np.zeros((_ENDPOINTS, _FRAMES), dtype=np.float64)
    numerator = np.zeros((_ENDPOINTS, _FRAMES, 3), dtype=np.float64)
    for endpoint in range(_ENDPOINTS):
        for window in range(_WINDOWS):
            if masks[endpoint, window] == 0:
                continue
            weight = float(weights[window])
            if weight <= 0.0:
                raise ValueError("selected teacher window must have positive weight")
            start = 50 * window
            stop = start + _WINDOW_FRAMES
            coverage[endpoint, start:stop] += weight
            numerator[endpoint, start:stop] += weight * aligned[window]
    mask = coverage > 0.0
    raw = np.full((_ENDPOINTS, _FRAMES, 3), np.nan, dtype=np.float64)
    raw[mask] = numerator[mask] / coverage[mask, None]
    filled = np.broadcast_to(
        baseline[None, :, :3, 3], (_ENDPOINTS, _FRAMES, 3)
    ).copy()
    filled[mask] = raw[mask]
    if not np.isfinite(filled).all():
        raise ValueError("independent teacher fusion produced non-finite centers")
    endpoint_values = np.zeros((_ENDPOINTS, _FRAMES, 3), dtype=np.float32)
    endpoint_indices, frame_indices = np.nonzero(mask)
    rotations_w2c = np.swapaxes(baseline[:, :3, :3], -1, -2)
    deltas = raw[endpoint_indices, frame_indices] - baseline[frame_indices, :3, 3]
    values = -np.einsum(
        "nij,nj->ni", rotations_w2c[frame_indices], deltas
    ) / scale
    with np.errstate(over="ignore", invalid="ignore"):
        values32 = values.astype(np.float32)
    if not np.isfinite(values32).all():
        raise ValueError("independent endpoints overflow float32")
    endpoint_values[endpoint_indices, frame_indices] = values32
    if not np.all(endpoint_values.view(np.uint32)[~mask] == 0):
        raise AssertionError("independent endpoint construction lost positive zero")
    return coverage, mask.astype(np.uint8), raw, filled, scale, endpoint_values


def _apply_endpoints(
    baseline_pose: np.ndarray,
    endpoints: np.ndarray,
    *,
    scale: float,
) -> np.ndarray:
    corrected = np.broadcast_to(
        baseline_pose[None], (_ENDPOINTS, _FRAMES, 9)
    ).copy()
    active = endpoints != np.float32(0.0)
    baseline_translation = np.broadcast_to(
        baseline_pose[None, :, :3], endpoints.shape
    )
    values = baseline_translation[active].astype(np.float64) + scale * endpoints[
        active
    ].astype(np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        values32 = values.astype(np.float32)
    if not np.isfinite(values32).all():
        raise ValueError("corrected endpoint pose overflows float32")
    corrected[..., :3][active] = values32
    return corrected


def _apply_oracle(
    poses: np.ndarray,
    *,
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    values = _validate_pose_stack(poses, name="oracle input poses").copy()
    values[:, :3, :3] = np.einsum(
        "ij,fjk->fik", rotation, values[:, :3, :3]
    )
    values[:, :3, 3] = scale * (poses[:, :3, 3] @ rotation.T) + translation
    return _validate_pose_stack(values, name="oracle output poses")


def _so3_error_deg(candidate: np.ndarray, reference: np.ndarray) -> np.ndarray:
    relative = np.einsum(
        "...ij,...kj->...ik",
        candidate[..., :3, :3],
        reference[..., :3, :3],
    )
    cosine = np.clip(
        (np.trace(relative, axis1=-2, axis2=-1) - 1.0) * 0.5,
        -1.0,
        1.0,
    )
    return np.rad2deg(np.arccos(cosine)).astype(np.float64, copy=False)


def _so3_delta_deg(candidate: np.ndarray, reference: np.ndarray) -> np.ndarray:
    left = np.asarray(candidate, dtype=np.float64)[..., :3, :3]
    right = np.asarray(reference, dtype=np.float64)[..., :3, :3]
    if left.shape != right.shape:
        raise ValueError("SO(3) delta inputs must have matching shapes")
    flat_left = left.reshape(-1, 3, 3)
    flat_right = right.reshape(-1, 3, 3)
    exact = np.all(
        flat_left.view(np.uint64) == flat_right.view(np.uint64), axis=(1, 2)
    )
    result = np.zeros(len(flat_left), dtype=np.float64)
    active = np.flatnonzero(~exact)
    if len(active):
        left_u, _, left_vh = np.linalg.svd(flat_left[active])
        right_u, _, right_vh = np.linalg.svd(flat_right[active])
        projected_left = left_u @ left_vh
        projected_right = right_u @ right_vh
        for projected, u, vh in (
            (projected_left, left_u, left_vh),
            (projected_right, right_u, right_vh),
        ):
            reflected = np.linalg.det(projected) < 0.0
            if np.any(reflected):
                u[reflected, :, -1] *= -1.0
                projected[reflected] = u[reflected] @ vh[reflected]
        relative = projected_left @ np.swapaxes(projected_right, -1, -2)
        sine = np.linalg.norm(
            relative - np.swapaxes(relative, -1, -2), axis=(1, 2)
        ) / (2.0 * math.sqrt(2.0))
        cosine = np.clip(
            (np.trace(relative, axis1=1, axis2=2) - 1.0) * 0.5,
            -1.0,
            1.0,
        )
        result[active] = np.rad2deg(
            np.arctan2(np.clip(sine, 0.0, 1.0), cosine)
        )
    return result.reshape(left.shape[:-2])


def _require_cross_device_baseline_match(
    decoded: np.ndarray,
    witness: np.ndarray,
    *,
    scale: float,
) -> None:
    """Gate native float32 pose decodes geometrically across CPU/GPU kernels."""
    actual = _validate_pose_stack(decoded, name="cross-device decoded baseline")
    expected = _validate_pose_stack(witness, name="cross-device baseline witness")
    if actual.shape != expected.shape or not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("cross-device baseline comparison is malformed")
    center_error = float(
        np.max(
            np.linalg.norm(
                actual[:, :3, 3] - expected[:, :3, 3],
                axis=1,
            )
        )
        / scale
    )
    rotation_error = float(np.max(_so3_delta_deg(actual, expected)))
    if (
        not math.isfinite(center_error)
        or center_error > _CROSS_DEVICE_CENTER_ATOL
        or not math.isfinite(rotation_error)
        or rotation_error > _CROSS_DEVICE_ROTATION_ATOL_DEG
    ):
        raise ValueError("cross-device decoded baseline mismatches GPU baseline witness")


def _rms_center_error(
    candidate: np.ndarray, ground_truth: np.ndarray, mask: np.ndarray
) -> float:
    if mask.shape != (_FRAMES,) or mask.dtype != np.bool_ or not np.any(mask):
        raise ValueError("RMS mask must cover at least one frame")
    delta = candidate[mask, :3, 3] - ground_truth[mask, :3, 3]
    value = float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("translation RMS must be finite and nonnegative")
    return value


def _utility(baseline: float, candidate: float, *, name: str) -> float:
    if (
        not math.isfinite(baseline)
        or baseline <= 0.0
        or not math.isfinite(candidate)
        or candidate < 0.0
    ):
        raise ValueError(f"{name} RMS values are invalid")
    return float((baseline - candidate) / baseline)


def _diagnostic_match(
    actual: np.ndarray,
    recorded: np.ndarray,
    *,
    name: str,
    mask: np.ndarray | None = None,
) -> None:
    left = np.asarray(actual, dtype=np.float64)
    right = np.asarray(recorded)
    if mask is not None:
        left = left[mask]
        right = right[mask]
    if (
        left.shape != right.shape
        or not np.isfinite(left).all()
        or not np.isfinite(right).all()
        or not np.allclose(left, right, atol=1e-12, rtol=1e-12)
    ):
        raise ValueError(f"{name} diagnostic mismatch")


def _replay_scene(
    bundle: Mapping[str, Mapping[str, np.ndarray]],
    *,
    scene: str,
    role: str,
    digests: Mapping[str, str],
    camera_head: nn.Module,
    device: torch.device,
) -> dict[str, object]:
    long = bundle["long"]
    short = bundle["short"]
    quality = bundle["quality"]
    target = bundle["target"]
    long_raw_batch, long_c2w_batch = _decode_tokens(
        camera_head, long["camera_tokens"][None], device=device
    )
    _, short_c2w = _decode_tokens(
        camera_head, short["short_camera_tokens"], device=device
    )
    if (
        long_raw_batch.shape != (1, _FRAMES, 9)
        or long_c2w_batch.shape != (1, _FRAMES, 4, 4)
        or short_c2w.shape != (_WINDOWS, _WINDOW_FRAMES, 4, 4)
    ):
        raise ValueError("Camera Head replay returned noncanonical shapes")
    baseline_pose = long_raw_batch[0]
    decoded_baseline = _validate_pose_stack(
        long_c2w_batch[0], name="replayed long baseline"
    )
    if not np.array_equal(baseline_pose, long["baseline_pose_encoding"]):
        raise ValueError("replayed Camera Head pose encoding mismatches stored witness")
    if not np.array_equal(decoded_baseline, long["baseline_c2w"]):
        scale = _prediction_scale(decoded_baseline)
        center_error = float(
            np.max(
                np.linalg.norm(
                    decoded_baseline[:, :3, 3]
                    - long["baseline_c2w"][:, :3, 3],
                    axis=1,
                )
            )
            / scale
        )
        rotation_error = float(
            np.max(_so3_delta_deg(decoded_baseline, long["baseline_c2w"]))
        )
        if center_error > 5e-6 or rotation_error > 2e-5:
            raise ValueError("replayed Camera Head baseline mismatches stored C2W witness")
        raise ValueError("stored baseline witness is not the unique deterministic replay")

    coverage, coverage_mask, raw_centers, filled_centers, replay_scale, endpoints = (
        _fuse_teacher(
            decoded_baseline,
            short_c2w,
            weights=quality["window_weights"],
            masks=quality["window_masks"],
        )
    )
    if not np.array_equal(coverage, quality["coverage_weights"]):
        raise ValueError("independent fusion coverage does not match quality sidecar")
    if not np.array_equal(coverage_mask, target["coverage_mask"]):
        raise ValueError("independent fusion coverage mask does not match target")
    if not np.array_equal(filled_centers, target["teacher_centers_raw_filled"]):
        if not np.allclose(
            filled_centers,
            target["teacher_centers_raw_filled"],
            atol=1e-12,
            rtol=1e-12,
        ):
            raise ValueError("independent fusion teacher centers mismatch target")
        raise ValueError("stored teacher centers are not the unique fusion replay")
    if not np.array_equal(endpoints, target["translation_endpoints"]):
        raise ValueError("independent translation endpoint replay mismatch")
    if np.asarray(replay_scale, dtype=np.float64).tobytes() != long[
        "prediction_scale"
    ].tobytes():
        raise ValueError("independent prediction-scale replay mismatch")

    corrected_pose = _apply_endpoints(
        baseline_pose, endpoints, scale=replay_scale
    )
    expected = np.broadcast_to(baseline_pose[None], corrected_pose.shape)
    quaternion_equal = [
        corrected_pose[index, :, 3:7].tobytes(order="C")
        == expected[index, :, 3:7].tobytes(order="C")
        for index in range(_ENDPOINTS)
    ]
    fov_equal = [
        corrected_pose[index, :, 7:9].tobytes(order="C")
        == expected[index, :, 7:9].tobytes(order="C")
        for index in range(_ENDPOINTS)
    ]
    positive_zero = [
        bool(
            np.all(
                endpoints[index].view(np.uint32)[coverage_mask[index] == 0] == 0
            )
        )
        for index in range(_ENDPOINTS)
    ]
    stacked_pose = np.concatenate((baseline_pose[None], corrected_pose), axis=0)
    try:
        with torch.no_grad():
            decoded = (
                pose_encoding_to_c2w(torch.from_numpy(stacked_pose))
                .detach()
                .to(device="cpu", dtype=torch.float64)
                .numpy()
            )
    except (RuntimeError, ValueError) as error:
        raise ValueError("true pose conversion failed for replayed endpoints") from error
    if decoded.shape != (1 + _ENDPOINTS, _FRAMES, 4, 4):
        raise ValueError("true pose conversion returned a noncanonical shape")
    decoded_pose_baseline = _validate_pose_stack(
        decoded[0], name="true-decoded baseline"
    )
    decoded_corrected = decoded[1:]
    for endpoint in range(_ENDPOINTS):
        _validate_pose_stack(
            decoded_corrected[endpoint], name=f"true-decoded endpoint {endpoint}"
        )
    _require_cross_device_baseline_match(
        decoded_pose_baseline,
        decoded_baseline,
        scale=replay_scale,
    )

    oracle_scale = float(quality["oracle_scale"])
    oracle_rotation = quality["oracle_rotation"]
    oracle_translation = quality["oracle_translation"]
    ground_truth = quality["gt_c2w"]
    gt_scale = float(quality["gt_scene_scale"])
    baseline_aligned = _apply_oracle(
        decoded_baseline,
        scale=oracle_scale,
        rotation=oracle_rotation,
        translation=oracle_translation,
    )
    corrected_aligned = _apply_oracle(
        decoded_corrected.reshape(_ENDPOINTS * _FRAMES, 4, 4),
        scale=oracle_scale,
        rotation=oracle_rotation,
        translation=oracle_translation,
    ).reshape(_ENDPOINTS, _FRAMES, 4, 4)
    teacher_raw = np.broadcast_to(
        decoded_baseline[None], (_ENDPOINTS, _FRAMES, 4, 4)
    ).copy()
    teacher_raw[:, :, :3, 3] = filled_centers
    teacher_aligned = _apply_oracle(
        teacher_raw.reshape(_ENDPOINTS * _FRAMES, 4, 4),
        scale=oracle_scale,
        rotation=oracle_rotation,
        translation=oracle_translation,
    ).reshape(_ENDPOINTS, _FRAMES, 4, 4)

    baseline_translation_error = (
        np.linalg.norm(
            baseline_aligned[:, :3, 3] - ground_truth[:, :3, 3], axis=1
        )
        / gt_scale
    )
    baseline_rotation_error = _so3_error_deg(baseline_aligned, ground_truth)
    _diagnostic_match(
        baseline_translation_error,
        quality["baseline_translation_error_normalized"],
        name="baseline translation",
    )
    _diagnostic_match(
        baseline_rotation_error,
        quality["baseline_rotation_error_deg"],
        name="baseline rotation",
    )

    full_mask = np.ones(_FRAMES, dtype=np.bool_)
    baseline_full_rms = _rms_center_error(
        baseline_aligned, ground_truth, full_mask
    )
    endpoint_rows: list[dict[str, object]] = []
    replayed_teacher_utilities = np.empty(_ENDPOINTS, dtype=np.float64)
    for endpoint in range(_ENDPOINTS):
        mask = coverage_mask[endpoint] != 0
        if not np.any(mask):
            raise ValueError(f"endpoint {endpoint} has no covered frame")
        uncovered = ~mask
        if np.any(uncovered) and not np.array_equal(
            filled_centers[endpoint, uncovered],
            decoded_baseline[uncovered, :3, 3],
        ):
            raise ValueError("uncovered teacher centers do not equal baseline")
        teacher_translation_error = (
            np.linalg.norm(
                teacher_aligned[endpoint, :, :3, 3]
                - ground_truth[:, :3, 3],
                axis=1,
            )
            / gt_scale
        )
        teacher_rotation_error = _so3_error_deg(
            teacher_aligned[endpoint], ground_truth
        )
        _diagnostic_match(
            teacher_translation_error,
            quality["teacher_translation_error_normalized"][endpoint],
            name=f"endpoint {endpoint} teacher translation",
            mask=mask,
        )
        _diagnostic_match(
            teacher_rotation_error,
            quality["teacher_rotation_error_deg"][endpoint],
            name=f"endpoint {endpoint} teacher rotation",
            mask=mask,
        )
        baseline_covered_rms = _rms_center_error(
            baseline_aligned, ground_truth, mask
        )
        corrected_covered_rms = _rms_center_error(
            corrected_aligned[endpoint], ground_truth, mask
        )
        teacher_covered_rms = _rms_center_error(
            teacher_aligned[endpoint], ground_truth, mask
        )
        corrected_full_rms = _rms_center_error(
            corrected_aligned[endpoint], ground_truth, full_mask
        )
        covered_utility = _utility(
            baseline_covered_rms,
            corrected_covered_rms,
            name=f"endpoint {endpoint} covered utility",
        )
        teacher_utility = _utility(
            baseline_covered_rms,
            teacher_covered_rms,
            name=f"endpoint {endpoint} teacher utility",
        )
        full_utility = _utility(
            baseline_full_rms,
            corrected_full_rms,
            name=f"endpoint {endpoint} full utility",
        )
        replayed_teacher_utilities[endpoint] = teacher_utility
        covered_roundtrip = float(
            np.max(
                np.linalg.norm(
                    decoded_corrected[endpoint, mask, :3, 3]
                    - filled_centers[endpoint, mask],
                    axis=1,
                )
            )
            / replay_scale
        )
        uncovered_drift = 0.0
        if np.any(uncovered):
            uncovered_drift = float(
                np.max(
                    np.linalg.norm(
                        decoded_corrected[endpoint, uncovered, :3, 3]
                        - decoded_pose_baseline[uncovered, :3, 3],
                        axis=1,
                    )
                )
                / replay_scale
            )
        rotation_delta = float(
            np.max(
                _so3_delta_deg(
                    decoded_corrected[endpoint], decoded_pose_baseline
                )
            )
        )
        endpoint_rms = float(
            np.sqrt(
                np.mean(
                    np.sum(endpoints[endpoint].astype(np.float64) ** 2, axis=1)
                )
            )
        )
        numeric = (
            covered_utility,
            teacher_utility,
            full_utility,
            covered_roundtrip,
            uncovered_drift,
            rotation_delta,
            endpoint_rms,
        )
        endpoint_rows.append(
            {
                "endpoint_id": endpoint,
                "covered_utility": covered_utility,
                "teacher_covered_utility": teacher_utility,
                "full_scene_utility": full_utility,
                "covered_roundtrip_fraction": covered_roundtrip,
                "uncovered_drift_fraction": uncovered_drift,
                "rotation_delta_deg": rotation_delta,
                "quaternion_bytes_equal": quaternion_equal[endpoint],
                "fov_bytes_equal": fov_equal[endpoint],
                "uncovered_positive_zero": positive_zero[endpoint],
                "endpoint_rms": endpoint_rms,
                "coverage_fraction": float(np.mean(mask)),
                "all_finite": bool(np.isfinite(numeric).all()),
            }
        )
    if not np.allclose(
        replayed_teacher_utilities,
        quality["variant_utilities"],
        atol=1e-12,
        rtol=1e-12,
    ):
        raise ValueError("teacher variant-utility diagnostic mismatch")
    covered_values = np.asarray(
        [row["covered_utility"] for row in endpoint_rows], dtype=np.float64
    )
    teacher_values = np.asarray(
        [row["teacher_covered_utility"] for row in endpoint_rows], dtype=np.float64
    )
    full_values = np.asarray(
        [row["full_scene_utility"] for row in endpoint_rows], dtype=np.float64
    )
    mean_teacher = float(np.mean(teacher_values))
    if not math.isfinite(mean_teacher) or mean_teacher <= 0.0:
        raise ValueError("mean teacher utility must be finite and positive")
    return {
        "scene": scene,
        "sample_id": f"{scene}:frames_500",
        "role": role,
        "endpoint_count": _ENDPOINTS,
        "endpoint_ids": list(range(_ENDPOINTS)),
        "endpoints": endpoint_rows,
        "mean_covered_utility": float(np.mean(covered_values)),
        "mean_teacher_covered_utility": mean_teacher,
        "teacher_retention": float(np.mean(covered_values) / mean_teacher),
        "mean_full_scene_utility": float(np.mean(full_values)),
        "max_covered_roundtrip_fraction": float(
            max(row["covered_roundtrip_fraction"] for row in endpoint_rows)
        ),
        "max_uncovered_drift_fraction": float(
            max(row["uncovered_drift_fraction"] for row in endpoint_rows)
        ),
        "max_rotation_delta_deg": float(
            max(row["rotation_delta_deg"] for row in endpoint_rows)
        ),
        "quaternion_bytes_equal": bool(all(quaternion_equal)),
        "fov_bytes_equal": bool(all(fov_equal)),
        "uncovered_positive_zero": bool(all(positive_zero)),
        "all_finite": bool(all(row["all_finite"] for row in endpoint_rows)),
        "provenance": {
            "long_sha256": digests["long"],
            "short_sha256": digests["short"],
            "quality_sha256": digests["quality"],
            "target_sha256": digests["target"],
            "source_sha256": str(long["source_sha256"]),
            "checkpoint_sha256": str(long["checkpoint_sha256"]),
            "teacher_reference_sha256": str(target["teacher_reference_sha256"]),
            "git_commit": str(long["git_commit"]),
        },
    }


def _classify(scene_metrics: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if len(scene_metrics) != len(_SCENES):
        raise ValueError("classifier requires the exact ten-scene cohort")
    retention = np.asarray(
        [row["teacher_retention"] for row in scene_metrics], dtype=np.float64
    )
    full = np.asarray(
        [row["mean_full_scene_utility"] for row in scene_metrics],
        dtype=np.float64,
    )
    roundtrip = np.asarray(
        [row["max_covered_roundtrip_fraction"] for row in scene_metrics],
        dtype=np.float64,
    )
    uncovered = np.asarray(
        [row["max_uncovered_drift_fraction"] for row in scene_metrics],
        dtype=np.float64,
    )
    rotation = np.asarray(
        [row["max_rotation_delta_deg"] for row in scene_metrics],
        dtype=np.float64,
    )
    mean_retention = float(np.mean(retention))
    mean_full = float(np.mean(full))
    minimum_full = float(np.min(full))
    positive_count = int(np.count_nonzero(full > 0.0))
    gates = {
        "finite": bool(all(row["all_finite"] for row in scene_metrics)),
        "uncovered_positive_zero": bool(
            all(row["uncovered_positive_zero"] for row in scene_metrics)
        ),
        "quaternion_bytes_equal": bool(
            all(row["quaternion_bytes_equal"] for row in scene_metrics)
        ),
        "fov_bytes_equal": bool(
            all(row["fov_bytes_equal"] for row in scene_metrics)
        ),
        "covered_roundtrip": float(np.max(roundtrip)) < 1e-5,
        "uncovered_anchor": float(np.max(uncovered)) < 1e-8,
        "rotation_guard": float(np.max(rotation)) <= 1e-6,
        "teacher_retention": mean_retention >= 0.95,
        "positive_scene_count": positive_count == len(_SCENES),
        "positive_mean": mean_full > 0.0,
        "minimum_full_utility": minimum_full >= 0.0,
        "physical_leakage_clean": True,
    }
    failed = [name for name in _GATE_NAMES if not gates[name]]
    return {
        "classification": _READY if not failed else "TRANSLATION_ENDPOINTS_FAILED",
        "failed_gates": failed,
        "gates": gates,
        "scene_count": len(_SCENES),
        "endpoint_count": len(_SCENES) * _ENDPOINTS,
        "mean_teacher_retention": mean_retention,
        "mean_full_scene_utility": mean_full,
        "minimum_full_scene_utility": minimum_full,
        "positive_scene_count": positive_count,
    }


def _compare_value(actual: object, expected: object, *, path: str) -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(f"report mismatch at {path}: mapping schema differs")
        for key in expected:
            _compare_value(actual[key], expected[key], path=f"{path}.{key}")
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f"report mismatch at {path}: sequence differs")
        for index, (left, right) in enumerate(zip(actual, expected)):
            _compare_value(left, right, path=f"{path}[{index}]")
        return
    if type(expected) is float:
        if (
            isinstance(actual, bool)
            or not isinstance(actual, (int, float))
            or not math.isfinite(float(actual))
            or not math.isclose(
                float(actual), expected, rel_tol=1e-15, abs_tol=1e-15
            )
        ):
            raise ValueError(f"report numeric mismatch at {path}")
        return
    if type(actual) is not type(expected) or actual != expected:
        raise ValueError(f"report mismatch at {path}")


def _format_float(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Markdown float must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Markdown float must be finite")
    return format(result, ".17g")


def _expected_markdown(payload: Mapping[str, object]) -> bytes:
    failed = payload["failed_gates"]
    if not isinstance(failed, list):
        raise ValueError("failed gates are malformed")
    lines = [
        "# Camera Translation H-VRFM Stage A-prime",
        "",
        f"- Run ID: `{payload['run_id']}`",
        f"- Git commit: `{payload['git_commit']}`",
        f"- Classification: `{payload['classification']}`",
        f"- Scene count: {payload['scene_count']}",
        f"- Endpoint count: {payload['endpoint_count']}",
        "- Mean teacher retention: "
        + _format_float(payload["mean_teacher_retention"]),
        "- Mean full-scene utility: "
        + _format_float(payload["mean_full_scene_utility"]),
        "- Minimum full-scene utility: "
        + _format_float(payload["minimum_full_scene_utility"]),
        f"- Positive scene count: {payload['positive_scene_count']}",
        "- Physical leakage clean: "
        + ("true" if payload["physical_leakage_clean"] else "false"),
        "- Failed gates: " + (", ".join(failed) if failed else "none"),
        "",
        "## Gates",
        "",
        "| Gate | Passed |",
        "|---|---:|",
    ]
    gates = payload["gates"]
    if not isinstance(gates, dict):
        raise ValueError("gates are malformed")
    for name in _GATE_NAMES:
        lines.append(f"| {name} | {'yes' if gates[name] else 'no'} |")
    lines.extend(
        [
            "",
            "## Scenes",
            "",
            "| Scene | Role | Sample ID | Teacher retention | Mean full utility | "
            "Max covered round-trip | Max uncovered drift | Max rotation delta (deg) |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    scene_metrics = payload["scene_metrics"]
    if not isinstance(scene_metrics, list):
        raise ValueError("scene metrics are malformed")
    for row in scene_metrics:
        if not isinstance(row, dict):
            raise ValueError("scene metric row is malformed")
        lines.append(
            f"| {row['scene']} | {row['role']} | {row['sample_id']} | "
            f"{_format_float(row['teacher_retention'])} | "
            f"{_format_float(row['mean_full_scene_utility'])} | "
            f"{_format_float(row['max_covered_roundtrip_fraction'])} | "
            f"{_format_float(row['max_uncovered_drift_fraction'])} | "
            f"{_format_float(row['max_rotation_delta_deg'])} |"
        )
    return ("\n".join(lines) + "\n").encode("utf-8")


def _validate_report_replay(
    report: Mapping[str, object],
    snapshots: Mapping[str, bytes],
    *,
    scene_metrics: list[dict[str, object]],
    cohort_records: Sequence[Mapping[str, object]],
    expected_run_id: str,
    expected_git_commit: str,
) -> dict[str, object]:
    classifier = _classify(scene_metrics)
    expected_cohort = [
        {
            "scene": row["scene"],
            "sample_id": row["sample_id"],
            "role": row["role"],
            "long_sha256": row["long_sha256"],
            "short_sha256": row["short_sha256"],
            "quality_sha256": row["quality_sha256"],
            "target_sha256": row["target_sha256"],
        }
        for row in cohort_records
    ]
    expected = {
        "schema": _REPORT_SCHEMA,
        "run_id": expected_run_id,
        "git_commit": expected_git_commit,
        **classifier,
        "physical_leakage_clean": True,
        "scene_metrics": scene_metrics,
        "cohort": expected_cohort,
    }
    _compare_value(dict(report), expected, path="report")
    if snapshots["reports/stage_a_prime.md"] != _expected_markdown(dict(report)):
        raise ValueError("report Markdown does not match independent replay")
    if classifier["classification"] != _READY:
        raise ValueError("independent Stage A-prime gates did not pass")
    return expected


@dataclass(frozen=True)
class _AtomicCreateResult:
    temporary_path: Path | None
    cleanup_error: OSError | None


def _atomic_create(path: Path, payload: bytes) -> _AtomicCreateResult:
    target = Path(path)
    _reject_symlink_components(target)
    if target.exists():
        raise ValueError(f"verification target appeared concurrently: {target}")
    temporary_path: Path | None = None
    cleanup_error: OSError | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_path, target)
    except OSError as error:
        raise ValueError(f"could not atomically create verification target: {target}") from error
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError as error:
                cleanup_error = error
    return _AtomicCreateResult(
        temporary_path=temporary_path if cleanup_error is not None else None,
        cleanup_error=cleanup_error,
    )


def _rollback_exact_new_file(path: Path, payload: bytes) -> None:
    target = Path(path)
    _reject_symlink_components(target)
    if target.is_symlink() or not target.is_file():
        raise ValueError("new final marker cannot be rolled back because it is missing")
    try:
        before = target.stat(follow_symlinks=False)
        current = target.read_bytes()
    except OSError as error:
        raise ValueError("could not authenticate new final marker for rollback") from error
    if current != payload:
        raise ValueError("new final marker changed before rollback")

    quarantine: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{target.name}.",
            suffix=".rollback",
            dir=target.parent,
            delete=False,
        ) as handle:
            quarantine = Path(handle.name)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(target, quarantine)
    except OSError as error:
        if quarantine is not None:
            try:
                quarantine.unlink(missing_ok=True)
            except OSError:
                pass
        raise ValueError("could not quarantine exact new final marker") from error

    try:
        quarantined_stat = quarantine.stat(follow_symlinks=False)
        quarantined = quarantine.read_bytes()
    except OSError as error:
        raise ValueError("could not authenticate quarantined final marker") from error
    same_inode = (
        int(quarantined_stat.st_dev),
        int(quarantined_stat.st_ino),
        int(quarantined_stat.st_size),
    ) == (int(before.st_dev), int(before.st_ino), int(before.st_size))
    if not same_inode or quarantined != payload:
        try:
            os.link(quarantine, target)
        except FileExistsError:
            pass
        except OSError as error:
            raise ValueError(
                "new final marker changed before rollback and could not be restored"
            ) from error
        raise ValueError("new final marker changed before rollback")
    try:
        quarantine.unlink()
    except OSError as error:
        raise ValueError("could not roll back exact new final marker") from error


def _signed_json_bytes(unsigned: Mapping[str, object]) -> bytes:
    return _canonical_json_bytes(
        {**dict(unsigned), "completion_digest": _canonical_digest(unsigned)}
    )


def _inventory_bytes(
    snapshots: Mapping[str, bytes],
    *,
    run_id: str,
    git_commit: str,
) -> tuple[bytes, int, int]:
    records = {
        relative: {"sha256": _sha256(payload), "bytes": len(payload)}
        for relative, payload in sorted(snapshots.items())
    }
    file_count = len(records)
    total_bytes = sum(len(payload) for payload in snapshots.values())
    unsigned = {
        "schema": INVENTORY_SCHEMA,
        "run_id": run_id,
        "git_commit": git_commit,
        "classification": _READY,
        "report_completion_sha256": _sha256(snapshots["reports/completed.json"]),
        "calibration_completion_sha256": _sha256(
            snapshots["calibration/completed.json"]
        ),
        "files": records,
        "file_count": file_count,
        "total_bytes": total_bytes,
    }
    return _signed_json_bytes(unsigned), file_count, total_bytes


def _final_bytes(
    inventory: bytes,
    *,
    run_id: str,
    git_commit: str,
    report_completion_sha256: str,
    file_count: int,
    total_bytes: int,
) -> bytes:
    unsigned = {
        "schema": VERIFIED_SCHEMA,
        "run_id": run_id,
        "git_commit": git_commit,
        "classification": _READY,
        "inventory_path": _INVENTORY_PATH,
        "inventory_sha256": _sha256(inventory),
        "report_completion_sha256": report_completion_sha256,
        "file_count": file_count,
        "total_bytes": total_bytes,
    }
    return _signed_json_bytes(unsigned)


def verify_completed_run(
    run_root: Path,
    *,
    expected_run_id: str,
    expected_git_commit: str,
    checkpoint_dir: Path,
) -> Path:
    """Independently replay one completed Stage A-prime run before sealing it."""
    root = Path(run_root)
    if not root.is_dir():
        raise ValueError("completed run root must be an existing directory")
    _reject_symlink_components(root)
    run_id = _require_run_id(expected_run_id)
    git_commit = _require_commit(expected_git_commit, name="expected_git_commit")
    if root.name != run_id:
        raise ValueError("run root basename does not match expected_run_id")
    base_snapshots, optional_snapshots, directory_identities = _snapshot_run(root)
    metadata = _parse_metadata(
        base_snapshots,
        expected_run_id=run_id,
        expected_git_commit=git_commit,
    )
    cohort_records, report, report_completion = _validate_manifests_and_chain(
        base_snapshots,
        metadata,
        expected_run_id=run_id,
        expected_git_commit=git_commit,
    )
    upstream = _authenticate_frozen_upstream(
        metadata, checkpoint_dir=Path(checkpoint_dir)
    )
    config = metadata["config"]
    preflight = metadata["preflight"]
    long_manifest = metadata["long_manifest"]
    if not all(isinstance(value, dict) for value in (config, preflight, long_manifest)):
        raise ValueError("metadata parse result is malformed")
    checkpoint_sha256 = _require_digest(
        config["checkpoint_sha256"], name="configured checkpoint digest"
    )
    checkpoint_file_value = config["checkpoint_file"]
    if not isinstance(checkpoint_file_value, str) or not checkpoint_file_value:
        raise ValueError("configured checkpoint path is malformed")
    checkpoint_file = Path(checkpoint_file_value)
    checkpoint_root = Path(checkpoint_dir)
    bindings = preflight["scene_bindings"]
    long_records = long_manifest["records"]
    if not isinstance(bindings, list) or not isinstance(long_records, list):
        raise ValueError("metadata scene records are malformed")

    with _private_checkpoint_copy(
        checkpoint_file, checkpoint_dir=checkpoint_root
    ) as (private_dir, copied_digest, stats):
        if copied_digest != checkpoint_sha256:
            raise ValueError("configured checkpoint digest mismatches authenticated bytes")
        camera_head, loaded_digest, device = _load_camera_head(private_dir)
        if (
            not isinstance(camera_head, nn.Module)
            or not isinstance(device, torch.device)
            or _require_digest(loaded_digest, name="loaded checkpoint digest")
            != checkpoint_sha256
        ):
            raise ValueError("private Camera Head load does not bind checkpoint bytes")
        scene_metrics: list[dict[str, object]] = []
        for index, scene in enumerate(_SCENES):
            cohort_row = cohort_records[index]
            binding = bindings[index]
            long_row = long_records[index]
            if not all(isinstance(value, dict) for value in (cohort_row, binding, long_row)):
                raise ValueError("scene metadata record is malformed")
            bundle = _parse_bundle(
                base_snapshots,
                scene=scene,
                frozen=upstream.scenes[scene],
                cohort_row=cohort_row,
                long_source_sha256=str(long_row["source_sha256"]),
                binding=binding,
                expected_git_commit=git_commit,
                checkpoint_sha256=checkpoint_sha256,
            )
            paths = _artifact_relpaths(scene)
            scene_metrics.append(
                _replay_scene(
                    bundle,
                    scene=scene,
                    role=_role(scene),
                    digests={
                        kind: _sha256(base_snapshots[path])
                        for kind, path in paths.items()
                    },
                    camera_head=camera_head,
                    device=device,
                )
            )
        _validate_report_replay(
            report,
            base_snapshots,
            scene_metrics=scene_metrics,
            cohort_records=cohort_records,
            expected_run_id=run_id,
            expected_git_commit=git_commit,
        )
        _rehash_checkpoint_identity(
            checkpoint_file,
            expected_stat=stats[0],
            expected_sha256=checkpoint_sha256,
        )

    _require_upstream_unchanged(upstream)
    _require_snapshot_unchanged(root, base_snapshots)
    _require_exact_run_topology(
        root,
        optional_files=frozenset(optional_snapshots),
        directory_identities=directory_identities,
    )
    inventory, file_count, total_bytes = _inventory_bytes(
        base_snapshots, run_id=run_id, git_commit=git_commit
    )
    existing_inventory = optional_snapshots.get(_INVENTORY_PATH)
    if existing_inventory is not None:
        if existing_inventory != inventory:
            raise ValueError("existing verification inventory conflicts with replay")
    else:
        inventory_publication = _atomic_create(root / _INVENTORY_PATH, inventory)
        if inventory_publication.cleanup_error is not None:
            try:
                _rollback_exact_new_file(root / _INVENTORY_PATH, inventory)
            except ValueError as rollback_error:
                raise ValueError(
                    "verification inventory temporary cleanup failed and exact "
                    f"rollback failed: {rollback_error}"
                ) from inventory_publication.cleanup_error
            raise ValueError(
                "verification inventory temporary cleanup failed"
            ) from inventory_publication.cleanup_error
    _require_snapshot_unchanged(root, base_snapshots)
    after_inventory = frozenset({*optional_snapshots, _INVENTORY_PATH})
    _require_exact_run_topology(
        root,
        optional_files=after_inventory,
        directory_identities=directory_identities,
    )
    if _snapshot_file(
        root / _INVENTORY_PATH, label="verification inventory"
    ) != inventory:
        raise ValueError("verification inventory changed before final seal")

    final = _final_bytes(
        inventory,
        run_id=run_id,
        git_commit=git_commit,
        report_completion_sha256=_sha256(base_snapshots["reports/completed.json"]),
        file_count=file_count,
        total_bytes=total_bytes,
    )
    existing_final = optional_snapshots.get(_FINAL_PATH)
    if existing_final is not None:
        if existing_final != final:
            raise ValueError("existing verified completion conflicts with replay")
        final_created = False
        final_publication = _AtomicCreateResult(None, None)
    else:
        final_publication = _atomic_create(root / _FINAL_PATH, final)
        final_created = True
    try:
        _require_snapshot_unchanged(root, base_snapshots)
        transient_files: frozenset[str] = frozenset()
        if final_publication.temporary_path is not None:
            temporary_relative = final_publication.temporary_path.relative_to(
                root
            ).as_posix()
            transient_files = frozenset({temporary_relative})
            if (
                _snapshot_file(
                    final_publication.temporary_path,
                    label="verification final temporary file",
                )
                != final
            ):
                raise ValueError(
                    "verification final temporary file changed after publication"
                )
        _require_exact_run_topology(
            root,
            optional_files=frozenset({_INVENTORY_PATH, _FINAL_PATH}),
            directory_identities=directory_identities,
            transient_files=transient_files,
        )
        _require_upstream_unchanged(upstream)
        if (
            _snapshot_file(
                root / _INVENTORY_PATH, label="verification inventory"
            )
            != inventory
        ):
            raise ValueError("verification inventory changed after final seal")
        if _snapshot_file(root / _FINAL_PATH, label="verified completion") != final:
            raise ValueError("verified completion changed after publication")
        if final_publication.cleanup_error is not None:
            raise ValueError(
                "verification final temporary cleanup failed after terminal "
                "revalidation"
            ) from final_publication.cleanup_error
    except Exception as publication_error:
        if final_created:
            try:
                _rollback_exact_new_file(root / _FINAL_PATH, final)
            except ValueError as rollback_error:
                raise ValueError(
                    "final validation failed and exact rollback failed: "
                    f"{rollback_error}"
                ) from publication_error
        raise
    return root / _FINAL_PATH


__all__ = ["INVENTORY_SCHEMA", "VERIFIED_SCHEMA", "verify_completed_run"]
