"""Fail-closed upstream authentication and Stage A-prime prepare publication."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Mapping, Sequence
import zipfile

import numpy as np
import torch

from pre_experiments.camera_translation_hvrfm.artifacts import (
    load_bound_bundle_bytes,
)
from pre_experiments.camera_translation_hvrfm.data import (
    PublishedTranslationSample,
    calibration_role,
    publish_translation_sample,
    validate_calibration_cohort,
)
from pre_experiments.camera_translation_hvrfm.teacher import (
    TeacherControls,
    _mint_teacher_controls,
    _scene_from_source,
)
from pre_experiments.common.model_io import find_checkpoint
from pre_experiments.conditional_hierarchical_vrfm.artifacts import (
    TEACHER_ARTIFACT_MEMBERS,
    _validate_teacher_artifact,
)
from pre_experiments.long_short_camera_head.data import (
    LONG_CONTEXT_MEMBERS as OLD_LONG_CONTEXT_MEMBERS,
    _validate_long_context as _validate_old_long_context,
)
from pre_experiments.long_short_camera_head.labels import (
    PRIVILEGED_MEMBERS,
    _validate_privileged,
)
from pre_experiments.long_short_camera_head.train import load_base_camera_head
from pre_experiments.variational_camera_latent.contracts import SourceShardRecord
from pre_experiments.variational_camera_latent.schema import (
    SOURCE_OPTIONAL_MEMBERS,
    SOURCE_REQUIRED_MEMBERS,
    SOURCE_SCHEMA,
    validate_source_shard,
)


PREFLIGHT_SCHEMA = "camera_translation_hvrfm.preflight_evidence.v1"
RUN_CONFIG_SCHEMA = "camera_translation_hvrfm.run_config.v1"
LONG_CONTEXT_MANIFEST_SCHEMA = (
    "camera_translation_hvrfm.long_context_manifest.v1"
)
COHORT_MANIFEST_SCHEMA = "camera_translation_hvrfm.cohort_manifest.v1"
PREPARE_COMPLETION_SCHEMA = "camera_translation_hvrfm.prepare_completion.v1"

FROZEN_SOURCE_RUN_NAME = "vrfm_camera_20260827T044926Z"
FROZEN_REFERENCE_RUN_NAME = (
    "privileged_teacher_lift_20260829T012716Z_tolfix"
)
FROZEN_FORMAL_RUN_NAME = "long_short_head_formal_20260828T072407Z"
FROZEN_SOURCE_COMPLETION_SHA256 = (
    "fd1b93caa16f45f0dbdc55fd7000aba9ab8bf166a7240f5ac2a716a0b3de9a32"
)
FROZEN_SOURCE_MANIFEST_SHA256 = (
    "be5aaa1b61be5e25709e40b3912e48aab38b6bbfac4be3b7ed183140219d6054"
)
FROZEN_REFERENCE_COMPLETION_SHA256 = (
    "7e63ca36e6fc4c08772e3356255f84c2853c9d46310ae546cc5e53dc1792048c"
)
FROZEN_FORMAL_COMPLETION_SHA256 = (
    "4d24b944792f348ccc8c180a99f3e0ee11397ce472900eb6abe38f6924732667"
)
FROZEN_FORMAL_DATA_MANIFEST_SHA256 = (
    "944ee57a75a68af45fc0ea6037070267552ea3f042bd2346638cdc65f2dd4a6e"
)
FROZEN_REFERENCE_LONG_MANIFEST_SHA256 = (
    "6b6ab434bb4cd8bd4afbeaf8a2d11354f321d8791501ada3ef2f9376eb064166"
)
FROZEN_CHECKPOINT_SHA256 = (
    "f164acf60724910d8fe1578bb499d800850c7bb0948db7555c413f9fbe60467e"
)
FROZEN_REFERENCE_GIT = "cee41a09ac4085c8d6b0b343ca07d8e8c53ace3c"
FROZEN_FORMAL_GIT = "2476a59f583ce4c39bbe66dc65d6a8e5cddfb52e"
FROZEN_BASIS_SHA256 = (
    "89fecc83d51be8a1923a0c177c20b45dec8d8aa611fae0b615ef9293511dd213"
)
FROZEN_REFERENCE_FILE_COUNT = 87
FROZEN_SCENES = (
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
SMOKE_SCENE = "scene0029_01"

PREFLIGHT_EVIDENCE_FIELDS = frozenset(
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
RUN_CONFIG_FIELDS = frozenset(
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
LONG_CONTEXT_MANIFEST_FIELDS = frozenset(
    {"schema", "run_id", "git_commit", "records"}
)
LONG_CONTEXT_RECORD_FIELDS = frozenset(
    {"sample_id", "scene", "role", "path", "sha256", "source_sha256"}
)
COHORT_MANIFEST_FIELDS = frozenset(
    {"schema", "run_id", "git_commit", "records"}
)
COHORT_RECORD_FIELDS = frozenset(
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
)
STAGE_COMPLETION_FIELDS = frozenset(
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
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


@dataclass(frozen=True)
class PipelineInputs:
    run_root: Path
    git_commit: str
    source_run: Path
    reference_run: Path
    formal_run: Path
    checkpoint_dir: Path
    expected_source_completion_sha256: str
    expected_reference_completion_sha256: str
    expected_formal_completion_sha256: str
    expected_checkpoint_sha256: str
    device: torch.device


@dataclass(frozen=True)
class AuthenticatedSceneInputs:
    scene: str
    role: str
    source_path: Path
    source_record: SourceShardRecord
    long_context_path: Path
    long_context_sha256: str
    teacher_reference_path: Path
    teacher_reference_sha256: str
    formal_label_path: Path
    formal_label_sha256: str


@dataclass(frozen=True)
class _FileSnapshot:
    path: Path
    payload: bytes
    sha256: str
    label: str
    identity: tuple[int, int, int, int, int]


@dataclass(frozen=True)
class _DirectorySnapshot:
    path: Path
    label: str
    identity: tuple[int, int, int, int, int]


@dataclass(frozen=True)
class _AuthenticatedUpstream:
    scenes: tuple[AuthenticatedSceneInputs, ...]
    snapshots: Mapping[str, Mapping[str, _FileSnapshot]]
    all_snapshots: tuple[_FileSnapshot, ...]
    checkpoint_snapshot: _FileSnapshot
    evidence: Mapping[str, object]


def _canonical_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a canonical lowercase SHA-256 digest")
    return value


def _canonical_commit(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _COMMIT_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a canonical lowercase Git commit")
    return value


def _canonical_digest(payload: object) -> str:
    try:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
            ensure_ascii=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("payload is not canonical JSON") from error
    return hashlib.sha256(encoded).hexdigest()


def _json_bytes(payload: Mapping[str, object]) -> bytes:
    try:
        return (
            json.dumps(
                dict(payload),
                indent=2,
                sort_keys=True,
                allow_nan=False,
                ensure_ascii=True,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("payload is not serializable canonical JSON") from error


def _absolute_without_resolving(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _reject_symlink_components(path: Path) -> None:
    candidate = Path(path)
    if ".." in candidate.parts:
        raise ValueError("authenticated paths may not contain lexical parent traversal")
    absolute = _absolute_without_resolving(candidate)
    for component in (absolute, *absolute.parents):
        if component.is_symlink():
            raise ValueError(f"authenticated paths may not contain symlinks: {component}")


def _sha256_file(path: Path) -> str:
    target = Path(path)
    _reject_symlink_components(target)
    if not target.is_file():
        raise ValueError(f"authenticated file must be regular: {target}")
    digest = hashlib.sha256()
    try:
        with target.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise ValueError(f"could not hash authenticated file: {target}") from error
    return digest.hexdigest()


def _snapshot_file(
    path: Path, expected_sha256: str | None, *, label: str
) -> _FileSnapshot:
    expected = (
        None
        if expected_sha256 is None
        else _canonical_sha256(
            expected_sha256, label=f"expected {label} digest"
        )
    )
    target = Path(path)
    _reject_symlink_components(target)
    if not target.is_file():
        raise ValueError(f"{label} must be a regular file")
    try:
        before = target.stat(follow_symlinks=False)
        with target.open("rb") as handle:
            opened_before = os.fstat(handle.fileno())
            payload = handle.read()
            opened_after = os.fstat(handle.fileno())
        after = target.stat(follow_symlinks=False)
    except OSError as error:
        raise ValueError(f"could not snapshot {label}") from error
    def identity(stat: os.stat_result) -> tuple[int, int, int, int, int]:
        return (
            int(stat.st_dev),
            int(stat.st_ino),
            int(stat.st_size),
            int(stat.st_mtime_ns),
            int(stat.st_ctime_ns),
        )
    signatures = {
        identity(before),
        identity(opened_before),
        identity(opened_after),
        identity(after),
    }
    if len(signatures) != 1:
        raise ValueError(f"{label} changed while taking its immutable snapshot")
    if type(payload) is not bytes:
        raise ValueError(f"{label} snapshot must be immutable bytes")
    actual = hashlib.sha256(payload).hexdigest()
    if expected is not None and actual != expected:
        raise ValueError(f"{label} digest mismatch")
    return _FileSnapshot(target.resolve(), payload, actual, label, signatures.pop())


def _require_snapshot_unchanged(snapshot: _FileSnapshot) -> None:
    try:
        current = snapshot.path.stat(follow_symlinks=False)
    except OSError as error:
        raise ValueError(f"{snapshot.label} changed during authentication") from error
    identity = (
        int(current.st_dev),
        int(current.st_ino),
        int(current.st_size),
        int(current.st_mtime_ns),
        int(current.st_ctime_ns),
    )
    if identity != snapshot.identity or _sha256_file(snapshot.path) != snapshot.sha256:
        raise ValueError(f"{snapshot.label} changed during authentication")


def _snapshot_directory(path: Path, *, label: str) -> _DirectorySnapshot:
    target = Path(path)
    _reject_symlink_components(target)
    if not target.is_dir():
        raise ValueError(f"{label} must be a directory")
    try:
        stat = target.stat(follow_symlinks=False)
    except OSError as error:
        raise ValueError(f"could not snapshot {label}") from error
    identity = (
        int(stat.st_dev),
        int(stat.st_ino),
        int(stat.st_size),
        int(stat.st_mtime_ns),
        int(stat.st_ctime_ns),
    )
    return _DirectorySnapshot(target.resolve(), label, identity)


def _require_directory_unchanged(snapshot: _DirectorySnapshot) -> None:
    try:
        current = snapshot.path.stat(follow_symlinks=False)
    except OSError as error:
        raise ValueError(f"{snapshot.label} changed during authentication") from error
    identity = (
        int(current.st_dev),
        int(current.st_ino),
        int(current.st_size),
        int(current.st_mtime_ns),
        int(current.st_ctime_ns),
    )
    if identity != snapshot.identity or not snapshot.path.is_dir():
        raise ValueError(f"{snapshot.label} changed during authentication")


def _snapshot_streaming_file(
    path: Path, expected_sha256: str | None, *, label: str
) -> _FileSnapshot:
    """Authenticate a large file without retaining its payload in memory."""
    expected = (
        None
        if expected_sha256 is None
        else _canonical_sha256(
            expected_sha256, label=f"expected {label} digest"
        )
    )
    target = Path(path)
    _reject_symlink_components(target)
    if not target.is_file():
        raise ValueError(f"{label} must be a regular file")
    digest = hashlib.sha256()
    try:
        before = target.stat(follow_symlinks=False)
        with target.open("rb") as handle:
            opened_before = os.fstat(handle.fileno())
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
            opened_after = os.fstat(handle.fileno())
        after = target.stat(follow_symlinks=False)
    except OSError as error:
        raise ValueError(f"could not snapshot {label}") from error

    def identity(stat: os.stat_result) -> tuple[int, int, int, int, int]:
        return (
            int(stat.st_dev),
            int(stat.st_ino),
            int(stat.st_size),
            int(stat.st_mtime_ns),
            int(stat.st_ctime_ns),
        )

    signatures = {
        identity(before),
        identity(opened_before),
        identity(opened_after),
        identity(after),
    }
    if len(signatures) != 1:
        raise ValueError(f"{label} changed while taking its streaming snapshot")
    actual = digest.hexdigest()
    if expected is not None and actual != expected:
        raise ValueError(f"{label} digest mismatch")
    return _FileSnapshot(target.resolve(), b"", actual, label, signatures.pop())


def _copy_streaming_snapshot(
    snapshot: _FileSnapshot, destination: Path
) -> _FileSnapshot:
    """Copy a previously authenticated large file from one stable descriptor."""
    _require_snapshot_unchanged(snapshot)
    target = Path(destination)
    _reject_symlink_components(target)
    if target.exists():
        raise ValueError(f"private checkpoint target already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(target.parent)
    digest = hashlib.sha256()
    try:
        source_before = snapshot.path.stat(follow_symlinks=False)
        with snapshot.path.open("rb") as source, target.open("xb") as output:
            opened_before = os.fstat(source.fileno())
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
                output.write(block)
            output.flush()
            os.fsync(output.fileno())
            opened_after = os.fstat(source.fileno())
        source_after = snapshot.path.stat(follow_symlinks=False)
    except OSError as error:
        raise ValueError("could not create the private checkpoint snapshot") from error
    source_identities = {
        (
            int(stat.st_dev),
            int(stat.st_ino),
            int(stat.st_size),
            int(stat.st_mtime_ns),
            int(stat.st_ctime_ns),
        )
        for stat in (source_before, opened_before, opened_after, source_after)
    }
    if source_identities != {snapshot.identity} or digest.hexdigest() != snapshot.sha256:
        raise ValueError("checkpoint changed while copying its private snapshot")
    private = _snapshot_streaming_file(
        target, snapshot.sha256, label="private checkpoint"
    )
    _require_snapshot_unchanged(snapshot)
    return private


def _json_object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f"JSON artifact contains duplicate key: {name}")
        result[name] = value
    return result


def _decode_json_snapshot(snapshot: _FileSnapshot) -> dict[str, object]:
    try:
        payload = json.loads(
            snapshot.payload.decode("utf-8"),
            object_pairs_hook=_json_object_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid {snapshot.label} JSON") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{snapshot.label} JSON root must be an object")
    return payload


def _decode_npz_snapshot(
    payload: bytes,
    expected_members: frozenset[str],
    *,
    label: str,
) -> dict[str, np.ndarray]:
    if type(payload) is not bytes:
        raise ValueError(f"{label} payload must be immutable bytes")
    expected_names = {f"{name}.npy" for name in expected_members}
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
            if len(names) != len(expected_names) or set(names) != expected_names:
                raise ValueError(f"{label} must use the exact schema")
        with np.load(BytesIO(payload), allow_pickle=False) as archive:
            arrays = {
                name: np.asarray(archive[name]).copy() for name in expected_members
            }
    except ValueError:
        raise
    except (OSError, KeyError, EOFError, zipfile.BadZipFile, zipfile.LargeZipFile) as error:
        raise ValueError(f"invalid {label} snapshot") from error
    if any(value.dtype.hasobject for value in arrays.values()):
        raise ValueError(f"{label} may not contain object arrays")
    return arrays


def _decode_source_snapshot(payload: bytes, **_: object) -> dict[str, np.ndarray]:
    members = frozenset(SOURCE_REQUIRED_MEMBERS | SOURCE_OPTIONAL_MEMBERS)
    arrays = _decode_npz_snapshot(payload, members, label="source shard")
    validate_source_shard(arrays)
    exact = {
        "global_frame_ids": ((500,), np.dtype(np.int64)),
        "global_camera_tokens": ((500, 2048), np.dtype(np.float32)),
        "short_frame_ids": ((9, 100), np.dtype(np.int64)),
        "short_camera_tokens": ((9, 100, 2048), np.dtype(np.float32)),
        "overlap_frame_ids": ((8, 50), np.dtype(np.int64)),
        "overlap_long_tokens": ((8, 50, 2048), np.dtype(np.float32)),
        "overlap_left_tokens": ((8, 50, 2048), np.dtype(np.float32)),
        "overlap_right_tokens": ((8, 50, 2048), np.dtype(np.float32)),
        "span_starts": ((8,), np.dtype(np.int64)),
        "sample_ids": ((8,), np.dtype("U64")),
        "global_pred_c2w": ((500, 4, 4), np.dtype(np.float64)),
        "overlap_long_c2w": ((8, 50, 4, 4), np.dtype(np.float64)),
    }
    for name, (shape, dtype) in exact.items():
        if arrays[name].shape != shape or arrays[name].dtype != dtype:
            raise ValueError(f"source shard {name} has a noncanonical dtype or shape")
    return arrays


def _decode_old_long_snapshot(payload: bytes, **_: object) -> dict[str, np.ndarray]:
    arrays = _decode_npz_snapshot(
        payload,
        frozenset(OLD_LONG_CONTEXT_MEMBERS),
        label="reference long-context artifact",
    )
    _validate_old_long_context(arrays)
    return arrays


def _decode_teacher_snapshot(
    payload: bytes,
    *,
    expected_source_sha256: str,
    expected_checkpoint_sha256: str,
    expected_formal_label_sha256: str,
    expected_teacher_sha256: str,
) -> TeacherControls:
    arrays = _decode_npz_snapshot(
        payload,
        frozenset(TEACHER_ARTIFACT_MEMBERS),
        label="teacher reference",
    )
    _validate_teacher_artifact(arrays)
    source_digest = _canonical_sha256(
        expected_source_sha256, label="teacher source binding"
    )
    checkpoint_digest = _canonical_sha256(
        expected_checkpoint_sha256, label="teacher checkpoint binding"
    )
    formal_digest = _canonical_sha256(
        expected_formal_label_sha256, label="teacher formal-label binding"
    )
    reference_digest = _canonical_sha256(
        expected_teacher_sha256, label="teacher reference binding"
    )
    exact = {
        "scene": ((), np.dtype("U32")),
        "frame_ids": ((500,), np.dtype(np.int64)),
        "window_weights": ((9,), np.dtype(np.float64)),
        "window_masks": ((4, 9), np.dtype(np.uint8)),
        "coverage_weights": ((4, 500), np.dtype(np.float64)),
        "source_sha256": ((), np.dtype("U64")),
        "checkpoint_sha256": ((), np.dtype("U64")),
        "formal_label_sha256": ((), np.dtype("U64")),
    }
    for name, (shape, dtype) in exact.items():
        if arrays[name].shape != shape or arrays[name].dtype != dtype:
            raise ValueError(
                f"teacher reference {name} must use exact canonical shape and dtype"
            )
    for name, expected in (
        ("source_sha256", source_digest),
        ("checkpoint_sha256", checkpoint_digest),
        ("formal_label_sha256", formal_digest),
    ):
        if str(arrays[name]) != expected:
            raise ValueError(f"teacher reference {name} binding mismatch")
    return _mint_teacher_controls(
        scene=str(arrays["scene"]),
        frame_ids=arrays["frame_ids"],
        window_weights=arrays["window_weights"],
        window_masks=arrays["window_masks"],
        expected_coverage_weights=arrays["coverage_weights"],
        source_sha256=source_digest,
        checkpoint_sha256=checkpoint_digest,
        formal_label_sha256=formal_digest,
        teacher_reference_sha256=reference_digest,
    )


def _decode_formal_label_snapshot(
    payload: bytes, **_: object
) -> dict[str, np.ndarray]:
    arrays = _decode_npz_snapshot(
        payload,
        frozenset(PRIVILEGED_MEMBERS),
        label="formal privileged label",
    )
    _validate_privileged(arrays)
    return arrays


def _require_fields(
    payload: Mapping[str, object], expected: frozenset[str], *, label: str
) -> None:
    if set(payload) != expected:
        raise ValueError(f"{label} schema fields are not exact")


def _require_signed_json(
    payload: Mapping[str, object], *, fields: frozenset[str], label: str
) -> None:
    _require_fields(payload, fields, label=label)
    unsigned = dict(payload)
    digest = unsigned.pop("completion_digest", None)
    if digest != _canonical_digest(unsigned):
        raise ValueError(f"{label} self-digest mismatch")


def _current_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("could not inspect the current Git commit") from error
    commit = result.stdout.strip()
    return _canonical_commit(commit, label="current Git commit")


def _resolve_inputs(inputs: PipelineInputs) -> dict[str, object]:
    if not isinstance(inputs, PipelineInputs):
        raise ValueError("inputs must be PipelineInputs")
    commit = _canonical_commit(inputs.git_commit, label="pipeline Git commit")
    if commit != _current_git_commit():
        raise ValueError("pipeline Git commit does not match the current repository HEAD")
    if not isinstance(inputs.device, torch.device):
        raise ValueError("pipeline device must be torch.device")
    expectations = {
        "source completion": (
            inputs.expected_source_completion_sha256,
            FROZEN_SOURCE_COMPLETION_SHA256,
        ),
        "reference completion": (
            inputs.expected_reference_completion_sha256,
            FROZEN_REFERENCE_COMPLETION_SHA256,
        ),
        "formal completion": (
            inputs.expected_formal_completion_sha256,
            FROZEN_FORMAL_COMPLETION_SHA256,
        ),
        "checkpoint": (
            inputs.expected_checkpoint_sha256,
            FROZEN_CHECKPOINT_SHA256,
        ),
    }
    for label, (provided, frozen) in expectations.items():
        canonical = _canonical_sha256(provided, label=f"expected {label}")
        if canonical != frozen:
            raise ValueError(f"expected {label} does not match the frozen experiment")
    paths = {
        "run_root": Path(inputs.run_root),
        "source_run": Path(inputs.source_run),
        "reference_run": Path(inputs.reference_run),
        "formal_run": Path(inputs.formal_run),
        "checkpoint_dir": Path(inputs.checkpoint_dir),
    }
    for path in paths.values():
        _reject_symlink_components(path)
    run_id = paths["run_root"].name
    if _RUN_ID_RE.fullmatch(run_id) is None or run_id in {".", ".."}:
        raise ValueError("run root must end in a canonical nonempty run ID")
    expected_names = {
        "source_run": FROZEN_SOURCE_RUN_NAME,
        "reference_run": FROZEN_REFERENCE_RUN_NAME,
        "formal_run": FROZEN_FORMAL_RUN_NAME,
    }
    for name, expected in expected_names.items():
        path = paths[name]
        if path.name != expected or not path.is_dir():
            raise ValueError(f"{name.replace('_', ' ')} is not the frozen upstream run")
    if not paths["checkpoint_dir"].is_dir():
        raise ValueError("checkpoint directory is unavailable")
    resolved = {name: path.resolve() for name, path in paths.items()}
    canonical_paths = tuple(resolved.values())
    for index, left in enumerate(canonical_paths):
        for right in canonical_paths[index + 1 :]:
            if left == right or left in right.parents or right in left.parents:
                raise ValueError(
                    "run root and upstream roots must be physically isolated"
                )
    return {**resolved, "run_id": run_id, "git_commit": commit}


def _expected_reference_inventory() -> frozenset[str]:
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
    for scene in FROZEN_SCENES:
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
    if len(files) != FROZEN_REFERENCE_FILE_COUNT:  # pragma: no cover - constant guard.
        raise RuntimeError("frozen reference inventory constant is inconsistent")
    return frozenset(files)


def _rows_by_scene(
    rows: object,
    *,
    fields: frozenset[str],
    label: str,
) -> dict[str, dict[str, object]]:
    if not isinstance(rows, list) or len(rows) != len(FROZEN_SCENES):
        raise ValueError(f"{label} must contain exactly ten records")
    result: dict[str, dict[str, object]] = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != fields:
            raise ValueError(f"{label} record schema is not exact")
        scene = row.get("scene")
        if not isinstance(scene, str) or scene not in FROZEN_SCENES or scene in result:
            raise ValueError(f"{label} scene cohort is invalid")
        role = row.get("role")
        if role != calibration_role(scene):
            raise ValueError(f"{label} role does not match the frozen split")
        result[scene] = dict(row)
    if set(result) != set(FROZEN_SCENES):
        raise ValueError(f"{label} scene cohort is incomplete")
    return result


def _reference_artifact_snapshots(
    reference_root: Path,
    files: Mapping[str, object],
) -> dict[str, _FileSnapshot]:
    expected = _expected_reference_inventory()
    if set(files) != expected:
        raise ValueError("reference inventory does not contain the exact 87-file cohort")
    retained = {
        "config.json",
        "manifests/long_context.json",
        "manifests/teacher.json",
        "reports/stage_a.json",
        "reports/stage_a.md",
        *(
            f"prediction_only/long_context/{scene}.npz"
            for scene in FROZEN_SCENES
        ),
        *(
            f"privileged_labels/teacher/{scene}.npz" for scene in FROZEN_SCENES
        ),
    }
    snapshots: dict[str, _FileSnapshot] = {}
    for relative in sorted(expected):
        digest = _canonical_sha256(
            files[relative], label=f"reference inventory digest for {relative}"
        )
        if relative in retained:
            snapshot = _snapshot_file(
                reference_root / relative,
                digest,
                label=f"reference inventory artifact {relative}",
            )
        else:
            snapshot = _snapshot_streaming_file(
                reference_root / relative,
                digest,
                label=f"reference inventory artifact {relative}",
            )
        snapshots[relative] = snapshot
    return snapshots


def _validate_source_completion(payload: Mapping[str, object]) -> None:
    _require_signed_json(
        payload,
        fields=_SOURCE_COMPLETION_FIELDS,
        label="source completion",
    )
    if (
        payload.get("schema") != "variational_camera_latent.verified_completion.v1"
        or payload.get("signal") != "WEAK_SIGNAL"
        or payload.get("scene_count") != 10
        or payload.get("overlap_count") != 80
        or payload.get("candidate_count") != 2560
    ):
        raise ValueError("source completion identity mismatch")
    for name in (
        "prediction_manifest_sha256",
        "privileged_manifest_sha256",
        "report_sha256",
    ):
        _canonical_sha256(payload.get(name), label=f"source completion {name}")


def _validate_formal_completion(
    payload: Mapping[str, object],
    *,
    source_manifest_sha256: str,
    formal_manifest_sha256: str,
    checkpoint_sha256: str,
) -> None:
    _require_fields(payload, _FORMAL_COMPLETION_FIELDS, label="formal completion")
    identities = (
        payload.get("schema") == "long_short_camera_head.verified_completion.v1",
        payload.get("git_revision") == FROZEN_FORMAL_GIT,
        isinstance(payload.get("verifier_git_revision"), str)
        and _COMMIT_RE.fullmatch(str(payload.get("verifier_git_revision"))) is not None,
        payload.get("source_manifest_sha256") == source_manifest_sha256,
        payload.get("base_checkpoint_sha256") == checkpoint_sha256,
        payload.get("data_manifest_sha256") == formal_manifest_sha256,
        payload.get("classification") == "NO_SOURCE_HEAD_SIGNAL",
        payload.get("scene_count") == 10,
        payload.get("train_scene_count") == 8,
        payload.get("locked_replay_scene_count") == 2,
        payload.get("inference_leakage_audit") is True,
    )
    if not all(identities):
        raise ValueError("formal completion identity mismatch")
    for name in (
        "config_sha256",
        "test_evidence_sha256",
        "report_sha256",
        "formal_protocol_sha256",
    ):
        _canonical_sha256(payload.get(name), label=f"formal completion {name}")
    stages = payload.get("stage_completion_sha256")
    expected_stages = {
        "evaluation_gt_only",
        "evaluation_long_short",
        "smoke",
        "training_gt_only",
        "training_long_short",
    }
    if not isinstance(stages, dict) or set(stages) != expected_stages:
        raise ValueError("formal completion stage bindings are not exact")
    for name, digest in stages.items():
        _canonical_sha256(digest, label=f"formal stage {name}")
    artifacts = payload.get("artifacts")
    artifact_fields = {
        "scene",
        "variant",
        "checkpoint_sha256",
        "prediction_sha256",
        "evaluation_sha256",
    }
    if not isinstance(artifacts, list) or len(artifacts) != 20:
        raise ValueError("formal completion artifact cohort is not exact")
    identities_seen: set[tuple[str, str]] = set()
    for row in artifacts:
        if not isinstance(row, dict) or set(row) != artifact_fields:
            raise ValueError("formal completion artifact schema is not exact")
        identity = (str(row.get("scene")), str(row.get("variant")))
        if (
            identity[0] not in FROZEN_SCENES
            or identity[1] not in {"gt_only", "long_short"}
            or identity in identities_seen
        ):
            raise ValueError("formal completion artifact identities are invalid")
        identities_seen.add(identity)
        for name in (
            "checkpoint_sha256",
            "prediction_sha256",
            "evaluation_sha256",
        ):
            _canonical_sha256(row.get(name), label=f"formal artifact {name}")


def _same_resolved_path(value: object, expected: Path) -> bool:
    if not isinstance(value, str):
        return False
    try:
        candidate = Path(value)
        _reject_symlink_components(candidate)
        return candidate.resolve() == expected.resolve()
    except (OSError, ValueError):
        return False


def _authenticate_upstream_full(inputs: PipelineInputs) -> _AuthenticatedUpstream:
    resolved = _resolve_inputs(inputs)
    source_root = resolved["source_run"]
    reference_root = resolved["reference_run"]
    formal_root = resolved["formal_run"]
    checkpoint_root = resolved["checkpoint_dir"]
    assert isinstance(source_root, Path)
    assert isinstance(reference_root, Path)
    assert isinstance(formal_root, Path)
    assert isinstance(checkpoint_root, Path)

    source_completion_snapshot = _snapshot_file(
        source_root / "verified_completion.json",
        inputs.expected_source_completion_sha256,
        label="source completion",
    )
    source_completion = _decode_json_snapshot(source_completion_snapshot)
    _validate_source_completion(source_completion)
    source_manifest_snapshot = _snapshot_file(
        source_root / "manifests" / "source_manifest.json",
        FROZEN_SOURCE_MANIFEST_SHA256,
        label="source manifest",
    )
    source_manifest = _decode_json_snapshot(source_manifest_snapshot)
    _require_fields(source_manifest, _SOURCE_MANIFEST_FIELDS, label="source manifest")
    if source_manifest.get("schema") != SOURCE_SCHEMA:
        raise ValueError("source manifest schema mismatch")
    _canonical_sha256(
        source_manifest.get("source_run_digest"), label="source run digest"
    )
    source_rows = _rows_by_scene(
        source_manifest.get("records"),
        fields=_SOURCE_RECORD_FIELDS,
        label="source manifest",
    )
    historic_root = str(source_manifest.get("dataset_root", "")).replace("\\", "/").rstrip("/")
    if not historic_root:
        raise ValueError("source manifest historical dataset root is missing")
    for scene, row in source_rows.items():
        historical = str(row["path"]).replace("\\", "/")
        if historical != f"{historic_root}/{scene}.npz":
            raise ValueError("source manifest historical path binding mismatch")
        if row["overlap_count"] != 8:
            raise ValueError("source manifest overlap count must be eight")
        _canonical_sha256(row["sha256"], label="source shard digest")

    reference_completion_snapshot = _snapshot_file(
        reference_root / "verified_completion.json",
        inputs.expected_reference_completion_sha256,
        label="reference completion",
    )
    reference_completion = _decode_json_snapshot(reference_completion_snapshot)
    _require_signed_json(
        reference_completion,
        fields=_REFERENCE_COMPLETION_FIELDS,
        label="reference completion",
    )
    if (
        reference_completion.get("schema")
        != "conditional_hierarchical_vrfm.verified_completion.v1"
        or reference_completion.get("git_commit") != FROZEN_REFERENCE_GIT
        or reference_completion.get("classification") != "LATENT_LIFT_FAILED"
        or reference_completion.get("file_count") != FROZEN_REFERENCE_FILE_COUNT
    ):
        raise ValueError("reference completion identity mismatch")
    inventory_snapshot = _snapshot_file(
        reference_root / "manifests" / "verification_inventory.json",
        _canonical_sha256(
            reference_completion.get("inventory_sha256"),
            label="reference inventory binding",
        ),
        label="reference inventory",
    )
    inventory = _decode_json_snapshot(inventory_snapshot)
    _require_fields(inventory, _REFERENCE_INVENTORY_FIELDS, label="reference inventory")
    if (
        inventory.get("schema")
        != "conditional_hierarchical_vrfm.verification_inventory.v1"
        or inventory.get("git_commit") != FROZEN_REFERENCE_GIT
        or inventory.get("classification") != "LATENT_LIFT_FAILED"
        or not isinstance(inventory.get("files"), dict)
        or len(inventory["files"]) != FROZEN_REFERENCE_FILE_COUNT
    ):
        raise ValueError("reference inventory identity mismatch")
    reference_snapshots = _reference_artifact_snapshots(
        reference_root, inventory["files"]
    )

    config = _decode_json_snapshot(reference_snapshots["config.json"])
    long_manifest = _decode_json_snapshot(
        reference_snapshots["manifests/long_context.json"]
    )
    teacher_manifest = _decode_json_snapshot(
        reference_snapshots["manifests/teacher.json"]
    )
    report = _decode_json_snapshot(reference_snapshots["reports/stage_a.json"])
    _require_fields(config, _REFERENCE_CONFIG_FIELDS, label="reference config")
    _require_fields(long_manifest, _REFERENCE_LONG_FIELDS, label="reference long manifest")
    _require_fields(
        teacher_manifest,
        _REFERENCE_TEACHER_FIELDS,
        label="reference teacher manifest",
    )
    _require_fields(report, _REFERENCE_REPORT_FIELDS, label="reference report")

    long_manifest_digest = reference_snapshots[
        "manifests/long_context.json"
    ].sha256
    teacher_manifest_digest = reference_snapshots["manifests/teacher.json"].sha256
    if long_manifest_digest != FROZEN_REFERENCE_LONG_MANIFEST_SHA256:
        raise ValueError("reference long-context manifest is not the frozen manifest")
    config_identity = (
        config.get("schema") == "conditional_hierarchical_vrfm.run_config.v1",
        config.get("git_commit") == FROZEN_REFERENCE_GIT,
        config.get("checkpoint_sha256") == inputs.expected_checkpoint_sha256,
        config.get("basis_sha256") == FROZEN_BASIS_SHA256,
        config.get("long_manifest_sha256") == long_manifest_digest,
        config.get("teacher_manifest_sha256") == teacher_manifest_digest,
        config.get("source_manifest_sha256") == source_manifest_snapshot.sha256,
        config.get("formal_completion_sha256")
        == inputs.expected_formal_completion_sha256,
        config.get("formal_data_manifest_sha256")
        == FROZEN_FORMAL_DATA_MANIFEST_SHA256,
        _same_resolved_path(config.get("source_run"), source_root),
        _same_resolved_path(config.get("formal_run_root"), formal_root),
        config.get("smoke_scene") == "scene0000_00",
        config.get("smoke_steps") == 20,
        config.get("calibration_steps") == 250,
        config.get("scene_count") == 10,
        config.get("variant_count") == 4,
    )
    if not all(config_identity):
        raise ValueError("reference config digest/provenance binding mismatch")

    long_rows = _rows_by_scene(
        long_manifest.get("records"),
        fields=_REFERENCE_LONG_RECORD_FIELDS,
        label="reference long manifest",
    )
    if long_manifest.get("schema") != (
        "conditional_hierarchical_vrfm.long_context_manifest.v1"
    ):
        raise ValueError("reference long manifest schema mismatch")
    teacher_rows = _rows_by_scene(
        teacher_manifest.get("records"),
        fields=_REFERENCE_TEACHER_RECORD_FIELDS,
        label="reference teacher manifest",
    )
    summary = teacher_manifest.get("teacher_upper_bound")
    expected_summary = {
        "scene_count": 10,
        "positive_scene_count": 10,
        "mean_coverage": 0.89,
        "mean_utility": 0.1293578270771188,
    }
    teacher_identity = (
        teacher_manifest.get("schema")
        == "conditional_hierarchical_vrfm.teacher_manifest.v1",
        teacher_manifest.get("git_commit") == FROZEN_REFERENCE_GIT,
        teacher_manifest.get("checkpoint_sha256")
        == inputs.expected_checkpoint_sha256,
        teacher_manifest.get("formal_completion_sha256")
        == inputs.expected_formal_completion_sha256,
        teacher_manifest.get("formal_data_manifest_sha256")
        == FROZEN_FORMAL_DATA_MANIFEST_SHA256,
        summary == expected_summary,
    )
    if not all(teacher_identity):
        raise ValueError("reference teacher manifest provenance binding mismatch")

    expected_failed_gates = [
        "teacher_retention",
        "per_scene_harm",
        "rotation_guard",
        "uncovered_anchor",
    ]
    provenance = report.get("provenance")
    report_rows = report.get("scene_metrics")
    if (
        report.get("schema") != "conditional_hierarchical_vrfm.stage_a_report.v1"
        or report.get("git_commit") != FROZEN_REFERENCE_GIT
        or report.get("classification") != "LATENT_LIFT_FAILED"
        or report.get("failed_gates") != expected_failed_gates
        or not isinstance(provenance, dict)
        or provenance
        != {
            "checkpoint_sha256": inputs.expected_checkpoint_sha256,
            "basis_sha256": FROZEN_BASIS_SHA256,
            "long_manifest_sha256": long_manifest_digest,
            "teacher_manifest_sha256": teacher_manifest_digest,
        }
        or not isinstance(report_rows, list)
        or len(report_rows) != 10
        or {row.get("scene") for row in report_rows if isinstance(row, dict)}
        != set(FROZEN_SCENES)
    ):
        raise ValueError("reference report provenance/classification mismatch")

    formal_completion_snapshot = _snapshot_file(
        formal_root / "verified_completion.json",
        inputs.expected_formal_completion_sha256,
        label="formal completion",
    )
    formal_manifest_snapshot = _snapshot_file(
        formal_root / "manifests" / "data_manifest.json",
        FROZEN_FORMAL_DATA_MANIFEST_SHA256,
        label="formal data manifest",
    )
    formal_completion = _decode_json_snapshot(formal_completion_snapshot)
    formal_manifest = _decode_json_snapshot(formal_manifest_snapshot)
    _validate_formal_completion(
        formal_completion,
        source_manifest_sha256=source_manifest_snapshot.sha256,
        formal_manifest_sha256=formal_manifest_snapshot.sha256,
        checkpoint_sha256=inputs.expected_checkpoint_sha256,
    )
    _require_fields(formal_manifest, _FORMAL_MANIFEST_FIELDS, label="formal data manifest")
    if (
        formal_manifest.get("schema") != "long_short_camera_head.data_manifest.v1"
        or formal_manifest.get("git_revision") != FROZEN_FORMAL_GIT
        or formal_manifest.get("source_manifest_sha256")
        != source_manifest_snapshot.sha256
        or formal_manifest.get("base_checkpoint_sha256")
        != inputs.expected_checkpoint_sha256
        or not _same_resolved_path(formal_manifest.get("source_run"), source_root)
        or not _same_resolved_path(
            formal_manifest.get("checkpoint_dir"), checkpoint_root
        )
    ):
        raise ValueError("formal data manifest provenance binding mismatch")
    formal_rows = _rows_by_scene(
        formal_manifest.get("records"),
        fields=_FORMAL_RECORD_FIELDS,
        label="formal data manifest",
    )

    checkpoint_file = find_checkpoint(checkpoint_root)
    _reject_symlink_components(checkpoint_file)
    checkpoint_snapshot = _snapshot_streaming_file(
        checkpoint_file,
        inputs.expected_checkpoint_sha256,
        label="checkpoint",
    )

    authenticated: list[AuthenticatedSceneInputs] = []
    scene_snapshots: dict[str, dict[str, _FileSnapshot]] = {}
    for scene in FROZEN_SCENES:
        source_row = source_rows[scene]
        long_row = long_rows[scene]
        teacher_row = teacher_rows[scene]
        formal_row = formal_rows[scene]
        role = calibration_role(scene)
        source_digest = str(source_row["sha256"])
        source_path = source_root / "prediction_only" / "source" / f"{scene}.npz"
        source_snapshot = _snapshot_file(
            source_path, source_digest, label=f"source shard {scene}"
        )
        source_arrays = _decode_source_snapshot(source_snapshot.payload, scene=scene)
        if _scene_from_source(source_arrays) != scene:
            raise ValueError("source shard scene identity mismatch")

        if (
            long_row["file"] != f"{scene}.npz"
            or long_row["source_sha256"] != source_digest
        ):
            raise ValueError("reference long-context source/path binding mismatch")
        long_digest = _canonical_sha256(
            long_row["sha256"], label="reference long-context digest"
        )
        long_relative = f"prediction_only/long_context/{scene}.npz"
        long_snapshot = reference_snapshots[long_relative]
        if long_snapshot.sha256 != long_digest:
            raise ValueError("reference long-context inventory binding mismatch")
        long_arrays = _decode_old_long_snapshot(long_snapshot.payload, scene=scene)
        if (
            str(long_arrays["scene"]) != scene
            or str(long_arrays["source_sha256"]) != source_digest
        ):
            raise ValueError("reference long-context semantic binding mismatch")

        formal_digest = _canonical_sha256(
            formal_row["privileged_sha256"], label="formal label digest"
        )
        formal_path = formal_root / "data" / "privileged_labels" / f"{scene}.npz"
        if (
            formal_row["source_sha256"] != source_digest
            or formal_row["long_context_sha256"] != long_digest
            or formal_row["role"] != role
            or formal_row["teacher_frame_count"] is True
            or not isinstance(formal_row["teacher_frame_count"], int)
            or int(formal_row["teacher_frame_count"]) <= 0
            or not _same_resolved_path(formal_row["source_path"], source_path)
            or not _same_resolved_path(
                formal_row["long_context_path"],
                formal_root / "data" / "long_context" / f"{scene}.npz",
            )
            or not _same_resolved_path(formal_row["privileged_path"], formal_path)
        ):
            raise ValueError("formal scene record binding mismatch")
        formal_snapshot = _snapshot_file(
            formal_path, formal_digest, label=f"formal label {scene}"
        )
        formal_arrays = _decode_formal_label_snapshot(
            formal_snapshot.payload, scene=scene
        )
        if (
            str(formal_arrays["scene"]) != scene
            or str(formal_arrays["source_sha256"]) != source_digest
            or str(formal_arrays["checkpoint_sha256"])
            != inputs.expected_checkpoint_sha256
        ):
            raise ValueError("formal label semantic binding mismatch")

        if (
            teacher_row["file"] != f"privileged_labels/teacher/{scene}.npz"
            or teacher_row["formal_label_sha256"] != formal_digest
        ):
            raise ValueError("teacher reference path/formal binding mismatch")
        teacher_digest = _canonical_sha256(
            teacher_row["sha256"], label="teacher reference digest"
        )
        teacher_snapshot = reference_snapshots[str(teacher_row["file"])]
        if teacher_snapshot.sha256 != teacher_digest:
            raise ValueError("teacher reference inventory binding mismatch")
        controls = _decode_teacher_snapshot(
            teacher_snapshot.payload,
            expected_source_sha256=source_digest,
            expected_checkpoint_sha256=inputs.expected_checkpoint_sha256,
            expected_formal_label_sha256=formal_digest,
            expected_teacher_sha256=teacher_digest,
        )
        if controls.scene != scene:
            raise ValueError("teacher reference scene identity mismatch")

        for snapshot in (
            source_snapshot,
            long_snapshot,
            teacher_snapshot,
            formal_snapshot,
        ):
            _require_snapshot_unchanged(snapshot)
        source_record = SourceShardRecord(
            scene=scene,
            role=role,
            path=source_path.resolve(),
            overlap_count=8,
            sha256=source_digest,
        )
        authenticated.append(
            AuthenticatedSceneInputs(
                scene=scene,
                role=role,
                source_path=source_path.resolve(),
                source_record=source_record,
                long_context_path=long_snapshot.path,
                long_context_sha256=long_digest,
                teacher_reference_path=teacher_snapshot.path,
                teacher_reference_sha256=teacher_digest,
                formal_label_path=formal_snapshot.path,
                formal_label_sha256=formal_digest,
            )
        )
        scene_snapshots[scene] = {
            "source": source_snapshot,
            "long": long_snapshot,
            "teacher": teacher_snapshot,
            "formal": formal_snapshot,
        }

    all_snapshots = (
        source_completion_snapshot,
        source_manifest_snapshot,
        reference_completion_snapshot,
        inventory_snapshot,
        formal_completion_snapshot,
        formal_manifest_snapshot,
        *reference_snapshots.values(),
        *(
            scene_snapshots[scene][name]
            for scene in FROZEN_SCENES
            for name in ("source", "formal")
        ),
        checkpoint_snapshot,
    )
    for snapshot in all_snapshots:
        _require_snapshot_unchanged(snapshot)

    scene_bindings = [
        {
            "scene": row.scene,
            "role": row.role,
            "source_sha256": row.source_record.sha256,
            "long_context_sha256": row.long_context_sha256,
            "teacher_reference_sha256": row.teacher_reference_sha256,
            "formal_label_sha256": row.formal_label_sha256,
        }
        for row in authenticated
    ]
    evidence = {
        "source_run": str(source_root),
        "source_completion_sha256": source_completion_snapshot.sha256,
        "source_manifest_sha256": source_manifest_snapshot.sha256,
        "reference_run": str(reference_root),
        "reference_completion_sha256": reference_completion_snapshot.sha256,
        "reference_inventory_sha256": inventory_snapshot.sha256,
        "reference_config_sha256": reference_snapshots["config.json"].sha256,
        "reference_report_json_sha256": reference_snapshots[
            "reports/stage_a.json"
        ].sha256,
        "reference_report_markdown_sha256": reference_snapshots[
            "reports/stage_a.md"
        ].sha256,
        "reference_long_manifest_sha256": long_manifest_digest,
        "reference_teacher_manifest_sha256": teacher_manifest_digest,
        "formal_run": str(formal_root),
        "formal_completion_sha256": formal_completion_snapshot.sha256,
        "formal_data_manifest_sha256": formal_manifest_snapshot.sha256,
        "checkpoint_file": str(checkpoint_snapshot.path),
        "checkpoint_sha256": inputs.expected_checkpoint_sha256,
        "scene_bindings": scene_bindings,
    }
    return _AuthenticatedUpstream(
        scenes=tuple(authenticated),
        snapshots=scene_snapshots,
        all_snapshots=all_snapshots,
        checkpoint_snapshot=checkpoint_snapshot,
        evidence=evidence,
    )


def authenticate_upstream(
    inputs: PipelineInputs,
) -> tuple[AuthenticatedSceneInputs, ...]:
    """Authenticate the frozen failed-run graph and return canonical scene inputs."""
    return _authenticate_upstream_full(inputs).scenes


def _publish_bytes_create_absent(path: Path, content: bytes) -> None:
    """Publish bytes atomically without any overwrite-capable operation."""
    if type(content) is not bytes:
        raise ValueError("publication content must be immutable bytes")
    target = Path(path)
    _reject_symlink_components(target)
    if target.exists():
        raise ValueError(f"publication target already exists: {target}")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise ValueError(f"could not create publication directory: {target.parent}") from error
    _reject_symlink_components(target.parent)
    temporary: Path | None = None
    publication_error: Exception | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, target)
    except Exception as error:
        publication_error = error
    cleanup_error: OSError | None = None
    if temporary is not None:
        try:
            temporary.unlink(missing_ok=True)
        except OSError as error:
            cleanup_error = error
    if publication_error is not None:
        detail = " and temporary cleanup failed" if cleanup_error else ""
        raise ValueError(
            f"could not create publication target without overwrite{detail}: {target}"
        ) from publication_error
    if cleanup_error is not None:
        raise ValueError(
            f"publication succeeded but temporary cleanup failed: {target}"
        ) from cleanup_error


def _require_exact_bytes(expected: Mapping[Path, bytes]) -> None:
    for path, content in expected.items():
        target = Path(path)
        _reject_symlink_components(target)
        if not target.is_file():
            raise ValueError(f"published stage artifact is missing: {target}")
        try:
            actual = target.read_bytes()
        except OSError as error:
            raise ValueError(f"could not revalidate stage artifact: {target}") from error
        if actual != content:
            raise ValueError(f"published stage artifact changed: {target}")


def _rollback_exact_new_file(path: Path, content: bytes) -> None:
    """Remove only the exact completion bytes just published by this call."""
    target = Path(path)
    _reject_symlink_components(target)
    if not target.is_file():
        raise ValueError(f"new completion target cannot be rolled back: {target}")
    try:
        current = target.read_bytes()
    except OSError as error:
        raise ValueError(
            f"could not authenticate new completion target: {target}"
        ) from error
    if current != content:
        raise ValueError(f"new completion target changed before rollback: {target}")
    try:
        target.unlink()
    except OSError as error:
        raise ValueError(f"could not roll back new completion target: {target}") from error


def _rollback_empty_preflight_prefix(root: Path) -> None:
    """Remove only the empty directory prefix created by this preflight call."""
    target = Path(root)
    _reject_symlink_components(target)
    if not target.exists():
        return
    if not target.is_dir():
        raise ValueError("new preflight run root is no longer a directory")
    entries = _root_entries(target)
    if entries not in (set(), {"manifests"}):
        raise ValueError("new preflight run root contains foreign or published artifacts")
    manifests = target / "manifests"
    if manifests.exists():
        _reject_symlink_components(manifests)
        if not manifests.is_dir() or any(manifests.iterdir()):
            raise ValueError("new preflight manifests prefix is not an empty directory")
        try:
            manifests.rmdir()
        except OSError as error:
            raise ValueError("could not remove empty preflight manifests prefix") from error
    try:
        target.rmdir()
    except OSError as error:
        raise ValueError("could not remove empty preflight run root") from error


def _preflight_payload(
    inputs: PipelineInputs, upstream: _AuthenticatedUpstream
) -> dict[str, object]:
    run_root = Path(inputs.run_root)
    unsigned: dict[str, object] = {
        "schema": PREFLIGHT_SCHEMA,
        "stage": "preflight",
        "run_id": run_root.name,
        "git_commit": inputs.git_commit,
        **dict(upstream.evidence),
    }
    payload = {**unsigned, "completion_digest": _canonical_digest(unsigned)}
    if set(payload) != PREFLIGHT_EVIDENCE_FIELDS:  # pragma: no cover - schema guard.
        raise RuntimeError("internal preflight evidence schema is inconsistent")
    return payload


def _validate_preflight_payload(
    payload: Mapping[str, object], *, run_id: str, git_commit: str
) -> None:
    _require_signed_json(
        payload, fields=PREFLIGHT_EVIDENCE_FIELDS, label="preflight evidence"
    )
    if (
        payload.get("schema") != PREFLIGHT_SCHEMA
        or payload.get("stage") != "preflight"
        or payload.get("run_id") != run_id
        or payload.get("git_commit") != git_commit
    ):
        raise ValueError("preflight evidence stage identity mismatch")
    rows = payload.get("scene_bindings")
    fields = {
        "scene",
        "role",
        "source_sha256",
        "long_context_sha256",
        "teacher_reference_sha256",
        "formal_label_sha256",
    }
    if not isinstance(rows, list) or len(rows) != 10:
        raise ValueError("preflight evidence must bind ten scenes")
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != fields:
            raise ValueError("preflight scene binding schema is not exact")
        scene = FROZEN_SCENES[index]
        if row.get("scene") != scene or row.get("role") != calibration_role(scene):
            raise ValueError("preflight scene identity/order mismatch")
        for name in (
            "source_sha256",
            "long_context_sha256",
            "teacher_reference_sha256",
            "formal_label_sha256",
        ):
            _canonical_sha256(row.get(name), label=f"preflight {name}")
    for name in (
        "source_completion_sha256",
        "source_manifest_sha256",
        "reference_completion_sha256",
        "reference_inventory_sha256",
        "reference_config_sha256",
        "reference_report_json_sha256",
        "reference_report_markdown_sha256",
        "reference_long_manifest_sha256",
        "reference_teacher_manifest_sha256",
        "formal_completion_sha256",
        "formal_data_manifest_sha256",
        "checkpoint_sha256",
    ):
        _canonical_sha256(payload.get(name), label=f"preflight {name}")
    frozen = {
        "source_completion_sha256": FROZEN_SOURCE_COMPLETION_SHA256,
        "source_manifest_sha256": FROZEN_SOURCE_MANIFEST_SHA256,
        "reference_completion_sha256": FROZEN_REFERENCE_COMPLETION_SHA256,
        "reference_long_manifest_sha256": FROZEN_REFERENCE_LONG_MANIFEST_SHA256,
        "formal_completion_sha256": FROZEN_FORMAL_COMPLETION_SHA256,
        "formal_data_manifest_sha256": FROZEN_FORMAL_DATA_MANIFEST_SHA256,
        "checkpoint_sha256": FROZEN_CHECKPOINT_SHA256,
    }
    if any(payload.get(name) != digest for name, digest in frozen.items()):
        raise ValueError("preflight evidence does not bind the frozen upstream digests")
    path_expectations = {
        "source_run": FROZEN_SOURCE_RUN_NAME,
        "reference_run": FROZEN_REFERENCE_RUN_NAME,
        "formal_run": FROZEN_FORMAL_RUN_NAME,
    }
    resolved_paths: list[Path] = []
    for name, basename in path_expectations.items():
        value = payload.get(name)
        if not isinstance(value, str):
            raise ValueError(f"preflight {name} path is malformed")
        path = Path(value)
        _reject_symlink_components(path)
        if path.name != basename or not path.is_dir():
            raise ValueError(f"preflight {name} path is not the frozen run")
        resolved_paths.append(path.resolve())
    checkpoint_value = payload.get("checkpoint_file")
    if not isinstance(checkpoint_value, str):
        raise ValueError("preflight checkpoint path is malformed")
    checkpoint_path = Path(checkpoint_value)
    _reject_symlink_components(checkpoint_path)
    if checkpoint_path.name not in {"model.safetensors", "model.pt"} or not checkpoint_path.is_file():
        raise ValueError("preflight checkpoint path is not a supported regular file")
    resolved_paths.append(checkpoint_path.resolve())
    if len(set(resolved_paths)) != len(resolved_paths):
        raise ValueError("preflight upstream paths are not physically distinct")


def _root_entries(root: Path) -> set[str]:
    if not root.is_dir():
        return set()
    return {path.relative_to(root).as_posix() for path in root.rglob("*")}


def _run_root_state(root: Path) -> str:
    _reject_symlink_components(root)
    if not root.exists():
        return "absent"
    if not root.is_dir():
        raise ValueError("existing run root is not a directory; use a new run ID")
    evidence = "manifests/preflight_evidence.json"
    entries = _root_entries(root)
    if entries == {"manifests", evidence}:
        return "preflight"
    if "prepare/completed.json" in entries:
        return "prepared"
    raise ValueError(
        "existing run root has no complete recognized marker prefix; preserve it and use a new run ID"
    )


def _load_json_path(path: Path, *, label: str) -> tuple[_FileSnapshot, dict[str, object]]:
    target = Path(path)
    snapshot = _snapshot_file(target, None, label=label)
    return snapshot, _decode_json_snapshot(snapshot)


def run_preflight(inputs: PipelineInputs) -> Path:
    """Authenticate everything first, then create only signed preflight evidence."""
    root = Path(inputs.run_root)
    state = _run_root_state(root)
    if state == "absent" and not root.parent.is_dir():
        raise ValueError("run-root parent must already exist")
    upstream = _authenticate_upstream_full(inputs)
    payload = _preflight_payload(inputs, upstream)
    content = _json_bytes(payload)
    evidence_path = root / "manifests" / "preflight_evidence.json"
    if state == "absent":
        try:
            root.mkdir(exist_ok=False)
        except (FileExistsError, OSError) as error:
            raise ValueError(
                "run root appeared after preflight; preserve it and use a new run ID"
            ) from error
        try:
            _publish_bytes_create_absent(evidence_path, content)
            evidence_snapshot = _snapshot_file(
                evidence_path,
                hashlib.sha256(content).hexdigest(),
                label="published preflight evidence",
            )
            _require_snapshot_unchanged(evidence_snapshot)
        except Exception as publication_error:
            try:
                _rollback_empty_preflight_prefix(root)
            except ValueError as rollback_error:
                raise ValueError(
                    "preflight publication failed and run-root rollback failed: "
                    f"{rollback_error}"
                ) from publication_error
            raise
        return evidence_path
    evidence_snapshot = _snapshot_file(
        evidence_path,
        hashlib.sha256(content).hexdigest(),
        label="existing preflight evidence",
    )
    if evidence_snapshot.payload != content:
        raise ValueError("existing preflight evidence does not match authenticated inputs")
    _validate_preflight_payload(
        payload, run_id=root.name, git_commit=inputs.git_commit
    )
    if state == "prepared":
        load_published_cohort(root)
    _require_snapshot_unchanged(evidence_snapshot)
    return evidence_path


def _canonical_sample_paths(root: Path, scene: str) -> dict[str, Path]:
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


def _require_pristine_prepare_prefix(root: Path) -> None:
    expected = {"manifests", "manifests/preflight_evidence.json"}
    if _root_entries(root) != expected:
        raise ValueError(
            "prepare artifacts exist without a complete marker; preserve them and use a new run ID"
        )


def _copy_authenticated_snapshot(path: Path, snapshot: _FileSnapshot) -> None:
    _publish_bytes_create_absent(path, snapshot.payload)
    _require_exact_bytes({path: snapshot.payload})


def _snapshot_staged_sample(
    published: PublishedTranslationSample,
    *,
    staging_root: Path,
    expected_scene: str,
    expected_role: str,
) -> tuple[dict[str, bytes], dict[str, str]]:
    if not isinstance(published, PublishedTranslationSample):
        raise ValueError("publisher returned a malformed sample record")
    if (
        published.scene != expected_scene
        or published.role != expected_role
        or published.sample_id != f"{expected_scene}:frames_500"
    ):
        raise ValueError("publisher returned the wrong sample identity")
    expected_paths = _canonical_sample_paths(staging_root, expected_scene)
    actual_paths = {
        "long": Path(published.long_path),
        "short": Path(published.short_path),
        "quality": Path(published.quality_path),
        "target": Path(published.target_path),
    }
    if any(actual_paths[name] != expected_paths[name] for name in actual_paths):
        raise ValueError("publisher returned a noncanonical staging path")
    expected_digests = {
        "long": published.long_sha256,
        "short": published.short_sha256,
        "quality": published.quality_sha256,
        "target": published.target_sha256,
    }
    snapshots = {
        name: _snapshot_file(
            actual_paths[name],
            _canonical_sha256(
                expected_digests[name], label=f"staged {name} digest"
            ),
            label=f"staged {name} artifact {expected_scene}",
        )
        for name in actual_paths
    }
    load_bound_bundle_bytes(
        snapshots["long"].payload,
        snapshots["short"].payload,
        snapshots["target"].payload,
        snapshots["quality"].payload,
    )
    for snapshot in snapshots.values():
        _require_snapshot_unchanged(snapshot)
    return (
        {name: snapshot.payload for name, snapshot in snapshots.items()},
        {name: snapshot.sha256 for name, snapshot in snapshots.items()},
    )


def _build_prepare_payloads(
    inputs: PipelineInputs,
    upstream: _AuthenticatedUpstream,
    samples: Sequence[PublishedTranslationSample],
    *,
    preflight_sha256: str,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    root = Path(inputs.run_root)
    long_records: list[dict[str, object]] = []
    cohort_records: list[dict[str, object]] = []
    source_by_scene = {
        row.scene: row.source_record.sha256 for row in upstream.scenes
    }
    for sample in samples:
        paths = _canonical_sample_paths(root, sample.scene)
        relative = {
            name: path.relative_to(root).as_posix() for name, path in paths.items()
        }
        long_records.append(
            {
                "sample_id": sample.sample_id,
                "scene": sample.scene,
                "role": sample.role,
                "path": relative["long"],
                "sha256": sample.long_sha256,
                "source_sha256": source_by_scene[sample.scene],
            }
        )
        cohort_records.append(
            {
                "sample_id": sample.sample_id,
                "scene": sample.scene,
                "role": sample.role,
                "long_path": relative["long"],
                "short_path": relative["short"],
                "quality_path": relative["quality"],
                "target_path": relative["target"],
                "long_sha256": sample.long_sha256,
                "short_sha256": sample.short_sha256,
                "quality_sha256": sample.quality_sha256,
                "target_sha256": sample.target_sha256,
            }
        )
    long_manifest: dict[str, object] = {
        "schema": LONG_CONTEXT_MANIFEST_SCHEMA,
        "run_id": root.name,
        "git_commit": inputs.git_commit,
        "records": long_records,
    }
    cohort_manifest: dict[str, object] = {
        "schema": COHORT_MANIFEST_SCHEMA,
        "run_id": root.name,
        "git_commit": inputs.git_commit,
        "records": cohort_records,
    }
    long_bytes = _json_bytes(long_manifest)
    cohort_bytes = _json_bytes(cohort_manifest)
    evidence = upstream.evidence
    config: dict[str, object] = {
        "schema": RUN_CONFIG_SCHEMA,
        "run_id": root.name,
        "git_commit": inputs.git_commit,
        "source_run": evidence["source_run"],
        "source_completion_sha256": evidence["source_completion_sha256"],
        "source_manifest_sha256": evidence["source_manifest_sha256"],
        "reference_run": evidence["reference_run"],
        "reference_completion_sha256": evidence[
            "reference_completion_sha256"
        ],
        "reference_inventory_sha256": evidence["reference_inventory_sha256"],
        "reference_long_manifest_sha256": evidence[
            "reference_long_manifest_sha256"
        ],
        "reference_teacher_manifest_sha256": evidence[
            "reference_teacher_manifest_sha256"
        ],
        "formal_run": evidence["formal_run"],
        "formal_completion_sha256": evidence["formal_completion_sha256"],
        "formal_data_manifest_sha256": evidence[
            "formal_data_manifest_sha256"
        ],
        "checkpoint_file": evidence["checkpoint_file"],
        "checkpoint_sha256": evidence["checkpoint_sha256"],
        "preflight_evidence_sha256": preflight_sha256,
        "long_context_manifest_sha256": hashlib.sha256(long_bytes).hexdigest(),
        "cohort_manifest_sha256": hashlib.sha256(cohort_bytes).hexdigest(),
        "scene_count": 10,
        "endpoint_count": 40,
        "smoke_scene": SMOKE_SCENE,
    }
    return config, long_manifest, cohort_manifest


def run_prepare(inputs: PipelineInputs) -> Path:
    """Stage via private files, then publish all final bytes create-if-absent."""
    root = Path(inputs.run_root)
    state = _run_root_state(root)
    if state == "absent":
        raise ValueError("prepare requires a completed preflight stage")
    upstream = _authenticate_upstream_full(inputs)
    expected_evidence = _preflight_payload(inputs, upstream)
    evidence_path = root / "manifests" / "preflight_evidence.json"
    evidence_bytes = _json_bytes(expected_evidence)
    evidence_snapshot = _snapshot_file(
        evidence_path,
        hashlib.sha256(evidence_bytes).hexdigest(),
        label="prepare preflight evidence",
    )
    if evidence_snapshot.payload != evidence_bytes:
        raise ValueError("prepare preflight evidence does not match authenticated inputs")
    _validate_preflight_payload(
        expected_evidence, run_id=root.name, git_commit=inputs.git_commit
    )
    completion_path = root / "prepare" / "completed.json"
    if state == "prepared":
        load_published_cohort(root)
        return completion_path
    _require_pristine_prepare_prefix(root)

    staged_payloads: dict[str, dict[str, bytes]] = {}
    staged_digests: dict[str, dict[str, str]] = {}
    with tempfile.TemporaryDirectory(
        prefix=f".{root.name}.prepare.", dir=root.parent
    ) as temporary:
        temporary_root = Path(temporary)
        input_root = temporary_root / "authenticated_inputs"
        staging_root = temporary_root / "published"
        private_paths: dict[str, dict[str, Path]] = {}
        for row in upstream.scenes:
            snapshots = upstream.snapshots[row.scene]
            paths = {
                "source": input_root / "source" / f"{row.scene}.npz",
                "teacher": input_root / "teacher" / f"{row.scene}.npz",
                "formal": input_root / "formal" / f"{row.scene}.npz",
            }
            for name, path in paths.items():
                _copy_authenticated_snapshot(path, snapshots[name])
            private_paths[row.scene] = paths

        private_checkpoint_dir = temporary_root / "checkpoint"
        private_checkpoint = _copy_streaming_snapshot(
            upstream.checkpoint_snapshot,
            private_checkpoint_dir / upstream.checkpoint_snapshot.path.name,
        )
        camera_head, checkpoint_sha256 = load_base_camera_head(
            private_checkpoint_dir
        )
        if checkpoint_sha256 != inputs.expected_checkpoint_sha256:
            raise ValueError("loaded Camera Head checkpoint digest mismatch")
        _require_snapshot_unchanged(private_checkpoint)
        _require_snapshot_unchanged(upstream.checkpoint_snapshot)
        _require_snapshot_unchanged(evidence_snapshot)
        camera_head = camera_head.to(inputs.device).eval()
        staged_samples: list[PublishedTranslationSample] = []
        for row in upstream.scenes:
            paths = private_paths[row.scene]
            published = publish_translation_sample(
                staging_root,
                role=row.role,
                source_path=paths["source"],
                source_record=row.source_record,
                teacher_reference_path=paths["teacher"],
                expected_teacher_reference_sha256=row.teacher_reference_sha256,
                formal_label_path=paths["formal"],
                expected_formal_label_sha256=row.formal_label_sha256,
                camera_head=camera_head,
                checkpoint_sha256=inputs.expected_checkpoint_sha256,
                git_commit=inputs.git_commit,
                device=inputs.device,
            )
            payloads, digests = _snapshot_staged_sample(
                published,
                staging_root=staging_root,
                expected_scene=row.scene,
                expected_role=row.role,
            )
            staged_payloads[row.scene] = payloads
            staged_digests[row.scene] = digests
            final_paths = _canonical_sample_paths(root, row.scene)
            staged_samples.append(
                PublishedTranslationSample(
                    sample_id=f"{row.scene}:frames_500",
                    scene=row.scene,
                    role=row.role,
                    long_path=final_paths["long"],
                    short_path=final_paths["short"],
                    quality_path=final_paths["quality"],
                    target_path=final_paths["target"],
                    long_sha256=digests["long"],
                    short_sha256=digests["short"],
                    quality_sha256=digests["quality"],
                    target_sha256=digests["target"],
                )
            )
        validate_calibration_cohort(staged_samples)
        for snapshot in upstream.all_snapshots:
            _require_snapshot_unchanged(snapshot)
        _require_snapshot_unchanged(private_checkpoint)
        _require_snapshot_unchanged(evidence_snapshot)

        # A race or foreign file appearing while compute ran invalidates this run ID.
        _require_pristine_prepare_prefix(root)
        expected_stage_bytes: dict[Path, bytes] = {}
        final_samples: list[PublishedTranslationSample] = []
        for row in upstream.scenes:
            paths = _canonical_sample_paths(root, row.scene)
            for name in ("long", "short", "quality", "target"):
                _publish_bytes_create_absent(
                    paths[name], staged_payloads[row.scene][name]
                )
                expected_stage_bytes[paths[name]] = staged_payloads[row.scene][name]
            digests = staged_digests[row.scene]
            final_samples.append(
                PublishedTranslationSample(
                    sample_id=f"{row.scene}:frames_500",
                    scene=row.scene,
                    role=row.role,
                    long_path=paths["long"],
                    short_path=paths["short"],
                    quality_path=paths["quality"],
                    target_path=paths["target"],
                    long_sha256=digests["long"],
                    short_sha256=digests["short"],
                    quality_sha256=digests["quality"],
                    target_sha256=digests["target"],
                )
            )

        preflight_sha256 = hashlib.sha256(evidence_bytes).hexdigest()
        config, long_manifest, cohort_manifest = _build_prepare_payloads(
            inputs,
            upstream,
            final_samples,
            preflight_sha256=preflight_sha256,
        )
        config_path = root / "config.json"
        long_manifest_path = root / "manifests" / "long_context.json"
        cohort_manifest_path = root / "manifests" / "cohort.json"
        json_publications = {
            long_manifest_path: _json_bytes(long_manifest),
            cohort_manifest_path: _json_bytes(cohort_manifest),
            config_path: _json_bytes(config),
        }
        for path in (long_manifest_path, cohort_manifest_path, config_path):
            _publish_bytes_create_absent(path, json_publications[path])
            expected_stage_bytes[path] = json_publications[path]

        relative_files = {
            path.relative_to(root).as_posix(): hashlib.sha256(content).hexdigest()
            for path, content in expected_stage_bytes.items()
        }
        unsigned_completion: dict[str, object] = {
            "schema": PREPARE_COMPLETION_SCHEMA,
            "stage": "prepare",
            "run_id": root.name,
            "git_commit": inputs.git_commit,
            "run_config_sha256": hashlib.sha256(
                json_publications[config_path]
            ).hexdigest(),
            "previous_marker_sha256": preflight_sha256,
            "files": dict(sorted(relative_files.items())),
            "metadata": {
                "scene_count": 10,
                "endpoint_count": 40,
                "smoke_scene": SMOKE_SCENE,
            },
        }
        completion = {
            **unsigned_completion,
            "completion_digest": _canonical_digest(unsigned_completion),
        }
        completion_bytes = _json_bytes(completion)
        _require_exact_bytes(expected_stage_bytes)
        _require_snapshot_unchanged(evidence_snapshot)
        dependency_snapshots = [
            _snapshot_file(
                path,
                hashlib.sha256(content).hexdigest(),
                label=f"prepare dependency {path.relative_to(root).as_posix()}",
            )
            for path, content in expected_stage_bytes.items()
        ]
        completion_published = False
        try:
            _publish_bytes_create_absent(completion_path, completion_bytes)
            completion_published = True
            for snapshot in upstream.all_snapshots:
                _require_snapshot_unchanged(snapshot)
            for snapshot in (evidence_snapshot, *dependency_snapshots):
                _require_snapshot_unchanged(snapshot)
            completion_snapshot = _snapshot_file(
                completion_path,
                hashlib.sha256(completion_bytes).hexdigest(),
                label="prepare completion",
            )
            _require_snapshot_unchanged(completion_snapshot)
        except Exception as validation_error:
            if completion_published:
                try:
                    _rollback_exact_new_file(completion_path, completion_bytes)
                except ValueError as rollback_error:
                    raise ValueError(
                        "prepare dependency validation failed and completion "
                        f"rollback failed: {rollback_error}"
                    ) from validation_error
            raise
    return completion_path


def _validate_prepare_config(
    config: Mapping[str, object],
    *,
    root: Path,
    preflight_sha256: str,
    long_manifest_sha256: str,
    cohort_manifest_sha256: str,
    preflight: Mapping[str, object],
) -> None:
    _require_fields(config, RUN_CONFIG_FIELDS, label="run config")
    if (
        config.get("schema") != RUN_CONFIG_SCHEMA
        or config.get("run_id") != root.name
        or _COMMIT_RE.fullmatch(str(config.get("git_commit", ""))) is None
        or config.get("preflight_evidence_sha256") != preflight_sha256
        or config.get("long_context_manifest_sha256")
        != long_manifest_sha256
        or config.get("cohort_manifest_sha256") != cohort_manifest_sha256
        or config.get("scene_count") != 10
        or config.get("endpoint_count") != 40
        or config.get("smoke_scene") != SMOKE_SCENE
    ):
        raise ValueError("run config identity/digest binding mismatch")
    shared = {
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
    }
    if any(config.get(name) != preflight.get(name) for name in shared):
        raise ValueError("run config does not match signed preflight provenance")
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
    ):
        _canonical_sha256(config.get(name), label=f"run config {name}")


def _canonical_downstream_file_allowlist(
    allowed_downstream_files: frozenset[str],
) -> frozenset[str]:
    if type(allowed_downstream_files) is not frozenset:
        raise ValueError("allowed_downstream_files must be a frozenset")
    canonical: set[str] = set()
    for relative in allowed_downstream_files:
        if not isinstance(relative, str) or not relative:
            raise ValueError("allowed downstream file paths must be nonempty strings")
        candidate = Path(relative)
        if (
            candidate.is_absolute()
            or candidate.drive
            or relative != candidate.as_posix()
            or any(part in {"", ".", ".."} for part in candidate.parts)
        ):
            raise ValueError("allowed downstream file path is not canonical relative")
        canonical.add(relative)
    return frozenset(canonical)


def _expected_prepared_entries(
    files: Mapping[str, object],
    *,
    allowed_downstream_files: frozenset[str] = frozenset(),
) -> set[str]:
    file_names = set(files) | {
        "manifests/preflight_evidence.json",
        "prepare/completed.json",
    } | set(allowed_downstream_files)
    directories: set[str] = set()
    for relative in file_names:
        candidate = Path(relative)
        for parent in candidate.parents:
            if str(parent) != ".":
                directories.add(parent.as_posix())
    return file_names | directories


def _validate_bundle_outer_bindings(
    bundle: Mapping[str, Mapping[str, np.ndarray]],
    *,
    scene: str,
    source_sha256: object,
    checkpoint_sha256: object,
    git_commit: str,
    formal_label_sha256: object,
    teacher_reference_sha256: object,
) -> None:
    if set(bundle) != {"long", "short", "target", "quality"}:
        raise ValueError("bundle outer binding schema mismatch")
    expected_common = {
        "sample_id": f"{scene}:frames_500",
        "scene": scene,
        "source_sha256": source_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "git_commit": git_commit,
    }
    for artifact_name, arrays in bundle.items():
        if not isinstance(arrays, Mapping):
            raise ValueError("bundle outer binding schema mismatch")
        for field, expected in expected_common.items():
            if field not in arrays or str(arrays[field]) != expected:
                raise ValueError(f"bundle {field} binding mismatch")
    quality = bundle["quality"]
    target = bundle["target"]
    if (
        "formal_label_sha256" not in quality
        or str(quality["formal_label_sha256"]) != formal_label_sha256
    ):
        raise ValueError("bundle formal_label_sha256 binding mismatch")
    for arrays in (quality, target):
        if (
            "teacher_reference_sha256" not in arrays
            or str(arrays["teacher_reference_sha256"])
            != teacher_reference_sha256
        ):
            raise ValueError("bundle teacher_reference_sha256 binding mismatch")


def load_published_cohort(
    run_root: Path,
    *,
    allowed_downstream_files: frozenset[str] = frozenset(),
) -> tuple[PublishedTranslationSample, ...]:
    """Load and reauthenticate one exact, completed, physically separated cohort."""
    allowed_downstream_files = _canonical_downstream_file_allowlist(
        allowed_downstream_files
    )
    root = Path(run_root)
    _reject_symlink_components(root)
    if not root.is_dir():
        raise ValueError("prepared run root is unavailable")
    root_directory_snapshot = _snapshot_directory(
        root, label="prepared run-root directory"
    )
    json_paths = {
        "preflight": root / "manifests" / "preflight_evidence.json",
        "config": root / "config.json",
        "long": root / "manifests" / "long_context.json",
        "cohort": root / "manifests" / "cohort.json",
        "completion": root / "prepare" / "completed.json",
    }
    snapshots: dict[str, _FileSnapshot] = {}
    payloads: dict[str, dict[str, object]] = {}
    for name, path in json_paths.items():
        snapshot, payload = _load_json_path(path, label=f"prepared {name}")
        snapshots[name] = snapshot
        payloads[name] = payload
    preflight = payloads["preflight"]
    config = payloads["config"]
    long_manifest = payloads["long"]
    cohort_manifest = payloads["cohort"]
    completion = payloads["completion"]
    git_commit = str(config.get("git_commit", ""))
    _validate_preflight_payload(
        preflight, run_id=root.name, git_commit=git_commit
    )
    _validate_prepare_config(
        config,
        root=root,
        preflight_sha256=snapshots["preflight"].sha256,
        long_manifest_sha256=snapshots["long"].sha256,
        cohort_manifest_sha256=snapshots["cohort"].sha256,
        preflight=preflight,
    )
    _require_fields(
        long_manifest,
        LONG_CONTEXT_MANIFEST_FIELDS,
        label="long-context manifest",
    )
    _require_fields(
        cohort_manifest, COHORT_MANIFEST_FIELDS, label="cohort manifest"
    )
    if (
        long_manifest.get("schema") != LONG_CONTEXT_MANIFEST_SCHEMA
        or cohort_manifest.get("schema") != COHORT_MANIFEST_SCHEMA
        or long_manifest.get("run_id") != root.name
        or cohort_manifest.get("run_id") != root.name
        or long_manifest.get("git_commit") != git_commit
        or cohort_manifest.get("git_commit") != git_commit
    ):
        raise ValueError("prepared manifest identity mismatch")
    long_rows = _rows_by_scene(
        long_manifest.get("records"),
        fields=LONG_CONTEXT_RECORD_FIELDS,
        label="long-context manifest",
    )
    cohort_rows = _rows_by_scene(
        cohort_manifest.get("records"),
        fields=COHORT_RECORD_FIELDS,
        label="cohort manifest",
    )
    _require_signed_json(
        completion,
        fields=STAGE_COMPLETION_FIELDS,
        label="prepare completion",
    )
    expected_metadata = {
        "scene_count": 10,
        "endpoint_count": 40,
        "smoke_scene": SMOKE_SCENE,
    }
    files = completion.get("files")
    if (
        completion.get("schema") != PREPARE_COMPLETION_SCHEMA
        or completion.get("stage") != "prepare"
        or completion.get("run_id") != root.name
        or completion.get("git_commit") != git_commit
        or completion.get("run_config_sha256") != snapshots["config"].sha256
        or completion.get("previous_marker_sha256")
        != snapshots["preflight"].sha256
        or completion.get("metadata") != expected_metadata
        or not isinstance(files, dict)
    ):
        raise ValueError("prepare completion stage binding mismatch")

    expected_file_names = {
        "config.json",
        "manifests/long_context.json",
        "manifests/cohort.json",
    }
    for scene in FROZEN_SCENES:
        expected_file_names.update(
            path.relative_to(root).as_posix()
            for path in _canonical_sample_paths(root, scene).values()
        )
    if set(files) != expected_file_names:
        raise ValueError("prepare completion file inventory is not exact")
    prepared_file_names = set(files) | {
        "manifests/preflight_evidence.json",
        "prepare/completed.json",
    }
    if prepared_file_names & set(allowed_downstream_files):
        raise ValueError("allowed downstream files must not name prepare artifacts")
    allowed_downstream_snapshots = []
    for relative in sorted(allowed_downstream_files):
        allowed_downstream_snapshots.append(
            _snapshot_streaming_file(
                root / relative,
                None,
                label=f"allowed downstream file {relative}",
            )
        )
    expected_entries = _expected_prepared_entries(
        files, allowed_downstream_files=allowed_downstream_files
    )
    if _root_entries(root) != expected_entries:
        raise ValueError("prepared run contains foreign files or directories")
    expected_file_names_with_markers = prepared_file_names | set(
        allowed_downstream_files
    )
    directory_snapshots = [root_directory_snapshot]
    for relative in sorted(expected_entries - expected_file_names_with_markers):
        directory_snapshots.append(
            _snapshot_directory(
                root / relative,
                label=f"prepared directory {relative}",
            )
        )
    fixed_json_digests = {
        "config.json": snapshots["config"].sha256,
        "manifests/long_context.json": snapshots["long"].sha256,
        "manifests/cohort.json": snapshots["cohort"].sha256,
    }
    for relative, digest in fixed_json_digests.items():
        if files.get(relative) != digest:
            raise ValueError("prepare completion JSON inventory mismatch")

    samples: list[PublishedTranslationSample] = []
    data_snapshots: list[_FileSnapshot] = []
    for scene in FROZEN_SCENES:
        long_row = long_rows[scene]
        row = cohort_rows[scene]
        preflight_row = preflight["scene_bindings"][FROZEN_SCENES.index(scene)]
        role = calibration_role(scene)
        expected_paths = _canonical_sample_paths(root, scene)
        expected_relative = {
            name: path.relative_to(root).as_posix()
            for name, path in expected_paths.items()
        }
        if (
            row["sample_id"] != f"{scene}:frames_500"
            or long_row["sample_id"] != row["sample_id"]
            or long_row["role"] != role
            or long_row["path"] != expected_relative["long"]
            or long_row["sha256"] != row["long_sha256"]
            or row["long_path"] != expected_relative["long"]
            or row["short_path"] != expected_relative["short"]
            or row["quality_path"] != expected_relative["quality"]
            or row["target_path"] != expected_relative["target"]
            or long_row["source_sha256"]
            != preflight_row["source_sha256"]
        ):
            raise ValueError("prepared cohort physical path or identity mismatch")
        digests = {
            "long": _canonical_sha256(row["long_sha256"], label="cohort long digest"),
            "short": _canonical_sha256(row["short_sha256"], label="cohort short digest"),
            "quality": _canonical_sha256(
                row["quality_sha256"], label="cohort quality digest"
            ),
            "target": _canonical_sha256(
                row["target_sha256"], label="cohort target digest"
            ),
        }
        scene_snapshots = {
            name: _snapshot_file(
                expected_paths[name],
                digests[name],
                label=f"prepared {name} artifact {scene}",
            )
            for name in expected_paths
        }
        for name, snapshot in scene_snapshots.items():
            relative = expected_relative[name]
            if files.get(relative) != snapshot.sha256:
                raise ValueError("prepare completion data inventory mismatch")
        bundle = load_bound_bundle_bytes(
            scene_snapshots["long"].payload,
            scene_snapshots["short"].payload,
            scene_snapshots["target"].payload,
            scene_snapshots["quality"].payload,
        )
        _validate_bundle_outer_bindings(
            bundle,
            scene=scene,
            source_sha256=preflight_row["source_sha256"],
            checkpoint_sha256=config["checkpoint_sha256"],
            git_commit=git_commit,
            formal_label_sha256=preflight_row["formal_label_sha256"],
            teacher_reference_sha256=preflight_row[
                "teacher_reference_sha256"
            ],
        )
        data_snapshots.extend(scene_snapshots.values())
        samples.append(
            PublishedTranslationSample(
                sample_id=f"{scene}:frames_500",
                scene=scene,
                role=role,
                long_path=expected_paths["long"],
                short_path=expected_paths["short"],
                quality_path=expected_paths["quality"],
                target_path=expected_paths["target"],
                long_sha256=digests["long"],
                short_sha256=digests["short"],
                quality_sha256=digests["quality"],
                target_sha256=digests["target"],
            )
        )
    validate_calibration_cohort(samples)
    for snapshot in (*snapshots.values(), *data_snapshots):
        _require_snapshot_unchanged(snapshot)
    for snapshot in allowed_downstream_snapshots:
        _require_snapshot_unchanged(snapshot)
    if _root_entries(root) != expected_entries:
        raise ValueError("prepared run contains late foreign files or directories")
    for snapshot in directory_snapshots:
        _require_directory_unchanged(snapshot)
    return tuple(samples)


__all__ = [
    "AuthenticatedSceneInputs",
    "COHORT_MANIFEST_FIELDS",
    "COHORT_MANIFEST_SCHEMA",
    "COHORT_RECORD_FIELDS",
    "FROZEN_REFERENCE_COMPLETION_SHA256",
    "LONG_CONTEXT_MANIFEST_FIELDS",
    "LONG_CONTEXT_MANIFEST_SCHEMA",
    "LONG_CONTEXT_RECORD_FIELDS",
    "PREFLIGHT_EVIDENCE_FIELDS",
    "PREFLIGHT_SCHEMA",
    "PREPARE_COMPLETION_SCHEMA",
    "PipelineInputs",
    "RUN_CONFIG_FIELDS",
    "RUN_CONFIG_SCHEMA",
    "STAGE_COMPLETION_FIELDS",
    "authenticate_upstream",
    "load_published_cohort",
    "run_preflight",
    "run_prepare",
]
