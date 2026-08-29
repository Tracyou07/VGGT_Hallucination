"""Fail-closed Stage A-prime evaluation and publication orchestration."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import tempfile

import torch

from pre_experiments.camera_translation_hvrfm.artifacts import (
    LONG_CONTEXT_MEMBERS,
    load_long_context,
)
from pre_experiments.camera_translation_hvrfm.data import (
    PublishedTranslationSample,
)
from pre_experiments.camera_translation_hvrfm.evaluate import (
    classify_stage_a_prime,
    evaluate_translation_sample,
)
from pre_experiments.camera_translation_hvrfm.pipeline import (
    PipelineInputs,
    load_published_cohort,
    run_preflight,
    run_prepare,
)
from pre_experiments.camera_translation_hvrfm.report import (
    build_stage_a_prime_report,
    write_stage_a_prime_report,
)
from pre_experiments.camera_translation_hvrfm.verify import (
    verify_completed_run,
)


SMOKE_COMPLETION_SCHEMA = "camera_translation_hvrfm.smoke_completion.v1"
CALIBRATION_COMPLETION_SCHEMA = (
    "camera_translation_hvrfm.calibration_completion.v1"
)
PREPARE_COMPLETION_SCHEMA = "camera_translation_hvrfm.prepare_completion.v1"
REPORT_SCHEMA = "camera_translation_hvrfm.stage_a_prime_report.v1"
REPORT_COMPLETION_SCHEMA = (
    "camera_translation_hvrfm.stage_a_prime_completion.v1"
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

_READY = "TRANSLATION_ENDPOINTS_READY"
_FAILED = "TRANSLATION_ENDPOINTS_FAILED"
_CLASSIFICATIONS = frozenset({_READY, _FAILED})
_SMOKE_SCENE = "scene0029_01"
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
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")

_SMOKE_PATH = "smoke/completed.json"
_CALIBRATION_PATH = "calibration/completed.json"
_REPORT_JSON_PATH = "reports/stage_a_prime.json"
_REPORT_MARKDOWN_PATH = "reports/stage_a_prime.md"
_REPORT_COMPLETION_PATH = "reports/completed.json"
_VERIFICATION_INVENTORY_PATH = "manifests/verification_inventory.json"
_VERIFIED_COMPLETION_PATH = "verified_completion.json"

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


class StageAPrimeFailed(RuntimeError):
    """The preserved calibration/report evidence classified Stage A-prime failed."""


@dataclass(frozen=True)
class _FileSnapshot:
    path: Path
    sha256: str
    identity: tuple[int, int, int, int, int, int]
    label: str


@dataclass(frozen=True)
class _DirectorySnapshot:
    path: Path
    identity: tuple[int, int, int]
    label: str


def _artifact_paths(scene: str) -> dict[str, str]:
    return {
        "long": f"prediction_only/long_context/{scene}.npz",
        "short": f"privileged_training/short_context/{scene}.npz",
        "quality": f"privileged_labels/quality/{scene}.npz",
        "target": f"privileged_labels/translation_targets/{scene}.npz",
    }


def _prepared_files() -> frozenset[str]:
    files = {
        "config.json",
        "manifests/preflight_evidence.json",
        "manifests/long_context.json",
        "manifests/cohort.json",
        "prepare/completed.json",
    }
    for scene in _SCENES:
        files.update(_artifact_paths(scene).values())
    return frozenset(files)


_PREPARED_FILES = _prepared_files()
_SMOKE_FILES = _PREPARED_FILES | {_SMOKE_PATH}
_CALIBRATION_FILES = _SMOKE_FILES | {_CALIBRATION_PATH}
_REPORT_FILES = _CALIBRATION_FILES | {
    _REPORT_JSON_PATH,
    _REPORT_MARKDOWN_PATH,
    _REPORT_COMPLETION_PATH,
}
_VERIFIED_FILES = _REPORT_FILES | {
    _VERIFICATION_INVENTORY_PATH,
    _VERIFIED_COMPLETION_PATH,
}
_STATE_FILES = {
    "prepared": _PREPARED_FILES,
    "smoke": _SMOKE_FILES,
    "calibration": _CALIBRATION_FILES,
    "report": _REPORT_FILES,
    "verified": _VERIFIED_FILES,
}
_STATE_ORDER = {
    "prepared": 0,
    "smoke": 1,
    "calibration": 2,
    "report": 3,
    "verified": 4,
}


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


def _json_bytes(payload: object) -> bytes:
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
        raise ValueError("payload is not canonical JSON") from error


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON object contains duplicate key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise ValueError(f"JSON contains non-finite constant: {value}")


def _decode_json_bytes(
    payload: bytes,
    *,
    label: str,
    fields: frozenset[str] | None = None,
) -> dict[str, object]:
    if type(payload) is not bytes:
        raise ValueError(f"{label} snapshot must be immutable bytes")
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{label} must be strict JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    if fields is not None and set(value) != set(fields):
        raise ValueError(f"{label} must use the exact schema")
    if _json_bytes(value) != payload:
        raise ValueError(f"{label} bytes are not canonical")
    return value


def _absolute_without_resolving(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _reject_symlink_components(path: Path) -> None:
    candidate = Path(path)
    if ".." in candidate.parts:
        raise ValueError("stage paths may not contain lexical parent traversal")
    absolute = _absolute_without_resolving(candidate)
    for component in (absolute, *absolute.parents):
        if component.is_symlink():
            raise ValueError(
                f"stage paths may not contain symlink components: {component}"
            )


def _root_from_inputs(inputs: PipelineInputs) -> Path:
    try:
        root = Path(inputs.run_root)
        git_commit = inputs.git_commit
    except (AttributeError, TypeError) as error:
        raise ValueError("pipeline inputs are malformed") from error
    _reject_symlink_components(root)
    absolute = _absolute_without_resolving(root)
    if not absolute.is_dir():
        raise ValueError("prepared run root must be an existing directory")
    if _RUN_ID_RE.fullmatch(absolute.name) is None:
        raise ValueError("run-root basename is not a canonical run ID")
    if not isinstance(git_commit, str) or _COMMIT_RE.fullmatch(git_commit) is None:
        raise ValueError("git_commit must be a canonical Git commit")
    return absolute


def _safe_child(root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if (
        not isinstance(relative, str)
        or not relative
        or candidate.is_absolute()
        or candidate.drive
        or relative != candidate.as_posix()
        or any(part in {"", ".", ".."} for part in candidate.parts)
    ):
        raise ValueError("stage artifact path must be canonical relative")
    target = root / candidate
    _reject_symlink_components(target)
    try:
        common = os.path.commonpath((os.fspath(root), os.fspath(target)))
    except ValueError as error:
        raise ValueError("stage artifact path escapes the run root") from error
    if os.path.normcase(common) != os.path.normcase(os.fspath(root)):
        raise ValueError("stage artifact path escapes the run root")
    return target


def _file_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _directory_identity(value: os.stat_result) -> tuple[int, int, int]:
    return (int(value.st_dev), int(value.st_ino), int(value.st_mode))


def _snapshot_file(path: Path, *, label: str) -> _FileSnapshot:
    target = Path(path)
    _reject_symlink_components(target)
    digest = hashlib.sha256()
    try:
        before = target.stat(follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} must be a regular file")
        with target.open("rb") as handle:
            opened_before = os.fstat(handle.fileno())
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
            opened_after = os.fstat(handle.fileno())
        after = target.stat(follow_symlinks=False)
    except ValueError:
        raise
    except OSError as error:
        raise ValueError(f"could not snapshot {label}") from error
    identities = {
        _file_identity(before),
        _file_identity(opened_before),
        _file_identity(opened_after),
        _file_identity(after),
    }
    if len(identities) != 1:
        raise ValueError(f"{label} changed while taking its snapshot")
    return _FileSnapshot(
        path=target,
        sha256=digest.hexdigest(),
        identity=identities.pop(),
        label=label,
    )


def _read_snapshot_bytes(snapshot: _FileSnapshot) -> bytes:
    _require_file_snapshot(snapshot)
    try:
        payload = snapshot.path.read_bytes()
    except OSError as error:
        raise ValueError(f"could not read {snapshot.label}") from error
    _require_file_snapshot(snapshot)
    if hashlib.sha256(payload).hexdigest() != snapshot.sha256:
        raise ValueError(f"{snapshot.label} changed during authentication")
    return payload


def _require_file_snapshot(snapshot: _FileSnapshot) -> None:
    try:
        current = snapshot.path.stat(follow_symlinks=False)
    except OSError as error:
        raise ValueError(f"{snapshot.label} changed during authentication") from error
    if _file_identity(current) != snapshot.identity:
        raise ValueError(f"{snapshot.label} changed during authentication")
    current_snapshot = _snapshot_file(snapshot.path, label=snapshot.label)
    if (
        current_snapshot.identity != snapshot.identity
        or current_snapshot.sha256 != snapshot.sha256
    ):
        raise ValueError(f"{snapshot.label} changed during authentication")


def _snapshot_directory(path: Path, *, label: str) -> _DirectorySnapshot:
    target = Path(path)
    _reject_symlink_components(target)
    try:
        current = target.stat(follow_symlinks=False)
    except OSError as error:
        raise ValueError(f"could not snapshot {label}") from error
    if not stat.S_ISDIR(current.st_mode):
        raise ValueError(f"{label} must be a directory")
    return _DirectorySnapshot(
        path=target,
        identity=_directory_identity(current),
        label=label,
    )


def _require_directory_snapshot(snapshot: _DirectorySnapshot) -> None:
    try:
        current = snapshot.path.stat(follow_symlinks=False)
    except OSError as error:
        raise ValueError(f"{snapshot.label} changed during authentication") from error
    if (
        not stat.S_ISDIR(current.st_mode)
        or _directory_identity(current) != snapshot.identity
    ):
        raise ValueError(f"{snapshot.label} changed during authentication")


def _inventory(root: Path) -> tuple[frozenset[str], frozenset[str]]:
    _reject_symlink_components(root)
    if not root.is_dir():
        raise ValueError("run root must remain a directory")
    files: set[str] = set()
    directories: set[str] = set()
    try:
        entries = list(root.rglob("*"))
    except OSError as error:
        raise ValueError("could not inventory the run root") from error
    for entry in entries:
        if entry.is_symlink():
            raise ValueError("run root may not contain symlinks")
        relative = entry.relative_to(root).as_posix()
        try:
            entry_stat = entry.stat(follow_symlinks=False)
        except OSError as error:
            raise ValueError("run entry changed during inventory") from error
        if stat.S_ISDIR(entry_stat.st_mode):
            directories.add(relative)
        elif stat.S_ISREG(entry_stat.st_mode):
            files.add(relative)
        else:
            raise ValueError("run root may contain only directories and regular files")
    return frozenset(files), frozenset(directories)


def _expected_directories(files: frozenset[str]) -> frozenset[str]:
    result: set[str] = set()
    for relative in files:
        parent = Path(relative).parent
        while parent != Path("."):
            result.add(parent.as_posix())
            parent = parent.parent
    return frozenset(result)


def _require_inventory(root: Path, files: frozenset[str], *, label: str) -> None:
    actual_files, actual_directories = _inventory(root)
    expected_directories = _expected_directories(files)
    if actual_files != files or actual_directories != expected_directories:
        missing_files = sorted(files - actual_files)
        extra_files = sorted(actual_files - files)
        missing_directories = sorted(expected_directories - actual_directories)
        extra_directories = sorted(actual_directories - expected_directories)
        raise ValueError(
            f"noncanonical {label} inventory; missing_files={missing_files}, "
            f"extra_files={extra_files}, missing_directories={missing_directories}, "
            f"extra_directories={extra_directories}"
        )


def _state(root: Path) -> str:
    files, directories = _inventory(root)
    for name, expected in _STATE_FILES.items():
        if files == expected and directories == _expected_directories(expected):
            return name
    raise ValueError(
        "run root contains an unknown, partial, or foreign stage artifact; preserve it"
    )


def _snapshot_state(
    root: Path, files: frozenset[str], *, label: str
) -> tuple[tuple[_FileSnapshot, ...], tuple[_DirectorySnapshot, ...]]:
    _require_inventory(root, files, label=label)
    file_snapshots = tuple(
        _snapshot_file(_safe_child(root, relative), label=f"{label} file {relative}")
        for relative in sorted(files)
    )
    directory_snapshots = (
        _snapshot_directory(root, label=f"{label} run root"),
        *(
            _snapshot_directory(
                _safe_child(root, relative), label=f"{label} directory {relative}"
            )
            for relative in sorted(_expected_directories(files))
        ),
    )
    _require_inventory(root, files, label=label)
    return file_snapshots, directory_snapshots


def _require_state_unchanged(
    root: Path,
    files: frozenset[str],
    file_snapshots: Sequence[_FileSnapshot],
    directory_snapshots: Sequence[_DirectorySnapshot],
    *,
    label: str,
) -> None:
    _require_inventory(root, files, label=label)
    for snapshot in file_snapshots:
        try:
            _require_file_snapshot(snapshot)
        except ValueError as error:
            raise ValueError(f"dependency changed during {label}: {snapshot.label}") from error
    for snapshot in directory_snapshots:
        try:
            _require_directory_snapshot(snapshot)
        except ValueError as error:
            raise ValueError(f"directory changed during {label}: {snapshot.label}") from error


def _load_cohort(
    root: Path, state: str
) -> tuple[PublishedTranslationSample, ...]:
    allowed = frozenset(_STATE_FILES[state] - _PREPARED_FILES)
    try:
        loaded = load_published_cohort(
            root, allowed_downstream_files=allowed
        )
    except TypeError as error:
        if allowed:
            raise ValueError(
                "pipeline.load_published_cohort lacks downstream authentication support"
            ) from error
        try:
            loaded = load_published_cohort(root)
        except TypeError:
            raise error
    if not isinstance(loaded, Sequence) or isinstance(loaded, (str, bytes)):
        raise ValueError("published cohort must be a sequence")
    cohort = tuple(loaded)
    scenes = [getattr(sample, "scene", None) for sample in cohort]
    if len(cohort) != 10 or len(set(scenes)) != 10 or set(scenes) != set(_SCENES):
        raise ValueError("stages require the exact ten-scene calibration cohort")
    by_scene = {sample.scene: sample for sample in cohort}
    for scene in _SCENES:
        sample = by_scene[scene]
        if (
            not isinstance(sample, PublishedTranslationSample)
            or sample.sample_id != f"{scene}:frames_500"
            or sample.role
            != ("validation" if scene in {"scene0325_01", "scene0675_00"} else "train")
        ):
            raise ValueError("stages require exact cohort identities and roles")
    return tuple(by_scene[scene] for scene in _SCENES)


def _read_json_path(
    path: Path, *, label: str, fields: frozenset[str] | None = None
) -> tuple[_FileSnapshot, dict[str, object]]:
    snapshot = _snapshot_file(path, label=label)
    return snapshot, _decode_json_bytes(
        _read_snapshot_bytes(snapshot), label=label, fields=fields
    )


def _validate_completion_digest(payload: Mapping[str, object], *, label: str) -> None:
    unsigned = dict(payload)
    recorded = unsigned.pop("completion_digest", None)
    if (
        not isinstance(recorded, str)
        or _SHA256_RE.fullmatch(recorded) is None
        or recorded != _canonical_digest(unsigned)
    ):
        raise ValueError(f"{label} completion digest mismatch")


def _actual_prepare_files(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in sorted(
        _PREPARED_FILES
        - {"manifests/preflight_evidence.json", "prepare/completed.json"}
    ):
        result[relative] = _snapshot_file(
            _safe_child(root, relative), label=f"prepare inventory {relative}"
        ).sha256
    return result


def _validate_prepare_marker(root: Path, *, git_commit: str) -> _FileSnapshot:
    marker_snapshot, marker = _read_json_path(
        _safe_child(root, "prepare/completed.json"),
        label="prepare marker",
        fields=STAGE_COMPLETION_FIELDS,
    )
    _validate_completion_digest(marker, label="prepare marker")
    config_sha256 = _snapshot_file(
        _safe_child(root, "config.json"), label="run config"
    ).sha256
    preflight_sha256 = _snapshot_file(
        _safe_child(root, "manifests/preflight_evidence.json"),
        label="preflight evidence",
    ).sha256
    if (
        marker.get("schema") != PREPARE_COMPLETION_SCHEMA
        or marker.get("stage") != "prepare"
        or marker.get("run_id") != root.name
        or marker.get("git_commit") != git_commit
        or marker.get("run_config_sha256") != config_sha256
        or marker.get("previous_marker_sha256") != preflight_sha256
        or marker.get("metadata")
        != {
            "scene_count": 10,
            "endpoint_count": 40,
            "smoke_scene": _SMOKE_SCENE,
        }
        or marker.get("files") != _actual_prepare_files(root)
    ):
        raise ValueError("prepare marker identity, metadata, or inventory mismatch")
    return marker_snapshot


def _stage_metadata(stage: str, classification: str = _READY) -> dict[str, object]:
    if stage == "smoke":
        if classification != _READY:
            raise ValueError("smoke marker may only record the ready classification")
        return {
            "scene": _SMOKE_SCENE,
            "endpoint_count": 4,
            "classification": _READY,
        }
    if stage == "calibration" and classification in _CLASSIFICATIONS:
        return {
            "scene_count": 10,
            "endpoint_count": 40,
            "classification": classification,
        }
    raise ValueError("unsupported stage marker metadata")


def _validate_stage_marker(
    root: Path,
    *,
    stage: str,
    git_commit: str,
    previous: _FileSnapshot,
) -> tuple[_FileSnapshot, str]:
    if stage == "smoke":
        relative = _SMOKE_PATH
        schema = SMOKE_COMPLETION_SCHEMA
    elif stage == "calibration":
        relative = _CALIBRATION_PATH
        schema = CALIBRATION_COMPLETION_SCHEMA
    else:
        raise ValueError("unsupported stage marker")
    snapshot, marker = _read_json_path(
        _safe_child(root, relative),
        label=f"{stage} marker",
        fields=STAGE_COMPLETION_FIELDS,
    )
    _validate_completion_digest(marker, label=f"{stage} marker")
    metadata = marker.get("metadata")
    classification = (
        metadata.get("classification") if isinstance(metadata, dict) else None
    )
    allowed = {_READY} if stage == "smoke" else _CLASSIFICATIONS
    if classification not in allowed:
        raise ValueError(f"{stage} marker classification mismatch")
    config_sha256 = _snapshot_file(
        _safe_child(root, "config.json"), label="run config"
    ).sha256
    if (
        marker.get("schema") != schema
        or marker.get("stage") != stage
        or marker.get("run_id") != root.name
        or marker.get("git_commit") != git_commit
        or marker.get("run_config_sha256") != config_sha256
        or marker.get("previous_marker_sha256") != previous.sha256
        or marker.get("files") != {}
        or metadata != _stage_metadata(stage, str(classification))
    ):
        raise ValueError(f"{stage} marker identity, metadata, or predecessor mismatch")
    return snapshot, str(classification)


def _authenticate_prefix(
    root: Path, *, git_commit: str, state: str
) -> tuple[tuple[PublishedTranslationSample, ...], _FileSnapshot, _FileSnapshot | None, str | None]:
    _require_inventory(root, _STATE_FILES[state], label=f"{state} stage")
    cohort = _load_cohort(root, state)
    prepare = _validate_prepare_marker(root, git_commit=git_commit)
    smoke: _FileSnapshot | None = None
    calibration_classification: str | None = None
    if _STATE_ORDER[state] >= _STATE_ORDER["smoke"]:
        smoke, _ = _validate_stage_marker(
            root,
            stage="smoke",
            git_commit=git_commit,
            previous=prepare,
        )
    if _STATE_ORDER[state] >= _STATE_ORDER["calibration"]:
        if smoke is None:
            raise ValueError("calibration cannot exist without smoke")
        _, calibration_classification = _validate_stage_marker(
            root,
            stage="calibration",
            git_commit=git_commit,
            previous=smoke,
        )
    _require_inventory(root, _STATE_FILES[state], label=f"{state} stage")
    return cohort, prepare, smoke, calibration_classification


def _snapshot_authenticated_state(
    root: Path,
    *,
    git_commit: str,
    state: str,
    label: str,
) -> tuple[
    tuple[PublishedTranslationSample, ...],
    _FileSnapshot,
    _FileSnapshot | None,
    str | None,
    tuple[_FileSnapshot, ...],
    tuple[_DirectorySnapshot, ...],
]:
    """Snapshot exact bytes first, then bind semantic authentication to them."""
    state_files = _STATE_FILES[state]
    file_snapshots, directory_snapshots = _snapshot_state(
        root, state_files, label=label
    )
    cohort, prepare, smoke, calibration_classification = _authenticate_prefix(
        root, git_commit=git_commit, state=state
    )
    _require_state_unchanged(
        root,
        state_files,
        file_snapshots,
        directory_snapshots,
        label=label,
    )
    return (
        cohort,
        prepare,
        smoke,
        calibration_classification,
        file_snapshots,
        directory_snapshots,
    )


def _all_numbers_finite(value: object) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, Mapping):
        return all(_all_numbers_finite(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return all(_all_numbers_finite(item) for item in value)
    return True


def _exact_bool(value: object) -> bool:
    return type(value) is bool and value


def _finite_metric(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"smoke {name} must be a finite scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"smoke {name} must be a finite scalar")
    return result


def _smoke_gate_failures(metrics: Mapping[str, object]) -> list[str]:
    if not isinstance(metrics, Mapping):
        raise ValueError("smoke evaluation must return a metric mapping")
    endpoints_value = metrics.get("endpoints")
    if not isinstance(endpoints_value, Sequence) or isinstance(
        endpoints_value, (str, bytes)
    ):
        raise ValueError("smoke endpoint metrics must be a sequence")
    endpoints = list(endpoints_value)
    if (
        metrics.get("scene") != _SMOKE_SCENE
        or metrics.get("endpoint_count") != 4
        or metrics.get("endpoint_ids") != [0, 1, 2, 3]
        or len(endpoints) != 4
    ):
        raise ValueError("smoke evaluation does not bind exactly four frozen endpoints")
    failures: list[str] = []
    if not _all_numbers_finite(metrics) or not _exact_bool(metrics.get("all_finite")):
        failures.append("finite")
    for name in (
        "uncovered_positive_zero",
        "quaternion_bytes_equal",
        "fov_bytes_equal",
    ):
        if not _exact_bool(metrics.get(name)):
            failures.append(name)
    if _finite_metric(
        metrics.get("max_covered_roundtrip_fraction"),
        name="covered round-trip",
    ) >= 1e-5:
        failures.append("covered_roundtrip")
    if _finite_metric(
        metrics.get("max_uncovered_drift_fraction"), name="uncovered drift"
    ) >= 1e-8:
        failures.append("uncovered_anchor")
    if _finite_metric(
        metrics.get("max_rotation_delta_deg"), name="rotation delta"
    ) > 1e-6:
        failures.append("rotation_guard")
    for index, endpoint in enumerate(endpoints):
        if not isinstance(endpoint, Mapping) or endpoint.get("endpoint_id") != index:
            raise ValueError("smoke endpoint ordering is not canonical")
        if not _exact_bool(endpoint.get("all_finite")):
            failures.append(f"endpoint_{index}_finite")
        for name in (
            "uncovered_positive_zero",
            "quaternion_bytes_equal",
            "fov_bytes_equal",
        ):
            if not _exact_bool(endpoint.get(name)):
                failures.append(f"endpoint_{index}_{name}")
        if _finite_metric(
            endpoint.get("covered_roundtrip_fraction"),
            name=f"endpoint {index} covered round-trip",
        ) >= 1e-5:
            failures.append(f"endpoint_{index}_covered_roundtrip")
        if _finite_metric(
            endpoint.get("uncovered_drift_fraction"),
            name=f"endpoint {index} uncovered drift",
        ) >= 1e-8:
            failures.append(f"endpoint_{index}_uncovered_anchor")
        if _finite_metric(
            endpoint.get("rotation_delta_deg"),
            name=f"endpoint {index} rotation delta",
        ) > 1e-6:
            failures.append(f"endpoint_{index}_rotation_guard")
    return failures


def _audit_prediction_only_tree(
    root: Path, cohort: Sequence[PublishedTranslationSample]
) -> bool:
    """Derive the production leakage verdict from exact paths and NPZ schemas."""
    prediction_root = _safe_child(root, "prediction_only")
    if not prediction_root.is_dir():
        return False
    expected_files = frozenset(f"long_context/{scene}.npz" for scene in _SCENES)
    actual_files: set[str] = set()
    actual_directories: set[str] = set()
    try:
        entries = list(prediction_root.rglob("*"))
    except OSError:
        return False
    for entry in entries:
        if entry.is_symlink():
            return False
        relative = entry.relative_to(prediction_root).as_posix()
        try:
            current = entry.stat(follow_symlinks=False)
        except OSError:
            return False
        if stat.S_ISDIR(current.st_mode):
            actual_directories.add(relative)
        elif stat.S_ISREG(current.st_mode):
            actual_files.add(relative)
        else:
            return False
    if actual_files != set(expected_files) or actual_directories != {"long_context"}:
        return False
    by_scene = {sample.scene: sample for sample in cohort}
    if set(by_scene) != set(_SCENES):
        return False
    for scene in _SCENES:
        sample = by_scene[scene]
        expected = _safe_child(
            root, f"prediction_only/long_context/{scene}.npz"
        )
        if _absolute_without_resolving(sample.long_path) != expected:
            return False
        for privileged in (
            sample.short_path,
            sample.quality_path,
            sample.target_path,
        ):
            try:
                common = os.path.commonpath(
                    (os.fspath(prediction_root), os.fspath(_absolute_without_resolving(privileged)))
                )
            except ValueError:
                return False
            if os.path.normcase(common) == os.path.normcase(os.fspath(prediction_root)):
                return False
        try:
            before = _snapshot_file(expected, label=f"prediction-only {scene}")
            arrays = load_long_context(expected)
            after = _snapshot_file(expected, label=f"prediction-only {scene}")
        except (OSError, ValueError):
            return False
        if (
            before.identity != after.identity
            or before.sha256 != after.sha256
            or before.sha256 != sample.long_sha256
            or set(arrays) != set(LONG_CONTEXT_MEMBERS)
        ):
            return False
    return True


def _live_calibration_decision(
    root: Path, cohort: Sequence[PublishedTranslationSample]
) -> str:
    metrics = [evaluate_translation_sample(sample) for sample in cohort]
    physical_leakage_clean = _audit_prediction_only_tree(root, cohort)
    classification = classify_stage_a_prime(
        metrics,
        cohort=cohort,
        physical_leakage_clean=physical_leakage_clean,
    )
    if not isinstance(classification, Mapping):
        raise ValueError("calibration classifier must return a mapping")
    decision = classification.get("classification")
    if (
        decision not in _CLASSIFICATIONS
        or classification.get("scene_count") != 10
        or classification.get("endpoint_count") != 40
    ):
        raise ValueError("calibration classifier returned a noncanonical decision")
    return str(decision)


def _stage_marker_bytes(
    root: Path,
    *,
    stage: str,
    schema: str,
    git_commit: str,
    previous_sha256: str,
    classification: str,
) -> bytes:
    config_sha256 = _snapshot_file(
        _safe_child(root, "config.json"), label="run config"
    ).sha256
    unsigned: dict[str, object] = {
        "schema": schema,
        "stage": stage,
        "run_id": root.name,
        "git_commit": git_commit,
        "run_config_sha256": config_sha256,
        "previous_marker_sha256": previous_sha256,
        "files": {},
        "metadata": _stage_metadata(stage, classification),
    }
    return _json_bytes(
        {**unsigned, "completion_digest": _canonical_digest(unsigned)}
    )


def _atomic_create_exact(path: Path, content: bytes) -> _FileSnapshot | None:
    """Create one exact file without overwrite; return its owned snapshot."""
    if type(content) is not bytes:
        raise ValueError("publication content must be immutable bytes")
    target = Path(path)
    _reject_symlink_components(target)
    if target.exists():
        if target.is_file() and not target.is_symlink() and target.read_bytes() == content:
            return None
        raise ValueError(f"publication target already exists with foreign bytes: {target}")
    if not target.parent.is_dir():
        raise ValueError("publication parent must already be a directory")
    temporary: Path | None = None
    linked = False
    linked_snapshot: _FileSnapshot | None = None
    publication_error: Exception | None = None
    cleanup_error: OSError | None = None
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
        linked = True
        source_snapshot = _snapshot_file(
            temporary, label="publication temporary file"
        )
        if source_snapshot.sha256 != hashlib.sha256(content).hexdigest():
            raise ValueError("publication temporary bytes changed before linking")
        linked_snapshot = _FileSnapshot(
            path=target,
            sha256=source_snapshot.sha256,
            identity=source_snapshot.identity,
            label="new completion target",
        )
        _require_file_snapshot(linked_snapshot)
    except Exception as error:
        publication_error = error
    if temporary is not None:
        try:
            temporary.unlink(missing_ok=True)
        except OSError as error:
            cleanup_error = error
    if cleanup_error is not None and linked:
        if linked_snapshot is None:
            raise ValueError(
                "publication succeeded but temporary cleanup failed and exact "
                f"completion rollback was unavailable: {target}"
            ) from cleanup_error
        try:
            _rollback_exact_snapshot(linked_snapshot, content)
            try:
                target.stat(follow_symlinks=False)
            except FileNotFoundError:
                pass
            except OSError as error:
                raise ValueError(
                    "could not verify completion rollback terminal state"
                ) from error
            else:
                raise ValueError("completion target remains after rollback")
        except ValueError as rollback_error:
            raise ValueError(
                "publication succeeded but temporary cleanup failed and completion "
                f"rollback failed: {rollback_error}"
            ) from cleanup_error
        raise ValueError(
            f"publication succeeded but temporary cleanup failed; completion was rolled back: {target}"
        ) from cleanup_error
    if publication_error is not None:
        try:
            exact_race = (
                not linked
                and cleanup_error is None
                and target.is_file()
                and not target.is_symlink()
                and target.read_bytes() == content
            )
        except OSError:
            exact_race = False
        if exact_race:
            return None
        detail = " and temporary cleanup failed" if cleanup_error else ""
        raise ValueError(
            f"could not publish create-if-absent target{detail}: {target}"
        ) from publication_error
    if cleanup_error is not None:
        raise ValueError(
            f"publication succeeded but temporary cleanup failed: {target}"
        ) from cleanup_error
    if not linked:
        raise ValueError("publication did not create or authenticate a target")
    if linked_snapshot is None:
        raise ValueError("publication did not capture an owned target snapshot")
    try:
        final_snapshot = _snapshot_file(target, label="new completion target")
    except ValueError as error:
        raise ValueError("publication target changed before terminal snapshot") from error
    if (
        final_snapshot.identity[:5] != linked_snapshot.identity[:5]
        or final_snapshot.sha256 != linked_snapshot.sha256
    ):
        raise ValueError("publication target changed before terminal snapshot")
    return final_snapshot


def _rollback_exact_new_file(path: Path, content: bytes) -> None:
    target = Path(path)
    _reject_symlink_components(target)
    if not target.is_file() or target.is_symlink():
        raise ValueError("new completion target cannot be rolled back")
    try:
        current = target.read_bytes()
    except OSError as error:
        raise ValueError("could not authenticate new completion target") from error
    if current != content:
        raise ValueError("new completion target changed before rollback")
    try:
        target.unlink()
    except OSError as error:
        raise ValueError("could not roll back new completion target") from error


def _rollback_exact_snapshot(snapshot: _FileSnapshot, content: bytes) -> None:
    """Remove only the unchanged file identity captured for this publication."""
    if (
        type(content) is not bytes
        or snapshot.sha256 != hashlib.sha256(content).hexdigest()
    ):
        raise ValueError("captured completion bytes do not match their snapshot")
    try:
        _require_file_snapshot(snapshot)
        current = _read_snapshot_bytes(snapshot)
        _require_file_snapshot(snapshot)
    except ValueError as error:
        raise ValueError("new completion target changed before rollback") from error
    if current != content:
        raise ValueError("new completion target changed before rollback")
    try:
        snapshot.path.unlink()
    except OSError as error:
        raise ValueError("could not roll back new completion target") from error


def _publish_stage_marker(
    root: Path,
    *,
    current_files: frozenset[str],
    target_relative: str,
    content: bytes,
    file_snapshots: Sequence[_FileSnapshot],
    directory_snapshots: Sequence[_DirectorySnapshot],
    label: str,
) -> Path:
    target = _safe_child(root, target_relative)
    expected_files = current_files | {target_relative}
    _require_state_unchanged(
        root,
        current_files,
        file_snapshots,
        directory_snapshots,
        label=label,
    )
    try:
        target.parent.mkdir(parents=False, exist_ok=False)
    except (FileExistsError, OSError) as error:
        raise ValueError(f"could not create pristine {label} directory") from error
    _reject_symlink_components(target.parent)
    publication_directories = (
        *directory_snapshots,
        _snapshot_directory(target.parent, label=f"{label} publication directory"),
    )
    for snapshot in file_snapshots:
        _require_file_snapshot(snapshot)
    for snapshot in publication_directories:
        _require_directory_snapshot(snapshot)
    created_snapshot: _FileSnapshot | None = None
    try:
        created_snapshot = _atomic_create_exact(target, content)
        if created_snapshot is not None:
            _require_file_snapshot(created_snapshot)
        for snapshot in file_snapshots:
            _require_file_snapshot(snapshot)
        for snapshot in publication_directories:
            _require_directory_snapshot(snapshot)
        _require_inventory(root, expected_files, label=f"{label} completion")
        marker = _snapshot_file(target, label=f"{label} completion")
        if (
            marker.sha256 != hashlib.sha256(content).hexdigest()
            or (
                created_snapshot is not None
                and marker.identity != created_snapshot.identity
            )
        ):
            raise ValueError(f"{label} completion bytes changed after publication")
        for snapshot in file_snapshots:
            _require_file_snapshot(snapshot)
        for snapshot in publication_directories:
            _require_directory_snapshot(snapshot)
        _require_inventory(root, expected_files, label=f"{label} completion")
    except Exception as validation_error:
        if created_snapshot is not None:
            try:
                _rollback_exact_snapshot(created_snapshot, content)
            except ValueError as rollback_error:
                raise ValueError(
                    f"{label} publication validation failed and completion rollback failed: "
                    f"{rollback_error}"
                ) from validation_error
        raise
    return target


def _validate_report(
    root: Path,
    *,
    git_commit: str,
    calibration_classification: str,
    cohort: Sequence[PublishedTranslationSample],
    expected_report: Mapping[str, object] | None = None,
) -> tuple[Path, Path, Path]:
    json_path = _safe_child(root, _REPORT_JSON_PATH)
    markdown_path = _safe_child(root, _REPORT_MARKDOWN_PATH)
    completion_path = _safe_child(root, _REPORT_COMPLETION_PATH)
    json_snapshot, report = _read_json_path(
        json_path, label="Stage A-prime report", fields=_REPORT_FIELDS
    )
    markdown_snapshot = _snapshot_file(
        markdown_path, label="Stage A-prime report Markdown"
    )
    _, completion = _read_json_path(
        completion_path,
        label="Stage A-prime report completion",
        fields=_REPORT_COMPLETION_FIELDS,
    )
    _validate_completion_digest(
        completion, label="Stage A-prime report completion"
    )
    classification = report.get("classification")
    if (
        classification not in _CLASSIFICATIONS
        or classification != calibration_classification
        or report.get("schema") != REPORT_SCHEMA
        or report.get("run_id") != root.name
        or report.get("git_commit") != git_commit
        or report.get("scene_count") != 10
        or report.get("endpoint_count") != 40
        or type(report.get("physical_leakage_clean")) is not bool
        or not isinstance(report.get("scene_metrics"), list)
        or len(report["scene_metrics"]) != 10
        or not isinstance(report.get("cohort"), list)
        or len(report["cohort"]) != 10
    ):
        raise ValueError("Stage A-prime report identity or classification mismatch")
    if (
        completion.get("schema") != REPORT_COMPLETION_SCHEMA
        or completion.get("run_id") != root.name
        or completion.get("git_commit") != git_commit
        or completion.get("classification") != classification
        or completion.get("scene_count") != 10
        or completion.get("endpoint_count") != 40
        or completion.get("report_json_path") != _REPORT_JSON_PATH
        or completion.get("report_json_sha256") != json_snapshot.sha256
        or completion.get("report_markdown_path") != _REPORT_MARKDOWN_PATH
        or completion.get("report_markdown_sha256") != markdown_snapshot.sha256
    ):
        raise ValueError("Stage A-prime report completion mismatch")
    if expected_report is None:
        live_metrics = [evaluate_translation_sample(sample) for sample in cohort]
        physical_leakage_clean = _audit_prediction_only_tree(root, cohort)
        expected = build_stage_a_prime_report(
            live_metrics,
            cohort=cohort,
            run_id=root.name,
            git_commit=git_commit,
            physical_leakage_clean=physical_leakage_clean,
        )
    else:
        expected = dict(expected_report)
    if (
        not isinstance(expected, Mapping)
        or set(expected) != set(_REPORT_FIELDS)
        or dict(expected) != report
    ):
        raise ValueError(
            "Stage A-prime report does not match deterministic reclassification"
        )
    return json_path, markdown_path, completion_path


def _terminal_report_publication_barrier(
    root: Path,
    *,
    payload: Mapping[str, object],
    git_commit: str,
    calibration_classification: str,
) -> tuple[Path, Path, Path]:
    """Bind the just-published report triplet to terminal immutable snapshots."""
    json_path = _safe_child(root, _REPORT_JSON_PATH)
    markdown_path = _safe_child(root, _REPORT_MARKDOWN_PATH)
    completion_path = _safe_child(root, _REPORT_COMPLETION_PATH)
    json_snapshot, report = _read_json_path(
        json_path, label="terminal Stage A-prime report", fields=_REPORT_FIELDS
    )
    markdown_snapshot = _snapshot_file(
        markdown_path, label="terminal Stage A-prime report Markdown"
    )
    completion_snapshot, completion = _read_json_path(
        completion_path,
        label="terminal Stage A-prime report completion",
        fields=_REPORT_COMPLETION_FIELDS,
    )
    _validate_completion_digest(
        completion, label="terminal Stage A-prime report completion"
    )
    if dict(report) != dict(payload):
        raise ValueError("terminal report JSON changed after semantic validation")
    if (
        completion.get("schema") != REPORT_COMPLETION_SCHEMA
        or completion.get("run_id") != root.name
        or completion.get("git_commit") != git_commit
        or completion.get("classification") != calibration_classification
        or completion.get("scene_count") != 10
        or completion.get("endpoint_count") != 40
        or completion.get("report_json_path") != _REPORT_JSON_PATH
        or completion.get("report_json_sha256") != json_snapshot.sha256
        or completion.get("report_markdown_path") != _REPORT_MARKDOWN_PATH
        or completion.get("report_markdown_sha256") != markdown_snapshot.sha256
    ):
        raise ValueError("terminal report completion binding mismatch")
    for snapshot in (json_snapshot, markdown_snapshot, completion_snapshot):
        try:
            _require_file_snapshot(snapshot)
        except ValueError as error:
            raise ValueError(
                f"report publication changed after terminal snapshot: {snapshot.label}"
            ) from error
    _require_inventory(root, _REPORT_FILES, label="terminal report completion")
    for snapshot in (json_snapshot, markdown_snapshot, completion_snapshot):
        try:
            _require_file_snapshot(snapshot)
        except ValueError as error:
            raise ValueError(
                f"report publication changed at terminal barrier: {snapshot.label}"
            ) from error
    return json_path, markdown_path, completion_path


def _snapshot_expected_report_completion(
    root: Path,
    payload: Mapping[str, object],
    *,
    git_commit: str,
    calibration_classification: str,
) -> tuple[_FileSnapshot, bytes] | None:
    """Capture only the completion bytes expected from this report publication."""
    try:
        if (
            payload.get("schema") != REPORT_SCHEMA
            or payload.get("run_id") != root.name
            or payload.get("git_commit") != git_commit
            or payload.get("classification") != calibration_classification
            or payload.get("scene_count") != 10
            or payload.get("endpoint_count") != 40
        ):
            return None
        markdown_snapshot = _snapshot_file(
            _safe_child(root, _REPORT_MARKDOWN_PATH),
            label="new Stage A-prime report Markdown",
        )
        json_content = _json_bytes(dict(payload))
        unsigned: dict[str, object] = {
            "schema": REPORT_COMPLETION_SCHEMA,
            "run_id": root.name,
            "git_commit": git_commit,
            "classification": calibration_classification,
            "scene_count": 10,
            "endpoint_count": 40,
            "report_json_path": _REPORT_JSON_PATH,
            "report_json_sha256": hashlib.sha256(json_content).hexdigest(),
            "report_markdown_path": _REPORT_MARKDOWN_PATH,
            "report_markdown_sha256": markdown_snapshot.sha256,
        }
        expected = _json_bytes(
            {**unsigned, "completion_digest": _canonical_digest(unsigned)}
        )
        completion_snapshot = _snapshot_file(
            _safe_child(root, _REPORT_COMPLETION_PATH),
            label="new Stage A-prime report completion",
        )
        if (
            completion_snapshot.sha256 != hashlib.sha256(expected).hexdigest()
            or _read_snapshot_bytes(completion_snapshot) != expected
        ):
            return None
        _require_file_snapshot(markdown_snapshot)
    except (KeyError, TypeError, ValueError):
        return None
    return completion_snapshot, expected


def _publish_report(
    root: Path,
    payload: Mapping[str, object],
    *,
    git_commit: str,
    calibration_classification: str,
    cohort: Sequence[PublishedTranslationSample],
    file_snapshots: Sequence[_FileSnapshot],
    directory_snapshots: Sequence[_DirectorySnapshot],
) -> tuple[Path, Path, Path]:
    report_directory = _safe_child(root, "reports")
    try:
        report_directory.mkdir(parents=False, exist_ok=False)
    except (FileExistsError, OSError) as error:
        raise ValueError("could not create pristine report directory") from error
    publication_directories = (
        *directory_snapshots,
        _snapshot_directory(report_directory, label="report publication directory"),
    )
    for snapshot in file_snapshots:
        _require_file_snapshot(snapshot)
    for snapshot in publication_directories:
        _require_directory_snapshot(snapshot)
    completion_path = _safe_child(root, _REPORT_COMPLETION_PATH)
    expected_completion: tuple[_FileSnapshot, bytes] | None = None
    try:
        try:
            paths = write_stage_a_prime_report(root, payload)
        finally:
            expected_completion = _snapshot_expected_report_completion(
                root,
                payload,
                git_commit=git_commit,
                calibration_classification=calibration_classification,
            )
        if (
            not isinstance(paths, tuple)
            or paths
            != (
                _safe_child(root, _REPORT_JSON_PATH),
                _safe_child(root, _REPORT_MARKDOWN_PATH),
                completion_path,
            )
        ):
            raise ValueError("report writer returned noncanonical publication paths")
        for snapshot in file_snapshots:
            _require_file_snapshot(snapshot)
        for snapshot in publication_directories:
            _require_directory_snapshot(snapshot)
        _require_inventory(root, _REPORT_FILES, label="report completion")
        _validate_report(
            root,
            git_commit=git_commit,
            calibration_classification=calibration_classification,
            cohort=cohort,
            expected_report=payload,
        )
        for snapshot in file_snapshots:
            _require_file_snapshot(snapshot)
        for snapshot in publication_directories:
            _require_directory_snapshot(snapshot)
        _require_inventory(root, _REPORT_FILES, label="report completion")
        terminal_paths = _terminal_report_publication_barrier(
            root,
            payload=payload,
            git_commit=git_commit,
            calibration_classification=calibration_classification,
        )
        if terminal_paths != paths:
            raise ValueError("terminal report publication paths are noncanonical")
        return terminal_paths
    except Exception as validation_error:
        if expected_completion is not None:
            completion_snapshot, completion_bytes = expected_completion
            try:
                _rollback_exact_snapshot(completion_snapshot, completion_bytes)
            except ValueError as rollback_error:
                raise ValueError(
                    "report dependency validation failed and completion rollback failed: "
                    f"{rollback_error}"
                ) from validation_error
        raise


def run_smoke(inputs: PipelineInputs) -> Path:
    """Evaluate only scene0029_01 and publish its structural-gate marker."""
    root = _root_from_inputs(inputs)
    state = _state(root)
    (
        cohort,
        prepare,
        _,
        _,
        file_snapshots,
        directory_snapshots,
    ) = _snapshot_authenticated_state(
        root,
        git_commit=inputs.git_commit,
        state=state,
        label="smoke",
    )
    if _STATE_ORDER[state] >= _STATE_ORDER["smoke"]:
        marker, _ = _validate_stage_marker(
            root,
            stage="smoke",
            git_commit=inputs.git_commit,
            previous=prepare,
        )
        smoke_sample = next(sample for sample in cohort if sample.scene == _SMOKE_SCENE)
        failures = _smoke_gate_failures(evaluate_translation_sample(smoke_sample))
        if failures:
            raise ValueError(
                "completed smoke live replay structural gates failed: "
                + ", ".join(sorted(set(failures)))
            )
        _require_state_unchanged(
            root,
            _STATE_FILES[state],
            file_snapshots,
            directory_snapshots,
            label="smoke",
        )
        return marker.path
    if state != "prepared":
        raise ValueError("smoke requires the exact completed prepare stage")
    smoke_sample = next(sample for sample in cohort if sample.scene == _SMOKE_SCENE)
    metrics = evaluate_translation_sample(smoke_sample)
    failures = _smoke_gate_failures(metrics)
    if failures:
        raise ValueError(
            "smoke structural gates failed: " + ", ".join(sorted(set(failures)))
        )
    _require_state_unchanged(
        root,
        _PREPARED_FILES,
        file_snapshots,
        directory_snapshots,
        label="smoke",
    )
    content = _stage_marker_bytes(
        root,
        stage="smoke",
        schema=SMOKE_COMPLETION_SCHEMA,
        git_commit=inputs.git_commit,
        previous_sha256=prepare.sha256,
        classification=_READY,
    )
    return _publish_stage_marker(
        root,
        current_files=_PREPARED_FILES,
        target_relative=_SMOKE_PATH,
        content=content,
        file_snapshots=file_snapshots,
        directory_snapshots=directory_snapshots,
        label="smoke",
    )


def run_calibration(inputs: PipelineInputs) -> Path:
    """Evaluate the exact ten-scene/four-endpoint cohort after signed smoke."""
    root = _root_from_inputs(inputs)
    state = _state(root)
    (
        cohort,
        _,
        smoke,
        existing_classification,
        file_snapshots,
        directory_snapshots,
    ) = _snapshot_authenticated_state(
        root,
        git_commit=inputs.git_commit,
        state=state,
        label="calibration",
    )
    if _STATE_ORDER[state] >= _STATE_ORDER["calibration"]:
        if existing_classification not in _CLASSIFICATIONS:
            raise ValueError("completed calibration classification is unavailable")
        live_classification = _live_calibration_decision(root, cohort)
        if live_classification != existing_classification:
            raise ValueError(
                "completed calibration marker disagrees with live replay classification"
            )
        _require_state_unchanged(
            root,
            _STATE_FILES[state],
            file_snapshots,
            directory_snapshots,
            label="calibration",
        )
        return _safe_child(root, _CALIBRATION_PATH)
    if state != "smoke" or smoke is None:
        raise ValueError("calibration requires a valid completed smoke stage")
    decision = _live_calibration_decision(root, cohort)
    _require_state_unchanged(
        root,
        _SMOKE_FILES,
        file_snapshots,
        directory_snapshots,
        label="calibration",
    )
    content = _stage_marker_bytes(
        root,
        stage="calibration",
        schema=CALIBRATION_COMPLETION_SCHEMA,
        git_commit=inputs.git_commit,
        previous_sha256=smoke.sha256,
        classification=decision,
    )
    return _publish_stage_marker(
        root,
        current_files=_SMOKE_FILES,
        target_relative=_CALIBRATION_PATH,
        content=content,
        file_snapshots=file_snapshots,
        directory_snapshots=directory_snapshots,
        label="calibration",
    )


def run_report(inputs: PipelineInputs) -> tuple[Path, Path, Path]:
    """Recompute and deterministically publish the ready or failed report."""
    root = _root_from_inputs(inputs)
    state = _state(root)
    (
        cohort,
        _,
        _,
        calibration_classification,
        file_snapshots,
        directory_snapshots,
    ) = _snapshot_authenticated_state(
        root,
        git_commit=inputs.git_commit,
        state=state,
        label="report",
    )
    if calibration_classification not in _CLASSIFICATIONS:
        raise ValueError("report requires a valid completed calibration stage")
    if _STATE_ORDER[state] >= _STATE_ORDER["report"]:
        paths = _validate_report(
            root,
            git_commit=inputs.git_commit,
            calibration_classification=calibration_classification,
            cohort=cohort,
        )
        _require_state_unchanged(
            root,
            _STATE_FILES[state],
            file_snapshots,
            directory_snapshots,
            label="report",
        )
        return paths
    if state != "calibration":
        raise ValueError("report requires the exact completed calibration stage")
    metrics = [evaluate_translation_sample(sample) for sample in cohort]
    physical_leakage_clean = _audit_prediction_only_tree(root, cohort)
    payload = build_stage_a_prime_report(
        metrics,
        cohort=cohort,
        run_id=root.name,
        git_commit=inputs.git_commit,
        physical_leakage_clean=physical_leakage_clean,
    )
    if (
        not isinstance(payload, Mapping)
        or payload.get("classification") != calibration_classification
    ):
        raise ValueError("report recomputation disagrees with calibration classification")
    _require_state_unchanged(
        root,
        _CALIBRATION_FILES,
        file_snapshots,
        directory_snapshots,
        label="report",
    )
    paths = _publish_report(
        root,
        payload,
        git_commit=inputs.git_commit,
        calibration_classification=calibration_classification,
        cohort=cohort,
        file_snapshots=file_snapshots,
        directory_snapshots=directory_snapshots,
    )
    _require_inventory(root, _REPORT_FILES, label="report completion")
    return paths


def _classification_from_returned_marker(path: Path) -> str:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        metadata = payload["metadata"]
        classification = metadata["classification"]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise ValueError("calibration stage returned an unreadable completion marker") from error
    if classification not in _CLASSIFICATIONS:
        raise ValueError("calibration stage returned an invalid classification")
    return str(classification)


def run_all(inputs: PipelineInputs) -> Path:
    """Run preflight -> prepare -> smoke -> calibration -> report -> verify."""
    try:
        root = _absolute_without_resolving(Path(inputs.run_root))
    except (AttributeError, TypeError) as error:
        raise ValueError("pipeline inputs are malformed") from error
    prepare_marker = root / "prepare" / "completed.json"
    if not prepare_marker.is_file():
        run_preflight(inputs)
        run_prepare(inputs)
    run_smoke(inputs)
    calibration_marker = run_calibration(inputs)
    run_report(inputs)
    classification = _classification_from_returned_marker(calibration_marker)
    if classification != _READY:
        raise StageAPrimeFailed(
            "Stage A-prime calibration failed; the signed failed marker and report were preserved"
        )
    try:
        checkpoint_dir = Path(inputs.checkpoint_dir)
    except (AttributeError, TypeError) as error:
        raise ValueError("pipeline inputs lack a valid checkpoint_dir") from error
    return verify_completed_run(
        root,
        expected_run_id=root.name,
        expected_git_commit=inputs.git_commit,
        checkpoint_dir=checkpoint_dir,
    )


def _commit_argument(value: str) -> str:
    if _COMMIT_RE.fullmatch(value) is None:
        raise argparse.ArgumentTypeError(
            "Git commit must be 40 lowercase hexadecimal characters"
        )
    return value


def _sha256_argument(value: str) -> str:
    if _SHA256_RE.fullmatch(value) is None:
        raise argparse.ArgumentTypeError(
            "SHA-256 values must be 64 lowercase hexadecimal characters"
        )
    return value


def _device_argument(value: str) -> torch.device:
    try:
        return torch.device(value)
    except (RuntimeError, ValueError) as error:
        raise argparse.ArgumentTypeError("device is not a valid torch device") from error


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m pre_experiments.camera_translation_hvrfm.stages",
        description="Run one fail-closed camera-translation Stage A-prime stage.",
    )
    parser.add_argument(
        "stage",
        choices=(
            "preflight",
            "prepare",
            "smoke",
            "calibration",
            "report",
            "verify",
            "all",
        ),
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--git-commit", type=_commit_argument, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--formal-run", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument(
        "--expected-source-completion-sha256",
        type=_sha256_argument,
        required=True,
    )
    parser.add_argument(
        "--expected-reference-completion-sha256",
        type=_sha256_argument,
        required=True,
    )
    parser.add_argument(
        "--expected-formal-completion-sha256",
        type=_sha256_argument,
        required=True,
    )
    parser.add_argument(
        "--expected-checkpoint-sha256",
        type=_sha256_argument,
        required=True,
    )
    parser.add_argument("--device", type=_device_argument, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Strict command-line adapter for the H20 runner."""
    arguments = _argument_parser().parse_args(argv)
    inputs = PipelineInputs(
        run_root=arguments.run_root,
        git_commit=arguments.git_commit,
        source_run=arguments.source_run,
        reference_run=arguments.reference_run,
        formal_run=arguments.formal_run,
        checkpoint_dir=arguments.checkpoint_dir,
        expected_source_completion_sha256=(
            arguments.expected_source_completion_sha256
        ),
        expected_reference_completion_sha256=(
            arguments.expected_reference_completion_sha256
        ),
        expected_formal_completion_sha256=(
            arguments.expected_formal_completion_sha256
        ),
        expected_checkpoint_sha256=arguments.expected_checkpoint_sha256,
        device=arguments.device,
    )
    stage = arguments.stage
    if stage == "preflight":
        result: Path | tuple[Path, Path, Path] = run_preflight(inputs)
    elif stage == "prepare":
        result = run_prepare(inputs)
    elif stage == "smoke":
        result = run_smoke(inputs)
    elif stage == "calibration":
        result = run_calibration(inputs)
    elif stage == "report":
        result = run_report(inputs)
    elif stage == "verify":
        result = verify_completed_run(
            inputs.run_root,
            expected_run_id=inputs.run_root.name,
            expected_git_commit=inputs.git_commit,
            checkpoint_dir=inputs.checkpoint_dir,
        )
    elif stage == "all":
        result = run_all(inputs)
    else:  # pragma: no cover - argparse choices make this unreachable.
        raise ValueError("unsupported stage")
    outputs = result if isinstance(result, tuple) else (result,)
    for output in outputs:
        print(output)
    return 0


__all__ = [
    "CALIBRATION_COMPLETION_SCHEMA",
    "SMOKE_COMPLETION_SCHEMA",
    "STAGE_COMPLETION_FIELDS",
    "StageAPrimeFailed",
    "main",
    "run_all",
    "run_calibration",
    "run_report",
    "run_smoke",
]


if __name__ == "__main__":
    raise SystemExit(main())
