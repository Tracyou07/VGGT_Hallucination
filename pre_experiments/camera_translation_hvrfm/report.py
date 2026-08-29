"""Deterministic, digest-bound Stage A-prime report publication."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from numbers import Integral, Real
import os
from pathlib import Path
import re
import tempfile

from pre_experiments.camera_translation_hvrfm.data import PublishedTranslationSample
from pre_experiments.camera_translation_hvrfm.evaluate import classify_stage_a_prime


REPORT_SCHEMA = "camera_translation_hvrfm.stage_a_prime_report.v1"
COMPLETION_SCHEMA = "camera_translation_hvrfm.stage_a_prime_completion.v1"
REPORT_JSON_PATH = "reports/stage_a_prime.json"
REPORT_MARKDOWN_PATH = "reports/stage_a_prime.md"
COMPLETION_PATH = "reports/completed.json"

_RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_ENDPOINT_KEYS = frozenset(
    {
        "endpoint_id",
        "covered_utility",
        "teacher_covered_utility",
        "full_scene_utility",
        "covered_roundtrip_fraction",
        "uncovered_drift_fraction",
        "rotation_delta_deg",
        "quaternion_bytes_equal",
        "fov_bytes_equal",
        "uncovered_positive_zero",
        "endpoint_rms",
        "coverage_fraction",
        "all_finite",
    }
)
_PROVENANCE_KEYS = frozenset(
    {
        "long_sha256",
        "short_sha256",
        "quality_sha256",
        "target_sha256",
        "source_sha256",
        "checkpoint_sha256",
        "teacher_reference_sha256",
        "git_commit",
    }
)
_SCENE_KEYS = frozenset(
    {
        "scene",
        "sample_id",
        "role",
        "endpoint_count",
        "endpoint_ids",
        "endpoints",
        "mean_covered_utility",
        "mean_teacher_covered_utility",
        "teacher_retention",
        "mean_full_scene_utility",
        "max_covered_roundtrip_fraction",
        "max_uncovered_drift_fraction",
        "max_rotation_delta_deg",
        "quaternion_bytes_equal",
        "fov_bytes_equal",
        "uncovered_positive_zero",
        "all_finite",
        "provenance",
    }
)
_COHORT_KEYS = frozenset(
    {
        "scene",
        "sample_id",
        "role",
        "long_sha256",
        "short_sha256",
        "quality_sha256",
        "target_sha256",
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
_CLASSIFIER_KEYS = frozenset(
    {
        "classification",
        "failed_gates",
        "gates",
        "scene_count",
        "endpoint_count",
        "mean_teacher_retention",
        "mean_full_scene_utility",
        "minimum_full_scene_utility",
        "positive_scene_count",
    }
)
_REPORT_KEYS = frozenset(
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


def _exact_mapping(
    value: object, members: frozenset[str], *, name: str
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != members:
        raise ValueError(f"{name} must use the exact schema")
    return value


def _text(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    return value


def _run_id(value: object) -> str:
    result = _text(value, name="run_id")
    if _RUN_ID_RE.fullmatch(result) is None:
        raise ValueError("run_id must be a canonical path-safe identifier")
    return result


def _sha256(value: object, *, name: str) -> str:
    result = _text(value, name=name)
    if _SHA256_RE.fullmatch(result) is None:
        raise ValueError(f"{name} must be a canonical lowercase SHA-256 digest")
    return result


def _git_commit(value: object) -> str:
    result = _text(value, name="git_commit")
    if _COMMIT_RE.fullmatch(result) is None:
        raise ValueError("git_commit must be a canonical lowercase 40-character hash")
    return result


def _boolean(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be an exact Boolean")
    return value


def _integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer, not a Boolean")
    return int(value)


def _number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number, not a Boolean")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _sequence(value: object, *, name: str) -> list[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a sequence")
    return list(value)


def _normalize_endpoint(value: object) -> dict[str, object]:
    row = _exact_mapping(value, _ENDPOINT_KEYS, name="endpoint metrics")
    return {
        "endpoint_id": _integer(row["endpoint_id"], name="endpoint_id"),
        "covered_utility": _number(
            row["covered_utility"], name="covered_utility"
        ),
        "teacher_covered_utility": _number(
            row["teacher_covered_utility"], name="teacher_covered_utility"
        ),
        "full_scene_utility": _number(
            row["full_scene_utility"], name="full_scene_utility"
        ),
        "covered_roundtrip_fraction": _number(
            row["covered_roundtrip_fraction"],
            name="covered_roundtrip_fraction",
        ),
        "uncovered_drift_fraction": _number(
            row["uncovered_drift_fraction"], name="uncovered_drift_fraction"
        ),
        "rotation_delta_deg": _number(
            row["rotation_delta_deg"], name="rotation_delta_deg"
        ),
        "quaternion_bytes_equal": _boolean(
            row["quaternion_bytes_equal"], name="quaternion_bytes_equal"
        ),
        "fov_bytes_equal": _boolean(
            row["fov_bytes_equal"], name="fov_bytes_equal"
        ),
        "uncovered_positive_zero": _boolean(
            row["uncovered_positive_zero"], name="uncovered_positive_zero"
        ),
        "endpoint_rms": _number(row["endpoint_rms"], name="endpoint_rms"),
        "coverage_fraction": _number(
            row["coverage_fraction"], name="coverage_fraction"
        ),
        "all_finite": _boolean(row["all_finite"], name="all_finite"),
    }


def _normalize_provenance(value: object) -> dict[str, str]:
    row = _exact_mapping(value, _PROVENANCE_KEYS, name="scene provenance")
    result = {
        name: _sha256(row[name], name=f"scene provenance {name}")
        for name in (
            "long_sha256",
            "short_sha256",
            "quality_sha256",
            "target_sha256",
            "source_sha256",
            "checkpoint_sha256",
            "teacher_reference_sha256",
        )
    }
    result["git_commit"] = _git_commit(row["git_commit"])
    return result


def _normalize_scene(value: object) -> dict[str, object]:
    row = _exact_mapping(value, _SCENE_KEYS, name="scene metrics")
    endpoints = [
        _normalize_endpoint(endpoint)
        for endpoint in _sequence(row["endpoints"], name="scene endpoints")
    ]
    endpoints.sort(key=lambda endpoint: int(endpoint["endpoint_id"]))
    endpoint_ids = [
        _integer(endpoint_id, name="endpoint_ids")
        for endpoint_id in _sequence(row["endpoint_ids"], name="endpoint_ids")
    ]
    return {
        "scene": _text(row["scene"], name="scene"),
        "sample_id": _text(row["sample_id"], name="sample_id"),
        "role": _text(row["role"], name="role"),
        "endpoint_count": _integer(
            row["endpoint_count"], name="endpoint_count"
        ),
        "endpoint_ids": endpoint_ids,
        "endpoints": endpoints,
        "mean_covered_utility": _number(
            row["mean_covered_utility"], name="mean_covered_utility"
        ),
        "mean_teacher_covered_utility": _number(
            row["mean_teacher_covered_utility"],
            name="mean_teacher_covered_utility",
        ),
        "teacher_retention": _number(
            row["teacher_retention"], name="teacher_retention"
        ),
        "mean_full_scene_utility": _number(
            row["mean_full_scene_utility"], name="mean_full_scene_utility"
        ),
        "max_covered_roundtrip_fraction": _number(
            row["max_covered_roundtrip_fraction"],
            name="max_covered_roundtrip_fraction",
        ),
        "max_uncovered_drift_fraction": _number(
            row["max_uncovered_drift_fraction"],
            name="max_uncovered_drift_fraction",
        ),
        "max_rotation_delta_deg": _number(
            row["max_rotation_delta_deg"], name="max_rotation_delta_deg"
        ),
        "quaternion_bytes_equal": _boolean(
            row["quaternion_bytes_equal"], name="quaternion_bytes_equal"
        ),
        "fov_bytes_equal": _boolean(
            row["fov_bytes_equal"], name="fov_bytes_equal"
        ),
        "uncovered_positive_zero": _boolean(
            row["uncovered_positive_zero"], name="uncovered_positive_zero"
        ),
        "all_finite": _boolean(row["all_finite"], name="all_finite"),
        "provenance": _normalize_provenance(row["provenance"]),
    }


def _normalize_scene_metrics(value: object) -> list[dict[str, object]]:
    rows = [
        _normalize_scene(row)
        for row in _sequence(value, name="scene_metrics")
    ]
    rows.sort(key=lambda row: str(row["scene"]))
    return rows


def _sorted_cohort_samples(value: object) -> list[PublishedTranslationSample]:
    samples = _sequence(value, name="cohort")
    if any(not isinstance(sample, PublishedTranslationSample) for sample in samples):
        raise ValueError("cohort must contain PublishedTranslationSample values")
    return sorted(samples, key=lambda sample: sample.scene)


def _cohort_record(sample: PublishedTranslationSample) -> dict[str, str]:
    return {
        "scene": _text(sample.scene, name="cohort scene"),
        "sample_id": _text(sample.sample_id, name="cohort sample_id"),
        "role": _text(sample.role, name="cohort role"),
        "long_sha256": _sha256(
            sample.long_sha256, name="cohort long_sha256"
        ),
        "short_sha256": _sha256(
            sample.short_sha256, name="cohort short_sha256"
        ),
        "quality_sha256": _sha256(
            sample.quality_sha256, name="cohort quality_sha256"
        ),
        "target_sha256": _sha256(
            sample.target_sha256, name="cohort target_sha256"
        ),
    }


def _normalize_cohort_records(value: object) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for value_row in _sequence(value, name="report cohort"):
        row = _exact_mapping(value_row, _COHORT_KEYS, name="cohort record")
        records.append(
            {
                "scene": _text(row["scene"], name="cohort scene"),
                "sample_id": _text(row["sample_id"], name="cohort sample_id"),
                "role": _text(row["role"], name="cohort role"),
                "long_sha256": _sha256(
                    row["long_sha256"], name="cohort long_sha256"
                ),
                "short_sha256": _sha256(
                    row["short_sha256"], name="cohort short_sha256"
                ),
                "quality_sha256": _sha256(
                    row["quality_sha256"], name="cohort quality_sha256"
                ),
                "target_sha256": _sha256(
                    row["target_sha256"], name="cohort target_sha256"
                ),
            }
        )
    records.sort(key=lambda row: row["scene"])
    return records


def _samples_from_records(
    records: Sequence[Mapping[str, str]],
) -> list[PublishedTranslationSample]:
    return [
        PublishedTranslationSample(
            sample_id=record["sample_id"],
            scene=record["scene"],
            role=record["role"],
            long_path=Path(),
            short_path=Path(),
            quality_path=Path(),
            target_path=Path(),
            long_sha256=record["long_sha256"],
            short_sha256=record["short_sha256"],
            quality_sha256=record["quality_sha256"],
            target_sha256=record["target_sha256"],
        )
        for record in records
    ]


def _normalize_classifier_result(value: object) -> dict[str, object]:
    row = _exact_mapping(value, _CLASSIFIER_KEYS, name="classifier result")
    gates_value = _exact_mapping(
        row["gates"], frozenset(_GATE_NAMES), name="classifier gates"
    )
    gates = {
        name: _boolean(gates_value[name], name=f"classifier gate {name}")
        for name in _GATE_NAMES
    }
    failed = [
        _text(name, name="failed gate")
        for name in _sequence(row["failed_gates"], name="failed_gates")
    ]
    expected_failed = [name for name in _GATE_NAMES if not gates[name]]
    if failed != expected_failed:
        raise ValueError("failed_gates must exactly match the failed classifier gates")
    classification = _text(row["classification"], name="classification")
    expected_classification = (
        "TRANSLATION_ENDPOINTS_READY"
        if not failed
        else "TRANSLATION_ENDPOINTS_FAILED"
    )
    if classification != expected_classification:
        raise ValueError("classification does not match failed_gates")
    return {
        "classification": classification,
        "failed_gates": failed,
        "gates": gates,
        "scene_count": _integer(row["scene_count"], name="scene_count"),
        "endpoint_count": _integer(
            row["endpoint_count"], name="endpoint_count"
        ),
        "mean_teacher_retention": _number(
            row["mean_teacher_retention"], name="mean_teacher_retention"
        ),
        "mean_full_scene_utility": _number(
            row["mean_full_scene_utility"], name="mean_full_scene_utility"
        ),
        "minimum_full_scene_utility": _number(
            row["minimum_full_scene_utility"],
            name="minimum_full_scene_utility",
        ),
        "positive_scene_count": _integer(
            row["positive_scene_count"], name="positive_scene_count"
        ),
    }


def _compose_report(
    scene_metrics: list[dict[str, object]],
    *,
    cohort: list[PublishedTranslationSample],
    run_id: str,
    git_commit: str,
    physical_leakage_clean: bool,
) -> dict[str, object]:
    if any(
        row["provenance"]["git_commit"] != git_commit
        for row in scene_metrics
    ):
        raise ValueError("report git_commit must match every scene provenance")
    classifier = _normalize_classifier_result(
        classify_stage_a_prime(
            scene_metrics,
            cohort=cohort,
            physical_leakage_clean=physical_leakage_clean,
        )
    )
    cohort_records = [_cohort_record(sample) for sample in cohort]
    return {
        "schema": REPORT_SCHEMA,
        "run_id": run_id,
        "git_commit": git_commit,
        **classifier,
        "physical_leakage_clean": physical_leakage_clean,
        "scene_metrics": scene_metrics,
        "cohort": cohort_records,
    }


def build_stage_a_prime_report(
    scene_metrics: Sequence[Mapping[str, object]],
    *,
    cohort: Sequence[PublishedTranslationSample],
    run_id: str,
    git_commit: str,
    physical_leakage_clean: bool,
) -> dict[str, object]:
    """Validate, classify, and normalize one complete Stage A-prime report."""
    normalized_metrics = _normalize_scene_metrics(scene_metrics)
    normalized_cohort = _sorted_cohort_samples(cohort)
    return _compose_report(
        normalized_metrics,
        cohort=normalized_cohort,
        run_id=_run_id(run_id),
        git_commit=_git_commit(git_commit),
        physical_leakage_clean=_boolean(
            physical_leakage_clean, name="physical_leakage_clean"
        ),
    )


def _normalize_report_payload(payload: object) -> dict[str, object]:
    row = _exact_mapping(payload, _REPORT_KEYS, name="Stage A-prime report")
    if row["schema"] != REPORT_SCHEMA:
        raise ValueError("Stage A-prime report schema mismatch")
    run_id = _run_id(row["run_id"])
    git_commit = _git_commit(row["git_commit"])
    physical_leakage_clean = _boolean(
        row["physical_leakage_clean"], name="physical_leakage_clean"
    )
    scene_metrics = _normalize_scene_metrics(row["scene_metrics"])
    cohort_records = _normalize_cohort_records(row["cohort"])
    cohort = _samples_from_records(cohort_records)
    provided_classifier = _normalize_classifier_result(
        {name: row[name] for name in _CLASSIFIER_KEYS}
    )
    expected = _compose_report(
        scene_metrics,
        cohort=cohort,
        run_id=run_id,
        git_commit=git_commit,
        physical_leakage_clean=physical_leakage_clean,
    )
    provided = {
        "schema": REPORT_SCHEMA,
        "run_id": run_id,
        "git_commit": git_commit,
        **provided_classifier,
        "physical_leakage_clean": physical_leakage_clean,
        "scene_metrics": scene_metrics,
        "cohort": cohort_records,
    }
    if provided != expected:
        raise ValueError("Stage A-prime report does not match independent classification")
    return expected


def _json_bytes(payload: Mapping[str, object]) -> bytes:
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


def _canonical_digest(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        dict(payload),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _format_float(value: object) -> str:
    return format(_number(value, name="Markdown float"), ".17g")


def _markdown_bytes(payload: Mapping[str, object]) -> bytes:
    failed = payload["failed_gates"]
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
    for row in payload["scene_metrics"]:
        lines.append(
            f"| {row['scene']} | {row['role']} | {row['sample_id']} | "
            f"{_format_float(row['teacher_retention'])} | "
            f"{_format_float(row['mean_full_scene_utility'])} | "
            f"{_format_float(row['max_covered_roundtrip_fraction'])} | "
            f"{_format_float(row['max_uncovered_drift_fraction'])} | "
            f"{_format_float(row['max_rotation_delta_deg'])} |"
        )
    return ("\n".join(lines) + "\n").encode("utf-8")


def _absolute_without_resolving(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _reject_symlink_components(path: Path) -> None:
    if ".." in Path(path).parts:
        raise ValueError("report paths may not contain lexical parent traversal")
    current = _absolute_without_resolving(path)
    for candidate in (current, *current.parents):
        if candidate.is_symlink():
            raise ValueError(f"report paths may not contain symlinks: {candidate}")


def _preflight_publication(
    expected: Mapping[Path, bytes], *, completion_path: Path
) -> dict[Path, bool]:
    existing: dict[Path, bool] = {}
    for path, content in expected.items():
        _reject_symlink_components(path)
        if path.exists():
            if not path.is_file():
                raise ValueError(f"existing report target is not a regular file: {path}")
            try:
                current = path.read_bytes()
            except OSError as error:
                raise ValueError(f"could not read existing report target: {path}") from error
            if current != content:
                raise ValueError(f"existing report conflicts with recomputed bytes: {path}")
            existing[path] = True
        else:
            existing[path] = False
    if existing[completion_path] and not all(existing.values()):
        raise ValueError("completion exists without both byte-exact report files")
    return existing


def _require_exact_files(expected: Mapping[Path, bytes]) -> None:
    for path, content in expected.items():
        _reject_symlink_components(path)
        if not path.is_file():
            raise ValueError(f"report file changed after publication preflight: {path}")
        try:
            current = path.read_bytes()
        except OSError as error:
            raise ValueError(f"could not revalidate report file: {path}") from error
        if current != content:
            raise ValueError(f"report file no longer matches recomputed bytes: {path}")


def _atomic_publish_bytes(path: Path, content: bytes) -> None:
    target = Path(path)
    _reject_symlink_components(target)
    if target.exists():
        raise ValueError(f"report target appeared after preflight: {target}")
    temporary_path: Path | None = None
    publication_error: Exception | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_path, target)
    except Exception as error:
        publication_error = error
    cleanup_error: OSError | None = None
    if temporary_path is not None:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError as error:
            cleanup_error = error
    if publication_error is not None:
        detail = " and temporary cleanup failed" if cleanup_error else ""
        raise ValueError(
            f"could not atomically publish report target{detail}: {target}"
        ) from publication_error
    if cleanup_error is not None:
        raise ValueError(
            f"report target was published but temporary cleanup failed: {target}"
        ) from cleanup_error


def _rollback_exact_new_file(path: Path, content: bytes) -> None:
    """Remove only the exact completion bytes just published by this call."""
    _reject_symlink_components(path)
    if not path.is_file():
        raise ValueError(f"new completion target cannot be rolled back: {path}")
    try:
        current = path.read_bytes()
    except OSError as error:
        raise ValueError(f"could not authenticate new completion target: {path}") from error
    if current != content:
        raise ValueError(f"new completion target changed before rollback: {path}")
    try:
        path.unlink()
    except OSError as error:
        raise ValueError(f"could not roll back new completion target: {path}") from error


def write_stage_a_prime_report(
    run_root: Path, payload: Mapping[str, object]
) -> tuple[Path, Path, Path]:
    """Publish deterministic JSON, Markdown, then the signed completion record."""
    normalized = _normalize_report_payload(payload)
    try:
        root = Path(run_root)
    except TypeError as error:
        raise ValueError("run_root must be path-like") from error
    json_path = root / Path(REPORT_JSON_PATH)
    markdown_path = root / Path(REPORT_MARKDOWN_PATH)
    completion_path = root / Path(COMPLETION_PATH)
    json_content = _json_bytes(normalized)
    markdown_content = _markdown_bytes(normalized)
    unsigned_completion = {
        "schema": COMPLETION_SCHEMA,
        "run_id": normalized["run_id"],
        "git_commit": normalized["git_commit"],
        "classification": normalized["classification"],
        "scene_count": normalized["scene_count"],
        "endpoint_count": normalized["endpoint_count"],
        "report_json_path": REPORT_JSON_PATH,
        "report_json_sha256": hashlib.sha256(json_content).hexdigest(),
        "report_markdown_path": REPORT_MARKDOWN_PATH,
        "report_markdown_sha256": hashlib.sha256(markdown_content).hexdigest(),
    }
    completion = {
        **unsigned_completion,
        "completion_digest": _canonical_digest(unsigned_completion),
    }
    completion_content = _json_bytes(completion)
    expected = {
        json_path: json_content,
        markdown_path: markdown_content,
        completion_path: completion_content,
    }
    existing = _preflight_publication(
        expected, completion_path=completion_path
    )
    if all(existing.values()):
        _require_exact_files(expected)
        return json_path, markdown_path, completion_path
    try:
        json_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise ValueError("could not create reports directory") from error
    _reject_symlink_components(json_path.parent)
    for path in (json_path, markdown_path):
        if not existing[path]:
            _atomic_publish_bytes(path, expected[path])
    _require_exact_files(
        {
            json_path: expected[json_path],
            markdown_path: expected[markdown_path],
        }
    )
    if not existing[completion_path]:
        _atomic_publish_bytes(completion_path, expected[completion_path])
        try:
            _require_exact_files(expected)
        except ValueError as validation_error:
            try:
                _rollback_exact_new_file(
                    completion_path, expected[completion_path]
                )
            except ValueError as rollback_error:
                raise ValueError(
                    "report dependency validation failed and completion "
                    f"rollback failed: {rollback_error}"
                ) from validation_error
            raise
    return json_path, markdown_path, completion_path


__all__ = [
    "COMPLETION_SCHEMA",
    "REPORT_SCHEMA",
    "build_stage_a_prime_report",
    "write_stage_a_prime_report",
]
