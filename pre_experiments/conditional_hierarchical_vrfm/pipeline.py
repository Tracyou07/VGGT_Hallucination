"""Content-bound calibration-first pipeline for privileged latent lifting."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import unittest
from types import SimpleNamespace
from typing import Mapping, Sequence

import numpy as np
import torch

from pre_experiments.camera_velocity_ambiguity_02.frozen_oracle import FrozenOracle
from pre_experiments.conditional_hierarchical_vrfm.artifacts import (
    load_latent_targets,
    load_teacher_artifact,
    reuse_or_save_teacher_artifact,
    save_latent_targets,
)
from pre_experiments.conditional_hierarchical_vrfm.basis import canonical_basis_sha256
from pre_experiments.conditional_hierarchical_vrfm.evaluate import (
    classify_stage_a,
    evaluate_latent_targets,
)
from pre_experiments.conditional_hierarchical_vrfm.lift import (
    LiftConfig,
    decode_coefficients,
    load_lift_checkpoint,
    optimize_latent_target,
)
from pre_experiments.conditional_hierarchical_vrfm.report import (
    REPORT_SCHEMA,
    write_stage_a_report,
)
from pre_experiments.conditional_hierarchical_vrfm.teacher import (
    TeacherVariantSet,
    build_teacher_variants,
    summarize_teacher_upper_bound,
)
from pre_experiments.long_short_camera_head.data import (
    LongContextRecord,
    SceneRecord,
    load_long_context,
    load_source_records,
    publish_long_context,
)
from pre_experiments.long_short_camera_head.labels import load_privileged_labels
from pre_experiments.long_short_camera_head.train import load_base_camera_head
from pre_experiments.variational_camera_latent.source import load_source_shard


LONG_MANIFEST_SCHEMA = "conditional_hierarchical_vrfm.long_context_manifest.v1"
INVENTORY_SCHEMA = "conditional_hierarchical_vrfm.verification_inventory.v1"
VERIFIED_SCHEMA = "conditional_hierarchical_vrfm.verified_completion.v1"
SMOKE_SCHEMA = "conditional_hierarchical_vrfm.smoke_completion.v1"
CALIBRATION_SCHEMA = "conditional_hierarchical_vrfm.calibration_completion.v1"
PREFLIGHT_SCHEMA = "conditional_hierarchical_vrfm.preflight_evidence.v1"
TEACHER_MANIFEST_SCHEMA = "conditional_hierarchical_vrfm.teacher_manifest.v1"
EXPECTED_SCENES = (
    "scene0000_00", "scene0013_02", "scene0029_01", "scene0084_01",
    "scene0121_01", "scene0207_01", "scene0280_00", "scene0325_01",
    "scene0675_00", "scene0691_00",
)
FROZEN_SPLIT_CONFIG_DIGEST = "81386f891b45ce8d2dc7706c9e64bf7783931d6eaf16154ff402beae13227fce"
FROZEN_FORMAL_RUN_NAME = "long_short_head_formal_20260828T072407Z"
FROZEN_FORMAL_GIT = "2476a59f583ce4c39bbe66dc65d6a8e5cddfb52e"
FROZEN_BASE_CHECKPOINT_SHA256 = "f164acf60724910d8fe1578bb499d800850c7bb0948db7555c413f9fbe60467e"
EXPECTED_TEACHER_COVERAGE = 0.89
EXPECTED_TEACHER_UTILITY = 0.1293578271441714
PREFLIGHT_SUITES = (
    ("tests/conditional_hierarchical_vrfm", 71),
    ("tests/variational_camera_latent", 64),
    ("tests/long_short_camera_head", 39),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_digest(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _read_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON artifact: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return payload


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)


def _reuse_or_publish_json(path: Path, payload: Mapping[str, object], label: str) -> None:
    expected = dict(payload)
    _reject_existing_symlink_components(path)
    if Path(path).exists():
        if not Path(path).is_file() or _read_json(path) != expected:
            raise ValueError(f"existing {label} does not match resumed publication")
        return
    _atomic_json(path, expected)


def validate_frozen_scene_identity(config_path: Path) -> None:
    """Cross-check the Task-4 cohort against the frozen CVA02 split identity."""
    payload = _read_json(config_path)
    cohorts = payload.get("cohorts")
    calibration = cohorts.get("calibration") if isinstance(cohorts, Mapping) else None
    scene_order = payload.get("scene_order")
    if (
        payload.get("config_digest") != FROZEN_SPLIT_CONFIG_DIGEST
        or not isinstance(calibration, list)
        or set(map(str, calibration)) != set(EXPECTED_SCENES)
        or len(calibration) != len(EXPECTED_SCENES)
        or not isinstance(scene_order, list)
        or not set(EXPECTED_SCENES).issubset(set(map(str, scene_order)))
    ):
        raise ValueError("frozen calibration scene identity mismatch")


def validate_source_scene_cohort(records: Sequence[object]) -> None:
    scenes = [str(getattr(record, "scene", "")) for record in records]
    roles = [str(getattr(record, "role", "")) for record in records]
    if len(scenes) != 10 or len(set(scenes)) != 10 or set(scenes) != set(EXPECTED_SCENES):
        raise ValueError("source manifest does not bind the exact ten calibration scenes")
    expected_roles = {
        scene: "validation" if scene in {"scene0325_01", "scene0675_00"} else "train"
        for scene in EXPECTED_SCENES
    }
    if any(expected_roles[scene] != role for scene, role in zip(scenes, roles)):
        raise ValueError("source scene roles must match the frozen eight/two split")


def build_long_context_manifest(source_manifest: Mapping[str, object]) -> dict[str, object]:
    """Derive a manifest that can name only physical long-only files under the run root."""
    rows = source_manifest.get("records")
    if not isinstance(rows, list) or not rows:
        raise ValueError("source manifest records are required")
    output: list[dict[str, object]] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("source manifest record must be an object")
        scene = str(row.get("scene", ""))
        role = str(row.get("role", ""))
        digest = str(row.get("sha256", ""))
        if not scene or scene in seen or role not in {"train", "validation", "smoke"} or len(digest) != 64:
            raise ValueError("source manifest record identity is invalid")
        output.append({
            "scene": scene,
            "role": role,
            "file": f"{scene}.npz",
            "sha256": "0" * 64,
            "source_sha256": digest,
        })
        seen.add(scene)
    return {"schema": LONG_MANIFEST_SCHEMA, "records": output}


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _reject_existing_symlink_components(path: Path) -> None:
    lexical = Path(path).absolute()
    for component in (lexical, *lexical.parents):
        if component.exists() and component.is_symlink():
            raise ValueError(f"publication path contains a symlink: {component}")


def validate_long_context_publication_root(run_root: Path) -> Path:
    """Reject lexical symlink/escape hazards before resolving a publication root."""
    lexical_root = Path(run_root).absolute()
    lexical_base = lexical_root / "prediction_only" / "long_context"
    _reject_existing_symlink_components(lexical_base)
    resolved_root = lexical_root.resolve()
    resolved_base = lexical_base.resolve()
    if not _within(resolved_base, resolved_root):
        raise ValueError("long-context publication path escapes the run root")
    return resolved_base


def reuse_or_publish_long_context(
    record: SceneRecord, destination: Path
) -> LongContextRecord:
    """Reuse only a byte-valid shard that exactly equals the source-derived payload."""
    target = Path(destination)
    _reject_existing_symlink_components(target)
    if not target.exists():
        return publish_long_context(record, target)
    try:
        existing = load_long_context(target)
        source = load_source_shard(record.path)
    except ValueError as error:
        raise ValueError("existing long-context shard is invalid") from error
    expected = {
        "scene": np.asarray(record.scene, dtype="U32"),
        "frame_ids": source["global_frame_ids"].astype(np.int64, copy=False),
        "camera_tokens": source["global_camera_tokens"].astype(np.float32, copy=False),
        "baseline_c2w": source["global_pred_c2w"].astype(np.float64, copy=False),
        "source_sha256": np.asarray(record.sha256, dtype="U64"),
    }
    if any(
        existing[name].dtype != value.dtype
        or existing[name].shape != value.shape
        or not np.array_equal(existing[name], value)
        for name, value in expected.items()
    ):
        raise ValueError("existing long-context shard does not match expected bytes/provenance")
    return LongContextRecord(
        scene=record.scene,
        role=record.role,
        path=target,
        sha256=sha256_file(target),
        source_sha256=record.sha256,
    )


def audit_long_context_manifest(run_root: Path, manifest: Mapping[str, object]) -> None:
    validate_long_context_publication_root(run_root)
    root = Path(run_root).resolve()
    if manifest.get("schema") != LONG_MANIFEST_SCHEMA:
        raise ValueError("long-context manifest schema mismatch")
    rows = manifest.get("records")
    if not isinstance(rows, list) or not rows:
        raise ValueError("long-context manifest is empty")
    base = (root / "prediction_only" / "long_context").resolve()
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {"scene", "role", "file", "sha256", "source_sha256"}:
            raise ValueError("long-context manifest record schema mismatch")
        name = str(row["file"])
        if name != f"{row['scene']}.npz" or Path(name).name != name:
            raise ValueError("long-context manifest file identity mismatch")
        lexical = base / name
        if lexical.is_symlink():
            raise ValueError("long-context shard may not be a symlink")
        resolved = lexical.resolve()
        if not _within(resolved, base) or not resolved.is_file():
            raise ValueError("long-context resolved-path escape or missing shard")
        if sha256_file(resolved) != row["sha256"]:
            raise ValueError("long-context shard digest mismatch")
        try:
            arrays = load_long_context(resolved)
        except ValueError as error:
            raise ValueError("physical shard is not strict long-only data") from error
        if str(arrays["scene"]) != row["scene"] or str(arrays["source_sha256"]) != row["source_sha256"]:
            raise ValueError("long-context semantic binding mismatch")


def validate_variant_zero_against_formal(
    teacher: TeacherVariantSet | object, formal: Mapping[str, object]
) -> None:
    """Require the rederived all-positive teacher to equal authenticated formal bytes."""
    comparisons = (
        (teacher.frame_ids, formal["frame_ids"]),
        (teacher.window_weights, formal["window_teacher_weight"]),
        (teacher.coverage_weights[0], formal["teacher_weight"]),
    )
    if any(not np.array_equal(np.asarray(left), np.asarray(right)) for left, right in comparisons):
        raise ValueError("rederived teacher variant zero differs from the formal label")
    if not np.allclose(
        np.asarray(teacher.fused_c2w[0]),
        np.asarray(formal["teacher_c2w_gt_gauge"]),
        atol=0.0,
        rtol=0.0,
        equal_nan=True,
    ):
        raise ValueError("rederived teacher variant zero differs from the formal label")


def authenticate_formal_run(
    formal_root: Path,
    source_run: Path,
    records: Sequence[object],
    checkpoint_sha256: str,
) -> dict[str, object]:
    """Authenticate the full frozen formal-run chain before trusting any label shard."""
    root = Path(formal_root).absolute()
    _reject_existing_symlink_components(root)
    if root.name != FROZEN_FORMAL_RUN_NAME or not root.is_dir():
        raise ValueError("labels must come from the verified formal run")
    root = root.resolve()
    marker_path = root / "verified_completion.json"
    manifest_path = root / "manifests" / "data_manifest.json"
    if not marker_path.is_file() or not manifest_path.is_file():
        raise ValueError("labels must come from the verified formal run")
    marker = _read_json(marker_path)
    manifest = _read_json(manifest_path)
    marker_fields = {
        "schema", "git_revision", "verifier_git_revision",
        "source_manifest_sha256", "base_checkpoint_sha256", "config_sha256",
        "data_manifest_sha256", "test_evidence_sha256", "stage_completion_sha256",
        "scene_count", "train_scene_count", "locked_replay_scene_count",
        "classification", "report_sha256", "artifacts", "inference_leakage_audit",
        "formal_protocol_sha256",
    }
    manifest_fields = {
        "schema", "git_revision", "source_run", "source_manifest_sha256",
        "prepared_root", "checkpoint_dir", "base_checkpoint_sha256", "records",
    }
    if set(marker) != marker_fields or marker.get("schema") != "long_short_camera_head.verified_completion.v1":
        raise ValueError("verified formal run completion schema mismatch")
    if set(manifest) != manifest_fields or manifest.get("schema") != "long_short_camera_head.data_manifest.v1":
        raise ValueError("verified formal run data manifest schema mismatch")
    if sha256_file(manifest_path) != marker.get("data_manifest_sha256"):
        raise ValueError("formal data manifest digest mismatch")
    source_root = Path(source_run).resolve()
    source_manifest = source_root / "manifests" / "source_manifest.json"
    if not source_manifest.is_file():
        raise ValueError("formal source manifest is unavailable")
    source_manifest_sha = sha256_file(source_manifest)
    identities = (
        marker.get("git_revision") == FROZEN_FORMAL_GIT,
        manifest.get("git_revision") == FROZEN_FORMAL_GIT,
        marker.get("classification") == "NO_SOURCE_HEAD_SIGNAL",
        marker.get("base_checkpoint_sha256") == FROZEN_BASE_CHECKPOINT_SHA256,
        manifest.get("base_checkpoint_sha256") == FROZEN_BASE_CHECKPOINT_SHA256,
        checkpoint_sha256 == FROZEN_BASE_CHECKPOINT_SHA256,
        marker.get("source_manifest_sha256") == source_manifest_sha,
        manifest.get("source_manifest_sha256") == source_manifest_sha,
        Path(str(manifest.get("source_run"))).resolve() == source_root,
        marker.get("scene_count") == 10,
        marker.get("train_scene_count") == 8,
        marker.get("locked_replay_scene_count") == 2,
        bool(marker.get("inference_leakage_audit")),
    )
    if not all(identities):
        raise ValueError("verified formal run identity mismatch")
    validate_source_scene_cohort(records)
    expected = {str(getattr(record, "scene")): record for record in records}
    rows = manifest.get("records")
    if not isinstance(rows, list) or len(rows) != 10:
        raise ValueError("formal data manifest must bind ten scenes")
    row_fields = {
        "scene", "role", "source_path", "source_sha256", "long_context_path",
        "long_context_sha256", "privileged_path", "privileged_sha256",
        "teacher_frame_count",
    }
    labels: dict[str, Path] = {}
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != row_fields:
            raise ValueError("formal data manifest record schema mismatch")
        scene = str(row["scene"])
        if scene not in expected or scene in labels:
            raise ValueError("formal data manifest scene cohort mismatch")
        record = expected[scene]
        if row["role"] != getattr(record, "role") or row["source_sha256"] != getattr(record, "sha256"):
            raise ValueError("formal source record binding mismatch")
        record_path = getattr(record, "path", None)
        if record_path is not None:
            actual_source = Path(record_path)
            if (
                actual_source.is_symlink()
                or not actual_source.is_file()
                or Path(str(row["source_path"])).resolve() != actual_source.resolve()
                or sha256_file(actual_source) != row["source_sha256"]
            ):
                raise ValueError("formal source record binding mismatch")
        label = root / "data" / "privileged_labels" / f"{scene}.npz"
        _reject_existing_symlink_components(label)
        if (
            Path(str(row["privileged_path"])).resolve() != label.resolve()
            or not label.is_file()
            or sha256_file(label) != row["privileged_sha256"]
        ):
            raise ValueError("formal privileged label digest/path mismatch")
        arrays = load_privileged_labels(label)
        if (
            str(arrays["scene"]) != scene
            or str(arrays["source_sha256"]) != row["source_sha256"]
            or str(arrays["checkpoint_sha256"]) != checkpoint_sha256
            or int(np.count_nonzero(arrays["teacher_weight"])) != row["teacher_frame_count"]
        ):
            raise ValueError("formal privileged label semantic binding mismatch")
        labels[scene] = label
    if set(labels) != set(EXPECTED_SCENES):
        raise ValueError("formal data manifest scene cohort mismatch")
    return {
        "labels": labels,
        "completion_sha256": sha256_file(marker_path),
        "data_manifest_sha256": sha256_file(manifest_path),
        "source_manifest_sha256": source_manifest_sha,
        "formal_root": root,
    }
def _teacher_arrays(
    teacher: TeacherVariantSet,
    formal: Mapping[str, np.ndarray],
    baseline_c2w_raw: np.ndarray,
    *,
    source_sha256: str,
    formal_label_sha256: str,
    git_commit: str,
) -> dict[str, np.ndarray]:
    oracle = teacher.oracle
    return {
        "scene": np.asarray(teacher.scene, dtype="U32"),
        "frame_ids": teacher.frame_ids.astype(np.int64, copy=True),
        "gt_c2w": np.asarray(formal["gt_c2w"], dtype=np.float64).copy(),
        "gt_scene_scale": np.asarray(formal["gt_scene_scale"], dtype=np.float64),
        "baseline_c2w_raw": np.asarray(baseline_c2w_raw, dtype=np.float64).copy(),
        "oracle_scene": np.asarray(oracle.scene, dtype="U32"),
        "oracle_frame_digest": np.asarray(oracle.frame_digest, dtype="U64"),
        "oracle_fit_count": np.asarray(oracle.fit_count, dtype=np.int64),
        "oracle_scale": np.asarray(oracle.scale, dtype=np.float64),
        "oracle_rotation": np.asarray(oracle.rotation, dtype=np.float64),
        "oracle_translation": np.asarray(oracle.translation, dtype=np.float64),
        "oracle_rank": np.asarray(oracle.rank, dtype=np.int64),
        "oracle_condition": np.asarray(oracle.condition, dtype=np.float64),
        "oracle_digest": np.asarray(oracle.transform_digest, dtype="U64"),
        "window_weights": teacher.window_weights.astype(np.float64, copy=True),
        "window_masks": teacher.window_masks.astype(np.uint8, copy=True),
        "coverage_weights": teacher.coverage_weights.astype(np.float64, copy=True),
        "fused_c2w": teacher.fused_c2w.astype(np.float64, copy=True),
        "variant_utilities": teacher.variant_utilities.astype(np.float64, copy=True),
        "source_sha256": np.asarray(source_sha256, dtype="U64"),
        "formal_label_sha256": np.asarray(formal_label_sha256, dtype="U64"),
        "checkpoint_sha256": np.asarray(teacher.checkpoint_sha256, dtype="U64"),
        "git_commit": np.asarray(git_commit, dtype="U40"),
    }


def _current_git() -> tuple[str, str]:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], text=True)
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("cannot inspect Git state") from error
    return commit, dirty


def _validate_git(expected: str) -> None:
    commit, dirty = _current_git()
    if commit != expected or dirty:
        raise ValueError("pipeline requires the exact clean Git commit")


def preflight_source_inventory(repository_root: Path | None = None) -> dict[str, dict[str, str]]:
    """Return exact byte inventories for the tested suites and their source trees."""
    root = Path.cwd().resolve() if repository_root is None else Path(repository_root).resolve()
    groups = {
        "tests": tuple(Path(suite) for suite, _ in PREFLIGHT_SUITES),
        "sources": (
            Path("pre_experiments"),
            Path("vggt"),
        ),
    }
    inventory: dict[str, dict[str, str]] = {}
    for group, directories in groups.items():
        rows: dict[str, str] = {}
        for directory in directories:
            if not (root / directory).exists():
                continue
            for path in sorted((root / directory).rglob("*.py")):
                if path.is_symlink() or not path.is_file():
                    raise ValueError("preflight source inventory contains an invalid path")
                rows[path.relative_to(root).as_posix()] = sha256_file(path)
        if not rows:
            raise ValueError("preflight source inventory is empty")
        inventory[group] = rows
    frozen = root / "configs" / "scannet50_camera_velocity_ambiguity_02_split_v2.json"
    inventory["frozen_config"] = {frozen.relative_to(root).as_posix(): sha256_file(frozen)}
    return inventory


def git_tree_identity() -> str:
    try:
        value = subprocess.check_output(
            ["git", "rev-parse", "HEAD^{tree}"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("cannot inspect Git tree identity") from error
    if re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise ValueError("Git tree identity is malformed")
    return value


def preflight_test_inventory(repository_root: Path | None = None) -> dict[str, list[str]]:
    """Discover and bind the exact stable unittest IDs for each required suite."""
    root = Path.cwd().resolve() if repository_root is None else Path(repository_root).resolve()

    def collect(suite: unittest.TestSuite) -> list[str]:
        output: list[str] = []
        for test in suite:
            if isinstance(test, unittest.TestSuite):
                output.extend(collect(test))
            else:
                output.append(test.id())
        return output

    inventory: dict[str, list[str]] = {}
    for suite, expected_count in PREFLIGHT_SUITES:
        discovered = unittest.defaultTestLoader.discover(
            str(root / suite), pattern="test_*.py", top_level_dir=str(root)
        )
        identifiers = sorted(collect(discovered))
        if len(identifiers) != expected_count or len(set(identifiers)) != expected_count:
            raise ValueError("preflight unittest inventory/count mismatch")
        inventory[suite] = identifiers
    return inventory


def _preflight_commands() -> list[list[str]]:
    return [
        [sys.executable, "-m", "unittest", "discover", "-s", suite, "-v"]
        for suite, _ in PREFLIGHT_SUITES
    ] + [[sys.executable, "-m", "compileall", "-q", "pre_experiments"]]


def _parse_unittest_results(
    content: str, suite: str, expected_ids: Sequence[str]
) -> list[dict[str, str]]:
    starts = list(re.finditer(r"(?m)^(test\S+) \(([^)\r\n]+)\)", content))
    observed: dict[str, str] = {}
    prefix = suite.replace("/", ".").replace("\\", ".")
    for index, match in enumerate(starts):
        block_end = starts[index + 1].start() if index + 1 < len(starts) else len(content)
        block = content[match.end():block_end]
        status_matches = list(re.finditer(r"(?m)\.\.\.\s+(ok|skipped\b[^\r\n]*)\s*$", block))
        if len(status_matches) != 1:
            raise ValueError("preflight unittest output lacks one terminal status per test")
        owner = match.group(2)
        identifier = f"{owner}.{match.group(1)}"
        if identifier not in expected_ids:
            identifier = f"{prefix}.{identifier}"
        status = "skipped" if status_matches[0].group(1).startswith("skipped") else "ok"
        if identifier in observed:
            raise ValueError("preflight unittest output contains duplicate test IDs")
        observed[identifier] = status
    if set(observed) != set(expected_ids) or len(observed) != len(expected_ids):
        raise ValueError("preflight unittest output does not match the stable test IDs")
    return [{"id": identifier, "status": observed[identifier]} for identifier in sorted(observed)]


def _execute_preflight_commands(
    test_inventory: Mapping[str, Sequence[str]], root: Path | None = None
) -> list[dict[str, object]]:
    completed_commands = [
        (
            command,
            subprocess.run(
                command, text=True, capture_output=True, check=False, timeout=1800,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
            ),
        )
        for command in _preflight_commands()
    ]
    rows: list[dict[str, object]] = []
    for index, (command, completed) in enumerate(completed_commands):
        content = completed.stdout + completed.stderr
        test_results: list[dict[str, str]] = []
        if index < len(PREFLIGHT_SUITES):
            suite, expected_count = PREFLIGHT_SUITES[index]
            test_results = _parse_unittest_results(content, suite, test_inventory[suite])
            if len(test_results) != expected_count:
                raise ValueError("preflight live test count mismatch")
        row: dict[str, object] = {
            "command": command,
            "returncode": completed.returncode,
            "test_count": len(test_results),
            "skipped_count": sum(result["status"] == "skipped" for result in test_results),
            "test_results": test_results,
        }
        if root is not None:
            log = root / "logs" / f"preflight_{index}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text(content, encoding="utf-8")
            row.update({
                "log": log.relative_to(root).as_posix(),
                "log_sha256": sha256_file(log),
            })
        rows.append(row)
    return rows


def run_preflight(args: argparse.Namespace | SimpleNamespace) -> Path:
    """Execute every CPU contract suite plus compileall and bind their actual logs."""
    _validate_git(args.git_commit)
    root = Path(args.run_root).resolve()
    evidence_path = root / "manifests" / "preflight_evidence.json"
    if evidence_path.is_file():
        _validate_preflight(root, args.git_commit)
        return evidence_path
    test_inventory = preflight_test_inventory()
    rows = _execute_preflight_commands(test_inventory, root)
    if any(row["returncode"] != 0 for row in rows):
        raise ValueError("preflight command failed")
    unsigned = {
        "schema": PREFLIGHT_SCHEMA,
        "git_commit": args.git_commit,
        "source_inventory": preflight_source_inventory(),
        "test_inventory": test_inventory,
        "git_tree": git_tree_identity(),
        "commands": rows,
    }
    payload = {**unsigned, "record_digest": _canonical_digest(unsigned)}
    _atomic_json(evidence_path, payload)
    return evidence_path


def _validate_preflight(root: Path, git_commit: str) -> dict[str, object]:
    path = root / "manifests" / "preflight_evidence.json"
    payload = _read_json(path)
    digest = payload.pop("record_digest", None)
    if payload.get("schema") != PREFLIGHT_SCHEMA or payload.get("git_commit") != git_commit:
        raise ValueError("preflight evidence commit binding mismatch")
    if digest != _canonical_digest(payload):
        raise ValueError("preflight evidence record digest mismatch")
    if payload.get("source_inventory") != preflight_source_inventory():
        raise ValueError("preflight evidence source inventory mismatch")
    if payload.get("test_inventory") != preflight_test_inventory():
        raise ValueError("preflight evidence unittest inventory mismatch")
    if payload.get("git_tree") != git_tree_identity():
        raise ValueError("preflight evidence Git tree mismatch")
    rows = payload.get("commands")
    if not isinstance(rows, list) or len(rows) != 4:
        raise ValueError("preflight evidence command count mismatch")
    expected_commands = _preflight_commands()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or row.get("returncode") != 0:
            raise ValueError("preflight evidence command failure")
        if row.get("command") != expected_commands[index]:
            raise ValueError("preflight evidence command mismatch")
        expected_count = 0
        expected_results: list[dict[str, str]] = []
        if index < 3:
            suite, expected_count = PREFLIGHT_SUITES[index]
            expected_ids = payload["test_inventory"][suite]
            results = row.get("test_results")
            if (
                not isinstance(results, list)
                or any(
                    not isinstance(result, Mapping)
                    or set(result) != {"id", "status"}
                    or result.get("status") not in {"ok", "skipped"}
                    for result in results
                )
            ):
                raise ValueError("preflight evidence test result schema mismatch")
            expected_results = [dict(result) for result in results]
            if [result["id"] for result in expected_results] != expected_ids:
                raise ValueError("preflight evidence stable test ID mismatch")
        if (
            row.get("test_count") != expected_count
            or not isinstance(row.get("skipped_count"), int)
            or int(row["skipped_count"])
            != sum(result["status"] == "skipped" for result in expected_results)
            or (index == 3 and row.get("test_results") != [])
        ):
            raise ValueError("preflight evidence test count mismatch")
        log = root / str(row.get("log", ""))
        if not log.is_file() or sha256_file(log) != row.get("log_sha256"):
            raise ValueError("preflight evidence log digest mismatch")
        if index < 3:
            content = log.read_text(encoding="utf-8")
            observed_results = _parse_unittest_results(
                content, PREFLIGHT_SUITES[index][0], payload["test_inventory"][PREFLIGHT_SUITES[index][0]]
            )
            if observed_results != expected_results or not re.search(r"^OK", content, re.MULTILINE):
                raise ValueError("preflight evidence log does not prove a passing test run")
    payload["record_digest"] = digest
    return payload


def validate_preflight_evidence(root: Path, git_commit: str) -> dict[str, object]:
    """Public verifier used by every later stage and barrier regressions."""
    return _validate_preflight(Path(root).resolve(), git_commit)


def _validate_preflight_live(root: Path, git_commit: str) -> dict[str, object]:
    """Re-execute the bound commands; static digests alone cannot authorize a stage."""
    payload = _validate_preflight(root, git_commit)
    live_rows = _execute_preflight_commands(payload["test_inventory"])
    evidence_rows = payload["commands"]
    for live, evidence in zip(live_rows, evidence_rows):
        if (
            live["command"] != evidence["command"]
            or live["returncode"] != 0
            or live["test_count"] != evidence["test_count"]
            or live["skipped_count"] != evidence["skipped_count"]
            or live["test_results"] != evidence["test_results"]
        ):
            raise ValueError("authoritative live preflight does not match bound evidence")
    return payload


def run_prepare(args: argparse.Namespace | SimpleNamespace) -> Path:
    """Publish physically separated long-only shards and strict teacher sidecars."""
    _validate_git(args.git_commit)
    validate_long_context_publication_root(args.run_root)
    root = Path(args.run_root).resolve()
    _validate_preflight_live(root, args.git_commit)
    existing = (
        root / "config.json", root / "manifests" / "long_context.json",
        root / "manifests" / "teacher.json",
    )
    if all(path.is_file() for path in existing):
        config, _, teacher_manifest = _load_prepared_manifests(root, args.git_commit)
        summary = teacher_manifest.get("teacher_upper_bound")
        if summary != {
            "scene_count": 10, "positive_scene_count": 10,
            "mean_coverage": EXPECTED_TEACHER_COVERAGE,
            "mean_utility": EXPECTED_TEACHER_UTILITY,
        }:
            raise ValueError("existing teacher replay summary mismatch")
        return root / "manifests" / "long_context.json"
    records = load_source_records(Path(args.source_run))
    validate_frozen_scene_identity(
        Path("configs/scannet50_camera_velocity_ambiguity_02_split_v2.json")
    )
    validate_source_scene_cohort(records)
    camera_head, checkpoint_sha256 = load_base_camera_head(Path(args.checkpoint_dir))
    formal_auth = authenticate_formal_run(
        Path(args.formal_label_root), Path(args.source_run), records, checkpoint_sha256
    )
    formal_labels = formal_auth["labels"]
    if not isinstance(formal_labels, Mapping):
        raise ValueError("formal label authentication result is malformed")
    device = torch.device(args.device)
    camera_head = camera_head.to(device).eval()
    manifest = build_long_context_manifest({
        "records": [
            {"scene": record.scene, "role": record.role, "sha256": record.sha256}
            for record in records
        ]
    })
    manifest_rows = {str(row["scene"]): row for row in manifest["records"]}
    teacher_rows: list[dict[str, object]] = []
    teachers: list[TeacherVariantSet] = []
    for record in records:
        long_path = root / "prediction_only" / "long_context" / f"{record.scene}.npz"
        published = reuse_or_publish_long_context(record, long_path)
        manifest_rows[record.scene]["sha256"] = published.sha256
        formal_path = Path(formal_labels[record.scene])
        formal = load_privileged_labels(formal_path)
        teacher = build_teacher_variants(
            record.path, Path(args.prepared_root) / record.scene, camera_head,
            checkpoint_sha256=checkpoint_sha256, device=device,
        )
        validate_variant_zero_against_formal(teacher, formal)
        long_context = load_long_context(long_path)
        teacher_path = root / "privileged_labels" / "teacher" / f"{record.scene}.npz"
        _reject_existing_symlink_components(teacher_path)
        if not _within(teacher_path.resolve(), root):
            raise ValueError("teacher publication path escapes the run root")
        arrays = _teacher_arrays(
            teacher, formal, long_context["baseline_c2w"], source_sha256=record.sha256,
            formal_label_sha256=sha256_file(formal_path), git_commit=args.git_commit,
        )
        teacher_sha = reuse_or_save_teacher_artifact(teacher_path, arrays)
        teacher_rows.append({
            "scene": record.scene, "role": record.role,
            "file": teacher_path.relative_to(root).as_posix(), "sha256": teacher_sha,
            "formal_label_sha256": sha256_file(formal_path),
        })
        teachers.append(teacher)
        del arrays, teacher, formal, long_context
        if device.type == "cuda":
            torch.cuda.empty_cache()
    summary = summarize_teacher_upper_bound(teachers)
    if (
        summary["scene_count"] != 10
        or summary["positive_scene_count"] != 10
        or float(summary["mean_coverage"]) != EXPECTED_TEACHER_COVERAGE
        or float(summary["mean_utility"]) != EXPECTED_TEACHER_UTILITY
    ):
        raise ValueError("teacher replay does not match the authenticated ten-scene upper bound")
    long_manifest_path = root / "manifests" / "long_context.json"
    teacher_manifest_path = root / "manifests" / "teacher.json"
    audit_long_context_manifest(root, manifest)
    _reuse_or_publish_json(long_manifest_path, manifest, "long-context manifest")
    teacher_manifest = {
        "schema": TEACHER_MANIFEST_SCHEMA, "git_commit": args.git_commit,
        "checkpoint_sha256": checkpoint_sha256, "teacher_upper_bound": summary,
        "formal_completion_sha256": formal_auth["completion_sha256"],
        "formal_data_manifest_sha256": formal_auth["data_manifest_sha256"],
        "records": teacher_rows,
    }
    _reuse_or_publish_json(teacher_manifest_path, teacher_manifest, "teacher manifest")
    config = {
        "schema": "conditional_hierarchical_vrfm.run_config.v1",
        "git_commit": args.git_commit,
        "checkpoint_sha256": checkpoint_sha256,
        "basis_sha256": canonical_basis_sha256(),
        "long_manifest_sha256": sha256_file(long_manifest_path),
        "teacher_manifest_sha256": sha256_file(teacher_manifest_path),
        "source_run": str(Path(args.source_run).resolve()),
        "source_manifest_sha256": formal_auth["source_manifest_sha256"],
        "formal_run_root": str(Path(formal_auth["formal_root"])),
        "formal_completion_sha256": formal_auth["completion_sha256"],
        "formal_data_manifest_sha256": formal_auth["data_manifest_sha256"],
        "smoke_scene": "scene0000_00", "smoke_steps": 20, "calibration_steps": 250,
        "scene_count": 10, "variant_count": 4,
    }
    config_path = root / "config.json"
    _reuse_or_publish_json(config_path, config, "immutable run config")
    return long_manifest_path


def _load_prepared_manifests(root: Path, git_commit: str) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    config = _read_json(root / "config.json")
    if config.get("git_commit") != git_commit or config.get("basis_sha256") != canonical_basis_sha256():
        raise ValueError("run config binding mismatch")
    long_manifest = _read_json(root / "manifests" / "long_context.json")
    teacher_manifest = _read_json(root / "manifests" / "teacher.json")
    if sha256_file(root / "manifests" / "long_context.json") != config.get("long_manifest_sha256"):
        raise ValueError("long manifest digest mismatch")
    if sha256_file(root / "manifests" / "teacher.json") != config.get("teacher_manifest_sha256"):
        raise ValueError("teacher manifest digest mismatch")
    audit_long_context_manifest(root, long_manifest)
    if teacher_manifest.get("schema") != TEACHER_MANIFEST_SCHEMA or teacher_manifest.get("git_commit") != git_commit:
        raise ValueError("teacher manifest binding mismatch")
    records = load_source_records(Path(str(config.get("source_run", ""))))
    formal_auth = authenticate_formal_run(
        Path(str(config.get("formal_run_root", ""))),
        Path(str(config.get("source_run", ""))),
        records,
        str(config.get("checkpoint_sha256", "")),
    )
    for name in ("completion_sha256", "data_manifest_sha256", "source_manifest_sha256"):
        config_name = "formal_" + name if name != "source_manifest_sha256" else name
        if config.get(config_name) != formal_auth[name]:
            raise ValueError("run config formal authentication binding mismatch")
    if (
        teacher_manifest.get("formal_completion_sha256") != formal_auth["completion_sha256"]
        or teacher_manifest.get("formal_data_manifest_sha256") != formal_auth["data_manifest_sha256"]
    ):
        raise ValueError("teacher manifest formal authentication binding mismatch")
    rows = teacher_manifest.get("records")
    if not isinstance(rows, list) or len(rows) != 10:
        raise ValueError("teacher manifest must bind ten scenes")
    for row in rows:
        path = root / str(row["file"])
        if path.is_symlink() or not _within(path.resolve(), root) or sha256_file(path) != row["sha256"]:
            raise ValueError("teacher manifest artifact mismatch")
        load_teacher_artifact(path)
        formal_label = formal_auth["labels"][str(row["scene"])]
        if row.get("formal_label_sha256") != sha256_file(Path(formal_label)):
            raise ValueError("teacher manifest formal label binding mismatch")
    return config, long_manifest, teacher_manifest


def _require_capacity(root: Path) -> None:
    usage = shutil.disk_usage(root if root.exists() else root.parent)
    if usage.free < 100 * 1024**3:
        raise ValueError("GPU stage requires at least 100 GiB free")


def _require_run_size(root: Path) -> None:
    total = sum(path.stat().st_size for path in root.rglob("*") if path.is_file() and not path.is_symlink())
    if total >= 20 * 1024**3:
        raise ValueError("run root exceeds 20 GiB")


def _oracle_from_teacher(arrays: Mapping[str, np.ndarray]) -> FrozenOracle:
    return FrozenOracle(
        scene=str(arrays["oracle_scene"]), frame_digest=str(arrays["oracle_frame_digest"]),
        fit_count=int(arrays["oracle_fit_count"]), scale=float(arrays["oracle_scale"]),
        rotation=tuple(tuple(float(value) for value in row) for row in arrays["oracle_rotation"]),
        translation=tuple(float(value) for value in arrays["oracle_translation"]),
        rank=int(arrays["oracle_rank"]), condition=float(arrays["oracle_condition"]),
        transform_digest=str(arrays["oracle_digest"]),
    )


def _target_bindings_valid(
    target: Mapping[str, np.ndarray], long: Mapping[str, np.ndarray], teacher_path: Path,
    teacher: Mapping[str, np.ndarray], *, steps: int, git_commit: str,
) -> None:
    validate_target_for_stage(target, steps=steps)
    if str(target["scene"]) != str(long["scene"]) or not np.array_equal(target["frame_ids"], long["frame_ids"]):
        raise ValueError("existing target scene/frame binding mismatch")
    if str(target["source_sha256"]) != str(long["source_sha256"]):
        raise ValueError("existing target source binding mismatch")
    if str(target["teacher_sha256"]) != sha256_file(teacher_path):
        raise ValueError("existing target teacher binding mismatch")
    if str(target["basis_sha256"]) != canonical_basis_sha256():
        raise ValueError("existing target basis binding mismatch")
    if str(target["checkpoint_sha256"]) != str(teacher["checkpoint_sha256"]) or str(target["git_commit"]) != git_commit:
        raise ValueError("existing target checkpoint/Git binding mismatch")
    if not np.array_equal(target["teacher_variant_ids"], np.arange(4)):
        raise ValueError("existing target variant binding mismatch")
    if not np.array_equal(target["teacher_window_masks"], teacher["window_masks"]):
        raise ValueError("existing target teacher masks mismatch")
    if not np.array_equal(target["coverage_masks"], (teacher["coverage_weights"] > 0).astype(np.uint8)):
        raise ValueError("existing target coverage mismatch")


def validate_target_for_stage(target: Mapping[str, object], *, steps: int) -> None:
    optimization_steps = np.asarray(target.get("optimization_steps"))
    initial = np.asarray(target.get("initial_losses"), dtype=np.float64)
    final = np.asarray(target.get("final_losses"), dtype=np.float64)
    if optimization_steps.shape != (4,) or not np.array_equal(optimization_steps, np.full(4, steps)):
        raise ValueError(f"target is not a valid {steps}-step final")
    if initial.shape != (4,) or final.shape != (4,) or not np.isfinite(initial).all() or not np.isfinite(final).all():
        raise ValueError("target losses must be finite four-vectors")
    if not np.all(final < initial):
        raise ValueError("target final losses must be strictly decreasing from their initial losses")


def validate_target_checkpoint_witness(
    target: Mapping[str, object],
    checkpoints: Sequence[Mapping[str, object]],
    *,
    steps: int,
) -> None:
    """Bind every saved target value to the exact optimizer checkpoint witness."""
    validate_target_for_stage(target, steps=steps)
    coefficients = np.asarray(target.get("residual_coefficients"))
    initial = np.asarray(target.get("initial_losses"))
    final = np.asarray(target.get("final_losses"))
    if (
        coefficients.shape != (4, 32, 2048)
        or coefficients.dtype != np.float32
        or not np.isfinite(coefficients).all()
        or initial.shape != (4,)
        or initial.dtype != np.float64
        or final.shape != (4,)
        or final.dtype != np.float64
        or len(checkpoints) != 4
    ):
        raise ValueError("target/checkpoint witness tensor contract mismatch")
    provenance = {
        "source_sha256": str(target.get("source_sha256")),
        "teacher_sha256": str(target.get("teacher_sha256")),
        "basis_sha256": str(target.get("basis_sha256")),
        "camera_head_checkpoint_sha256": str(target.get("checkpoint_sha256")),
        "git_commit": str(target.get("git_commit")),
    }
    for variant, checkpoint in enumerate(checkpoints):
        try:
            best = checkpoint["best_coefficients"]
            loss_trace = checkpoint["loss_trace"]
            initial_loss = float(checkpoint["initial_loss"])
            best_loss = float(checkpoint["best_loss"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("target/checkpoint witness is incomplete") from error
        if (
            not isinstance(best, torch.Tensor)
            or best.device.type != "cpu"
            or best.dtype != torch.float32
            or tuple(best.shape) != (1, 32, 2048)
            or not torch.isfinite(best).all().item()
            or checkpoint.get("variant_index") != variant
            or checkpoint.get("next_step") != steps
            or not isinstance(loss_trace, list)
            or len(loss_trace) != steps
            or not np.isfinite(np.asarray(loss_trace, dtype=np.float64)).all()
            or not np.isfinite((initial_loss, best_loss)).all()
            or best_loss != min(float(value) for value in loss_trace)
            or initial_loss != float(initial[variant])
            or best_loss != float(final[variant])
            or any(checkpoint.get(name) != value for name, value in provenance.items())
            or not np.array_equal(best.numpy()[0], coefficients[variant])
        ):
            raise ValueError("target/checkpoint witness mismatch")


def _load_checkpoint_witnesses(checkpoint_dir: Path) -> list[dict[str, object]]:
    return [
        load_lift_checkpoint(Path(checkpoint_dir) / f"variant_{variant}.pt")
        for variant in range(4)
    ]


def verify_target_redecode(
    camera_head: torch.nn.Module,
    long_tokens: torch.Tensor,
    coefficients: torch.Tensor,
    *,
    expected: np.ndarray | None,
) -> np.ndarray:
    """Decode coefficients through the real Camera Head and reject fabricated summaries."""
    with torch.no_grad():
        decoded = decode_coefficients(camera_head, long_tokens, coefficients).cpu().numpy()[0]
    if expected is not None:
        expected_array = np.asarray(expected)
        if expected_array.shape != (500, 4, 4) or not np.allclose(
            decoded, expected_array, atol=2e-5, rtol=2e-5
        ):
            raise ValueError("saved decoded poses do not match coefficient re-decode")
    return decoded


def select_resume_checkpoint(*, final_target: Path, checkpoint: Path) -> Path | None:
    """A summarized target is never an optimizer state; only a checkpoint may resume."""
    if Path(final_target).exists():
        return None
    candidate = Path(checkpoint)
    return candidate if candidate.is_file() else None


def _run_scene_lift(
    *,
    camera_head: torch.nn.Module,
    device: torch.device,
    long_path: Path,
    teacher_path: Path,
    target_path: Path,
    checkpoint_dir: Path,
    steps: int,
    git_commit: str,
    smoke_split_resume: bool = False,
) -> Path:
    long = load_long_context(long_path)
    teacher = load_teacher_artifact(teacher_path)
    if target_path.exists():
        existing = load_latent_targets(target_path)
        _target_bindings_valid(existing, long, teacher_path, teacher, steps=steps, git_commit=git_commit)
        validate_target_checkpoint_witness(
            existing, _load_checkpoint_witnesses(checkpoint_dir), steps=steps
        )
        return target_path
    tokens = torch.from_numpy(long["camera_tokens"]).unsqueeze(0).to(device=device, dtype=torch.float32)
    oracle = _oracle_from_teacher(teacher)
    results = []
    teacher_digest = sha256_file(teacher_path)
    for variant in range(4):
        checkpoint = checkpoint_dir / f"variant_{variant}.pt"
        teacher_pose = torch.from_numpy(teacher["fused_c2w"][variant]).unsqueeze(0).to(device)
        coverage = torch.from_numpy(teacher["coverage_weights"][variant]).to(device)
        common = dict(
            coverage_weight=coverage, variant_index=variant, checkpoint_path=checkpoint,
            source_sha256=str(long["source_sha256"]), teacher_sha256=teacher_digest,
            basis_sha256=canonical_basis_sha256(),
            camera_head_checkpoint_sha256=str(teacher["checkpoint_sha256"]),
            git_commit=git_commit,
        )
        resume_checkpoint = select_resume_checkpoint(final_target=target_path, checkpoint=checkpoint)
        if smoke_split_resume and resume_checkpoint is None:
            optimize_latent_target(
                camera_head, tokens, teacher_pose, oracle,
                LiftConfig(max_steps=10), resume=False, **common,
            )
        result = optimize_latent_target(
            camera_head, tokens, teacher_pose, oracle,
            LiftConfig(max_steps=steps),
            resume=select_resume_checkpoint(final_target=target_path, checkpoint=checkpoint) is not None,
            **common,
        )
        if not result.finite or result.completed_steps != steps or result.final_loss >= result.initial_loss:
            raise ValueError("latent lift did not produce a finite decreasing final")
        checkpoint_payload = load_lift_checkpoint(checkpoint)
        if checkpoint_payload["next_step"] != steps or len(checkpoint_payload["loss_trace"]) != steps:
            raise ValueError("lift checkpoint is not an exact resumable final")
        results.append(result)
        del teacher_pose, coverage, result
        if device.type == "cuda":
            torch.cuda.empty_cache()
    arrays = {
        "scene": np.asarray(str(long["scene"]), dtype="U32"),
        "frame_ids": long["frame_ids"].astype(np.int64, copy=True),
        "teacher_variant_ids": np.arange(4, dtype=np.int64),
        "teacher_window_masks": teacher["window_masks"].astype(np.uint8, copy=True),
        "coverage_masks": (teacher["coverage_weights"] > 0).astype(np.uint8),
        "residual_coefficients": np.stack([value.coefficients.cpu().numpy()[0] for value in results]),
        "decoded_c2w_raw": np.stack([value.decoded_c2w_raw.cpu().numpy()[0] for value in results]),
        "optimization_steps": np.full(4, steps, dtype=np.int64),
        "initial_losses": np.asarray([value.initial_loss for value in results], dtype=np.float64),
        "final_losses": np.asarray([value.final_loss for value in results], dtype=np.float64),
        "basis_sha256": np.asarray(canonical_basis_sha256(), dtype="U64"),
        "source_sha256": np.asarray(str(long["source_sha256"]), dtype="U64"),
        "teacher_sha256": np.asarray(teacher_digest, dtype="U64"),
        "checkpoint_sha256": np.asarray(str(teacher["checkpoint_sha256"]), dtype="U64"),
        "git_commit": np.asarray(git_commit, dtype="U40"),
    }
    save_latent_targets(target_path, arrays, teacher_artifact=teacher_path)
    validate_target_checkpoint_witness(
        arrays, _load_checkpoint_witnesses(checkpoint_dir), steps=steps
    )
    del arrays, results, tokens, long, teacher
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return target_path


def _write_stage_record(
    path: Path, *, schema: str, git_commit: str, root: Path,
    files: Sequence[Path], metadata: Mapping[str, object],
) -> Path:
    bound = {item.relative_to(root).as_posix(): sha256_file(item) for item in files}
    unsigned = {
        "schema": schema, "git_commit": git_commit, "files": bound,
        "metadata": dict(metadata),
    }
    payload = {**unsigned, "record_digest": _canonical_digest(unsigned)}
    if path.exists() and _read_json(path) != payload:
        raise ValueError("existing stage completion does not match recomputation")
    if not path.exists():
        _atomic_json(path, payload)
    return path


def run_smoke(args: argparse.Namespace | SimpleNamespace) -> Path:
    root = Path(args.run_root).resolve()
    _validate_git(args.git_commit)
    _validate_preflight_live(root, args.git_commit)
    config, _, teacher_manifest = _load_prepared_manifests(root, args.git_commit)
    _require_capacity(root)
    device = torch.device(args.device)
    camera_head, digest = load_base_camera_head(Path(args.checkpoint_dir))
    if digest != config["checkpoint_sha256"]:
        raise ValueError("Camera Head checkpoint changed since prepare")
    camera_head = camera_head.to(device).eval()
    teacher_row = next(row for row in teacher_manifest["records"] if row["scene"] == "scene0000_00")
    target = _run_scene_lift(
        camera_head=camera_head, device=device,
        long_path=root / "prediction_only" / "long_context" / "scene0000_00.npz",
        teacher_path=root / str(teacher_row["file"]),
        target_path=root / "smoke" / "latent_targets" / "scene0000_00.npz",
        checkpoint_dir=root / "smoke" / "checkpoints" / "scene0000_00",
        steps=20, git_commit=args.git_commit, smoke_split_resume=True,
    )
    teacher_path = root / str(teacher_row["file"])
    long_path = root / "prediction_only" / "long_context" / "scene0000_00.npz"
    teacher = load_teacher_artifact(teacher_path)
    teacher["artifact_sha256"] = sha256_file(teacher_path)
    target_arrays = load_latent_targets(target)
    validate_target_checkpoint_witness(
        target_arrays,
        _load_checkpoint_witnesses(root / "smoke" / "checkpoints" / "scene0000_00"),
        steps=20,
    )
    metrics = evaluate_latent_targets(load_long_context(long_path), target_arrays, teacher)
    if not metrics["all_finite"]:
        raise ValueError("smoke metrics are non-finite")
    checkpoints = sorted((root / "smoke" / "checkpoints" / "scene0000_00").glob("variant_*.pt"))
    marker = _write_stage_record(
        root / "smoke" / "completed.json", schema=SMOKE_SCHEMA,
        git_commit=args.git_commit, root=root,
        files=[long_path, teacher_path, target, *checkpoints],
        metadata={"scene": "scene0000_00", "variant_count": 4, "steps": 20, "exact_resume": True},
    )
    _require_run_size(root)
    return marker


def _recompute_scene_metrics(
    root: Path, teacher_manifest: Mapping[str, object]
) -> list[dict[str, object]]:
    by_scene = {str(row["scene"]): row for row in teacher_manifest["records"]}
    metrics: list[dict[str, object]] = []
    for scene in EXPECTED_SCENES:
        teacher_path = root / str(by_scene[scene]["file"])
        teacher = load_teacher_artifact(teacher_path)
        teacher["artifact_sha256"] = sha256_file(teacher_path)
        metrics.append(evaluate_latent_targets(
            load_long_context(root / "prediction_only" / "long_context" / f"{scene}.npz"),
            load_latent_targets(root / "privileged_labels" / "latent_targets" / f"{scene}.npz"),
            teacher,
        ))
    return metrics


def _expected_formal_files() -> set[str]:
    exact = {
        "config.json", "manifests/preflight_evidence.json", "manifests/long_context.json",
        "manifests/teacher.json", "smoke/completed.json", "calibration/completed.json",
        "reports/stage_a.json", "reports/stage_a.md",
    }
    exact.update(f"logs/preflight_{index}.log" for index in range(4))
    for scene in EXPECTED_SCENES:
        exact.add(f"prediction_only/long_context/{scene}.npz")
        exact.add(f"privileged_labels/teacher/{scene}.npz")
        exact.add(f"privileged_labels/latent_targets/{scene}.npz")
        exact.update(
            f"checkpoints/calibration/{scene}/variant_{variant}.pt"
            for variant in range(4)
        )
    exact.add("smoke/latent_targets/scene0000_00.npz")
    exact.update(
        f"smoke/checkpoints/scene0000_00/variant_{variant}.pt"
        for variant in range(4)
    )
    return exact


def is_expected_formal_file(relative: str) -> bool:
    return relative.replace("\\", "/") in _expected_formal_files()


def run_report(args: argparse.Namespace | SimpleNamespace) -> Path:
    root = Path(args.run_root).resolve()
    _validate_git(args.git_commit)
    _validate_preflight_live(root, args.git_commit)
    config, _, teacher_manifest = _load_prepared_manifests(root, args.git_commit)
    calibration_path = root / "calibration" / "completed.json"
    record = _validate_bound_record(
        calibration_path, schema=CALIBRATION_SCHEMA, expected_git_commit=args.git_commit
    )
    if record.get("metadata") != {
        "scenes": list(EXPECTED_SCENES), "variant_count": 40, "steps": 250
    }:
        raise ValueError("calibration completion metadata mismatch")
    metrics = _recompute_scene_metrics(root, teacher_manifest)
    classification = classify_stage_a(
        metrics, expected_scenes=EXPECTED_SCENES, prediction_contract_clean=True
    )
    payload = {
        "schema": REPORT_SCHEMA, "git_commit": args.git_commit,
        "classification": classification["classification"],
        "failed_gates": classification["failed_gates"], "scene_metrics": metrics,
        "provenance": {
            "checkpoint_sha256": config["checkpoint_sha256"],
            "basis_sha256": config["basis_sha256"],
            "long_manifest_sha256": config["long_manifest_sha256"],
            "teacher_manifest_sha256": config["teacher_manifest_sha256"],
        },
    }
    write_stage_a_report(root, payload)
    files: dict[str, str] = {}
    for path in root.rglob("*"):
        if path.is_symlink() or path.name.endswith(".tmp"):
            raise ValueError("run contains forbidden symlink or temporary file")
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative in {"manifests/verification_inventory.json", "verified_completion.json"}:
            continue
        if not is_expected_formal_file(relative):
            raise ValueError("run does not have the exact directory whitelist")
        files[relative] = sha256_file(path)
    if set(files) != _expected_formal_files():
        raise ValueError("run does not contain the exact formal file cohort")
    inventory = {
        "schema": INVENTORY_SCHEMA, "git_commit": args.git_commit,
        "classification": classification["classification"], "files": files,
    }
    inventory_path = root / "manifests" / "verification_inventory.json"
    if inventory_path.exists():
        if _read_json(inventory_path) != inventory:
            raise ValueError("existing verification inventory does not match recomputation")
    else:
        _atomic_json(inventory_path, inventory)
    verify_inventory_exactness(
        root,
        inventory,
        allow_verified_completion=True,
        expected_files=_expected_formal_files(),
    )
    return inventory_path


def _validate_bound_record(path: Path, *, schema: str, expected_git_commit: str) -> dict[str, object]:
    payload = _read_json(path)
    digest = payload.pop("record_digest", None)
    if payload.get("schema") != schema or payload.get("git_commit") != expected_git_commit:
        raise ValueError("stage completion binding mismatch")
    if digest != _canonical_digest(payload):
        raise ValueError("stage completion digest mismatch")
    files = payload.get("files")
    if not isinstance(files, dict):
        raise ValueError("stage completion files are invalid")
    root = path.parent.parent.resolve()
    for relative, expected in files.items():
        candidate = (root / str(relative)).resolve()
        if not _within(candidate, root) or not candidate.is_file() or sha256_file(candidate) != expected:
            raise ValueError("stage completion artifact mismatch")
    payload["record_digest"] = digest
    return payload


def run_calibration(args: argparse.Namespace | SimpleNamespace) -> Path:
    """Enter calibration only through an independently replayed smoke completion."""
    root = Path(args.run_root).resolve()
    marker = root / "smoke" / "completed.json"
    if not marker.is_file():
        raise ValueError("valid smoke completion is required before calibration")
    expected_commit = getattr(args, "git_commit", None)
    if not isinstance(expected_commit, str):
        try:
            expected_commit = str(_read_json(root / "config.json")["git_commit"])
        except (ValueError, KeyError) as error:
            raise ValueError("valid smoke completion requires a bound run config") from error
    smoke_record = _validate_bound_record(marker, schema=SMOKE_SCHEMA, expected_git_commit=expected_commit)
    if smoke_record.get("metadata") != {
        "scene": "scene0000_00", "variant_count": 4, "steps": 20, "exact_resume": True
    }:
        raise ValueError("smoke completion metadata mismatch")
    _validate_git(expected_commit)
    _validate_preflight_live(root, expected_commit)
    executor = getattr(args, "calibration_executor", None)
    if executor is not None:
        result = executor(args)
        if not isinstance(result, Path):
            raise ValueError("calibration executor must return a completion path")
        return result
    if not hasattr(args, "device") or not hasattr(args, "checkpoint_dir"):
        return marker
    config, long_manifest, teacher_manifest = _load_prepared_manifests(root, expected_commit)
    _require_capacity(root)
    device = torch.device(args.device)
    camera_head, digest = load_base_camera_head(Path(args.checkpoint_dir))
    if digest != config["checkpoint_sha256"]:
        raise ValueError("Camera Head checkpoint changed since prepare")
    camera_head = camera_head.to(device).eval()
    teacher_by_scene = {str(row["scene"]): row for row in teacher_manifest["records"]}
    smoke_scene = "scene0000_00"
    smoke_long = load_long_context(root / "prediction_only" / "long_context" / f"{smoke_scene}.npz")
    smoke_teacher_path = root / str(teacher_by_scene[smoke_scene]["file"])
    smoke_teacher = load_teacher_artifact(smoke_teacher_path)
    smoke_target = load_latent_targets(root / "smoke" / "latent_targets" / f"{smoke_scene}.npz")
    _target_bindings_valid(
        smoke_target, smoke_long, smoke_teacher_path, smoke_teacher,
        steps=20, git_commit=expected_commit,
    )
    validate_target_checkpoint_witness(
        smoke_target,
        _load_checkpoint_witnesses(root / "smoke" / "checkpoints" / smoke_scene),
        steps=20,
    )
    if not np.all(smoke_target["final_losses"] < smoke_target["initial_losses"]):
        raise ValueError("smoke completion losses did not decrease")
    smoke_tokens = torch.from_numpy(smoke_long["camera_tokens"]).unsqueeze(0).to(device)
    for variant in range(4):
        checkpoint = load_lift_checkpoint(
            root / "smoke" / "checkpoints" / smoke_scene / f"variant_{variant}.pt"
        )
        if checkpoint["next_step"] != 20 or checkpoint["variant_index"] != variant:
            raise ValueError("smoke completion checkpoint is not an exact 20-step resume witness")
        coefficients = torch.from_numpy(smoke_target["residual_coefficients"][variant]).unsqueeze(0).to(device)
        verify_target_redecode(
            camera_head, smoke_tokens, coefficients,
            expected=smoke_target["decoded_c2w_raw"][variant],
        )
    del smoke_tokens, smoke_long, smoke_teacher, smoke_target
    outputs: list[Path] = []
    checkpoints: list[Path] = []
    for scene in EXPECTED_SCENES:
        target = root / "privileged_labels" / "latent_targets" / f"{scene}.npz"
        checkpoint_dir = root / "checkpoints" / "calibration" / scene
        if scene == "scene0000_00" and not target.exists() and not checkpoint_dir.exists():
            smoke_dir = root / "smoke" / "checkpoints" / scene
            checkpoint_dir.mkdir(parents=True, exist_ok=False)
            for smoke_checkpoint in sorted(smoke_dir.glob("variant_*.pt")):
                shutil.copy2(smoke_checkpoint, checkpoint_dir / smoke_checkpoint.name)
        teacher_path = root / str(teacher_by_scene[scene]["file"])
        output = _run_scene_lift(
            camera_head=camera_head, device=device,
            long_path=root / "prediction_only" / "long_context" / f"{scene}.npz",
            teacher_path=teacher_path, target_path=target, checkpoint_dir=checkpoint_dir,
            steps=250, git_commit=expected_commit,
        )
        outputs.append(output)
        checkpoints.extend(sorted(checkpoint_dir.glob("variant_*.pt")))
        _require_run_size(root)
    return _write_stage_record(
        root / "calibration" / "completed.json", schema=CALIBRATION_SCHEMA,
        git_commit=expected_commit, root=root, files=[*outputs, *checkpoints],
        metadata={"scenes": list(EXPECTED_SCENES), "variant_count": 40, "steps": 250},
    )


def _inventory_files(root: Path, inventory: Mapping[str, object]) -> set[str]:
    files = inventory.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("verification inventory files are invalid")
    allowed: set[str] = set()
    for relative, expected_digest in files.items():
        if not isinstance(relative, str) or not isinstance(expected_digest, str):
            raise ValueError("verification inventory row is invalid")
        candidate = root / relative
        if candidate.is_symlink() or not _within(candidate.resolve(), root):
            raise ValueError("verification inventory path escape or symlink")
        if not candidate.is_file() or sha256_file(candidate) != expected_digest:
            raise ValueError("verification inventory digest mismatch")
        allowed.add(relative.replace("\\", "/"))
    return allowed


def verify_inventory_exactness(
    run_root: Path,
    inventory: Mapping[str, object],
    *,
    allow_verified_completion: bool = False,
    expected_files: set[str] | None = None,
) -> None:
    """Rehash every inventoried byte and reject every unlisted filesystem entry."""
    root = Path(run_root).resolve()
    allowed = _inventory_files(root, inventory)
    if expected_files is not None and allowed != expected_files:
        raise ValueError("verification inventory does not bind the exact formal cohort")
    inventory_path = root / "manifests" / "verification_inventory.json"
    if inventory_path.is_file():
        allowed.add("manifests/verification_inventory.json")
    completion_path = root / "verified_completion.json"
    if allow_verified_completion and completion_path.is_file():
        allowed.add("verified_completion.json")
    actual: set[str] = set()
    actual_directories: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink() or path.name.endswith(".tmp"):
            raise ValueError("run contains forbidden symlink or temporary artifact")
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
        elif path.is_dir():
            actual_directories.add(path.relative_to(root).as_posix())
    if actual != allowed:
        raise ValueError("run does not have the exact directory inventory")
    expected_directories: set[str] = set()
    for relative in allowed:
        parent = Path(relative).parent
        while parent != Path("."):
            expected_directories.add(parent.as_posix())
            parent = parent.parent
    if actual_directories != expected_directories:
        raise ValueError("run does not have the exact directory inventory")
    size = sum((root / relative).stat().st_size for relative in actual)
    if size >= 20 * 1024**3:
        raise ValueError("run root exceeds 20 GiB")


def verify_completed_run(
    run_root: Path,
    *,
    expected_git_commit: str,
    checkpoint_dir: Path | None = None,
    device: torch.device | str = torch.device("cpu"),
) -> Path:
    """Rehash the exact inventory and publish completion only after byte integrity passes."""
    root = Path(run_root).resolve()
    inventory_path = root / "manifests" / "verification_inventory.json"
    inventory = _read_json(inventory_path)
    if set(inventory) != {"schema", "git_commit", "classification", "files"}:
        raise ValueError("verification inventory schema is not exact")
    if inventory.get("schema") != INVENTORY_SCHEMA or inventory.get("git_commit") != expected_git_commit:
        raise ValueError("verification inventory commit binding mismatch")
    if inventory.get("classification") not in {"LATENT_TARGETS_READY", "LATENT_LIFT_FAILED"}:
        raise ValueError("verification inventory classification is invalid")
    config_path = root / "config.json"
    if not config_path.is_file():
        raise ValueError("formal config is required for completed-run verification")
    if config_path.is_file():
        _validate_git(expected_git_commit)
        _validate_preflight_live(root, expected_git_commit)
        config, _, teacher_manifest = _load_prepared_manifests(root, expected_git_commit)
        calibration = _validate_bound_record(
            root / "calibration" / "completed.json",
            schema=CALIBRATION_SCHEMA,
            expected_git_commit=expected_git_commit,
        )
        if calibration.get("metadata") != {
            "scenes": list(EXPECTED_SCENES), "variant_count": 40, "steps": 250
        }:
            raise ValueError("calibration completion metadata mismatch")
        if checkpoint_dir is None:
            raise ValueError("formal verification must reload the Camera Head checkpoint")
        resolved_device = torch.device(device)
        camera_head, checkpoint_sha = load_base_camera_head(Path(checkpoint_dir))
        if checkpoint_sha != config["checkpoint_sha256"]:
            raise ValueError("verification Camera Head checkpoint digest mismatch")
        camera_head = camera_head.to(resolved_device).eval()
        teacher_by_scene = {str(row["scene"]): row for row in teacher_manifest["records"]}
        for scene in EXPECTED_SCENES:
            long = load_long_context(root / "prediction_only" / "long_context" / f"{scene}.npz")
            target_path = root / "privileged_labels" / "latent_targets" / f"{scene}.npz"
            target = load_latent_targets(target_path)
            teacher_path = root / str(teacher_by_scene[scene]["file"])
            teacher = load_teacher_artifact(teacher_path)
            _target_bindings_valid(target, long, teacher_path, teacher, steps=250, git_commit=expected_git_commit)
            validate_target_checkpoint_witness(
                target,
                _load_checkpoint_witnesses(
                    root / "checkpoints" / "calibration" / scene
                ),
                steps=250,
            )
            for variant in range(4):
                checkpoint = load_lift_checkpoint(
                    root / "checkpoints" / "calibration" / scene / f"variant_{variant}.pt"
                )
                expected_checkpoint = {
                    "variant_index": variant, "next_step": 250,
                    "source_sha256": str(long["source_sha256"]),
                    "teacher_sha256": sha256_file(teacher_path),
                    "basis_sha256": canonical_basis_sha256(),
                    "camera_head_checkpoint_sha256": str(teacher["checkpoint_sha256"]),
                    "git_commit": expected_git_commit,
                }
                if any(checkpoint[name] != value for name, value in expected_checkpoint.items()):
                    raise ValueError("calibration lift checkpoint binding mismatch")
            tokens = torch.from_numpy(long["camera_tokens"]).unsqueeze(0).to(resolved_device)
            for variant in range(4):
                coefficients = torch.from_numpy(target["residual_coefficients"][variant]).unsqueeze(0).to(resolved_device)
                verify_target_redecode(
                    camera_head, tokens, coefficients,
                    expected=target["decoded_c2w_raw"][variant],
                )
                del coefficients
            del tokens, long, target, teacher
            gc.collect()
            if resolved_device.type == "cuda":
                torch.cuda.empty_cache()
        metrics = _recompute_scene_metrics(root, teacher_manifest)
        classification = classify_stage_a(
            metrics, expected_scenes=EXPECTED_SCENES, prediction_contract_clean=True
        )
        expected_report = {
            "schema": REPORT_SCHEMA, "git_commit": expected_git_commit,
            "classification": classification["classification"],
            "failed_gates": classification["failed_gates"], "scene_metrics": metrics,
            "provenance": {
                "checkpoint_sha256": config["checkpoint_sha256"],
                "basis_sha256": config["basis_sha256"],
                "long_manifest_sha256": config["long_manifest_sha256"],
                "teacher_manifest_sha256": config["teacher_manifest_sha256"],
            },
        }
        if _read_json(root / "reports" / "stage_a.json") != expected_report:
            raise ValueError("trusted report values do not match independent replay")
        if inventory["classification"] != classification["classification"]:
            raise ValueError("inventory classification does not match independent replay")
    completion_path = root / "verified_completion.json"
    verify_inventory_exactness(
        root,
        inventory,
        allow_verified_completion=True,
        expected_files=_expected_formal_files(),
    )
    unsigned = {
        "schema": VERIFIED_SCHEMA,
        "git_commit": expected_git_commit,
        "classification": inventory["classification"],
        "inventory_sha256": sha256_file(inventory_path),
        "file_count": len(inventory["files"]),
    }
    expected = {**unsigned, "completion_digest": _canonical_digest(unsigned)}
    if completion_path.exists():
        if _read_json(completion_path) != expected:
            raise ValueError("existing verified completion does not match recomputation")
    else:
        _atomic_json(completion_path, expected)
    return completion_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("preflight", "prepare", "smoke", "calibration", "report", "verify"))
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--source-run", type=Path)
    parser.add_argument("--formal-label-root", type=Path)
    parser.add_argument("--prepared-root", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.stage == "preflight":
        run_preflight(args)
    elif args.stage == "prepare":
        required = (args.source_run, args.formal_label_root, args.prepared_root, args.checkpoint_dir)
        if any(value is None for value in required):
            raise ValueError("prepare requires source, formal-label, prepared, and checkpoint roots")
        run_prepare(args)
    elif args.stage == "smoke":
        if args.checkpoint_dir is None:
            raise ValueError("smoke requires checkpoint-dir")
        run_smoke(args)
    elif args.stage == "calibration":
        if args.checkpoint_dir is None:
            raise ValueError("calibration requires checkpoint-dir")
        run_calibration(args)
    elif args.stage == "report":
        run_report(args)
    elif args.stage == "verify":
        verify_completed_run(
            args.run_root, expected_git_commit=args.git_commit,
            checkpoint_dir=args.checkpoint_dir, device=args.device,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
