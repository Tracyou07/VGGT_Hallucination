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
from types import SimpleNamespace
from typing import Mapping, Sequence

import numpy as np
import torch

from pre_experiments.camera_velocity_ambiguity_02.frozen_oracle import FrozenOracle
from pre_experiments.conditional_hierarchical_vrfm.artifacts import (
    load_latent_targets,
    load_teacher_artifact,
    save_latent_targets,
    save_teacher_artifact,
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
    load_long_context,
    load_source_records,
    publish_long_context,
)
from pre_experiments.long_short_camera_head.labels import load_privileged_labels
from pre_experiments.long_short_camera_head.train import load_base_camera_head


LONG_MANIFEST_SCHEMA = "conditional_hierarchical_vrfm.long_context_manifest.v1"
INVENTORY_SCHEMA = "conditional_hierarchical_vrfm.verification_inventory.v1"
VERIFIED_SCHEMA = "conditional_hierarchical_vrfm.verified_completion.v1"
SMOKE_SCHEMA = "conditional_hierarchical_vrfm.smoke_completion.v1"
CALIBRATION_SCHEMA = "conditional_hierarchical_vrfm.calibration_completion.v1"
PREFLIGHT_SCHEMA = "conditional_hierarchical_vrfm.preflight_evidence.v1"
TEACHER_MANIFEST_SCHEMA = "conditional_hierarchical_vrfm.teacher_manifest.v1"
EXPECTED_SCENES = (
    "scene0000_00", "scene0013_00", "scene0029_00", "scene0084_00",
    "scene0121_00", "scene0207_00", "scene0280_00", "scene0325_00",
    "scene0675_00", "scene0691_00",
)
EXPECTED_TEACHER_COVERAGE = 0.89
EXPECTED_TEACHER_UTILITY = 0.1293578271441714


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


def audit_long_context_manifest(run_root: Path, manifest: Mapping[str, object]) -> None:
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


def _locate_formal_label(root: Path, scene: str) -> Path:
    base = Path(root).resolve()
    fixed = (
        base / "data" / "privileged_labels" / f"{scene}.npz",
        base / "privileged_labels" / f"{scene}.npz",
    )
    candidates = [path for path in fixed if path.is_file() and not path.is_symlink()]
    if len(candidates) != 1:
        raise ValueError(f"formal label must resolve uniquely for {scene}")
    return candidates[0]


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


def run_preflight(args: argparse.Namespace | SimpleNamespace) -> Path:
    """Execute every CPU contract suite plus compileall and bind their actual logs."""
    _validate_git(args.git_commit)
    root = Path(args.run_root).resolve()
    evidence_path = root / "manifests" / "preflight_evidence.json"
    suites = (
        "tests/conditional_hierarchical_vrfm",
        "tests/variational_camera_latent",
        "tests/long_short_camera_head",
    )
    commands = [
        [sys.executable, "-m", "unittest", "discover", "-s", suite, "-v"]
        for suite in suites
    ] + [[sys.executable, "-m", "compileall", "-q", "pre_experiments"]]
    rows: list[dict[str, object]] = []
    for index, command in enumerate(commands):
        completed = subprocess.run(
            command, text=True, capture_output=True, check=False, timeout=1800,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        )
        log = root / "logs" / f"preflight_{index}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(completed.stdout + completed.stderr, encoding="utf-8")
        count = 0
        if index < len(suites):
            match = re.search(r"Ran\s+(\d+)\s+tests?", completed.stdout + completed.stderr)
            count = int(match.group(1)) if match else 0
            if count < 1:
                raise ValueError("preflight test evidence contains no executed tests")
        if completed.returncode != 0:
            raise ValueError(f"preflight command failed: {' '.join(command)}")
        rows.append({
            "command": command, "returncode": completed.returncode, "test_count": count,
            "log": log.relative_to(root).as_posix(), "log_sha256": sha256_file(log),
        })
    unsigned = {"schema": PREFLIGHT_SCHEMA, "git_commit": args.git_commit, "commands": rows}
    payload = {**unsigned, "record_digest": _canonical_digest(unsigned)}
    if evidence_path.exists() and _read_json(evidence_path) != payload:
        raise ValueError("existing preflight evidence is stale or fabricated")
    if not evidence_path.exists():
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
    rows = payload.get("commands")
    if not isinstance(rows, list) or len(rows) != 4:
        raise ValueError("preflight evidence command count mismatch")
    expected_commands = [
        [sys.executable, "-m", "unittest", "discover", "-s", suite, "-v"]
        for suite in (
            "tests/conditional_hierarchical_vrfm", "tests/variational_camera_latent",
            "tests/long_short_camera_head",
        )
    ] + [[sys.executable, "-m", "compileall", "-q", "pre_experiments"]]
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or row.get("returncode") != 0:
            raise ValueError("preflight evidence command failure")
        if row.get("command") != expected_commands[index]:
            raise ValueError("preflight evidence command mismatch")
        if index < 3 and (not isinstance(row.get("test_count"), int) or int(row["test_count"]) < 1):
            raise ValueError("preflight evidence test count mismatch")
        log = root / str(row.get("log", ""))
        if not log.is_file() or sha256_file(log) != row.get("log_sha256"):
            raise ValueError("preflight evidence log digest mismatch")
        if index < 3:
            content = log.read_text(encoding="utf-8")
            if not re.search(r"Ran\s+\d+\s+tests?", content) or not re.search(r"^OK", content, re.MULTILINE):
                raise ValueError("preflight evidence log does not prove a passing test run")
    payload["record_digest"] = digest
    return payload


def validate_preflight_evidence(root: Path, git_commit: str) -> dict[str, object]:
    """Public verifier used by every later stage and barrier regressions."""
    return _validate_preflight(Path(root).resolve(), git_commit)


def run_prepare(args: argparse.Namespace | SimpleNamespace) -> Path:
    """Publish physically separated long-only shards and strict teacher sidecars."""
    _validate_git(args.git_commit)
    root = Path(args.run_root).resolve()
    _validate_preflight(root, args.git_commit)
    existing = (
        root / "config.json", root / "manifests" / "long_context.json",
        root / "manifests" / "teacher.json",
    )
    if any(path.exists() for path in existing):
        if not all(path.is_file() for path in existing):
            raise ValueError("prepare found an incomplete existing publication")
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
    if tuple(sorted(record.scene for record in records)) != tuple(sorted(EXPECTED_SCENES)):
        raise ValueError("source manifest does not bind the exact ten calibration scenes")
    if sum(record.role == "train" for record in records) != 8 or sum(record.role == "validation" for record in records) != 2:
        raise ValueError("source scene roles must be exactly eight train and two validation")
    camera_head, checkpoint_sha256 = load_base_camera_head(Path(args.checkpoint_dir))
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
        if long_path.exists():
            raise ValueError("prepare refuses to overwrite an existing long-context shard")
        published = publish_long_context(record, long_path)
        manifest_rows[record.scene]["sha256"] = published.sha256
        formal_path = _locate_formal_label(Path(args.formal_label_root), record.scene)
        formal = load_privileged_labels(formal_path)
        teacher = build_teacher_variants(
            record.path, Path(args.prepared_root) / record.scene, camera_head,
            checkpoint_sha256=checkpoint_sha256, device=device,
        )
        validate_variant_zero_against_formal(teacher, formal)
        long_context = load_long_context(long_path)
        teacher_path = root / "privileged_labels" / "teacher" / f"{record.scene}.npz"
        arrays = _teacher_arrays(
            teacher, formal, long_context["baseline_c2w"], source_sha256=record.sha256,
            formal_label_sha256=sha256_file(formal_path), git_commit=args.git_commit,
        )
        teacher_sha = save_teacher_artifact(teacher_path, arrays)
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
    _atomic_json(long_manifest_path, manifest)
    teacher_manifest = {
        "schema": TEACHER_MANIFEST_SCHEMA, "git_commit": args.git_commit,
        "checkpoint_sha256": checkpoint_sha256, "teacher_upper_bound": summary,
        "records": teacher_rows,
    }
    _atomic_json(teacher_manifest_path, teacher_manifest)
    config = {
        "schema": "conditional_hierarchical_vrfm.run_config.v1",
        "git_commit": args.git_commit,
        "checkpoint_sha256": checkpoint_sha256,
        "basis_sha256": canonical_basis_sha256(),
        "long_manifest_sha256": sha256_file(long_manifest_path),
        "teacher_manifest_sha256": sha256_file(teacher_manifest_path),
        "smoke_scene": "scene0000_00", "smoke_steps": 20, "calibration_steps": 250,
        "scene_count": 10, "variant_count": 4,
    }
    config_path = root / "config.json"
    if config_path.exists() and _read_json(config_path) != config:
        raise ValueError("existing immutable run config does not match")
    if not config_path.exists():
        _atomic_json(config_path, config)
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
    rows = teacher_manifest.get("records")
    if not isinstance(rows, list) or len(rows) != 10:
        raise ValueError("teacher manifest must bind ten scenes")
    for row in rows:
        path = root / str(row["file"])
        if path.is_symlink() or not _within(path.resolve(), root) or sha256_file(path) != row["sha256"]:
            raise ValueError("teacher manifest artifact mismatch")
        load_teacher_artifact(path)
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
    _validate_preflight(root, args.git_commit)
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
    metrics = evaluate_latent_targets(load_long_context(long_path), load_latent_targets(target), teacher)
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


def _allowed_formal_file(relative: str) -> bool:
    exact = {
        "config.json", "manifests/preflight_evidence.json", "manifests/long_context.json",
        "manifests/teacher.json", "smoke/completed.json", "calibration/completed.json",
        "reports/stage_a.json", "reports/stage_a.md",
    }
    if relative in exact:
        return True
    patterns = (
        r"logs/preflight_[0-3]\.log",
        r"prediction_only/long_context/scene\d{4}_\d{2}\.npz",
        r"privileged_labels/teacher/scene\d{4}_\d{2}\.npz",
        r"privileged_labels/latent_targets/scene\d{4}_\d{2}\.npz",
        r"smoke/latent_targets/scene0000_00\.npz",
        r"smoke/checkpoints/scene0000_00/variant_[0-3]\.pt",
        r"checkpoints/calibration/scene\d{4}_\d{2}/variant_[0-3]\.pt",
    )
    return any(re.fullmatch(pattern, relative) for pattern in patterns)


def run_report(args: argparse.Namespace | SimpleNamespace) -> Path:
    root = Path(args.run_root).resolve()
    _validate_git(args.git_commit)
    _validate_preflight(root, args.git_commit)
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
        if not _allowed_formal_file(relative):
            raise ValueError("run does not have the exact directory whitelist")
        files[relative] = sha256_file(path)
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
    executor = getattr(args, "calibration_executor", None)
    if executor is not None:
        result = executor(args)
        if not isinstance(result, Path):
            raise ValueError("calibration executor must return a completion path")
        return result
    if not hasattr(args, "device") or not hasattr(args, "checkpoint_dir"):
        return marker
    _validate_git(expected_commit)
    _validate_preflight(root, expected_commit)
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
) -> None:
    """Rehash every inventoried byte and reject every unlisted filesystem entry."""
    root = Path(run_root).resolve()
    allowed = _inventory_files(root, inventory)
    inventory_path = root / "manifests" / "verification_inventory.json"
    if inventory_path.is_file():
        allowed.add("manifests/verification_inventory.json")
    completion_path = root / "verified_completion.json"
    if allow_verified_completion and completion_path.is_file():
        allowed.add("verified_completion.json")
    actual: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink() or path.name.endswith(".tmp"):
            raise ValueError("run contains forbidden symlink or temporary artifact")
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
    if actual != allowed:
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
        _validate_preflight(root, expected_git_commit)
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
    verify_inventory_exactness(root, inventory, allow_verified_completion=True)
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
