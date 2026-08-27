from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Mapping, Sequence

import numpy as np

from pre_experiments.variational_camera_latent.alpha_scan import DEFAULT_ALPHAS

from .contracts import LongContextRecord
from .dataset import (
    CANDIDATE_BINDING_SCHEMA,
    PRIVILEGED_BINDING_SCHEMA,
    PredictionCandidateDataset,
)
from .evaluate import (
    evaluate_scene_scores,
    load_evaluation_sidecar,
    load_selection_shard,
    load_score_shard,
    score_scene_candidates,
    summarize_calibration,
    write_scene_selections,
)
from .schema import build_long_context_shard, write_prediction_binding_manifest
from .train import (
    FROZEN_TRAIN_SCENES,
    FROZEN_VALIDATION_SCENES,
    SelectorTrainConfig,
    train_selectors,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_ROOT = Path(
    "/data/yjh/output/variational_camera_latent/vrfm_camera_20260827T044926Z"
)
DEFAULT_OUTPUT_ROOT = Path("/data/yjh/output/variational_camera_selector")
ALL_SCENES = FROZEN_TRAIN_SCENES + FROZEN_VALIDATION_SCENES
_PREPARE_SCHEMA = "variational_camera_selector.prepare_complete.v1"
_SMOKE_SCHEMA = "variational_camera_selector.smoke_complete.v1"
_CALIBRATION_SCHEMA = "variational_camera_selector.calibration_complete.v1"
_SCORE_MANIFEST_SCHEMA = "variational_camera_selector.score_manifest.v1"
_SCORE_COMPLETE_SCHEMA = "variational_camera_selector.score_complete.v1"
_EVALUATION_MANIFEST_SCHEMA = "variational_camera_selector.evaluation_manifest.v1"
_PRIVILEGED_COMPLETE_SCHEMA = "variational_camera_selector.privileged_complete.v1"
_REPORT_COMPLETE_SCHEMA = "variational_camera_selector.report_complete.v1"
_VERIFIED_SCHEMA = "variational_camera_selector.verified_completion.v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> dict[str, object]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid or missing {label}: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _atomic_json(path: Path, payload: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def write_exact_json(path: Path, payload: Mapping[str, object]) -> dict[str, object]:
    """Create an immutable JSON artifact, or accept an equal payload."""
    path = Path(path)
    normalized = dict(payload)
    if path.exists():
        observed = _read_json(path, "immutable JSON artifact")
        if observed != normalized:
            raise ValueError(f"immutable JSON artifact differs: {path}")
        return observed
    _atomic_json(path, normalized)
    return normalized


def _require_records(
    payload: Mapping[str, object],
    *,
    schema: str,
    scenes: Sequence[str],
    label: str,
) -> list[dict[str, object]]:
    if payload.get("schema") != schema:
        raise ValueError(f"{label} schema does not match")
    records = payload.get("records")
    if not isinstance(records, list) or any(not isinstance(row, dict) for row in records):
        raise ValueError(f"{label} records are invalid")
    if [row.get("scene") for row in records] != list(scenes):
        raise ValueError(f"{label} scene order does not match")
    return records  # type: ignore[return-value]


def _require_exact_validation_scenes(values: object) -> tuple[str, str]:
    if values != list(FROZEN_VALIDATION_SCENES) and values != tuple(
        FROZEN_VALIDATION_SCENES
    ):
        raise ValueError("validation scenes must be exactly the frozen two-scene split")
    return FROZEN_VALIDATION_SCENES


def _require_clean_git_commit(root: Path = ROOT) -> str:
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        text=True,
        capture_output=True,
        check=True,
    ).stdout
    if status.strip():
        raise ValueError("selector pipeline requires a clean Git checkout")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise ValueError("selector pipeline cannot authenticate the current Git commit")
    return commit


def _verify_record_files(records: Sequence[Mapping[str, object]], label: str) -> None:
    for record in records:
        path = Path(str(record.get("path")))
        expected = record.get("sha256")
        if not path.is_file() or path.is_symlink() or _sha256_file(path) != expected:
            raise ValueError(f"{label} file digest does not match: {path}")


def _load_input_context(input_root: Path) -> dict[str, object]:
    input_root = Path(input_root)
    phase1_path = input_root / "verified_completion.json"
    phase1 = _read_json(phase1_path, "Phase 1 completion")
    if phase1.get("schema") != "variational_camera_latent.verified_completion.v1":
        raise ValueError("Phase 1 completion schema does not match")
    for field, relative in (
        ("prediction_manifest_sha256", "manifests/prediction_manifest.json"),
        ("privileged_manifest_sha256", "manifests/privileged_manifest.json"),
        ("report_sha256", "reports/exploration_report.json"),
    ):
        if phase1.get(field) != _sha256_file(input_root / relative):
            raise ValueError(f"Phase 1 completion does not bind {relative}")

    residual_completion_path = (
        input_root / "vrfm_residual_alpha_scan_full_context_verified_completion.json"
    )
    residual_completion = _read_json(
        residual_completion_path, "VRFM residual completion"
    )
    if (
        residual_completion.get("schema")
        != "variational_camera_latent.vrfm_residual_alpha_scan_full_context_verified_completion.v1"
        or residual_completion.get("phase1_completion_sha256") != _sha256_file(phase1_path)
    ):
        raise ValueError("VRFM residual completion does not bind sealed Phase 1")

    paths = {
        "source": input_root / "manifests/source_manifest.json",
        "candidate": input_root / "manifests/calibration_prediction_manifest.json",
        "residual_prediction": input_root
        / "manifests/vrfm_residual_alpha_scan_full_context_prediction_manifest.json",
        "residual_labels": input_root
        / "manifests/vrfm_residual_alpha_scan_full_context_privileged_manifest.json",
    }
    source_manifest = _read_json(paths["source"], "source manifest")
    candidate_manifest = _read_json(paths["candidate"], "candidate manifest")
    prediction_manifest = _read_json(
        paths["residual_prediction"], "residual prediction manifest"
    )
    label_manifest = _read_json(paths["residual_labels"], "residual label manifest")
    source_records = _require_records(
        source_manifest,
        schema="variational_camera_latent.source.v1",
        scenes=ALL_SCENES,
        label="source manifest",
    )
    candidate_records = _require_records(
        candidate_manifest,
        schema="variational_camera_latent.calibration_prediction_manifest.v1",
        scenes=ALL_SCENES,
        label="candidate manifest",
    )
    prediction_records = _require_records(
        prediction_manifest,
        schema=(
            "variational_camera_latent."
            "vrfm_residual_alpha_scan_full_context_prediction_manifest.v1"
        ),
        scenes=ALL_SCENES,
        label="residual prediction manifest",
    )
    label_records = _require_records(
        label_manifest,
        schema=(
            "variational_camera_latent."
            "vrfm_residual_alpha_scan_full_context_privileged_manifest.v1"
        ),
        scenes=ALL_SCENES,
        label="residual label manifest",
    )
    if [record.get("role") for record in source_records] != ["train"] * 8 + [
        "validation"
    ] * 2:
        raise ValueError("sealed source roles do not match the frozen 8/2 split")
    if candidate_manifest.get("samples") != 32:
        raise ValueError("sealed candidate manifest must contain 32 directions")
    if prediction_manifest.get("alphas") != list(DEFAULT_ALPHAS):
        raise ValueError("sealed residual alpha grid does not match")
    for records, label in (
        (source_records, "source"),
        (candidate_records, "candidate"),
        (prediction_records, "residual prediction"),
        (label_records, "residual label"),
    ):
        _verify_record_files(records, label)
    if residual_completion.get("candidate_manifest_sha256") != _sha256_file(
        paths["candidate"]
    ):
        raise ValueError("residual completion does not bind the candidate manifest")
    if residual_completion.get("prediction_manifest_sha256") != _sha256_file(
        paths["residual_prediction"]
    ):
        raise ValueError("residual completion does not bind the prediction manifest")
    if residual_completion.get("privileged_manifest_sha256") != _sha256_file(
        paths["residual_labels"]
    ):
        raise ValueError("residual completion does not bind the label manifest")
    return {
        "input_root": input_root,
        "phase1_completion_sha256": _sha256_file(phase1_path),
        "residual_completion_sha256": _sha256_file(residual_completion_path),
        "source_records": source_records,
        "candidate_records": candidate_records,
        "prediction_records": prediction_records,
        "label_records": label_records,
    }


def _prepare_paths(run_root: Path) -> dict[str, Path]:
    manifests = Path(run_root) / "manifests"
    return {
        "long_manifest": manifests / "long_context_manifest.json",
        "candidate_binding": manifests / "candidate_binding_manifest.json",
        "label_binding": manifests / "privileged_binding_manifest.json",
        "metadata": Path(run_root) / "run_metadata.json",
        "completion": manifests / "prepare_complete.json",
    }


def _require_prepare_barrier(run_root: Path) -> dict[str, object]:
    paths = _prepare_paths(run_root)
    completion = _read_json(paths["completion"], "prepare completion")
    if completion.get("schema") != _PREPARE_SCHEMA:
        raise ValueError("prepare completion schema does not match")
    for field, path in (
        ("long_context_manifest_sha256", paths["long_manifest"]),
        ("candidate_binding_manifest_sha256", paths["candidate_binding"]),
        ("privileged_binding_manifest_sha256", paths["label_binding"]),
        ("run_metadata_sha256", paths["metadata"]),
    ):
        if not path.is_file() or completion.get(field) != _sha256_file(path):
            raise ValueError("prepare barrier artifact digest does not match")
    candidate = _read_json(paths["candidate_binding"], "candidate binding manifest")
    labels = _read_json(paths["label_binding"], "privileged binding manifest")
    _require_records(
        candidate,
        schema=CANDIDATE_BINDING_SCHEMA,
        scenes=ALL_SCENES,
        label="candidate binding manifest",
    )
    _require_records(
        labels,
        schema=PRIVILEGED_BINDING_SCHEMA,
        scenes=ALL_SCENES,
        label="privileged binding manifest",
    )
    return completion


def run_prepare(args: argparse.Namespace) -> Path:
    run_root = Path(args.run_root)
    paths = _prepare_paths(run_root)
    if paths["completion"].is_file():
        _require_prepare_barrier(run_root)
        return run_root
    context = _load_input_context(Path(args.input_root))
    git_commit = _require_clean_git_commit()
    source_records = context["source_records"]
    candidate_records = context["candidate_records"]
    prediction_records = context["prediction_records"]
    label_records = context["label_records"]
    assert isinstance(source_records, list)
    assert isinstance(candidate_records, list)
    assert isinstance(prediction_records, list)
    assert isinstance(label_records, list)
    long_records: list[LongContextRecord] = []
    candidate_bindings: list[dict[str, object]] = []
    label_bindings: list[dict[str, object]] = []
    for source, candidate, prediction, label in zip(
        source_records, candidate_records, prediction_records, label_records
    ):
        scene = str(source["scene"])
        role = str(source["role"])
        long_path = run_root / "prediction_only" / "long_context" / f"{scene}.npz"
        record = build_long_context_shard(
            Path(str(source["path"])),
            Path(str(candidate["path"])),
            long_path,
            role=role,
            producer_git_commit=git_commit,
        )
        long_records.append(record)
        candidate_bindings.append(
            {
                "scene": scene,
                "role": role,
                "long_context_path": str(long_path),
                "long_context_sha256": record.sha256,
                "source_sha256": str(source["sha256"]),
                "candidate_path": str(candidate["path"]),
                "candidate_sha256": str(candidate["sha256"]),
                "residual_prediction_path": str(prediction["path"]),
                "residual_prediction_sha256": str(prediction["sha256"]),
            }
        )
        label_bindings.append(
            {
                "scene": scene,
                "role": role,
                "path": str(label["path"]),
                "sha256": str(label["sha256"]),
                "prediction_sha256": str(prediction["sha256"]),
                "source_sha256": str(source["sha256"]),
                "candidate_sha256": str(candidate["sha256"]),
            }
        )
    write_prediction_binding_manifest(
        paths["long_manifest"],
        records=long_records,
        upstream_run_root=Path(args.input_root),
        upstream_completion_sha256=str(context["phase1_completion_sha256"]),
        producer_git_commit=git_commit,
    )
    write_exact_json(
        paths["candidate_binding"],
        {
            "schema": CANDIDATE_BINDING_SCHEMA,
            "alphas": list(DEFAULT_ALPHAS),
            "records": candidate_bindings,
        },
    )
    write_exact_json(
        paths["label_binding"],
        {"schema": PRIVILEGED_BINDING_SCHEMA, "records": label_bindings},
    )
    write_exact_json(
        paths["metadata"],
        {
            "schema": "variational_camera_selector.run.v1",
            "run_id": run_root.name,
            "input_root": str(Path(args.input_root)),
            "output_root": str(run_root),
            "git_commit": git_commit,
            "train_scenes": list(FROZEN_TRAIN_SCENES),
            "validation_scenes": list(FROZEN_VALIDATION_SCENES),
            "phase1_completion_sha256": context["phase1_completion_sha256"],
            "residual_completion_sha256": context["residual_completion_sha256"],
        },
    )
    write_exact_json(
        paths["completion"],
        {
            "schema": _PREPARE_SCHEMA,
            "scene_count": 10,
            "overlap_count": 80,
            "choice_count_per_overlap": 225,
            "git_commit": git_commit,
            "long_context_manifest_sha256": _sha256_file(paths["long_manifest"]),
            "candidate_binding_manifest_sha256": _sha256_file(
                paths["candidate_binding"]
            ),
            "privileged_binding_manifest_sha256": _sha256_file(paths["label_binding"]),
            "run_metadata_sha256": _sha256_file(paths["metadata"]),
            "phase1_completion_sha256": context["phase1_completion_sha256"],
            "residual_completion_sha256": context["residual_completion_sha256"],
        },
    )
    return run_root


def _metrics_rows(path: Path) -> list[dict[str, object]]:
    try:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
        ]
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid training metrics: {path}") from error
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError("training metrics are empty or malformed")
    return rows


def run_smoke(args: argparse.Namespace) -> Path:
    run_root = Path(args.run_root)
    prepare = _require_prepare_barrier(run_root)
    completion_path = run_root / "manifests" / "smoke_complete.json"
    if completion_path.is_file():
        completion = _read_json(completion_path, "smoke completion")
        if completion.get("schema") != _SMOKE_SCHEMA:
            raise ValueError("smoke completion schema does not match")
        return run_root
    result = train_selectors(
        SelectorTrainConfig(
            prediction_manifest=_prepare_paths(run_root)["candidate_binding"],
            privileged_manifest=_prepare_paths(run_root)["label_binding"],
            run_root=run_root / "smoke",
            max_steps=args.smoke_steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            tau=args.tau,
            seed=args.seed,
            d_model=args.d_model,
            device=args.device,
            checkpoint_interval=args.checkpoint_interval,
            git_commit=str(prepare["git_commit"]),
            train_scenes=(FROZEN_TRAIN_SCENES[0],),
        )
    )
    rows = _metrics_rows(result.metrics_path)
    for field in ("full_context_loss", "residual_only_loss"):
        values = np.asarray([row[field] for row in rows], dtype=np.float64)
        if not np.isfinite(values).all() or float(values.min()) >= float(values[0]):
            raise ValueError(f"smoke did not demonstrate finite trainability for {field}")
    report_path = run_root / "reports" / "smoke_summary.json"
    write_exact_json(
        report_path,
        {
            "schema": "variational_camera_selector.smoke_summary.v1",
            "scene": FROZEN_TRAIN_SCENES[0],
            "completed_step": result.completed_step,
            "full_context_initial_loss": rows[0]["full_context_loss"],
            "full_context_final_loss": rows[-1]["full_context_loss"],
            "residual_only_initial_loss": rows[0]["residual_only_loss"],
            "residual_only_final_loss": rows[-1]["residual_only_loss"],
        },
    )
    write_exact_json(
        completion_path,
        {
            "schema": _SMOKE_SCHEMA,
            "scene": FROZEN_TRAIN_SCENES[0],
            "completed_step": result.completed_step,
            "checkpoint_sha256": _sha256_file(result.checkpoint_path),
            "metrics_sha256": _sha256_file(result.metrics_path),
            "report_sha256": _sha256_file(report_path),
            "candidate_binding_manifest_sha256": prepare[
                "candidate_binding_manifest_sha256"
            ],
        },
    )
    return run_root


def _require_smoke_barrier(run_root: Path) -> dict[str, object]:
    completion = _read_json(
        run_root / "manifests/smoke_complete.json", "smoke completion"
    )
    if completion.get("schema") != _SMOKE_SCHEMA:
        raise ValueError("smoke barrier does not match")
    checkpoint = run_root / "smoke/training/checkpoints/latest.pt"
    if completion.get("checkpoint_sha256") != _sha256_file(checkpoint):
        raise ValueError("smoke checkpoint digest does not match")
    return completion


def run_calibration(args: argparse.Namespace) -> Path:
    run_root = Path(args.run_root)
    prepare = _require_prepare_barrier(run_root)
    _require_smoke_barrier(run_root)
    completion_path = run_root / "manifests/calibration_complete.json"
    if completion_path.is_file():
        completion = _read_json(completion_path, "calibration completion")
        if completion.get("schema") != _CALIBRATION_SCHEMA:
            raise ValueError("calibration completion schema does not match")
        return run_root
    result = train_selectors(
        SelectorTrainConfig(
            prediction_manifest=_prepare_paths(run_root)["candidate_binding"],
            privileged_manifest=_prepare_paths(run_root)["label_binding"],
            run_root=run_root,
            max_steps=args.calibration_steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            tau=args.tau,
            seed=args.seed,
            d_model=args.d_model,
            device=args.device,
            checkpoint_interval=args.checkpoint_interval,
            git_commit=str(prepare["git_commit"]),
            train_scenes=FROZEN_TRAIN_SCENES,
        )
    )
    write_exact_json(
        completion_path,
        {
            "schema": _CALIBRATION_SCHEMA,
            "train_scenes": list(FROZEN_TRAIN_SCENES),
            "completed_step": result.completed_step,
            "checkpoint_sha256": _sha256_file(result.checkpoint_path),
            "metrics_sha256": _sha256_file(result.metrics_path),
            "training_state_sha256": _sha256_file(result.training_state_path),
            "config_digest": result.config_digest,
            "input_digest": result.input_digest,
        },
    )
    return run_root


def _require_calibration_barrier(run_root: Path) -> dict[str, object]:
    completion = _read_json(
        run_root / "manifests/calibration_complete.json", "calibration completion"
    )
    if (
        completion.get("schema") != _CALIBRATION_SCHEMA
        or completion.get("train_scenes") != list(FROZEN_TRAIN_SCENES)
    ):
        raise ValueError("calibration barrier does not match the frozen train split")
    checkpoint = run_root / "training/checkpoints/latest.pt"
    if completion.get("checkpoint_sha256") != _sha256_file(checkpoint):
        raise ValueError("calibration checkpoint digest does not match")
    return completion


def _score_paths(run_root: Path) -> tuple[Path, Path]:
    return (
        run_root / "manifests/score_manifest.json",
        run_root / "manifests/score_complete.json",
    )


def run_score(args: argparse.Namespace) -> Path:
    run_root = Path(args.run_root)
    _require_prepare_barrier(run_root)
    calibration = _require_calibration_barrier(run_root)
    _, completion_path = _score_paths(run_root)
    if completion_path.is_file():
        _require_score_barrier(run_root)
        return run_root
    dataset = PredictionCandidateDataset(
        _prepare_paths(run_root)["candidate_binding"],
        roles=("validation",),
        scenes=FROZEN_VALIDATION_SCENES,
    )
    checkpoint = run_root / "training/checkpoints/latest.pt"
    records: list[dict[str, object]] = []
    for scene in FROZEN_VALIDATION_SCENES:
        destination = run_root / "prediction_only/scores" / f"{scene}.npz"
        score_scene_candidates(dataset, scene, checkpoint, destination, device=args.device)
        selection_destination = run_root / "prediction_only/selections" / f"{scene}.npz"
        write_scene_selections(
            dataset,
            scene,
            destination,
            selection_destination,
        )
        records.append(
            {
                "scene": scene,
                "path": str(destination),
                "sha256": _sha256_file(destination),
                "selection_path": str(selection_destination),
                "selection_sha256": _sha256_file(selection_destination),
            }
        )
    manifest_path, completion_path = _score_paths(run_root)
    write_exact_json(
        manifest_path,
        {
            "schema": _SCORE_MANIFEST_SCHEMA,
            "validation_scenes": list(FROZEN_VALIDATION_SCENES),
            "scene_count": 2,
            "overlap_count": 16,
            "choice_count_per_overlap": 225,
            "checkpoint_sha256": calibration["checkpoint_sha256"],
            "candidate_binding_manifest_sha256": _sha256_file(
                _prepare_paths(run_root)["candidate_binding"]
            ),
            "records": records,
        },
    )
    write_exact_json(
        completion_path,
        {
            "schema": _SCORE_COMPLETE_SCHEMA,
            "validation_scenes": list(FROZEN_VALIDATION_SCENES),
            "score_artifact_count": 2,
            "selection_artifact_count": 2,
            "score_manifest_sha256": _sha256_file(manifest_path),
            "checkpoint_sha256": calibration["checkpoint_sha256"],
        },
    )
    return run_root


def _require_score_barrier(
    run_root: Path,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    manifest_path, completion_path = _score_paths(run_root)
    try:
        completion = _read_json(completion_path, "prediction barrier completion")
    except ValueError as error:
        raise ValueError("prediction barrier is absent or invalid") from error
    if completion.get("schema") != _SCORE_COMPLETE_SCHEMA:
        raise ValueError("prediction barrier schema does not match")
    _require_exact_validation_scenes(completion.get("validation_scenes"))
    if completion.get("score_manifest_sha256") != _sha256_file(manifest_path):
        raise ValueError("prediction barrier does not bind the score manifest")
    manifest = _read_json(manifest_path, "score manifest")
    records = _require_records(
        manifest,
        schema=_SCORE_MANIFEST_SCHEMA,
        scenes=FROZEN_VALIDATION_SCENES,
        label="score manifest",
    )
    _require_exact_validation_scenes(manifest.get("validation_scenes"))
    _verify_record_files(records, "score")
    for record in records:
        score_path = Path(str(record["path"]))
        load_score_shard(score_path)
        selection_path = Path(str(record.get("selection_path")))
        if (
            not selection_path.is_file()
            or selection_path.is_symlink()
            or _sha256_file(selection_path) != record.get("selection_sha256")
        ):
            raise ValueError(f"selection file digest does not match: {selection_path}")
        selection = load_selection_shard(selection_path)
        if str(selection["score_sha256"]) != _sha256_file(score_path):
            raise ValueError("selection artifact does not bind its score artifact")
    return completion, records


def run_privileged(args: argparse.Namespace) -> Path:
    run_root = Path(args.run_root)
    # Do not move any label-bearing import/load above this prediction-only barrier.
    _, score_records = _require_score_barrier(run_root)
    completion_path = run_root / "manifests/privileged_complete.json"
    if completion_path.is_file():
        _require_privileged_barrier(run_root)
        return run_root
    candidate = _read_json(
        _prepare_paths(run_root)["candidate_binding"], "candidate binding manifest"
    )
    labels = _read_json(
        _prepare_paths(run_root)["label_binding"], "privileged binding manifest"
    )
    candidate_records = _require_records(
        candidate,
        schema=CANDIDATE_BINDING_SCHEMA,
        scenes=ALL_SCENES,
        label="candidate binding manifest",
    )
    label_records = _require_records(
        labels,
        schema=PRIVILEGED_BINDING_SCHEMA,
        scenes=ALL_SCENES,
        label="privileged binding manifest",
    )
    candidates_by_scene = {str(row["scene"]): row for row in candidate_records}
    labels_by_scene = {str(row["scene"]): row for row in label_records}
    evaluation_records: list[dict[str, object]] = []
    for score_record in score_records:
        scene = str(score_record["scene"])
        prediction_record = candidates_by_scene[scene]
        label_record = labels_by_scene[scene]
        destination = run_root / "privileged_labels/evaluation" / f"{scene}.npz"
        evaluate_scene_scores(
            Path(str(score_record["path"])),
            Path(str(prediction_record["residual_prediction_path"])),
            Path(str(label_record["path"])),
            destination,
            random_seed=args.seed,
        )
        evaluation_records.append(
            {"scene": scene, "path": str(destination), "sha256": _sha256_file(destination)}
        )
    manifest_path = run_root / "manifests/evaluation_manifest.json"
    write_exact_json(
        manifest_path,
        {
            "schema": _EVALUATION_MANIFEST_SCHEMA,
            "validation_scenes": list(FROZEN_VALIDATION_SCENES),
            "scene_count": 2,
            "overlap_count": 16,
            "random_seed": args.seed,
            "score_manifest_sha256": _sha256_file(_score_paths(run_root)[0]),
            "privileged_binding_manifest_sha256": _sha256_file(
                _prepare_paths(run_root)["label_binding"]
            ),
            "records": evaluation_records,
        },
    )
    write_exact_json(
        completion_path,
        {
            "schema": _PRIVILEGED_COMPLETE_SCHEMA,
            "validation_scenes": list(FROZEN_VALIDATION_SCENES),
            "evaluation_manifest_sha256": _sha256_file(manifest_path),
            "score_completion_sha256": _sha256_file(_score_paths(run_root)[1]),
        },
    )
    return run_root


def _require_privileged_barrier(
    run_root: Path,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    completion = _read_json(
        run_root / "manifests/privileged_complete.json", "privileged completion"
    )
    if completion.get("schema") != _PRIVILEGED_COMPLETE_SCHEMA:
        raise ValueError("privileged completion schema does not match")
    _require_exact_validation_scenes(completion.get("validation_scenes"))
    manifest_path = run_root / "manifests/evaluation_manifest.json"
    if completion.get("evaluation_manifest_sha256") != _sha256_file(manifest_path):
        raise ValueError("privileged completion does not bind evaluation manifest")
    manifest = _read_json(manifest_path, "evaluation manifest")
    records = _require_records(
        manifest,
        schema=_EVALUATION_MANIFEST_SCHEMA,
        scenes=FROZEN_VALIDATION_SCENES,
        label="evaluation manifest",
    )
    _verify_record_files(records, "evaluation")
    for record in records:
        load_evaluation_sidecar(Path(str(record["path"])))
    return completion, records


def run_report(args: argparse.Namespace) -> Path:
    run_root = Path(args.run_root)
    _, records = _require_privileged_barrier(run_root)
    completion_path = run_root / "manifests/report_complete.json"
    if completion_path.is_file():
        completion = _read_json(completion_path, "report completion")
        if completion.get("schema") != _REPORT_COMPLETE_SCHEMA:
            raise ValueError("report completion schema does not match")
        return run_root
    report = summarize_calibration(
        [Path(str(record["path"])) for record in records], random_seed=args.seed
    )
    report_path = run_root / "reports/calibration_summary.json"
    write_exact_json(report_path, report)
    write_exact_json(
        completion_path,
        {
            "schema": _REPORT_COMPLETE_SCHEMA,
            "validation_scenes": list(FROZEN_VALIDATION_SCENES),
            "classification": report["classification"],
            "report_sha256": _sha256_file(report_path),
            "evaluation_manifest_sha256": _sha256_file(
                run_root / "manifests/evaluation_manifest.json"
            ),
        },
    )
    return run_root


def _require_exact_files(
    directory: Path, expected_names: Sequence[str], label: str
) -> None:
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError(f"{label} directory is missing or symlinked")
    observed = sorted(path.name for path in directory.iterdir())
    if observed != sorted(expected_names):
        raise ValueError(f"{label} directory does not contain the exact artifact set")
    if any(path.is_symlink() or not path.is_file() for path in directory.iterdir()):
        raise ValueError(f"{label} artifacts must be regular files")


def verify_completed_run(run_root: Path) -> dict[str, object]:
    run_root = Path(run_root)
    report_completion = _read_json(
        run_root / "manifests/report_complete.json", "report completion"
    )
    _require_exact_validation_scenes(report_completion.get("validation_scenes"))
    if report_completion.get("schema") != _REPORT_COMPLETE_SCHEMA:
        raise ValueError("report completion schema does not match")
    prepare = _require_prepare_barrier(run_root)
    calibration = _require_calibration_barrier(run_root)
    score_completion, score_records = _require_score_barrier(run_root)
    privileged_completion, evaluation_records = _require_privileged_barrier(run_root)
    _require_exact_files(
        run_root / "prediction_only/long_context",
        [f"{scene}.npz" for scene in ALL_SCENES],
        "long-context",
    )
    _require_exact_files(
        run_root / "prediction_only/scores",
        [f"{scene}.npz" for scene in FROZEN_VALIDATION_SCENES],
        "score",
    )
    _require_exact_files(
        run_root / "prediction_only/selections",
        [f"{scene}.npz" for scene in FROZEN_VALIDATION_SCENES],
        "selection",
    )
    _require_exact_files(
        run_root / "privileged_labels/evaluation",
        [f"{scene}.npz" for scene in FROZEN_VALIDATION_SCENES],
        "evaluation",
    )
    report_path = run_root / "reports/calibration_summary.json"
    if report_completion.get("report_sha256") != _sha256_file(report_path):
        raise ValueError("report completion does not bind calibration report")
    report = _read_json(report_path, "calibration report")
    recomputed = summarize_calibration(
        [Path(str(record["path"])) for record in evaluation_records],
        random_seed=int(report.get("random_seed", -1)),
    )
    if report != recomputed:
        raise ValueError("calibration report does not recompute from evaluation sidecars")
    completion = {
        "schema": _VERIFIED_SCHEMA,
        "run_id": run_root.name,
        "git_commit": prepare["git_commit"],
        "train_scenes": list(FROZEN_TRAIN_SCENES),
        "validation_scenes": list(FROZEN_VALIDATION_SCENES),
        "scene_count": 10,
        "training_overlap_count": 64,
        "validation_overlap_count": 16,
        "choice_count_per_overlap": 225,
        "completed_step": calibration["completed_step"],
        "classification": report["classification"],
        "prepare_completion_sha256": _sha256_file(
            run_root / "manifests/prepare_complete.json"
        ),
        "calibration_completion_sha256": _sha256_file(
            run_root / "manifests/calibration_complete.json"
        ),
        "score_completion_sha256": _sha256_file(_score_paths(run_root)[1]),
        "privileged_completion_sha256": _sha256_file(
            run_root / "manifests/privileged_complete.json"
        ),
        "score_manifest_sha256": score_completion["score_manifest_sha256"],
        "evaluation_manifest_sha256": privileged_completion[
            "evaluation_manifest_sha256"
        ],
        "report_sha256": _sha256_file(report_path),
        "checkpoint_sha256": calibration["checkpoint_sha256"],
        "score_artifact_count": len(score_records),
        "selection_artifact_count": len(score_records),
        "evaluation_artifact_count": len(evaluation_records),
    }
    write_exact_json(run_root / "verified_completion.json", completion)
    return completion


def run_stage(args: argparse.Namespace) -> Path | dict[str, object]:
    if args.stage == "prepare":
        return run_prepare(args)
    if args.stage == "smoke":
        return run_smoke(args)
    if args.stage == "calibration":
        return run_calibration(args)
    if args.stage == "score":
        return run_score(args)
    if args.stage == "privileged":
        return run_privileged(args)
    if args.stage == "report":
        return run_report(args)
    if args.stage == "verify":
        return verify_completed_run(args.run_root)
    if args.stage == "auto":
        run_prepare(args)
        run_smoke(args)
        run_calibration(args)
        run_score(args)
        run_privileged(args)
        run_report(args)
        return verify_completed_run(args.run_root)
    raise ValueError(f"unknown selector stage: {args.stage}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Variational Camera latent candidate selector"
    )
    parser.add_argument(
        "--stage",
        required=True,
        choices=(
            "prepare",
            "smoke",
            "calibration",
            "score",
            "privileged",
            "report",
            "verify",
            "auto",
        ),
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke-steps", type=int, default=30)
    parser.add_argument("--calibration-steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--tau", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--checkpoint-interval", type=int, default=25)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    run_stage(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
