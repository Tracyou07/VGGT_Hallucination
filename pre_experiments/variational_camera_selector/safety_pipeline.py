from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Mapping, Sequence

import numpy as np

from .dataset import PredictionCandidateDataset
from .evaluate import (
    evaluate_scene_scores,
    load_evaluation_sidecar,
    load_score_shard,
    score_scene_candidates,
)
from .safety_calibration import (
    GateAcceptance,
    GateMetrics,
    FrozenGateResult,
    default_gate_acceptance,
    default_policy_candidates,
    fit_frozen_gate,
    gated_evaluation_from_arrays,
    observation_from_arrays,
    summarize_gate_validation,
)
from .safety_gate import (
    GatePolicy,
    load_gate_policy,
    load_gated_selection,
    write_gate_policy,
    write_gated_scene_selection,
)
from .train import (
    FROZEN_TRAIN_SCENES,
    FROZEN_VALIDATION_SCENES,
    SelectorTrainConfig,
    train_selectors,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE_RUN = Path(
    "/data/yjh/output/variational_camera_selector/selector_A_20260827T153304Z"
)
DEFAULT_OUTPUT_ROOT = Path("/data/yjh/output/variational_camera_selector_safety")
_FOLD_SCHEMA = "variational_camera_selector.safety_oof_fold.v1"
_OOF_SCORE_SCHEMA = "variational_camera_selector.safety_oof_scores.v1"
_OOF_EVALUATION_SCHEMA = "variational_camera_selector.safety_oof_evaluations.v1"
_FIT_SCHEMA = "variational_camera_selector.safety_fit.v1"
_APPLY_SCHEMA = "variational_camera_selector.safety_apply.v1"
_EVALUATE_SCHEMA = "variational_camera_selector.safety_evaluate.v1"
_VERIFIED_SCHEMA = "variational_camera_selector.safety_verified.v1"


def fold_training_scenes(held_scene: str) -> tuple[str, ...]:
    if held_scene not in FROZEN_TRAIN_SCENES:
        raise ValueError("OOF held scene must belong to the frozen train split")
    return tuple(scene for scene in FROZEN_TRAIN_SCENES if scene != held_scene)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: object) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)
    return path


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)
    return path


def _read_json(path: Path, label: str) -> dict[str, object]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {label}: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _load_npz(path: Path, label: str) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as payload:
            return {name: payload[name] for name in payload.files}
    except (OSError, ValueError, KeyError) as error:
        raise ValueError(f"invalid {label}: {path}") from error


def _current_commit() -> str:
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    if status:
        raise ValueError("safety experiment requires a clean tracked worktree")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    if len(commit) != 40:
        raise ValueError("safety experiment git commit is invalid")
    return commit


def _records(
    path: Path, *, schema: str, scenes: Sequence[str], label: str
) -> tuple[dict[str, object], list[dict[str, object]]]:
    payload = _read_json(path, label)
    records = payload.get("records")
    if (
        payload.get("schema") != schema
        or not isinstance(records, list)
        or any(not isinstance(row, dict) for row in records)
        or [str(row.get("scene")) for row in records] != list(scenes)
    ):
        raise ValueError(f"{label} scene order or schema does not match")
    return payload, [dict(row) for row in records]


def _verify_base_artifact_records(
    candidate_records: Sequence[Mapping[str, object]],
    generic_records: Sequence[Mapping[str, object]],
) -> None:
    candidate_fields = (
        ("long_context_path", "long_context_sha256"),
        ("candidate_path", "candidate_sha256"),
        ("residual_prediction_path", "residual_prediction_sha256"),
    )
    for row in candidate_records:
        for path_field, digest_field in candidate_fields:
            path = Path(str(row.get(path_field)))
            digest = row.get(digest_field)
            if not path.is_file() or _sha256_file(path) != digest:
                raise ValueError(
                    f"base artifact digest differs for {path_field}: {path}"
                )
    for row in generic_records:
        path = Path(str(row.get("path")))
        digest = row.get("sha256")
        if not path.is_file() or _sha256_file(path) != digest:
            raise ValueError(f"base artifact digest differs for path: {path}")


def _base_context(
    base_run: Path,
    *,
    require_privileged: bool = False,
    require_evaluation: bool = False,
    privileged_scenes: Sequence[str] | None = None,
) -> dict[str, object]:
    if not require_privileged and privileged_scenes is not None:
        raise ValueError("privileged scenes require privileged base context")
    base_run = Path(base_run).resolve()
    verified = _read_json(base_run / "verified_completion.json", "base completion")
    if (
        verified.get("schema") != "variational_camera_selector.verified_completion.v1"
        or tuple(verified.get("train_scenes", ())) != FROZEN_TRAIN_SCENES
        or tuple(verified.get("validation_scenes", ())) != FROZEN_VALIDATION_SCENES
        or verified.get("completed_step") != 800
    ):
        raise ValueError("base selector completion does not match the frozen experiment")
    candidate_manifest = base_run / "manifests/candidate_binding_manifest.json"
    _, candidate_records = _records(
        candidate_manifest,
        schema="variational_camera_selector.candidate_binding_manifest.v1",
        scenes=FROZEN_TRAIN_SCENES + FROZEN_VALIDATION_SCENES,
        label="base candidate manifest",
    )
    score_manifest = base_run / "manifests/score_manifest.json"
    _, score_records = _records(
        score_manifest,
        schema="variational_camera_selector.score_manifest.v1",
        scenes=FROZEN_VALIDATION_SCENES,
        label="base score manifest",
    )
    _verify_base_artifact_records(candidate_records, score_records)
    context: dict[str, object] = {
        "base_run": base_run,
        "verified": verified,
        "candidate_manifest": candidate_manifest,
        "candidate_records": candidate_records,
        "score_manifest": score_manifest,
        "score_records": score_records,
    }
    if require_privileged:
        selected_privileged_scenes = (
            FROZEN_TRAIN_SCENES
            if privileged_scenes is None
            else tuple(privileged_scenes)
        )
        if (
            not selected_privileged_scenes
            or len(set(selected_privileged_scenes)) != len(selected_privileged_scenes)
            or not set(selected_privileged_scenes).issubset(FROZEN_TRAIN_SCENES)
        ):
            raise ValueError("privileged scenes must be a unique train-scene subset")
        privileged_manifest = base_run / "manifests/privileged_binding_manifest.json"
        _, privileged_records = _records(
            privileged_manifest,
            schema="variational_camera_selector.privileged_binding_manifest.v1",
            scenes=FROZEN_TRAIN_SCENES + FROZEN_VALIDATION_SCENES,
            label="base privileged manifest",
        )
        training_records = [
            row
            for row in privileged_records
            if str(row["scene"]) in selected_privileged_scenes
        ]
        if tuple(str(row["scene"]) for row in training_records) != tuple(
            selected_privileged_scenes
        ):
            raise ValueError("privileged manifest does not match requested scene order")
        _verify_base_artifact_records((), training_records)
        context.update(
            {
                "privileged_manifest": privileged_manifest,
                "privileged_records": privileged_records,
            }
        )
    if require_evaluation:
        evaluation_manifest = base_run / "manifests/evaluation_manifest.json"
        _, evaluation_records = _records(
            evaluation_manifest,
            schema="variational_camera_selector.evaluation_manifest.v1",
            scenes=FROZEN_VALIDATION_SCENES,
            label="base evaluation manifest",
        )
        _verify_base_artifact_records((), evaluation_records)
        context.update(
            {
                "evaluation_manifest": evaluation_manifest,
                "evaluation_records": evaluation_records,
            }
        )
    return context


def _fold_completion(run_root: Path, scene: str) -> Path:
    return Path(run_root) / "oof/folds" / scene / "fold_complete.json"


def _require_fold(run_root: Path, scene: str) -> dict[str, object]:
    path = _fold_completion(run_root, scene)
    payload = _read_json(path, f"OOF fold {scene}")
    score_path = Path(str(payload.get("score_path")))
    checkpoint_path = Path(str(payload.get("checkpoint_path")))
    if (
        payload.get("schema") != _FOLD_SCHEMA
        or payload.get("held_scene") != scene
        or tuple(payload.get("training_scenes", ())) != fold_training_scenes(scene)
        or not score_path.is_file()
        or _sha256_file(score_path) != payload.get("score_sha256")
        or not checkpoint_path.is_file()
        or _sha256_file(checkpoint_path) != payload.get("checkpoint_sha256")
    ):
        raise ValueError(f"OOF fold completion differs for {scene}")
    scores = load_score_shard(score_path)
    if str(scores["scene"]) != scene or str(scores["checkpoint_sha256"]) != payload.get(
        "checkpoint_sha256"
    ):
        raise ValueError(f"OOF score identity differs for {scene}")
    return payload


def run_oof_fold(args: argparse.Namespace) -> Path:
    run_root = Path(args.run_root).resolve()
    held = str(args.fold_scene)
    training_scenes = fold_training_scenes(held)
    context = _base_context(
        Path(args.base_run),
        require_privileged=True,
        privileged_scenes=training_scenes,
    )
    completion = _fold_completion(run_root, held)
    if completion.is_file():
        _require_fold(run_root, held)
        return completion
    commit = _current_commit()
    fold_root = run_root / "oof/folds" / held
    result = train_selectors(
        SelectorTrainConfig(
            prediction_manifest=Path(context["candidate_manifest"]),
            privileged_manifest=Path(context["privileged_manifest"]),
            run_root=fold_root,
            max_steps=int(args.steps),
            batch_size=int(args.batch_size),
            learning_rate=float(args.learning_rate),
            tau=float(args.tau),
            seed=int(args.seed),
            d_model=int(args.d_model),
            device=str(args.device),
            checkpoint_interval=int(args.checkpoint_interval),
            git_commit=commit,
            train_scenes=training_scenes,
        )
    )
    dataset = PredictionCandidateDataset(
        Path(context["candidate_manifest"]),
        roles=("train",),
        scenes=FROZEN_TRAIN_SCENES,
    )
    score_path = fold_root / "prediction_only/scores" / f"{held}.npz"
    if score_path.is_file():
        scores = load_score_shard(score_path)
        if (
            str(scores["scene"]) != held
            or str(scores["checkpoint_sha256"])
            != _sha256_file(result.checkpoint_path)
        ):
            raise ValueError("existing OOF score does not match its resumed checkpoint")
    else:
        score_scene_candidates(
            dataset,
            held,
            result.checkpoint_path,
            score_path,
            device=str(args.device),
        )
    _atomic_json(
        completion,
        {
            "schema": _FOLD_SCHEMA,
            "held_scene": held,
            "training_scenes": list(training_scenes),
            "completed_step": result.completed_step,
            "checkpoint_path": str(result.checkpoint_path),
            "checkpoint_sha256": _sha256_file(result.checkpoint_path),
            "score_path": str(score_path),
            "score_sha256": _sha256_file(score_path),
            "base_completion_sha256": _sha256_file(
                Path(context["base_run"]) / "verified_completion.json"
            ),
            "git_commit": commit,
        },
    )
    _require_fold(run_root, held)
    return completion


def _oof_score_paths(run_root: Path) -> tuple[Path, Path]:
    return (
        Path(run_root) / "manifests/oof_score_manifest.json",
        Path(run_root) / "manifests/oof_score_complete.json",
    )


def run_collect(args: argparse.Namespace) -> Path:
    run_root = Path(args.run_root).resolve()
    context = _base_context(Path(args.base_run))
    records = []
    commits: set[str] = set()
    for scene in FROZEN_TRAIN_SCENES:
        fold = _require_fold(run_root, scene)
        commits.add(str(fold["git_commit"]))
        records.append(
            {
                "scene": scene,
                "path": str(fold["score_path"]),
                "sha256": str(fold["score_sha256"]),
                "checkpoint_sha256": str(fold["checkpoint_sha256"]),
                "fold_completion_sha256": _sha256_file(_fold_completion(run_root, scene)),
            }
        )
    if len(commits) != 1:
        raise ValueError("OOF folds were not produced by one exact git commit")
    manifest_path, completion_path = _oof_score_paths(run_root)
    _atomic_json(
        manifest_path,
        {
            "schema": _OOF_SCORE_SCHEMA,
            "records": records,
            "training_scenes": list(FROZEN_TRAIN_SCENES),
            "base_candidate_manifest_sha256": _sha256_file(
                Path(context["candidate_manifest"])
            ),
            "git_commit": next(iter(commits)),
        },
    )
    _atomic_json(
        completion_path,
        {
            "schema": _OOF_SCORE_SCHEMA,
            "manifest_sha256": _sha256_file(manifest_path),
            "score_count": 8,
        },
    )
    _require_oof_score_barrier(run_root)
    return completion_path


def _require_oof_score_barrier(run_root: Path) -> list[dict[str, object]]:
    manifest_path, completion_path = _oof_score_paths(run_root)
    completion = _read_json(completion_path, "OOF score completion")
    payload, records = _records(
        manifest_path,
        schema=_OOF_SCORE_SCHEMA,
        scenes=FROZEN_TRAIN_SCENES,
        label="OOF score manifest",
    )
    if (
        completion.get("schema") != _OOF_SCORE_SCHEMA
        or completion.get("manifest_sha256") != _sha256_file(manifest_path)
        or completion.get("score_count") != 8
        or tuple(payload.get("training_scenes", ())) != FROZEN_TRAIN_SCENES
    ):
        raise ValueError("OOF score barrier is invalid")
    for row in records:
        path = Path(str(row.get("path")))
        if not path.is_file() or _sha256_file(path) != row.get("sha256"):
            raise ValueError("OOF score artifact digest differs")
        scores = load_score_shard(path)
        if str(scores["scene"]) != row.get("scene"):
            raise ValueError("OOF score scene differs")
    return records


def _oof_evaluation_paths(run_root: Path) -> tuple[Path, Path]:
    return (
        Path(run_root) / "manifests/oof_evaluation_manifest.json",
        Path(run_root) / "manifests/oof_evaluation_complete.json",
    )


def run_privileged(args: argparse.Namespace) -> Path:
    run_root = Path(args.run_root).resolve()
    score_records = _require_oof_score_barrier(run_root)
    context = _base_context(Path(args.base_run), require_privileged=True)
    candidates = {str(row["scene"]): row for row in context["candidate_records"]}
    labels = {str(row["scene"]): row for row in context["privileged_records"]}
    records = []
    for score_record in score_records:
        scene = str(score_record["scene"])
        destination = run_root / "privileged_labels/oof_evaluation" / f"{scene}.npz"
        if not destination.is_file():
            evaluate_scene_scores(
                Path(str(score_record["path"])),
                Path(str(candidates[scene]["residual_prediction_path"])),
                Path(str(labels[scene]["path"])),
                destination,
                random_seed=int(args.seed),
            )
        arrays = load_evaluation_sidecar(destination)
        if str(arrays["scene"]) != scene or str(arrays["score_sha256"]) != str(
            score_record["sha256"]
        ):
            raise ValueError("OOF privileged outcome does not bind its score")
        records.append(
            {
                "scene": scene,
                "path": str(destination),
                "sha256": _sha256_file(destination),
                "score_sha256": str(score_record["sha256"]),
            }
        )
    manifest_path, completion_path = _oof_evaluation_paths(run_root)
    _atomic_json(
        manifest_path,
        {
            "schema": _OOF_EVALUATION_SCHEMA,
            "records": records,
            "oof_score_manifest_sha256": _sha256_file(_oof_score_paths(run_root)[0]),
        },
    )
    _atomic_json(
        completion_path,
        {
            "schema": _OOF_EVALUATION_SCHEMA,
            "manifest_sha256": _sha256_file(manifest_path),
            "evaluation_count": 8,
        },
    )
    _require_oof_evaluation_barrier(run_root)
    return completion_path


def _require_oof_evaluation_barrier(run_root: Path) -> list[dict[str, object]]:
    _require_oof_score_barrier(run_root)
    manifest_path, completion_path = _oof_evaluation_paths(run_root)
    completion = _read_json(completion_path, "OOF evaluation completion")
    payload, records = _records(
        manifest_path,
        schema=_OOF_EVALUATION_SCHEMA,
        scenes=FROZEN_TRAIN_SCENES,
        label="OOF evaluation manifest",
    )
    if (
        completion.get("schema") != _OOF_EVALUATION_SCHEMA
        or completion.get("manifest_sha256") != _sha256_file(manifest_path)
        or completion.get("evaluation_count") != 8
        or payload.get("oof_score_manifest_sha256")
        != _sha256_file(_oof_score_paths(run_root)[0])
    ):
        raise ValueError("OOF evaluation barrier is invalid")
    for row in records:
        path = Path(str(row.get("path")))
        if not path.is_file() or _sha256_file(path) != row.get("sha256"):
            raise ValueError("OOF evaluation artifact digest differs")
        arrays = load_evaluation_sidecar(path)
        if str(arrays["score_sha256"]) != row.get("score_sha256"):
            raise ValueError("OOF evaluation score binding differs")
    return records


def _acceptance() -> GateAcceptance:
    return default_gate_acceptance()


def _fit_result(run_root: Path) -> FrozenGateResult:
    score_records = _require_oof_score_barrier(run_root)
    evaluation_records = _require_oof_evaluation_barrier(run_root)
    scores = {str(row["scene"]): row for row in score_records}
    observations = [
        observation_from_arrays(
            load_score_shard(Path(str(scores[str(row["scene"])]["path"]))),
            load_evaluation_sidecar(Path(str(row["path"]))),
        )
        for row in evaluation_records
    ]
    return fit_frozen_gate(
        observations,
        acceptance=_acceptance(),
        candidates=default_policy_candidates(),
    )


def _fit_report(result: FrozenGateResult) -> dict[str, object]:
    def metrics_payload(metrics: GateMetrics) -> dict[str, object]:
        payload = asdict(metrics)
        payload["per_scene_mean"] = dict(metrics.per_scene_mean)
        return payload

    return {
        "schema": "variational_camera_selector.safety_fit_report.v1",
        "acceptance": asdict(_acceptance()),
        "candidate_policy_count": len(default_policy_candidates()),
        "deployable": result.policy.deployable,
        "policy": asdict(result.policy),
        "calibration_metrics": metrics_payload(result.calibration_metrics),
        "crossfit_metrics": metrics_payload(result.crossfit_metrics),
        "fold_policies": [
            {"scene": scene, "policy": asdict(policy)}
            for scene, policy in result.fold_policies
        ],
    }


def _fit_paths(run_root: Path) -> tuple[Path, Path, Path, Path]:
    return (
        Path(run_root) / "reports/oof_gate_fit.json",
        Path(run_root) / "manifests/gate_fit_manifest.json",
        Path(run_root) / "frozen_policy/policy.json",
        Path(run_root) / "manifests/gate_fit_complete.json",
    )


def run_fit(args: argparse.Namespace) -> Path:
    run_root = Path(args.run_root).resolve()
    result = _fit_result(run_root)
    report_path, manifest_path, policy_path, completion_path = _fit_paths(run_root)
    _atomic_json(report_path, _fit_report(result))
    _atomic_json(
        manifest_path,
        {
            "schema": _FIT_SCHEMA,
            "training_scenes": list(FROZEN_TRAIN_SCENES),
            "oof_score_manifest_sha256": _sha256_file(_oof_score_paths(run_root)[0]),
            "oof_evaluation_manifest_sha256": _sha256_file(
                _oof_evaluation_paths(run_root)[0]
            ),
            "fit_report_path": str(report_path),
            "fit_report_sha256": _sha256_file(report_path),
        },
    )
    write_gate_policy(
        policy_path,
        result.policy,
        training_scenes=FROZEN_TRAIN_SCENES,
        fit_manifest_sha256=_sha256_file(manifest_path),
    )
    _atomic_json(
        completion_path,
        {
            "schema": _FIT_SCHEMA,
            "fit_manifest_sha256": _sha256_file(manifest_path),
            "fit_report_sha256": _sha256_file(report_path),
            "policy_sha256": _sha256_file(policy_path),
            "deployable": result.policy.deployable,
        },
    )
    _require_fit_barrier(run_root)
    return completion_path


def _require_fit_barrier(run_root: Path) -> tuple[GatePolicy, dict[str, object]]:
    report_path, manifest_path, policy_path, completion_path = _fit_paths(run_root)
    manifest = _read_json(manifest_path, "gate fit manifest")
    completion = _read_json(completion_path, "gate fit completion")
    policy = load_gate_policy(policy_path)
    if (
        manifest.get("schema") != _FIT_SCHEMA
        or tuple(manifest.get("training_scenes", ())) != FROZEN_TRAIN_SCENES
        or manifest.get("fit_report_sha256") != _sha256_file(report_path)
        or completion.get("schema") != _FIT_SCHEMA
        or completion.get("fit_manifest_sha256") != _sha256_file(manifest_path)
        or completion.get("fit_report_sha256") != _sha256_file(report_path)
        or completion.get("policy_sha256") != _sha256_file(policy_path)
        or completion.get("deployable") is not policy.deployable
    ):
        raise ValueError("gate fit barrier is invalid")
    return policy, completion


def _apply_paths(run_root: Path) -> tuple[Path, Path]:
    return (
        Path(run_root) / "manifests/gated_selection_manifest.json",
        Path(run_root) / "manifests/gated_selection_complete.json",
    )


def run_apply(args: argparse.Namespace) -> Path:
    run_root = Path(args.run_root).resolve()
    policy, _ = _require_fit_barrier(run_root)
    context = _base_context(Path(args.base_run))
    dataset = PredictionCandidateDataset(
        Path(context["candidate_manifest"]),
        roles=("validation",),
        scenes=FROZEN_VALIDATION_SCENES,
    )
    policy_path = _fit_paths(run_root)[2]
    policy_sha = _sha256_file(policy_path)
    records = []
    for row in context["score_records"]:
        scene = str(row["scene"])
        destination = run_root / "prediction_only/gated_selections" / f"{scene}.npz"
        if not destination.is_file():
            write_gated_scene_selection(
                dataset,
                scene,
                Path(str(row["path"])),
                policy,
                policy_sha,
                destination,
            )
        gated = load_gated_selection(destination)
        if (
            str(gated["scene"]) != scene
            or str(gated["score_sha256"]) != row["sha256"]
            or str(gated["policy_sha256"]) != policy_sha
        ):
            raise ValueError("gated selection does not bind score and policy")
        records.append(
            {"scene": scene, "path": str(destination), "sha256": _sha256_file(destination)}
        )
    manifest_path, completion_path = _apply_paths(run_root)
    _atomic_json(
        manifest_path,
        {
            "schema": _APPLY_SCHEMA,
            "records": records,
            "validation_scenes": list(FROZEN_VALIDATION_SCENES),
            "base_score_manifest_sha256": _sha256_file(Path(context["score_manifest"])),
            "policy_sha256": policy_sha,
        },
    )
    _atomic_json(
        completion_path,
        {
            "schema": _APPLY_SCHEMA,
            "manifest_sha256": _sha256_file(manifest_path),
            "selection_count": 2,
        },
    )
    _require_apply_barrier(run_root)
    return completion_path


def _require_apply_barrier(run_root: Path) -> list[dict[str, object]]:
    policy, _ = _require_fit_barrier(run_root)
    del policy
    manifest_path, completion_path = _apply_paths(run_root)
    completion = _read_json(completion_path, "gated selection completion")
    payload, records = _records(
        manifest_path,
        schema=_APPLY_SCHEMA,
        scenes=FROZEN_VALIDATION_SCENES,
        label="gated selection manifest",
    )
    if (
        completion.get("schema") != _APPLY_SCHEMA
        or completion.get("manifest_sha256") != _sha256_file(manifest_path)
        or completion.get("selection_count") != 2
        or tuple(payload.get("validation_scenes", ())) != FROZEN_VALIDATION_SCENES
        or payload.get("policy_sha256") != _sha256_file(_fit_paths(run_root)[2])
    ):
        raise ValueError("gated selection barrier is invalid")
    for row in records:
        path = Path(str(row["path"]))
        if not path.is_file() or _sha256_file(path) != row["sha256"]:
            raise ValueError("gated selection digest differs")
        load_gated_selection(path)
    return records


def _evaluation_paths(run_root: Path) -> tuple[Path, Path, Path]:
    return (
        Path(run_root) / "manifests/gated_evaluation_manifest.json",
        Path(run_root) / "reports/validation_summary.json",
        Path(run_root) / "manifests/gated_evaluation_complete.json",
    )


def run_evaluate(args: argparse.Namespace) -> Path:
    run_root = Path(args.run_root).resolve()
    gated_records = _require_apply_barrier(run_root)
    context = _base_context(Path(args.base_run), require_evaluation=True)
    raw_records = {str(row["scene"]): row for row in context["evaluation_records"]}
    evaluation_records = []
    evaluated = []
    for gated_record in gated_records:
        scene = str(gated_record["scene"])
        gated = load_gated_selection(Path(str(gated_record["path"])))
        raw = load_evaluation_sidecar(Path(str(raw_records[scene]["path"])))
        arrays = gated_evaluation_from_arrays(gated, raw)
        destination = run_root / "privileged_labels/gated_evaluation" / f"{scene}.npz"
        _atomic_npz(destination, arrays)
        evaluated.append(arrays)
        evaluation_records.append(
            {"scene": scene, "path": str(destination), "sha256": _sha256_file(destination)}
        )
    manifest_path, summary_path, completion_path = _evaluation_paths(run_root)
    _atomic_json(
        manifest_path,
        {
            "schema": _EVALUATE_SCHEMA,
            "records": evaluation_records,
            "gated_selection_manifest_sha256": _sha256_file(_apply_paths(run_root)[0]),
        },
    )
    _atomic_json(summary_path, summarize_gate_validation(evaluated))
    _atomic_json(
        completion_path,
        {
            "schema": _EVALUATE_SCHEMA,
            "manifest_sha256": _sha256_file(manifest_path),
            "summary_sha256": _sha256_file(summary_path),
            "evaluation_count": 2,
        },
    )
    _require_evaluation_barrier(run_root)
    return completion_path


def _require_evaluation_barrier(run_root: Path) -> list[dict[str, object]]:
    _require_apply_barrier(run_root)
    manifest_path, summary_path, completion_path = _evaluation_paths(run_root)
    completion = _read_json(completion_path, "gated evaluation completion")
    payload, records = _records(
        manifest_path,
        schema=_EVALUATE_SCHEMA,
        scenes=FROZEN_VALIDATION_SCENES,
        label="gated evaluation manifest",
    )
    if (
        completion.get("schema") != _EVALUATE_SCHEMA
        or completion.get("manifest_sha256") != _sha256_file(manifest_path)
        or completion.get("summary_sha256") != _sha256_file(summary_path)
        or completion.get("evaluation_count") != 2
        or payload.get("gated_selection_manifest_sha256")
        != _sha256_file(_apply_paths(run_root)[0])
    ):
        raise ValueError("gated evaluation barrier is invalid")
    for row in records:
        path = Path(str(row["path"]))
        if not path.is_file() or _sha256_file(path) != row["sha256"]:
            raise ValueError("gated evaluation digest differs")
        _load_npz(path, "gated evaluation")
    return records


def run_verify(args: argparse.Namespace) -> Path:
    run_root = Path(args.run_root).resolve()
    context = _base_context(Path(args.base_run), require_evaluation=True)
    score_records = _require_oof_score_barrier(run_root)
    evaluation_records = _require_oof_evaluation_barrier(run_root)
    policy, _ = _require_fit_barrier(run_root)
    gated_records = _require_apply_barrier(run_root)
    final_records = _require_evaluation_barrier(run_root)

    recomputed = _fit_result(run_root)
    if recomputed.policy != policy:
        raise ValueError("frozen policy does not recompute from OOF artifacts")
    fit_report = _read_json(_fit_paths(run_root)[0], "OOF gate fit report")
    if fit_report != _fit_report(recomputed):
        raise ValueError("OOF gate fit report does not recompute")
    final_by_scene = {str(row["scene"]): row for row in final_records}
    evaluated = [
        _load_npz(Path(str(final_by_scene[scene]["path"])), "gated evaluation")
        for scene in FROZEN_VALIDATION_SCENES
    ]
    summary = _read_json(_evaluation_paths(run_root)[1], "validation summary")
    if summary != summarize_gate_validation(evaluated):
        raise ValueError("gated validation summary does not recompute")

    expected_sets = {
        run_root / "prediction_only/gated_selections": {
            f"{scene}.npz" for scene in FROZEN_VALIDATION_SCENES
        },
        run_root / "privileged_labels/oof_evaluation": {
            f"{scene}.npz" for scene in FROZEN_TRAIN_SCENES
        },
        run_root / "privileged_labels/gated_evaluation": {
            f"{scene}.npz" for scene in FROZEN_VALIDATION_SCENES
        },
    }
    for directory, expected in expected_sets.items():
        observed = {path.name for path in directory.iterdir() if path.is_file()}
        if observed != expected:
            raise ValueError(f"verified artifact set differs: {directory}")
    verified_path = run_root / "verified_completion.json"
    _atomic_json(
        verified_path,
        {
            "schema": _VERIFIED_SCHEMA,
            "run_id": run_root.name,
            "base_run": str(context["base_run"]),
            "base_completion_sha256": _sha256_file(
                Path(context["base_run"]) / "verified_completion.json"
            ),
            "training_scenes": list(FROZEN_TRAIN_SCENES),
            "validation_scenes": list(FROZEN_VALIDATION_SCENES),
            "oof_score_count": len(score_records),
            "oof_evaluation_count": len(evaluation_records),
            "gated_selection_count": len(gated_records),
            "gated_evaluation_count": len(final_records),
            "policy_deployable": policy.deployable,
            "policy_sha256": _sha256_file(_fit_paths(run_root)[2]),
            "fit_report_sha256": _sha256_file(_fit_paths(run_root)[0]),
            "validation_summary_sha256": _sha256_file(_evaluation_paths(run_root)[1]),
            "validation_classification": summary["classification"],
            "git_commit": _current_commit(),
        },
    )
    return verified_path


def run_stage(args: argparse.Namespace) -> object:
    if args.stage == "oof-fold":
        return run_oof_fold(args)
    if args.stage == "collect":
        return run_collect(args)
    if args.stage == "privileged":
        return run_privileged(args)
    if args.stage == "fit":
        return run_fit(args)
    if args.stage == "apply":
        return run_apply(args)
    if args.stage == "evaluate":
        return run_evaluate(args)
    if args.stage == "verify":
        return run_verify(args)
    if args.stage == "finalize":
        run_collect(args)
        run_privileged(args)
        run_fit(args)
        run_apply(args)
        run_evaluate(args)
        return run_verify(args)
    raise ValueError(f"unknown safety stage: {args.stage}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OOF-calibrated safety gate for the variational Camera selector"
    )
    parser.add_argument(
        "--stage",
        required=True,
        choices=(
            "oof-fold",
            "collect",
            "privileged",
            "fit",
            "apply",
            "evaluate",
            "verify",
            "finalize",
        ),
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--base-run", type=Path, default=DEFAULT_BASE_RUN)
    parser.add_argument("--fold-scene", choices=FROZEN_TRAIN_SCENES)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--tau", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--checkpoint-interval", type=int, default=25)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if args.stage == "oof-fold" and args.fold_scene is None:
        parser.error("--fold-scene is required for --stage oof-fold")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    result = run_stage(parse_args(argv))
    if isinstance(result, Path):
        print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
