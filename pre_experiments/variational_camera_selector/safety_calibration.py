from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Sequence

import numpy as np

from .safety_gate import (
    GateFeatures,
    GatePolicy,
    apply_gate_policy,
    extract_gate_features,
)


@dataclass(frozen=True)
class GateAcceptance:
    min_coverage: float
    min_positive_precision: float
    max_catastrophic_rate: float
    catastrophic_utility: float
    min_mean_utility: float
    min_worst_scene_mean: float


def default_gate_acceptance() -> GateAcceptance:
    return GateAcceptance(
        min_coverage=0.125,
        min_positive_precision=0.70,
        max_catastrophic_rate=0.05,
        catastrophic_utility=-0.05,
        min_mean_utility=0.0,
        min_worst_scene_mean=-0.01,
    )


@dataclass(frozen=True)
class GateSceneObservation:
    scene: str
    features: GateFeatures
    proposed_utility: np.ndarray

    def __post_init__(self) -> None:
        utility = np.asarray(self.proposed_utility)
        if not self.scene or utility.shape != (8,) or not np.isfinite(utility).all():
            raise ValueError("gate scene observations require a scene and finite utility [8]")


@dataclass(frozen=True)
class GateMetrics:
    scene_count: int
    overlap_count: int
    selected_count: int
    positive_count: int
    catastrophic_count: int
    coverage: float
    positive_precision: float
    catastrophic_rate: float
    mean_utility: float
    median_utility: float
    worst_scene_mean: float
    per_scene_mean: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class GateSearchResult:
    policy: GatePolicy
    metrics: GateMetrics


@dataclass(frozen=True)
class FrozenGateResult:
    policy: GatePolicy
    calibration_metrics: GateMetrics
    crossfit_metrics: GateMetrics
    fold_policies: tuple[tuple[str, GatePolicy], ...]


def default_policy_candidates() -> tuple[GatePolicy, ...]:
    return tuple(
        GatePolicy(
            deployable=True,
            max_alpha=max_alpha,
            min_advantage_z=advantage,
            min_prominence_z=prominence,
            min_residual_support_z=residual_support,
            require_top_agreement=agreement,
        )
        for max_alpha, advantage, prominence, residual_support, agreement in product(
            (0.02, 0.05, 0.1, 0.2),
            (0.0, 0.5, 1.0, 1.5, 2.0),
            (0.0, 0.1, 0.25, 0.5),
            (-1.0, 0.0, 0.5, 1.0),
            (False, True),
        )
    )


def observation_from_arrays(
    score: dict[str, np.ndarray], evaluation: dict[str, np.ndarray]
) -> GateSceneObservation:
    try:
        score_scene = str(np.asarray(score["scene"]))
        evaluation_scene = str(np.asarray(evaluation["scene"]))
        score_ids = np.asarray(score["source_sample_ids"])
        evaluation_ids = np.asarray(evaluation["source_sample_ids"])
        score_indices = np.asarray(
            score["full_context_selected_indices"], dtype=np.int64
        )
        evaluation_indices = np.asarray(
            evaluation["full_context_selected_indices"], dtype=np.int64
        )
        proposed_utility = np.asarray(
            evaluation["full_context_utility"], dtype=np.float64
        )
    except KeyError as error:
        raise ValueError("gate observation inputs are incomplete") from error
    if not score_scene or score_scene != evaluation_scene:
        raise ValueError("gate observation scene identities differ")
    if (
        score_ids.shape != (8,)
        or evaluation_ids.shape != (8,)
        or not np.array_equal(score_ids, evaluation_ids)
    ):
        raise ValueError("gate observation sample IDs differ")
    features = extract_gate_features(score)
    if (
        score_indices.shape != (8,)
        or evaluation_indices.shape != (8,)
        or not np.array_equal(score_indices, features.proposed_indices)
        or not np.array_equal(score_indices, evaluation_indices)
    ):
        raise ValueError("gate observation selected indices differ")
    if proposed_utility.shape != (8,) or not np.isfinite(proposed_utility).all():
        raise ValueError("gate observation outcomes must be finite [8]")
    return GateSceneObservation(
        scene=score_scene,
        features=features,
        proposed_utility=proposed_utility,
    )


def gated_evaluation_from_arrays(
    gated: dict[str, np.ndarray], raw_evaluation: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    try:
        gated_scene = str(np.asarray(gated["scene"]))
        raw_scene = str(np.asarray(raw_evaluation["scene"]))
        gated_ids = np.asarray(gated["source_sample_ids"])
        raw_ids = np.asarray(raw_evaluation["source_sample_ids"])
        proposed = np.asarray(gated["proposed_indices"], dtype=np.int64)
        selected = np.asarray(gated["selected_indices"], dtype=np.int64)
        gate_pass = np.asarray(gated["gate_pass"])
        raw_indices = np.asarray(
            raw_evaluation["full_context_selected_indices"], dtype=np.int64
        )
        proposed_utility = np.asarray(
            raw_evaluation["full_context_utility"], dtype=np.float64
        )
        gated_score_sha = str(np.asarray(gated["score_sha256"]))
        raw_score_sha = str(np.asarray(raw_evaluation["score_sha256"]))
        policy_sha = str(np.asarray(gated["policy_sha256"]))
    except KeyError as error:
        raise ValueError("gated evaluation inputs are incomplete") from error
    if not gated_scene or gated_scene != raw_scene:
        raise ValueError("gated evaluation scene identities differ")
    if (
        gated_ids.shape != (8,)
        or raw_ids.shape != (8,)
        or not np.array_equal(gated_ids, raw_ids)
    ):
        raise ValueError("gated evaluation sample IDs differ")
    if (
        proposed.shape != (8,)
        or selected.shape != (8,)
        or raw_indices.shape != (8,)
        or not np.array_equal(proposed, raw_indices)
    ):
        raise ValueError("gated evaluation proposed indices differ")
    if gate_pass.shape != (8,) or gate_pass.dtype != np.bool_:
        raise ValueError("gated evaluation pass mask is invalid")
    expected_selected = np.where(gate_pass, proposed, 0)
    if not np.array_equal(selected, expected_selected):
        raise ValueError("gated evaluation selected indices do not match the gate")
    if (
        proposed_utility.shape != (8,)
        or not np.isfinite(proposed_utility).all()
        or len(gated_score_sha) != 64
        or gated_score_sha != raw_score_sha
        or len(policy_sha) != 64
    ):
        raise ValueError("gated evaluation score or outcome binding is invalid")
    return {
        "scene": np.asarray(gated_scene, dtype="U64"),
        "source_sample_ids": gated_ids.astype("U96"),
        "proposed_indices": proposed,
        "selected_indices": selected,
        "gate_pass": gate_pass.astype(np.bool_),
        "proposed_utility": proposed_utility,
        "gated_utility": np.where(gate_pass, proposed_utility, 0.0).astype(
            np.float64
        ),
        "score_sha256": np.asarray(gated_score_sha, dtype="U64"),
        "policy_sha256": np.asarray(policy_sha, dtype="U64"),
    }


def summarize_gate_validation(
    records: Sequence[dict[str, np.ndarray]],
    *,
    acceptance: GateAcceptance | None = None,
) -> dict[str, object]:
    acceptance = acceptance or default_gate_acceptance()
    if not records:
        raise ValueError("gated validation requires at least one scene")
    scenes: list[str] = []
    selected: list[np.ndarray] = []
    passed: list[np.ndarray] = []
    for record in records:
        try:
            scene = str(np.asarray(record["scene"]))
            gate_pass = np.asarray(record["gate_pass"])
            utility = np.asarray(record["gated_utility"], dtype=np.float64)
        except KeyError as error:
            raise ValueError("gated validation record is incomplete") from error
        if (
            not scene
            or scene in scenes
            or gate_pass.shape != (8,)
            or gate_pass.dtype != np.bool_
            or utility.shape != (8,)
            or not np.isfinite(utility).all()
            or np.any(utility[~gate_pass] != 0.0)
        ):
            raise ValueError("gated validation record is invalid")
        scenes.append(scene)
        selected.append(utility)
        passed.append(gate_pass)
    metrics = _metrics_from_selected(
        scenes,
        selected,
        passed,
        catastrophic_utility=acceptance.catastrophic_utility,
    )
    if metrics.selected_count == 0:
        classification = "SAFE_NOOP"
    elif _accepted(metrics, acceptance):
        classification = "SAFE_IMPROVEMENT"
    else:
        classification = "UNSAFE_GENERALIZATION"
    return {
        "schema": "variational_camera_selector.safety_validation_summary.v1",
        "classification": classification,
        "scene_count": metrics.scene_count,
        "overlap_count": metrics.overlap_count,
        "selected_count": metrics.selected_count,
        "positive_count": metrics.positive_count,
        "catastrophic_count": metrics.catastrophic_count,
        "coverage": metrics.coverage,
        "positive_precision": metrics.positive_precision,
        "catastrophic_rate": metrics.catastrophic_rate,
        "mean_utility": metrics.mean_utility,
        "median_utility": metrics.median_utility,
        "worst_scene_mean": metrics.worst_scene_mean,
        "per_scene_mean": dict(metrics.per_scene_mean),
    }


def _metrics_from_selected(
    scenes: Sequence[str], selected: Sequence[np.ndarray], passed: Sequence[np.ndarray],
    *, catastrophic_utility: float = -0.05,
) -> GateMetrics:
    if not scenes or len(scenes) != len(selected) or len(scenes) != len(passed):
        raise ValueError("gate metrics require matching non-empty scene vectors")
    per_scene = tuple(
        (scene, float(np.asarray(values, dtype=np.float64).mean()))
        for scene, values in zip(scenes, selected)
    )
    utility = np.concatenate([np.asarray(value, dtype=np.float64) for value in selected])
    mask = np.concatenate([np.asarray(value, dtype=np.bool_) for value in passed])
    if utility.shape != (8 * len(scenes),) or mask.shape != utility.shape:
        raise ValueError("gate metric vectors must contain eight overlaps per scene")
    executed = utility[mask]
    positive_count = int(np.count_nonzero(executed > 0.0))
    catastrophic_count = int(np.count_nonzero(executed < catastrophic_utility))
    selected_count = int(np.count_nonzero(mask))
    return GateMetrics(
        scene_count=len(scenes),
        overlap_count=int(len(utility)),
        selected_count=selected_count,
        positive_count=positive_count,
        catastrophic_count=catastrophic_count,
        coverage=float(selected_count / len(utility)),
        positive_precision=float(positive_count / selected_count) if selected_count else 0.0,
        catastrophic_rate=(
            float(catastrophic_count / selected_count) if selected_count else 0.0
        ),
        mean_utility=float(utility.mean()),
        median_utility=float(np.median(utility)),
        worst_scene_mean=min(value for _, value in per_scene),
        per_scene_mean=per_scene,
    )


def evaluate_gate_policy(
    observations: Sequence[GateSceneObservation],
    policy: GatePolicy,
    *,
    catastrophic_utility: float = -0.05,
) -> GateMetrics:
    if not observations:
        raise ValueError("gate evaluation requires at least one scene")
    selected: list[np.ndarray] = []
    passed: list[np.ndarray] = []
    for observation in observations:
        decision = apply_gate_policy(observation.features, policy)
        selected.append(
            np.where(decision.gate_pass, observation.proposed_utility, 0.0).astype(
                np.float64
            )
        )
        passed.append(decision.gate_pass)
    return _metrics_from_selected(
        [observation.scene for observation in observations],
        selected,
        passed,
        catastrophic_utility=catastrophic_utility,
    )


def _accepted(metrics: GateMetrics, acceptance: GateAcceptance) -> bool:
    return (
        metrics.coverage >= acceptance.min_coverage
        and metrics.positive_precision >= acceptance.min_positive_precision
        and metrics.catastrophic_rate <= acceptance.max_catastrophic_rate
        and metrics.mean_utility > acceptance.min_mean_utility
        and metrics.worst_scene_mean >= acceptance.min_worst_scene_mean
    )


def _policy_key(result: GateSearchResult) -> tuple[float, ...]:
    metrics = result.metrics
    policy = result.policy
    return (
        metrics.worst_scene_mean,
        metrics.mean_utility,
        metrics.positive_precision,
        -metrics.catastrophic_rate,
        metrics.coverage,
        -policy.max_alpha,
        policy.min_advantage_z,
        policy.min_prominence_z,
        policy.min_residual_support_z,
        float(policy.require_top_agreement),
    )


def search_gate_policy(
    observations: Sequence[GateSceneObservation],
    *,
    acceptance: GateAcceptance,
    candidates: Sequence[GatePolicy],
) -> GateSearchResult:
    if not observations or not candidates:
        raise ValueError("gate search requires observations and candidate policies")
    feasible: list[GateSearchResult] = []
    for policy in candidates:
        if not policy.deployable:
            continue
        metrics = evaluate_gate_policy(
            observations,
            policy,
            catastrophic_utility=acceptance.catastrophic_utility,
        )
        if _accepted(metrics, acceptance):
            feasible.append(GateSearchResult(policy=policy, metrics=metrics))
    if feasible:
        return max(feasible, key=_policy_key)
    fallback = GatePolicy.fail_closed()
    return GateSearchResult(
        policy=fallback,
        metrics=evaluate_gate_policy(
            observations,
            fallback,
            catastrophic_utility=acceptance.catastrophic_utility,
        ),
    )


def fit_frozen_gate(
    observations: Sequence[GateSceneObservation],
    *,
    acceptance: GateAcceptance,
    candidates: Sequence[GatePolicy],
) -> FrozenGateResult:
    if len(observations) < 2 or len({row.scene for row in observations}) != len(
        observations
    ):
        raise ValueError("crossfit gate calibration requires unique scenes")
    fold_policies: list[tuple[str, GatePolicy]] = []
    selected: list[np.ndarray] = []
    passed: list[np.ndarray] = []
    for held in observations:
        training = [row for row in observations if row.scene != held.scene]
        fold = search_gate_policy(
            training, acceptance=acceptance, candidates=candidates
        )
        decision = apply_gate_policy(held.features, fold.policy)
        fold_policies.append((held.scene, fold.policy))
        selected.append(
            np.where(decision.gate_pass, held.proposed_utility, 0.0).astype(np.float64)
        )
        passed.append(decision.gate_pass)
    crossfit = _metrics_from_selected(
        [row.scene for row in observations],
        selected,
        passed,
        catastrophic_utility=acceptance.catastrophic_utility,
    )
    fitted = search_gate_policy(
        observations, acceptance=acceptance, candidates=candidates
    )
    if not _accepted(crossfit, acceptance) or not fitted.policy.deployable:
        fallback = GatePolicy.fail_closed()
        return FrozenGateResult(
            policy=fallback,
            calibration_metrics=evaluate_gate_policy(
                observations,
                fallback,
                catastrophic_utility=acceptance.catastrophic_utility,
            ),
            crossfit_metrics=crossfit,
            fold_policies=tuple(fold_policies),
        )
    return FrozenGateResult(
        policy=fitted.policy,
        calibration_metrics=fitted.metrics,
        crossfit_metrics=crossfit,
        fold_policies=tuple(fold_policies),
    )
