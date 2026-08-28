from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from .dataset import PredictionCandidateDataset


GATED_SELECTION_SCHEMA = "variational_camera_selector.gated_selection.v1"
GATE_POLICY_SCHEMA = "variational_camera_selector.safety_policy.v1"
MAX_GATE_ALPHA = 0.2
_FORBIDDEN_PARTS = (
    "gt",
    "ground_truth",
    "privileged",
    "depth",
    "quality",
    "error",
    "utility",
)
_GATED_SELECTION_MEMBERS = {
    "scene",
    "role",
    "source_sample_ids",
    "span_starts",
    "proposed_indices",
    "selected_indices",
    "gate_pass",
    "proposed_choice_ids",
    "selected_choice_ids",
    "proposed_alphas",
    "selected_alphas",
    "proposed_z",
    "selected_z",
    "advantage_z",
    "prominence_z",
    "residual_support_z",
    "top_agreement",
    "selected_scores",
    "corrected_camera_tokens",
    "source_sha256",
    "candidate_sha256",
    "residual_prediction_sha256",
    "binding_manifest_sha256",
    "checkpoint_sha256",
    "score_sha256",
    "policy_sha256",
}


@dataclass(frozen=True)
class GatePolicy:
    deployable: bool
    max_alpha: float
    min_advantage_z: float
    min_prominence_z: float
    min_residual_support_z: float
    require_top_agreement: bool

    @classmethod
    def fail_closed(cls) -> "GatePolicy":
        return cls(
            deployable=False,
            max_alpha=0.0,
            min_advantage_z=0.0,
            min_prominence_z=0.0,
            min_residual_support_z=0.0,
            require_top_agreement=True,
        )


@dataclass(frozen=True)
class GateFeatures:
    proposed_indices: np.ndarray
    proposed_alphas: np.ndarray
    advantage_z: np.ndarray
    prominence_z: np.ndarray
    residual_support_z: np.ndarray
    top_agreement: np.ndarray

    def __post_init__(self) -> None:
        arrays = (
            self.proposed_indices,
            self.proposed_alphas,
            self.advantage_z,
            self.prominence_z,
            self.residual_support_z,
            self.top_agreement,
        )
        lengths = {np.asarray(value).shape for value in arrays}
        if len(lengths) != 1 or not lengths or next(iter(lengths)) != (8,):
            raise ValueError("gate features must be matching vectors with shape [8]")
        if not np.issubdtype(np.asarray(self.proposed_indices).dtype, np.integer):
            raise ValueError("gate proposed indices must be integers")
        for value in (
            self.proposed_alphas,
            self.advantage_z,
            self.prominence_z,
            self.residual_support_z,
        ):
            if not np.isfinite(np.asarray(value)).all():
                raise ValueError("gate features must be finite")
        if np.asarray(self.top_agreement).dtype != np.bool_:
            raise ValueError("gate agreement must be boolean")


@dataclass(frozen=True)
class GateDecision:
    selected_indices: np.ndarray
    gate_pass: np.ndarray


def _finite_matrix(value: object, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (8, 225) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite matrix with shape [8,225]")
    return array


def extract_gate_features(score_shard: Mapping[str, object]) -> GateFeatures:
    """Derive scale-free safety signals from prediction-only ranker outputs."""
    try:
        full = _finite_matrix(score_shard["full_context_scores"], "full scores")
        residual = _finite_matrix(
            score_shard["residual_only_scores"], "residual scores"
        )
        alphas = _finite_matrix(score_shard["alphas"], "alphas")
    except KeyError as error:
        raise ValueError("score shard is missing gate inputs") from error
    if not np.allclose(alphas[:, 0], 0.0) or np.any(alphas[:, 1:] <= 0.0):
        raise ValueError("gate score shard must contain one no-op followed by positive alphas")

    rows = np.arange(8)
    proposed = np.argmax(full, axis=1).astype(np.int64)
    full_scale = np.maximum(np.std(full, axis=1), 1e-12)
    residual_scale = np.maximum(np.std(residual, axis=1), 1e-12)
    top_score = full[rows, proposed]
    second_score = np.partition(full, -2, axis=1)[:, -2]
    residual_top = np.argmax(residual, axis=1).astype(np.int64)
    return GateFeatures(
        proposed_indices=proposed,
        proposed_alphas=alphas[rows, proposed].astype(np.float64),
        advantage_z=((top_score - full[:, 0]) / full_scale).astype(np.float64),
        prominence_z=((top_score - second_score) / full_scale).astype(np.float64),
        residual_support_z=(
            (residual[rows, proposed] - residual[:, 0]) / residual_scale
        ).astype(np.float64),
        top_agreement=(proposed == residual_top),
    )


def apply_gate_policy(features: GateFeatures, policy: GatePolicy) -> GateDecision:
    """Choose the ranker's original top candidate or fail closed to index zero."""
    if not policy.deployable:
        gate_pass = np.zeros(8, dtype=np.bool_)
    else:
        thresholds = (
            policy.max_alpha,
            policy.min_advantage_z,
            policy.min_prominence_z,
            policy.min_residual_support_z,
        )
        if not np.isfinite(np.asarray(thresholds, dtype=np.float64)).all():
            raise ValueError("gate policy thresholds must be finite")
        if policy.max_alpha <= 0.0 or policy.max_alpha > MAX_GATE_ALPHA:
            raise ValueError("a deployable gate must keep max_alpha in (0, 0.2]")
        alpha_cap = float(np.float32(policy.max_alpha))
        gate_pass = (
            (features.proposed_indices != 0)
            & (features.proposed_alphas <= alpha_cap)
            & (features.advantage_z >= policy.min_advantage_z)
            & (features.prominence_z >= policy.min_prominence_z)
            & (features.residual_support_z >= policy.min_residual_support_z)
        )
        if policy.require_top_agreement:
            gate_pass &= features.top_agreement
    selected = np.where(gate_pass, features.proposed_indices, 0).astype(np.int64)
    return GateDecision(selected_indices=selected, gate_pass=gate_pass.astype(np.bool_))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _valid_digest(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def write_gated_scene_selection(
    dataset: PredictionCandidateDataset,
    scene: str,
    score_path: Path,
    policy: GatePolicy,
    policy_sha256: str,
    destination: Path,
) -> Path:
    """Materialize gated latents without accepting a label-bearing input."""
    from .evaluate import load_score_shard

    if not _valid_digest(policy_sha256):
        raise ValueError("gate policy SHA-256 is invalid")
    scores = load_score_shard(score_path)
    if dataset.scenes.count(scene) != 1 or str(scores["scene"]) != scene:
        raise ValueError("gated selection scene must occur exactly once and match scores")
    if str(scores["binding_manifest_sha256"]) != _sha256_file(dataset.binding_manifest):
        raise ValueError("gated score does not bind the prediction manifest")
    scene_index = dataset.scenes.index(scene)
    groups = [dataset[scene_index * 8 + overlap] for overlap in range(8)]
    if [group.sample_id for group in groups] != scores["source_sample_ids"].tolist():
        raise ValueError("gated selection groups do not match score sample IDs")

    features = extract_gate_features(scores)
    decision = apply_gate_policy(features, policy)
    rows = np.arange(8)
    proposed = features.proposed_indices
    selected = decision.selected_indices
    corrected = np.stack(
        [groups[row].x0 + groups[row].delta_tokens[int(selected[row])] for row in rows]
    ).astype(np.float32)
    arrays: dict[str, np.ndarray] = {
        "scene": scores["scene"].copy(),
        "role": scores["role"].copy(),
        "source_sample_ids": scores["source_sample_ids"].copy(),
        "span_starts": scores["span_starts"].copy(),
        "proposed_indices": proposed.astype(np.int64),
        "selected_indices": selected.astype(np.int64),
        "gate_pass": decision.gate_pass.astype(np.bool_),
        "proposed_choice_ids": scores["choice_ids"][rows, proposed].astype("U160"),
        "selected_choice_ids": scores["choice_ids"][rows, selected].astype("U160"),
        "proposed_alphas": features.proposed_alphas.astype(np.float32),
        "selected_alphas": scores["alphas"][rows, selected].astype(np.float32),
        "proposed_z": scores["z"][rows, proposed].astype(np.float32),
        "selected_z": scores["z"][rows, selected].astype(np.float32),
        "advantage_z": features.advantage_z.astype(np.float32),
        "prominence_z": features.prominence_z.astype(np.float32),
        "residual_support_z": features.residual_support_z.astype(np.float32),
        "top_agreement": features.top_agreement.astype(np.bool_),
        "selected_scores": scores["full_context_scores"][rows, selected].astype(
            np.float32
        ),
        "corrected_camera_tokens": corrected,
        "source_sha256": scores["source_sha256"].copy(),
        "candidate_sha256": scores["candidate_sha256"].copy(),
        "residual_prediction_sha256": scores["residual_prediction_sha256"].copy(),
        "binding_manifest_sha256": scores["binding_manifest_sha256"].copy(),
        "checkpoint_sha256": scores["checkpoint_sha256"].copy(),
        "score_sha256": np.asarray(_sha256_file(score_path), dtype="U64"),
        "policy_sha256": np.asarray(policy_sha256, dtype="U64"),
    }
    _validate_gated_selection(arrays)
    _atomic_npz(destination, arrays)
    return Path(destination)


def _validate_gated_selection(arrays: Mapping[str, np.ndarray]) -> None:
    names = set(arrays)
    forbidden = sorted(
        name for name in names if any(part in name.lower() for part in _FORBIDDEN_PARTS)
    )
    if forbidden:
        raise ValueError(f"gated prediction contains forbidden members: {forbidden}")
    if names != _GATED_SELECTION_MEMBERS:
        raise ValueError("gated prediction members do not match the schema")
    normalized = {name: np.asarray(value) for name, value in arrays.items()}
    if any(value.dtype.hasobject for value in normalized.values()):
        raise ValueError("gated prediction may not contain object arrays")
    for name in ("scene", "role"):
        if normalized[name].shape != () or normalized[name].dtype.kind != "U":
            raise ValueError(f"gated prediction {name} must be a Unicode scalar")
    for name in ("source_sample_ids", "proposed_choice_ids", "selected_choice_ids"):
        if normalized[name].shape != (8,) or normalized[name].dtype.kind != "U":
            raise ValueError(f"gated prediction {name} is invalid")
    if normalized["span_starts"].shape != (8,) or not np.issubdtype(
        normalized["span_starts"].dtype, np.integer
    ):
        raise ValueError("gated prediction spans are invalid")
    for name in ("proposed_indices", "selected_indices"):
        value = normalized[name]
        if value.shape != (8,) or not np.issubdtype(value.dtype, np.integer) or np.any(
            (value < 0) | (value >= 225)
        ):
            raise ValueError(f"gated prediction {name} is invalid")
    for name in ("gate_pass", "top_agreement"):
        if normalized[name].shape != (8,) or normalized[name].dtype != np.bool_:
            raise ValueError(f"gated prediction {name} must be boolean [8]")
    for name in (
        "proposed_alphas",
        "selected_alphas",
        "advantage_z",
        "prominence_z",
        "residual_support_z",
        "selected_scores",
    ):
        value = normalized[name]
        if value.shape != (8,) or not np.issubdtype(value.dtype, np.floating) or not np.isfinite(
            value
        ).all():
            raise ValueError(f"gated prediction {name} must be finite floating [8]")
    for name in ("proposed_z", "selected_z"):
        value = normalized[name]
        if value.ndim != 2 or value.shape[0] != 8 or value.shape[1] < 1 or not np.isfinite(
            value
        ).all():
            raise ValueError(f"gated prediction {name} is invalid")
    corrected = normalized["corrected_camera_tokens"]
    if corrected.shape != (8, 50, 2048) or corrected.dtype != np.float32 or not np.isfinite(
        corrected
    ).all():
        raise ValueError("gated corrected latent must be finite float32 [8,50,2048]")
    if not np.array_equal(normalized["selected_indices"] != 0, normalized["gate_pass"]):
        raise ValueError("gated pass mask must exactly match non-noop selections")
    for name in (
        "source_sha256",
        "candidate_sha256",
        "residual_prediction_sha256",
        "binding_manifest_sha256",
        "checkpoint_sha256",
        "score_sha256",
        "policy_sha256",
    ):
        if normalized[name].shape != () or normalized[name].dtype.kind != "U" or not _valid_digest(
            str(normalized[name])
        ):
            raise ValueError(f"gated prediction {name} must be a SHA-256 scalar")


def load_gated_selection(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as payload:
            arrays = {name: payload[name] for name in payload.files}
    except (OSError, ValueError, KeyError) as error:
        raise ValueError(f"invalid gated selection: {path}") from error
    _validate_gated_selection(arrays)
    return arrays


def _policy_values(policy: GatePolicy) -> dict[str, object]:
    return {
        "deployable": policy.deployable,
        "max_alpha": policy.max_alpha,
        "min_advantage_z": policy.min_advantage_z,
        "min_prominence_z": policy.min_prominence_z,
        "min_residual_support_z": policy.min_residual_support_z,
        "require_top_agreement": policy.require_top_agreement,
    }


def write_gate_policy(
    path: Path,
    policy: GatePolicy,
    *,
    training_scenes: tuple[str, ...],
    fit_manifest_sha256: str,
) -> Path:
    if (
        len(training_scenes) != 8
        or len(set(training_scenes)) != 8
        or any(not scene for scene in training_scenes)
    ):
        raise ValueError("gate policy requires exactly eight unique training scenes")
    if not _valid_digest(fit_manifest_sha256):
        raise ValueError("gate fit manifest SHA-256 is invalid")
    payload = {
        "schema": GATE_POLICY_SCHEMA,
        "policy": _policy_values(policy),
        "training_scenes": list(training_scenes),
        "fit_manifest_sha256": fit_manifest_sha256,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)
    return path


def load_gate_policy(path: Path) -> GatePolicy:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid gate policy: {path}") from error
    if not isinstance(payload, dict) or set(payload) != {
        "schema",
        "policy",
        "training_scenes",
        "fit_manifest_sha256",
    }:
        raise ValueError("gate policy members do not match the schema")
    if payload["schema"] != GATE_POLICY_SCHEMA:
        raise ValueError("gate policy schema does not match")
    scenes = payload["training_scenes"]
    if (
        not isinstance(scenes, list)
        or len(scenes) != 8
        or len(set(scenes)) != 8
        or any(not isinstance(scene, str) or not scene for scene in scenes)
        or not _valid_digest(payload["fit_manifest_sha256"])
    ):
        raise ValueError("gate policy provenance is invalid")
    values = payload["policy"]
    expected = set(_policy_values(GatePolicy.fail_closed()))
    if not isinstance(values, dict) or set(values) != expected:
        raise ValueError("gate policy thresholds do not match the schema")
    try:
        policy = GatePolicy(
            deployable=values["deployable"],
            max_alpha=float(values["max_alpha"]),
            min_advantage_z=float(values["min_advantage_z"]),
            min_prominence_z=float(values["min_prominence_z"]),
            min_residual_support_z=float(values["min_residual_support_z"]),
            require_top_agreement=values["require_top_agreement"],
        )
    except (TypeError, ValueError) as error:
        raise ValueError("gate policy thresholds are invalid") from error
    if type(policy.deployable) is not bool or type(policy.require_top_agreement) is not bool:
        raise ValueError("gate policy booleans are invalid")
    numeric = np.asarray(
        (
            policy.max_alpha,
            policy.min_advantage_z,
            policy.min_prominence_z,
            policy.min_residual_support_z,
        ),
        dtype=np.float64,
    )
    if not np.isfinite(numeric).all() or (
        policy.deployable
        and (policy.max_alpha <= 0.0 or policy.max_alpha > MAX_GATE_ALPHA)
    ):
        raise ValueError("gate policy numeric thresholds are invalid")
    return policy
