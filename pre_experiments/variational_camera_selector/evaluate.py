from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from pre_experiments.variational_camera_latent.alpha_scan import DEFAULT_ALPHAS
from pre_experiments.variational_camera_latent.vrfm_residual_scan import (
    load_vrfm_residual_alpha_scan,
)

from .dataset import CandidateGroup, PredictionCandidateDataset
from .model import CandidateRanker


SCORE_SCHEMA = "variational_camera_selector.prediction_scores.v1"
EVALUATION_SCHEMA = "variational_camera_selector.privileged_evaluation.v1"
_CHECKPOINT_SCHEMA = "variational_camera_selector.training_checkpoint.v1"
_SCORE_MEMBERS = {
    "scene",
    "role",
    "source_sample_ids",
    "span_starts",
    "choice_ids",
    "alphas",
    "z",
    "sample_seeds",
    "full_context_scores",
    "residual_only_scores",
    "full_context_selected_indices",
    "residual_only_selected_indices",
    "source_sha256",
    "candidate_sha256",
    "residual_prediction_sha256",
    "binding_manifest_sha256",
    "checkpoint_sha256",
    "checkpoint_step",
}
_FORBIDDEN_SCORE_PARTS = (
    "gt",
    "ground_truth",
    "privileged",
    "depth",
    "quality",
    "error",
    "utility",
)
_EVALUATION_MEMBERS = {
    "scene",
    "role",
    "source_sample_ids",
    "full_context_selected_indices",
    "residual_only_selected_indices",
    "random_selected_indices",
    "oracle_selected_indices",
    "full_context_utility",
    "residual_only_utility",
    "random_utility",
    "noop_utility",
    "oracle_utility",
    "full_context_oracle_rank",
    "residual_only_oracle_rank",
    "full_context_spearman",
    "residual_only_spearman",
    "score_sha256",
    "sidecar_sha256",
    "prediction_sha256",
    "random_seed",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _valid_digest(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _load_npz(path: Path, label: str) -> dict[str, np.ndarray]:
    try:
        with np.load(Path(path), allow_pickle=False) as archive:
            return {name: np.asarray(archive[name]).copy() for name in archive.files}
    except (OSError, ValueError, KeyError) as error:
        raise ValueError(f"invalid {label}: {path}; object arrays are forbidden") from error


def _load_rankers(
    checkpoint_path: Path, device: torch.device
) -> tuple[CandidateRanker, CandidateRanker, int]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or checkpoint.get("schema") != _CHECKPOINT_SCHEMA:
        raise ValueError("selector checkpoint schema does not match")
    try:
        completed_step = int(checkpoint["completed_step"])
        model_config = checkpoint["model_config"]
        if completed_step < 1 or not isinstance(model_config, dict):
            raise ValueError
        expected_keys = {"d_model", "z_dim", "input_dim", "span_count"}
        if set(model_config) != expected_keys:
            raise ValueError
        full = CandidateRanker(
            **model_config, include_global_context=True
        ).to(device)  # type: ignore[arg-type]
        residual = CandidateRanker(
            **model_config, include_global_context=False
        ).to(device)  # type: ignore[arg-type]
        full.load_state_dict(checkpoint["full_context_model"])
        residual.load_state_dict(checkpoint["residual_only_model"])
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise ValueError("selector checkpoint model state is invalid") from error
    return full.eval(), residual.eval(), completed_step


def _group_tensors(
    group: CandidateGroup, device: torch.device
) -> tuple[torch.Tensor, ...]:
    return (
        torch.from_numpy(group.global_tokens.astype(np.float32))[None].to(device),
        torch.from_numpy(group.x0.astype(np.float32))[None].to(device),
        torch.from_numpy(group.delta_tokens.astype(np.float32))[None].to(device),
        torch.from_numpy(group.alphas.astype(np.float32))[None].to(device),
        torch.as_tensor([group.span_start], dtype=torch.long, device=device),
        torch.from_numpy(group.z.astype(np.float32))[None].to(device),
    )


def score_scene_candidates(
    dataset: PredictionCandidateDataset,
    scene: str,
    checkpoint_path: Path,
    destination: Path,
    *,
    device: str = "cuda",
) -> Path:
    """Write prediction-only scores for all eight overlap groups in one scene."""
    if dataset.scenes.count(scene) != 1:
        raise ValueError("requested scene must occur exactly once in the prediction dataset")
    scene_index = dataset.scenes.index(scene)
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA scoring was requested but CUDA is unavailable")
    full_model, residual_model, completed_step = _load_rankers(
        Path(checkpoint_path), torch_device
    )
    groups = [dataset[scene_index * 8 + overlap] for overlap in range(8)]
    if any(group.scene != scene for group in groups):
        raise ValueError("prediction dataset returned a different scene")
    full_rows: list[np.ndarray] = []
    residual_rows: list[np.ndarray] = []
    with torch.inference_mode():
        for group in groups:
            inputs = _group_tensors(group, torch_device)
            full_rows.append(full_model(*inputs)[0].float().cpu().numpy())
            residual_rows.append(residual_model(*inputs)[0].float().cpu().numpy())
    full_scores = np.stack(full_rows).astype(np.float32)
    residual_scores = np.stack(residual_rows).astype(np.float32)
    arrays = {
        "scene": np.asarray(scene, dtype="U64"),
        "role": np.asarray(groups[0].role, dtype="U16"),
        "source_sample_ids": np.asarray([group.sample_id for group in groups], dtype="U96"),
        "span_starts": np.asarray([group.span_start for group in groups], dtype=np.int64),
        "choice_ids": np.stack([group.choice_ids for group in groups]),
        "alphas": np.stack([group.alphas for group in groups]).astype(np.float32),
        "z": np.stack([group.z for group in groups]).astype(np.float32),
        "sample_seeds": np.stack([group.sample_seeds for group in groups]).astype(np.int64),
        "full_context_scores": full_scores,
        "residual_only_scores": residual_scores,
        "full_context_selected_indices": np.argmax(full_scores, axis=1).astype(np.int64),
        "residual_only_selected_indices": np.argmax(residual_scores, axis=1).astype(np.int64),
        "source_sha256": np.asarray(groups[0].source_sha256, dtype="U64"),
        "candidate_sha256": np.asarray(groups[0].candidate_sha256, dtype="U64"),
        "residual_prediction_sha256": np.asarray(
            groups[0].residual_prediction_sha256, dtype="U64"
        ),
        "binding_manifest_sha256": np.asarray(
            _sha256_file(dataset.binding_manifest), dtype="U64"
        ),
        "checkpoint_sha256": np.asarray(_sha256_file(checkpoint_path), dtype="U64"),
        "checkpoint_step": np.asarray(completed_step, dtype=np.int64),
    }
    _validate_score_shard(arrays)
    _atomic_npz(destination, arrays)
    return Path(destination)


def _validate_score_shard(arrays: Mapping[str, np.ndarray]) -> None:
    names = set(arrays)
    forbidden = sorted(
        name for name in names if any(part in name.lower() for part in _FORBIDDEN_SCORE_PARTS)
    )
    if forbidden:
        raise ValueError(f"prediction score shard contains forbidden members: {forbidden}")
    if names != _SCORE_MEMBERS:
        raise ValueError("prediction score shard members do not match the schema")
    normalized = {name: np.asarray(value) for name, value in arrays.items()}
    if any(value.dtype.hasobject for value in normalized.values()):
        raise ValueError("prediction score shard may not contain object arrays")
    expected = {
        "source_sample_ids": (8,),
        "span_starts": (8,),
        "choice_ids": (8, 225),
        "alphas": (8, 225),
        "sample_seeds": (8, 225),
        "full_context_scores": (8, 225),
        "residual_only_scores": (8, 225),
        "full_context_selected_indices": (8,),
        "residual_only_selected_indices": (8,),
        "checkpoint_step": (),
    }
    for name, shape in expected.items():
        if normalized[name].shape != shape:
            raise ValueError(f"prediction score member {name} has invalid shape")
    z = normalized["z"]
    if z.ndim != 3 or z.shape[:2] != (8, 225) or z.shape[2] < 1:
        raise ValueError("prediction score z must have shape [8, 225, z_dim]")
    for name in ("scene", "role"):
        if normalized[name].shape != () or normalized[name].dtype.kind != "U":
            raise ValueError(f"prediction score {name} must be a Unicode scalar")
    if normalized["source_sample_ids"].dtype.kind != "U" or normalized["choice_ids"].dtype.kind != "U":
        raise ValueError("prediction score IDs must be Unicode")
    expected_alphas = np.concatenate(
        (
            np.zeros(1, dtype=np.float32),
            np.repeat(np.asarray(DEFAULT_ALPHAS[1:], dtype=np.float32), 32),
        )
    )
    if not np.array_equal(normalized["alphas"], np.tile(expected_alphas, (8, 1))):
        raise ValueError("prediction score alpha order does not match the frozen grid")
    for name in ("alphas", "z", "full_context_scores", "residual_only_scores"):
        if not np.issubdtype(normalized[name].dtype, np.floating) or not np.isfinite(
            normalized[name]
        ).all():
            raise ValueError(f"prediction score member {name} must be finite floating point")
    for name, score_name in (
        ("full_context_selected_indices", "full_context_scores"),
        ("residual_only_selected_indices", "residual_only_scores"),
    ):
        if not np.array_equal(normalized[name], np.argmax(normalized[score_name], axis=1)):
            raise ValueError(f"{name} does not match score argmax")
    if int(normalized["checkpoint_step"]) < 1:
        raise ValueError("prediction score checkpoint step must be positive")
    for name in (
        "source_sha256",
        "candidate_sha256",
        "residual_prediction_sha256",
        "binding_manifest_sha256",
        "checkpoint_sha256",
    ):
        if normalized[name].shape != () or normalized[name].dtype.kind != "U" or not _valid_digest(
            str(normalized[name])
        ):
            raise ValueError(f"prediction score {name} must be a SHA-256 scalar")


def load_score_shard(path: Path) -> dict[str, np.ndarray]:
    arrays = _load_npz(path, "prediction score shard")
    _validate_score_shard(arrays)
    return arrays


def _rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    first = 0
    while first < len(values):
        last = first + 1
        while last < len(values) and values[order[last]] == values[order[first]]:
            last += 1
        ranks[order[first:last]] = 0.5 * (first + last - 1) + 1.0
        first = last
    return ranks


def _spearman(first: np.ndarray, second: np.ndarray) -> float:
    first_rank = _rankdata(first)
    second_rank = _rankdata(second)
    first_centered = first_rank - first_rank.mean()
    second_centered = second_rank - second_rank.mean()
    denominator = float(
        np.sqrt(np.sum(first_centered**2) * np.sum(second_centered**2))
    )
    if denominator == 0.0:
        return 0.0
    return float(np.sum(first_centered * second_centered) / denominator)


def _oracle_rank(scores: np.ndarray, oracle_index: int) -> int:
    order = np.argsort(-np.asarray(scores), kind="mergesort")
    return int(np.flatnonzero(order == oracle_index)[0]) + 1


def evaluate_scene_scores(
    score_path: Path,
    residual_prediction_path: Path,
    sidecar_path: Path,
    destination: Path,
    *,
    random_seed: int = 20260827,
) -> Path:
    """Attach sealed utilities only after prediction scores are complete and validated."""
    scores = load_score_shard(score_path)
    prediction = load_vrfm_residual_alpha_scan(residual_prediction_path)
    # Import the GT-bearing loader only in this privileged evaluation function.
    from pre_experiments.variational_camera_latent.vrfm_residual_scan import (
        load_vrfm_residual_privileged,
    )

    sidecar = load_vrfm_residual_privileged(sidecar_path)
    if str(scores["residual_prediction_sha256"]) != _sha256_file(residual_prediction_path):
        raise ValueError("score shard does not bind the residual prediction")
    if str(sidecar["prediction_sha256"]) != _sha256_file(residual_prediction_path):
        raise ValueError("sidecar does not bind the residual prediction")
    if str(scores["source_sha256"]) != str(prediction["source_shard_sha256"]):
        raise ValueError("score and residual prediction source digests differ")
    if str(scores["candidate_sha256"]) != str(prediction["candidate_shard_sha256"]):
        raise ValueError("score and residual prediction candidate digests differ")
    if not np.array_equal(scores["source_sample_ids"], prediction["source_sample_ids"]):
        raise ValueError("score and prediction sample IDs differ")
    if not np.array_equal(prediction["source_sample_ids"], sidecar["source_sample_ids"]):
        raise ValueError("prediction and sidecar sample IDs differ")
    if not np.array_equal(prediction["sample_seeds"], sidecar["sample_seeds"]):
        raise ValueError("prediction and sidecar sample seeds differ")
    if not np.array_equal(prediction["alphas"], sidecar["alphas"]):
        raise ValueError("prediction and sidecar alpha grids differ")

    utilities = np.concatenate(
        (
            np.zeros((8, 1), dtype=np.float64),
            sidecar["relative_improvement"][:, :, 1:].transpose(0, 2, 1).reshape(8, -1),
        ),
        axis=1,
    )
    expanded_seeds = np.concatenate(
        (
            np.full((8, 1), -1, dtype=np.int64),
            np.tile(prediction["sample_seeds"], (1, len(DEFAULT_ALPHAS) - 1)),
        ),
        axis=1,
    )
    if not np.array_equal(scores["sample_seeds"], expanded_seeds):
        raise ValueError("score choice seeds do not match residual prediction")
    full_indices = scores["full_context_selected_indices"].astype(np.int64)
    residual_indices = scores["residual_only_selected_indices"].astype(np.int64)
    rows = np.arange(8)
    scene_seed = int.from_bytes(
        hashlib.sha256(f"{str(scores['scene'])}:{random_seed}".encode("utf-8")).digest()[:8],
        "little",
    )
    generator = np.random.default_rng(scene_seed)
    random_indices = generator.integers(0, 225, size=8, dtype=np.int64)
    oracle_indices = np.argmax(utilities, axis=1).astype(np.int64)
    full_ranks = np.asarray(
        [_oracle_rank(scores["full_context_scores"][row], int(oracle_indices[row])) for row in rows],
        dtype=np.int64,
    )
    residual_ranks = np.asarray(
        [_oracle_rank(scores["residual_only_scores"][row], int(oracle_indices[row])) for row in rows],
        dtype=np.int64,
    )
    full_spearman = np.asarray(
        [_spearman(scores["full_context_scores"][row], utilities[row]) for row in rows],
        dtype=np.float64,
    )
    residual_spearman = np.asarray(
        [_spearman(scores["residual_only_scores"][row], utilities[row]) for row in rows],
        dtype=np.float64,
    )
    arrays = {
        "scene": scores["scene"].copy(),
        "role": scores["role"].copy(),
        "source_sample_ids": scores["source_sample_ids"].copy(),
        "full_context_selected_indices": full_indices,
        "residual_only_selected_indices": residual_indices,
        "random_selected_indices": random_indices,
        "oracle_selected_indices": oracle_indices,
        "full_context_utility": utilities[rows, full_indices],
        "residual_only_utility": utilities[rows, residual_indices],
        "random_utility": utilities[rows, random_indices],
        "noop_utility": np.zeros(8, dtype=np.float64),
        "oracle_utility": utilities[rows, oracle_indices],
        "full_context_oracle_rank": full_ranks,
        "residual_only_oracle_rank": residual_ranks,
        "full_context_spearman": full_spearman,
        "residual_only_spearman": residual_spearman,
        "score_sha256": np.asarray(_sha256_file(score_path), dtype="U64"),
        "sidecar_sha256": np.asarray(_sha256_file(sidecar_path), dtype="U64"),
        "prediction_sha256": np.asarray(_sha256_file(residual_prediction_path), dtype="U64"),
        "random_seed": np.asarray(random_seed, dtype=np.int64),
    }
    _validate_evaluation(arrays)
    _atomic_npz(destination, arrays)
    return Path(destination)


def _validate_evaluation(arrays: Mapping[str, np.ndarray]) -> None:
    if set(arrays) != _EVALUATION_MEMBERS:
        raise ValueError("privileged evaluation members do not match the schema")
    normalized = {name: np.asarray(value) for name, value in arrays.items()}
    if any(value.dtype.hasobject for value in normalized.values()):
        raise ValueError("privileged evaluation may not contain object arrays")
    for name in ("scene", "role"):
        if normalized[name].shape != () or normalized[name].dtype.kind != "U":
            raise ValueError(f"privileged evaluation {name} must be a Unicode scalar")
    if normalized["source_sample_ids"].shape != (8,) or normalized[
        "source_sample_ids"
    ].dtype.kind != "U":
        raise ValueError("privileged evaluation sample IDs are invalid")
    vector_names = _EVALUATION_MEMBERS - {
        "scene",
        "role",
        "source_sample_ids",
        "score_sha256",
        "sidecar_sha256",
        "prediction_sha256",
        "random_seed",
    }
    for name in vector_names:
        if normalized[name].shape != (8,):
            raise ValueError(f"privileged evaluation member {name} must have shape [8]")
    for name in (
        "full_context_utility",
        "residual_only_utility",
        "random_utility",
        "noop_utility",
        "oracle_utility",
        "full_context_spearman",
        "residual_only_spearman",
    ):
        if not np.issubdtype(normalized[name].dtype, np.floating) or not np.isfinite(
            normalized[name]
        ).all():
            raise ValueError(f"privileged evaluation member {name} must be finite")
    for name in ("score_sha256", "sidecar_sha256", "prediction_sha256"):
        if normalized[name].shape != () or normalized[name].dtype.kind != "U" or not _valid_digest(
            str(normalized[name])
        ):
            raise ValueError(f"privileged evaluation {name} is invalid")


def load_evaluation_sidecar(path: Path) -> dict[str, np.ndarray]:
    arrays = _load_npz(path, "privileged selector evaluation")
    _validate_evaluation(arrays)
    return arrays


def classify_signal(scene_rows: Sequence[Mapping[str, float]]) -> str:
    full_beats_noop = all(
        row["full_context_mean"] > row["noop_mean"] for row in scene_rows
    )
    full_beats_controls = np.mean(
        [
            row["full_context_mean"]
            - max(row["residual_only_mean"], row["random_mean"])
            for row in scene_rows
        ]
    ) > 0.0
    if full_beats_noop and full_beats_controls:
        return "LEARNABLE_SIGNAL"
    if any(row["full_context_mean"] > row["noop_mean"] for row in scene_rows):
        return "WEAK_SIGNAL"
    return "NO_GENERALIZATION"


def _evaluation_mapping(value: Path | Mapping[str, object]) -> Mapping[str, object]:
    return load_evaluation_sidecar(value) if isinstance(value, Path) else value


def _vector(record: Mapping[str, object], name: str) -> np.ndarray:
    value = np.asarray(record[name])
    if value.shape != (8,) or not np.isfinite(value).all():
        raise ValueError(f"calibration record {name} must be a finite vector [8]")
    return value


def summarize_calibration(
    scene_evaluations: Sequence[Path | Mapping[str, object]],
    *,
    random_seed: int = 20260827,
) -> dict[str, object]:
    if not scene_evaluations:
        raise ValueError("calibration requires at least one scene evaluation")
    records = [_evaluation_mapping(value) for value in scene_evaluations]
    per_scene: list[dict[str, float | str | int]] = []
    seen: set[str] = set()
    vectors: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "full_context_utility",
            "residual_only_utility",
            "random_utility",
            "noop_utility",
            "oracle_utility",
            "full_context_oracle_rank",
            "full_context_spearman",
        )
    }
    for record in records:
        scene = str(record["scene"])
        if not scene or scene in seen:
            raise ValueError("calibration scene IDs must be non-empty and unique")
        seen.add(scene)
        for name in vectors:
            vectors[name].append(_vector(record, name))
        row: dict[str, float | str | int] = {
            "scene": scene,
            "full_context_mean": float(vectors["full_context_utility"][-1].mean()),
            "residual_only_mean": float(vectors["residual_only_utility"][-1].mean()),
            "random_mean": float(vectors["random_utility"][-1].mean()),
            "noop_mean": float(vectors["noop_utility"][-1].mean()),
            "oracle_mean": float(vectors["oracle_utility"][-1].mean()),
            "full_context_positive_over_1pct_count": int(
                np.count_nonzero(vectors["full_context_utility"][-1] > 0.01)
            ),
            "full_context_oracle_regret_mean": float(
                (
                    vectors["oracle_utility"][-1]
                    - vectors["full_context_utility"][-1]
                ).mean()
            ),
        }
        per_scene.append(row)

    def metric(name: str) -> dict[str, object]:
        scene_values = np.asarray([values.mean() for values in vectors[name]])
        overlap_values = np.concatenate(vectors[name])
        payload: dict[str, object] = {
            "scene_mean": float(scene_values.mean()),
            "scene_median": float(np.median(scene_values)),
            "overlap_mean": float(overlap_values.mean()),
            "overlap_median": float(np.median(overlap_values)),
            "per_scene_mean": {
                str(row["scene"]): float(scene_values[index])
                for index, row in enumerate(per_scene)
            },
        }
        if name.endswith("utility"):
            payload["positive_over_1pct_count"] = int(
                np.count_nonzero(overlap_values > 0.01)
            )
        return payload

    full_rank = np.concatenate(vectors["full_context_oracle_rank"])
    full_metric = metric("full_context_utility")
    full_metric.update(
        {
            "oracle_regret_scene_mean": float(
                np.mean(
                    [
                        row["oracle_mean"] - row["full_context_mean"]  # type: ignore[operator]
                        for row in per_scene
                    ]
                )
            ),
            "top1_oracle_coverage": float(np.mean(full_rank <= 1)),
            "top4_oracle_coverage": float(np.mean(full_rank <= 4)),
            "top8_oracle_coverage": float(np.mean(full_rank <= 8)),
            "spearman_overlap_mean": float(
                np.concatenate(vectors["full_context_spearman"]).mean()
            ),
        }
    )
    classification_rows = [
        {
            "full_context_mean": float(row["full_context_mean"]),
            "residual_only_mean": float(row["residual_only_mean"]),
            "random_mean": float(row["random_mean"]),
            "noop_mean": float(row["noop_mean"]),
        }
        for row in per_scene
    ]
    return {
        "schema": "variational_camera_selector.calibration_summary.v1",
        "scene_count": len(records),
        "overlap_count": 8 * len(records),
        "inference_unit": "overlap",
        "aggregate_unit": "scene",
        "random_seed": random_seed,
        "classification": classify_signal(classification_rows),
        "full_context": full_metric,
        "residual_only": metric("residual_only_utility"),
        "random": metric("random_utility"),
        "noop": metric("noop_utility"),
        "oracle": metric("oracle_utility"),
        "full_context_minus_residual_only_scene_mean": float(
            np.mean(
                [row["full_context_mean"] - row["residual_only_mean"] for row in per_scene]  # type: ignore[operator]
            )
        ),
        "full_context_minus_random_scene_mean": float(
            np.mean(
                [row["full_context_mean"] - row["random_mean"] for row in per_scene]  # type: ignore[operator]
            )
        ),
        "full_context_minus_noop_scene_mean": float(
            np.mean(
                [row["full_context_mean"] - row["noop_mean"] for row in per_scene]  # type: ignore[operator]
            )
        ),
        "per_scene": per_scene,
    }
