"""Frozen 20-transform structured-null protocol for matched-random controls.

This module deliberately contains only prediction-safe planning and pure
statistics.  Loading ground truth or privileged labels belongs to the separate
finalization phase in :mod:`pipeline`.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .matched_random_ablation import (
    _CAMERA_ITERATIONS,
    _CONTEXT_FRAMES,
    _DECODE_PROTOCOL,
    _RNG_ALGORITHM,
    _TRANSFORM_PROTOCOL,
    _build_shared_transform,
)


FORMAL_REPLICATE_COUNT = 20
SCENE_COUNT = 10
OVERLAPS_PER_SCENE = 8
DIRECTIONS_PER_OVERLAP = 32
NONZERO_ALPHAS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)
PLAN_SCHEMA = "variational_camera_latent.matched_random_20q_plan.v1"
SEED_DERIVATION_PROTOCOL = (
    "sha256(protocol_master_transform_reference_replicate_index).v1"
)


def _canonical_digest(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _validate_digest(value: str, label: str, *, length: int = 64) -> str:
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase hexadecimal digest")
    return value


def _validate_master_seed(value: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError("master_seed must be an integer")
    normalized = int(value)
    if normalized < 0 or normalized >= 2**63:
        raise ValueError("master_seed must be in [0, 2**63)")
    return normalized


def derive_replicate_seed(
    *,
    master_seed: int,
    transform_identity_sha256: str,
    reference_prediction_manifest_sha256: str,
    replicate_index: int,
) -> tuple[int, str]:
    """Derive one preregistered seed without consulting evaluation labels."""
    normalized_master_seed = _validate_master_seed(master_seed)
    _validate_digest(transform_identity_sha256, "transform_identity_sha256")
    _validate_digest(
        reference_prediction_manifest_sha256,
        "reference_prediction_manifest_sha256",
    )
    if (
        isinstance(replicate_index, (bool, np.bool_))
        or not isinstance(replicate_index, (int, np.integer))
        or not 0 <= int(replicate_index) < FORMAL_REPLICATE_COUNT
    ):
        raise ValueError("replicate index must be in [0, 20)")
    material = "|".join(
        (
            SEED_DERIVATION_PROTOCOL,
            str(normalized_master_seed),
            transform_identity_sha256,
            reference_prediction_manifest_sha256,
            str(int(replicate_index)),
        )
    ).encode("ascii")
    digest = hashlib.sha256(material).hexdigest()
    seed = int.from_bytes(bytes.fromhex(digest)[:8], "big") & (2**63 - 1)
    return seed, digest


def build_matched_random_ensemble_plan(
    *,
    master_seed: int,
    transform_identity_sha256: str,
    reference_prediction_manifest_sha256: str,
    reference_transform_sha256: str,
    producer_git_commit: str,
    camera_head_checkpoint_sha256: str,
    source_manifest_sha256: str,
    candidate_manifest_sha256: str,
    vrfm_prediction_manifest_sha256: str,
    scenes: Sequence[str],
) -> dict[str, object]:
    """Build the immutable prediction-only plan for all formal null replicates."""
    normalized_master_seed = _validate_master_seed(master_seed)
    for label, digest in (
        ("transform_identity_sha256", transform_identity_sha256),
        ("reference_prediction_manifest_sha256", reference_prediction_manifest_sha256),
        ("reference_transform_sha256", reference_transform_sha256),
        ("camera_head_checkpoint_sha256", camera_head_checkpoint_sha256),
        ("source_manifest_sha256", source_manifest_sha256),
        ("candidate_manifest_sha256", candidate_manifest_sha256),
        ("vrfm_prediction_manifest_sha256", vrfm_prediction_manifest_sha256),
    ):
        _validate_digest(digest, label)
    _validate_digest(producer_git_commit, "producer_git_commit", length=40)
    scene_values = list(scenes)
    if (
        len(scene_values) != SCENE_COUNT
        or len(set(scene_values)) != SCENE_COUNT
        or any(not isinstance(scene, str) or not scene for scene in scene_values)
    ):
        raise ValueError("20-Q plan requires ten unique scene names")

    replicates: list[dict[str, object]] = []
    seeds: set[int] = set()
    transforms: set[str] = set()
    for replicate_index in range(FORMAL_REPLICATE_COUNT):
        seed, seed_material_sha256 = derive_replicate_seed(
            master_seed=normalized_master_seed,
            transform_identity_sha256=transform_identity_sha256,
            reference_prediction_manifest_sha256=(
                reference_prediction_manifest_sha256
            ),
            replicate_index=replicate_index,
        )
        _, _, _, _, transform_sha256 = _build_shared_transform(
            base_seed=seed,
            identity_sha256=transform_identity_sha256,
        )
        if seed in seeds or transform_sha256 in transforms:
            raise ValueError("20-Q plan produced a duplicate seed or transform")
        if transform_sha256 == reference_transform_sha256:
            raise ValueError("formal 20-Q plan repeats the observed pilot transform")
        seeds.add(seed)
        transforms.add(transform_sha256)
        prefix = f"replicate_{replicate_index:03d}"
        replicates.append(
            {
                "replicate_index": replicate_index,
                "replicate_id": f"formal_null_{replicate_index:03d}",
                "replicate_seed": seed,
                "seed_material_sha256": seed_material_sha256,
                "expected_transform_sha256": transform_sha256,
                "prediction_root": (
                    "prediction_only/"
                    f"matched_random_ablation_20q_full_context/{prefix}"
                ),
                "prediction_manifest_path": (
                    "manifests/matched_random_20q/"
                    f"{prefix}_prediction_manifest.json"
                ),
                "prediction_completion_path": (
                    "manifests/matched_random_20q/"
                    f"{prefix}_prediction_complete.json"
                ),
            }
        )

    directions_per_replicate = (
        SCENE_COUNT * OVERLAPS_PER_SCENE * DIRECTIONS_PER_OVERLAP
    )
    grid_cells_per_replicate = directions_per_replicate * (
        1 + len(NONZERO_ALPHAS)
    )
    unique_poses_per_replicate = SCENE_COUNT * OVERLAPS_PER_SCENE * (
        1 + DIRECTIONS_PER_OVERLAP * len(NONZERO_ALPHAS)
    )
    family_unique_poses = SCENE_COUNT * OVERLAPS_PER_SCENE * (
        1
        + FORMAL_REPLICATE_COUNT
        * DIRECTIONS_PER_OVERLAP
        * len(NONZERO_ALPHAS)
    )
    return {
        "schema": PLAN_SCHEMA,
        "protocol": "matched_random_structured_null_20q.v1",
        "replicate_count": FORMAL_REPLICATE_COUNT,
        "formal_replicate_indices": list(range(FORMAL_REPLICATE_COUNT)),
        "master_seed": normalized_master_seed,
        "seed_derivation_protocol": SEED_DERIVATION_PROTOCOL,
        "transform_identity_sha256": transform_identity_sha256,
        "reference_prediction_manifest_sha256": (
            reference_prediction_manifest_sha256
        ),
        "reference_transform_sha256": reference_transform_sha256,
        "observed_pilot_included_in_formal_null": False,
        "producer_git_commit": producer_git_commit,
        "camera_head_checkpoint_sha256": camera_head_checkpoint_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "candidate_manifest_sha256": candidate_manifest_sha256,
        "vrfm_prediction_manifest_sha256": vrfm_prediction_manifest_sha256,
        "scenes": scene_values,
        "matched_oracle_budget_per_replicate": {
            "directions_per_overlap": DIRECTIONS_PER_OVERLAP,
            "nonzero_alpha_count": len(NONZERO_ALPHAS),
            "no_op_count_per_overlap": 1,
            "alphas": [0.0, *NONZERO_ALPHAS],
            "directions": directions_per_replicate,
            "grid_cells": grid_cells_per_replicate,
            "unique_poses": unique_poses_per_replicate,
        },
        "random_family_budget": {
            "directions": directions_per_replicate * FORMAL_REPLICATE_COUNT,
            "grid_cells": grid_cells_per_replicate * FORMAL_REPLICATE_COUNT,
            "unique_poses": family_unique_poses,
            "shared_no_op_count_per_overlap": 1,
        },
        "prediction_contract": {
            "transform_protocol": _TRANSFORM_PROTOCOL,
            "rng_algorithm": _RNG_ALGORITHM,
            "decode_protocol": _DECODE_PROTOCOL,
            "decode_context_frames": _CONTEXT_FRAMES,
            "camera_iterations": _CAMERA_ITERATIONS,
            "same_transform_across_all_scenes": True,
            "preserves_feature_row_gram_geometry": True,
        },
        "replicates": replicates,
    }


def write_matched_random_ensemble_plan(
    destination: Path, plan: dict[str, object]
) -> dict[str, object]:
    if "plan_digest" in plan:
        raise ValueError("invalid unsigned 20-Q plan")
    validate_matched_random_ensemble_plan(plan)
    signed = {**plan, "plan_digest": _canonical_digest(plan)}
    destination = Path(destination)
    if destination.is_file():
        try:
            existing = json.loads(destination.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("existing 20-Q plan is invalid") from error
        if existing != signed:
            raise ValueError("existing 20-Q plan differs")
        return existing
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(signed, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(destination)
    return signed


def validate_matched_random_ensemble_plan(
    plan: dict[str, object],
) -> dict[str, object]:
    """Recompute every derived plan field and reject contract drift."""
    if not isinstance(plan, dict) or plan.get("schema") != PLAN_SCHEMA:
        raise ValueError("invalid 20-Q plan schema")
    try:
        expected = build_matched_random_ensemble_plan(
            master_seed=plan["master_seed"],
            transform_identity_sha256=plan["transform_identity_sha256"],
            reference_prediction_manifest_sha256=plan[
                "reference_prediction_manifest_sha256"
            ],
            reference_transform_sha256=plan["reference_transform_sha256"],
            producer_git_commit=plan["producer_git_commit"],
            camera_head_checkpoint_sha256=plan[
                "camera_head_checkpoint_sha256"
            ],
            source_manifest_sha256=plan["source_manifest_sha256"],
            candidate_manifest_sha256=plan["candidate_manifest_sha256"],
            vrfm_prediction_manifest_sha256=plan[
                "vrfm_prediction_manifest_sha256"
            ],
            scenes=plan["scenes"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid 20-Q plan payload") from error
    if plan != expected:
        raise ValueError("20-Q plan derived fields differ")
    return plan


def load_matched_random_ensemble_plan(path: Path) -> dict[str, object]:
    try:
        signed = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("20-Q plan is missing or invalid") from error
    if not isinstance(signed, dict):
        raise ValueError("20-Q plan must be a JSON object")
    unsigned = dict(signed)
    digest = unsigned.pop("plan_digest", None)
    if unsigned.get("schema") != PLAN_SCHEMA or digest != _canonical_digest(unsigned):
        raise ValueError("20-Q plan digest does not match")
    validate_matched_random_ensemble_plan(unsigned)
    return signed


def summarize_matched_random_ensemble(
    identity_overlap_best: np.ndarray,
    random_overlap_best: np.ndarray,
    *,
    replicate_indices: Sequence[int],
    expected_replicates: Sequence[dict[str, object]],
    actual_replicate_bindings: Sequence[dict[str, object]],
    observed_pilot_score: float | None = None,
) -> dict[str, object]:
    """Rank the learned orientation against 20 new structured-null Qs."""
    identity = np.asarray(identity_overlap_best, dtype=np.float64)
    random = np.asarray(random_overlap_best, dtype=np.float64)
    indices = list(replicate_indices)
    if identity.shape != (SCENE_COUNT, OVERLAPS_PER_SCENE):
        raise ValueError("identity scores must have shape [10, 8]")
    if random.shape != (
        FORMAL_REPLICATE_COUNT,
        SCENE_COUNT,
        OVERLAPS_PER_SCENE,
    ):
        raise ValueError("formal null requires exactly 20 score tensors [10, 8]")
    if len(indices) != FORMAL_REPLICATE_COUNT:
        raise ValueError("formal null requires exactly 20 replicate indices")
    if any(
        isinstance(index, (bool, np.bool_))
        or not isinstance(index, (int, np.integer))
        for index in indices
    ):
        raise ValueError("formal replicate indices must be integers")
    indices = [int(index) for index in indices]
    if len(set(indices)) != FORMAL_REPLICATE_COUNT:
        raise ValueError("formal null requires unique replicate indices")
    if set(indices) != set(range(FORMAL_REPLICATE_COUNT)):
        raise ValueError("formal null replicate indices must be 0 through 19")
    if not np.isfinite(identity).all() or not np.isfinite(random).all():
        raise ValueError("ensemble scores must be finite")
    tolerance = 1e-12
    if (
        np.any(identity < -tolerance)
        or np.any(identity > 1.0 + tolerance)
        or np.any(random < -tolerance)
        or np.any(random > 1.0 + tolerance)
    ):
        raise ValueError("overlap-best relative improvements must be in [0, 1]")
    identity = np.clip(identity, 0.0, 1.0)
    random = np.clip(random, 0.0, 1.0)
    bindings = validate_formal_replicate_bindings(
        expected_replicates,
        actual_replicate_bindings,
    )
    if [row["replicate_index"] for row in bindings] != indices:
        raise ValueError("score order does not match formal replicate bindings")

    identity_scene = np.median(identity, axis=1)
    random_scene = np.median(random, axis=2)
    identity_score = float(np.mean(identity_scene))
    random_scores = np.mean(random_scene, axis=1)
    greater = int(np.count_nonzero(random_scores > identity_score))
    greater_equal = int(np.count_nonzero(random_scores >= identity_score))
    less_equal = int(np.count_nonzero(random_scores <= identity_score))
    less = int(np.count_nonzero(random_scores < identity_score))
    tied = int(np.count_nonzero(random_scores == identity_score))
    denominator = FORMAL_REPLICATE_COUNT + 1
    replicates = [
        {
            "replicate_index": int(index),
            "score": float(random_scores[position]),
            "effect_random_minus_identity": float(
                random_scores[position] - identity_score
            ),
            "scene_scores": random_scene[position].tolist(),
        }
        for position, index in enumerate(indices)
    ]
    summary: dict[str, object] = {
        "formal_null_replicate_count": FORMAL_REPLICATE_COUNT,
        "formal_replicate_indices": indices,
        "randomization_unit": "structured_null_transform",
        "scene_count": SCENE_COUNT,
        "overlaps_per_scene": OVERLAPS_PER_SCENE,
        "scene_statistic": "median_overlap_oracle_best_relative_improvement",
        "global_statistic": "mean_scene_statistic",
        "higher_is_better": True,
        "identity_score": identity_score,
        "identity_scene_scores": identity_scene.tolist(),
        "identity_rank_descending_best_tie": 1 + greater,
        "identity_rank_descending_worst_tie": 1 + greater_equal,
        "null_less_count": less,
        "null_tie_count": tied,
        "null_greater_count": greater,
        "p_identity_unusually_good": float((1 + greater_equal) / denominator),
        "p_identity_unusually_bad": float((1 + less_equal) / denominator),
        "p_identity_two_sided": float(
            min(
                1.0,
                2.0
                * min(
                    (1 + greater_equal) / denominator,
                    (1 + less_equal) / denominator,
                ),
            )
        ),
        "ties_count_conservatively_against_tested_alternative": True,
        "minimum_attainable_one_sided_p": float(1 / denominator),
        "minimum_attainable_two_sided_p": float(2 / denominator),
        "null_score_summary": {
            "median": float(np.median(random_scores)),
            "q1": float(np.quantile(random_scores, 0.25)),
            "q3": float(np.quantile(random_scores, 0.75)),
            "minimum": float(np.min(random_scores)),
            "maximum": float(np.max(random_scores)),
        },
        "replicates": replicates,
        "formal_replicate_bindings": bindings,
        "observed_pilot_included_in_p_values": False,
        "formal_scope": (
            "conditional_on_fixed_ten_calibration_scenes_and_oracle_budget"
        ),
        "formal_training_attribution": False,
        "oracle_upper_bound": True,
    }
    if observed_pilot_score is not None:
        if not np.isfinite(observed_pilot_score):
            raise ValueError("observed_pilot_score must be finite")
        summary["observed_pilot_score"] = float(observed_pilot_score)
    return summary


def validate_formal_replicate_bindings(
    expected_replicates: Sequence[dict[str, object]],
    actual_replicate_bindings: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    """Bind every score row to the preregistered seed and transform digest."""
    expected = list(expected_replicates)
    actual = list(actual_replicate_bindings)
    if len(expected) != FORMAL_REPLICATE_COUNT or len(actual) != FORMAL_REPLICATE_COUNT:
        raise ValueError("formal score bindings require exactly 20 replicates")
    normalized: list[dict[str, object]] = []
    for position, (expected_row, actual_row) in enumerate(zip(expected, actual)):
        try:
            expected_binding = {
                "replicate_index": int(expected_row["replicate_index"]),
                "replicate_seed": int(expected_row["replicate_seed"]),
                "transform_sha256": str(expected_row["expected_transform_sha256"]),
            }
            actual_binding = {
                "replicate_index": int(actual_row["replicate_index"]),
                "replicate_seed": int(actual_row["replicate_seed"]),
                "transform_sha256": str(actual_row["transform_sha256"]),
            }
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("formal replicate binding is incomplete") from error
        if expected_binding["replicate_index"] != position:
            raise ValueError("expected formal replicate order differs")
        if actual_binding != expected_binding:
            raise ValueError("actual replicate binding differs from frozen 20-Q plan")
        _validate_master_seed(actual_binding["replicate_seed"])
        _validate_digest(actual_binding["transform_sha256"], "transform_sha256")
        normalized.append(actual_binding)
    if len({row["replicate_seed"] for row in normalized}) != FORMAL_REPLICATE_COUNT:
        raise ValueError("formal replicate seeds must be unique")
    if len({row["transform_sha256"] for row in normalized}) != FORMAL_REPLICATE_COUNT:
        raise ValueError("formal replicate transforms must be unique")
    return normalized
