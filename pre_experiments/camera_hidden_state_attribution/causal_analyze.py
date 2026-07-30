"""Aggregate and validate Camera Head hidden causal preference results."""

from __future__ import annotations

import csv
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from pre_experiments.camera_hidden_state_attribution.artifacts import (
    canonical_digest,
)
from pre_experiments.camera_hidden_state_attribution.attribution import (
    group_specificity,
)
from pre_experiments.camera_hidden_state_attribution.causal_preference import (
    GROUPS,
    apply_causal_normalization,
    fit_causal_normalization,
)
from pre_experiments.common.contracts import atomic_write_json
from pre_experiments.local_global_consistency.metrics import (
    spearman_correlation,
)


EFFECT_KEYS = {
    "translation": "translation_effect",
    "rotation": "rotation_effect_deg",
    "fov": "fov_effect",
}


def aggregate_scene_effects(
    scene_effects: Sequence[Mapping[str, object]],
    *,
    require_complete_basis: bool = False,
) -> dict[str, object]:
    """Aggregate every scene with equal weight."""
    if not scene_effects:
        raise ValueError("at least one scene is required")
    scene_names = [str(scene.get("scene", "")) for scene in scene_effects]
    if any(not name for name in scene_names) or len(set(scene_names)) != len(
        scene_names
    ):
        raise ValueError("scene names must be non-empty and unique")

    first_activation = np.asarray(
        scene_effects[0]["activation_scale"],
        dtype=np.float64,
    )
    if first_activation.ndim != 2 or min(first_activation.shape) < 1:
        raise ValueError("activation_scale must have shape [iteration, unit]")
    shape = first_activation.shape
    activations = []
    per_group: dict[str, list[np.ndarray]] = {group: [] for group in GROUPS}
    basis_shape: tuple[int, int] | None = None
    for scene in scene_effects:
        activation = np.asarray(scene["activation_scale"], dtype=np.float64)
        if (
            activation.shape != shape
            or not np.isfinite(activation).all()
            or np.any(activation <= 0)
        ):
            raise ValueError("all activation scales must share one finite shape")
        activations.append(activation)
        for group, key in EFFECT_KEYS.items():
            effect = np.asarray(scene[key], dtype=np.float64)
            if (
                effect.shape != shape
                or not np.isfinite(effect).all()
                or np.any(effect < 0)
            ):
                raise ValueError("all scene effect arrays must share one finite shape")
            per_group[group].append(effect)

        measured = np.asarray(scene["measured_basis_mask"])
        if measured.dtype != np.bool_ or measured.ndim != 2:
            raise ValueError("measured basis mask must be a boolean matrix")
        if basis_shape is None:
            basis_shape = measured.shape
        elif measured.shape != basis_shape:
            raise ValueError("all measured basis masks must share one shape")
        if measured.shape[0] != shape[0]:
            raise ValueError("measured basis iteration count mismatch")
        if require_complete_basis and (
            measured.shape[1] != 9 or not bool(measured.all())
        ):
            raise ValueError(
                "formal aggregation requires all nine basis dimensions"
            )

    activation_stack = np.stack(activations)
    group_stacks = {
        group: np.stack(values)
        for group, values in per_group.items()
    }
    return {
        "scenes": scene_names,
        "scene_count": len(scene_names),
        "activation_scale_mean": activation_stack.mean(axis=0),
        "activation_scale_std": activation_stack.std(axis=0),
        "effects_mean": {
            group: values.mean(axis=0)
            for group, values in group_stacks.items()
        },
        "effects_std": {
            group: values.std(axis=0)
            for group, values in group_stacks.items()
        },
    }


def freeze_causal_normalization(
    aggregate: Mapping[str, object],
    *,
    split_digest: str,
    calibration_scenes: Sequence[str],
    measurement_config: Mapping[str, object],
    quantile: float = 0.9,
) -> dict[str, object]:
    """Freeze calibration-only group scales and the reference causal atlas."""
    scenes = [str(scene) for scene in calibration_scenes]
    if not split_digest or scenes != list(aggregate.get("scenes", [])):
        raise ValueError("calibration provenance does not match aggregate scenes")
    effects = _group_arrays(aggregate["effects_mean"])
    scales = fit_causal_normalization(effects, quantile=quantile)
    payload: dict[str, object] = {
        "schema_version": 1,
        "method": "camera_hidden_causal_preference",
        "split_digest": split_digest,
        "calibration_scenes": scenes,
        "measurement_config": dict(measurement_config),
        "normalization_quantile": float(quantile),
        "normalization_scales": scales,
        "calibration_effects": {
            group: effects[group].tolist()
            for group in GROUPS
        },
    }
    payload["frozen_digest"] = canonical_digest(payload)
    return payload


def validate_frozen_causal_normalization(
    frozen: Mapping[str, object],
    *,
    split_digest: str,
    calibration_scenes: Sequence[str],
    measurement_config: Mapping[str, object],
) -> dict[str, object]:
    """Authenticate frozen calibration data and bind it to this invocation."""
    try:
        value = json.loads(json.dumps(frozen))
    except (TypeError, ValueError) as error:
        raise ValueError("invalid frozen causal normalization") from error
    if not isinstance(value, dict):
        raise ValueError("invalid frozen causal normalization")
    digest = value.pop("frozen_digest", None)
    if (
        not isinstance(digest, str)
        or digest != canonical_digest(value)
        or value.get("schema_version") != 1
        or value.get("method") != "camera_hidden_causal_preference"
        or value.get("split_digest") != split_digest
        or value.get("calibration_scenes")
        != [str(scene) for scene in calibration_scenes]
        or value.get("measurement_config") != dict(measurement_config)
    ):
        raise ValueError("frozen causal normalization provenance mismatch")
    effects = _group_arrays(value.get("calibration_effects"))
    scales = value.get("normalization_scales")
    if not isinstance(scales, dict):
        raise ValueError("frozen causal normalization provenance mismatch")
    apply_causal_normalization(effects, scales)
    expected_iterations = int(measurement_config.get("num_iterations", 0))
    if effects[GROUPS[0]].shape[0] != expected_iterations:
        raise ValueError("frozen causal normalization provenance mismatch")
    value["frozen_digest"] = digest
    return value


def write_causal_numeric_summary(
    run_dir: Path,
    scene_effects: Sequence[Mapping[str, object]],
    *,
    output_weight: np.ndarray,
    partition: str,
    frozen: Mapping[str, object],
) -> dict[str, object]:
    """Write the strict numeric atlas and calibration/holdout comparison."""
    if partition not in {"calibration", "holdout"}:
        raise ValueError("partition must be calibration or holdout")
    aggregate = aggregate_scene_effects(
        scene_effects,
        require_complete_basis=True,
    )
    unsigned = dict(frozen)
    digest = unsigned.pop("frozen_digest", None)
    if not isinstance(digest, str) or digest != canonical_digest(unsigned):
        raise ValueError("invalid frozen causal normalization digest")
    effects_mean = _group_arrays(aggregate["effects_mean"])
    effects_std = _group_arrays(aggregate["effects_std"])
    scales = frozen.get("normalization_scales")
    if not isinstance(scales, dict):
        raise ValueError("frozen normalization scales are missing")
    normalized = apply_causal_normalization(effects_mean, scales)

    weight = np.asarray(output_weight, dtype=np.float64)
    iterations, hidden_dim = effects_mean[GROUPS[0]].shape
    if (
        weight.shape != (9, hidden_dim)
        or not np.isfinite(weight).all()
    ):
        raise ValueError("output_weight must be finite with shape [9, hidden]")
    structural = group_specificity(weight)
    activation_mean = np.asarray(
        aggregate["activation_scale_mean"],
        dtype=np.float64,
    )
    activation_std = np.asarray(
        aggregate["activation_scale_std"],
        dtype=np.float64,
    )

    run_dir.mkdir(parents=True, exist_ok=True)
    position_fieldnames = [
        "partition",
        "iteration",
        "unit",
        "scene_count",
        "activation_scale_mean",
        "activation_scale_std",
    ]
    for group in GROUPS:
        position_fieldnames.extend(
            (
                f"{group}_effect_mean",
                f"{group}_effect_std",
                f"normalized_{group}_effect",
                f"{group}_preference",
                f"structural_{group}_specificity",
            )
        )
    position_fieldnames.append("preferred_group")
    with (run_dir / "per_position.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=position_fieldnames)
        writer.writeheader()
        for iteration in range(iterations):
            for unit in range(hidden_dim):
                row: dict[str, object] = {
                    "partition": partition,
                    "iteration": iteration,
                    "unit": unit,
                    "scene_count": aggregate["scene_count"],
                    "activation_scale_mean": activation_mean[iteration, unit],
                    "activation_scale_std": activation_std[iteration, unit],
                    "preferred_group": normalized["preferred_group"][
                        iteration, unit
                    ],
                }
                for group in GROUPS:
                    row.update(
                        {
                            f"{group}_effect_mean": effects_mean[group][
                                iteration, unit
                            ],
                            f"{group}_effect_std": effects_std[group][
                                iteration, unit
                            ],
                            f"normalized_{group}_effect": normalized[
                                "normalized"
                            ][group][iteration, unit],
                            f"{group}_preference": normalized["preferences"][
                                group
                            ][iteration, unit],
                            f"structural_{group}_specificity": structural[
                                group
                            ][unit],
                        }
                    )
                writer.writerow(row)

    direct_rows = _direct_check_rows(scene_effects)
    direct_fieldnames = [
        "scene",
        "iteration",
        "unit",
    ]
    for group in GROUPS:
        direct_fieldnames.extend(
            (
                f"projected_{group}",
                f"measured_{group}",
                f"{group}_relative_error",
            )
        )
    with (run_dir / "direct_checks.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=direct_fieldnames)
        writer.writeheader()
        writer.writerows(direct_rows)

    preferred_counts = Counter(
        np.asarray(normalized["preferred_group"]).reshape(-1).tolist()
    )
    calibration_comparison = {}
    preferred_agreement = None
    if partition == "holdout":
        calibration_effects = _group_arrays(
            frozen.get("calibration_effects")
        )
        calibration_normalized = apply_causal_normalization(
            calibration_effects,
            scales,
        )
        preferred_agreement = float(
            np.mean(
                calibration_normalized["preferred_group"]
                == normalized["preferred_group"]
            )
        )
        for group in GROUPS:
            calibration_values = calibration_effects[group].reshape(-1)
            partition_values = effects_mean[group].reshape(-1)
            top_k = min(64, len(calibration_values))
            calibration_top = set(
                np.argsort(
                    -calibration_values,
                    kind="mergesort",
                )[:top_k].tolist()
            )
            partition_top = set(
                np.argsort(
                    -partition_values,
                    kind="mergesort",
                )[:top_k].tolist()
            )
            calibration_comparison[group] = {
                "spearman": spearman_correlation(
                    calibration_values,
                    partition_values,
                ),
                "top_k": top_k,
                "top_k_overlap": len(calibration_top & partition_top),
                "top_k_overlap_fraction": (
                    len(calibration_top & partition_top) / top_k
                ),
            }

    direct_error_summary = {}
    for group in GROUPS:
        values = np.asarray(
            [float(row[f"{group}_relative_error"]) for row in direct_rows],
            dtype=np.float64,
        )
        direct_error_summary[group] = {
            "count": int(len(values)),
            "mean": float(values.mean()) if len(values) else None,
            "median": float(np.median(values)) if len(values) else None,
            "max": float(values.max()) if len(values) else None,
        }
    summary: dict[str, object] = {
        "partition": partition,
        "scene_count": aggregate["scene_count"],
        "aggregation_unit": "scene",
        "normalization_scales": dict(scales),
        "preferred_group_counts": {
            group: int(preferred_counts.get(group, 0))
            for group in GROUPS
        },
        "direct_projection_relative_error": direct_error_summary,
        "preferred_group_agreement": preferred_agreement,
        "calibration_comparison": calibration_comparison,
    }
    atomic_write_json(run_dir / "summary.json", summary)
    atomic_write_json(
        run_dir / "frozen_causal_normalization.json",
        dict(frozen),
    )
    return summary


def _direct_check_rows(
    scene_effects: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    rows = []
    suffixes = {
        "translation": "translation",
        "rotation": "rotation_deg",
        "fov": "fov",
    }
    for scene in scene_effects:
        iterations = np.asarray(scene["direct_iteration"], dtype=np.int64)
        units = np.asarray(scene["direct_unit"], dtype=np.int64)
        for index, (iteration, unit) in enumerate(zip(iterations, units)):
            row: dict[str, object] = {
                "scene": str(scene["scene"]),
                "iteration": int(iteration),
                "unit": int(unit),
            }
            for group, suffix in suffixes.items():
                projected = float(
                    np.asarray(scene[f"direct_projected_{suffix}"])[index]
                )
                measured = float(
                    np.asarray(scene[f"direct_measured_{suffix}"])[index]
                )
                denominator = max(
                    abs(projected),
                    abs(measured),
                    np.finfo(np.float64).eps,
                )
                row[f"projected_{group}"] = projected
                row[f"measured_{group}"] = measured
                row[f"{group}_relative_error"] = (
                    abs(projected - measured) / denominator
                )
            rows.append(row)
    return rows


def _group_arrays(value: object) -> dict[str, np.ndarray]:
    if not isinstance(value, Mapping) or set(value) != set(GROUPS):
        raise ValueError(f"group arrays must contain exactly {GROUPS}")
    arrays = {
        group: np.asarray(value[group], dtype=np.float64)
        for group in GROUPS
    }
    shape = arrays[GROUPS[0]].shape
    if len(shape) != 2 or min(shape) < 1:
        raise ValueError("group arrays must have shape [iteration, unit]")
    for array in arrays.values():
        if (
            array.shape != shape
            or not np.isfinite(array).all()
            or np.any(array < 0)
        ):
            raise ValueError("group arrays must share one finite non-negative shape")
    return arrays
