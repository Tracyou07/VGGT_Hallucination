"""Scene-equal calibration and frozen holdout analysis."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
from pathlib import Path

import numpy as np

from pre_experiments.camera_hidden_state_attribution.artifacts import (
    canonical_digest,
)
from pre_experiments.common.contracts import atomic_write_json


def _bootstrap_mean_ci(
    values: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) < 1 or not np.isfinite(array).all():
        raise ValueError("bootstrap values must be a non-empty finite vector")
    if samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(samples, len(array)))
    means = array[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def _validated_candidate_arrays(
    shard: Mapping[str, object],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    names = np.asarray(shard["candidate_names"])
    alpha = np.asarray(shard["candidate_alpha"], dtype=np.float64)
    beta = np.asarray(shard["candidate_beta"], dtype=np.float64)
    translation = np.asarray(shard["translation_error_aligned"], dtype=np.float64)
    rotation = np.asarray(shard["rotation_error_deg_aligned"], dtype=np.float64)
    fov = np.asarray(shard["fov_change_mean"], dtype=np.float64)
    hidden = np.asarray(shard["hidden_displacement_rms"], dtype=np.float64)
    candidate_count = len(names) if names.ndim == 1 else -1
    if (
        candidate_count < 2
        or names.dtype.kind not in "US"
        or names[0] != "baseline"
        or len(np.unique(names)) != candidate_count
        or alpha.shape != (candidate_count,)
        or beta.shape != (candidate_count, 3)
        or translation.ndim != 2
        or translation.shape[0] != candidate_count
        or rotation.shape != translation.shape
        or fov.shape != (candidate_count,)
        or hidden.shape != (candidate_count,)
        or not all(
            np.isfinite(item).all()
            for item in (alpha, beta, translation, rotation, fov, hidden)
        )
        or np.any(translation < 0.0)
        or np.any(rotation < 0.0)
        or np.any(fov < 0.0)
        or np.any(hidden < 0.0)
    ):
        raise ValueError("scene candidate arrays are invalid")
    return names.astype(str), alpha, beta, translation, rotation, fov, hidden


def _policy_candidate(frozen: Mapping[str, object]) -> tuple[str, float, np.ndarray]:
    value = dict(frozen)
    digest = value.pop("frozen_digest", None)
    if not isinstance(digest, str) or digest != canonical_digest(value):
        raise ValueError("frozen policy digest is invalid")
    name = value.get("selected_candidate")
    try:
        alpha = float(value["selected_alpha"])
        beta = np.asarray(value["selected_beta"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("frozen candidate is invalid") from error
    if (
        not isinstance(name, str)
        or not name
        or not np.isfinite(alpha)
        or not 0.0 < alpha <= 1.0
        or beta.shape != (3,)
        or not np.isfinite(beta).all()
        or np.any(beta < 0.0)
        or not np.isclose(beta.sum(), 1.0, atol=1e-8, rtol=0.0)
    ):
        raise ValueError("frozen candidate is invalid")
    return name, alpha, beta


def summarize_scene_shards(
    shards: Sequence[Mapping[str, object]],
    *,
    partition: str,
    frozen_policy: Mapping[str, object] | None = None,
    bootstrap_samples: int = 10000,
    seed: int = 33,
) -> dict[str, object]:
    """Reduce frame metrics within scenes before scene-equal aggregation."""
    if partition not in {"calibration", "holdout"} or not shards:
        raise ValueError("partition and at least one scene shard are required")
    scenes = [str(shard.get("scene", "")) for shard in shards]
    if any(not scene for scene in scenes) or len(set(scenes)) != len(scenes):
        raise ValueError("scene shards must have unique identities")

    parsed = [_validated_candidate_arrays(shard) for shard in shards]
    reference_names, reference_alpha, reference_beta = parsed[0][:3]
    for names, alpha, beta, *_ in parsed[1:]:
        if (
            not np.array_equal(names, reference_names)
            or not np.array_equal(alpha, reference_alpha)
            or not np.array_equal(beta, reference_beta)
        ):
            raise ValueError("scene shards do not share candidate identities")

    if partition == "holdout":
        if frozen_policy is None:
            raise ValueError("holdout requires a frozen policy")
        selected_name, selected_alpha, selected_beta = _policy_candidate(frozen_policy)
        if (
            reference_names.tolist() != ["baseline", selected_name]
            or not np.isclose(reference_alpha[1], selected_alpha, atol=0.0, rtol=0.0)
            or not np.allclose(reference_beta[1], selected_beta, atol=1e-12, rtol=0.0)
        ):
            raise ValueError("holdout shards must contain exactly baseline and frozen candidate")

    rows = []
    for candidate_index in range(1, len(reference_names)):
        translation_delta = []
        rotation_delta = []
        fov_change = []
        hidden_displacement = []
        improved_frame_fraction = []
        for _, _, _, translation, rotation, fov, hidden in parsed:
            translation_delta.append(
                float(translation[candidate_index].mean() - translation[0].mean())
            )
            rotation_delta.append(
                float(rotation[candidate_index].mean() - rotation[0].mean())
            )
            fov_change.append(float(fov[candidate_index]))
            hidden_displacement.append(float(hidden[candidate_index]))
            improved_frame_fraction.append(
                float(np.mean(translation[candidate_index] < translation[0]))
            )
        translation_array = np.asarray(translation_delta)
        rotation_array = np.asarray(rotation_delta)
        low, high = _bootstrap_mean_ci(
            translation_array,
            samples=bootstrap_samples,
            seed=seed,
        )
        if len(translation_array) == 1:
            leave_one_out_max = float(translation_array[0])
        else:
            leave_one_out_max = max(
                float(np.delete(translation_array, index).mean())
                for index in range(len(translation_array))
            )
        beta = reference_beta[candidate_index]
        rows.append(
            {
                "candidate": str(reference_names[candidate_index]),
                "alpha": float(reference_alpha[candidate_index]),
                "beta100": float(beta[0]),
                "beta200": float(beta[1]),
                "beta300": float(beta[2]),
                "scene_count": len(scenes),
                "translation_delta_mean": float(translation_array.mean()),
                "translation_delta_median": float(np.median(translation_array)),
                "translation_delta_ci95_low": low,
                "translation_delta_ci95_high": high,
                "leave_one_out_max_mean": leave_one_out_max,
                "improved_scene_fraction": float(np.mean(translation_array < 0.0)),
                "improved_frame_fraction_scene_mean": float(
                    np.mean(improved_frame_fraction)
                ),
                "rotation_delta_deg_mean": float(rotation_array.mean()),
                "fov_change_mean": float(np.mean(fov_change)),
                "hidden_displacement_rms_mean": float(
                    np.mean(hidden_displacement)
                ),
            }
        )
    return {
        "schema_version": 1,
        "partition": partition,
        "scenes": scenes,
        "scene_count": len(scenes),
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": seed,
        "candidate_rows": rows,
    }


def freeze_candidate_policy(
    summary: Mapping[str, object],
    *,
    split_digest: str,
    calibration_scenes: Sequence[str],
    source_run_id: str,
    scale_run_ids: Mapping[int, str],
    max_rotation_delta_deg: float = 0.05,
    max_fov_change: float = 0.01,
    min_improved_scene_fraction: float = 0.5,
) -> dict[str, object]:
    """Freeze the best robust and safety-compatible calibration candidate."""
    scenes = [str(scene) for scene in calibration_scenes]
    rows = summary.get("candidate_rows")
    if (
        summary.get("partition") != "calibration"
        or summary.get("scenes") != scenes
        or not isinstance(rows, Sequence)
        or not rows
        or not split_digest
        or not source_run_id
        or set(scale_run_ids) != {100, 200, 300}
        or not np.isfinite(max_rotation_delta_deg)
        or max_rotation_delta_deg < 0.0
        or not np.isfinite(max_fov_change)
        or max_fov_change < 0.0
        or not 0.0 <= min_improved_scene_fraction <= 1.0
    ):
        raise ValueError("calibration summary or freeze provenance is invalid")

    eligible = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("candidate summary row is invalid")
        if (
            float(row["translation_delta_mean"]) < 0.0
            and float(row["translation_delta_ci95_high"]) <= 0.0
            and float(row["leave_one_out_max_mean"]) < 0.0
            and float(row["improved_scene_fraction"])
            >= min_improved_scene_fraction
            and float(row["rotation_delta_deg_mean"])
            <= max_rotation_delta_deg
            and float(row["fov_change_mean"]) <= max_fov_change
        ):
            eligible.append(row)
    if not eligible:
        raise ValueError("no calibration candidate satisfies robustness and safety gates")
    selected = min(
        eligible,
        key=lambda row: (
            float(row["translation_delta_mean"]),
            float(row["rotation_delta_deg_mean"]),
            float(row["fov_change_mean"]),
            str(row["candidate"]),
        ),
    )
    frozen: dict[str, object] = {
        "schema_version": 1,
        "method": "multiscale_camera_hidden_interpolation",
        "selection_partition": "calibration",
        "selection_metric": "scene_mean_aligned_translation_error_delta",
        "split_digest": split_digest,
        "calibration_scenes": scenes,
        "source_run_id": source_run_id,
        "scale_run_ids": {
            str(scale): str(scale_run_ids[scale]) for scale in (100, 200, 300)
        },
        "safety_limits": {
            "max_rotation_delta_deg": float(max_rotation_delta_deg),
            "max_fov_change": float(max_fov_change),
            "min_improved_scene_fraction": float(min_improved_scene_fraction),
            "require_nonpositive_translation_ci95_high": True,
            "require_negative_leave_one_out_max_mean": True,
        },
        "selected_candidate": str(selected["candidate"]),
        "selected_alpha": float(selected["alpha"]),
        "selected_beta": [
            float(selected["beta100"]),
            float(selected["beta200"]),
            float(selected["beta300"]),
        ],
        "selected_calibration_metrics": dict(selected),
    }
    frozen["frozen_digest"] = canonical_digest(frozen)
    return frozen


def validate_frozen_policy(
    frozen: Mapping[str, object],
    *,
    split_digest: str,
    calibration_scenes: Sequence[str],
    source_run_id: str,
    scale_run_ids: Mapping[int, str],
) -> dict[str, object]:
    """Authenticate a calibration policy before any holdout replay."""
    _policy_candidate(frozen)
    expected_scale_runs = {
        str(scale): str(scale_run_ids[scale]) for scale in (100, 200, 300)
    }
    if (
        frozen.get("method") != "multiscale_camera_hidden_interpolation"
        or frozen.get("selection_partition") != "calibration"
        or frozen.get("split_digest") != split_digest
        or frozen.get("calibration_scenes")
        != [str(scene) for scene in calibration_scenes]
        or frozen.get("source_run_id") != source_run_id
        or frozen.get("scale_run_ids") != expected_scale_runs
    ):
        raise ValueError("frozen policy provenance mismatch")
    return dict(frozen)


def write_numeric_summary(path: Path, summary: Mapping[str, object]) -> None:
    """Write only scene-reduced scalar candidate evidence."""
    rows = summary.get("candidate_rows")
    if not isinstance(rows, Sequence) or not rows:
        raise ValueError("candidate summary rows are required")
    destination = Path(path)
    destination.mkdir(parents=True, exist_ok=True)
    atomic_write_json(destination / "candidate_summary.json", dict(summary))
    fieldnames = list(rows[0].keys())
    temporary = destination / "candidate_summary.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            if not isinstance(row, Mapping) or list(row.keys()) != fieldnames:
                raise ValueError("candidate summary rows do not share a schema")
            writer.writerow(row)
    temporary.replace(destination / "candidate_summary.csv")
