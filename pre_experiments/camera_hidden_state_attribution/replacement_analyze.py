"""Numeric summaries for short-to-long hidden replacement."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import csv
from pathlib import Path

import numpy as np

from pre_experiments.camera_hidden_state_attribution.analyze import (
    intervention_metrics,
)
from pre_experiments.common.contracts import atomic_write_json


SCENE_METRICS = (
    "aligned_translation_error_mean",
    "aligned_translation_error_median",
    "aligned_translation_ate_rmse",
    "aligned_rotation_error_deg_mean",
    "aligned_rotation_error_deg_median",
    "aligned_translation_error_delta",
    "aligned_rotation_error_deg_delta",
    "camera_center_displacement_mean",
    "rotation_change_deg_mean",
    "fov_change_mean",
)


def _estimate(values: Sequence[float], *, seed: int = 33) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) < 1 or not np.isfinite(array).all():
        raise ValueError("summary values must be a non-empty finite vector")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(10000, len(array)))
    means = array[indices].mean(axis=1)
    low, high = np.percentile(means, (2.5, 97.5))
    return {
        "estimate": float(array.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
    }


def summarize_replacement_rows(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Build scene-paired selected/control tests for every alpha."""
    by_scene: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        scene = str(row.get("scene", ""))
        if not scene:
            raise ValueError("replacement row has no scene")
        by_scene[scene].append(row)
    if not by_scene:
        raise ValueError("at least one replacement row is required")

    alphas = sorted(
        {
            float(row["alpha"])
            for row in rows
            if row.get("condition_family") == "selected"
        }
    )
    if not alphas or any(
        not np.isfinite(alpha) or not 0.0 < alpha <= 1.0
        for alpha in alphas
    ):
        raise ValueError("replacement rows contain no valid alpha")
    alpha_tests = []
    for alpha in alphas:
        selected_deltas = []
        control_deltas = []
        selected_minus_control = []
        for scene, scene_rows in sorted(by_scene.items()):
            baseline = [
                row
                for row in scene_rows
                if row.get("condition_family") == "baseline"
            ]
            selected = [
                row
                for row in scene_rows
                if row.get("condition_family") == "selected"
                and float(row["alpha"]) == alpha
            ]
            controls = [
                row
                for row in scene_rows
                if row.get("condition_family") == "control"
                and float(row["alpha"]) == alpha
            ]
            if len(baseline) != 1 or len(selected) != 1 or not controls:
                raise ValueError(
                    f"incomplete alpha={alpha} conditions for {scene}"
                )
            selected_delta = float(
                selected[0]["aligned_translation_error_delta"]
            )
            control_delta = float(
                np.mean(
                    [
                        float(row["aligned_translation_error_delta"])
                        for row in controls
                    ]
                )
            )
            if not np.isfinite(selected_delta) or not np.isfinite(
                control_delta
            ):
                raise ValueError("replacement deltas must be finite")
            selected_deltas.append(selected_delta)
            control_deltas.append(control_delta)
            selected_minus_control.append(selected_delta - control_delta)
        selected_array = np.asarray(selected_deltas)
        comparison_array = np.asarray(selected_minus_control)
        alpha_tests.append(
            {
                "alpha": alpha,
                "selected_delta": _estimate(selected_deltas),
                "control_mean_delta": _estimate(control_deltas),
                "selected_minus_control": _estimate(
                    selected_minus_control
                ),
                "selected_improved_scene_fraction": float(
                    np.mean(selected_array < 0)
                ),
                "selected_beat_control_scene_fraction": float(
                    np.mean(comparison_array < 0)
                ),
            }
        )
    condition_aggregates = []
    conditions = sorted(
        {str(row["condition"]) for row in rows},
        key=lambda value: (
            value != "baseline",
            value != "selected",
            value,
        ),
    )
    for condition in conditions:
        condition_rows = [
            row for row in rows if row["condition"] == condition
        ]
        aggregate: dict[str, object] = {
            "condition": condition,
            "scene_count": len(condition_rows),
        }
        for metric in SCENE_METRICS:
            if all(metric in row for row in condition_rows):
                aggregate[metric] = _estimate(
                    [float(row[metric]) for row in condition_rows]
                )
        condition_aggregates.append(aggregate)

    return {
        "scene_count": len(by_scene),
        "bootstrap_unit": "scene",
        "bootstrap_samples": 10000,
        "bootstrap_seed": 33,
        "alpha_tests": alpha_tests,
        "condition_aggregates": condition_aggregates,
    }


def select_calibration_alpha(
    summary: Mapping[str, object],
) -> dict[str, float | str]:
    """Freeze the alpha with the lowest calibration translation delta."""
    tests = summary.get("alpha_tests")
    if not isinstance(tests, Sequence) or not tests:
        raise ValueError("calibration summary has no alpha tests")
    candidates = []
    for item in tests:
        if not isinstance(item, Mapping):
            raise ValueError("invalid calibration alpha test")
        try:
            alpha = float(item["alpha"])
            estimate = float(item["selected_delta"]["estimate"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("invalid calibration alpha test") from error
        if not np.isfinite(alpha) or not np.isfinite(estimate):
            raise ValueError("invalid calibration alpha test")
        candidates.append((estimate, alpha))
    estimate, alpha = min(candidates, key=lambda item: (item[0], item[1]))
    return {
        "selected_alpha": alpha,
        "alpha_selection_metric": (
            "minimum_scene_mean_aligned_translation_error_delta"
        ),
        "calibration_selected_delta": estimate,
    }


def build_replacement_rows(
    result: Mapping[str, object],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Convert one strict scene artifact into scene and frame CSV rows."""
    scene = str(result.get("scene", ""))
    names = np.asarray(result["condition_names"]).astype(str)
    families = np.asarray(result["condition_family"]).astype(str)
    alphas = np.asarray(result["condition_alpha"], dtype=np.float64)
    counts = np.asarray(result["replacement_count"], dtype=np.int64)
    frame_ids = np.asarray(result["frame_ids"], dtype=np.int64)
    predictions = np.asarray(result["pred_c2w_raw"], dtype=np.float64)
    pose_enc = np.asarray(result["pose_enc"], dtype=np.float64)
    translation = np.asarray(
        result["translation_error_aligned"],
        dtype=np.float64,
    )
    rotation = np.asarray(
        result["rotation_error_deg_aligned"],
        dtype=np.float64,
    )
    if not scene or names[0] != "baseline":
        raise ValueError("invalid replacement scene identity")

    baseline_translation = float(translation[0].mean())
    baseline_rotation = float(rotation[0].mean())
    rows = []
    frame_rows = []
    for index, condition in enumerate(names):
        metrics = {
            "aligned_translation_error_mean": float(
                translation[index].mean()
            ),
            "aligned_translation_error_median": float(
                np.median(translation[index])
            ),
            "aligned_translation_ate_rmse": float(
                np.sqrt(np.mean(np.square(translation[index])))
            ),
            "aligned_rotation_error_deg_mean": float(rotation[index].mean()),
            "aligned_rotation_error_deg_median": float(
                np.median(rotation[index])
            ),
        }
        if index == 0:
            changes = {
                "camera_center_displacement_mean": 0.0,
                "rotation_change_deg_mean": 0.0,
                "fov_change_mean": 0.0,
            }
        else:
            changes = intervention_metrics(
                predictions[0],
                predictions[index],
                pose_enc[0],
                pose_enc[index],
            )
        rows.append(
            {
                "scene": scene,
                "condition": condition,
                "condition_family": families[index],
                "alpha": float(alphas[index]),
                "replacement_count": int(counts[index]),
                **metrics,
                "aligned_translation_error_delta": (
                    metrics["aligned_translation_error_mean"]
                    - baseline_translation
                ),
                "aligned_rotation_error_deg_delta": (
                    metrics["aligned_rotation_error_deg_mean"]
                    - baseline_rotation
                ),
                **changes,
            }
        )
        for frame_index, frame_id in enumerate(frame_ids):
            frame_rows.append(
                {
                    "scene": scene,
                    "condition": condition,
                    "condition_family": families[index],
                    "alpha": float(alphas[index]),
                    "frame_id": int(frame_id),
                    "sequence_index": frame_index,
                    "selected_local_window_index": int(
                        result["selected_window_index"][frame_index]
                    ),
                    "selected_boundary_distance": int(
                        result["selected_boundary_distance"][frame_index]
                    ),
                    "local_observation_count": int(
                        result["local_observation_count"][frame_index]
                    ),
                    "translation_error_aligned": float(
                        translation[index, frame_index]
                    ),
                    "rotation_error_deg_aligned": float(
                        rotation[index, frame_index]
                    ),
                }
            )
    return rows, frame_rows


def write_replacement_numeric_summary(
    run_dir: Path,
    results: Sequence[Mapping[str, object]],
    frozen: Mapping[str, object],
    *,
    partition: str,
) -> dict[str, object]:
    """Write strict scene/frame tables and the scene-paired summary."""
    if partition not in {"calibration", "holdout"}:
        raise ValueError("partition must be calibration or holdout")
    scene_rows = []
    frame_rows = []
    for result in results:
        result_scene_rows, result_frame_rows = build_replacement_rows(result)
        scene_rows.extend(result_scene_rows)
        frame_rows.extend(result_frame_rows)
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "per_scene.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "scene",
                "condition",
                "condition_family",
                "alpha",
                "replacement_count",
                *SCENE_METRICS,
            ),
        )
        writer.writeheader()
        writer.writerows(scene_rows)
    with (run_dir / "per_frame.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "scene",
                "condition",
                "condition_family",
                "alpha",
                "frame_id",
                "sequence_index",
                "selected_local_window_index",
                "selected_boundary_distance",
                "local_observation_count",
                "translation_error_aligned",
                "rotation_error_deg_aligned",
            ),
        )
        writer.writeheader()
        writer.writerows(frame_rows)
    summary = {
        "partition": partition,
        "frozen_digest": frozen["frozen_digest"],
        "selected_count": frozen["selected_count"],
        "control_repeats": frozen["control_repeats"],
        **summarize_replacement_rows(scene_rows),
    }
    atomic_write_json(run_dir / "summary.json", summary)
    atomic_write_json(run_dir / "frozen_replacement.json", dict(frozen))
    return summary
