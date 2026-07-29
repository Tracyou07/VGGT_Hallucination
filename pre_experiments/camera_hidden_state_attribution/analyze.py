"""Intervention metrics and numeric aggregation for hidden-state attribution."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from pre_experiments.common.contracts import atomic_write_json
from pre_experiments.common.pose_metrics import rotation_angle_deg


METRICS = (
    "camera_center_displacement_mean",
    "rotation_change_deg_mean",
    "fov_change_mean",
    "aligned_translation_error_mean",
    "aligned_translation_error_delta",
)


def unit_mask(
    frozen: dict[str, object],
    group: str,
    set_name: str,
    iterations: int,
    hidden_dim: int,
) -> np.ndarray:
    mask = np.zeros((iterations, hidden_dim), dtype=bool)
    collection = frozen.get(set_name)
    if not isinstance(collection, dict) or group not in collection:
        raise ValueError(f"missing frozen {set_name}/{group} units")
    for item in collection[group]:
        iteration = int(item["iteration"])
        unit = int(item["unit"])
        if not (0 <= iteration < iterations and 0 <= unit < hidden_dim):
            raise ValueError("frozen unit index is out of range")
        mask[iteration, unit] = True
    return mask


def intervention_metrics(
    baseline_c2w: np.ndarray,
    changed_c2w: np.ndarray,
    baseline_pose_enc: np.ndarray,
    changed_pose_enc: np.ndarray,
) -> dict[str, float]:
    baseline = np.asarray(baseline_c2w, dtype=np.float64)
    changed = np.asarray(changed_c2w, dtype=np.float64)
    if baseline.shape != changed.shape or baseline.ndim != 3:
        raise ValueError("pose trajectories must share shape [S, 4, 4]")
    center_change = np.linalg.norm(
        changed[:, :3, 3] - baseline[:, :3, 3], axis=1
    )
    rotation_change = np.asarray(
        [
            rotation_angle_deg(left[:3, :3].T @ right[:3, :3])
            for left, right in zip(baseline, changed)
        ]
    )
    baseline_encoding = np.asarray(baseline_pose_enc, dtype=np.float64)
    changed_encoding = np.asarray(changed_pose_enc, dtype=np.float64)
    fov_change = np.linalg.norm(
        changed_encoding[:, 7:9] - baseline_encoding[:, 7:9], axis=1
    )
    return {
        "camera_center_displacement_mean": float(center_change.mean()),
        "rotation_change_deg_mean": float(rotation_change.mean()),
        "fov_change_mean": float(fov_change.mean()),
    }


def _bootstrap(values: np.ndarray, seed: int = 33) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(10000, len(values)))
    means = values[indices].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def write_numeric_summary(
    run_dir: Path,
    frozen: dict[str, object],
    rows: list[dict[str, object]],
    *,
    partition: str,
    scene_statistics: list[dict[str, object]] | None = None,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = ["scene", "group", "set", *METRICS]
    with (run_dir / "per_scene.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    partition_scores: dict[str, np.ndarray] = {}
    edge_scores: dict[str, np.ndarray] = {}
    interior_scores: dict[str, np.ndarray] = {}
    if scene_statistics:
        for group in ("translation", "rotation", "fov"):
            partition_scores[group] = np.mean(
                np.stack(
                    [
                        np.asarray(scene["drift"][group])
                        * np.asarray(scene["specificity"][group])[None, :]
                        for scene in scene_statistics
                    ]
                ),
                axis=0,
            )
            for target, stratum in (
                (edge_scores, "edge"),
                (interior_scores, "interior"),
            ):
                target[group] = np.mean(
                    np.stack(
                        [
                            np.asarray(scene["boundary_drift"][stratum][group])
                            * np.asarray(scene["specificity"][group])[None, :]
                            for scene in scene_statistics
                        ]
                    ),
                    axis=0,
                )

    unit_rows = []
    ranking_overlap: dict[str, dict[str, object]] = {}
    scores = frozen.get("scores", {})
    if isinstance(scores, dict):
        for group, entries in scores.items():
            selected_pairs = {
                (int(item["iteration"]), int(item["unit"]))
                for item in frozen["selected"][group]
            }
            control_pairs = {
                (int(item["iteration"]), int(item["unit"]))
                for item in frozen["controls"][group]
            }
            partition_rank = {}
            if group in partition_scores:
                candidates = [
                    (float(partition_scores[group][iteration, unit]), iteration, unit)
                    for iteration in range(partition_scores[group].shape[0])
                    for unit in range(partition_scores[group].shape[1])
                ]
                candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
                partition_rank = {
                    (iteration, unit): rank
                    for rank, (_, iteration, unit) in enumerate(candidates, start=1)
                }
                top = {
                    (iteration, unit)
                    for _, iteration, unit in candidates[: len(selected_pairs)]
                }
                ranking_overlap[group] = {
                    "top_k": len(selected_pairs),
                    "overlap_count": len(top.intersection(selected_pairs)),
                    "overlap_fraction": (
                        len(top.intersection(selected_pairs)) / len(selected_pairs)
                        if selected_pairs
                        else 0.0
                    ),
                }
            for rank, item in enumerate(entries, start=1):
                pair = (int(item["iteration"]), int(item["unit"]))
                unit_rows.append(
                    {
                        "group": group,
                        "calibration_rank": rank,
                        "iteration": item["iteration"],
                        "unit": item["unit"],
                        "calibration_score": item["score"],
                        "partition_rank": partition_rank.get(pair, ""),
                        "partition_score": (
                            float(partition_scores[group][pair])
                            if group in partition_scores
                            else ""
                        ),
                        "edge_score": (
                            float(edge_scores[group][pair])
                            if group in edge_scores
                            else ""
                        ),
                        "interior_score": (
                            float(interior_scores[group][pair])
                            if group in interior_scores
                            else ""
                        ),
                        "selected": pair in selected_pairs,
                        "control": pair in control_pairs,
                    }
                )
    with (run_dir / "per_unit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "group",
                "calibration_rank",
                "iteration",
                "unit",
                "calibration_score",
                "partition_rank",
                "partition_score",
                "edge_score",
                "interior_score",
                "selected",
                "control",
            ),
        )
        writer.writeheader()
        writer.writerows(unit_rows)

    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((str(row["group"]), str(row["set"])), []).append(row)
    aggregates = []
    for (group, set_name), group_rows in sorted(grouped.items()):
        aggregate: dict[str, object] = {
            "group": group,
            "set": set_name,
            "scene_count": len(group_rows),
        }
        for metric in METRICS:
            values = np.asarray([float(row[metric]) for row in group_rows])
            low, high = _bootstrap(values)
            aggregate[metric] = {
                "estimate": float(values.mean()),
                "ci95_low": low,
                "ci95_high": high,
            }
        aggregates.append(aggregate)
    atomic_write_json(
        run_dir / "summary.json",
        {
            "partition": partition,
            "bootstrap_unit": "scene",
            "bootstrap_samples": 10000,
            "bootstrap_seed": 33,
            "frozen_top_k_overlap": ranking_overlap,
            "aggregates": aggregates,
        },
    )
    atomic_write_json(run_dir / "frozen_units.json", frozen)
