"""Scene-level holdout summaries and deterministic bootstrap intervals."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from pre_experiments.local_global_consistency.metrics import (
    pearson_correlation,
    spearman_correlation,
)


SCORE_FIELDS = (
    "global_local_token_cosine",
    "global_local_pose_translation",
    "global_local_pose_rotation_deg",
    "local_local_token_cosine",
    "local_local_pose_translation",
    "local_local_pose_rotation_deg",
)


def _finite_float(value: object, *, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be numeric") from error
    if not np.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _correlations(
    score_rows: Sequence[Mapping[str, object]],
    labels: Mapping[tuple[str, int], Mapping[str, object]],
    *,
    score_field: str,
    growth_field: str,
) -> tuple[float | None, float | None]:
    pairs = []
    for row in score_rows:
        score = row.get(score_field)
        if score is None:
            continue
        identity = (str(row["scene"]), int(row["frame_id"]))
        label = labels[identity]
        pairs.append(
            (
                _finite_float(score, field=score_field),
                _finite_float(label[growth_field], field=growth_field),
            )
        )
    if not pairs:
        return None, None
    score_values = np.asarray([pair[0] for pair in pairs], dtype=np.float64)
    growth_values = np.asarray([pair[1] for pair in pairs], dtype=np.float64)
    return (
        pearson_correlation(score_values, growth_values),
        spearman_correlation(score_values, growth_values),
    )


def summarize_holdout_scenes(
    score_rows: Sequence[Mapping[str, object]],
    validation_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Reduce frame-level prediction scores and GT labels to independent scenes."""
    if not score_rows or not validation_rows:
        raise ValueError("holdout summaries require score and validation rows")
    labels: dict[tuple[str, int], Mapping[str, object]] = {}
    for row in validation_rows:
        identity = (str(row["scene"]), int(row["frame_id"]))
        if identity in labels:
            raise ValueError(f"duplicate validation identity: {identity}")
        labels[identity] = row
    score_identities = [(str(row["scene"]), int(row["frame_id"])) for row in score_rows]
    if len(score_identities) != len(set(score_identities)):
        raise ValueError("prediction scores contain duplicate scene/frame identities")
    if set(score_identities) != set(labels):
        raise ValueError("prediction and validation scene/frame identities differ")

    summaries: list[dict[str, object]] = []
    for scene in sorted({identity[0] for identity in score_identities}):
        scene_scores = [row for row in score_rows if str(row["scene"]) == scene]
        scene_labels = [labels[(scene, int(row["frame_id"]))] for row in scene_scores]
        translation = np.asarray(
            [
                _finite_float(
                    row["translation_error_growth_global_minus_local"],
                    field="translation_error_growth_global_minus_local",
                )
                for row in scene_labels
            ],
            dtype=np.float64,
        )
        rotation = np.asarray(
            [
                _finite_float(
                    row["rotation_error_growth_global_minus_local_deg"],
                    field="rotation_error_growth_global_minus_local_deg",
                )
                for row in scene_labels
            ],
            dtype=np.float64,
        )
        row: dict[str, object] = {
            "scene": scene,
            "frame_count": len(scene_scores),
            "translation_growth_mean": float(np.mean(translation)),
            "translation_growth_median": float(np.median(translation)),
            "translation_growth_positive_fraction": float(np.mean(translation > 0)),
            "rotation_growth_mean": float(np.mean(rotation)),
            "rotation_growth_median": float(np.median(rotation)),
            "rotation_growth_positive_fraction": float(np.mean(rotation > 0)),
        }
        for prefix, field in (
            ("token", "token_local_reliable"),
            ("pose", "pose_local_reliable"),
        ):
            evaluable = [item[field] for item in scene_scores if item.get(field) is not None]
            if not evaluable or not all(isinstance(value, bool) for value in evaluable):
                raise ValueError(f"{field} must contain at least one boolean value")
            row[f"{prefix}_reliable_coverage"] = float(np.mean(evaluable))
            row[f"{prefix}_reliability_evaluable_fraction"] = float(
                len(evaluable) / len(scene_scores)
            )

        for score_field in SCORE_FIELDS:
            for growth_name, growth_field in (
                ("translation_growth", "translation_error_growth_global_minus_local"),
                ("rotation_growth", "rotation_error_growth_global_minus_local_deg"),
            ):
                pearson, spearman = _correlations(
                    scene_scores,
                    labels,
                    score_field=score_field,
                    growth_field=growth_field,
                )
                row[f"{score_field}_vs_{growth_name}_pearson"] = pearson
                row[f"{score_field}_vs_{growth_name}_spearman"] = spearman
        summaries.append(row)
    return summaries


def bootstrap_holdout(
    scene_rows: Sequence[Mapping[str, object]],
    *,
    samples: int = 10_000,
    seed: int = 33,
) -> list[dict[str, object]]:
    """Bootstrap scene-level estimates; frames are never sampled independently."""
    if len(scene_rows) < 2:
        raise ValueError("bootstrap requires at least two scene summaries")
    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    scenes = [str(row.get("scene")) for row in scene_rows]
    if len(set(scenes)) != len(scenes):
        raise ValueError("scene summaries must contain unique scene IDs")

    metric_names: list[str] = []
    excluded = {"scene", "frame_count"}
    for row in scene_rows:
        for name, value in row.items():
            if name in excluded or name in metric_names or isinstance(value, bool):
                continue
            if value is None or isinstance(value, (int, float, np.number)):
                metric_names.append(name)

    rng = np.random.default_rng(seed)
    summaries: list[dict[str, object]] = []
    for metric in metric_names:
        values = np.asarray(
            [
                float(row[metric])
                for row in scene_rows
                if row.get(metric) is not None and np.isfinite(float(row[metric]))
            ],
            dtype=np.float64,
        )
        if len(values) < 2:
            continue
        indices = rng.integers(0, len(values), size=(samples, len(values)))
        estimates = np.mean(values[indices], axis=1)
        estimate = float(np.mean(values))
        ci_low, ci_high = np.percentile(estimates, [2.5, 97.5])
        summaries.append(
            {
                "metric": metric,
                "estimate": estimate,
                "ci95_low": float(ci_low),
                "ci95_high": float(ci_high),
                "contributing_scene_count": len(values),
                "bootstrap_samples": samples,
                "bootstrap_seed": seed,
                "bootstrap_unit": "scene",
                "estimator": "mean_of_scene_statistics",
            }
        )
    if not summaries:
        raise ValueError("no finite scene-level metrics are available for bootstrap")
    return summaries
