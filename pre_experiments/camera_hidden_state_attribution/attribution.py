"""Prediction-only hidden-unit contribution and ranking utilities."""

from __future__ import annotations

import numpy as np


GROUP_SLICES = {
    "translation": slice(0, 3),
    "rotation": slice(3, 7),
    "fov": slice(7, 9),
}


def group_weight_norms(weight: np.ndarray) -> dict[str, np.ndarray]:
    """Return the output-group L2 weight norm for every hidden unit."""
    array = np.asarray(weight, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] != 9:
        raise ValueError("pose output weight must have shape [9, hidden_dim]")
    return {
        name: np.linalg.norm(array[indices], axis=0)
        for name, indices in GROUP_SLICES.items()
    }


def group_specificity(weight: np.ndarray) -> dict[str, np.ndarray]:
    norms = group_weight_norms(weight)
    total = sum(norms.values())
    denominator = np.maximum(total, np.finfo(np.float64).eps)
    return {name: values / denominator for name, values in norms.items()}


def contribution_drift(
    global_hidden: np.ndarray,
    local_hidden: np.ndarray,
    weight: np.ndarray,
) -> dict[str, np.ndarray]:
    """Average matched-frame contribution drift with shape [iteration, hidden]."""
    global_array = np.asarray(global_hidden, dtype=np.float64)
    local_array = np.asarray(local_hidden, dtype=np.float64)
    if global_array.shape != local_array.shape or global_array.ndim != 3:
        raise ValueError("matched hidden arrays must share shape [iteration, sample, hidden]")
    activation_drift = np.abs(local_array - global_array).mean(axis=1)
    norms = group_weight_norms(weight)
    return {
        name: activation_drift * values[None, :]
        for name, values in norms.items()
    }


def freeze_unit_sets(
    scene_statistics: list[dict[str, object]],
    *,
    top_k: int = 64,
    seed: int = 33,
) -> dict[str, object]:
    """Freeze scene-equal unit rankings and iteration-matched random controls."""
    if not scene_statistics:
        raise ValueError("at least one calibration scene is required")
    if top_k < 1:
        raise ValueError("top_k must be positive")

    first_drift = scene_statistics[0]["drift"]
    if not isinstance(first_drift, dict):
        raise ValueError("scene drift must be a group dictionary")
    iterations, hidden_dim = np.asarray(first_drift["translation"]).shape
    frozen: dict[str, object] = {
        "top_k": min(top_k, iterations * hidden_dim),
        "seed": seed,
        "selected": {},
        "controls": {},
        "scores": {},
    }
    rng = np.random.default_rng(seed)

    for group in GROUP_SLICES:
        per_scene = []
        for scene in scene_statistics:
            drift = scene["drift"]
            specificity = scene["specificity"]
            if not isinstance(drift, dict) or not isinstance(specificity, dict):
                raise ValueError("invalid scene statistic dictionaries")
            values = np.asarray(drift[group], dtype=np.float64)
            group_specificity_values = np.asarray(
                specificity[group], dtype=np.float64
            )
            if values.shape != (iterations, hidden_dim):
                raise ValueError("all scene drift arrays must share one shape")
            if group_specificity_values.shape != (hidden_dim,):
                raise ValueError("specificity must have shape [hidden_dim]")
            per_scene.append(values * group_specificity_values[None, :])
        scores = np.mean(np.stack(per_scene), axis=0)
        candidates = [
            (float(scores[iteration, unit]), iteration, unit)
            for iteration in range(iterations)
            for unit in range(hidden_dim)
        ]
        candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
        selected_tuples = [
            (iteration, unit)
            for _, iteration, unit in candidates[: int(frozen["top_k"])]
        ]
        selected = [
            {"iteration": iteration, "unit": unit}
            for iteration, unit in selected_tuples
        ]

        controls: list[dict[str, int]] = []
        selected_set = set(selected_tuples)
        for iteration in range(iterations):
            count = sum(item_iteration == iteration for item_iteration, _ in selected_tuples)
            available = [
                unit
                for unit in range(hidden_dim)
                if (iteration, unit) not in selected_set
            ]
            if len(available) < count:
                raise ValueError("not enough unselected units for matched controls")
            chosen = rng.choice(available, size=count, replace=False)
            controls.extend(
                {"iteration": iteration, "unit": int(unit)}
                for unit in sorted(chosen.tolist())
            )

        frozen["selected"][group] = selected
        frozen["controls"][group] = controls
        frozen["scores"][group] = [
            {
                "iteration": iteration,
                "unit": unit,
                "score": score,
            }
            for score, iteration, unit in candidates
        ]

    return frozen
