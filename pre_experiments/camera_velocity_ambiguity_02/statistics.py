"""Primary-only scene aggregation and deterministic paired scene bootstrap."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class BootstrapDifference:
    mean_difference: float
    ci_low: float
    ci_high: float
    seed: int
    samples: int
    scene_count: int


def aggregate_scene_prevalence(
    rows: Sequence[Mapping[str, object]], *, positive_field: str
) -> dict[str, float]:
    grouped: dict[str, list[bool]] = {}
    for row in rows:
        if row.get("route") != "primary":
            continue
        scene = row.get("scene")
        value = row.get(positive_field)
        if not isinstance(scene, str) or not isinstance(value, (bool, np.bool_)):
            raise ValueError("primary prevalence rows require scene and boolean outcome")
        grouped.setdefault(scene, []).append(bool(value))
    if not grouped:
        raise ValueError("no primary rows available for scene prevalence")
    return {
        scene: float(np.mean(values))
        for scene, values in sorted(grouped.items())
    }


def paired_scene_bootstrap(
    left: Mapping[str, float],
    right: Mapping[str, float],
    *,
    seed: int = 33,
    samples: int = 10_000,
) -> BootstrapDifference:
    if set(left) != set(right) or len(left) < 2:
        raise ValueError("paired bootstrap requires matching scene sets")
    if samples < 1:
        raise ValueError("bootstrap samples must be positive")
    scenes = sorted(left)
    differences = np.asarray([float(left[scene]) - float(right[scene]) for scene in scenes])
    if not np.isfinite(differences).all():
        raise ValueError("bootstrap values must be finite")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(scenes), size=(samples, len(scenes)))
    distribution = differences[indices].mean(axis=1)
    return BootstrapDifference(
        mean_difference=float(np.mean(differences)),
        ci_low=float(np.quantile(distribution, 0.025)),
        ci_high=float(np.quantile(distribution, 0.975)),
        seed=seed,
        samples=samples,
        scene_count=len(scenes),
    )
