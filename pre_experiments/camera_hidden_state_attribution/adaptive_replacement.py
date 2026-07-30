"""Numeric summaries for scene-adaptive hidden interpolation."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence

import numpy as np

from pre_experiments.camera_hidden_state_attribution.replacement_analyze import (
    _estimate,
)


def summarize_adaptive_rows(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Pair adaptive selected/control deltas within each independent scene."""
    scenes = sorted({str(row.get("scene", "")) for row in rows})
    if not scenes or scenes[0] == "":
        raise ValueError("adaptive rows require scene identities")
    selected_deltas = []
    control_deltas = []
    selected_minus_control = []
    alpha_counts: Counter[str] = Counter()
    control_counts = set()
    for scene in scenes:
        scene_rows = [
            row for row in rows if str(row.get("scene", "")) == scene
        ]
        baseline = [
            row
            for row in scene_rows
            if row.get("condition_family") == "baseline"
        ]
        selected = [
            row
            for row in scene_rows
            if row.get("condition_family") == "selected"
        ]
        controls = [
            row
            for row in scene_rows
            if row.get("condition_family") == "control"
        ]
        if len(baseline) != 1 or len(selected) != 1 or not controls:
            raise ValueError(f"incomplete adaptive conditions for {scene}")
        alpha = float(selected[0]["alpha"])
        if (
            not np.isfinite(alpha)
            or alpha <= 0
            or any(float(row["alpha"]) != alpha for row in controls)
        ):
            raise ValueError(f"adaptive alpha mismatch for {scene}")
        control_names = [str(row["condition"]) for row in controls]
        if len(control_names) != len(set(control_names)):
            raise ValueError(f"duplicate adaptive controls for {scene}")
        control_counts.add(len(controls))
        alpha_counts[f"{alpha:.8g}"] += 1
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
        if not np.isfinite(selected_delta) or not np.isfinite(control_delta):
            raise ValueError("adaptive deltas must be finite")
        selected_deltas.append(selected_delta)
        control_deltas.append(control_delta)
        selected_minus_control.append(selected_delta - control_delta)
    if len(control_counts) != 1:
        raise ValueError("adaptive control count differs across scenes")
    return {
        "scene_count": len(scenes),
        "alpha_scene_counts": dict(sorted(alpha_counts.items())),
        "evaluated_control_repeats": control_counts.pop(),
        "selected_delta": _estimate(selected_deltas),
        "control_mean_delta": _estimate(control_deltas),
        "selected_minus_control": _estimate(selected_minus_control),
        "selected_improved_scene_fraction": float(
            np.mean(np.asarray(selected_deltas) < 0)
        ),
        "selected_beat_control_scene_fraction": float(
            np.mean(np.asarray(selected_minus_control) < 0)
        ),
    }


def compare_adaptive_to_fixed(
    adaptive_rows: Sequence[Mapping[str, object]],
    fixed_rows: Sequence[Mapping[str, object]],
    *,
    fixed_alpha: float = 0.02,
) -> dict[str, object]:
    """Compare adaptive and preregistered fixed-alpha deltas by scene."""
    adaptive = {
        str(row["scene"]): float(row["aligned_translation_error_delta"])
        for row in adaptive_rows
        if row.get("condition_family") == "selected"
    }
    fixed = {
        str(row["scene"]): float(row["aligned_translation_error_delta"])
        for row in fixed_rows
        if row.get("condition_family") == "selected"
        and float(row["alpha"]) == fixed_alpha
    }
    if not adaptive or set(adaptive) != set(fixed):
        raise ValueError("adaptive and fixed scene sets differ")
    differences = [
        adaptive[scene] - fixed[scene] for scene in sorted(adaptive)
    ]
    if not np.isfinite(differences).all():
        raise ValueError("adaptive-fixed deltas must be finite")
    return {
        "fixed_alpha": fixed_alpha,
        "scene_count": len(differences),
        "adaptive_minus_fixed": _estimate(differences),
        "adaptive_beat_fixed_scene_fraction": float(
            np.mean(np.asarray(differences) < 0)
        ),
    }
