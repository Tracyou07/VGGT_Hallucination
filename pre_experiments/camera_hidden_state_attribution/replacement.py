"""Frozen candidate selection and matched short-hidden assembly."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence

import numpy as np

from pre_experiments.camera_hidden_state_attribution.artifacts import (
    canonical_digest,
)


def _position(row: Mapping[str, object]) -> tuple[int, int]:
    try:
        iteration = int(row["iteration"])
        unit = int(row["unit"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid replacement position") from error
    if iteration < 0 or unit < 0:
        raise ValueError("replacement positions must be non-negative")
    return iteration, unit


def freeze_replacement_manifest(
    old_rows: Sequence[Mapping[str, object]],
    causal_rows: Sequence[Mapping[str, object]],
    *,
    split_digest: str,
    calibration_scenes: Sequence[str],
    group: str = "translation",
    source_top_k: int = 64,
    control_repeats: int = 5,
    seed: int = 33,
) -> dict[str, object]:
    """Freeze the calibration intersection and iteration-matched controls."""
    if not split_digest or not calibration_scenes:
        raise ValueError("split provenance is required")
    if source_top_k < 1 or control_repeats < 1:
        raise ValueError("top-k and control repeats must be positive")

    old_group = [row for row in old_rows if row.get("group") == group]
    try:
        old_group.sort(key=lambda row: int(row["partition_rank"]))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid {group} attribution rank") from error
    old_positions = [_position(row) for row in old_group]
    if len(set(old_positions)) != len(old_positions) or len(old_positions) < source_top_k:
        raise ValueError(f"insufficient unique {group} attribution positions")
    old_rank = {
        position: rank
        for rank, position in enumerate(old_positions, start=1)
    }
    old_top = set(old_positions[:source_top_k])

    effect_field = f"{group}_effect_mean"
    causal_ranked = []
    for row in causal_rows:
        position = _position(row)
        try:
            effect = float(row[effect_field])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid causal field {effect_field}") from error
        if not np.isfinite(effect) or effect < 0:
            raise ValueError(f"invalid causal field {effect_field}")
        causal_ranked.append((effect, position))
    if len({position for _, position in causal_ranked}) != len(causal_ranked):
        raise ValueError("causal positions must be unique")
    if len(causal_ranked) < source_top_k:
        raise ValueError("source_top_k exceeds the causal universe")
    causal_ranked.sort(key=lambda item: (-item[0], item[1]))
    causal_rank = {
        position: rank
        for rank, (_, position) in enumerate(causal_ranked, start=1)
    }
    causal_top = {
        position for _, position in causal_ranked[:source_top_k]
    }

    selected_positions = sorted(
        old_top & causal_top,
        key=lambda position: (
            max(old_rank[position], causal_rank[position]),
            old_rank[position] + causal_rank[position],
            position,
        ),
    )
    if not selected_positions:
        raise ValueError("calibration top-k sets have no intersection")

    source_union = old_top | causal_top
    universe = {position for _, position in causal_ranked}
    iteration_counts = Counter(iteration for iteration, _ in selected_positions)
    rng = np.random.default_rng(seed)
    control_sets = [
        {"name": f"control_{index:02d}", "positions": []}
        for index in range(control_repeats)
    ]
    for iteration, count in sorted(iteration_counts.items()):
        available = sorted(
            position
            for position in universe - source_union
            if position[0] == iteration
        )
        required = count * control_repeats
        if len(available) < required:
            raise ValueError(
                f"not enough iteration-{iteration} units for disjoint controls"
            )
        chosen = rng.choice(len(available), size=required, replace=False)
        for control_index in range(control_repeats):
            start = control_index * count
            positions = [
                available[int(index)]
                for index in chosen[start : start + count]
            ]
            control_sets[control_index]["positions"].extend(
                [
                    {"iteration": item[0], "unit": item[1]}
                    for item in sorted(positions)
                ]
            )

    selected = [
        {
            "iteration": iteration,
            "unit": unit,
            "attribution_rank": old_rank[(iteration, unit)],
            "causal_rank": causal_rank[(iteration, unit)],
        }
        for iteration, unit in selected_positions
    ]
    frozen: dict[str, object] = {
        "schema_version": 1,
        "method": "short_to_long_pose_hidden_replacement",
        "group": group,
        "source_top_k": source_top_k,
        "selected_count": len(selected),
        "selection": "calibration_attribution_top_k_intersection_causal_top_k",
        "control_method": (
            "disjoint_seeded_random_outside_source_union_matched_by_iteration"
        ),
        "control_repeats": control_repeats,
        "seed": seed,
        "split_digest": split_digest,
        "calibration_scenes": list(calibration_scenes),
        "selected": selected,
        "control_sets": control_sets,
    }
    frozen["frozen_digest"] = canonical_digest(frozen)
    return frozen


def assemble_short_hidden(
    global_frame_ids: np.ndarray,
    windows: Sequence[Mapping[str, object]],
) -> dict[str, np.ndarray]:
    """Choose the most interior short-window hidden for every global frame."""
    frame_ids = np.asarray(global_frame_ids, dtype=np.int64)
    if (
        frame_ids.ndim != 1
        or len(frame_ids) < 2
        or len(np.unique(frame_ids)) != len(frame_ids)
    ):
        raise ValueError("global_frame_ids must contain unique frame IDs")
    if not windows:
        raise ValueError("at least one short window is required")

    global_index = {
        int(frame_id): index for index, frame_id in enumerate(frame_ids)
    }
    selected_hidden: np.ndarray | None = None
    selected_window = np.full(len(frame_ids), -1, dtype=np.int64)
    selected_boundary = np.full(len(frame_ids), -1, dtype=np.int64)
    observation_count = np.zeros(len(frame_ids), dtype=np.int64)
    seen_window_indices: set[int] = set()
    expected_shape: tuple[int, int] | None = None

    for window in sorted(windows, key=lambda item: int(item["window_index"])):
        window_index = int(window["window_index"])
        if window_index in seen_window_indices:
            raise ValueError("short window indices must be unique")
        seen_window_indices.add(window_index)
        local_ids = np.asarray(window["frame_ids"], dtype=np.int64)
        hidden = np.asarray(window["hidden"], dtype=np.float32)
        if (
            local_ids.ndim != 1
            or len(local_ids) < 2
            or len(np.unique(local_ids)) != len(local_ids)
            or hidden.ndim != 3
            or hidden.shape[1] != len(local_ids)
            or not np.isfinite(hidden).all()
        ):
            raise ValueError("invalid short-window hidden artifact")
        shape = (hidden.shape[0], hidden.shape[2])
        if expected_shape is None:
            expected_shape = shape
            selected_hidden = np.empty(
                (shape[0], len(frame_ids), shape[1]),
                dtype=np.float32,
            )
        elif shape != expected_shape:
            raise ValueError("short windows must share iteration and hidden dimensions")

        for local_index, frame_id_value in enumerate(local_ids):
            try:
                sequence_index = global_index[int(frame_id_value)]
            except KeyError as error:
                raise ValueError(
                    "short window contains a frame absent from global context"
                ) from error
            observation_count[sequence_index] += 1
            boundary = min(local_index, len(local_ids) - 1 - local_index)
            if boundary > selected_boundary[sequence_index]:
                assert selected_hidden is not None
                selected_hidden[:, sequence_index] = hidden[:, local_index]
                selected_boundary[sequence_index] = boundary
                selected_window[sequence_index] = window_index

    missing = frame_ids[selected_window < 0]
    if len(missing):
        raise ValueError(
            f"short windows do not cover global frames: {missing[:5].tolist()}"
        )
    assert selected_hidden is not None
    return {
        "hidden": selected_hidden,
        "selected_window_index": selected_window,
        "selected_boundary_distance": selected_boundary,
        "observation_count": observation_count,
    }


def replacement_mask(
    frozen: Mapping[str, object],
    set_name: str,
    *,
    iterations: int,
    hidden_dim: int,
) -> np.ndarray:
    """Build one strict boolean replacement mask from a frozen manifest."""
    if set_name == "selected":
        positions = frozen.get("selected")
    else:
        controls = frozen.get("control_sets")
        if not isinstance(controls, Sequence):
            raise ValueError("frozen control sets are missing")
        match = [
            item
            for item in controls
            if isinstance(item, Mapping) and item.get("name") == set_name
        ]
        positions = match[0].get("positions") if len(match) == 1 else None
    if not isinstance(positions, Sequence):
        raise ValueError(f"frozen replacement set is missing: {set_name}")

    mask = np.zeros((iterations, hidden_dim), dtype=bool)
    for item in positions:
        if not isinstance(item, Mapping):
            raise ValueError("invalid frozen replacement position")
        iteration, unit = _position(item)
        if iteration >= iterations or unit >= hidden_dim:
            raise ValueError("frozen replacement position is out of range")
        mask[iteration, unit] = True
    if int(mask.sum()) != len(positions):
        raise ValueError("frozen replacement positions contain duplicates")
    return mask
