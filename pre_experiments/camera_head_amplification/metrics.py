"""Matched-frame and layerwise amplification metrics."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def _finite_array(name: str, value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def match_frame_indices(short_ids: np.ndarray, long_ids: np.ndarray) -> np.ndarray:
    """Return indices into long_ids in the exact order provided by short_ids."""
    short = _finite_array("short_ids", short_ids).astype(np.int64, copy=False)
    long = _finite_array("long_ids", long_ids).astype(np.int64, copy=False)
    if short.ndim != 1 or long.ndim != 1:
        raise ValueError("frame IDs must be one-dimensional")
    if len(np.unique(short)) != len(short) or len(np.unique(long)) != len(long):
        raise ValueError("frame IDs must be unique")
    lookup = {int(frame_id): index for index, frame_id in enumerate(long)}
    missing = [int(frame_id) for frame_id in short if int(frame_id) not in lookup]
    if missing:
        raise ValueError(f"long context is missing frame IDs: {missing[:5]}")
    return np.asarray([lookup[int(frame_id)] for frame_id in short], dtype=np.int64)


def _vector_drift(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    baseline = _finite_array("baseline", left).astype(np.float64, copy=False)
    perturbed = _finite_array("perturbed", right).astype(np.float64, copy=False)
    if baseline.shape != perturbed.shape or baseline.ndim != 2:
        raise ValueError("compared representations must have matching shape [S, D]")
    return np.linalg.norm(perturbed - baseline, axis=-1)


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values, dtype=np.float64))))


def build_stage_summary_rows(
    *,
    scene: str,
    comparison: str,
    baseline_tokens: np.ndarray,
    perturbed_tokens: np.ndarray,
    baseline_stages: Mapping[str, np.ndarray],
    perturbed_stages: Mapping[str, np.ndarray],
    allow_zero_input: bool = False,
) -> list[dict[str, object]]:
    """Summarize layer drift relative to normalized input-token drift."""
    input_per_frame = _vector_drift(baseline_tokens, perturbed_tokens)
    input_rms = _rms(input_per_frame)
    zero_input = input_rms <= np.finfo(np.float64).eps
    if zero_input and not allow_zero_input:
        raise ValueError("input token drift must be non-zero")
    if list(baseline_stages) != list(perturbed_stages):
        raise ValueError("baseline and perturbed stage names must match")

    rows: list[dict[str, object]] = []
    for stage, baseline in baseline_stages.items():
        perturbed = perturbed_stages[stage]
        if baseline.shape != perturbed.shape or baseline.ndim != 3:
            raise ValueError(f"stage {stage} must have matching shape [I, S, D]")
        for iteration in range(baseline.shape[0]):
            per_frame = _vector_drift(baseline[iteration], perturbed[iteration])
            stage_rms = _rms(per_frame)
            rows.append(
                {
                    "scene": scene,
                    "comparison": comparison,
                    "iteration": iteration + 1,
                    "stage": stage,
                    "feature_dim": int(baseline.shape[-1]),
                    "input_token_rms_drift": input_rms,
                    "stage_rms_drift": stage_rms,
                    "stage_mean_drift": float(np.mean(per_frame)),
                    "stage_median_drift": float(np.median(per_frame)),
                    "stage_p95_drift": float(np.percentile(per_frame, 95)),
                    "amplification_ratio": None if zero_input else stage_rms / input_rms,
                }
            )
    return rows


def build_stage_per_frame_rows(
    *,
    scene: str,
    comparison: str,
    frame_ids: np.ndarray,
    baseline_tokens: np.ndarray,
    perturbed_tokens: np.ndarray,
    baseline_stages: Mapping[str, np.ndarray],
    perturbed_stages: Mapping[str, np.ndarray],
) -> list[dict[str, object]]:
    """Build compact per-frame scalar drift rows without saving activations."""
    ids = np.asarray(frame_ids, dtype=np.int64)
    input_drift = _vector_drift(baseline_tokens, perturbed_tokens)
    if len(ids) != len(input_drift):
        raise ValueError("frame_ids and token arrays must have the same length")
    rows: list[dict[str, object]] = []
    for stage, baseline in baseline_stages.items():
        perturbed = perturbed_stages[stage]
        if baseline.shape != perturbed.shape or baseline.ndim != 3:
            raise ValueError(f"stage {stage} must have matching shape [I, S, D]")
        for iteration in range(baseline.shape[0]):
            stage_drift = _vector_drift(baseline[iteration], perturbed[iteration])
            for frame_id, token_value, stage_value in zip(
                ids, input_drift, stage_drift
            ):
                rows.append(
                    {
                        "scene": scene,
                        "comparison": comparison,
                        "frame_id": int(frame_id),
                        "iteration": iteration + 1,
                        "stage": stage,
                        "input_token_drift": float(token_value),
                        "stage_drift": float(stage_value),
                    }
                )
    return rows


def validate_replay_baseline(
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    atol: float,
    rtol: float,
) -> dict[str, float]:
    """Fail closed unless replayed activated pose encodings match saved output."""
    replayed = _finite_array("actual replay", actual).astype(np.float64, copy=False)
    saved = _finite_array("expected replay", expected).astype(np.float64, copy=False)
    if replayed.shape != saved.shape:
        raise ValueError(
            f"replay baseline shape mismatch: {replayed.shape} != {saved.shape}"
        )
    difference = np.abs(replayed - saved)
    diagnostics = {
        "max_abs_error": float(np.max(difference, initial=0.0)),
        "mean_abs_error": float(np.mean(difference)),
    }
    if not np.allclose(replayed, saved, atol=atol, rtol=rtol):
        raise ValueError(
            "replay baseline mismatch: "
            f"max_abs_error={diagnostics['max_abs_error']:.8g}, "
            f"atol={atol:.8g}, rtol={rtol:.8g}"
        )
    return diagnostics
