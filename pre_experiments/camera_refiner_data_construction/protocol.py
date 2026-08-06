"""Frozen multiscale candidate and frame-matching protocol."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from pre_experiments.camera_hidden_state_attribution.replacement import (
    assemble_short_hidden,
)
from pre_experiments.local_global_consistency.windows import (
    build_sliding_windows,
)


LOCAL_SCALES = (100, 200, 300)
DEFAULT_ALPHAS = (0.01, 0.02, 0.05, 0.10)
PURE_BETAS = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)
MIXTURE_BETAS = (
    (0.5, 0.5, 0.0),
    (0.0, 0.5, 0.5),
    (0.2, 0.3, 0.5),
)


def _validated_beta(beta: Sequence[float]) -> tuple[float, float, float]:
    try:
        values = tuple(float(value) for value in beta)
    except (TypeError, ValueError) as error:
        raise ValueError("beta must contain three finite weights") from error
    if (
        len(values) != len(LOCAL_SCALES)
        or not np.isfinite(values).all()
        or any(value < 0.0 for value in values)
        or not np.isclose(sum(values), 1.0, atol=1e-8, rtol=0.0)
    ):
        raise ValueError("beta must be a non-negative three-weight simplex")
    return values  # type: ignore[return-value]


def _number_label(value: float) -> str:
    return f"{value:.8g}".replace(".", "p")


@dataclass(frozen=True)
class Candidate:
    """One bounded hidden interpolation candidate over 100/200/300 scales."""

    alpha: float
    beta: tuple[float, float, float]

    def __post_init__(self) -> None:
        alpha = float(self.alpha)
        if not np.isfinite(alpha) or not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be finite and in (0, 1]")
        beta = _validated_beta(self.beta)
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "beta", beta)

    @property
    def name(self) -> str:
        beta_label = "_".join(_number_label(value) for value in self.beta)
        return f"a{_number_label(self.alpha)}_b{beta_label}"


def _candidates(
    betas: Sequence[tuple[float, float, float]],
    alphas: Sequence[float] = DEFAULT_ALPHAS,
) -> tuple[Candidate, ...]:
    candidates = tuple(
        Candidate(alpha=alpha, beta=beta)
        for beta in betas
        for alpha in alphas
    )
    if len({candidate.name for candidate in candidates}) != len(candidates):
        raise ValueError("candidate identities must be unique")
    return candidates


def default_pure_candidates() -> tuple[Candidate, ...]:
    """Return the frozen single-scale calibration grid."""
    return _candidates(PURE_BETAS)


def default_mixture_candidates() -> tuple[Candidate, ...]:
    """Return the predeclared multiscale calibration grid."""
    return _candidates(MIXTURE_BETAS)


def assemble_multiscale_hidden(
    global_frame_ids: np.ndarray,
    scale_windows: Mapping[int, Sequence[Mapping[str, object]]],
) -> dict[str, np.ndarray]:
    """Assemble exact local hidden arrays as `[scale, iteration, frame, hidden]`."""
    frame_ids = np.asarray(global_frame_ids)
    if set(scale_windows) != set(LOCAL_SCALES):
        raise ValueError(f"scale windows must contain exactly scales {LOCAL_SCALES}")

    assembled_by_scale = []
    starts_by_scale = []
    stops_by_scale = []
    for scale in LOCAL_SCALES:
        expected = build_sliding_windows(
            frame_ids,
            length=scale,
            stride=scale // 2,
        )
        records = sorted(
            scale_windows[scale],
            key=lambda item: int(item["window_index"]),
        )
        if len(records) != len(expected):
            raise ValueError(f"scale {scale} windows do not match canonical count")
        normalized_records = []
        bounds = {}
        for record, window in zip(records, expected):
            try:
                index = int(record["window_index"])
                local_ids = np.asarray(record["frame_ids"], dtype=np.int64)
                hidden = np.asarray(record["hidden"], dtype=np.float32)
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"scale {scale} window is invalid") from error
            if index != window.index or not np.array_equal(
                local_ids,
                np.asarray(window.frame_ids, dtype=np.int64),
            ):
                raise ValueError(f"scale {scale} windows are not canonical")
            normalized_records.append(
                {
                    "window_index": index,
                    "frame_ids": local_ids,
                    "hidden": hidden,
                }
            )
            bounds[index] = (window.start, window.stop)

        assembled = assemble_short_hidden(frame_ids, normalized_records)
        selected = assembled["selected_window_index"]
        starts_by_scale.append(
            np.asarray([bounds[int(index)][0] for index in selected], dtype=np.int64)
        )
        stops_by_scale.append(
            np.asarray([bounds[int(index)][1] for index in selected], dtype=np.int64)
        )
        assembled_by_scale.append(assembled)

    shapes = {item["hidden"].shape for item in assembled_by_scale}
    if len(shapes) != 1:
        raise ValueError("all scales must share iteration, frame, and hidden dimensions")
    return {
        "scales": np.asarray(LOCAL_SCALES, dtype=np.int64),
        "hidden": np.stack([item["hidden"] for item in assembled_by_scale]),
        "selected_window_index": np.stack(
            [item["selected_window_index"] for item in assembled_by_scale]
        ),
        "selected_boundary_distance": np.stack(
            [item["selected_boundary_distance"] for item in assembled_by_scale]
        ),
        "observation_count": np.stack(
            [item["observation_count"] for item in assembled_by_scale]
        ),
        "selected_window_start": np.stack(starts_by_scale),
        "selected_window_stop": np.stack(stops_by_scale),
    }


def mix_local_hidden(
    local_hidden: np.ndarray,
    beta: Sequence[float],
) -> np.ndarray:
    """Mix `[3, iteration, frame, hidden]` local states over the scale axis."""
    hidden = np.asarray(local_hidden, dtype=np.float32)
    if hidden.ndim != 4 or hidden.shape[0] != len(LOCAL_SCALES):
        raise ValueError("local_hidden must have shape [3, iteration, frame, hidden]")
    if not np.isfinite(hidden).all():
        raise ValueError("local_hidden must contain only finite values")
    weights = np.asarray(_validated_beta(beta), dtype=np.float32)
    return np.tensordot(weights, hidden, axes=(0, 0)).astype(np.float32)
