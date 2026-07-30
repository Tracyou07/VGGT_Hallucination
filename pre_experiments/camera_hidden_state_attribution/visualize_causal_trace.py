"""Visualize calibration-frozen causal tracing evidence."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np


GROUPS = ("translation", "rotation", "fov")
COLORS = {
    "translation": "#2563EB",
    "rotation": "#D97706",
    "fov": "#15803D",
    "ink": "#172033",
    "muted": "#64748B",
    "line": "#CBD5E1",
    "surface": "#F8FAFC",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as error:
        raise ValueError(f"cannot read causal-trace input: {path}") from error
    if not rows:
        raise ValueError(f"causal-trace input is empty: {path}")
    return rows


def _position(row: Mapping[str, str]) -> tuple[int, int]:
    try:
        position = (int(row["iteration"]), int(row["unit"]))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid (iteration, unit) identity") from error
    if min(position) < 0:
        raise ValueError("iteration and unit must be non-negative")
    return position


def _causal_index(
    rows: Sequence[Mapping[str, str]],
) -> dict[tuple[int, int], Mapping[str, str]]:
    index = {_position(row): row for row in rows}
    if len(index) != len(rows):
        raise ValueError("causal rows contain duplicate positions")
    return index


def _old_top(
    rows: Sequence[Mapping[str, str]],
    group: str,
    top_k: int,
) -> set[tuple[int, int]]:
    selected = [row for row in rows if row.get("group") == group]
    try:
        selected.sort(key=lambda row: int(row["partition_rank"]))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid {group} partition rank") from error
    positions = [_position(row) for row in selected]
    if len(set(positions)) != len(positions) or len(positions) < top_k:
        raise ValueError(f"invalid {group} old-attribution positions")
    return set(positions[:top_k])


def _causal_top(
    rows: Sequence[Mapping[str, str]],
    group: str,
    top_k: int,
) -> set[tuple[int, int]]:
    field = f"{group}_effect_mean"
    ranked = []
    for row in rows:
        try:
            value = float(row[field])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid causal field {field}") from error
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"invalid causal field {field}")
        ranked.append((value, _position(row)))
    if len({position for _, position in ranked}) != len(ranked):
        raise ValueError("causal rows contain duplicate positions")
    ranked.sort(key=lambda item: (-item[0], item[1]))
    if len(ranked) < top_k:
        raise ValueError("top_k exceeds the causal position count")
    return {position for _, position in ranked[:top_k]}


def _iteration_counts(
    positions: set[tuple[int, int]],
) -> dict[str, int]:
    counts = Counter(iteration for iteration, _ in positions)
    return {str(iteration): counts[iteration] for iteration in sorted(counts)}


def build_trace_summary(
    old_calibration_rows: Sequence[Mapping[str, str]],
    old_holdout_rows: Sequence[Mapping[str, str]],
    causal_calibration_rows: Sequence[Mapping[str, str]],
    causal_holdout_rows: Sequence[Mapping[str, str]],
    *,
    top_k: int = 64,
) -> dict[str, object]:
    """Freeze calibration intersections and validate them on holdout."""
    if top_k < 1:
        raise ValueError("top_k must be positive")
    calibration_index = _causal_index(causal_calibration_rows)
    holdout_index = _causal_index(causal_holdout_rows)
    if set(calibration_index) != set(holdout_index):
        raise ValueError("calibration and holdout causal positions differ")
    if top_k > len(calibration_index):
        raise ValueError("top_k exceeds the causal position count")

    summaries = {}
    for group in GROUPS:
        old_calibration_top = _old_top(
            old_calibration_rows,
            group,
            top_k,
        )
        old_holdout_top = _old_top(old_holdout_rows, group, top_k)
        causal_calibration_top = _causal_top(
            causal_calibration_rows,
            group,
            top_k,
        )
        causal_holdout_top = _causal_top(
            causal_holdout_rows,
            group,
            top_k,
        )
        frozen = old_calibration_top & causal_calibration_top
        holdout_joint = old_holdout_top & causal_holdout_top
        recovered = frozen & old_holdout_top & causal_holdout_top
        stable_preference = {
            position
            for position in frozen
            if calibration_index[position].get("preferred_group") == group
            and holdout_index[position].get("preferred_group") == group
        }
        summaries[group] = {
            "calibration_overlap": len(frozen),
            "holdout_overlap": len(holdout_joint),
            "frozen_candidate_count": len(frozen),
            "frozen_recovered_count": len(recovered),
            "stable_preference_count": len(stable_preference),
            "calibration_causal_iteration_counts": _iteration_counts(
                causal_calibration_top
            ),
            "holdout_causal_iteration_counts": _iteration_counts(
                causal_holdout_top
            ),
            "frozen_positions": [
                {"iteration": iteration, "unit": unit}
                for iteration, unit in sorted(frozen)
            ],
        }
    return {
        "top_k": top_k,
        "universe_size": len(calibration_index),
        "groups": summaries,
    }


def _method_box(
    axis: plt.Axes,
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    detail: str,
    color: str,
) -> None:
    box = Rectangle(
        (x, y),
        width,
        height,
        linewidth=0.9,
        edgecolor=color,
        facecolor="#FAFAFA",
        transform=axis.transAxes,
    )
    axis.add_patch(box)
    axis.text(
        x + 0.025,
        y + height * 0.66,
        title,
        transform=axis.transAxes,
        fontsize=8.4,
        fontweight="bold",
        color=COLORS["ink"],
        va="center",
    )
    axis.text(
        x + 0.025,
        y + height * 0.31,
        detail,
        transform=axis.transAxes,
        fontsize=7.2,
        color=COLORS["muted"],
        va="center",
    )


def render_trace_overview(
    summary: Mapping[str, object],
    output: Path,
) -> Path:
    """Render a compact paper-style figure without changing numeric evidence."""
    top_k = int(summary["top_k"])
    universe_size = int(summary["universe_size"])
    groups = summary["groups"]
    if not isinstance(groups, Mapping) or set(groups) != set(GROUPS):
        raise ValueError("summary has invalid group data")

    rc = {
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "axes.linewidth": 0.7,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
    }
    with plt.rc_context(rc):
        figure = plt.figure(figsize=(12.2, 4.25), facecolor="white")
        grid = figure.add_gridspec(
            1,
            3,
            width_ratios=(1.42, 1.0, 1.0),
            left=0.035,
            right=0.985,
            top=0.89,
            bottom=0.19,
            wspace=0.34,
        )
        method = figure.add_subplot(grid[0, 0])
        overlap = figure.add_subplot(grid[0, 1])
        recovery = figure.add_subplot(grid[0, 2])

        method.set_axis_off()
        method.text(
            -0.02,
            1.03,
            "(a)",
            transform=method.transAxes,
            fontsize=10,
            fontweight="bold",
        )
        method.text(
            0.07,
            1.03,
            "Calibration-frozen causal tracing",
            transform=method.transAxes,
            fontsize=9,
            fontweight="bold",
        )
        _method_box(
            method,
            x=0.02,
            y=0.71,
            width=0.31,
            height=0.13,
            title="Short context",
            detail="matched targets",
            color=COLORS["muted"],
        )
        _method_box(
            method,
            x=0.02,
            y=0.49,
            width=0.31,
            height=0.13,
            title="Long context",
            detail="same targets",
            color=COLORS["muted"],
        )
        _method_box(
            method,
            x=0.51,
            y=0.62,
            width=0.46,
            height=0.15,
            title=r"Hidden drift  $\Delta h_{i,u}$",
            detail=f"rank all {universe_size} positions",
            color=COLORS["translation"],
        )
        _method_box(
            method,
            x=0.51,
            y=0.37,
            width=0.46,
            height=0.15,
            title=r"End-to-end effect  $E_{i,u}$",
            detail=r"$J_i W_{:,u}\,\sigma_{i,u}$",
            color=COLORS["rotation"],
        )
        _method_box(
            method,
            x=0.51,
            y=0.12,
            width=0.46,
            height=0.15,
            title=rf"Frozen intersection  $C_g$",
            detail=rf"$\mathrm{{Top}}_{{{top_k}}}(D_g)\cap"
            rf"\mathrm{{Top}}_{{{top_k}}}(E_g)$",
            color=COLORS["fov"],
        )
        method.annotate(
            "",
            xy=(0.50, 0.695),
            xytext=(0.34, 0.775),
            xycoords=method.transAxes,
            arrowprops={
                "arrowstyle": "-|>",
                "color": COLORS["line"],
                "linewidth": 1.0,
            },
        )
        method.annotate(
            "",
            xy=(0.50, 0.695),
            xytext=(0.34, 0.555),
            xycoords=method.transAxes,
            arrowprops={
                "arrowstyle": "-|>",
                "color": COLORS["line"],
                "linewidth": 1.0,
            },
        )
        for start_y, end_y in ((0.62, 0.52), (0.37, 0.27)):
            method.annotate(
                "",
                xy=(0.74, end_y),
                xytext=(0.74, start_y),
                xycoords=method.transAxes,
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": COLORS["line"],
                    "linewidth": 1.0,
                },
            )
        method.text(
            0.74,
            0.045,
            "local-to-global replacement (next)",
            transform=method.transAxes,
            ha="center",
            va="center",
            fontsize=7.2,
            color=COLORS["muted"],
        )
        method.annotate(
            "",
            xy=(0.74, 0.065),
            xytext=(0.74, 0.12),
            xycoords=method.transAxes,
            arrowprops={
                "arrowstyle": "-|>",
                "color": COLORS["line"],
                "linewidth": 1.0,
                "linestyle": "--",
            },
        )

        overlap.text(
            -0.18,
            1.03,
            "(b)",
            transform=overlap.transAxes,
            fontsize=10,
            fontweight="bold",
        )
        overlap.set_title(
            f"Cross-experiment top-{top_k} overlap",
            loc="left",
            pad=8,
            fontweight="bold",
        )
        names = ("Translation", "Rotation", "FoV")
        y = np.arange(len(GROUPS))[::-1]
        calibration_values = np.asarray(
            [
                int(groups[group]["calibration_overlap"]) / top_k
                for group in GROUPS
            ]
        )
        holdout_values = np.asarray(
            [
                int(groups[group]["holdout_overlap"]) / top_k
                for group in GROUPS
            ]
        )
        for index, group in enumerate(GROUPS):
            axis_y = y[index]
            overlap.plot(
                [calibration_values[index], holdout_values[index]],
                [axis_y + 0.09, axis_y - 0.09],
                color=COLORS[group],
                linewidth=1.1,
                alpha=0.55,
                zorder=1,
            )
            overlap.scatter(
                calibration_values[index],
                axis_y + 0.09,
                s=38,
                marker="o",
                facecolor="white",
                edgecolor=COLORS[group],
                linewidth=1.1,
                zorder=3,
            )
            overlap.scatter(
                holdout_values[index],
                axis_y - 0.09,
                s=34,
                marker="s",
                facecolor=COLORS[group],
                edgecolor=COLORS[group],
                linewidth=0.8,
                zorder=3,
            )
            overlap.text(
                calibration_values[index] + 0.018,
                axis_y + 0.09,
                f'{int(groups[group]["calibration_overlap"])}/{top_k}',
                va="center",
                fontsize=7,
                color=COLORS["ink"],
            )
            overlap.text(
                holdout_values[index] + 0.018,
                axis_y - 0.09,
                f'{int(groups[group]["holdout_overlap"])}/{top_k}',
                va="center",
                fontsize=7,
                color=COLORS["ink"],
            )
        overlap.set_yticks(y, names)
        overlap.set_xlim(0.0, 0.76)
        overlap.set_xticks(
            np.linspace(0, 0.75, 4),
            [f"{value:.0%}" for value in np.linspace(0, 0.75, 4)],
        )
        overlap.set_xlabel("Overlap fraction")
        overlap.grid(axis="x", color="#E5E7EB", linewidth=0.6)
        overlap.spines[["top", "right", "left"]].set_visible(False)
        overlap.tick_params(axis="y", length=0)
        overlap.legend(
            handles=[
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    markerfacecolor="white",
                    markeredgecolor=COLORS["muted"],
                    linewidth=0,
                    label="Calibration",
                ),
                Line2D(
                    [0],
                    [0],
                    marker="s",
                    markerfacecolor=COLORS["muted"],
                    markeredgecolor=COLORS["muted"],
                    linewidth=0,
                    label="Holdout",
                ),
            ],
            frameon=False,
            fontsize=7,
            loc="lower right",
        )

        recovery.text(
            -0.18,
            1.03,
            "(c)",
            transform=recovery.transAxes,
            fontsize=10,
            fontweight="bold",
        )
        recovery.set_title(
            "Frozen-set recovery on holdout",
            loc="left",
            pad=8,
            fontweight="bold",
        )
        recovered = np.asarray(
            [
                int(groups[group]["frozen_recovered_count"])
                for group in GROUPS
            ]
        )
        frozen = np.asarray(
            [
                int(groups[group]["frozen_candidate_count"])
                for group in GROUPS
            ]
        )
        fractions = np.divide(
            recovered,
            frozen,
            out=np.zeros(3, dtype=np.float64),
            where=frozen > 0,
        )
        bars = recovery.barh(
            y,
            fractions,
            height=0.44,
            color=[COLORS[group] for group in GROUPS],
            alpha=0.88,
        )
        recovery.set_yticks(y, names)
        recovery.set_xlim(0.0, 1.08)
        recovery.set_xticks(
            np.linspace(0, 1, 5),
            [f"{value:.0%}" for value in np.linspace(0, 1, 5)],
        )
        recovery.set_xlabel("Recovered in both holdout top-K lists")
        recovery.grid(axis="x", color="#E5E7EB", linewidth=0.6)
        recovery.spines[["top", "right", "left"]].set_visible(False)
        recovery.tick_params(axis="y", length=0)
        for bar, numerator, denominator in zip(bars, recovered, frozen):
            recovery.text(
                min(bar.get_width() + 0.025, 1.025),
                bar.get_y() + bar.get_height() / 2,
                f"{numerator}/{denominator}",
                va="center",
                fontsize=7.2,
                fontweight="bold",
                color=COLORS["ink"],
            )

        figure.text(
            0.035,
            0.065,
            (
                "All end-to-end causal top-64 positions occur at refinement "
                "iteration 0. Rotation direct validation remains pending."
            ),
            fontsize=7.5,
            color=COLORS["ink"],
        )
        figure.text(
            0.985,
            0.065,
            "Overlap supports candidate mediation, not GT-error causation.",
            ha="right",
            fontsize=7.3,
            color=COLORS["muted"],
        )

        output = output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(
            output,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close(figure)
        return output


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-calibration", type=Path, required=True)
    parser.add_argument("--old-holdout", type=Path, required=True)
    parser.add_argument("--causal-calibration", type=Path, required=True)
    parser.add_argument("--causal-holdout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=64)
    args = parser.parse_args(argv)
    summary = build_trace_summary(
        _read_csv(args.old_calibration.resolve()),
        _read_csv(args.old_holdout.resolve()),
        _read_csv(args.causal_calibration.resolve()),
        _read_csv(args.causal_holdout.resolve()),
        top_k=args.top_k,
    )
    output = render_trace_overview(summary, args.output)
    print(f"[plot] {output}")


if __name__ == "__main__":
    main()
