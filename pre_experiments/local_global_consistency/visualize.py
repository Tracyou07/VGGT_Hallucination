"""Render diagnostic figures from completed scalar analysis outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

from pre_experiments.local_global_consistency.metrics import spearman_correlation


COLORS = {
    "blue": "#2563EB",
    "green": "#15803D",
    "amber": "#D97706",
    "red": "#DC2626",
    "gray": "#64748B",
    "easy": "#2A9D8F",
    "medium": "#E9C46A",
    "hard": "#E76F51",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as error:
        raise ValueError(f"cannot read visualization input: {path}") from error
    if not rows:
        raise ValueError(f"visualization input is empty: {path}")
    return rows


def _float(row: dict[str, str], field: str) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid numeric field {field}") from error
    if not np.isfinite(value):
        raise ValueError(f"non-finite numeric field {field}")
    return value


def _save(figure: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def _plot_split(split_manifest: Path, output: Path) -> Path:
    try:
        payload = json.loads(split_manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid split manifest: {split_manifest}") from error
    records = payload.get("scene_difficulty") if isinstance(payload, dict) else None
    if not isinstance(records, dict) or not records:
        raise ValueError("split manifest has no scene_difficulty records")
    ordered = sorted(
        records.items(),
        key=lambda item: (float(item[1]["difficulty_score"]), item[0]),
    )
    figure, axis = plt.subplots(figsize=(12, 5.5))
    for index, (scene, record) in enumerate(ordered):
        stratum = str(record["stratum"])
        selected = bool(record.get("selected_for_calibration"))
        axis.scatter(
            index,
            float(record["difficulty_score"]),
            s=72 if selected else 34,
            color=COLORS.get(stratum, COLORS["gray"]),
            edgecolor="black" if selected else "white",
            linewidth=1.2 if selected else 0.4,
            zorder=3,
        )
        if selected:
            axis.annotate(
                scene,
                (index, float(record["difficulty_score"])),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=7,
                rotation=45,
            )
    axis.set(
        title="ScanNet-50 raw-motion difficulty split",
        xlabel="Candidate scenes ordered by raw-GT motion difficulty",
        ylabel="Average percentile-rank difficulty",
    )
    axis.grid(axis="y", alpha=0.25)
    axis.legend(
        handles=[
            Patch(color=COLORS[name], label=name.title())
            for name in ("easy", "medium", "hard")
        ]
        + [
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markeredgecolor="black",
                markerfacecolor="white",
                label="Selected calibration scene",
            )
        ],
        frameon=False,
        ncol=4,
        loc="upper left",
    )
    return _save(figure, output)


def _plot_calibration_correlations(run_dir: Path, output: Path) -> Path:
    rows = _read_csv(run_dir / "calibration_summary.csv")
    score_names = sorted(
        {row["score"] for row in rows if row.get("gate") == "all"}
    )
    if not score_names:
        raise ValueError("calibration summary has no all-frame score rows")
    translation = []
    rotation = []
    for score in score_names:
        selected = [row for row in rows if row["score"] == score and row["gate"] == "all"]
        translation.append(
            [
                _float(row, "translation_growth_spearman")
                for row in selected
                if row.get("translation_growth_spearman") not in ("", None)
            ]
        )
        rotation.append(
            [
                _float(row, "rotation_growth_spearman")
                for row in selected
                if row.get("rotation_growth_spearman") not in ("", None)
            ]
        )
    figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for axis, values, title, color in (
        (axes[0], translation, "Translation-growth Spearman", COLORS["blue"]),
        (axes[1], rotation, "Rotation-growth Spearman", COLORS["amber"]),
    ):
        boxes = axis.boxplot(values, patch_artist=True, showfliers=False)
        for box in boxes["boxes"]:
            box.set(facecolor=color, alpha=0.65)
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_ylabel(title)
        axis.grid(axis="y", alpha=0.25)
    axes[1].set_xticks(range(1, len(score_names) + 1), score_names, rotation=30, ha="right")
    figure.suptitle("Calibration score-to-GT associations across scenes")
    return _save(figure, output)


def _plot_holdout_error_growth(run_dir: Path, output: Path) -> Path:
    rows = _read_csv(run_dir / "holdout_per_scene_summary.csv")
    rows.sort(key=lambda row: _float(row, "translation_growth_mean"))
    scenes = [row["scene"] for row in rows]
    x = np.arange(len(rows))
    figure, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    for axis, field, title, color in (
        (
            axes[0],
            "translation_growth_mean",
            "Mean translation error growth (global - local)",
            COLORS["blue"],
        ),
        (
            axes[1],
            "rotation_growth_mean",
            "Mean rotation error growth (global - local, deg)",
            COLORS["amber"],
        ),
    ):
        values = np.asarray([_float(row, field) for row in rows])
        axis.bar(x, values, color=np.where(values >= 0, color, COLORS["green"]))
        axis.axhline(0.0, color="black", linewidth=0.9)
        axis.set_ylabel(title)
        axis.grid(axis="y", alpha=0.2)
    axes[1].set_xticks(x, scenes, rotation=90, fontsize=7)
    figure.suptitle("Holdout error growth by scene")
    return _save(figure, output)


def _joined_holdout_rows(run_dir: Path) -> list[dict[str, float | str | int]]:
    scores = _read_csv(run_dir / "holdout_prediction_scores_per_frame.csv")
    validation = _read_csv(run_dir / "holdout_gt_validation_per_frame.csv")
    labels = {
        (row["scene"], int(row["frame_id"])): row for row in validation
    }
    if len(labels) != len(validation):
        raise ValueError("duplicate holdout validation identities")
    joined = []
    for row in scores:
        identity = (row["scene"], int(row["frame_id"]))
        if identity not in labels:
            raise ValueError(f"score has no GT validation row: {identity}")
        label = labels[identity]
        joined.append(
            {
                "scene": row["scene"],
                "frame_id": identity[1],
                "pose_translation": _float(row, "global_local_pose_translation"),
                "pose_rotation": _float(row, "global_local_pose_rotation_deg"),
                "translation_growth": _float(
                    label, "translation_error_growth_global_minus_local"
                ),
                "rotation_growth": _float(
                    label, "rotation_error_growth_global_minus_local_deg"
                ),
            }
        )
    if len(joined) != len(validation):
        raise ValueError("score and validation row counts differ")
    return joined


def _plot_holdout_score_growth(run_dir: Path, output: Path) -> Path:
    rows = _joined_holdout_rows(run_dir)
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    pairs = (
        (
            "pose_translation",
            "translation_growth",
            "Pose translation disagreement",
            "Translation error growth",
        ),
        (
            "pose_rotation",
            "rotation_growth",
            "Pose rotation disagreement (deg)",
            "Rotation error growth (deg)",
        ),
    )
    for axis, (x_field, y_field, x_label, y_label) in zip(axes, pairs):
        x = np.asarray([float(row[x_field]) for row in rows])
        y = np.asarray([float(row[y_field]) for row in rows])
        density = axis.hexbin(
            x,
            y,
            gridsize=35,
            mincnt=1,
            cmap="viridis",
            linewidths=0.15,
        )
        correlation = spearman_correlation(x, y)
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set(xlabel=x_label, ylabel=y_label)
        axis.set_title(
            f"Spearman = {correlation:.3f}"
            if correlation is not None
            else "Spearman unavailable"
        )
        figure.colorbar(density, ax=axis, label="Frame count")
    figure.suptitle("Prediction-only disagreement versus raw-GT validation")
    return _save(figure, output)


def _plot_holdout_coverage(run_dir: Path, output: Path) -> Path:
    rows = _read_csv(run_dir / "holdout_per_scene_summary.csv")
    rows.sort(
        key=lambda row: (
            _float(row, "token_reliable_coverage")
            + _float(row, "pose_reliable_coverage")
        )
        / 2.0
    )
    x = np.arange(len(rows))
    width = 0.42
    figure, axis = plt.subplots(figsize=(14, 5.5))
    axis.bar(
        x - width / 2,
        [_float(row, "token_reliable_coverage") for row in rows],
        width,
        label="Token reliability",
        color=COLORS["blue"],
    )
    axis.bar(
        x + width / 2,
        [_float(row, "pose_reliable_coverage") for row in rows],
        width,
        label="Pose reliability",
        color=COLORS["green"],
    )
    axis.set(
        title="Frozen-threshold reliability coverage by holdout scene",
        ylabel="Reliable fraction among overlap-evaluable frames",
        ylim=(0.0, 1.05),
    )
    axis.set_xticks(x, [row["scene"] for row in rows], rotation=90, fontsize=7)
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False)
    return _save(figure, output)


def _plot_holdout_ci(run_dir: Path, output: Path) -> Path:
    rows = _read_csv(run_dir / "holdout_aggregate_summary.csv")
    selected_names = (
        "translation_growth_mean",
        "translation_growth_median",
        "translation_growth_positive_fraction",
        "rotation_growth_mean",
        "rotation_growth_median",
        "rotation_growth_positive_fraction",
        "token_reliable_coverage",
        "pose_reliable_coverage",
    )
    by_name = {row["metric"]: row for row in rows}
    selected = [by_name[name] for name in selected_names if name in by_name]
    if not selected:
        raise ValueError("holdout aggregate summary has no primary metrics")
    estimates = np.asarray([_float(row, "estimate") for row in selected])
    lows = np.asarray([_float(row, "ci95_low") for row in selected])
    highs = np.asarray([_float(row, "ci95_high") for row in selected])
    y = np.arange(len(selected))
    figure, axis = plt.subplots(figsize=(10, 6.5))
    axis.errorbar(
        estimates,
        y,
        xerr=np.vstack([estimates - lows, highs - estimates]),
        fmt="o",
        color=COLORS["blue"],
        ecolor=COLORS["gray"],
        capsize=4,
    )
    axis.axvline(0.0, color="black", linewidth=0.8)
    axis.set_yticks(y, [row["metric"] for row in selected])
    axis.set(
        title="Holdout scene-bootstrap estimates and 95% intervals",
        xlabel="Mean of scene-level statistics",
    )
    axis.grid(axis="x", alpha=0.2)
    axis.invert_yaxis()
    return _save(figure, output)


def write_visualizations(
    run_dir: Path,
    *,
    mode: str,
    split_manifest: Path | None = None,
) -> list[Path]:
    """Write deterministic PNG diagnostics without changing numeric evidence."""
    run_dir = run_dir.resolve()
    if mode not in {"calibration", "holdout"}:
        raise ValueError("visualization mode must be calibration or holdout")
    output_dir = run_dir / "visualizations"
    outputs: list[Path] = []
    if split_manifest is not None:
        outputs.append(
            _plot_split(
                split_manifest.resolve(),
                output_dir / "split_difficulty.png",
            )
        )
    if mode == "calibration":
        outputs.append(
            _plot_calibration_correlations(
                run_dir,
                output_dir / "calibration_score_correlations.png",
            )
        )
    else:
        outputs.extend(
            [
                _plot_holdout_error_growth(
                    run_dir, output_dir / "holdout_error_growth_by_scene.png"
                ),
                _plot_holdout_score_growth(
                    run_dir, output_dir / "holdout_score_vs_error_growth.png"
                ),
                _plot_holdout_coverage(
                    run_dir, output_dir / "holdout_reliability_coverage.png"
                ),
                _plot_holdout_ci(
                    run_dir, output_dir / "holdout_aggregate_ci.png"
                ),
            ]
        )
    return outputs


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["calibration", "holdout"], required=True)
    parser.add_argument("--split-manifest", type=Path)
    args = parser.parse_args(argv)
    outputs = write_visualizations(
        args.run_dir,
        mode=args.mode,
        split_manifest=args.split_manifest,
    )
    for output in outputs:
        print(f"[plot] {output}")


if __name__ == "__main__":
    main()
