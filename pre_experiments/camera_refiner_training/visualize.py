"""Compact diagnostic plots for training and trajectory refinement."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot

    return pyplot


def _save(figure: object, path: Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    figure.savefig(temporary, format="png", dpi=150, bbox_inches="tight")
    temporary.replace(destination)


def write_history_plot(path: Path, history: list[dict[str, float | int]]) -> None:
    if not history:
        raise ValueError("history must not be empty")
    pyplot = _pyplot()
    figure, axis = pyplot.subplots(figsize=(6.0, 3.5))
    epochs = [int(row["epoch"]) for row in history]
    axis.plot(epochs, [float(row["train_loss"]) for row in history], label="train")
    axis.plot(
        epochs,
        [float(row["validation_loss"]) for row in history],
        label="validation",
    )
    axis.set(xlabel="Epoch", ylabel="Loss", title="Translation Refiner Training")
    axis.grid(alpha=0.25)
    axis.legend()
    _save(figure, path)
    pyplot.close(figure)


def write_trajectory_plot(
    path: Path,
    *,
    baseline_c2w: np.ndarray,
    refined_c2w: np.ndarray,
    gt_c2w: np.ndarray,
) -> None:
    baseline = np.asarray(baseline_c2w)[:, :3, 3]
    refined = np.asarray(refined_c2w)[:, :3, 3]
    gt = np.asarray(gt_c2w)[:, :3, 3]
    if baseline.shape != refined.shape or baseline.shape != gt.shape:
        raise ValueError("trajectory pose arrays must have matching shapes")
    pyplot = _pyplot()
    figure, axes = pyplot.subplots(1, 2, figsize=(9.0, 4.0))
    for axis, dimensions, labels in (
        (axes[0], (0, 1), ("X", "Y")),
        (axes[1], (0, 2), ("X", "Z")),
    ):
        axis.plot(gt[:, dimensions[0]], gt[:, dimensions[1]], label="GT", linewidth=2)
        axis.plot(
            baseline[:, dimensions[0]],
            baseline[:, dimensions[1]],
            label="VGGT",
            alpha=0.8,
        )
        axis.plot(
            refined[:, dimensions[0]],
            refined[:, dimensions[1]],
            label="Refined",
            alpha=0.9,
        )
        axis.set(xlabel=labels[0], ylabel=labels[1])
        axis.set_aspect("equal", adjustable="datalim")
        axis.grid(alpha=0.25)
    axes[0].legend()
    figure.suptitle("Camera Center Trajectory")
    _save(figure, path)
    pyplot.close(figure)
