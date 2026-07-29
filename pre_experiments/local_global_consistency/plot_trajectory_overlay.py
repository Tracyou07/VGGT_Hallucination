"""Plot raw and aligned global/local camera trajectories against raw GT."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from pre_experiments.common.pose_metrics import align_pose_sequence


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise ValueError(f"missing trajectory artifact: {path}")
    with np.load(path, allow_pickle=False) as artifact:
        required = {"frame_ids", "pred_c2w_raw", "gt_c2w_raw"}
        missing = required.difference(artifact.files)
        if missing:
            raise ValueError(f"{path} is missing arrays: {sorted(missing)}")
        return {name: np.asarray(artifact[name]) for name in required}


def _validate_poses(name: str, poses: np.ndarray, count: int) -> np.ndarray:
    array = np.asarray(poses, dtype=np.float64)
    if array.shape != (count, 4, 4) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite with shape [{count}, 4, 4]")
    return array


def _aligned_c2w(pred_c2w_raw: np.ndarray, gt_c2w_raw: np.ndarray) -> np.ndarray:
    result = align_pose_sequence(np.linalg.inv(pred_c2w_raw), gt_c2w_raw)
    return np.asarray(result["aligned_c2w"], dtype=np.float64)


def load_trajectories(
    global_artifact_path: Path,
    local_run_dir: Path,
    scene: str,
) -> dict[str, object]:
    global_artifact = _load_npz(global_artifact_path)
    frame_ids = np.asarray(global_artifact["frame_ids"], dtype=np.int64)
    if frame_ids.ndim != 1 or len(np.unique(frame_ids)) != len(frame_ids):
        raise ValueError("global frame_ids must be a unique one-dimensional array")
    global_pred_raw = _validate_poses(
        "global pred_c2w_raw", global_artifact["pred_c2w_raw"], len(frame_ids)
    )
    gt_raw = _validate_poses(
        "global gt_c2w_raw", global_artifact["gt_c2w_raw"], len(frame_ids)
    )
    global_index = {int(frame_id): index for index, frame_id in enumerate(frame_ids)}

    local_windows = []
    window_paths = sorted(
        (local_run_dir / scene).glob("window_*/window_diagnostics.npz")
    )
    if not window_paths:
        raise ValueError(f"no local windows found for {scene}")
    for path in window_paths:
        artifact = _load_npz(path)
        ids = np.asarray(artifact["frame_ids"], dtype=np.int64)
        if ids.ndim != 1 or len(np.unique(ids)) != len(ids):
            raise ValueError(f"invalid local frame_ids: {path}")
        try:
            indices = np.asarray([global_index[int(frame_id)] for frame_id in ids])
        except KeyError as error:
            raise ValueError(f"local frame is absent from global artifact: {path}") from error
        pred_raw = _validate_poses(
            f"local pred_c2w_raw in {path}", artifact["pred_c2w_raw"], len(ids)
        )
        local_gt = _validate_poses(
            f"local gt_c2w_raw in {path}", artifact["gt_c2w_raw"], len(ids)
        )
        if not np.allclose(local_gt, gt_raw[indices], atol=1e-10, rtol=0):
            raise ValueError(f"local raw GT differs from global raw GT: {path}")
        local_windows.append(
            {
                "name": path.parent.name,
                "ids": ids,
                "pred_raw": pred_raw,
                "pred_aligned": _aligned_c2w(pred_raw, local_gt),
            }
        )

    return {
        "frame_ids": frame_ids,
        "gt_raw": gt_raw,
        "global_raw": global_pred_raw,
        "global_aligned": _aligned_c2w(global_pred_raw, gt_raw),
        "local_windows": local_windows,
    }


def _centers(poses: np.ndarray) -> np.ndarray:
    return np.asarray(poses, dtype=np.float64)[:, :3, 3]


def _set_equal_3d(axis: plt.Axes, point_sets: list[np.ndarray]) -> None:
    points = np.concatenate(point_sets, axis=0)
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    center = (minimum + maximum) / 2.0
    radius = max(float(np.max(maximum - minimum)) / 2.0, 1e-6)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1, 1, 1))


def _plot_row(
    axes: list[plt.Axes],
    gt: np.ndarray,
    global_poses: np.ndarray,
    local_windows: list[dict[str, object]],
    local_field: str,
    row_title: str,
) -> None:
    gt_points = _centers(gt)
    global_points = _centers(global_poses)
    local_points = [_centers(np.asarray(window[local_field])) for window in local_windows]
    colors = plt.get_cmap("viridis")(np.linspace(0.05, 0.95, len(local_windows)))

    for axis, dimensions, labels in (
        (axes[0], (0, 1), ("X", "Y")),
        (axes[1], (0, 2), ("X", "Z")),
    ):
        axis.plot(
            gt_points[:, dimensions[0]],
            gt_points[:, dimensions[1]],
            color="black",
            linewidth=2.2,
            zorder=4,
        )
        axis.plot(
            global_points[:, dimensions[0]],
            global_points[:, dimensions[1]],
            color="#DC2626",
            linewidth=1.8,
            zorder=3,
        )
        for points, color in zip(local_points, colors):
            axis.plot(
                points[:, dimensions[0]],
                points[:, dimensions[1]],
                color=color,
                linewidth=1.0,
                alpha=0.78,
            )
        axis.scatter(
            gt_points[0, dimensions[0]],
            gt_points[0, dimensions[1]],
            color="black",
            marker="o",
            s=28,
            zorder=5,
        )
        axis.scatter(
            gt_points[-1, dimensions[0]],
            gt_points[-1, dimensions[1]],
            color="black",
            marker="X",
            s=38,
            zorder=5,
        )
        axis.set_xlabel(labels[0])
        axis.set_ylabel(labels[1])
        axis.set_aspect("equal", adjustable="datalim")
        axis.grid(alpha=0.2)

    axis_3d = axes[2]
    axis_3d.plot(*gt_points.T, color="black", linewidth=2.2)
    axis_3d.plot(*global_points.T, color="#DC2626", linewidth=1.8)
    for points, color in zip(local_points, colors):
        axis_3d.plot(*points.T, color=color, linewidth=1.0, alpha=0.78)
    axis_3d.scatter(*gt_points[0], color="black", marker="o", s=28)
    axis_3d.scatter(*gt_points[-1], color="black", marker="X", s=38)
    axis_3d.set_xlabel("X")
    axis_3d.set_ylabel("Y")
    axis_3d.set_zlabel("Z")
    _set_equal_3d(axis_3d, [gt_points, global_points, *local_points])

    axes[0].set_title(f"{row_title}: XY")
    axes[1].set_title(f"{row_title}: XZ")
    axes[2].set_title(f"{row_title}: 3D")


def plot_trajectory_overlay(
    trajectories: dict[str, object],
    scene: str,
    output: Path,
) -> Path:
    figure = plt.figure(figsize=(18, 11))
    axes = [
        figure.add_subplot(2, 3, 1),
        figure.add_subplot(2, 3, 2),
        figure.add_subplot(2, 3, 3, projection="3d"),
        figure.add_subplot(2, 3, 4),
        figure.add_subplot(2, 3, 5),
        figure.add_subplot(2, 3, 6, projection="3d"),
    ]
    local_windows = list(trajectories["local_windows"])
    _plot_row(
        axes[:3],
        np.asarray(trajectories["gt_raw"]),
        np.asarray(trajectories["global_raw"]),
        local_windows,
        "pred_raw",
        "Raw coordinates",
    )
    _plot_row(
        axes[3:],
        np.asarray(trajectories["gt_raw"]),
        np.asarray(trajectories["global_aligned"]),
        local_windows,
        "pred_aligned",
        "Aligned to raw GT",
    )
    figure.suptitle(
        f"{scene}: raw and aligned 500-frame global / 100-frame local trajectories",
        fontsize=16,
    )
    figure.legend(
        handles=[
            Line2D([0], [0], color="black", linewidth=2.2, label="Raw GT"),
            Line2D([0], [0], color="#DC2626", linewidth=1.8, label="500-frame global"),
            Line2D([0], [0], color="#21918C", linewidth=1.2, label="100-frame local windows"),
            Line2D([0], [0], color="black", marker="o", linestyle="", label="GT start"),
            Line2D([0], [0], color="black", marker="X", linestyle="", label="GT end"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=5,
        frameon=False,
    )
    figure.text(
        0.5,
        0.012,
        "Aligned local windows use independent Sim(3) transforms and are not a stitched trajectory. "
        "GT remains raw in both rows.",
        ha="center",
        fontsize=10,
    )
    figure.tight_layout(rect=(0.02, 0.04, 0.98, 0.91))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    if not output.is_file() or output.stat().st_size == 0:
        raise ValueError(f"failed to create trajectory figure: {output}")
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot raw and aligned global/local trajectories against raw GT."
    )
    parser.add_argument("--scene", required=True)
    parser.add_argument("--global-artifact", type=Path, required=True)
    parser.add_argument("--local-run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    trajectories = load_trajectories(
        args.global_artifact, args.local_run_dir, args.scene
    )
    output = plot_trajectory_overlay(trajectories, args.scene, args.output)
    print(output)


if __name__ == "__main__":
    main()
