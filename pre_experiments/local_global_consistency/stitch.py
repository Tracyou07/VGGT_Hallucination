"""Prediction-only sequential stitching of overlapping local trajectories."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from pre_experiments.common.pose_metrics import umeyama


def _window_arrays(
    window: dict[str, np.ndarray], index: int
) -> tuple[np.ndarray, np.ndarray]:
    try:
        frame_ids = np.asarray(window["frame_ids"], dtype=np.int64)
        poses = np.asarray(window["pred_c2w_raw"], dtype=np.float64)
    except KeyError as error:
        raise ValueError(f"window {index} is missing {error.args[0]}") from error
    if (
        frame_ids.ndim != 1
        or len(frame_ids) < 3
        or len(np.unique(frame_ids)) != len(frame_ids)
    ):
        raise ValueError(f"window {index} frame_ids must contain at least three unique IDs")
    if poses.shape != (len(frame_ids), 4, 4) or not np.isfinite(poses).all():
        raise ValueError(f"window {index} poses must be finite with shape [S, 4, 4]")
    return frame_ids, poses


def _apply_sim3(
    poses: np.ndarray,
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    aligned = poses.copy()
    aligned[:, :3, :3] = np.einsum(
        "ij,sjk->sik", rotation, poses[:, :3, :3]
    )
    aligned[:, :3, 3] = scale * (poses[:, :3, 3] @ rotation.T) + translation
    return aligned


def stitch_local_windows(
    windows: Sequence[dict[str, np.ndarray]],
) -> dict[str, object]:
    """Stitch ordered windows using only overlapping predicted camera poses."""
    if not windows:
        raise ValueError("at least one local window is required")

    stitched_by_id: dict[int, np.ndarray] = {}
    boundary_by_id: dict[int, int] = {}
    ordered_ids: list[int] = []
    overlap_counts: list[int] = []
    transforms: list[dict[str, np.ndarray | float]] = []

    for window_index, window in enumerate(windows):
        frame_ids, poses = _window_arrays(window, window_index)
        if window_index == 0:
            aligned = poses.copy()
        else:
            overlap_indices = [
                local_index
                for local_index, frame_id in enumerate(frame_ids)
                if int(frame_id) in stitched_by_id
            ]
            if len(overlap_indices) < 3:
                raise ValueError(
                    f"window {window_index} must have at least three overlapping predicted frames"
                )
            moving_overlap = poses[overlap_indices]
            reference_overlap = np.stack(
                [stitched_by_id[int(frame_ids[i])] for i in overlap_indices]
            )
            scale, rotation, translation = umeyama(
                moving_overlap[:, :3, 3],
                reference_overlap[:, :3, 3],
            )
            aligned = _apply_sim3(poses, scale, rotation, translation)
            overlap_counts.append(len(overlap_indices))
            transforms.append(
                {
                    "scale": scale,
                    "rotation": rotation,
                    "translation": translation,
                }
            )

        for local_index, frame_id_value in enumerate(frame_ids):
            frame_id = int(frame_id_value)
            boundary_distance = min(local_index, len(frame_ids) - 1 - local_index)
            if frame_id not in stitched_by_id:
                ordered_ids.append(frame_id)
                stitched_by_id[frame_id] = aligned[local_index]
                boundary_by_id[frame_id] = boundary_distance
            elif boundary_distance > boundary_by_id[frame_id]:
                stitched_by_id[frame_id] = aligned[local_index]
                boundary_by_id[frame_id] = boundary_distance

    return {
        "frame_ids": np.asarray(ordered_ids, dtype=np.int64),
        "stitched_c2w": np.stack([stitched_by_id[frame_id] for frame_id in ordered_ids]),
        "overlap_counts": overlap_counts,
        "transforms": transforms,
    }


def assemble_windows_in_reference_gauge(
    windows: Sequence[dict[str, np.ndarray]],
    *,
    reference_frame_ids: np.ndarray,
    reference_c2w: np.ndarray,
) -> dict[str, object]:
    """Align each local window to a prediction-only reference and assemble it."""
    frame_ids = np.asarray(reference_frame_ids, dtype=np.int64)
    reference = np.asarray(reference_c2w, dtype=np.float64)
    if (
        frame_ids.ndim != 1
        or len(frame_ids) < 3
        or len(np.unique(frame_ids)) != len(frame_ids)
    ):
        raise ValueError("reference_frame_ids must contain at least three unique IDs")
    if reference.shape != (len(frame_ids), 4, 4) or not np.isfinite(reference).all():
        raise ValueError("reference_c2w must be finite with shape [S, 4, 4]")
    if not windows:
        raise ValueError("at least one local window is required")

    reference_index = {int(frame_id): index for index, frame_id in enumerate(frame_ids)}
    assembled_by_id: dict[int, np.ndarray] = {}
    boundary_by_id: dict[int, int] = {}
    observation_count = {int(frame_id): 0 for frame_id in frame_ids}
    transforms: list[dict[str, np.ndarray | float]] = []

    for window_index, window in enumerate(windows):
        local_ids, poses = _window_arrays(window, window_index)
        try:
            indices = np.asarray(
                [reference_index[int(frame_id)] for frame_id in local_ids],
                dtype=np.int64,
            )
        except KeyError as error:
            raise ValueError(
                f"window {window_index} contains a frame absent from the reference"
            ) from error
        reference_segment = reference[indices]
        scale, rotation, translation = umeyama(
            poses[:, :3, 3],
            reference_segment[:, :3, 3],
        )
        aligned = _apply_sim3(poses, scale, rotation, translation)
        transforms.append(
            {
                "scale": scale,
                "rotation": rotation,
                "translation": translation,
            }
        )
        for local_index, frame_id_value in enumerate(local_ids):
            frame_id = int(frame_id_value)
            observation_count[frame_id] += 1
            boundary_distance = min(local_index, len(local_ids) - 1 - local_index)
            if (
                frame_id not in assembled_by_id
                or boundary_distance > boundary_by_id[frame_id]
            ):
                assembled_by_id[frame_id] = aligned[local_index]
                boundary_by_id[frame_id] = boundary_distance

    missing = [int(frame_id) for frame_id in frame_ids if int(frame_id) not in assembled_by_id]
    if missing:
        raise ValueError(f"local windows do not cover reference frames: {missing[:5]}")
    return {
        "frame_ids": frame_ids.copy(),
        "assembled_c2w": np.stack(
            [assembled_by_id[int(frame_id)] for frame_id in frame_ids]
        ),
        "observation_count": np.asarray(
            [observation_count[int(frame_id)] for frame_id in frame_ids],
            dtype=np.int64,
        ),
        "transforms": transforms,
    }


def build_translation_hybrid(
    reference_c2w: np.ndarray,
    assembled_c2w: np.ndarray,
) -> np.ndarray:
    """Use assembled camera centers while retaining reference orientations."""
    reference = np.asarray(reference_c2w, dtype=np.float64)
    assembled = np.asarray(assembled_c2w, dtype=np.float64)
    if (
        reference.shape != assembled.shape
        or reference.ndim != 3
        or reference.shape[1:] != (4, 4)
        or not np.isfinite(reference).all()
        or not np.isfinite(assembled).all()
    ):
        raise ValueError("reference and assembled poses must be finite matching [S, 4, 4] arrays")
    hybrid = reference.copy()
    hybrid[:, :3, 3] = assembled[:, :3, 3]
    return hybrid
