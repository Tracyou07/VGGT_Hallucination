"""Prediction-only local-global scores and separate raw-GT validation labels."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

import numpy as np

from pre_experiments.camera_iteration.pose_metrics import align_pose_sequence
from pre_experiments.local_global_consistency.alignment import (
    align_prediction_trajectories,
)


def _cosine_distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError("token arrays must have matching shape [S, C]")
    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    if np.any(a_norm <= 1e-12) or np.any(b_norm <= 1e-12):
        raise ValueError("Camera Tokens must have non-zero norms")
    return 1.0 - np.sum((a / a_norm) * (b / b_norm), axis=1)


def _median(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    if len(array) == 0 or not np.isfinite(array).all():
        raise ValueError("median requires finite values")
    return float(np.median(array))


def _aligned_errors(pred_c2w: np.ndarray, gt_c2w_raw: np.ndarray) -> dict[str, np.ndarray]:
    result = align_pose_sequence(np.linalg.inv(pred_c2w), gt_c2w_raw)
    return {
        "translation": np.asarray(result["translation_error_aligned"], dtype=np.float64),
        "rotation": np.asarray(result["rotation_error_deg_aligned"], dtype=np.float64),
    }


def build_scene_rows(
    scene: str,
    global_artifact: dict[str, np.ndarray],
    windows: list[dict[str, object]],
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    """Build observation, overlap, score, and separate GT-validation rows."""
    global_ids = np.asarray(global_artifact["frame_ids"], dtype=np.int64)
    global_tokens = np.asarray(global_artifact["normalized_camera_tokens"])
    global_pred = np.asarray(global_artifact["pred_c2w_raw"], dtype=np.float64)
    global_gt = np.asarray(global_artifact["gt_c2w_raw"], dtype=np.float64)
    if (
        global_tokens.ndim != 2
        or len(global_tokens) != len(global_ids)
        or global_pred.shape != (len(global_ids), 4, 4)
        or global_gt.shape != (len(global_ids), 4, 4)
    ):
        raise ValueError("global artifact arrays must match frame_ids")
    global_index = {int(frame_id): index for index, frame_id in enumerate(global_ids)}
    if len(global_index) != len(global_ids):
        raise ValueError("global frame_ids must be unique")
    global_errors = _aligned_errors(global_pred, global_gt)

    observations: list[dict[str, object]] = []
    validation_observations: dict[int, list[dict[str, float]]] = defaultdict(list)
    normalized_windows: list[dict[str, object]] = []
    for record in sorted(windows, key=lambda item: int(item["index"])):
        artifact = record["artifact"]
        if not isinstance(artifact, dict):
            raise ValueError("window record must contain an artifact dictionary")
        ids = np.asarray(artifact["frame_ids"], dtype=np.int64)
        tokens = np.asarray(artifact["normalized_camera_tokens"])
        pred = np.asarray(artifact["pred_c2w_raw"], dtype=np.float64)
        gt = np.asarray(artifact["gt_c2w_raw"], dtype=np.float64)
        indices = np.asarray([global_index[int(frame_id)] for frame_id in ids], dtype=np.int64)
        start = int(record["start"])
        stop = int(record["stop"])
        if not np.array_equal(indices, np.arange(start, stop)):
            raise ValueError("window frame IDs do not match global sequence boundaries")
        if not np.allclose(gt, global_gt[indices], atol=1e-10, rtol=0):
            raise ValueError("window raw GT does not match global raw GT")
        global_local_alignment = align_prediction_trajectories(global_pred[indices], pred)
        token_distance = _cosine_distance(global_tokens[indices], tokens)
        local_errors = _aligned_errors(pred, gt)
        window_index = int(record["index"])
        for local_index, (frame_id, sequence_index) in enumerate(zip(ids, indices)):
            boundary_distance = min(local_index, len(ids) - 1 - local_index)
            observations.append(
                {
                    "scene": scene,
                    "frame_id": int(frame_id),
                    "sequence_index": int(sequence_index),
                    "window_index": window_index,
                    "window_start": start,
                    "window_stop": stop,
                    "boundary_distance": boundary_distance,
                    "global_local_token_cosine": float(token_distance[local_index]),
                    "global_local_pose_translation": float(
                        global_local_alignment["translation_residual"][local_index]
                    ),
                    "global_local_pose_rotation_deg": float(
                        global_local_alignment["rotation_residual_deg"][local_index]
                    ),
                }
            )
            validation_observations[int(frame_id)].append(
                {
                    "local_translation_error_aligned": float(
                        local_errors["translation"][local_index]
                    ),
                    "local_rotation_error_aligned_deg": float(
                        local_errors["rotation"][local_index]
                    ),
                }
            )
        normalized_windows.append(
            {
                "index": window_index,
                "start": start,
                "stop": stop,
                "ids": ids,
                "tokens": tokens,
                "pred": pred,
            }
        )

    overlap_rows: list[dict[str, object]] = []
    overlap_by_frame: dict[int, dict[str, float]] = {}
    for left, right in zip(normalized_windows, normalized_windows[1:]):
        left_lookup = {int(frame_id): index for index, frame_id in enumerate(left["ids"])}
        right_lookup = {int(frame_id): index for index, frame_id in enumerate(right["ids"])}
        shared_ids = [frame_id for frame_id in left_lookup if frame_id in right_lookup]
        if len(shared_ids) < 2:
            raise ValueError("adjacent local windows must overlap by at least two frames")
        left_indices = np.asarray([left_lookup[frame_id] for frame_id in shared_ids])
        right_indices = np.asarray([right_lookup[frame_id] for frame_id in shared_ids])
        alignment = align_prediction_trajectories(
            left["pred"][left_indices], right["pred"][right_indices]
        )
        token_distance = _cosine_distance(
            left["tokens"][left_indices], right["tokens"][right_indices]
        )
        for index, frame_id in enumerate(shared_ids):
            row = {
                "scene": scene,
                "frame_id": int(frame_id),
                "left_window_index": int(left["index"]),
                "right_window_index": int(right["index"]),
                "local_local_token_cosine": float(token_distance[index]),
                "local_local_pose_translation": float(
                    alignment["translation_residual"][index]
                ),
                "local_local_pose_rotation_deg": float(
                    alignment["rotation_residual_deg"][index]
                ),
            }
            overlap_rows.append(row)
            overlap_by_frame[int(frame_id)] = {
                key: float(row[key])
                for key in (
                    "local_local_token_cosine",
                    "local_local_pose_translation",
                    "local_local_pose_rotation_deg",
                )
            }

    observations_by_frame: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in observations:
        observations_by_frame[int(row["frame_id"])].append(row)
    score_rows: list[dict[str, object]] = []
    validation_rows: list[dict[str, object]] = []
    for sequence_index, frame_id_value in enumerate(global_ids):
        frame_id = int(frame_id_value)
        frame_observations = observations_by_frame[frame_id]
        selected = max(
            frame_observations,
            key=lambda row: (int(row["boundary_distance"]), -int(row["window_start"])),
        )
        overlap = overlap_by_frame.get(frame_id, {})
        score_rows.append(
            {
                "scene": scene,
                "frame_id": frame_id,
                "sequence_index": sequence_index,
                "local_observation_count": len(frame_observations),
                "selected_local_window_index": int(selected["window_index"]),
                "selected_boundary_distance": int(selected["boundary_distance"]),
                "global_local_token_cosine": _median(
                    row["global_local_token_cosine"] for row in frame_observations
                ),
                "global_local_pose_translation": _median(
                    row["global_local_pose_translation"] for row in frame_observations
                ),
                "global_local_pose_rotation_deg": _median(
                    row["global_local_pose_rotation_deg"] for row in frame_observations
                ),
                "local_local_token_cosine": overlap.get("local_local_token_cosine"),
                "local_local_pose_translation": overlap.get(
                    "local_local_pose_translation"
                ),
                "local_local_pose_rotation_deg": overlap.get(
                    "local_local_pose_rotation_deg"
                ),
            }
        )
        local_validation = validation_observations[frame_id]
        global_translation = float(global_errors["translation"][sequence_index])
        global_rotation = float(global_errors["rotation"][sequence_index])
        local_translation = _median(
            row["local_translation_error_aligned"] for row in local_validation
        )
        local_rotation = _median(
            row["local_rotation_error_aligned_deg"] for row in local_validation
        )
        validation_rows.append(
            {
                "scene": scene,
                "frame_id": frame_id,
                "sequence_index": sequence_index,
                "global_translation_error_aligned": global_translation,
                "median_local_translation_error_aligned": local_translation,
                "translation_error_growth_global_minus_local": global_translation
                - local_translation,
                "global_rotation_error_aligned_deg": global_rotation,
                "median_local_rotation_error_aligned_deg": local_rotation,
                "rotation_error_growth_global_minus_local_deg": global_rotation
                - local_rotation,
            }
        )
    return observations, overlap_rows, score_rows, validation_rows


def fit_reliability_thresholds(
    score_rows: list[dict[str, object]],
    *,
    stable_scenes: set[str],
) -> dict[str, float]:
    """Fit p95 reliability thresholds from prediction-only stable controls."""
    mapping = {
        "token_cosine_p95": "local_local_token_cosine",
        "pose_translation_p95": "local_local_pose_translation",
        "pose_rotation_deg_p95": "local_local_pose_rotation_deg",
    }
    thresholds = {}
    for output_name, field in mapping.items():
        values = [
            float(row[field])
            for row in score_rows
            if row["scene"] in stable_scenes and row.get(field) is not None
        ]
        if not values:
            raise ValueError(f"no stable-control values available for {field}")
        thresholds[output_name] = float(np.percentile(values, 95))
    return thresholds


def apply_reliability(
    score_rows: list[dict[str, object]],
    thresholds: dict[str, float] | None,
) -> list[dict[str, object]]:
    """Annotate local reliability without using GT evaluation labels."""
    output = []
    for row in score_rows:
        annotated = dict(row)
        has_overlap = row.get("local_local_token_cosine") is not None
        if thresholds is None or not has_overlap:
            annotated["token_local_reliable"] = None
            annotated["pose_local_reliable"] = None
        else:
            annotated["token_local_reliable"] = bool(
                float(row["local_local_token_cosine"])
                <= thresholds["token_cosine_p95"]
            )
            annotated["pose_local_reliable"] = bool(
                float(row["local_local_pose_translation"])
                <= thresholds["pose_translation_p95"]
                and float(row["local_local_pose_rotation_deg"])
                <= thresholds["pose_rotation_deg_p95"]
            )
        output.append(annotated)
    return output


def _pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 3 or np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0
        start = stop
    return ranks


def _spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    return _pearson(_ranks(left), _ranks(right))


def summarize_scores(
    score_rows: list[dict[str, object]],
    validation_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Correlate prediction-only scores with separately supplied GT labels."""
    labels = {
        (str(row["scene"]), int(row["frame_id"])): row for row in validation_rows
    }
    candidates = (
        ("global_local_token_cosine", "token_local_reliable"),
        ("global_local_pose_translation", "pose_local_reliable"),
        ("global_local_pose_rotation_deg", "pose_local_reliable"),
        ("local_local_token_cosine", None),
        ("local_local_pose_translation", None),
        ("local_local_pose_rotation_deg", None),
    )
    scenes = sorted({str(row["scene"]) for row in score_rows})
    summaries: list[dict[str, object]] = []
    for scene in scenes:
        scene_rows = [row for row in score_rows if row["scene"] == scene]
        for score_name, reliability_name in candidates:
            for gate in ("all", "reliable") if reliability_name else ("all",):
                pairs = []
                for row in scene_rows:
                    value = row.get(score_name)
                    if value is None:
                        continue
                    if gate == "reliable" and row.get(reliability_name) is not True:
                        continue
                    label = labels.get((scene, int(row["frame_id"])))
                    if label is None:
                        raise ValueError("score row has no matching validation row")
                    pairs.append((float(value), label))
                if not pairs:
                    continue
                scores = np.asarray([pair[0] for pair in pairs], dtype=np.float64)
                translation = np.asarray(
                    [
                        pair[1]["translation_error_growth_global_minus_local"]
                        for pair in pairs
                    ],
                    dtype=np.float64,
                )
                rotation = np.asarray(
                    [
                        pair[1]["rotation_error_growth_global_minus_local_deg"]
                        for pair in pairs
                    ],
                    dtype=np.float64,
                )
                order = np.argsort(scores)
                quartile_count = max(1, len(scores) // 4)
                summaries.append(
                    {
                        "scene": scene,
                        "score": score_name,
                        "gate": gate,
                        "frame_count": len(scores),
                        "score_mean": float(np.mean(scores)),
                        "score_p95": float(np.percentile(scores, 95)),
                        "translation_growth_mean": float(np.mean(translation)),
                        "translation_growth_pearson": _pearson(scores, translation),
                        "translation_growth_spearman": _spearman(scores, translation),
                        "rotation_growth_pearson": _pearson(scores, rotation),
                        "rotation_growth_spearman": _spearman(scores, rotation),
                        "translation_growth_bottom_quartile_mean": float(
                            np.mean(translation[order[:quartile_count]])
                        ),
                        "translation_growth_top_quartile_mean": float(
                            np.mean(translation[order[-quartile_count:]])
                        ),
                    }
                )
    return summaries
