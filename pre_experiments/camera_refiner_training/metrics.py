"""Translation metrics for raw and aligned camera trajectories."""

from __future__ import annotations

import numpy as np

from pre_experiments.camera_refiner_training.geometry import align_centers_to_reference


def translation_metrics(predicted_c2w: np.ndarray, gt_c2w: np.ndarray) -> dict[str, float]:
    predicted = np.asarray(predicted_c2w, dtype=np.float64)
    gt = np.asarray(gt_c2w, dtype=np.float64)
    if predicted.shape != gt.shape or predicted.ndim != 3 or predicted.shape[1:] != (4, 4):
        raise ValueError("predicted and GT poses must have matching [S, 4, 4] shape")
    predicted_centers = predicted[:, :3, 3]
    gt_centers = gt[:, :3, 3]
    aligned, _ = align_centers_to_reference(predicted_centers, gt_centers)
    raw_error = np.linalg.norm(predicted_centers - gt_centers, axis=1)
    aligned_error = np.linalg.norm(aligned - gt_centers, axis=1)
    return {
        "ate_raw": float(np.sqrt(np.mean(raw_error ** 2))),
        "ate_aligned": float(np.sqrt(np.mean(aligned_error ** 2))),
        "median_raw": float(np.median(raw_error)),
        "median_aligned": float(np.median(aligned_error)),
    }
