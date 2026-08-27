from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ClusterResult:
    labels: np.ndarray
    centers: np.ndarray
    one_cluster_sse: float
    two_cluster_sse: float
    one_to_two_sse_ratio: float


def two_means(features: np.ndarray, *, max_iterations: int = 50) -> ClusterResult:
    """Deterministic two-means with farthest-pair initialization."""
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 1:
        raise ValueError("features must have shape [samples, dimensions]")
    if not np.isfinite(values).all() or max_iterations < 1:
        raise ValueError("features must be finite and max_iterations positive")
    sample_count = values.shape[0]
    mean = values.mean(axis=0)
    one_sse = float(np.sum((values - mean) ** 2))
    if sample_count == 1 or one_sse == 0.0:
        labels = np.zeros(sample_count, dtype=np.int64)
        centers = np.stack((mean, mean)).astype(np.float32)
        return ClusterResult(labels, centers, one_sse, one_sse, 1.0)

    norms = np.sum(values * values, axis=1)
    distances = norms[:, None] + norms[None, :] - 2.0 * (values @ values.T)
    first, second = np.unravel_index(int(np.argmax(distances)), distances.shape)
    centers = np.stack((values[first], values[second]))
    labels = np.full(sample_count, -1, dtype=np.int64)
    for _ in range(max_iterations):
        squared = np.stack(
            (np.sum((values - centers[0]) ** 2, axis=1), np.sum((values - centers[1]) ** 2, axis=1)),
            axis=1,
        )
        updated_labels = np.argmin(squared, axis=1).astype(np.int64)
        if np.array_equal(updated_labels, labels):
            break
        labels = updated_labels
        for cluster in (0, 1):
            members = values[labels == cluster]
            if len(members):
                centers[cluster] = members.mean(axis=0)

    order = sorted(range(2), key=lambda index: tuple(centers[index].tolist()))
    if order != [0, 1]:
        centers = centers[order]
        labels = 1 - labels
    two_sse = float(
        sum(np.sum((values[labels == cluster] - centers[cluster]) ** 2) for cluster in (0, 1))
    )
    ratio = one_sse / max(two_sse, np.finfo(np.float64).eps)
    return ClusterResult(
        labels=labels,
        centers=centers.astype(np.float32),
        one_cluster_sse=one_sse,
        two_cluster_sse=two_sse,
        one_to_two_sse_ratio=float(ratio),
    )
