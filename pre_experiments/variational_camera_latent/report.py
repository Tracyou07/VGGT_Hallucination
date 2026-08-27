from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _read_object(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid run manifest: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError("run manifest must be a JSON object")
    return payload


def summarize_run(
    prediction_manifest: Path,
    privileged_manifest: Path,
    destination: Path,
) -> dict[str, object]:
    """Classify exploratory signal without turning weak science into technical failure."""
    prediction = _read_object(prediction_manifest)
    privileged = _read_object(privileged_manifest)
    if prediction.get("schema") != "variational_camera_latent.prediction_manifest.v1":
        raise ValueError("prediction manifest schema mismatch")
    if privileged.get("schema") != "variational_camera_latent.privileged_manifest.v1":
        raise ValueError("privileged manifest schema mismatch")
    improvements = np.asarray(privileged.get("best_relative_improvements"), dtype=np.float64)
    if improvements.ndim != 1 or improvements.size < 1 or not np.isfinite(improvements).all():
        raise ValueError("privileged manifest has invalid best_relative_improvements")
    z_sensitivity = float(prediction.get("z_sensitivity", 0.0))
    ratio = float(prediction.get("median_one_to_two_sse_ratio", 1.0))
    if not np.isfinite([z_sensitivity, ratio]).all() or z_sensitivity < 0.0 or ratio < 1.0:
        raise ValueError("prediction manifest has invalid diversity metrics")
    positive = int(np.count_nonzero(improvements > 0.0))
    median_improvement = float(np.median(improvements))
    if median_improvement >= 0.05 and positive >= max(3, len(improvements) // 4) and ratio >= 1.1:
        signal = "PROMISING"
    elif z_sensitivity > 1e-6 or positive > 0 or ratio > 1.0 + 1e-6:
        signal = "WEAK_SIGNAL"
    else:
        signal = "NO_SIGNAL"
    report: dict[str, object] = {
        "schema": "variational_camera_latent.exploration_report.v1",
        "technically_complete": True,
        "signal": signal,
        "scene_count": int(prediction.get("scene_count", 0)),
        "candidate_count": int(prediction.get("candidate_count", 0)),
        "z_sensitivity": z_sensitivity,
        "median_one_to_two_sse_ratio": ratio,
        "median_best_relative_improvement": median_improvement,
        "positive_overlap_count": positive,
        "limitations": [
            "Phase 1 evaluates independent 50-frame overlaps and does not stitch a 500-frame trajectory.",
            "Two-means evidence is exploratory and is not proof of discrete multimodality.",
        ],
    }
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return report
