"""Frozen scene-level alpha selection from prediction-only consistency."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import json
from pathlib import Path

import numpy as np

from pre_experiments.camera_hidden_state_attribution.artifacts import (
    canonical_digest,
)


ALPHA_CHOICES = (0.01, 0.02, 0.05)
RIDGE = 1.0
SOURCE_FEATURE_FIELDS = (
    "global_local_token_cosine",
    "global_local_pose_translation",
    "local_local_pose_translation",
)
FEATURE_FIELDS = tuple(
    f"{field}_median" for field in SOURCE_FEATURE_FIELDS
)


def _finite(value: object, *, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be numeric") from error
    if not np.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def build_scene_features(
    score_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Reduce strict prediction-only frame scores to scene medians."""
    if not score_rows:
        raise ValueError("prediction score rows are required")
    by_scene: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    identities = set()
    for row in score_rows:
        scene = str(row.get("scene", ""))
        if not scene:
            raise ValueError("prediction score row has no scene")
        identity = (scene, int(row["frame_id"]))
        if identity in identities:
            raise ValueError(f"duplicate prediction score identity: {identity}")
        identities.add(identity)
        by_scene[scene].append(row)

    output = []
    for scene, rows in sorted(by_scene.items()):
        feature_row: dict[str, object] = {"scene": scene}
        for source, destination in zip(
            SOURCE_FEATURE_FIELDS,
            FEATURE_FIELDS,
        ):
            values = [
                _finite(row[source], field=source)
                for row in rows
                if row.get(source) not in (None, "")
            ]
            if not values:
                raise ValueError(f"{scene} has no finite {source} values")
            feature_row[destination] = float(np.median(values))
        output.append(feature_row)
    return output


def build_oracle_labels(
    replacement_rows: Sequence[Mapping[str, object]],
    *,
    alpha_choices: Sequence[float] = ALPHA_CHOICES,
) -> dict[str, float]:
    """Choose calibration-only alpha labels from aligned prediction errors."""
    choices = tuple(float(alpha) for alpha in alpha_choices)
    if choices != ALPHA_CHOICES:
        raise ValueError(f"alpha choices must be {ALPHA_CHOICES}")
    by_scene: dict[str, dict[float, float]] = defaultdict(dict)
    for row in replacement_rows:
        if row.get("condition_family") != "selected":
            continue
        scene = str(row.get("scene", ""))
        alpha = _finite(row.get("alpha"), field="alpha")
        if alpha not in choices:
            continue
        if alpha in by_scene[scene]:
            raise ValueError(f"duplicate selected alpha={alpha} for {scene}")
        by_scene[scene][alpha] = _finite(
            row.get("aligned_translation_error_delta"),
            field="aligned_translation_error_delta",
        )
    if not by_scene:
        raise ValueError("replacement rows contain no selected alpha curves")
    labels = {}
    for scene, curve in sorted(by_scene.items()):
        if set(curve) != set(choices):
            raise ValueError(f"incomplete alpha curve for {scene}")
        labels[scene] = min(choices, key=lambda alpha: (curve[alpha], alpha))
    return labels


def fit_frozen_selector(
    scene_features: Sequence[Mapping[str, object]],
    labels: Mapping[str, float],
    *,
    split_digest: str,
    score_run_id: str,
    replacement_run_id: str,
) -> dict[str, object]:
    """Fit and authenticate a fixed standardized ridge selector."""
    feature_by_scene = {
        str(row.get("scene", "")): row for row in scene_features
    }
    if (
        not feature_by_scene
        or "" in feature_by_scene
        or len(feature_by_scene) != len(scene_features)
        or set(feature_by_scene) != set(labels)
    ):
        raise ValueError("selector feature and label scenes must match")
    scenes = sorted(feature_by_scene)
    x = np.asarray(
        [
            [
                _finite(feature_by_scene[scene][field], field=field)
                for field in FEATURE_FIELDS
            ]
            for scene in scenes
        ],
        dtype=np.float64,
    )
    y = np.asarray(
        [_finite(labels[scene], field="oracle_alpha") for scene in scenes],
        dtype=np.float64,
    )
    if any(value not in ALPHA_CHOICES for value in y):
        raise ValueError("oracle labels contain an unsupported alpha")
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    normalized = (x - mean) / scale
    coefficients = np.linalg.solve(
        normalized.T @ normalized
        + RIDGE * np.eye(normalized.shape[1], dtype=np.float64),
        normalized.T @ (y - y.mean()),
    )
    frozen: dict[str, object] = {
        "schema_version": 1,
        "method": "scene_prediction_only_ridge_alpha_selector",
        "feature_fields": list(FEATURE_FIELDS),
        "alpha_choices": list(ALPHA_CHOICES),
        "ridge": RIDGE,
        "feature_mean": mean.tolist(),
        "feature_scale": scale.tolist(),
        "coefficients": coefficients.tolist(),
        "intercept": float(y.mean()),
        "calibration_scenes": scenes,
        "calibration_feature_digest": canonical_digest(
            [
                {
                    "scene": scene,
                    **{
                        field: float(feature_by_scene[scene][field])
                        for field in FEATURE_FIELDS
                    },
                }
                for scene in scenes
            ]
        ),
        "calibration_label_digest": canonical_digest(
            {scene: float(labels[scene]) for scene in scenes}
        ),
        "split_digest": str(split_digest),
        "source_score_run_id": str(score_run_id),
        "source_replacement_run_id": str(replacement_run_id),
    }
    frozen["selector_digest"] = canonical_digest(frozen)
    return frozen


def load_frozen_selector(
    source: Mapping[str, object] | Path,
    *,
    expected_split_digest: str,
) -> dict[str, object]:
    """Load a frozen selector and verify its complete numeric contract."""
    if isinstance(source, Path):
        try:
            value = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid frozen selector: {source}") from error
    else:
        try:
            value = json.loads(json.dumps(source))
        except (TypeError, ValueError) as error:
            raise ValueError("invalid frozen selector") from error
    if not isinstance(value, dict):
        raise ValueError("frozen selector must be an object")
    digest = value.pop("selector_digest", None)
    if not isinstance(digest, str) or digest != canonical_digest(value):
        raise ValueError("frozen selector digest mismatch")
    if (
        value.get("schema_version") != 1
        or value.get("method")
        != "scene_prediction_only_ridge_alpha_selector"
        or value.get("feature_fields") != list(FEATURE_FIELDS)
        or value.get("alpha_choices") != list(ALPHA_CHOICES)
        or value.get("ridge") != RIDGE
        or value.get("split_digest") != expected_split_digest
    ):
        raise ValueError("frozen selector provenance mismatch")
    for field in (
        "feature_mean",
        "feature_scale",
        "coefficients",
    ):
        array = np.asarray(value.get(field), dtype=np.float64)
        if (
            array.shape != (len(FEATURE_FIELDS),)
            or not np.isfinite(array).all()
            or (field == "feature_scale" and np.any(array <= 0))
        ):
            raise ValueError(f"frozen selector {field} is invalid")
    _finite(value.get("intercept"), field="intercept")
    value["selector_digest"] = digest
    return value


def predict_alpha(
    frozen: Mapping[str, object],
    scene_features: Mapping[str, object],
) -> float:
    """Predict and quantize one scene alpha without reading GT."""
    values = np.asarray(
        [
            _finite(scene_features[field], field=field)
            for field in FEATURE_FIELDS
        ],
        dtype=np.float64,
    )
    mean = np.asarray(frozen["feature_mean"], dtype=np.float64)
    scale = np.asarray(frozen["feature_scale"], dtype=np.float64)
    coefficients = np.asarray(frozen["coefficients"], dtype=np.float64)
    prediction = float(
        frozen["intercept"]
        + ((values - mean) / scale) @ coefficients
    )
    return min(
        ALPHA_CHOICES,
        key=lambda alpha: (abs(alpha - prediction), alpha),
    )


def evaluate_leave_one_out(
    scene_features: Sequence[Mapping[str, object]],
    labels: Mapping[str, float],
    replacement_rows: Sequence[Mapping[str, object]],
    *,
    split_digest: str,
    score_run_id: str,
    replacement_run_id: str,
    fixed_alpha: float = 0.02,
) -> dict[str, object]:
    """Estimate selector headroom without fitting on the evaluated scene."""
    if fixed_alpha not in ALPHA_CHOICES:
        raise ValueError("fixed alpha must be one of the candidate choices")
    feature_by_scene = {
        str(row.get("scene", "")): row for row in scene_features
    }
    if set(feature_by_scene) != set(labels) or len(labels) < 3:
        raise ValueError("LOOCV requires matching features for at least 3 scenes")
    curves: dict[str, dict[float, float]] = defaultdict(dict)
    for row in replacement_rows:
        if row.get("condition_family") != "selected":
            continue
        scene = str(row.get("scene", ""))
        alpha = _finite(row.get("alpha"), field="alpha")
        if alpha not in ALPHA_CHOICES:
            continue
        curves[scene][alpha] = _finite(
            row.get("aligned_translation_error_delta"),
            field="aligned_translation_error_delta",
        )
    if set(curves) != set(labels) or any(
        set(curve) != set(ALPHA_CHOICES) for curve in curves.values()
    ):
        raise ValueError("LOOCV requires complete candidate curves")

    rows = []
    for held_scene in sorted(labels):
        training_scenes = [
            scene for scene in sorted(labels) if scene != held_scene
        ]
        frozen = fit_frozen_selector(
            [feature_by_scene[scene] for scene in training_scenes],
            {scene: labels[scene] for scene in training_scenes},
            split_digest=split_digest,
            score_run_id=score_run_id,
            replacement_run_id=replacement_run_id,
        )
        predicted_alpha = predict_alpha(
            frozen,
            feature_by_scene[held_scene],
        )
        oracle_alpha = float(labels[held_scene])
        rows.append(
            {
                "scene": held_scene,
                "training_scene_count": len(training_scenes),
                "predicted_alpha": predicted_alpha,
                "oracle_alpha": oracle_alpha,
                "adaptive_delta": curves[held_scene][predicted_alpha],
                "fixed_alpha": fixed_alpha,
                "fixed_alpha_delta": curves[held_scene][fixed_alpha],
                "oracle_delta": curves[held_scene][oracle_alpha],
            }
        )
    return {
        "scene_count": len(rows),
        "fixed_alpha": fixed_alpha,
        "adaptive_delta_mean": float(
            np.mean([row["adaptive_delta"] for row in rows])
        ),
        "fixed_alpha_delta_mean": float(
            np.mean([row["fixed_alpha_delta"] for row in rows])
        ),
        "oracle_delta_mean": float(
            np.mean([row["oracle_delta"] for row in rows])
        ),
        "alpha_match_fraction": float(
            np.mean(
                [
                    row["predicted_alpha"] == row["oracle_alpha"]
                    for row in rows
                ]
            )
        ),
        "rows": rows,
    }
