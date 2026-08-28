from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from pre_experiments.variational_camera_latent.camera import (
    decode_camera_tokens,
    pose_encoding_to_c2w,
)

from .data import load_long_context
from .labels import load_privileged_labels
from .train import load_camera_head_checkpoint


PREDICTION_SCHEMA = "long_short_camera_head.prediction.v1"
EVALUATION_SCHEMA = "long_short_camera_head.evaluation.v1"
PREDICTION_MEMBERS = {
    "scene",
    "frame_ids",
    "pose_encoding",
    "predicted_c2w",
    "source_sha256",
    "checkpoint_sha256",
}


@dataclass(frozen=True)
class PredictionRecord:
    scene: str
    path: Path
    sha256: str


@dataclass(frozen=True)
class EvaluationRecord:
    scene: str
    path: Path
    metrics: dict[str, object]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _validate_prediction(arrays: dict[str, np.ndarray]) -> None:
    if set(arrays) != PREDICTION_MEMBERS:
        raise ValueError("prediction members do not match the strict schema")
    if any(value.dtype.hasobject for value in arrays.values()):
        raise ValueError("prediction arrays may not use object dtype")
    if arrays["scene"].shape != () or arrays["scene"].dtype.kind != "U":
        raise ValueError("prediction scene must be a Unicode scalar")
    if arrays["frame_ids"].shape != (500,) or not np.issubdtype(
        arrays["frame_ids"].dtype, np.integer
    ):
        raise ValueError("prediction frame IDs must have shape [500]")
    if arrays["pose_encoding"].shape != (500, 9):
        raise ValueError("prediction pose encoding must have shape [500,9]")
    if arrays["predicted_c2w"].shape != (500, 4, 4):
        raise ValueError("prediction poses must have shape [500,4,4]")
    if not np.isfinite(arrays["pose_encoding"]).all() or not np.isfinite(
        arrays["predicted_c2w"]
    ).all():
        raise ValueError("prediction tensors must be finite")
    if not np.allclose(arrays["predicted_c2w"][:, 3, :], [0.0, 0.0, 0.0, 1.0]):
        raise ValueError("prediction poses must be homogeneous")
    for name in ("source_sha256", "checkpoint_sha256"):
        value = arrays[name]
        if value.shape != () or value.dtype.kind != "U" or len(str(value)) != 64:
            raise ValueError(f"prediction {name} is malformed")


def load_prediction(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(Path(path), allow_pickle=False) as archive:
            arrays = {name: np.asarray(archive[name]).copy() for name in archive.files}
    except (OSError, ValueError, KeyError) as error:
        raise ValueError(f"invalid prediction shard: {path}") from error
    _validate_prediction(arrays)
    return arrays


def run_long_only_inference(
    long_context_path: Path,
    checkpoint_path: Path,
    checkpoint_dir: Path,
    destination: Path,
    device: torch.device,
) -> PredictionRecord:
    """Decode a fine-tuned Camera Head from long tokens and nothing privileged."""
    long_context = load_long_context(long_context_path)
    model = load_camera_head_checkpoint(checkpoint_path, checkpoint_dir, device)
    tokens = torch.from_numpy(long_context["camera_tokens"]).unsqueeze(0).to(device)
    pose = decode_camera_tokens(model, tokens, iterations=4)
    c2w = pose_encoding_to_c2w(pose.float())
    arrays = {
        "scene": long_context["scene"].copy(),
        "frame_ids": long_context["frame_ids"].copy(),
        "pose_encoding": pose[0].float().cpu().numpy(),
        "predicted_c2w": c2w[0].double().cpu().numpy(),
        "source_sha256": long_context["source_sha256"].copy(),
        "checkpoint_sha256": np.asarray(_sha256_file(checkpoint_path), dtype="U64"),
    }
    _validate_prediction(arrays)
    destination = Path(destination)
    _atomic_npz(destination, arrays)
    return PredictionRecord(str(arrays["scene"]), destination, _sha256_file(destination))


def _apply_frozen_transform(poses: np.ndarray, labels: dict[str, np.ndarray]) -> np.ndarray:
    rotation = labels["oracle_rotation"].astype(np.float64)
    scale = float(labels["oracle_scale"])
    translation = labels["oracle_translation"].astype(np.float64)
    aligned = poses.astype(np.float64, copy=True)
    aligned[:, :3, :3] = np.einsum("ij,sjk->sik", rotation, poses[:, :3, :3])
    aligned[:, :3, 3] = scale * (poses[:, :3, 3] @ rotation.T) + translation
    return aligned


def _translation_rms(predicted: np.ndarray, target: np.ndarray) -> float:
    delta = predicted[:, :3, 3] - target[:, :3, 3]
    return float(np.sqrt(np.mean(np.sum(delta * delta, axis=-1))))


def _rotation_error_degrees(predicted: np.ndarray, target: np.ndarray) -> float:
    relative = np.einsum(
        "sji,sjk->sik", predicted[:, :3, :3], target[:, :3, :3]
    )
    cosine = np.clip((np.trace(relative, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)).mean())


def _relative_translation_rms(
    predicted: np.ndarray,
    target: np.ndarray,
    scene_scale: float,
) -> dict[str, float]:
    values: dict[str, float] = {}
    for lag in (1, 5, 10, 25):
        pred_delta = predicted[lag:, :3, 3] - predicted[:-lag, :3, 3]
        gt_delta = target[lag:, :3, 3] - target[:-lag, :3, 3]
        residual = (pred_delta - gt_delta) / scene_scale
        values[str(lag)] = float(np.sqrt(np.mean(np.sum(residual * residual, axis=-1))))
    return values


def evaluate_prediction(
    prediction_path: Path,
    privileged_path: Path,
    destination: Path,
) -> EvaluationRecord:
    """Score a completed long-only prediction with separate privileged labels."""
    prediction = load_prediction(prediction_path)
    labels = load_privileged_labels(privileged_path)
    if str(prediction["scene"]) != str(labels["scene"]):
        raise ValueError("prediction and privileged labels have different scenes")
    if not np.array_equal(prediction["frame_ids"], labels["frame_ids"]):
        raise ValueError("prediction and privileged labels have different frame IDs")
    if str(prediction["source_sha256"]) != str(labels["source_sha256"]):
        raise ValueError("prediction and privileged labels have different source identity")

    baseline_pose = torch.from_numpy(labels["baseline_pose_encoding"]).unsqueeze(0)
    baseline_c2w = pose_encoding_to_c2w(baseline_pose.float())[0].double().numpy()
    gt = labels["gt_c2w"].astype(np.float64)
    baseline_aligned = _apply_frozen_transform(baseline_c2w, labels)
    predicted_aligned = _apply_frozen_transform(prediction["predicted_c2w"], labels)
    baseline_rms = _translation_rms(baseline_aligned, gt)
    predicted_rms = _translation_rms(predicted_aligned, gt)
    utility = (
        float((baseline_rms - predicted_rms) / baseline_rms)
        if baseline_rms > np.finfo(np.float64).eps
        else (0.0 if predicted_rms <= np.finfo(np.float64).eps else float("-inf"))
    )
    overlap = slice(50, 450)
    scene_scale = float(labels["gt_scene_scale"])
    metrics: dict[str, object] = {
        "schema": EVALUATION_SCHEMA,
        "scene": str(prediction["scene"]),
        "frame_count": 500,
        "source_sha256": str(prediction["source_sha256"]),
        "checkpoint_sha256": str(prediction["checkpoint_sha256"]),
        "oracle_digest": str(labels["oracle_digest"]),
        "baseline_rms": baseline_rms,
        "predicted_rms": predicted_rms,
        "utility": utility,
        "baseline_overlap_rms": _translation_rms(baseline_aligned[overlap], gt[overlap]),
        "predicted_overlap_rms": _translation_rms(predicted_aligned[overlap], gt[overlap]),
        "baseline_rotation_deg": _rotation_error_degrees(baseline_aligned, gt),
        "predicted_rotation_deg": _rotation_error_degrees(predicted_aligned, gt),
        "relative_translation_rms": _relative_translation_rms(
            predicted_aligned, gt, scene_scale
        ),
        "baseline_relative_translation_rms": _relative_translation_rms(
            baseline_aligned, gt, scene_scale
        ),
        "pose_encoding_correction_rms": float(
            np.sqrt(
                np.mean(
                    (
                        prediction["pose_encoding"].astype(np.float64)
                        - labels["baseline_pose_encoding"].astype(np.float64)
                    )
                    ** 2
                )
            )
        ),
    }
    numeric = [
        value
        for name, value in metrics.items()
        if isinstance(value, (float, int)) and name != "frame_count"
    ]
    if not np.isfinite(numeric).all():
        raise ValueError("evaluation produced non-finite metrics")
    destination = Path(destination)
    _atomic_json(destination, metrics)
    return EvaluationRecord(str(prediction["scene"]), destination, metrics)
